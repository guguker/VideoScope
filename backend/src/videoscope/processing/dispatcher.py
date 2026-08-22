from __future__ import annotations

from dataclasses import dataclass
import logging
import re
import secrets
from threading import Event, Lock, Thread, current_thread
from typing import Callable, Protocol

from videoscope.jobs import JobFenceError, JobState, VideoIndexJob, VideoIndexPlanSnapshot
from videoscope.processing.indexer import (
    JobCancelled,
    VideoIndexExecutionContext,
)
from videoscope.repository import Repository, VideoIndexRecoveryReport


logger = logging.getLogger(__name__)


_EXECUTOR_IDENTITY_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class DurableVideoIndexer(Protocol):
    def process_durable(
        self,
        video_id: str,
        *,
        context: VideoIndexExecutionContext,
    ) -> tuple[str, ...]: ...


@dataclass(frozen=True, slots=True)
class DispatcherRecoveryReport:
    examined_count: int
    retry_job_ids: tuple[str, ...]
    cancelled_job_ids: tuple[str, ...]
    quarantined_count: int


class _DispatchFailure(RuntimeError):
    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


class DurableVideoIndexDispatcher:
    """Sequential DB-backed dispatcher for fenced video-index attempts.

    The wake event only reduces polling latency. Queued work remains owned by
    SQLite and therefore survives process shutdown without an in-memory backlog.
    """

    def __init__(
        self,
        repository: Repository,
        indexer: DurableVideoIndexer,
        *,
        executor_identity_resolver: Callable[[], str],
        poll_interval: float = 0.5,
        recovery_batch_size: int = 100,
        token_factory: Callable[[], str] | None = None,
        start_immediately: bool = True,
    ) -> None:
        if not isinstance(repository, Repository):
            raise TypeError("dispatcher repository must be validated")
        if not callable(getattr(indexer, "process_durable", None)):
            raise TypeError("dispatcher indexer must support durable execution")
        if not callable(executor_identity_resolver):
            raise TypeError("dispatcher executor identity resolver must be callable")
        if (
            isinstance(poll_interval, bool)
            or not isinstance(poll_interval, (int, float))
            or not 0.01 <= float(poll_interval) <= 60.0
        ):
            raise ValueError("dispatcher poll interval must be between 0.01 and 60 seconds")
        if (
            isinstance(recovery_batch_size, bool)
            or not isinstance(recovery_batch_size, int)
            or not 1 <= recovery_batch_size <= 1000
        ):
            raise ValueError("dispatcher recovery batch size must be between 1 and 1000")
        resolved_token_factory = token_factory or (lambda: secrets.token_urlsafe(32))
        if not callable(resolved_token_factory):
            raise TypeError("dispatcher token factory must be callable")

        self.repository = repository
        self.indexer = indexer
        self.executor_identity_resolver = executor_identity_resolver
        self.poll_interval = float(poll_interval)
        self.recovery_batch_size = recovery_batch_size
        self.token_factory = resolved_token_factory
        self._state_lock = Lock()
        self._stop = Event()
        self._wake = Event()
        self._started = False
        self._closed = False
        self._worker: Thread | None = None
        if start_immediately:
            self.start()

    def start(self) -> None:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("video index dispatcher is closed")
            if self._started:
                raise RuntimeError("video index dispatcher is already started")
            worker = Thread(
                target=self._run,
                name="videoscope-video-index-dispatcher",
                daemon=True,
            )
            try:
                worker.start()
            except Exception:
                self._worker = None
                raise
            self._worker = worker
            self._started = True

    def wake(self) -> None:
        """Advisory notification after a durable enqueue transaction commits."""
        with self._state_lock:
            if self._closed:
                raise RuntimeError("video index dispatcher is closed")
        self._wake.set()

    def recover_startup(self) -> DispatcherRecoveryReport:
        """Terminalize abandoned leases and queue exact-plan retry children."""
        with self._state_lock:
            if self._closed:
                raise RuntimeError("video index dispatcher is closed")
            if self._started:
                raise RuntimeError("video index recovery must run before dispatcher start")

        examined = 0
        retries: list[str] = []
        cancelled: list[str] = []
        quarantined = 0
        for _batch in range(1024):
            report: VideoIndexRecoveryReport = (
                self.repository.recover_abandoned_video_index_jobs(
                    limit=self.recovery_batch_size,
                )
            )
            examined += report.examined_count
            retries.extend(report.retry_job_ids)
            cancelled.extend(report.cancelled_job_ids)
            quarantined += report.quarantined_count
            if not report.has_more:
                return DispatcherRecoveryReport(
                    examined_count=examined,
                    retry_job_ids=tuple(retries),
                    cancelled_job_ids=tuple(cancelled),
                    quarantined_count=quarantined,
                )
            if report.examined_count < 1:
                break
        raise RuntimeError("video index startup recovery exceeded its bounded work")

    def _new_execution_token(self) -> str:
        token = self.token_factory()
        if type(token) is not str:
            raise ValueError("dispatcher token factory returned invalid data")
        return token

    def _current_executor_identity(self) -> str:
        try:
            identity = self.executor_identity_resolver()
        except Exception as error:
            raise _DispatchFailure("job_executor_unavailable") from error
        if type(identity) is not str or not _EXECUTOR_IDENTITY_PATTERN.fullmatch(identity):
            raise _DispatchFailure("job_executor_unavailable")
        return identity

    def _load_context(
        self,
        job: VideoIndexJob,
        *,
        execution_token: str,
    ) -> VideoIndexExecutionContext:
        try:
            plan = self.repository.get_video_index_job_plan(job.job_id)
        except Exception as error:
            raise _DispatchFailure("job_plan_unavailable") from error
        if (
            not isinstance(plan, VideoIndexPlanSnapshot)
            or plan.plan_hash != job.plan_hash
        ):
            raise _DispatchFailure("job_plan_unavailable")
        if self._current_executor_identity() != plan.executor_identity:
            raise _DispatchFailure("job_executor_identity_mismatch")
        return VideoIndexExecutionContext(
            job_id=job.job_id,
            execution_token=execution_token,
            plan=plan,
        )

    def _terminalize_owned_failure(
        self,
        job: VideoIndexJob,
        *,
        execution_token: str,
        error_code: str,
    ) -> None:
        try:
            current = self.repository.get_video_index_job(job.job_id)
            if (
                current is None
                or current.state is not JobState.RUNNING
                or current.execution_token != execution_token
            ):
                return
            if current.cancel_requested_at is not None:
                self.repository.cancel_video_index_job(
                    job.job_id,
                    execution_token=execution_token,
                )
            else:
                self.repository.fail_video_index_job(
                    job.job_id,
                    execution_token=execution_token,
                    error_code=error_code,
                )
        except JobFenceError:
            return
        except Exception:
            logger.exception(
                "Video index job terminalization failed",
                extra={"job_id": job.job_id, "error_code": error_code},
            )

    def _execute(self, job: VideoIndexJob, *, execution_token: str) -> None:
        try:
            context = self._load_context(job, execution_token=execution_token)
            self.indexer.process_durable(job.video_id, context=context)
            if self._current_executor_identity() != context.plan.executor_identity:
                raise _DispatchFailure("job_executor_identity_mismatch")
            self.repository.complete_video_index_job(
                job.job_id,
                execution_token=execution_token,
            )
            return
        except JobCancelled:
            error_code = "job_execution_cancelled_unexpected"
        except _DispatchFailure as error:
            error_code = error.error_code
        except JobFenceError:
            return
        except Exception:
            logger.exception(
                "Durable video indexing failed",
                extra={"job_id": job.job_id, "video_id": job.video_id},
            )
            error_code = "job_execution_failed"
        self._terminalize_owned_failure(
            job,
            execution_token=execution_token,
            error_code=error_code,
        )

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                execution_token = self._new_execution_token()
                job = self.repository.claim_next_video_index_job(
                    execution_token=execution_token,
                )
            except Exception:
                logger.exception("Video index job claim failed")
                self._wake.wait(timeout=self.poll_interval)
                self._wake.clear()
                continue
            if job is None:
                self._wake.wait(timeout=self.poll_interval)
                self._wake.clear()
                continue
            self._execute(job, execution_token=execution_token)

    def close(self, *, timeout: float | None = 5) -> bool:
        with self._state_lock:
            if not self._closed:
                self._closed = True
                self._stop.set()
                self._wake.set()
            worker = self._worker
        if worker is None:
            return True
        if worker is current_thread():
            return False
        worker.join(timeout=timeout)
        return not worker.is_alive()
