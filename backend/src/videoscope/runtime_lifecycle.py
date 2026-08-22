from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import errno
import fcntl
import logging
import os
from pathlib import Path
import stat
from threading import Condition, Event, Lock, Thread, current_thread
from typing import Protocol
from uuid import uuid4

from videoscope.artifacts import (
    ArtifactGCJob,
    ArtifactGCOutcome,
    TextVectorBuildQuarantine,
    TextVectorBuildRecoveryReport,
)


logger = logging.getLogger(__name__)


class ArtifactGCRepository(Protocol):
    def recover_abandoned_text_vector_builds(
        self,
        *,
        limit: int = 100,
    ) -> TextVectorBuildRecoveryReport: ...

    def list_quarantined_text_vector_builds(
        self,
        *,
        limit: int = 100,
    ) -> tuple[TextVectorBuildQuarantine, ...]: ...

    def recover_expired_text_vector_builds(
        self,
        *,
        limit: int = 100,
    ) -> TextVectorBuildRecoveryReport: ...

    def claim_next_artifact_gc_job(
        self,
        *,
        worker_id: str,
        lease_seconds: int = 30,
    ) -> ArtifactGCJob | None: ...

    def heartbeat_artifact_gc_job(
        self,
        job_id: str,
        *,
        worker_id: str,
        lease_token: str,
        attempt: int,
        lease_seconds: int = 30,
    ) -> ArtifactGCJob: ...

    def finish_artifact_gc_job(
        self,
        job_id: str,
        *,
        worker_id: str,
        lease_token: str,
        attempt: int,
        outcome: ArtifactGCOutcome,
        error_code: str | None = None,
    ) -> ArtifactGCJob: ...


class GenerationStore(Protocol):
    def delete_generation_from_storage(
        self,
        generation_id: str,
        *,
        index_specification_hash: str,
        collection_name: str,
    ) -> None: ...


class TextVectorStorageGate:
    """Keep generation writers and destructive maintenance mutually exclusive."""

    def __init__(self) -> None:
        self._condition = Condition(Lock())
        self._active_builds = 0
        self._waiting_builds = 0
        self._maintenance_active = False

    @contextmanager
    def build_activity(self) -> Iterator[None]:
        with self._condition:
            self._waiting_builds += 1
            try:
                while self._maintenance_active:
                    self._condition.wait()
                self._active_builds += 1
            finally:
                self._waiting_builds -= 1
        try:
            yield
        finally:
            with self._condition:
                self._active_builds -= 1
                if self._active_builds < 0:  # pragma: no cover - invariant guard
                    self._active_builds = 0
                    raise RuntimeError("text vector storage gate is unbalanced")
                self._condition.notify_all()

    @contextmanager
    def try_maintenance(self) -> Iterator[bool]:
        acquired = False
        with self._condition:
            if (
                self._active_builds == 0
                and self._waiting_builds == 0
                and not self._maintenance_active
            ):
                self._maintenance_active = True
                acquired = True
        try:
            yield acquired
        finally:
            if acquired:
                with self._condition:
                    self._maintenance_active = False
                    self._condition.notify_all()


class _ArtifactGCLeaseLost(RuntimeError):
    pass


class _ArtifactGCLeaseHeartbeat:
    def __init__(
        self,
        repository: ArtifactGCRepository,
        job: ArtifactGCJob,
        *,
        lease_seconds: int,
        interval: float,
    ) -> None:
        self.repository = repository
        self.job = job
        self.lease_seconds = lease_seconds
        self.interval = interval
        self._stop = Event()
        self._error: Exception | None = None
        self._thread = Thread(
            target=self._run,
            name=f"videoscope-gc-lease-{job.job_id[:8]}",
            daemon=True,
        )

    def _run(self) -> None:
        assert self.job.worker_id is not None and self.job.lease_token is not None
        while not self._stop.wait(self.interval):
            try:
                self.repository.heartbeat_artifact_gc_job(
                    self.job.job_id,
                    worker_id=self.job.worker_id,
                    lease_token=self.job.lease_token,
                    attempt=self.job.attempt,
                    lease_seconds=self.lease_seconds,
                )
            except Exception as error:
                self._error = error
                return

    def __enter__(self) -> _ArtifactGCLeaseHeartbeat:
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore[no-untyped-def]
        self._stop.set()
        self._thread.join()
        if exc_type is None and self._error is not None:
            raise _ArtifactGCLeaseLost("artifact GC lease heartbeat failed") from self._error


class ArtifactGarbageCollector:
    """Bounded leased worker for idempotent deletion of orphaned generations."""

    def __init__(
        self,
        repository: ArtifactGCRepository,
        generation_store: GenerationStore,
        *,
        worker_id: str | None = None,
        batch_size: int = 32,
        startup_recovery_limit: int = 1000,
        lease_seconds: int = 60,
        heartbeat_interval: float = 20.0,
        poll_interval: float = 2.0,
        storage_gate: TextVectorStorageGate | None = None,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 1000:
            raise ValueError("artifact GC batch size must be between 1 and 1000")
        if (
            isinstance(startup_recovery_limit, bool)
            or not isinstance(startup_recovery_limit, int)
            or not 1 <= startup_recovery_limit <= 1000
        ):
            raise ValueError("artifact GC recovery limit must be between 1 and 1000")
        if (
            isinstance(lease_seconds, bool)
            or not isinstance(lease_seconds, int)
            or not 1 <= lease_seconds <= 300
        ):
            raise ValueError("artifact GC lease must be between 1 and 300 seconds")
        if (
            isinstance(heartbeat_interval, bool)
            or not isinstance(heartbeat_interval, (int, float))
            or not 0 < float(heartbeat_interval) < lease_seconds
        ):
            raise ValueError("artifact GC heartbeat must be shorter than its lease")
        if (
            isinstance(poll_interval, bool)
            or not isinstance(poll_interval, (int, float))
            or float(poll_interval) <= 0
        ):
            raise ValueError("artifact GC poll interval must be positive")
        self.repository = repository
        self.generation_store = generation_store
        self.worker_id = worker_id or f"gc-{uuid4().hex}"
        self.batch_size = batch_size
        self.startup_recovery_limit = startup_recovery_limit
        self.lease_seconds = lease_seconds
        self.heartbeat_interval = float(heartbeat_interval)
        self.poll_interval = float(poll_interval)
        self.storage_gate = storage_gate or TextVectorStorageGate()
        self.recovery_degraded = False
        self.last_recovery_report: TextVectorBuildRecoveryReport | None = None
        self._stop = Event()
        self._state_lock = Lock()
        self._thread: Thread | None = None

    def recover_startup(self) -> tuple[str, ...]:
        request_limit = min(self.startup_recovery_limit + 1, 1000)
        report = self.repository.recover_abandoned_text_vector_builds(
            limit=request_limit,
        )
        if not isinstance(report, TextVectorBuildRecoveryReport):
            self.recovery_degraded = True
            raise RuntimeError("text vector startup recovery returned an invalid report")
        if report.examined_count > request_limit or (
            report.has_more and report.examined_count < request_limit
        ):
            self.recovery_degraded = True
            raise RuntimeError("text vector startup recovery returned an invalid report")
        self.last_recovery_report = report
        quarantines = self.repository.list_quarantined_text_vector_builds(limit=1)
        if type(quarantines) is not tuple:
            self.recovery_degraded = True
            raise RuntimeError("text vector startup recovery returned invalid health")
        if report.quarantined_count or quarantines:
            self.recovery_degraded = True
            logger.error(
                "Text vector startup recovery quarantined corrupt metadata",
                extra={
                    "new_quarantined_count": report.quarantined_count,
                    "has_persisted_quarantine": bool(quarantines),
                },
            )
        if (
            report.examined_count > self.startup_recovery_limit
            or report.has_more
        ):
            raise RuntimeError("text vector startup recovery limit was exceeded")
        return report.terminalized_generation_ids

    def run_batch(self) -> int:
        with self.storage_gate.try_maintenance() as acquired:
            if not acquired:
                return 0
            try:
                recovery_report = self.repository.recover_expired_text_vector_builds(
                    limit=self.batch_size,
                )
            except Exception as error:
                self.recovery_degraded = True
                logger.error("Text vector build recovery failed", exc_info=error)
                return 0
            if not isinstance(recovery_report, TextVectorBuildRecoveryReport):
                self.recovery_degraded = True
                logger.error("Text vector build recovery returned an invalid report")
                return 0
            self.last_recovery_report = recovery_report
            if recovery_report.quarantined_count:
                self.recovery_degraded = True
                logger.error(
                    "Text vector build recovery quarantined corrupt metadata",
                    extra={"quarantined_count": recovery_report.quarantined_count},
                )
            return self._run_claimed_batch()

    def _run_claimed_batch(self) -> int:
        processed = 0
        for _ in range(self.batch_size):
            try:
                job = self.repository.claim_next_artifact_gc_job(
                    worker_id=self.worker_id,
                    lease_seconds=self.lease_seconds,
                )
            except Exception as error:
                logger.error("Artifact GC claim failed", exc_info=error)
                break
            if job is None:
                break
            processed += 1
            if (
                not isinstance(job, ArtifactGCJob)
                or job.state != "running"
                or job.worker_id != self.worker_id
                or job.lease_token is None
            ):
                logger.error("Artifact GC repository returned an invalid lease")
                break
            try:
                with _ArtifactGCLeaseHeartbeat(
                    self.repository,
                    job,
                    lease_seconds=self.lease_seconds,
                    interval=self.heartbeat_interval,
                ):
                    self.generation_store.delete_generation_from_storage(
                        job.generation_id,
                        index_specification_hash=job.index_specification_hash,
                        collection_name=job.collection_name,
                    )
            except _ArtifactGCLeaseLost as error:
                logger.error("Artifact GC lease was lost", exc_info=error)
                continue
            except ValueError as error:
                outcome = ArtifactGCOutcome.PERMANENT_FAILURE
                error_code = "artifact_gc_contract_invalid"
                logger.error("Artifact GC contract validation failed", exc_info=error)
            except Exception as error:
                outcome = ArtifactGCOutcome.TRANSIENT_FAILURE
                error_code = "artifact_gc_storage_unavailable"
                logger.warning("Artifact GC storage deletion failed", exc_info=error)
            else:
                outcome = ArtifactGCOutcome.SUCCESS
                error_code = None
            try:
                assert job.worker_id is not None and job.lease_token is not None
                self.repository.finish_artifact_gc_job(
                    job.job_id,
                    worker_id=job.worker_id,
                    lease_token=job.lease_token,
                    attempt=job.attempt,
                    outcome=outcome,
                    error_code=error_code,
                )
            except Exception as error:
                logger.error("Artifact GC completion fencing failed", exc_info=error)
        return processed

    def _run(self) -> None:
        while not self._stop.is_set():
            processed = self.run_batch()
            if processed < self.batch_size:
                self._stop.wait(self.poll_interval)

    def start(self) -> None:
        with self._state_lock:
            if self._thread is not None:
                raise RuntimeError("artifact GC worker is already started")
            self._stop.clear()
            thread = Thread(
                target=self._run,
                name="videoscope-artifact-gc",
                daemon=True,
            )
            try:
                thread.start()
            except Exception:
                self._stop.set()
                raise
            self._thread = thread

    def close(self, *, timeout: float | None = 5) -> bool:
        self._stop.set()
        with self._state_lock:
            thread = self._thread
        if thread is None:
            return True
        if thread is current_thread():
            return False
        thread.join(timeout=timeout)
        return not thread.is_alive()


class ExclusiveRuntimeLock:
    """One VideoScope runtime owns a data directory at a time.

    The lock file is intentionally persistent: unlinking a lock file creates an
    inode race in which two processes can each hold a different lock.
    """

    filename = ".videoscope-runtime.lock"

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.path = self.data_dir / self.filename
        self._file_descriptor: int | None = None

    @property
    def is_acquired(self) -> bool:
        return self._file_descriptor is not None

    @staticmethod
    def _reject_symbolic_link_ancestors(path: Path) -> None:
        absolute_path = Path(os.path.abspath(path))
        for candidate in reversed((absolute_path, *absolute_path.parents)):
            try:
                candidate_stat = os.lstat(candidate)
            except FileNotFoundError:
                continue
            except OSError as error:
                raise RuntimeError("runtime data path is unavailable") from error
            if stat.S_ISLNK(candidate_stat.st_mode):
                raise RuntimeError(
                    "runtime data path must not contain a symbolic link"
                )

    @staticmethod
    def _stable_identity(metadata: os.stat_result) -> tuple[int, ...]:
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_nlink,
            metadata.st_uid,
            metadata.st_gid,
        )

    @classmethod
    def _open_existing_data_directory(cls, path: Path) -> int:
        """Retain an existing directory through an all-component no-follow chain."""
        cls._reject_symbolic_link_ancestors(path)
        absolute = Path(os.path.abspath(path))
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        directory = getattr(os, "O_DIRECTORY", 0)
        if not nofollow or not directory:
            raise RuntimeError("secure runtime lock access is unavailable")
        flags = os.O_RDONLY | nofollow | directory | getattr(os, "O_CLOEXEC", 0)
        try:
            descriptor = os.open(absolute.anchor, flags)
        except OSError as error:
            raise RuntimeError("runtime data directory is unavailable") from error
        try:
            for part in absolute.parts[1:]:
                try:
                    observed = os.stat(
                        part,
                        dir_fd=descriptor,
                        follow_symlinks=False,
                    )
                except OSError as error:
                    raise RuntimeError("runtime data directory is unavailable") from error
                if stat.S_ISLNK(observed.st_mode):
                    raise RuntimeError(
                        "runtime data path must not contain a symbolic link"
                    )
                if not stat.S_ISDIR(observed.st_mode):
                    raise RuntimeError(
                        "runtime data directory must be a regular directory"
                    )
                child: int | None = None
                try:
                    child = os.open(part, flags, dir_fd=descriptor)
                    opened = os.fstat(child)
                    current = os.stat(
                        part,
                        dir_fd=descriptor,
                        follow_symlinks=False,
                    )
                except OSError as error:
                    if child is not None:
                        os.close(child)
                    raise RuntimeError("runtime data directory is unavailable") from error
                assert child is not None
                if not (
                    cls._stable_identity(observed)
                    == cls._stable_identity(opened)
                    == cls._stable_identity(current)
                ):
                    os.close(child)
                    raise RuntimeError("runtime data path changed while opening")
                os.close(descriptor)
                descriptor = child
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    @staticmethod
    def _is_private_managed_lock(metadata: os.stat_result) -> bool:
        return (
            stat.S_ISREG(metadata.st_mode)
            and metadata.st_nlink == 1
            and metadata.st_uid == os.geteuid()
            and stat.S_IMODE(metadata.st_mode) == 0o600
        )

    def acquire(self) -> None:
        if self._file_descriptor is not None:
            raise RuntimeError("runtime lock is already acquired")
        self._reject_symbolic_link_ancestors(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._reject_symbolic_link_ancestors(self.data_dir)
        try:
            directory_stat = os.stat(self.data_dir, follow_symlinks=False)
        except OSError as error:
            raise RuntimeError("runtime data directory is unavailable") from error
        if not stat.S_ISDIR(directory_stat.st_mode):
            raise RuntimeError("runtime data directory must be a regular directory")

        flags = os.O_RDWR | os.O_CREAT
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            file_descriptor = os.open(self.path, flags, 0o600)
        except OSError as error:
            if error.errno in {errno.ELOOP, errno.EMLINK}:
                raise RuntimeError("runtime lock must be a managed regular file") from error
            raise RuntimeError("runtime lock could not be opened") from error
        try:
            file_stat = os.fstat(file_descriptor)
            if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
                raise RuntimeError("runtime lock must be a managed regular file")
            os.fchmod(file_descriptor, 0o600)
            try:
                fcntl.flock(file_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    "another VideoScope process already owns this data directory"
                ) from error
        except Exception:
            os.close(file_descriptor)
            raise
        self._file_descriptor = file_descriptor

    def acquire_existing(self) -> None:
        """Acquire the persistent lock without creating or changing filesystem state."""
        if self._file_descriptor is not None:
            raise RuntimeError("runtime lock is already acquired")
        directory_descriptor = self._open_existing_data_directory(self.data_dir)
        file_descriptor: int | None = None
        try:
            try:
                observed = os.stat(
                    self.filename,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError as error:
                raise RuntimeError("runtime lock is unavailable") from error
            except OSError as error:
                raise RuntimeError("runtime lock could not be inspected") from error
            if not self._is_private_managed_lock(observed):
                raise RuntimeError("runtime lock must be a managed regular file")

            flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            try:
                file_descriptor = os.open(
                    self.filename,
                    flags,
                    dir_fd=directory_descriptor,
                )
                opened = os.fstat(file_descriptor)
                current = os.stat(
                    self.filename,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
            except OSError as error:
                raise RuntimeError("runtime lock could not be opened") from error
            if not (
                self._is_private_managed_lock(opened)
                and self._is_private_managed_lock(current)
                and self._stable_identity(observed)
                == self._stable_identity(opened)
                == self._stable_identity(current)
            ):
                raise RuntimeError("runtime lock must be a managed regular file")
            try:
                fcntl.flock(file_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    "another VideoScope process already owns this data directory"
                ) from error
        except Exception:
            if file_descriptor is not None:
                os.close(file_descriptor)
            raise
        finally:
            os.close(directory_descriptor)
        self._file_descriptor = file_descriptor

    def close(self) -> None:
        file_descriptor = self._file_descriptor
        if file_descriptor is None:
            return
        self._file_descriptor = None
        try:
            fcntl.flock(file_descriptor, fcntl.LOCK_UN)
        finally:
            os.close(file_descriptor)
