from __future__ import annotations

import logging
from typing import Callable, Protocol

from videoscope.jobs import VideoIndexJob, VideoIndexPlanSnapshot
from videoscope.repository import Repository, VideoRecord


logger = logging.getLogger(__name__)


class DispatcherWakeup(Protocol):
    def wake(self, video_id: str) -> None: ...


class VideoIndexPlanUnavailable(RuntimeError):
    """The current runtime cannot produce an attested durable plan."""


class VideoIndexCoordinator:
    """Own durable enqueue transitions while treating wake-up as advisory."""

    def __init__(
        self,
        repository: Repository,
        *,
        plan_factory: Callable[[], VideoIndexPlanSnapshot],
        wakeup: DispatcherWakeup,
    ) -> None:
        if not isinstance(repository, Repository):
            raise TypeError("video index coordinator repository must be validated")
        if not callable(plan_factory):
            raise TypeError("video index plan factory must be callable")
        if not callable(getattr(wakeup, "wake", None)):
            raise TypeError("video index dispatcher wakeup must be callable")
        self.repository = repository
        self.plan_factory = plan_factory
        self.wakeup = wakeup

    def _new_plan(self) -> VideoIndexPlanSnapshot:
        try:
            plan = self.plan_factory()
        except Exception as error:
            raise VideoIndexPlanUnavailable(
                "video index plan is unavailable"
            ) from error
        if not isinstance(plan, VideoIndexPlanSnapshot):
            raise VideoIndexPlanUnavailable("video index plan is unavailable")
        return plan

    def _wake_after_commit(self, *, job_id: str, video_id: str) -> None:
        try:
            self.wakeup.wake(video_id)
        except Exception:
            logger.exception(
                "Durable video index wake-up failed",
                extra={"job_id": job_id},
            )

    def create_ingest(
        self,
        *,
        video_id: str,
        original_name: str,
        stored_name: str,
        media_path: str,
        size_bytes: int,
        source_sha256: str,
        job_id: str | None = None,
    ) -> tuple[VideoRecord, VideoIndexJob]:
        plan = self._new_plan()
        video, job = self.repository.create_video_with_asset_and_index_job(
            video_id=video_id,
            original_name=original_name,
            stored_name=stored_name,
            media_path=media_path,
            size_bytes=size_bytes,
            source_sha256=source_sha256,
            plan=plan,
            job_id=job_id,
        )
        self._wake_after_commit(job_id=job.job_id, video_id=job.video_id)
        return video, job

    def enqueue_reindex(
        self,
        video_id: str,
        *,
        job_id: str | None = None,
    ) -> VideoIndexJob:
        job = self.repository.enqueue_video_reindex_job(
            video_id,
            plan=self._new_plan(),
            job_id=job_id,
        )
        self._wake_after_commit(job_id=job.job_id, video_id=job.video_id)
        return job

    def get_job(self, job_id: str) -> VideoIndexJob | None:
        return self.repository.get_video_index_job(job_id)

    def request_cancellation(self, job_id: str) -> VideoIndexJob:
        return self.repository.request_video_index_job_cancellation(job_id)

    def retry(
        self,
        job_id: str,
        *,
        retry_job_id: str | None = None,
    ) -> VideoIndexJob:
        child = self.repository.retry_video_index_job(
            job_id,
            retry_job_id=retry_job_id,
        )
        self._wake_after_commit(job_id=child.job_id, video_id=child.video_id)
        return child
