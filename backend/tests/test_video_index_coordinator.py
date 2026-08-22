from __future__ import annotations

import hashlib

import pytest

from videoscope.artifacts import (
    IndexingSpecifications,
    StageKind,
    StageSpecification,
    StageState,
)
from videoscope.jobs import JobState, VideoIndexPlanSnapshot
from videoscope.processing.coordinator import (
    VideoIndexCoordinator,
    VideoIndexPlanUnavailable,
)
from videoscope.providers.whisper import WhisperPromptSnapshot
from videoscope.repository import Repository


def _stage(kind: StageKind) -> StageSpecification:
    parameters: dict[str, object] = {}
    if kind is StageKind.SPEECH:
        parameters = {
            "effective_prompt_sha256": hashlib.sha256(b"").hexdigest(),
            "glossary_state": "not_configured",
        }
    return StageSpecification(
        kind=kind,
        schema_version=1,
        implementation_revision=f"test.{kind.value}.v1",
        parameters=parameters,
        dependencies={"videoscope": "test"},
    )


def _plan() -> VideoIndexPlanSnapshot:
    prompt = WhisperPromptSnapshot(
        effective_prompt=None,
        effective_prompt_sha256=hashlib.sha256(b"").hexdigest(),
        glossary_state="not_configured",
    )
    return VideoIndexPlanSnapshot(
        schema_version=2,
        specifications=IndexingSpecifications(
            scenes=_stage(StageKind.SCENES),
            speech=_stage(StageKind.SPEECH),
            ocr=_stage(StageKind.OCR),
            objects=_stage(StageKind.OBJECTS),
            text_vectors=_stage(StageKind.TEXT_VECTORS),
        ),
        visual_dense_specification=_stage(StageKind.VISUAL_DENSE),
        lighthouse_specification=_stage(StageKind.LIGHTHOUSE),
        whisper_prompt_snapshot=prompt,
        executor_identity="sha256:" + "a" * 64,
    )


class Wakeup:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.video_ids: list[str] = []
        self.fail = fail

    def wake(self, video_id: str) -> None:
        self.calls += 1
        self.video_ids.append(video_id)
        if self.fail:
            raise RuntimeError("wake unavailable")


def _coordinator(tmp_path, *, wakeup: Wakeup | None = None):  # type: ignore[no-untyped-def]
    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()
    resolved_wakeup = wakeup or Wakeup()
    coordinator = VideoIndexCoordinator(
        repository,
        plan_factory=_plan,
        wakeup=resolved_wakeup,
    )
    return repository, resolved_wakeup, coordinator


def _record_terminal_receipts(
    repository: Repository,
    *,
    job_id: str,
    execution_token: str,
) -> None:
    plan = _plan()
    specifications = (
        *plan.specifications.segment_specifications,
        plan.specifications.text_vectors,
        plan.visual_dense_specification,
        plan.lighthouse_specification,
    )
    for specification in specifications:
        run = repository.create_stage_run(
            video_id="video-1",
            specification=specification,
            run_id=f"{job_id}-{specification.kind.value}",
            job_id=job_id,
            execution_token=execution_token,
        )
        repository.transition_stage_run(
            run.run_id,
            StageState.NOT_CONFIGURED,
            execution_token=execution_token,
        )


def test_coordinator_persists_ingest_before_advisory_wake(tmp_path) -> None:
    repository, wakeup, coordinator = _coordinator(tmp_path)

    video, job = coordinator.create_ingest(
        video_id="video-1",
        original_name="match.mp4",
        stored_name="video-1.mp4",
        media_path="/managed/video-1.mp4",
        size_bytes=100,
        source_sha256="b" * 64,
        job_id="job-1",
    )

    assert wakeup.calls == 1
    assert wakeup.video_ids == ["video-1"]
    assert video.id == job.video_id
    assert job.state is JobState.QUEUED
    assert repository.get_video_index_job_plan(job.job_id) == _plan()


def test_wake_failure_does_not_rollback_or_mask_a_durable_enqueue(tmp_path) -> None:
    wakeup = Wakeup(fail=True)
    repository, _wakeup, coordinator = _coordinator(tmp_path, wakeup=wakeup)

    video, job = coordinator.create_ingest(
        video_id="video-1",
        original_name="match.mp4",
        stored_name="video-1.mp4",
        media_path="/managed/video-1.mp4",
        size_bytes=100,
        source_sha256="b" * 64,
        job_id="job-1",
    )

    assert video.id == "video-1"
    assert repository.get_video_index_job(job.job_id) == job
    assert wakeup.calls == 1


def test_plan_failure_happens_before_any_ingest_mutation(tmp_path) -> None:
    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()

    def fail_plan() -> VideoIndexPlanSnapshot:
        raise ValueError("runtime is not attested")

    coordinator = VideoIndexCoordinator(
        repository,
        plan_factory=fail_plan,
        wakeup=Wakeup(),
    )
    with pytest.raises(VideoIndexPlanUnavailable):
        coordinator.create_ingest(
            video_id="video-1",
            original_name="match.mp4",
            stored_name="video-1.mp4",
            media_path="/managed/video-1.mp4",
            size_bytes=100,
            source_sha256="b" * 64,
            job_id="job-1",
        )
    assert repository.get_video("video-1") is None


def test_reindex_deduplicates_active_work_and_wakes_each_request(tmp_path) -> None:
    repository, wakeup, coordinator = _coordinator(tmp_path)
    _video, ingest = coordinator.create_ingest(
        video_id="video-1",
        original_name="match.mp4",
        stored_name="video-1.mp4",
        media_path="/managed/video-1.mp4",
        size_bytes=100,
        source_sha256="b" * 64,
        job_id="job-1",
    )
    claimed = repository.claim_next_video_index_job(execution_token="t" * 20)
    assert claimed is not None
    _record_terminal_receipts(
        repository,
        job_id=ingest.job_id,
        execution_token=claimed.execution_token or "",
    )
    repository.complete_video_index_job(
        ingest.job_id,
        execution_token=claimed.execution_token or "",
    )

    first = coordinator.enqueue_reindex("video-1", job_id="job-2")
    second = coordinator.enqueue_reindex("video-1", job_id="job-3")

    assert first.job_id == second.job_id == "job-2"
    assert wakeup.calls == 3


def test_cancel_and_retry_preserve_the_persisted_plan(tmp_path) -> None:
    repository, wakeup, coordinator = _coordinator(tmp_path)
    _video, parent = coordinator.create_ingest(
        video_id="video-1",
        original_name="match.mp4",
        stored_name="video-1.mp4",
        media_path="/managed/video-1.mp4",
        size_bytes=100,
        source_sha256="b" * 64,
        job_id="job-1",
    )
    cancelled = coordinator.request_cancellation(parent.job_id)
    assert cancelled.state is JobState.CANCELLED

    child = coordinator.retry(parent.job_id, retry_job_id="job-2")

    assert child.retry_of_job_id == parent.job_id
    assert repository.get_video_index_job_plan(child.job_id) == _plan()
    assert wakeup.calls == 2


def test_coordinator_get_job_is_bounded_to_existing_resources(tmp_path) -> None:
    _repository, _wakeup, coordinator = _coordinator(tmp_path)
    assert coordinator.get_job("missing-job") is None
    with pytest.raises(ValueError):
        coordinator.get_job("../escape")
