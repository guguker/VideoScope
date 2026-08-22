from __future__ import annotations

import hashlib
from pathlib import Path
from threading import Event

import pytest

from videoscope.artifacts import (
    IndexingSpecifications,
    StageKind,
    StageSpecification,
    StageState,
)
from videoscope.jobs import JobFenceError, JobState, VideoIndexPlanSnapshot
from videoscope.media.ffmpeg import MediaProbe
from videoscope.processing.indexer import (
    Indexer,
    JobCancelled,
    VideoIndexExecutionContext,
)
from videoscope.providers.types import TimedText
from videoscope.providers.whisper import WhisperPromptSnapshot
from videoscope.repository import Repository
from videoscope.search.vector_index import MemoryVectorIndex


TOKEN = "worker-token-aaaaaaaaaaaaaaaa"


def _stage(
    kind: StageKind,
    *,
    parameters: dict[str, object] | None = None,
) -> StageSpecification:
    return StageSpecification(
        kind=kind,
        schema_version=1,
        implementation_revision=f"test.{kind.value}.v1",
        parameters=parameters or {},
        dependencies={"videoscope": "test"},
    )


def _plan(
    *,
    visual_boundary: str = "disabled",
    lighthouse_boundary: str = "disabled",
) -> VideoIndexPlanSnapshot:
    prompt_text = "Мозгов, контратака"
    prompt = WhisperPromptSnapshot(
        effective_prompt=prompt_text,
        effective_prompt_sha256=hashlib.sha256(prompt_text.encode()).hexdigest(),
        glossary_state="ready",
    )
    return VideoIndexPlanSnapshot(
        schema_version=2,
        specifications=IndexingSpecifications(
            scenes=_stage(StageKind.SCENES),
            speech=_stage(
                StageKind.SPEECH,
                parameters={
                    "effective_prompt_sha256": prompt.effective_prompt_sha256,
                    "glossary_state": prompt.glossary_state,
                },
            ),
            ocr=_stage(StageKind.OCR),
            objects=_stage(StageKind.OBJECTS),
            text_vectors=_stage(StageKind.TEXT_VECTORS),
        ),
        visual_dense_specification=_stage(
            StageKind.VISUAL_DENSE,
            parameters={
                "boundary": visual_boundary,
                "visual_specification_hash": "d" * 64,
            },
        ),
        lighthouse_specification=_stage(
            StageKind.LIGHTHOUSE,
            parameters={
                "boundary": lighthouse_boundary,
                "specification_hash": "e" * 64,
            },
        ),
        whisper_prompt_snapshot=prompt,
        executor_identity="sha256:" + "e" * 64,
    )


class _FFmpeg:
    def probe(self, _source: Path) -> MediaProbe:
        return MediaProbe(duration=10.0, width=1280, height=720, fps=25.0)

    def extract_frame(self, _source: Path, destination: Path, _timestamp: float) -> None:
        destination.write_bytes(b"jpeg")


class _Scenes:
    def detect(self, _source: Path, _duration: float) -> list[tuple[float, float]]:
        return [(0.0, 10.0)]


class _UnverifiedVectorIndex:
    available = True
    supports_generation_provenance = False

    def __init__(self) -> None:
        self.calls = 0

    def replace_video(self, _video_id: str, _segments: object) -> None:
        self.calls += 1


class _BuildOnlyExternalProvider:
    def __init__(
        self,
        *,
        kind: StageKind,
        specification: StageSpecification,
        repository: Repository | None = None,
        cancel_during_build: bool = False,
    ) -> None:
        self.kind = kind
        self.specification = specification
        self.repository = repository
        self.cancel_during_build = cancel_during_build
        self.build_calls: list[tuple[str, Path, float]] = []
        self.legacy_calls = 0

    def build_video_source(self, video_id: str, source: Path, duration: float, *args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        self.build_calls.append((video_id, source, duration))
        if self.cancel_during_build:
            assert self.repository is not None
            self.repository.request_video_index_job_cancellation("job-1")
        payload = source.read_bytes()
        descriptor: dict[str, object] = {
            "duration_seconds": duration,
            "generation_id": (
                "1" * 32
                if self.kind is StageKind.VISUAL_DENSE
                else "2" * 32
            ),
            "source_sha256": hashlib.sha256(payload).hexdigest(),
            "source_size_bytes": len(payload),
            "specification_hash": (
                self.specification.parameters["visual_specification_hash"]
                if self.kind is StageKind.VISUAL_DENSE
                else self.specification.parameters["specification_hash"]
            ),
            "video_id": video_id,
        }
        descriptor[
            "content_sha256"
            if self.kind is StageKind.VISUAL_DENSE
            else "manifest_sha256"
        ] = "c" * 64
        return descriptor

    def replace_video_source(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        self.legacy_calls += 1
        raise AssertionError("durable visual path used legacy activation")

    def prepare(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        self.legacy_calls += 1
        raise AssertionError("durable Lighthouse path used legacy activation")


def _claimed_job(
    tmp_path: Path,
    *,
    plan: VideoIndexPlanSnapshot | None = None,
) -> tuple[Repository, VideoIndexPlanSnapshot, Path]:
    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()
    source = tmp_path / "video.mp4"
    content = b"video"
    source.write_bytes(content)
    resolved_plan = plan or _plan()
    repository.create_video_with_asset_and_index_job(
        video_id="video-1",
        original_name="video.mp4",
        stored_name="video.mp4",
        media_path=str(source),
        size_bytes=len(content),
        source_sha256=hashlib.sha256(content).hexdigest(),
        plan=resolved_plan,
        job_id="job-1",
    )
    claimed = repository.claim_next_video_index_job(execution_token=TOKEN)
    assert claimed is not None
    return repository, resolved_plan, source


def test_durable_indexer_uses_only_persisted_plan_and_forwards_prompt_and_fence(
    tmp_path: Path,
) -> None:
    repository, plan, _source = _claimed_job(tmp_path)
    resolver_calls = 0
    received_prompts: list[object | None] = []

    def forbidden_resolver() -> object:
        nonlocal resolver_calls
        resolver_calls += 1
        raise AssertionError("durable execution must not resolve mutable settings")

    class Speech:
        def transcribe(
            self,
            _source: Path,
            *,
            prompt_snapshot: object | None = None,
        ) -> list[TimedText]:
            received_prompts.append(prompt_snapshot)
            return [TimedText(1.0, 3.0, "контратака", 0.9)]

    indexer = Indexer(
        repository=repository,
        media_root=tmp_path,
        thumbnails_dir=tmp_path / "thumbs",
        specification_resolver=forbidden_resolver,
        ffmpeg=_FFmpeg(),  # type: ignore[arg-type]
        scenes=_Scenes(),  # type: ignore[arg-type]
        speech=Speech(),
        vector_index=MemoryVectorIndex(),
    )

    warnings = indexer.process_durable(
        "video-1",
        context=VideoIndexExecutionContext(
            job_id="job-1",
            execution_token=TOKEN,
            plan=plan,
        ),
    )

    assert resolver_calls == 0
    assert received_prompts == [plan.whisper_prompt_snapshot]
    assert warnings == ()
    job = repository.get_video_index_job("job-1")
    video = repository.get_video("video-1")
    assert job is not None and job.state is JobState.RUNNING
    assert job.progress == 0.99 and job.stage == "finalizing"
    assert video is not None and video.status == "processing"
    assert video.stage == "finalizing"
    assert (video.duration, video.width, video.height, video.fps) == (
        10.0,
        1280,
        720,
        25.0,
    )
    with repository._connect() as connection:
        linked = connection.execute(
            """
            SELECT specifications.stage_kind, runs.state, runs.job_id
            FROM stage_runs AS runs
            JOIN stage_specifications AS specifications
              ON specifications.specification_hash = runs.specification_hash
            ORDER BY runs.created_at, runs.run_id
            """
        ).fetchall()
    assert {StageKind(row["stage_kind"]) for row in linked} == {
        StageKind.SCENES,
        StageKind.SPEECH,
        StageKind.OCR,
        StageKind.OBJECTS,
        StageKind.TEXT_VECTORS,
        StageKind.VISUAL_DENSE,
        StageKind.LIGHTHOUSE,
    }
    assert {row["job_id"] for row in linked} == {"job-1"}
    assert all(row["state"] != StageState.RUNNING.value for row in linked)
    external = {
        StageKind(row["stage_kind"]): StageState(row["state"])
        for row in linked
        if StageKind(row["stage_kind"])
        in {StageKind.VISUAL_DENSE, StageKind.LIGHTHOUSE}
    }
    assert external == {
        StageKind.VISUAL_DENSE: StageState.NOT_CONFIGURED,
        StageKind.LIGHTHOUSE: StageState.NOT_CONFIGURED,
    }


@pytest.mark.parametrize(
    ("kind", "error_code", "warning"),
    [
        (
            StageKind.VISUAL_DENSE,
            "visual_dense_provider_unavailable",
            "dense visual stage failed",
        ),
        (
            StageKind.LIGHTHOUSE,
            "lighthouse_provider_unavailable",
            "lighthouse stage failed",
        ),
    ],
)
def test_durable_indexer_marks_missing_configured_external_provider_failed(
    tmp_path: Path,
    kind: StageKind,
    error_code: str,
    warning: str,
) -> None:
    plan = _plan(
        visual_boundary=(
            "isolated-worker" if kind is StageKind.VISUAL_DENSE else "disabled"
        ),
        lighthouse_boundary=(
            "isolated-worker" if kind is StageKind.LIGHTHOUSE else "disabled"
        ),
    )
    repository, plan, _source = _claimed_job(tmp_path, plan=plan)
    indexer = Indexer(
        repository=repository,
        media_root=tmp_path,
        thumbnails_dir=tmp_path / "thumbs",
        specification_resolver=lambda: pytest.fail("resolver must not be called"),
        ffmpeg=_FFmpeg(),  # type: ignore[arg-type]
        scenes=_Scenes(),  # type: ignore[arg-type]
        vector_index=MemoryVectorIndex(),
    )

    warnings = indexer.process_durable(
        "video-1",
        context=VideoIndexExecutionContext("job-1", TOKEN, plan),
    )

    run = repository.get_latest_stage_run("video-1", kind)
    assert run is not None and run.state is StageState.FAILED
    assert run.error_code == error_code
    assert warnings == (warning,)


def test_durable_indexer_propagates_cancellation_through_stage_error_boundaries(
    tmp_path: Path,
) -> None:
    repository, plan, _source = _claimed_job(tmp_path)

    class CancellingSpeech:
        def transcribe(
            self,
            _source: Path,
            *,
            prompt_snapshot: object | None = None,
        ) -> list[TimedText]:
            assert prompt_snapshot == plan.whisper_prompt_snapshot
            repository.request_video_index_job_cancellation("job-1")
            return [TimedText(1.0, 3.0, "контратака", 0.9)]

    indexer = Indexer(
        repository=repository,
        media_root=tmp_path,
        thumbnails_dir=tmp_path / "thumbs",
        specification_resolver=lambda: pytest.fail("resolver must not be called"),
        ffmpeg=_FFmpeg(),  # type: ignore[arg-type]
        scenes=_Scenes(),  # type: ignore[arg-type]
        speech=CancellingSpeech(),
        vector_index=_UnverifiedVectorIndex(),  # type: ignore[arg-type]
    )

    with pytest.raises(JobCancelled):
        indexer.process_durable(
            "video-1",
            context=VideoIndexExecutionContext(
                job_id="job-1",
                execution_token=TOKEN,
                plan=plan,
            ),
        )

    job = repository.get_video_index_job("job-1")
    speech_run = repository.get_latest_stage_run("video-1", StageKind.SPEECH)
    assert job is not None and job.state is JobState.RUNNING
    assert job.cancel_requested_at is not None
    assert speech_run is not None and speech_run.state is StageState.RUNNING
    assert repository.get_active_segment_generation(
        "video-1",
        StageKind.SPEECH,
    ) is None


def test_durable_indexer_rejects_an_obsolete_execution_token_before_providers(
    tmp_path: Path,
) -> None:
    repository, plan, _source = _claimed_job(tmp_path)
    indexer = Indexer(
        repository=repository,
        media_root=tmp_path,
        thumbnails_dir=tmp_path / "thumbs",
        specification_resolver=lambda: pytest.fail("resolver must not be called"),
        ffmpeg=_FFmpeg(),  # type: ignore[arg-type]
        scenes=_Scenes(),  # type: ignore[arg-type]
        vector_index=_UnverifiedVectorIndex(),  # type: ignore[arg-type]
    )

    with pytest.raises(JobFenceError):
        indexer.process_durable(
            "video-1",
            context=VideoIndexExecutionContext(
                job_id="job-1",
                execution_token="worker-token-bbbbbbbbbbbbbbbb",
                plan=plan,
            ),
        )

    assert repository.get_latest_stage_run("video-1", StageKind.SCENES) is None


def test_durable_indexer_never_uses_unverified_in_process_vector_replacement(
    tmp_path: Path,
) -> None:
    repository, plan, _source = _claimed_job(tmp_path)
    vector_index = _UnverifiedVectorIndex()
    indexer = Indexer(
        repository=repository,
        media_root=tmp_path,
        thumbnails_dir=tmp_path / "thumbs",
        specification_resolver=lambda: pytest.fail("resolver must not be called"),
        ffmpeg=_FFmpeg(),  # type: ignore[arg-type]
        scenes=_Scenes(),  # type: ignore[arg-type]
        vector_index=vector_index,  # type: ignore[arg-type]
    )
    warnings: list[str] = []
    context = VideoIndexExecutionContext(
        job_id="job-1",
        execution_token=TOKEN,
        plan=plan,
    )

    indexer._replace_text_vectors(
        "video-1",
        plan.specifications,
        warnings,
        context=context,
    )

    assert vector_index.calls == 0
    assert warnings == ["text vector generation unverified"]
    run = repository.get_latest_stage_run("video-1", StageKind.TEXT_VECTORS)
    assert run is not None and run.state is StageState.FAILED


def test_durable_text_vector_heartbeat_and_publish_keep_the_job_fence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, plan, _source = _claimed_job(tmp_path)
    heartbeat_seen = Event()
    original_heartbeat = repository.heartbeat_text_vector_build

    def recording_heartbeat(
        generation_id: str,
        *,
        lease_seconds: int = 900,
        execution_token: str | None = None,
    ) -> str:
        assert execution_token == TOKEN
        heartbeat_seen.set()
        return original_heartbeat(
            generation_id,
            lease_seconds=lease_seconds,
            execution_token=execution_token,
        )

    class WaitingVectorIndex(MemoryVectorIndex):
        def build_generation(self, build_plan):  # type: ignore[no-untyped-def]
            assert heartbeat_seen.wait(timeout=2)
            return super().build_generation(build_plan)

    monkeypatch.setattr(
        repository,
        "heartbeat_text_vector_build",
        recording_heartbeat,
    )
    indexer = Indexer(
        repository=repository,
        media_root=tmp_path,
        thumbnails_dir=tmp_path / "thumbs",
        specification_resolver=lambda: pytest.fail("resolver must not be called"),
        ffmpeg=_FFmpeg(),  # type: ignore[arg-type]
        scenes=_Scenes(),  # type: ignore[arg-type]
        vector_index=WaitingVectorIndex(),
        text_vector_lease_seconds=1,
        text_vector_heartbeat_interval=0.01,
    )

    warnings = indexer.process_durable(
        "video-1",
        context=VideoIndexExecutionContext("job-1", TOKEN, plan),
    )

    assert warnings == ()
    assert heartbeat_seen.is_set()
    run = repository.get_latest_stage_run("video-1", StageKind.TEXT_VECTORS)
    assert run is not None and run.state is StageState.COMPLETE


def test_durable_text_vector_cancellation_is_not_downgraded_to_stage_failure(
    tmp_path: Path,
) -> None:
    repository, plan, _source = _claimed_job(tmp_path)

    class CancellingVectorIndex(MemoryVectorIndex):
        def build_generation(self, build_plan):  # type: ignore[no-untyped-def]
            repository.request_video_index_job_cancellation("job-1")
            return super().build_generation(build_plan)

    indexer = Indexer(
        repository=repository,
        media_root=tmp_path,
        thumbnails_dir=tmp_path / "thumbs",
        specification_resolver=lambda: pytest.fail("resolver must not be called"),
        ffmpeg=_FFmpeg(),  # type: ignore[arg-type]
        scenes=_Scenes(),  # type: ignore[arg-type]
        vector_index=CancellingVectorIndex(),
    )

    with pytest.raises(JobCancelled):
        indexer.process_durable(
            "video-1",
            context=VideoIndexExecutionContext("job-1", TOKEN, plan),
        )

    run = repository.get_latest_stage_run("video-1", StageKind.TEXT_VECTORS)
    job = repository.get_video_index_job("job-1")
    assert run is not None and run.state is StageState.RUNNING
    assert job is not None and job.cancel_requested_at is not None
    assert repository.get_active_text_vector_generation("video-1") is None


def test_durable_external_stages_build_and_commit_without_provider_activation(
    tmp_path: Path,
) -> None:
    repository, plan, _source = _claimed_job(tmp_path)
    visual = _BuildOnlyExternalProvider(
        kind=StageKind.VISUAL_DENSE,
        specification=plan.visual_dense_specification,
    )
    lighthouse = _BuildOnlyExternalProvider(
        kind=StageKind.LIGHTHOUSE,
        specification=plan.lighthouse_specification,
    )
    indexer = Indexer(
        repository=repository,
        media_root=tmp_path,
        thumbnails_dir=tmp_path / "thumbs",
        specification_resolver=lambda: pytest.fail("resolver must not be called"),
        ffmpeg=_FFmpeg(),  # type: ignore[arg-type]
        scenes=_Scenes(),  # type: ignore[arg-type]
        vector_index=MemoryVectorIndex(),
        visual_index=visual,  # type: ignore[arg-type]
        moment_retriever=lighthouse,  # type: ignore[arg-type]
    )

    warnings = indexer.process_durable(
        "video-1",
        context=VideoIndexExecutionContext("job-1", TOKEN, plan),
    )

    assert warnings == ()
    assert visual.legacy_calls == lighthouse.legacy_calls == 0
    assert len(visual.build_calls) == len(lighthouse.build_calls) == 1
    visual_run = repository.get_latest_stage_run(
        "video-1", StageKind.VISUAL_DENSE
    )
    lighthouse_run = repository.get_latest_stage_run(
        "video-1", StageKind.LIGHTHOUSE
    )
    assert visual_run is not None and visual_run.state is StageState.COMPLETE
    assert lighthouse_run is not None and lighthouse_run.state is StageState.COMPLETE
    assert repository.get_active_external_index_generation(
        "video-1", StageKind.VISUAL_DENSE
    ) is None
    assert repository.get_active_external_index_generation(
        "video-1", StageKind.LIGHTHOUSE
    ) is None

    repository.complete_video_index_job("job-1", execution_token=TOKEN)

    assert repository.get_active_external_index_generation(
        "video-1", StageKind.VISUAL_DENSE
    ) is not None
    assert repository.get_active_external_index_generation(
        "video-1", StageKind.LIGHTHOUSE
    ) is not None


def test_external_cancel_after_build_is_observed_before_receipt_publication(
    tmp_path: Path,
) -> None:
    repository, plan, _source = _claimed_job(tmp_path)
    visual = _BuildOnlyExternalProvider(
        kind=StageKind.VISUAL_DENSE,
        specification=plan.visual_dense_specification,
        repository=repository,
        cancel_during_build=True,
    )
    indexer = Indexer(
        repository=repository,
        media_root=tmp_path,
        thumbnails_dir=tmp_path / "thumbs",
        specification_resolver=lambda: pytest.fail("resolver must not be called"),
        ffmpeg=_FFmpeg(),  # type: ignore[arg-type]
        scenes=_Scenes(),  # type: ignore[arg-type]
        vector_index=MemoryVectorIndex(),
        visual_index=visual,  # type: ignore[arg-type]
    )

    with pytest.raises(JobCancelled):
        indexer.process_durable(
            "video-1",
            context=VideoIndexExecutionContext("job-1", TOKEN, plan),
        )

    run = repository.get_latest_stage_run("video-1", StageKind.VISUAL_DENSE)
    assert run is not None and run.state is StageState.RUNNING
    with repository._connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM external_index_generations"
        ).fetchone()[0] == 0
