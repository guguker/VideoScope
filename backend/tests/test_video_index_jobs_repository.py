from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import sqlite3

import pytest

from videoscope.artifacts import (
    IndexingSpecifications,
    StageKind,
    StageSpecification,
    StageState,
    TextVectorBuildReceipt,
    TextVectorIndexSpecification,
)
from videoscope.jobs import (
    JobFenceError,
    JobState,
    JobTransitionError,
    VideoIndexIntent,
    VideoIndexPlanSnapshot,
)
from videoscope.providers.whisper import WhisperPromptSnapshot
from videoscope.repository import LATEST_SCHEMA_VERSION, Repository, SegmentRecord


TOKEN_A = "worker-token-aaaaaaaaaaaaaaaa"
TOKEN_B = "worker-token-bbbbbbbbbbbbbbbb"


def _stage(
    kind: StageKind,
    *,
    parameters: dict[str, object] | None = None,
    revision: str = "v1",
) -> StageSpecification:
    return StageSpecification(
        kind=kind,
        schema_version=1,
        implementation_revision=f"test.{kind.value}.{revision}",
        parameters=parameters or {},
        dependencies={"videoscope": "test"},
    )


def _plan(
    *,
    executor_identity: str = "videoscope.indexer.test-v1",
    stage_revision: str = "v1",
) -> VideoIndexPlanSnapshot:
    prompt_text = "Мозгов, контратака"
    prompt = WhisperPromptSnapshot(
        effective_prompt=prompt_text,
        effective_prompt_sha256=hashlib.sha256(prompt_text.encode()).hexdigest(),
        glossary_state="ready",
    )
    resolved_executor_identity = (
        executor_identity
        if executor_identity.startswith("sha256:")
        else "sha256:" + hashlib.sha256(executor_identity.encode()).hexdigest()
    )
    return VideoIndexPlanSnapshot(
        schema_version=2,
        specifications=IndexingSpecifications(
            scenes=_stage(StageKind.SCENES, revision=stage_revision),
            speech=_stage(
                StageKind.SPEECH,
                parameters={
                    "effective_prompt_sha256": prompt.effective_prompt_sha256,
                    "glossary_state": prompt.glossary_state,
                },
                revision=stage_revision,
            ),
            ocr=_stage(StageKind.OCR, revision=stage_revision),
            objects=_stage(StageKind.OBJECTS, revision=stage_revision),
            text_vectors=_stage(StageKind.TEXT_VECTORS, revision=stage_revision),
        ),
        visual_dense_specification=_stage(
            StageKind.VISUAL_DENSE,
            revision=stage_revision,
        ),
        lighthouse_specification=_stage(
            StageKind.LIGHTHOUSE,
            revision=stage_revision,
        ),
        whisper_prompt_snapshot=prompt,
        executor_identity=resolved_executor_identity,
    )


def _repository(tmp_path) -> Repository:  # type: ignore[no-untyped-def]
    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()
    return repository


def _create_ingest_job(
    repository: Repository,
    *,
    video_id: str = "video-1",
    job_id: str = "job-1",
    plan: VideoIndexPlanSnapshot | None = None,
):  # type: ignore[no-untyped-def]
    return repository.create_video_with_asset_and_index_job(
        video_id=video_id,
        original_name=f"{video_id}.mp4",
        stored_name=f"{video_id}.mp4",
        media_path=f"/tmp/{video_id}.mp4",
        size_bytes=1024,
        source_sha256="a" * 64,
        plan=plan or _plan(),
        job_id=job_id,
    )


def _segment(
    segment_id: str,
    *,
    modality: str,
    text: str,
    thumbnail_path: str | None = None,
) -> SegmentRecord:
    return SegmentRecord(
        id=segment_id,
        video_id="video-1",
        start=1.0,
        end=2.0,
        modality=modality,
        text=text,
        confidence=0.9,
        metadata={},
        thumbnail_path=thumbnail_path,
    )


def _create_ready_legacy_video(repository: Repository) -> None:
    repository.create_video_with_asset(
        video_id="video-1",
        original_name="video-1.mp4",
        stored_name="video-1.mp4",
        media_path="/tmp/video-1.mp4",
        size_bytes=1024,
        source_sha256="a" * 64,
    )
    repository.update_video(
        "video-1",
        status="ready",
        progress=1.0,
        stage="ready",
    )


def _publish_legacy_segment_generation(
    repository: Repository,
    specification: StageSpecification,
    *,
    run_id: str,
    generation_id: str,
    segment: SegmentRecord,
    video_thumbnail_path: str | None | object = None,
):  # type: ignore[no-untyped-def]
    repository.create_stage_run(
        video_id="video-1",
        specification=specification,
        run_id=run_id,
    )
    repository.transition_stage_run(run_id, StageState.RUNNING)
    keyword: dict[str, object] = {}
    if specification.kind is StageKind.SCENES:
        keyword["video_thumbnail_path"] = video_thumbnail_path
    return repository.commit_segment_generation(
        run_id,
        generation_id=generation_id,
        segments=[segment],
        **keyword,
    )


def _start_owned_stage_run(
    repository: Repository,
    *,
    job_id: str,
    specification: StageSpecification,
    run_id: str,
    execution_token: str = TOKEN_B,
):  # type: ignore[no-untyped-def]
    run = repository.create_stage_run(
        video_id="video-1",
        specification=specification,
        run_id=run_id,
        job_id=job_id,
        execution_token=execution_token,
    )
    return repository.transition_stage_run(
        run.run_id,
        StageState.RUNNING,
        execution_token=execution_token,
    )


def _planned_stage_specifications(
    plan: VideoIndexPlanSnapshot,
) -> tuple[StageSpecification, ...]:
    return (
        *plan.specifications.segment_specifications,
        plan.specifications.text_vectors,
        plan.visual_dense_specification,
        plan.lighthouse_specification,
    )


def _record_terminal_job_receipts(
    repository: Repository,
    *,
    job_id: str,
    plan: VideoIndexPlanSnapshot,
    execution_token: str,
    specifications: tuple[StageSpecification, ...] | None = None,
) -> None:
    resolved = specifications or _planned_stage_specifications(plan)
    for index, specification in enumerate(resolved):
        run = repository.create_stage_run(
            video_id="video-1",
            specification=specification,
            run_id=f"receipt-{index}-{specification.kind.value}",
            job_id=job_id,
            execution_token=execution_token,
        )
        repository.transition_stage_run(
            run.run_id,
            StageState.NOT_CONFIGURED,
            execution_token=execution_token,
        )


def _receipt(build) -> TextVectorBuildReceipt:  # type: ignore[no-untyped-def]
    return TextVectorBuildReceipt(
        generation_id=build.generation_id,
        index_specification_hash=build.index_specification.specification_hash,
        point_count=len(build.points),
        point_manifest_sha256=build.point_manifest_sha256,
        vector_manifest_sha256="f" * 64,
    )


def test_schema_v9_adds_durable_jobs_without_fabricating_legacy_history(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    repository = _repository(tmp_path)
    repository.create_video(
        video_id="legacy-video",
        original_name="legacy.mp4",
        stored_name="legacy.mp4",
        media_path="/tmp/legacy.mp4",
        size_bytes=1024,
    )

    repository.initialize()

    assert repository.schema_version() == LATEST_SCHEMA_VERSION == 9
    assert repository.list_video_index_jobs(video_id="legacy-video", limit=10) == ()
    with repository._connect() as connection:
        job_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(video_index_jobs)")
        }
        stage_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(stage_runs)")
        }
        generation_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(segment_generations)")
        }
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }
        triggers = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            )
        }
    assert {"plan_json", "execution_token", "retry_of_job_id"} <= job_columns
    assert "job_id" in stage_columns
    assert {"updates_video_thumbnail", "video_thumbnail_path"} <= generation_columns
    assert {
        "idx_video_index_jobs_active_video",
        "idx_video_index_jobs_active_idempotency",
        "idx_video_index_jobs_retry_child",
    } <= indexes
    assert {
        "active_segment_generations_job_owner_insert",
        "active_segment_generations_job_owner_update",
        "active_text_vector_generations_job_owner_insert",
        "active_text_vector_generations_job_owner_update",
        "video_index_jobs_require_exact_stage_receipts",
    } <= triggers


def test_schema_v9_repairs_the_exact_stage_receipt_completion_trigger(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    repository = _repository(tmp_path)
    with repository._connect() as connection:
        connection.execute(
            "DROP TRIGGER video_index_jobs_require_exact_stage_receipts"
        )

    repository.initialize()

    with repository._connect() as connection:
        trigger = connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'trigger'
              AND name = 'video_index_jobs_require_exact_stage_receipts'
            """
        ).fetchone()
    assert trigger is not None


def test_upload_persists_video_asset_and_ingest_job_atomically(tmp_path) -> None:  # type: ignore[no-untyped-def]
    repository = _repository(tmp_path)
    plan = _plan()

    video, job = _create_ingest_job(repository, plan=plan)

    assert video.id == "video-1"
    assert job.job_id == "job-1"
    assert job.intent is VideoIndexIntent.INGEST
    assert job.state is JobState.QUEUED
    assert job.source_sha256 == "a" * 64
    assert job.plan_hash == plan.plan_hash
    assert repository.get_video_index_job("job-1") == job
    assert repository.get_video_index_job_plan("job-1") == plan
    with repository._connect() as connection:
        plan_json = connection.execute(
            "SELECT plan_json FROM video_index_jobs WHERE job_id = 'job-1'"
        ).fetchone()[0]
    assert plan_json == plan.canonical_json


def test_upload_job_failure_rolls_back_video_and_new_asset(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    repository = _repository(tmp_path)

    def reject_job(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("injected job failure")

    monkeypatch.setattr(repository, "_insert_video_index_job", reject_job)

    with pytest.raises(RuntimeError, match="injected job failure"):
        _create_ingest_job(repository)

    assert repository.get_video("video-1") is None
    with repository._connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM assets").fetchone()[0] == 0
        assert (
            connection.execute("SELECT COUNT(*) FROM video_index_jobs").fetchone()[0]
            == 0
        )


def test_reindex_atomically_snapshots_normalized_state_and_deduplicates(tmp_path) -> None:  # type: ignore[no-untyped-def]
    repository = _repository(tmp_path)
    _video, ingest = _create_ingest_job(repository)
    running = repository.claim_next_video_index_job(
        execution_token=TOKEN_A,
        stage="probe",
    )
    assert running is not None and running.job_id == ingest.job_id
    _record_terminal_job_receipts(
        repository,
        job_id=ingest.job_id,
        plan=_plan(),
        execution_token=TOKEN_A,
    )
    repository.complete_video_index_job(ingest.job_id, execution_token=TOKEN_A)
    repository.update_video(
        "video-1",
        status="ready",
        progress=1.0,
        stage="ready",
        error="raw warning; must not become durable state",
    )
    plan = _plan(executor_identity="videoscope.indexer.test-v2")

    first = repository.enqueue_video_reindex_job(
        "video-1",
        plan=plan,
        job_id="reindex-1",
    )
    second = repository.enqueue_video_reindex_job(
        "video-1",
        plan=plan,
        job_id="ignored-by-dedup",
    )

    assert second == first
    assert first.intent is VideoIndexIntent.REINDEX
    assert first.prior_video_state is not None
    assert first.prior_video_state.status == "ready"
    assert first.prior_video_state.progress == 1.0
    assert first.prior_video_state.stage == "ready"
    assert first.prior_video_state.error_code is None
    video = repository.get_video("video-1")
    assert video is not None
    assert (video.status, video.progress, video.stage, video.error) == (
        "queued",
        0.0,
        "queued",
        None,
    )
    jobs = repository.list_video_index_jobs(video_id="video-1", limit=10)
    assert [job.job_id for job in jobs] == ["job-1", "reindex-1"]


def test_concurrent_reindex_creates_exactly_one_active_job(tmp_path) -> None:  # type: ignore[no-untyped-def]
    repository = _repository(tmp_path)
    _video, ingest = _create_ingest_job(repository)
    repository.claim_next_video_index_job(execution_token=TOKEN_A, stage="probe")
    _record_terminal_job_receipts(
        repository,
        job_id=ingest.job_id,
        plan=_plan(),
        execution_token=TOKEN_A,
    )
    repository.complete_video_index_job(ingest.job_id, execution_token=TOKEN_A)
    plan = _plan(executor_identity="videoscope.indexer.concurrent-v1")

    def enqueue(index: int):  # type: ignore[no-untyped-def]
        return repository.enqueue_video_reindex_job(
            "video-1",
            plan=plan,
            job_id=f"concurrent-{index}",
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        jobs = list(executor.map(enqueue, range(8)))

    assert len({job.job_id for job in jobs}) == 1
    with repository._connect() as connection:
        active_count = connection.execute(
            """
            SELECT COUNT(*) FROM video_index_jobs
            WHERE video_id = 'video-1' AND state IN ('queued', 'running')
            """
        ).fetchone()[0]
    assert active_count == 1


def test_latest_video_index_jobs_bulk_is_bounded_exact_and_quarantine_closed(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    repository = _repository(tmp_path)
    _video, first = _create_ingest_job(
        repository,
        video_id="video-1",
        job_id="job-video-1",
    )
    repository.claim_next_video_index_job(
        execution_token=TOKEN_A,
        stage="probe",
    )
    _record_terminal_job_receipts(
        repository,
        job_id=first.job_id,
        plan=_plan(),
        execution_token=TOKEN_A,
    )
    repository.complete_video_index_job(first.job_id, execution_token=TOKEN_A)
    latest_first = repository.enqueue_video_reindex_job(
        "video-1",
        plan=_plan(executor_identity="videoscope.indexer.latest-v2"),
        job_id="reindex-video-1",
    )
    _other_video, latest_second = _create_ingest_job(
        repository,
        video_id="video-2",
        job_id="job-video-2",
    )

    latest = repository.get_latest_video_index_jobs(
        ("video-2", "missing-video", "video-1")
    )

    assert latest == {
        "video-1": latest_first,
        "video-2": latest_second,
    }
    assert repository.get_latest_video_index_jobs(()) == {}
    with pytest.raises(ValueError, match="unique"):
        repository.get_latest_video_index_jobs(("video-1", "video-1"))
    with pytest.raises(ValueError, match="1000"):
        repository.get_latest_video_index_jobs(
            tuple(f"video-{index}" for index in range(1001))
        )
    with repository._connect() as connection:
        sequence = connection.execute(
            "SELECT sequence FROM video_index_jobs WHERE job_id = ?",
            (latest_first.job_id,),
        ).fetchone()[0]
        connection.execute(
            """
            INSERT INTO video_index_job_quarantines (
                job_sequence, error_code, quarantined_at
            ) VALUES (?, 'job_record_corrupt', '2026-01-01T00:00:00+00:00')
            """,
            (sequence,),
        )
    with pytest.raises(ValueError, match="quarantined"):
        repository.get_latest_video_index_jobs(("video-1", "video-2"))


def test_persisted_plan_corruption_fails_closed_without_leaking_value(tmp_path) -> None:  # type: ignore[no-untyped-def]
    repository = _repository(tmp_path)
    _create_ingest_job(repository)
    with repository._connect() as connection:
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(
            "DROP TRIGGER video_index_jobs_identity_immutable_update"
        )
        connection.execute(
            "UPDATE video_index_jobs SET plan_json = ? WHERE job_id = ?",
            ("private-corrupt-plan", "job-1"),
        )

    with pytest.raises(ValueError) as captured:
        repository.get_video_index_job("job-1")

    assert str(captured.value) == "persisted video index job is corrupt"
    assert "private-corrupt-plan" not in str(captured.value)


def test_latest_video_index_jobs_bulk_fails_closed_on_any_corrupt_latest_row(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    repository = _repository(tmp_path)
    _create_ingest_job(repository, video_id="video-corrupt", job_id="job-corrupt")
    _create_ingest_job(repository, video_id="video-valid", job_id="job-valid")
    with repository._connect() as connection:
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute("DROP TRIGGER video_index_jobs_identity_immutable_update")
        connection.execute(
            "UPDATE video_index_jobs SET plan_json = ? WHERE job_id = ?",
            ("private-corrupt-plan", "job-corrupt"),
        )

    with pytest.raises(ValueError) as captured:
        repository.get_latest_video_index_jobs(
            ("video-valid", "video-corrupt")
        )

    assert str(captured.value) == "persisted video index job is corrupt"
    assert "private-corrupt-plan" not in str(captured.value)


def test_claim_quarantines_corrupt_head_and_continues_to_valid_job(tmp_path) -> None:  # type: ignore[no-untyped-def]
    repository = _repository(tmp_path)
    _create_ingest_job(repository, video_id="video-corrupt", job_id="job-corrupt")
    _create_ingest_job(repository, video_id="video-valid", job_id="job-valid")
    with repository._connect() as connection:
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute("DROP TRIGGER video_index_jobs_identity_immutable_update")
        connection.execute(
            "UPDATE video_index_jobs SET plan_json = ? WHERE job_id = ?",
            ("private-corrupt-plan", "job-corrupt"),
        )

    claimed = repository.claim_next_video_index_job(
        execution_token=TOKEN_A,
        stage="probe",
        scan_limit=16,
    )

    assert claimed is not None and claimed.job_id == "job-valid"
    assert repository.count_video_index_job_quarantines() == 1
    with repository._connect() as connection:
        quarantined = connection.execute(
            """
            SELECT quarantines.error_code, jobs.job_id
            FROM video_index_job_quarantines AS quarantines
            JOIN video_index_jobs AS jobs
              ON jobs.sequence = quarantines.job_sequence
            """
        ).fetchone()
        assert tuple(quarantined) == ("job_record_corrupt", "job-corrupt")
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute("DELETE FROM video_index_job_quarantines")


def test_claim_and_checkpoint_are_fenced_and_update_video_atomically(tmp_path) -> None:  # type: ignore[no-untyped-def]
    repository = _repository(tmp_path)
    _video, queued = _create_ingest_job(repository)

    running = repository.claim_next_video_index_job(
        execution_token=TOKEN_A,
        stage="probe",
    )
    assert running is not None
    advanced = repository.checkpoint_video_index_job(
        queued.job_id,
        execution_token=TOKEN_A,
        progress=0.45,
        stage="speech",
    )

    assert (advanced.progress, advanced.stage) == (0.45, "speech")
    video = repository.get_video("video-1")
    assert video is not None
    assert (video.status, video.progress, video.stage) == (
        "processing",
        0.45,
        "speech",
    )
    with pytest.raises(JobFenceError):
        repository.checkpoint_video_index_job(
            queued.job_id,
            execution_token=TOKEN_B,
            progress=0.5,
            stage="ocr",
        )
    with pytest.raises(JobTransitionError, match="monotonic"):
        repository.checkpoint_video_index_job(
            queued.job_id,
            execution_token=TOKEN_A,
            progress=0.4,
            stage="ocr",
        )
    assert repository.get_video_index_job(queued.job_id) == advanced


def test_reindex_cancellation_is_cooperative_and_restores_prior_state(tmp_path) -> None:  # type: ignore[no-untyped-def]
    repository = _repository(tmp_path)
    _video, ingest = _create_ingest_job(repository)
    repository.claim_next_video_index_job(execution_token=TOKEN_A, stage="probe")
    _record_terminal_job_receipts(
        repository,
        job_id=ingest.job_id,
        plan=_plan(),
        execution_token=TOKEN_A,
    )
    repository.complete_video_index_job(ingest.job_id, execution_token=TOKEN_A)
    reindex = repository.enqueue_video_reindex_job(
        "video-1",
        plan=_plan(executor_identity="videoscope.indexer.cancel-v1"),
        job_id="reindex-cancel",
    )
    repository.claim_next_video_index_job(execution_token=TOKEN_B, stage="probe")
    repository.checkpoint_video_index_job(
        reindex.job_id,
        execution_token=TOKEN_B,
        progress=0.25,
        stage="scenes",
    )

    requested = repository.request_video_index_job_cancellation(reindex.job_id)
    assert requested.state is JobState.RUNNING
    assert requested.cancel_requested_at is not None
    with pytest.raises(JobTransitionError, match="cancellation"):
        repository.checkpoint_video_index_job(
            reindex.job_id,
            execution_token=TOKEN_B,
            progress=0.3,
            stage="speech",
        )
    cancelled = repository.cancel_video_index_job(
        reindex.job_id,
        execution_token=TOKEN_B,
    )

    assert cancelled.state is JobState.CANCELLED
    video = repository.get_video("video-1")
    assert video is not None
    assert (video.status, video.progress, video.stage, video.error) == (
        "ready",
        1.0,
        "ready",
        None,
    )


def test_job_owned_scene_generation_stays_inactive_and_cancel_preserves_prior(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    repository = _repository(tmp_path)
    plan = _plan(executor_identity="videoscope.indexer.staged-cancel-v1")
    _create_ready_legacy_video(repository)
    old_thumbnail = "/tmp/thumbnails/video-1/old-scene.jpg"
    old = _publish_legacy_segment_generation(
        repository,
        plan.specifications.scenes,
        run_id="legacy-scene-run",
        generation_id="legacy-scene-generation",
        segment=_segment(
            "legacy-scene",
            modality="scene",
            text="old scene",
            thumbnail_path=old_thumbnail,
        ),
        video_thumbnail_path=old_thumbnail,
    )
    job = repository.enqueue_video_reindex_job(
        "video-1",
        plan=plan,
        job_id="reindex-staged-cancel",
    )
    repository.claim_next_video_index_job(
        execution_token=TOKEN_B,
        stage="scenes",
    )
    run = _start_owned_stage_run(
        repository,
        job_id=job.job_id,
        specification=plan.specifications.scenes,
        run_id="owned-scene-run",
    )
    new_thumbnail = "/tmp/thumbnails/video-1/new-scene.jpg"

    staged = repository.commit_segment_generation(
        run.run_id,
        generation_id="owned-scene-generation",
        segments=[
            _segment(
                "owned-scene",
                modality="scene",
                text="new scene",
                thumbnail_path=new_thumbnail,
            )
        ],
        video_thumbnail_path=new_thumbnail,
        execution_token=TOKEN_B,
    )

    assert repository.get_segment_generation(staged.generation_id) == staged
    assert repository.get_active_segment_generation(
        "video-1", StageKind.SCENES
    ) == old
    video = repository.get_video("video-1")
    assert video is not None and video.thumbnail_path == old_thumbnail
    with repository._connect() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="job is not complete"):
            connection.execute(
                """
                UPDATE active_segment_generations
                SET generation_id = ?, activated_at = ?
                WHERE video_id = ? AND stage_kind = ?
                """,
                (
                    staged.generation_id,
                    staged.completed_at,
                    "video-1",
                    StageKind.SCENES.value,
                ),
            )

    repository.request_video_index_job_cancellation(job.job_id)
    repository.cancel_video_index_job(job.job_id, execution_token=TOKEN_B)

    assert repository.get_active_segment_generation(
        "video-1", StageKind.SCENES
    ) == old
    video = repository.get_video("video-1")
    assert video is not None and video.thumbnail_path == old_thumbnail


def test_job_completion_atomically_activates_staged_segments_text_and_thumbnail(
    tmp_path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    repository = _repository(tmp_path)
    plan = _plan(executor_identity="videoscope.indexer.staged-complete-v1")
    _create_ready_legacy_video(repository)
    old_thumbnail = "/tmp/thumbnails/video-1/old-scene.jpg"
    old_scene = _publish_legacy_segment_generation(
        repository,
        plan.specifications.scenes,
        run_id="legacy-scene-run",
        generation_id="legacy-scene-generation",
        segment=_segment(
            "legacy-scene",
            modality="scene",
            text="old scene",
            thumbnail_path=old_thumbnail,
        ),
        video_thumbnail_path=old_thumbnail,
    )
    old_speech = _publish_legacy_segment_generation(
        repository,
        plan.specifications.speech,
        run_id="legacy-speech-run",
        generation_id="legacy-speech-generation",
        segment=_segment(
            "legacy-speech",
            modality="speech",
            text="old transcript",
        ),
    )
    repository.create_stage_run(
        video_id="video-1",
        specification=plan.specifications.text_vectors,
        run_id="legacy-text-run",
    )
    repository.transition_stage_run("legacy-text-run", StageState.RUNNING)
    index_specification = TextVectorIndexSpecification(
        embedding_identity="tests/hash-embedding@v1",
        dimensions=8,
    )
    old_build = repository.reserve_text_vector_generation(
        "legacy-text-run",
        index_specification=index_specification,
        semantic_specifications=plan.specifications.semantic_segment_specifications,
        generation_id="legacy-text-generation",
    )
    old_text = repository.commit_text_vector_generation(
        "legacy-text-run",
        receipt=_receipt(old_build),
    )
    job = repository.enqueue_video_reindex_job(
        "video-1",
        plan=plan,
        job_id="reindex-staged-complete",
    )
    repository.claim_next_video_index_job(
        execution_token=TOKEN_B,
        stage="scenes",
    )
    scene_run = _start_owned_stage_run(
        repository,
        job_id=job.job_id,
        specification=plan.specifications.scenes,
        run_id="owned-scene-run",
    )
    new_thumbnail = "/tmp/thumbnails/video-1/new-scene.jpg"
    new_scene = repository.commit_segment_generation(
        scene_run.run_id,
        generation_id="owned-scene-generation",
        segments=[
            _segment(
                "owned-scene",
                modality="scene",
                text="new scene",
                thumbnail_path=new_thumbnail,
            )
        ],
        video_thumbnail_path=new_thumbnail,
        execution_token=TOKEN_B,
    )
    speech_run = _start_owned_stage_run(
        repository,
        job_id=job.job_id,
        specification=plan.specifications.speech,
        run_id="owned-speech-run",
    )
    new_speech = repository.commit_segment_generation(
        speech_run.run_id,
        generation_id="owned-speech-generation",
        segments=[
            _segment(
                "owned-speech",
                modality="speech",
                text="new transcript",
            )
        ],
        execution_token=TOKEN_B,
    )
    text_run = _start_owned_stage_run(
        repository,
        job_id=job.job_id,
        specification=plan.specifications.text_vectors,
        run_id="owned-text-run",
    )
    new_build = repository.reserve_text_vector_generation(
        text_run.run_id,
        index_specification=index_specification,
        semantic_specifications=plan.specifications.semantic_segment_specifications,
        generation_id="owned-text-generation",
        execution_token=TOKEN_B,
    )
    by_kind = {item.stage_kind: item for item in new_build.inputs}
    assert by_kind[StageKind.SPEECH].segment_generation_id == new_speech.generation_id
    new_text = repository.commit_text_vector_generation(
        text_run.run_id,
        receipt=_receipt(new_build),
        execution_token=TOKEN_B,
    )
    _record_terminal_job_receipts(
        repository,
        job_id=job.job_id,
        plan=plan,
        execution_token=TOKEN_B,
        specifications=tuple(
            specification
            for specification in _planned_stage_specifications(plan)
            if specification.kind
            not in {
                StageKind.SCENES,
                StageKind.SPEECH,
                StageKind.TEXT_VECTORS,
            }
        ),
    )

    assert repository.get_active_segment_generation(
        "video-1", StageKind.SCENES
    ) == old_scene
    assert repository.get_active_segment_generation(
        "video-1", StageKind.SPEECH
    ) == old_speech
    assert repository.get_active_text_vector_generation("video-1") == old_text
    video = repository.get_video("video-1")
    assert video is not None and video.thumbnail_path == old_thumbnail
    with repository._connect() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="job is not complete"):
            connection.execute(
                """
                UPDATE active_text_vector_generations
                SET generation_id = ?, activated_at = ?
                WHERE video_id = ?
                """,
                (new_text.generation_id, new_text.completed_at, "video-1"),
            )

    activate = repository._activate_video_index_job_outputs

    def fail_after_activation(*args, **kwargs):  # type: ignore[no-untyped-def]
        activate(*args, **kwargs)
        raise RuntimeError("injected post-activation failure")

    monkeypatch.setattr(
        repository,
        "_activate_video_index_job_outputs",
        fail_after_activation,
    )
    with pytest.raises(RuntimeError, match="post-activation"):
        repository.complete_video_index_job(job.job_id, execution_token=TOKEN_B)
    assert repository.get_video_index_job(job.job_id).state is JobState.RUNNING  # type: ignore[union-attr]
    assert repository.get_active_segment_generation(
        "video-1", StageKind.SCENES
    ) == old_scene
    assert repository.get_active_text_vector_generation("video-1") == old_text
    monkeypatch.setattr(repository, "_activate_video_index_job_outputs", activate)

    completed = repository.complete_video_index_job(
        job.job_id,
        execution_token=TOKEN_B,
    )

    assert completed.state is JobState.COMPLETE
    assert repository.get_active_segment_generation(
        "video-1", StageKind.SCENES
    ) == new_scene
    assert repository.get_active_segment_generation(
        "video-1", StageKind.SPEECH
    ) == new_speech
    assert repository.get_active_text_vector_generation("video-1") == new_text
    video = repository.get_video("video-1")
    assert video is not None and video.thumbnail_path == new_thumbnail


def test_changed_plan_without_outputs_releases_no_mismatched_active_generations(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    repository = _repository(tmp_path)
    old_plan = _plan(
        executor_identity="videoscope.indexer.old-plan-v1",
        stage_revision="old",
    )
    _create_ready_legacy_video(repository)
    old_thumbnail = "/tmp/thumbnails/video-1/old-scene.jpg"
    segment_details = (
        (StageKind.SCENES, "scene", old_thumbnail),
        (StageKind.SPEECH, "speech", None),
        (StageKind.OCR, "ocr", None),
        (StageKind.OBJECTS, "objects", None),
    )
    for stage_kind, modality, thumbnail_path in segment_details:
        specification = old_plan.specifications.for_kind(stage_kind)
        _publish_legacy_segment_generation(
            repository,
            specification,
            run_id=f"legacy-{stage_kind.value}-run",
            generation_id=f"legacy-{stage_kind.value}-generation",
            segment=_segment(
                f"legacy-{stage_kind.value}-segment",
                modality=modality,
                text=f"old {modality}",
                thumbnail_path=thumbnail_path,
            ),
            video_thumbnail_path=thumbnail_path,
        )
    repository.create_stage_run(
        video_id="video-1",
        specification=old_plan.specifications.text_vectors,
        run_id="legacy-text-run",
    )
    repository.transition_stage_run("legacy-text-run", StageState.RUNNING)
    index_specification = TextVectorIndexSpecification(
        embedding_identity="tests/hash-embedding@v1",
        dimensions=8,
    )
    old_build = repository.reserve_text_vector_generation(
        "legacy-text-run",
        index_specification=index_specification,
        semantic_specifications=(
            old_plan.specifications.semantic_segment_specifications
        ),
        generation_id="legacy-text-generation",
    )
    repository.commit_text_vector_generation(
        "legacy-text-run",
        receipt=_receipt(old_build),
    )
    new_plan = _plan(
        executor_identity="videoscope.indexer.new-plan-v2",
        stage_revision="new",
    )
    job = repository.enqueue_video_reindex_job(
        "video-1",
        plan=new_plan,
        job_id="reindex-no-compatible-output",
    )
    repository.claim_next_video_index_job(
        execution_token=TOKEN_B,
        stage="finalizing",
    )
    _record_terminal_job_receipts(
        repository,
        job_id=job.job_id,
        plan=new_plan,
        execution_token=TOKEN_B,
    )

    repository.complete_video_index_job(job.job_id, execution_token=TOKEN_B)

    for stage_kind, _modality, _thumbnail_path in segment_details:
        assert repository.get_active_segment_generation(
            "video-1",
            stage_kind,
        ) is None
    assert repository.get_active_text_vector_generation("video-1") is None
    video = repository.get_video("video-1")
    assert video is not None and video.thumbnail_path is None


def test_active_reads_reject_cancelled_job_owned_segment_and_text_pointers(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    repository = _repository(tmp_path)
    plan = _plan(executor_identity="videoscope.indexer.corrupt-owner-v1")
    _create_ready_legacy_video(repository)
    _publish_legacy_segment_generation(
        repository,
        plan.specifications.speech,
        run_id="legacy-speech-run",
        generation_id="legacy-speech-generation",
        segment=_segment(
            "legacy-speech",
            modality="speech",
            text="old transcript",
        ),
    )
    repository.create_stage_run(
        video_id="video-1",
        specification=plan.specifications.text_vectors,
        run_id="legacy-text-run",
    )
    repository.transition_stage_run("legacy-text-run", StageState.RUNNING)
    index_specification = TextVectorIndexSpecification(
        embedding_identity="tests/hash-embedding@v1",
        dimensions=8,
    )
    legacy_build = repository.reserve_text_vector_generation(
        "legacy-text-run",
        index_specification=index_specification,
        semantic_specifications=plan.specifications.semantic_segment_specifications,
        generation_id="legacy-text-generation",
    )
    repository.commit_text_vector_generation(
        "legacy-text-run",
        receipt=_receipt(legacy_build),
    )
    job = repository.enqueue_video_reindex_job(
        "video-1",
        plan=plan,
        job_id="reindex-corrupt-owner",
    )
    repository.claim_next_video_index_job(
        execution_token=TOKEN_B,
        stage="speech",
    )
    speech_run = _start_owned_stage_run(
        repository,
        job_id=job.job_id,
        specification=plan.specifications.speech,
        run_id="owned-speech-run",
    )
    staged_speech = repository.commit_segment_generation(
        speech_run.run_id,
        generation_id="owned-speech-generation",
        segments=[
            _segment(
                "owned-speech",
                modality="speech",
                text="new transcript",
            )
        ],
        execution_token=TOKEN_B,
    )
    text_run = _start_owned_stage_run(
        repository,
        job_id=job.job_id,
        specification=plan.specifications.text_vectors,
        run_id="owned-text-run",
    )
    staged_build = repository.reserve_text_vector_generation(
        text_run.run_id,
        index_specification=index_specification,
        semantic_specifications=plan.specifications.semantic_segment_specifications,
        generation_id="owned-text-generation",
        execution_token=TOKEN_B,
    )
    staged_text = repository.commit_text_vector_generation(
        text_run.run_id,
        receipt=_receipt(staged_build),
        execution_token=TOKEN_B,
    )
    with repository._connect() as connection:
        connection.execute(
            "DROP TRIGGER active_segment_generations_job_owner_update"
        )
        connection.execute(
            "DROP TRIGGER active_text_vector_generations_job_owner_update"
        )
        connection.execute(
            """
            UPDATE active_segment_generations
            SET generation_id = ?, activated_at = ?
            WHERE video_id = ? AND stage_kind = ?
            """,
            (
                staged_speech.generation_id,
                staged_speech.completed_at,
                "video-1",
                StageKind.SPEECH.value,
            ),
        )
        connection.execute(
            """
            UPDATE active_text_vector_generations
            SET generation_id = ?, activated_at = ?
            WHERE video_id = ?
            """,
            (staged_text.generation_id, staged_text.completed_at, "video-1"),
        )
    repository.request_video_index_job_cancellation(job.job_id)
    repository.cancel_video_index_job(job.job_id, execution_token=TOKEN_B)

    with pytest.raises(ValueError, match="owner job is not complete"):
        repository.get_active_segment_generation("video-1", StageKind.SPEECH)
    with pytest.raises(ValueError, match="owner job is not complete"):
        repository.get_active_text_vector_generation("video-1")
    with pytest.raises(ValueError, match="owner job is not complete"):
        repository.get_active_text_vector_generation_id("video-1")


def test_retry_is_linear_idempotent_and_preserves_exact_work_identity(tmp_path) -> None:  # type: ignore[no-untyped-def]
    repository = _repository(tmp_path)
    _video, root = _create_ingest_job(repository)
    repository.claim_next_video_index_job(execution_token=TOKEN_A, stage="probe")
    failed = repository.fail_video_index_job(
        root.job_id,
        execution_token=TOKEN_A,
        error_code="probe_failed",
    )

    retry = repository.retry_video_index_job(
        failed.job_id,
        retry_job_id="job-2",
    )
    repeated = repository.retry_video_index_job(
        failed.job_id,
        retry_job_id="ignored-retry-id",
    )

    assert retry == repeated
    assert retry.state is JobState.QUEUED
    assert retry.attempt == 2
    assert retry.retry_of_job_id == failed.job_id
    assert retry.source_sha256 == failed.source_sha256
    assert retry.plan_hash == failed.plan_hash
    assert retry.idempotency_hash == failed.idempotency_hash


def test_startup_recovery_terminalizes_abandoned_attempt_and_creates_child(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    repository = _repository(tmp_path)
    _video, root = _create_ingest_job(repository)
    repository.claim_next_video_index_job(execution_token=TOKEN_A, stage="probe")
    repository.checkpoint_video_index_job(
        root.job_id,
        execution_token=TOKEN_A,
        progress=0.2,
        stage="scenes",
    )

    report = repository.recover_abandoned_video_index_jobs(limit=10)

    assert report.examined_count == 1
    assert report.cancelled_job_ids == ()
    assert len(report.retry_job_ids) == 1
    parent = repository.get_video_index_job(root.job_id)
    child = repository.get_video_index_job(report.retry_job_ids[0])
    assert parent is not None and child is not None
    assert (parent.state, parent.error_code) == (
        JobState.FAILED,
        "job_runtime_abandoned",
    )
    assert child.state is JobState.QUEUED
    assert child.retry_of_job_id == parent.job_id
    assert child.attempt == 2
    with pytest.raises(JobTransitionError):
        repository.checkpoint_video_index_job(
            parent.job_id,
            execution_token=TOKEN_A,
            progress=0.3,
            stage="speech",
        )


def test_startup_recovery_acknowledges_requested_cancel_without_retry(tmp_path) -> None:  # type: ignore[no-untyped-def]
    repository = _repository(tmp_path)
    _video, root = _create_ingest_job(repository)
    repository.claim_next_video_index_job(execution_token=TOKEN_A, stage="probe")
    repository.request_video_index_job_cancellation(root.job_id)

    report = repository.recover_abandoned_video_index_jobs(limit=10)

    assert report.examined_count == 1
    assert report.retry_job_ids == ()
    assert report.cancelled_job_ids == (root.job_id,)
    cancelled = repository.get_video_index_job(root.job_id)
    assert cancelled is not None and cancelled.state is JobState.CANCELLED


def test_startup_recovery_quarantines_corrupt_row_and_continues(tmp_path) -> None:  # type: ignore[no-untyped-def]
    repository = _repository(tmp_path)
    _create_ingest_job(repository, video_id="video-corrupt", job_id="job-corrupt")
    _create_ingest_job(repository, video_id="video-valid", job_id="job-valid")
    repository.claim_next_video_index_job(execution_token=TOKEN_A, stage="probe")
    repository.claim_next_video_index_job(execution_token=TOKEN_B, stage="probe")
    with repository._connect() as connection:
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute("DROP TRIGGER video_index_jobs_identity_immutable_update")
        connection.execute(
            "UPDATE video_index_jobs SET plan_json = ? WHERE job_id = ?",
            ("private-corrupt-plan", "job-corrupt"),
        )

    report = repository.recover_abandoned_video_index_jobs(limit=10)

    assert report.examined_count == 2
    assert report.quarantined_count == 1
    assert len(report.retry_job_ids) == 1
    child = repository.get_video_index_job(report.retry_job_ids[0])
    assert child is not None and child.retry_of_job_id == "job-valid"
    assert repository.count_video_index_job_quarantines() == 1


def test_legacy_adoption_is_explicit_bounded_and_requires_asset_identity(tmp_path) -> None:  # type: ignore[no-untyped-def]
    repository = _repository(tmp_path)
    repository.create_video_with_asset(
        video_id="legacy-video",
        original_name="legacy.mp4",
        stored_name="legacy.mp4",
        media_path="/tmp/legacy.mp4",
        size_bytes=1024,
        source_sha256="c" * 64,
    )
    repository.update_video(
        "legacy-video",
        status="processing",
        progress=0.4,
        stage="speech",
    )

    candidates = repository.list_legacy_video_index_candidates(limit=10)
    adopted = repository.adopt_legacy_video_index_job(
        "legacy-video",
        plan=_plan(executor_identity="videoscope.indexer.legacy-v1"),
        job_id="legacy-job",
    )
    repeated = repository.adopt_legacy_video_index_job(
        "legacy-video",
        plan=_plan(executor_identity="videoscope.indexer.legacy-v1"),
        job_id="ignored-legacy-job",
    )

    assert [video.id for video in candidates] == ["legacy-video"]
    assert adopted == repeated
    assert adopted.intent is VideoIndexIntent.INGEST
    assert adopted.prior_video_state is None
    assert repository.list_legacy_video_index_candidates(limit=10) == ()


def test_stage_run_job_link_requires_matching_active_plan_and_is_immutable(tmp_path) -> None:  # type: ignore[no-untyped-def]
    repository = _repository(tmp_path)
    plan = _plan()
    _video, root = _create_ingest_job(repository, plan=plan)
    repository.claim_next_video_index_job(execution_token=TOKEN_A, stage="scenes")

    run = repository.create_stage_run(
        video_id="video-1",
        specification=plan.specifications.scenes,
        run_id="scene-run",
        job_id=root.job_id,
        execution_token=TOKEN_A,
    )

    with repository._connect() as connection:
        persisted_job_id = connection.execute(
            "SELECT job_id FROM stage_runs WHERE run_id = ?",
            (run.run_id,),
        ).fetchone()[0]
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE stage_runs SET job_id = NULL WHERE run_id = ?",
                (run.run_id,),
            )
    assert persisted_job_id == root.job_id
    with pytest.raises(JobFenceError):
        repository.create_stage_run(
            video_id="video-1",
            specification=plan.specifications.speech,
            run_id="speech-run",
            job_id=root.job_id,
            execution_token=TOKEN_B,
        )

    with pytest.raises(JobFenceError):
        repository.transition_stage_run(
            run.run_id,
            StageState.RUNNING,
            execution_token=TOKEN_B,
        )
    repository.transition_stage_run(
        run.run_id,
        StageState.RUNNING,
        execution_token=TOKEN_A,
    )
    with pytest.raises(JobFenceError):
        repository.commit_segment_generation(
            run.run_id,
            segments=(),
            execution_token=TOKEN_B,
        )

    with pytest.raises(JobFenceError, match="must own"):
        repository.create_stage_run(
            video_id="video-1",
            specification=plan.specifications.speech,
            run_id="unowned-speech-run",
        )


def test_job_cancellation_terminalizes_linked_active_stage_runs(tmp_path) -> None:  # type: ignore[no-untyped-def]
    repository = _repository(tmp_path)
    plan = _plan()
    _video, root = _create_ingest_job(repository, plan=plan)
    repository.claim_next_video_index_job(execution_token=TOKEN_A, stage="scenes")
    run = repository.create_stage_run(
        video_id="video-1",
        specification=plan.specifications.scenes,
        run_id="scene-run",
        job_id=root.job_id,
        execution_token=TOKEN_A,
    )
    repository.transition_stage_run(
        run.run_id,
        StageState.RUNNING,
        execution_token=TOKEN_A,
    )

    repository.request_video_index_job_cancellation(root.job_id)
    repository.cancel_video_index_job(root.job_id, execution_token=TOKEN_A)

    persisted = repository.get_stage_run(run.run_id)
    assert persisted is not None
    assert persisted.state is StageState.CANCELLED


def test_job_failure_terminalizes_linked_active_stage_runs(tmp_path) -> None:  # type: ignore[no-untyped-def]
    repository = _repository(tmp_path)
    plan = _plan()
    _video, root = _create_ingest_job(repository, plan=plan)
    repository.claim_next_video_index_job(execution_token=TOKEN_A, stage="scenes")
    run = repository.create_stage_run(
        video_id="video-1",
        specification=plan.specifications.scenes,
        run_id="scene-run",
        job_id=root.job_id,
        execution_token=TOKEN_A,
    )
    repository.transition_stage_run(
        run.run_id,
        StageState.RUNNING,
        execution_token=TOKEN_A,
    )

    repository.fail_video_index_job(
        root.job_id,
        execution_token=TOKEN_A,
        error_code="scene_failed",
    )

    persisted = repository.get_stage_run(run.run_id)
    assert persisted is not None
    assert persisted.state is StageState.FAILED
    assert persisted.error_code == "scene_failed"


def test_job_checkpoint_preserves_linked_running_stage(tmp_path) -> None:  # type: ignore[no-untyped-def]
    repository = _repository(tmp_path)
    plan = _plan()
    _video, root = _create_ingest_job(repository, plan=plan)
    repository.claim_next_video_index_job(execution_token=TOKEN_A, stage="scenes")
    run = repository.create_stage_run(
        video_id="video-1",
        specification=plan.specifications.scenes,
        run_id="scene-run",
        job_id=root.job_id,
        execution_token=TOKEN_A,
    )
    repository.transition_stage_run(
        run.run_id,
        StageState.RUNNING,
        execution_token=TOKEN_A,
    )

    repository.checkpoint_video_index_job(
        root.job_id,
        execution_token=TOKEN_A,
        progress=0.2,
        stage="scenes",
    )

    persisted = repository.get_stage_run(run.run_id)
    assert persisted is not None
    assert persisted.state is StageState.RUNNING


def test_job_completion_requires_every_exact_planned_stage_receipt(tmp_path) -> None:  # type: ignore[no-untyped-def]
    repository = _repository(tmp_path)
    plan = _plan()
    _video, root = _create_ingest_job(repository, plan=plan)
    repository.claim_next_video_index_job(execution_token=TOKEN_A, stage="finalizing")

    with pytest.raises(JobTransitionError, match="exact planned stage receipts"):
        repository.complete_video_index_job(
            root.job_id,
            execution_token=TOKEN_A,
        )

    persisted = repository.get_video_index_job(root.job_id)
    assert persisted is not None and persisted.state is JobState.RUNNING


def test_job_completion_rejects_duplicate_receipt_substituted_for_missing_stage(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    repository = _repository(tmp_path)
    plan = _plan()
    _video, root = _create_ingest_job(repository, plan=plan)
    repository.claim_next_video_index_job(execution_token=TOKEN_A, stage="finalizing")
    planned = _planned_stage_specifications(plan)
    _record_terminal_job_receipts(
        repository,
        job_id=root.job_id,
        plan=plan,
        execution_token=TOKEN_A,
        specifications=(*planned[:-1], planned[0]),
    )

    with pytest.raises(JobTransitionError, match="exact planned stage receipts"):
        repository.complete_video_index_job(
            root.job_id,
            execution_token=TOKEN_A,
        )


def test_completion_rejects_seven_distinct_receipts_with_an_unplanned_specification(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    repository = _repository(tmp_path)
    plan = _plan()
    _video, root = _create_ingest_job(repository, plan=plan)
    repository.claim_next_video_index_job(execution_token=TOKEN_A, stage="finalizing")
    planned = _planned_stage_specifications(plan)
    _record_terminal_job_receipts(
        repository,
        job_id=root.job_id,
        plan=plan,
        execution_token=TOKEN_A,
        specifications=planned[:-1],
    )
    wrong = _stage(StageKind.LIGHTHOUSE, revision="unplanned")
    with repository._connect() as connection:
        timestamp = connection.execute(
            "SELECT updated_at FROM video_index_jobs WHERE job_id = ?",
            (root.job_id,),
        ).fetchone()[0]
        connection.execute(
            """
            INSERT INTO stage_specifications (
                specification_hash, stage_kind, canonical_json, created_at
            ) VALUES (?, ?, ?, ?)
            """,
            (
                wrong.specification_hash,
                wrong.kind.value,
                wrong.canonical_json,
                timestamp,
            ),
        )
        connection.execute(
            """
            INSERT INTO stage_runs (
                run_id, video_id, specification_hash, source_sha256,
                attempt, state, output_generation, error_code,
                retry_of_run_id, created_at, started_at, finished_at,
                updated_at, job_id
            ) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?, NULL, ?, ?, ?)
            """,
            (
                "receipt-unplanned-lighthouse",
                "video-1",
                wrong.specification_hash,
                "a" * 64,
                1,
                StageState.NOT_CONFIGURED.value,
                timestamp,
                timestamp,
                timestamp,
                root.job_id,
            ),
        )

    with pytest.raises(JobTransitionError, match="exact planned stage receipts"):
        repository.complete_video_index_job(
            root.job_id,
            execution_token=TOKEN_A,
        )
    with repository._connect() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="exact planned stage receipts"):
            connection.execute(
                """
                UPDATE video_index_jobs
                SET state = 'complete', progress = 1.0, stage = 'complete',
                    execution_token = NULL, finished_at = updated_at
                WHERE job_id = ?
                """,
                (root.job_id,),
            )


def test_raw_sql_job_completion_requires_exact_planned_stage_receipts(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    repository = _repository(tmp_path)
    plan = _plan()
    _video, root = _create_ingest_job(repository, plan=plan)
    repository.claim_next_video_index_job(execution_token=TOKEN_A, stage="finalizing")
    planned = _planned_stage_specifications(plan)
    _record_terminal_job_receipts(
        repository,
        job_id=root.job_id,
        plan=plan,
        execution_token=TOKEN_A,
        specifications=(*planned[:-1], planned[0]),
    )

    with repository._connect() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="exact planned stage receipts"):
            connection.execute(
                """
                UPDATE video_index_jobs
                SET state = 'complete', progress = 1.0, stage = 'complete',
                    execution_token = NULL, finished_at = updated_at
                WHERE job_id = ?
                """,
                (root.job_id,),
            )


def test_job_completion_accepts_all_exact_terminal_non_output_receipts(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    repository = _repository(tmp_path)
    plan = _plan()
    _video, root = _create_ingest_job(repository, plan=plan)
    repository.claim_next_video_index_job(execution_token=TOKEN_A, stage="finalizing")
    _record_terminal_job_receipts(
        repository,
        job_id=root.job_id,
        plan=plan,
        execution_token=TOKEN_A,
    )

    completed = repository.complete_video_index_job(
        root.job_id,
        execution_token=TOKEN_A,
    )

    assert completed.state is JobState.COMPLETE


def test_job_completion_rejects_linked_active_stage_run(tmp_path) -> None:  # type: ignore[no-untyped-def]
    repository = _repository(tmp_path)
    plan = _plan()
    _video, root = _create_ingest_job(repository, plan=plan)
    repository.claim_next_video_index_job(execution_token=TOKEN_A, stage="scenes")
    run = repository.create_stage_run(
        video_id="video-1",
        specification=plan.specifications.scenes,
        run_id="scene-run",
        job_id=root.job_id,
        execution_token=TOKEN_A,
    )
    repository.transition_stage_run(
        run.run_id,
        StageState.RUNNING,
        execution_token=TOKEN_A,
    )

    with pytest.raises(JobTransitionError, match="active stage runs"):
        repository.complete_video_index_job(
            root.job_id,
            execution_token=TOKEN_A,
        )

    persisted_job = repository.get_video_index_job(root.job_id)
    persisted_run = repository.get_stage_run(run.run_id)
    assert persisted_job is not None and persisted_job.state is JobState.RUNNING
    assert persisted_run is not None and persisted_run.state is StageState.RUNNING

    with repository._connect() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="active stage runs"):
            connection.execute(
                """
                UPDATE video_index_jobs
                SET state = 'complete', progress = 1.0, stage = 'complete',
                    execution_token = NULL, finished_at = updated_at
                WHERE job_id = ?
                """,
                (root.job_id,),
            )


def test_text_vector_build_mutations_require_job_token_and_cancel_cleans_build(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    repository = _repository(tmp_path)
    plan = _plan()
    _video, root = _create_ingest_job(repository, plan=plan)
    repository.claim_next_video_index_job(execution_token=TOKEN_A, stage="text_vectors")
    run = repository.create_stage_run(
        video_id="video-1",
        specification=plan.specifications.text_vectors,
        run_id="text-run",
        job_id=root.job_id,
        execution_token=TOKEN_A,
    )
    repository.transition_stage_run(
        run.run_id,
        StageState.RUNNING,
        execution_token=TOKEN_A,
    )
    index_specification = TextVectorIndexSpecification(
        embedding_identity="tests/hash-embedding@v1",
        dimensions=8,
    )

    with pytest.raises(JobFenceError):
        repository.reserve_text_vector_generation(
            run.run_id,
            index_specification=index_specification,
            semantic_specifications=plan.specifications.semantic_segment_specifications,
            generation_id="text-generation-wrong",
            execution_token=TOKEN_B,
        )
    build = repository.reserve_text_vector_generation(
        run.run_id,
        index_specification=index_specification,
        semantic_specifications=plan.specifications.semantic_segment_specifications,
        generation_id="text-generation",
        execution_token=TOKEN_A,
    )
    receipt = TextVectorBuildReceipt(
        generation_id=build.generation_id,
        index_specification_hash=build.index_specification.specification_hash,
        point_count=len(build.points),
        point_manifest_sha256=build.point_manifest_sha256,
        vector_manifest_sha256="f" * 64,
    )
    with pytest.raises(JobFenceError):
        repository.heartbeat_text_vector_build(
            build.generation_id,
            execution_token=TOKEN_B,
        )
    with pytest.raises(JobFenceError):
        repository.commit_text_vector_generation(
            run.run_id,
            receipt=receipt,
            execution_token=TOKEN_B,
        )
    with pytest.raises(JobFenceError):
        repository.fail_text_vector_build(
            run.run_id,
            error_code="text_vector_failed",
            execution_token=TOKEN_B,
        )

    repository.request_video_index_job_cancellation(root.job_id)
    repository.cancel_video_index_job(root.job_id, execution_token=TOKEN_A)

    with repository._connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM text_vector_builds").fetchone()[0] == 0
        tombstone = connection.execute(
            """
            SELECT lifecycle_state FROM text_vector_generation_tombstones
            WHERE generation_id = ?
            """,
            (build.generation_id,),
        ).fetchone()
        assert tombstone[0] == "gc_pending"
        assert connection.execute(
            "SELECT COUNT(*) FROM artifact_gc_jobs WHERE generation_id = ?",
            (build.generation_id,),
        ).fetchone()[0] == 1


def test_recovery_cleans_linked_text_vector_build_before_retry(tmp_path) -> None:  # type: ignore[no-untyped-def]
    repository = _repository(tmp_path)
    plan = _plan()
    _video, root = _create_ingest_job(repository, plan=plan)
    repository.claim_next_video_index_job(execution_token=TOKEN_A, stage="text_vectors")
    run = repository.create_stage_run(
        video_id="video-1",
        specification=plan.specifications.text_vectors,
        run_id="text-run-parent",
        job_id=root.job_id,
        execution_token=TOKEN_A,
    )
    repository.transition_stage_run(
        run.run_id,
        StageState.RUNNING,
        execution_token=TOKEN_A,
    )
    index_specification = TextVectorIndexSpecification(
        embedding_identity="tests/hash-embedding@v1",
        dimensions=8,
    )
    first_build = repository.reserve_text_vector_generation(
        run.run_id,
        index_specification=index_specification,
        semantic_specifications=plan.specifications.semantic_segment_specifications,
        generation_id="text-generation-parent",
        execution_token=TOKEN_A,
    )

    report = repository.recover_abandoned_video_index_jobs(limit=10)
    child_id = report.retry_job_ids[0]
    child = repository.claim_next_video_index_job(
        execution_token=TOKEN_B,
        stage="text_vectors",
    )
    assert child is not None and child.job_id == child_id
    retry_run = repository.create_stage_run(
        video_id="video-1",
        specification=plan.specifications.text_vectors,
        run_id="text-run-child",
        job_id=child.job_id,
        execution_token=TOKEN_B,
    )
    repository.transition_stage_run(
        retry_run.run_id,
        StageState.RUNNING,
        execution_token=TOKEN_B,
    )
    retry_build = repository.reserve_text_vector_generation(
        retry_run.run_id,
        index_specification=index_specification,
        semantic_specifications=plan.specifications.semantic_segment_specifications,
        generation_id="text-generation-child",
        execution_token=TOKEN_B,
    )

    assert retry_build.generation_id == "text-generation-child"
    with repository._connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM text_vector_builds WHERE generation_id = ?",
            (first_build.generation_id,),
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM artifact_gc_jobs WHERE generation_id = ?",
            (first_build.generation_id,),
        ).fetchone()[0] == 1
