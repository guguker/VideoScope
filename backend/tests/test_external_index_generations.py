from __future__ import annotations

import hashlib
import json
import sqlite3

import pytest

from videoscope.artifacts import IndexingSpecifications, StageKind, StageSpecification, StageState
from videoscope.jobs import JobFenceError, JobState, VideoIndexPlanSnapshot
from videoscope.providers.whisper import WhisperPromptSnapshot
from videoscope.repository import Repository


TOKEN = "worker-token-aaaaaaaaaaaaaaaa"
WRONG_TOKEN = "worker-token-bbbbbbbbbbbbbbbb"
SOURCE_SHA256 = "a" * 64


def _stage(
    kind: StageKind,
    revision: str = "v1",
    *,
    parameters: dict[str, object] | None = None,
) -> StageSpecification:
    return StageSpecification(
        kind=kind,
        schema_version=1,
        implementation_revision=f"test.{kind.value}.{revision}",
        parameters=parameters or {},
        dependencies={"videoscope": "test"},
    )


def _plan(revision: str = "v1") -> VideoIndexPlanSnapshot:
    prompt_text = "Мозгов"
    prompt = WhisperPromptSnapshot(
        effective_prompt=prompt_text,
        effective_prompt_sha256=hashlib.sha256(prompt_text.encode()).hexdigest(),
        glossary_state="ready",
    )
    return VideoIndexPlanSnapshot(
        schema_version=2,
        specifications=IndexingSpecifications(
            scenes=_stage(StageKind.SCENES, revision),
            speech=_stage(
                StageKind.SPEECH,
                revision,
                parameters={
                    "effective_prompt_sha256": prompt.effective_prompt_sha256,
                    "glossary_state": prompt.glossary_state,
                },
            ),
            ocr=_stage(StageKind.OCR, revision),
            objects=_stage(StageKind.OBJECTS, revision),
            text_vectors=_stage(StageKind.TEXT_VECTORS, revision),
        ),
        visual_dense_specification=_stage(
            StageKind.VISUAL_DENSE,
            revision,
            parameters={"visual_specification_hash": "d" * 64},
        ),
        lighthouse_specification=_stage(
            StageKind.LIGHTHOUSE,
            revision,
            parameters={"specification_hash": "e" * 64},
        ),
        whisper_prompt_snapshot=prompt,
        executor_identity="sha256:" + "e" * 64,
    )


def _repository(tmp_path) -> Repository:  # type: ignore[no-untyped-def]
    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()
    return repository


def _create_ready_legacy_video(repository: Repository) -> None:
    repository.create_video_with_asset(
        video_id="video-1",
        original_name="video.mp4",
        stored_name="video.mp4",
        media_path="/tmp/video.mp4",
        size_bytes=1024,
        source_sha256=SOURCE_SHA256,
    )
    repository.update_video(
        "video-1",
        status="ready",
        progress=1.0,
        stage="ready",
        duration=4.0,
    )


def _descriptor(
    kind: StageKind,
    specification: StageSpecification,
    generation_id: str,
) -> dict[str, object]:
    descriptor: dict[str, object] = {
        "duration_seconds": 4.0,
        "generation_id": generation_id,
        "source_sha256": SOURCE_SHA256,
        "source_size_bytes": 1024,
        "specification_hash": (
            specification.parameters["visual_specification_hash"]
            if kind is StageKind.VISUAL_DENSE
            else specification.parameters["specification_hash"]
        ),
        "video_id": "video-1",
    }
    descriptor[
        "content_sha256" if kind is StageKind.VISUAL_DENSE else "manifest_sha256"
    ] = "c" * 64
    return descriptor


def _legacy_generation(
    repository: Repository,
    specification: StageSpecification,
    *,
    run_id: str,
    generation_id: str,
):  # type: ignore[no-untyped-def]
    repository.create_stage_run(
        video_id="video-1",
        specification=specification,
        run_id=run_id,
    )
    repository.transition_stage_run(run_id, StageState.RUNNING)
    return repository.commit_external_index_generation(
        run_id,
        descriptor=_descriptor(specification.kind, specification, generation_id),
    )


def _owned_generation(
    repository: Repository,
    *,
    job_id: str,
    specification: StageSpecification,
    run_id: str,
    generation_id: str,
):  # type: ignore[no-untyped-def]
    repository.create_stage_run(
        video_id="video-1",
        specification=specification,
        run_id=run_id,
        job_id=job_id,
        execution_token=TOKEN,
    )
    repository.transition_stage_run(
        run_id,
        StageState.RUNNING,
        execution_token=TOKEN,
    )
    return repository.commit_external_index_generation(
        run_id,
        descriptor=_descriptor(specification.kind, specification, generation_id),
        execution_token=TOKEN,
    )


def _start_reindex(repository: Repository, plan: VideoIndexPlanSnapshot, job_id: str) -> None:
    repository.enqueue_video_reindex_job("video-1", plan=plan, job_id=job_id)
    claimed = repository.claim_next_video_index_job(execution_token=TOKEN)
    assert claimed is not None and claimed.job_id == job_id


def _record_non_external_receipts(
    repository: Repository,
    *,
    plan: VideoIndexPlanSnapshot,
    job_id: str,
) -> None:
    specifications = (
        *plan.specifications.segment_specifications,
        plan.specifications.text_vectors,
    )
    for specification in specifications:
        run = repository.create_stage_run(
            video_id="video-1",
            specification=specification,
            run_id=f"{job_id}-{specification.kind.value}",
            job_id=job_id,
            execution_token=TOKEN,
        )
        repository.transition_stage_run(
            run.run_id,
            StageState.NOT_CONFIGURED,
            execution_token=TOKEN,
        )


def test_v9_external_generation_schema_is_idempotent_and_guarded(tmp_path) -> None:  # type: ignore[no-untyped-def]
    repository = _repository(tmp_path)
    repository.initialize()

    with repository._connect() as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        triggers = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            )
        }

    assert {"external_index_generations", "active_external_index_generations"} <= tables
    assert {
        "external_index_generations_immutable_update",
        "external_index_generations_immutable_delete",
        "external_index_generation_run_identity_insert",
        "active_external_index_generations_job_owner_insert",
        "active_external_index_generations_job_owner_update",
    } <= triggers


@pytest.mark.parametrize("kind", [StageKind.VISUAL_DENSE, StageKind.LIGHTHOUSE])
def test_cancelled_job_never_activates_staged_external_generation(
    tmp_path,
    kind: StageKind,
) -> None:  # type: ignore[no-untyped-def]
    repository = _repository(tmp_path)
    plan = _plan()
    _create_ready_legacy_video(repository)
    specification = plan.for_kind(kind)
    old = _legacy_generation(
        repository,
        specification,
        run_id=f"legacy-{kind.value}",
        generation_id="1" * 32,
    )
    _start_reindex(repository, plan, f"job-{kind.value}")

    staged = _owned_generation(
        repository,
        job_id=f"job-{kind.value}",
        specification=specification,
        run_id=f"run-{kind.value}",
        generation_id="2" * 32,
    )

    assert staged.generation_id == "2" * 32
    assert repository.get_active_external_index_generation("video-1", kind) == old
    repository.request_video_index_job_cancellation(f"job-{kind.value}")
    repository.cancel_video_index_job(
        f"job-{kind.value}",
        execution_token=TOKEN,
    )
    assert repository.get_active_external_index_generation("video-1", kind) == old


def test_job_completion_atomically_releases_both_external_generations(tmp_path) -> None:  # type: ignore[no-untyped-def]
    repository = _repository(tmp_path)
    plan = _plan()
    _create_ready_legacy_video(repository)
    old_visual = _legacy_generation(
        repository,
        plan.visual_dense_specification,
        run_id="legacy-visual",
        generation_id="1" * 32,
    )
    old_lighthouse = _legacy_generation(
        repository,
        plan.lighthouse_specification,
        run_id="legacy-lighthouse",
        generation_id="3" * 32,
    )
    _start_reindex(repository, plan, "job-release")
    visual = _owned_generation(
        repository,
        job_id="job-release",
        specification=plan.visual_dense_specification,
        run_id="run-visual",
        generation_id="2" * 32,
    )
    lighthouse = _owned_generation(
        repository,
        job_id="job-release",
        specification=plan.lighthouse_specification,
        run_id="run-lighthouse",
        generation_id="4" * 32,
    )
    _record_non_external_receipts(
        repository,
        plan=plan,
        job_id="job-release",
    )

    assert repository.get_active_external_index_generation(
        "video-1", StageKind.VISUAL_DENSE
    ) == old_visual
    assert repository.get_active_external_index_generation(
        "video-1", StageKind.LIGHTHOUSE
    ) == old_lighthouse

    completed = repository.complete_video_index_job(
        "job-release",
        execution_token=TOKEN,
    )

    assert completed.state is JobState.COMPLETE
    assert repository.get_active_external_index_generation(
        "video-1", StageKind.VISUAL_DENSE
    ) == visual
    assert repository.get_active_external_index_generation(
        "video-1", StageKind.LIGHTHOUSE
    ) == lighthouse


def test_external_release_rolls_back_job_and_both_bindings_on_corrupt_receipt(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    repository = _repository(tmp_path)
    plan = _plan()
    _create_ready_legacy_video(repository)
    old_visual = _legacy_generation(
        repository,
        plan.visual_dense_specification,
        run_id="legacy-visual",
        generation_id="1" * 32,
    )
    old_lighthouse = _legacy_generation(
        repository,
        plan.lighthouse_specification,
        run_id="legacy-lighthouse",
        generation_id="3" * 32,
    )
    _start_reindex(repository, plan, "job-rollback")
    _owned_generation(
        repository,
        job_id="job-rollback",
        specification=plan.visual_dense_specification,
        run_id="run-visual",
        generation_id="2" * 32,
    )
    _owned_generation(
        repository,
        job_id="job-rollback",
        specification=plan.lighthouse_specification,
        run_id="run-lighthouse",
        generation_id="4" * 32,
    )
    _record_non_external_receipts(
        repository,
        plan=plan,
        job_id="job-rollback",
    )
    with repository._connect() as connection:
        connection.execute("DROP TRIGGER external_index_generations_immutable_update")
        connection.execute(
            """
            UPDATE external_index_generations
            SET descriptor_sha256 = ?
            WHERE stage_kind = 'lighthouse' AND generation_id = ?
            """,
            ("f" * 64, "4" * 32),
        )

    with pytest.raises(ValueError, match="corrupt"):
        repository.complete_video_index_job(
            "job-rollback",
            execution_token=TOKEN,
        )

    job = repository.get_video_index_job("job-rollback")
    assert job is not None and job.state is JobState.RUNNING
    assert repository.get_active_external_index_generation(
        "video-1", StageKind.VISUAL_DENSE
    ) == old_visual
    assert repository.get_active_external_index_generation(
        "video-1", StageKind.LIGHTHOUSE
    ) == old_lighthouse


def test_external_commit_is_fenced_bounded_pathless_and_raw_sql_guarded(tmp_path) -> None:  # type: ignore[no-untyped-def]
    repository = _repository(tmp_path)
    plan = _plan()
    _create_ready_legacy_video(repository)
    _start_reindex(repository, plan, "job-fence")
    specification = plan.visual_dense_specification
    repository.create_stage_run(
        video_id="video-1",
        specification=specification,
        run_id="run-fence",
        job_id="job-fence",
        execution_token=TOKEN,
    )
    repository.transition_stage_run(
        "run-fence",
        StageState.RUNNING,
        execution_token=TOKEN,
    )
    descriptor = _descriptor(StageKind.VISUAL_DENSE, specification, "5" * 32)

    raw_mismatch = {**descriptor, "duration_seconds": 5.0}
    raw_canonical = json.dumps(
        raw_mismatch,
        sort_keys=True,
        separators=(",", ":"),
    )
    with repository._connect() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="run identity mismatch"):
            connection.execute(
                """
                INSERT INTO external_index_generations (
                    video_id, stage_kind, generation_id, specification_hash,
                    artifact_specification_hash, source_sha256, run_id,
                    descriptor_json, descriptor_sha256, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "video-1",
                    StageKind.VISUAL_DENSE.value,
                    descriptor["generation_id"],
                    specification.specification_hash,
                    specification.parameters["visual_specification_hash"],
                    SOURCE_SHA256,
                    "run-fence",
                    raw_canonical,
                    hashlib.sha256(raw_canonical.encode("utf-8")).hexdigest(),
                    "2026-08-21T12:00:00+00:00",
                ),
            )

    with pytest.raises(JobFenceError):
        repository.commit_external_index_generation(
            "run-fence",
            descriptor=descriptor,
            execution_token=WRONG_TOKEN,
        )
    with pytest.raises(ValueError, match="canonical descriptor"):
        repository.commit_external_index_generation(
            "run-fence",
            descriptor={**descriptor, "path": "/private/cache/model"},
            execution_token=TOKEN,
        )
    with pytest.raises(ValueError, match="canonical descriptor"):
        repository.commit_external_index_generation(
            "run-fence",
            descriptor={**descriptor, "content_sha256": "x" * 20_000},
            execution_token=TOKEN,
        )
    with pytest.raises(ValueError, match="canonical descriptor"):
        repository.commit_external_index_generation(
            "run-fence",
            descriptor={**descriptor, "duration_seconds": 5.0},
            execution_token=TOKEN,
        )

    generation = repository.commit_external_index_generation(
        "run-fence",
        descriptor=descriptor,
        execution_token=TOKEN,
    )
    with repository._connect() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="not complete"):
            connection.execute(
                """
                INSERT INTO active_external_index_generations (
                    video_id, stage_kind, generation_id, activated_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    "video-1",
                    StageKind.VISUAL_DENSE.value,
                    generation.generation_id,
                    generation.completed_at,
                ),
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE external_index_generations SET descriptor_json = ?",
                (json.dumps({"corrupt": True}),),
            )

    repository.request_video_index_job_cancellation("job-fence")
    with pytest.raises(JobFenceError):
        repository.commit_external_index_generation(
            "run-fence",
            descriptor=descriptor,
            execution_token=TOKEN,
        )
