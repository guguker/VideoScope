from __future__ import annotations

from datetime import UTC, datetime, timedelta
import sqlite3

import pytest

import videoscope.repository as repository_module
from videoscope.artifacts import (
    StageKind,
    StageSpecification,
    StageState,
    TextVectorBuildReceipt,
    TextVectorIndexSpecification,
)
from videoscope.repository import LATEST_SCHEMA_VERSION, Repository, SegmentRecord


def _stage_specification(kind: StageKind, *, revision: str = "v1") -> StageSpecification:
    return StageSpecification(
        kind=kind,
        schema_version=1,
        implementation_revision=f"tests.{kind.value}.{revision}",
        parameters={"contract": f"{kind.value}-segments-v1"},
        model_identity=f"tests/{kind.value}@{revision}",
    )


def _semantic_specifications(*, revision: str = "v1") -> tuple[StageSpecification, ...]:
    return tuple(
        _stage_specification(kind, revision=revision)
        for kind in (StageKind.SPEECH, StageKind.OCR, StageKind.OBJECTS)
    )


def _text_specification(
    semantic: tuple[StageSpecification, ...],
    *,
    revision: str = "v1",
) -> StageSpecification:
    by_kind = {item.kind: item for item in semantic}
    return StageSpecification(
        kind=StageKind.TEXT_VECTORS,
        schema_version=1,
        implementation_revision=f"tests.text-vectors.{revision}",
        parameters={"semantic_modalities": ["speech", "ocr", "objects"]},
        model_identity=f"tests/embedding@{revision}",
        dependencies={
            "speech_specification": by_kind[StageKind.SPEECH].specification_hash,
            "ocr_specification": by_kind[StageKind.OCR].specification_hash,
            "objects_specification": by_kind[StageKind.OBJECTS].specification_hash,
        },
    )


def _index_specification(*, dimensions: int = 8) -> TextVectorIndexSpecification:
    return TextVectorIndexSpecification(
        embedding_identity="tests/hash-embedding@v1",
        dimensions=dimensions,
        distance="cosine",
        payload_schema_version=1,
        point_id_algorithm="uuid5-generation-segment-v1",
    )


def _repository_with_video(tmp_path) -> Repository:  # type: ignore[no-untyped-def]
    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()
    repository.create_video_with_asset(
        video_id="video-1",
        original_name="match.mp4",
        stored_name="video-1.mp4",
        media_path=str(tmp_path / "video-1.mp4"),
        size_bytes=1024,
        source_sha256="a" * 64,
    )
    return repository


def _segment(segment_id: str, modality: str, text: str) -> SegmentRecord:
    return SegmentRecord(
        id=segment_id,
        video_id="video-1",
        start=1.0,
        end=2.0,
        modality=modality,
        text=text,
        confidence=0.9,
        metadata={},
        thumbnail_path=None,
    )


def _publish_segment_generation(
    repository: Repository,
    specification: StageSpecification,
    *,
    run_id: str,
    generation_id: str,
    segments: list[SegmentRecord],
) -> None:
    repository.create_stage_run(
        video_id="video-1",
        specification=specification,
        run_id=run_id,
    )
    repository.transition_stage_run(run_id, StageState.RUNNING)
    repository.commit_segment_generation(
        run_id,
        generation_id=generation_id,
        segments=segments,
    )


def _start_text_run(
    repository: Repository,
    specification: StageSpecification,
    *,
    run_id: str,
) -> None:
    repository.create_stage_run(
        video_id="video-1",
        specification=specification,
        run_id=run_id,
    )
    repository.transition_stage_run(run_id, StageState.RUNNING)


def _complete_receipt(plan) -> TextVectorBuildReceipt:  # type: ignore[no-untyped-def]
    return TextVectorBuildReceipt(
        generation_id=plan.generation_id,
        index_specification_hash=plan.index_specification.specification_hash,
        point_count=len(plan.points),
        point_manifest_sha256=plan.point_manifest_sha256,
        vector_manifest_sha256="f" * 64,
    )


def test_text_vector_index_specification_is_canonical_and_collection_safe() -> None:
    first = _index_specification()
    second = TextVectorIndexSpecification.from_canonical_json(first.canonical_json)

    assert second == first
    assert second.specification_hash == first.specification_hash
    assert second.collection_name.startswith("videoscope_text_v1_")
    assert "/" not in second.collection_name


@pytest.mark.parametrize(
    "changes",
    [
        {"embedding_identity": ""},
        {"dimensions": 0},
        {"dimensions": True},
        {"distance": "dot"},
        {"payload_schema_version": 0},
        {"point_id_algorithm": "../../unsafe"},
    ],
)
def test_text_vector_index_specification_rejects_ambiguous_identity(changes) -> None:  # type: ignore[no-untyped-def]
    values = {
        "embedding_identity": "tests/hash-embedding@v1",
        "dimensions": 8,
        "distance": "cosine",
        "payload_schema_version": 1,
        "point_id_algorithm": "uuid5-generation-segment-v1",
    }
    values.update(changes)

    with pytest.raises(ValueError):
        TextVectorIndexSpecification(**values)


def test_schema_v8_migration_preserves_v5_rows_and_is_idempotent(tmp_path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_path = source_dir / "videoscope.sqlite3"
    with sqlite3.connect(source_path) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("BEGIN IMMEDIATE")
        for version, migration in repository_module._SCHEMA_MIGRATIONS[:5]:
            migration(connection)
            connection.execute(f"PRAGMA user_version = {version}")
        connection.commit()
    source = Repository(source_path)
    source.create_video_with_asset(
        video_id="video-1",
        original_name="match.mp4",
        stored_name="video-1.mp4",
        media_path=str(source_dir / "video-1.mp4"),
        size_bytes=1024,
        source_sha256="a" * 64,
    )
    semantic = _semantic_specifications()
    _publish_segment_generation(
        source,
        semantic[0],
        run_id="speech-run-1",
        generation_id="speech-generation-1",
        segments=[_segment("speech-1", "speech", "made basket")],
    )
    migrated_path = tmp_path / "migrated.sqlite3"
    with source._connect() as source_connection, sqlite3.connect(migrated_path) as target:
        source_connection.backup(target)
    migrated = Repository(migrated_path)
    migrated.initialize()
    migrated.initialize()

    assert migrated.schema_version() == LATEST_SCHEMA_VERSION == 9
    assert migrated.get_video("video-1") is not None
    assert [item.id for item in migrated.list_active_segments("video-1", StageKind.SPEECH)] == [
        "speech-1"
    ]
    assert migrated.get_active_text_vector_generation("video-1") is None
    with migrated._connect() as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_reservation_snapshots_exact_upstream_lineage_and_searchable_points(tmp_path) -> None:
    repository = _repository_with_video(tmp_path)
    semantic = _semantic_specifications()
    text = _text_specification(semantic)
    _publish_segment_generation(
        repository,
        semantic[0],
        run_id="speech-run-1",
        generation_id="speech-generation-1",
        segments=[
            _segment("speech-1", "speech", "made basket"),
            _segment("speech-short", "speech", "yes"),
        ],
    )
    _publish_segment_generation(
        repository,
        semantic[1],
        run_id="ocr-run-1",
        generation_id="ocr-generation-1",
        segments=[_segment("ocr-1", "ocr", "HOME 87")],
    )
    _start_text_run(repository, text, run_id="text-run-1")

    plan = repository.reserve_text_vector_generation(
        "text-run-1",
        index_specification=_index_specification(),
        semantic_specifications=semantic,
        generation_id="text-generation-1",
    )

    by_kind = {item.stage_kind: item for item in plan.inputs}
    assert by_kind[StageKind.SPEECH].segment_generation_id == "speech-generation-1"
    assert by_kind[StageKind.OCR].segment_generation_id == "ocr-generation-1"
    assert by_kind[StageKind.OBJECTS].segment_generation_id is None
    assert by_kind[StageKind.OBJECTS].segment_count == 0
    assert [(point.modality, point.segment_id) for point in plan.points] == [
        ("speech", "speech-1"),
        ("ocr", "ocr-1"),
    ]
    assert plan.expected_previous_generation_id is None


def test_atomic_commit_activates_complete_generation_and_retains_old_generation(tmp_path) -> None:
    repository = _repository_with_video(tmp_path)
    semantic = _semantic_specifications()
    text = _text_specification(semantic)
    _publish_segment_generation(
        repository,
        semantic[0],
        run_id="speech-run-1",
        generation_id="speech-generation-1",
        segments=[_segment("speech-1", "speech", "made basket")],
    )

    _start_text_run(repository, text, run_id="text-run-1")
    first_plan = repository.reserve_text_vector_generation(
        "text-run-1",
        index_specification=_index_specification(),
        semantic_specifications=semantic,
        generation_id="text-generation-1",
    )
    first = repository.commit_text_vector_generation(
        "text-run-1",
        receipt=_complete_receipt(first_plan),
    )

    _start_text_run(repository, text, run_id="text-run-2")
    second_plan = repository.reserve_text_vector_generation(
        "text-run-2",
        index_specification=_index_specification(),
        semantic_specifications=semantic,
        generation_id="text-generation-2",
    )
    assert second_plan.expected_previous_generation_id == first.generation_id
    second = repository.commit_text_vector_generation(
        "text-run-2",
        receipt=_complete_receipt(second_plan),
    )

    assert repository.get_active_text_vector_generation("video-1") == second
    assert repository.get_text_vector_generation(first.generation_id) == first
    assert repository.get_stage_run("text-run-2").state is StageState.COMPLETE  # type: ignore[union-attr]
    assert repository.get_stage_run("text-run-2").output_generation == second.generation_id  # type: ignore[union-attr]


def test_empty_generation_is_complete_capability_not_missing(tmp_path) -> None:
    repository = _repository_with_video(tmp_path)
    semantic = _semantic_specifications()
    text = _text_specification(semantic)
    for index, specification in enumerate(semantic):
        _publish_segment_generation(
            repository,
            specification,
            run_id=f"{specification.kind.value}-run-empty",
            generation_id=f"{specification.kind.value}-generation-empty",
            segments=[],
        )
    _start_text_run(repository, text, run_id="text-run-empty")
    plan = repository.reserve_text_vector_generation(
        "text-run-empty",
        index_specification=_index_specification(),
        semantic_specifications=semantic,
        generation_id="text-generation-empty",
    )

    generation = repository.commit_text_vector_generation(
        "text-run-empty",
        receipt=_complete_receipt(plan),
    )

    assert generation.point_count == 0
    bindings = repository.list_current_text_vector_bindings(
        video_ids=["video-1"],
        text_specification=text,
        semantic_specifications=semantic,
        required_modalities={"speech", "ocr", "objects"},
    )
    assert len(bindings) == 1
    assert bindings[0].generation_id == generation.generation_id


def test_receipt_or_upstream_change_cannot_swap_old_active_generation(tmp_path) -> None:
    repository = _repository_with_video(tmp_path)
    semantic = _semantic_specifications()
    text = _text_specification(semantic)
    _publish_segment_generation(
        repository,
        semantic[0],
        run_id="speech-run-1",
        generation_id="speech-generation-1",
        segments=[_segment("speech-1", "speech", "made basket")],
    )
    _start_text_run(repository, text, run_id="text-run-1")
    first_plan = repository.reserve_text_vector_generation(
        "text-run-1",
        index_specification=_index_specification(),
        semantic_specifications=semantic,
        generation_id="text-generation-1",
    )
    first = repository.commit_text_vector_generation(
        "text-run-1", receipt=_complete_receipt(first_plan)
    )

    _start_text_run(repository, text, run_id="text-run-2")
    second_plan = repository.reserve_text_vector_generation(
        "text-run-2",
        index_specification=_index_specification(),
        semantic_specifications=semantic,
        generation_id="text-generation-2",
    )
    _publish_segment_generation(
        repository,
        semantic[0],
        run_id="speech-run-2",
        generation_id="speech-generation-2",
        segments=[_segment("speech-2", "speech", "new transcript")],
    )

    with pytest.raises(ValueError, match="inputs changed"):
        repository.commit_text_vector_generation(
            "text-run-2", receipt=_complete_receipt(second_plan)
        )

    assert repository.get_active_text_vector_generation("video-1") == first
    assert repository.get_text_vector_generation("text-generation-2") is None
    assert repository.get_stage_run("text-run-2").state is StageState.RUNNING  # type: ignore[union-attr]


def test_direct_complete_and_concurrent_reservation_are_rejected(tmp_path) -> None:
    repository = _repository_with_video(tmp_path)
    semantic = _semantic_specifications()
    text = _text_specification(semantic)
    _start_text_run(repository, text, run_id="text-run-1")
    repository.reserve_text_vector_generation(
        "text-run-1",
        index_specification=_index_specification(),
        semantic_specifications=semantic,
        generation_id="text-generation-1",
    )
    _start_text_run(repository, text, run_id="text-run-2")

    with pytest.raises(ValueError, match="atomic text vector generation commit"):
        repository.transition_stage_run(
            "text-run-2",
            StageState.COMPLETE,
            output_generation="made-up-generation",
        )
    with pytest.raises(ValueError, match="build already active"):
        repository.reserve_text_vector_generation(
            "text-run-2",
            index_specification=_index_specification(),
            semantic_specifications=semantic,
            generation_id="text-generation-2",
        )


def test_failed_and_expired_builds_become_terminal_and_enqueue_exact_gc(tmp_path) -> None:
    repository = _repository_with_video(tmp_path)
    semantic = _semantic_specifications()
    text = _text_specification(semantic)
    _start_text_run(repository, text, run_id="text-run-failed")
    plan = repository.reserve_text_vector_generation(
        "text-run-failed",
        index_specification=_index_specification(),
        semantic_specifications=semantic,
        generation_id="text-generation-failed",
    )

    repository.fail_text_vector_build(
        "text-run-failed", error_code="text_vector_index_failed"
    )

    failed = repository.get_stage_run("text-run-failed")
    assert failed is not None and failed.state is StageState.FAILED
    jobs = repository.list_pending_artifact_gc_jobs()
    assert [(job.generation_id, job.collection_name) for job in jobs] == [
        (plan.generation_id, plan.index_specification.collection_name)
    ]

    _start_text_run(repository, text, run_id="text-run-expired")
    expired = repository.reserve_text_vector_generation(
        "text-run-expired",
        index_specification=_index_specification(),
        semantic_specifications=semantic,
        generation_id="text-generation-expired",
        lease_seconds=1,
    )
    cutoff = (
        datetime.fromisoformat(expired.lease_expires_at) + timedelta(seconds=1)
    ).astimezone(UTC).isoformat()

    expired_ids = repository.expire_text_vector_builds(before=cutoff)

    assert expired_ids == (expired.generation_id,)
    expired_run = repository.get_stage_run("text-run-expired")
    assert expired_run is not None and expired_run.state is StageState.FAILED
    assert expired_run.error_code == "text_vector_build_interrupted"


def test_corrupt_active_vector_pointer_and_manifest_fail_closed(tmp_path) -> None:
    repository = _repository_with_video(tmp_path)
    semantic = _semantic_specifications()
    text = _text_specification(semantic)
    _start_text_run(repository, text, run_id="text-run-1")
    plan = repository.reserve_text_vector_generation(
        "text-run-1",
        index_specification=_index_specification(),
        semantic_specifications=semantic,
        generation_id="text-generation-1",
    )
    repository.commit_text_vector_generation("text-run-1", receipt=_complete_receipt(plan))
    with repository._connect() as connection:
        connection.execute("DROP TRIGGER text_vector_generations_immutable_update")
        connection.execute(
            "UPDATE text_vector_generations SET point_manifest_sha256 = ? WHERE generation_id = ?",
            ("0" * 64, plan.generation_id),
        )

    with pytest.raises(ValueError, match="corrupt"):
        repository.get_active_text_vector_generation("video-1")
    with pytest.raises(ValueError, match="corrupt"):
        repository.list_current_text_vector_bindings(
            video_ids=["video-1"],
            text_specification=text,
            semantic_specifications=semantic,
        )


def test_benchmark_text_binding_rejects_declared_size_before_materialization(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository_with_video(tmp_path)
    semantic = _semantic_specifications()
    text = _text_specification(semantic)
    _start_text_run(repository, text, run_id="text-run-1")
    plan = repository.reserve_text_vector_generation(
        "text-run-1",
        index_specification=_index_specification(),
        semantic_specifications=semantic,
        generation_id="text-generation-1",
    )
    repository.commit_text_vector_generation(
        "text-run-1",
        receipt=_complete_receipt(plan),
    )
    with repository._connect() as connection:
        connection.execute("DROP TRIGGER text_vector_generations_immutable_update")
        connection.execute(
            "UPDATE text_vector_generations SET point_count = ? WHERE generation_id = ?",
            (1_000_000_000, plan.generation_id),
        )
    monkeypatch.setattr(
        Repository,
        "_validate_referenced_text_vector_inputs",
        classmethod(
            lambda *_args, **_kwargs: pytest.fail(
                "oversized point declarations must fail before materialization"
            )
        ),
    )

    with pytest.raises(ValueError, match="benchmark point limit"):
        repository.get_text_vector_search_binding_bounded(
            plan.generation_id,
            max_points=10,
            max_segments=10,
            max_text_bytes=1024,
            max_metadata_bytes=1024,
        )
