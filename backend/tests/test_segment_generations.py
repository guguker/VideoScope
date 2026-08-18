from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import shutil
import sqlite3
from threading import Barrier
from threading import Event

import pytest

import videoscope.repository as repository_module
from videoscope.artifacts import SegmentGeneration, StageKind, StageSpecification, StageState
from videoscope.repository import LATEST_SCHEMA_VERSION, Repository, SegmentRecord


def _specification(kind: StageKind = StageKind.SPEECH) -> StageSpecification:
    return StageSpecification(
        kind=kind,
        schema_version=1,
        implementation_revision=f"{kind.value}-provider-v1",
        model_identity=f"{kind.value}-model@revision",
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


def _start_run(
    repository: Repository,
    specification: StageSpecification,
    run_id: str,
) -> None:
    repository.create_stage_run(
        video_id="video-1",
        specification=specification,
        run_id=run_id,
    )
    repository.transition_stage_run(run_id, StageState.RUNNING)


def _segment(
    segment_id: str,
    *,
    modality: str = "speech",
    text: str = "made basket",
    video_id: str = "video-1",
    thumbnail_path: str | None = None,
    metadata: dict[str, object] | None = None,
) -> SegmentRecord:
    return SegmentRecord(
        id=segment_id,
        video_id=video_id,
        start=1.0,
        end=2.0,
        modality=modality,
        text=text,
        confidence=0.9,
        metadata={} if metadata is None else metadata,
        thumbnail_path=thumbnail_path,
    )


def _commit_speech_generation(
    repository: Repository,
    *,
    run_id: str = "speech-run-1",
    generation_id: str = "speech-generation-1",
    segment_id: str = "speech-segment-1",
) -> tuple[StageSpecification, SegmentGeneration]:
    specification = _specification()
    _start_run(repository, specification, run_id)
    generation = repository.commit_segment_generation(
        run_id,
        generation_id=generation_id,
        segments=[_segment(segment_id)],
    )
    return specification, generation


def test_generation_commit_atomically_activates_segments_and_completes_run(tmp_path) -> None:
    repository = _repository_with_video(tmp_path)
    repository.add_segment(
        segment_id="legacy-segment",
        video_id="video-1",
        start=0.0,
        end=1.0,
        modality="speech",
        text="legacy unproven text",
        confidence=1.0,
    )
    specification = _specification()
    _start_run(repository, specification, "speech-run-1")

    generation = repository.commit_segment_generation(
        "speech-run-1",
        generation_id="speech-generation-1",
        segments=[
            _segment("speech-segment-1"),
            _segment("speech-segment-2", text="three pointer"),
        ],
    )

    assert generation.segment_count == 2
    assert repository.get_segment_generation(generation.generation_id) == generation
    assert repository.get_active_segment_generation("video-1", StageKind.SPEECH) == generation
    assert repository.is_active_segment_generation_current("video-1", specification)
    assert [item.id for item in repository.list_active_segments("video-1", StageKind.SPEECH)] == [
        "speech-segment-1",
        "speech-segment-2",
    ]
    assert {item.id for item in repository.list_segments("video-1")} == {
        "legacy-segment",
        "speech-segment-1",
        "speech-segment-2",
    }
    run = repository.get_stage_run("speech-run-1")
    assert run is not None
    assert run.state is StageState.COMPLETE
    assert run.output_generation == generation.generation_id
    with repository._connect() as connection:
        legacy_generation = connection.execute(
            "SELECT generation_id FROM segments WHERE id = ?",
            ("legacy-segment",),
        ).fetchone()[0]
    assert legacy_generation is None


def test_empty_active_generation_is_distinct_from_missing_generation(tmp_path) -> None:
    repository = _repository_with_video(tmp_path)
    specification = _specification()

    assert repository.get_active_segment_generation("video-1", StageKind.SPEECH) is None
    assert not repository.is_active_segment_generation_current("video-1", specification)
    assert repository.list_active_segments("video-1", StageKind.SPEECH) == []

    _start_run(repository, specification, "speech-run-empty")
    generation = repository.commit_segment_generation(
        "speech-run-empty",
        generation_id="speech-generation-empty",
        segments=[],
    )

    assert generation.segment_count == 0
    assert repository.get_active_segment_generation("video-1", StageKind.SPEECH) == generation
    assert repository.is_active_segment_generation_current("video-1", specification)
    assert repository.list_active_segments("video-1", StageKind.SPEECH) == []


def test_new_activation_retains_previous_immutable_generation(tmp_path) -> None:
    repository = _repository_with_video(tmp_path)
    specification, first = _commit_speech_generation(repository)
    _start_run(repository, specification, "speech-run-2")

    second = repository.commit_segment_generation(
        "speech-run-2",
        generation_id="speech-generation-2",
        segments=[_segment("speech-segment-2", text="new transcript")],
    )

    assert repository.get_active_segment_generation("video-1", StageKind.SPEECH) == second
    assert repository.get_segment_generation(first.generation_id) == first
    assert [item.id for item in repository.list_active_segments("video-1", StageKind.SPEECH)] == [
        "speech-segment-2"
    ]
    with repository._connect() as connection:
        generations = connection.execute(
            "SELECT generation_id FROM segment_generations ORDER BY generation_id"
        ).fetchall()
        segment_ids = connection.execute(
            "SELECT id FROM segments WHERE generation_id = ?",
            (first.generation_id,),
        ).fetchall()
    assert [row[0] for row in generations] == [first.generation_id, second.generation_id]
    assert [row[0] for row in segment_ids] == ["speech-segment-1"]

    with repository._connect() as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE segment_generations SET segment_count = 0 WHERE generation_id = ?",
                (first.generation_id,),
            )
    with repository._connect() as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE segments SET text = ? WHERE id = ?",
                ("tampered", "speech-segment-1"),
            )
    with repository._connect() as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "DELETE FROM segments WHERE id = ?",
                ("speech-segment-1",),
            )


def test_empty_previous_generation_cannot_be_deleted_after_replacement(tmp_path) -> None:
    repository = _repository_with_video(tmp_path)
    specification = _specification()
    _start_run(repository, specification, "speech-run-empty-first")
    first = repository.commit_segment_generation(
        "speech-run-empty-first",
        generation_id="speech-generation-empty-first",
        segments=[],
    )
    _start_run(repository, specification, "speech-run-replacement")
    repository.commit_segment_generation(
        "speech-run-replacement",
        generation_id="speech-generation-replacement",
        segments=[_segment("replacement-segment")],
    )

    with repository._connect() as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "DELETE FROM segment_generations WHERE generation_id = ?",
                (first.generation_id,),
            )

    assert repository.get_segment_generation(first.generation_id) == first


def test_video_deletion_can_cascade_immutable_generation_rows(tmp_path) -> None:
    repository = _repository_with_video(tmp_path)
    _, generation = _commit_speech_generation(repository)

    with repository._connect() as connection:
        connection.execute("DELETE FROM videos WHERE id = ?", ("video-1",))

    assert repository.get_video("video-1") is None
    assert repository.get_segment_generation(generation.generation_id) is None
    with repository._connect() as connection:
        segment_count = connection.execute(
            "SELECT COUNT(*) FROM segments WHERE generation_id = ?",
            (generation.generation_id,),
        ).fetchone()[0]
        pointer_count = connection.execute(
            "SELECT COUNT(*) FROM active_segment_generations WHERE generation_id = ?",
            (generation.generation_id,),
        ).fetchone()[0]
    assert segment_count == 0
    assert pointer_count == 0


@pytest.mark.parametrize(
    "bad_segment",
    [
        _segment("unsafe/id"),
        _segment("wrong-video", video_id="video-2"),
        _segment("wrong-modality", modality="ocr"),
        _segment("unsafe-path", thumbnail_path="../outside.jpg"),
        _segment("unsafe-empty-path", thumbnail_path=""),
        _segment("bad-json", metadata={"score": float("nan")}),
        SegmentRecord(
            id="bad-confidence",
            video_id="video-1",
            start=1.0,
            end=2.0,
            modality="speech",
            text="bad",
            confidence=float("inf"),
            metadata={},
            thumbnail_path=None,
        ),
    ],
)
def test_generation_commit_rejects_invalid_candidate_without_partial_state(
    tmp_path,
    bad_segment: SegmentRecord,
) -> None:
    repository = _repository_with_video(tmp_path)
    specification, active = _commit_speech_generation(repository)
    _start_run(repository, specification, "speech-run-invalid")

    with pytest.raises(ValueError):
        repository.commit_segment_generation(
            "speech-run-invalid",
            generation_id="speech-generation-invalid",
            segments=[bad_segment],
        )

    assert repository.get_active_segment_generation("video-1", StageKind.SPEECH) == active
    assert repository.get_segment_generation("speech-generation-invalid") is None
    run = repository.get_stage_run("speech-run-invalid")
    assert run is not None and run.state is StageState.RUNNING


def test_generation_commit_requires_unique_candidate_ids(tmp_path) -> None:
    repository = _repository_with_video(tmp_path)
    specification, active = _commit_speech_generation(repository)
    _start_run(repository, specification, "speech-run-duplicate")

    with pytest.raises(ValueError, match="unique"):
        repository.commit_segment_generation(
            "speech-run-duplicate",
            generation_id="speech-generation-duplicate",
            segments=[_segment("duplicate"), _segment("duplicate")],
        )

    assert repository.get_active_segment_generation("video-1", StageKind.SPEECH) == active
    assert repository.get_stage_run("speech-run-duplicate").state is StageState.RUNNING  # type: ignore[union-attr]


def test_insert_or_activation_failure_rolls_back_generation_and_old_pointer(tmp_path) -> None:
    repository = _repository_with_video(tmp_path)
    specification, active = _commit_speech_generation(repository)
    _start_run(repository, specification, "speech-run-rejected")
    with repository._connect() as connection:
        connection.execute(
            """
            CREATE TRIGGER reject_segment_generation_activation
            BEFORE UPDATE ON active_segment_generations
            BEGIN
                SELECT RAISE(ABORT, 'simulated activation rejection');
            END
            """
        )

    with pytest.raises(sqlite3.IntegrityError):
        repository.commit_segment_generation(
            "speech-run-rejected",
            generation_id="speech-generation-rejected",
            segments=[_segment("speech-segment-rejected")],
        )

    assert repository.get_active_segment_generation("video-1", StageKind.SPEECH) == active
    assert repository.get_segment_generation("speech-generation-rejected") is None
    assert all(item.id != "speech-segment-rejected" for item in repository.list_segments("video-1"))
    assert repository.get_stage_run("speech-run-rejected").state is StageState.RUNNING  # type: ignore[union-attr]


def test_run_completion_failure_rolls_back_segments_and_activation(tmp_path) -> None:
    repository = _repository_with_video(tmp_path)
    specification, active = _commit_speech_generation(repository)
    _start_run(repository, specification, "speech-run-transition-rejected")
    with repository._connect() as connection:
        connection.execute(
            """
            CREATE TRIGGER reject_segment_generation_run_completion
            BEFORE UPDATE OF state ON stage_runs
            WHEN OLD.run_id = 'speech-run-transition-rejected'
            BEGIN
                SELECT RAISE(ABORT, 'simulated run completion rejection');
            END
            """
        )

    with pytest.raises(sqlite3.IntegrityError):
        repository.commit_segment_generation(
            "speech-run-transition-rejected",
            generation_id="speech-generation-transition-rejected",
            segments=[_segment("speech-segment-transition-rejected")],
        )

    assert repository.get_active_segment_generation("video-1", StageKind.SPEECH) == active
    assert repository.get_segment_generation("speech-generation-transition-rejected") is None
    assert all(
        item.id != "speech-segment-transition-rejected"
        for item in repository.list_segments("video-1")
    )
    run = repository.get_stage_run("speech-run-transition-rejected")
    assert run is not None and run.state is StageState.RUNNING


def test_only_running_segment_stage_can_publish_a_generation(tmp_path) -> None:
    repository = _repository_with_video(tmp_path)
    speech = _specification()
    repository.create_stage_run(
        video_id="video-1",
        specification=speech,
        run_id="queued-speech-run",
    )

    with pytest.raises(ValueError, match="running"):
        repository.commit_segment_generation(
            "queued-speech-run",
            generation_id="queued-generation",
            segments=[],
        )

    text_vectors = _specification(StageKind.TEXT_VECTORS)
    _start_run(repository, text_vectors, "text-vector-run")
    with pytest.raises(ValueError, match="segment"):
        repository.commit_segment_generation(
            "text-vector-run",
            generation_id="text-vector-generation",
            segments=[],
        )


def test_segment_stage_cannot_bypass_atomic_generation_commit(tmp_path) -> None:
    repository = _repository_with_video(tmp_path)
    specification = _specification()
    _start_run(repository, specification, "speech-run-bypass")

    with pytest.raises(ValueError, match="segment generation"):
        repository.transition_stage_run(
            "speech-run-bypass",
            StageState.COMPLETE,
            output_generation="unproven-generation",
        )

    run = repository.get_stage_run("speech-run-bypass")
    assert run is not None and run.state is StageState.RUNNING


@pytest.mark.parametrize(
    ("stage_kind", "modality"),
    [
        (StageKind.SCENES, "scene"),
        (StageKind.SPEECH, "speech"),
        (StageKind.OCR, "ocr"),
        (StageKind.OBJECTS, "objects"),
    ],
)
def test_generation_stage_requires_its_exact_segment_modality(
    tmp_path,
    stage_kind: StageKind,
    modality: str,
) -> None:
    repository = _repository_with_video(tmp_path)
    specification = _specification(stage_kind)
    run_id = f"{stage_kind.value}-run"
    generation_id = f"{stage_kind.value}-generation"
    _start_run(repository, specification, run_id)

    generation = repository.commit_segment_generation(
        run_id,
        generation_id=generation_id,
        segments=[_segment(f"{stage_kind.value}-segment", modality=modality)],
    )

    assert generation.stage_kind is stage_kind
    assert repository.list_active_segments("video-1", stage_kind)[0].modality == modality


def test_currentness_requires_the_expected_stage_specification(tmp_path) -> None:
    repository = _repository_with_video(tmp_path)
    specification, _ = _commit_speech_generation(repository)
    changed = StageSpecification(
        kind=StageKind.SPEECH,
        schema_version=2,
        implementation_revision="speech-provider-v2",
        model_identity="speech-model@new-revision",
    )

    assert repository.is_active_segment_generation_current("video-1", specification)
    assert not repository.is_active_segment_generation_current("video-1", changed)


@pytest.mark.parametrize(
    "tamper",
    ["pointer", "run", "specification", "source", "registry"],
)
def test_active_generation_reads_fail_closed_on_corrupt_identity(tmp_path, tamper: str) -> None:
    repository = _repository_with_video(tmp_path)
    _, generation = _commit_speech_generation(repository)
    with repository._connect() as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        if tamper == "pointer":
            connection.execute(
                "UPDATE active_segment_generations SET generation_id = ? WHERE generation_id = ?",
                ("missing-generation", generation.generation_id),
            )
        elif tamper == "registry":
            connection.execute(
                "UPDATE stage_specifications SET canonical_json = ? WHERE specification_hash = ?",
                ("{}", generation.specification_hash),
            )
        else:
            connection.execute("DROP TRIGGER segment_generations_immutable_update")
            column, value = {
                "run": ("run_id", "missing-run"),
                "specification": ("specification_hash", "b" * 64),
                "source": ("source_sha256", "b" * 64),
            }[tamper]
            connection.execute(
                f"UPDATE segment_generations SET {column} = ? WHERE generation_id = ?",
                (value, generation.generation_id),
            )

    with pytest.raises(ValueError):
        repository.get_active_segment_generation("video-1", StageKind.SPEECH)
    with pytest.raises(ValueError):
        repository.list_active_segments("video-1", StageKind.SPEECH)


def test_active_segment_read_rejects_corrupt_modality(tmp_path) -> None:
    repository = _repository_with_video(tmp_path)
    _commit_speech_generation(repository)
    with repository._connect() as connection:
        connection.execute("DROP TRIGGER generated_segments_immutable_update")
        connection.execute(
            "UPDATE segments SET modality = ? WHERE id = ?",
            ("ocr", "speech-segment-1"),
        )

    with pytest.raises(ValueError, match="modality"):
        repository.list_active_segments("video-1", StageKind.SPEECH)


def test_concurrent_generation_commits_are_whole_and_last_activation_wins(tmp_path) -> None:
    repository = _repository_with_video(tmp_path)
    specification = _specification()
    _start_run(repository, specification, "speech-run-a")
    _start_run(repository, specification, "speech-run-b")
    starting_line = Barrier(2)

    def commit(label: str) -> SegmentGeneration:
        starting_line.wait()
        concurrent_repository = Repository(repository.database_path)
        return concurrent_repository.commit_segment_generation(
            f"speech-run-{label}",
            generation_id=f"speech-generation-{label}",
            segments=[_segment(f"speech-segment-{label}", text=label)],
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        generations = list(executor.map(commit, ("a", "b")))

    active = repository.get_active_segment_generation("video-1", StageKind.SPEECH)
    assert active is not None
    assert active.generation_id in {item.generation_id for item in generations}
    assert {repository.get_stage_run(item.run_id).state for item in generations} == {  # type: ignore[union-attr]
        StageState.COMPLETE
    }
    assert all(repository.get_segment_generation(item.generation_id) == item for item in generations)
    assert [item.id for item in repository.list_active_segments("video-1", StageKind.SPEECH)] == [
        active.generation_id.replace("generation", "segment")
    ]


def test_concurrent_scene_activation_keeps_video_thumbnail_with_active_generation(
    tmp_path,
) -> None:
    repository = _repository_with_video(tmp_path)
    specification = _specification(StageKind.SCENES)
    _start_run(repository, specification, "scene-run-a")
    _start_run(repository, specification, "scene-run-b")
    starting_line = Barrier(2)

    def commit(label: str) -> SegmentGeneration:
        thumbnail = f"/managed/generations/scene-generation-{label}/scene.jpg"
        starting_line.wait()
        return Repository(repository.database_path).commit_segment_generation(
            f"scene-run-{label}",
            generation_id=f"scene-generation-{label}",
            segments=[
                _segment(
                    f"scene-segment-{label}",
                    modality="scene",
                    thumbnail_path=thumbnail,
                )
            ],
            video_thumbnail_path=thumbnail,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(commit, ("a", "b")))

    active_segments = repository.list_active_segments("video-1", StageKind.SCENES)
    video = repository.get_video("video-1")
    assert video is not None
    assert len(active_segments) == 1
    assert video.thumbnail_path == active_segments[0].thumbnail_path


def test_generation_completion_timestamp_is_taken_after_write_lock(
    tmp_path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    repository = _repository_with_video(tmp_path)
    specification = _specification()
    _start_run(repository, specification, "speech-run-waits-for-lock")
    timestamp_requested = Event()
    real_now = repository_module._now

    def tracked_now() -> str:
        timestamp_requested.set()
        return real_now()

    monkeypatch.setattr(repository_module, "_now", tracked_now)
    write_lock = repository._connect()
    write_lock.execute("BEGIN IMMEDIATE")
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            repository.commit_segment_generation,
            "speech-run-waits-for-lock",
            generation_id="speech-generation-waits-for-lock",
            segments=[],
        )
        requested_while_blocked = timestamp_requested.wait(0.05)
        write_lock.rollback()
        write_lock.close()
        generation = future.result()

    assert not requested_while_blocked
    assert timestamp_requested.is_set()
    assert generation.completed_at == repository.get_stage_run(
        "speech-run-waits-for-lock"
    ).finished_at  # type: ignore[union-attr]


def test_v5_migrates_a_copy_without_claiming_legacy_segments_current(tmp_path) -> None:
    source_path = tmp_path / "source-v4.sqlite3"
    with sqlite3.connect(source_path) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("BEGIN IMMEDIATE")
        for version, migration in repository_module._SCHEMA_MIGRATIONS[:4]:
            migration(connection)
            connection.execute(f"PRAGMA user_version = {version}")
        connection.commit()
    source = Repository(source_path)
    source.create_video(
        video_id="legacy-video",
        original_name="legacy.mp4",
        stored_name="legacy-video.mp4",
        media_path="/tmp/legacy-video.mp4",
        size_bytes=100,
    )
    source.add_segment(
        segment_id="legacy-speech",
        video_id="legacy-video",
        start=1.0,
        end=2.0,
        modality="speech",
        text="legacy text",
        confidence=1.0,
    )
    copied_path = tmp_path / "copied.sqlite3"
    shutil.copy2(source_path, copied_path)

    copied = Repository(copied_path)
    copied.initialize()
    copied.initialize()

    assert source.schema_version() == 4
    assert copied.schema_version() == LATEST_SCHEMA_VERSION == 6
    assert [item.id for item in copied.list_segments("legacy-video")] == ["legacy-speech"]
    assert copied.get_active_segment_generation("legacy-video", StageKind.SPEECH) is None
    assert copied.list_active_segments("legacy-video", StageKind.SPEECH) == []
    with copied._connect() as connection:
        generation_id = connection.execute(
            "SELECT generation_id FROM segments WHERE id = ?",
            ("legacy-speech",),
        ).fetchone()[0]
        generation_count = connection.execute(
            "SELECT COUNT(*) FROM segment_generations"
        ).fetchone()[0]
    assert generation_id is None
    assert generation_count == 0


def test_interrupted_v5_migration_rolls_back_all_schema_and_preserves_v4_data(
    tmp_path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    database_path = tmp_path / "interrupted-v5.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("BEGIN IMMEDIATE")
        for version, migration in repository_module._SCHEMA_MIGRATIONS[:4]:
            migration(connection)
            connection.execute(f"PRAGMA user_version = {version}")
        connection.execute(
            """
            INSERT INTO videos (
                id, original_name, stored_name, media_path, size_bytes, asset_id,
                status, progress, stage, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-video",
                "legacy.mp4",
                "legacy.mp4",
                "/tmp/legacy.mp4",
                100,
                "ready",
                1.0,
                "ready",
                "2026-08-18T10:00:00+00:00",
                "2026-08-18T10:00:00+00:00",
            ),
        )
        connection.execute(
            """
            INSERT INTO segments (
                id, video_id, start, end, modality, text,
                confidence, metadata_json, thumbnail_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
            """,
            (
                "legacy-segment",
                "legacy-video",
                1.0,
                2.0,
                "speech",
                "legacy",
                1.0,
                "{}",
            ),
        )
        connection.commit()

    real_migration = repository_module._SCHEMA_MIGRATIONS[4][1]

    def fail_after_v5_schema(connection) -> None:  # type: ignore[no-untyped-def]
        real_migration(connection)
        connection.execute("CREATE TABLE migration_v5_must_rollback (value TEXT)")
        raise RuntimeError("simulated v5 interruption")

    monkeypatch.setattr(
        repository_module,
        "_SCHEMA_MIGRATIONS",
        (*repository_module._SCHEMA_MIGRATIONS[:4], (5, fail_after_v5_schema)),
    )

    with pytest.raises(RuntimeError, match="simulated v5 interruption"):
        Repository(database_path).initialize()

    with sqlite3.connect(database_path) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        segment_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(segments)").fetchall()
        }
        generation_table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            ("segment_generations",),
        ).fetchone()
        rollback_table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            ("migration_v5_must_rollback",),
        ).fetchone()
        legacy_count = connection.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
    assert version == 4
    assert "generation_id" not in segment_columns
    assert generation_table is None
    assert rollback_table is None
    assert legacy_count == 1
