from concurrent.futures import ThreadPoolExecutor
import sqlite3

import pytest

import videoscope.repository as repository_module
from videoscope.artifacts import StageKind, StageSpecification, StageState
from videoscope.repository import LATEST_SCHEMA_VERSION, Repository


def test_repository_round_trips_video_and_segments(tmp_path) -> None:
    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()
    video = repository.create_video(
        video_id="video-1",
        original_name="match.mp4",
        stored_name="video-1.mp4",
        media_path="/tmp/video-1.mp4",
        size_bytes=1024,
    )
    repository.add_segment(
        segment_id="segment-1",
        video_id=video.id,
        start=4.0,
        end=9.5,
        modality="speech",
        text="important moment",
        confidence=0.8,
        metadata={"language": "en"},
    )

    loaded = repository.get_video(video.id)
    segments = repository.list_segments(video.id)

    assert loaded is not None
    assert loaded.original_name == "match.mp4"
    assert segments[0].metadata == {"language": "en"}


def test_repository_updates_processing_state_atomically(tmp_path) -> None:
    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()
    repository.create_video(
        video_id="video-1",
        original_name="match.mp4",
        stored_name="video-1.mp4",
        media_path="/tmp/video-1.mp4",
        size_bytes=1024,
    )

    repository.update_video("video-1", status="processing", progress=0.45, stage="ocr")

    loaded = repository.get_video("video-1")
    assert loaded is not None
    assert (loaded.status, loaded.progress, loaded.stage) == ("processing", 0.45, "ocr")


@pytest.mark.parametrize(
    "changes",
    [
        {"start": -1.0},
        {"start": float("nan")},
        {"end": float("inf")},
        {"confidence": float("nan")},
        {"confidence": 1.1},
        {"modality": "qwen_video"},
        {"metadata": []},
        {"metadata": {"score": float("nan")}},
    ],
)
def test_repository_rejects_invalid_segment_contract(tmp_path, changes) -> None:  # type: ignore[no-untyped-def]
    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()
    repository.create_video(
        video_id="video-1",
        original_name="match.mp4",
        stored_name="video-1.mp4",
        media_path="/tmp/video-1.mp4",
        size_bytes=1024,
    )
    payload = {
        "segment_id": "segment-1",
        "video_id": "video-1",
        "start": 1.0,
        "end": 2.0,
        "modality": "speech",
        "text": "moment",
        "confidence": 0.8,
        "metadata": {},
    }
    payload.update(changes)

    with pytest.raises(ValueError):
        repository.add_segment(**payload)


def test_repository_ignores_invalid_legacy_segment_rows(tmp_path) -> None:
    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()
    repository.create_video(
        video_id="video-1",
        original_name="match.mp4",
        stored_name="video-1.mp4",
        media_path="/tmp/video-1.mp4",
        size_bytes=1024,
    )
    with repository._connect() as connection:
        connection.execute(
            """
            INSERT INTO segments (
                id, video_id, start, end, modality, text,
                confidence, metadata_json, thumbnail_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-bad",
                "video-1",
                -1.0,
                2.0,
                "speech",
                "bad",
                0.8,
                "{}",
                None,
            ),
        )

    assert repository.list_segments("video-1") == []


def _speech_specification() -> StageSpecification:
    return StageSpecification(
        kind=StageKind.SPEECH,
        schema_version=1,
        implementation_revision="speech-provider-v1",
        model_identity="whisper@pinned-revision",
        parameters={"language": "auto"},
        dependencies={"mlx-whisper": "0.4.2"},
    )


def _visual_specification() -> StageSpecification:
    return StageSpecification(
        kind=StageKind.VISUAL_DENSE,
        schema_version=1,
        implementation_revision="visual-provider-v1",
        model_identity="visual-model@pinned-revision",
    )


def _create_video_with_asset_identity(repository: Repository) -> None:
    repository.create_video_with_asset(
        video_id="video-1",
        original_name="match.mp4",
        stored_name="video-1.mp4",
        media_path="/tmp/video-1.mp4",
        size_bytes=1024,
        source_sha256="a" * 64,
    )


def test_numbered_migrations_preserve_legacy_database_data(tmp_path) -> None:
    database_path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE videos (
                id TEXT PRIMARY KEY,
                original_name TEXT NOT NULL,
                stored_name TEXT NOT NULL UNIQUE,
                media_path TEXT NOT NULL UNIQUE,
                size_bytes INTEGER NOT NULL,
                status TEXT NOT NULL,
                progress REAL NOT NULL,
                stage TEXT NOT NULL,
                duration REAL,
                width INTEGER,
                height INTEGER,
                fps REAL,
                thumbnail_path TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE segments (
                id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
                start REAL NOT NULL,
                end REAL NOT NULL,
                modality TEXT NOT NULL,
                text TEXT NOT NULL,
                confidence REAL NOT NULL,
                metadata_json TEXT NOT NULL,
                thumbnail_path TEXT
            );
            INSERT INTO videos (
                id, original_name, stored_name, media_path, size_bytes,
                status, progress, stage, created_at, updated_at
            ) VALUES (
                'legacy-video', 'legacy.mp4', 'legacy-video.mp4',
                '/private/legacy-video.mp4', 2048, 'ready', 1.0, 'ready',
                '2026-07-01T00:00:00+00:00', '2026-07-02T00:00:00+00:00'
            );
            INSERT INTO segments (
                id, video_id, start, end, modality, text,
                confidence, metadata_json, thumbnail_path
            ) VALUES (
                'legacy-speech', 'legacy-video', 1.0, 2.0, 'speech',
                'legacy words', 0.9, '{}', NULL
            );
            """
        )

    repository = Repository(database_path)
    repository.initialize()
    repository.initialize()

    video = repository.get_video("legacy-video")
    assert video is not None
    assert video.original_name == "legacy.mp4"
    assert video.display_name is None
    assert video.asset_id is None
    assert repository.get_video_asset("legacy-video") is None
    assert repository.list_segments("legacy-video")[0].text == "legacy words"
    assert repository.schema_version() == LATEST_SCHEMA_VERSION


def test_repository_refuses_a_database_from_a_newer_schema(tmp_path) -> None:
    database_path = tmp_path / "future.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute(f"PRAGMA user_version = {LATEST_SCHEMA_VERSION + 1}")

    with pytest.raises(RuntimeError, match="newer"):
        Repository(database_path).initialize()


def test_migration_from_user_version_zero_with_display_name_is_idempotent(tmp_path) -> None:
    database_path = tmp_path / "legacy-with-display-name.sqlite3"
    repository = Repository(database_path)
    repository.initialize()
    repository.create_video(
        video_id="legacy-video",
        original_name="legacy.mp4",
        stored_name="legacy-video.mp4",
        media_path="/tmp/legacy-video.mp4",
        size_bytes=1024,
    )
    repository.update_video("legacy-video", display_name="Important game")
    with repository._connect() as connection:
        connection.execute("DROP TABLE stage_runs")
        connection.execute("DROP TABLE stage_specifications")
        connection.execute("PRAGMA user_version = 0")

    repository.initialize()
    repository.initialize()

    migrated = repository.get_video("legacy-video")
    assert migrated is not None
    assert migrated.display_name == "Important game"
    assert repository.schema_version() == LATEST_SCHEMA_VERSION


def test_failed_numbered_migration_rolls_back_schema_version_and_ddl(
    tmp_path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    database_path = tmp_path / "interrupted.sqlite3"
    repository = Repository(database_path)
    repository.initialize()
    repository.create_video(
        video_id="legacy-video",
        original_name="legacy.mp4",
        stored_name="legacy-video.mp4",
        media_path="/tmp/legacy-video.mp4",
        size_bytes=1024,
    )
    with repository._connect() as connection:
        connection.execute("DROP TABLE stage_runs")
        connection.execute("DROP TABLE stage_specifications")
        connection.execute("PRAGMA user_version = 0")

    def fail_final_migration(connection) -> None:  # type: ignore[no-untyped-def]
        connection.execute("CREATE TABLE migration_must_rollback (value TEXT)")
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(
        repository_module,
        "_SCHEMA_MIGRATIONS",
        (*repository_module._SCHEMA_MIGRATIONS[:2], (3, fail_final_migration)),
    )

    with pytest.raises(RuntimeError, match="simulated interruption"):
        repository.initialize()

    with sqlite3.connect(database_path) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        rollback_table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            ("migration_must_rollback",),
        ).fetchone()
        video_count = connection.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
    assert version == 0
    assert rollback_table is None
    assert video_count == 1


def test_concurrent_initialization_serializes_numbered_migrations(tmp_path) -> None:
    database_path = tmp_path / "concurrent.sqlite3"

    def initialize() -> None:
        Repository(database_path).initialize()

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(initialize) for _ in range(8)]
        for future in futures:
            future.result()

    repository = Repository(database_path)
    assert repository.schema_version() == LATEST_SCHEMA_VERSION
    with repository._connect() as connection:
        tables = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    assert {
        "videos",
        "segments",
        "assets",
        "stage_specifications",
        "stage_runs",
        "segment_generations",
        "active_segment_generations",
    } <= tables


def test_migration_never_infers_complete_provenance_from_legacy_ready_state(tmp_path) -> None:
    repository = Repository(tmp_path / "legacy.sqlite3")
    repository.initialize()
    repository.create_video(
        video_id="video-1",
        original_name="match.mp4",
        stored_name="video-1.mp4",
        media_path="/tmp/video-1.mp4",
        size_bytes=1024,
    )
    repository.update_video("video-1", status="ready", progress=1.0, stage="ready")
    repository.add_segment(
        segment_id="speech-1",
        video_id="video-1",
        start=1.0,
        end=2.0,
        modality="speech",
        text="legacy evidence",
        confidence=1.0,
    )

    repository.initialize()

    assert repository.list_stage_runs("video-1") == []
    assert repository.get_latest_stage_run("video-1", StageKind.SPEECH) is None


def test_repository_persists_stage_run_lifecycle_and_specification(tmp_path) -> None:
    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()
    _create_video_with_asset_identity(repository)
    specification = _visual_specification()

    queued = repository.create_stage_run(
        video_id="video-1",
        specification=specification,
        run_id="speech-run-1",
    )
    running = repository.transition_stage_run("speech-run-1", StageState.RUNNING)
    complete = repository.transition_stage_run(
        "speech-run-1",
        StageState.COMPLETE,
        output_generation="speech-generation-1",
    )

    assert queued.attempt == 1
    assert queued.state is StageState.QUEUED
    assert running.started_at is not None
    assert complete.state is StageState.COMPLETE
    assert complete.finished_at is not None
    assert complete.output_generation == "speech-generation-1"
    assert repository.get_stage_specification(specification.specification_hash) == specification
    assert repository.get_latest_stage_run("video-1", StageKind.VISUAL_DENSE) == complete

    repository.initialize()
    assert repository.get_stage_run("speech-run-1") == complete


def test_repository_rejects_explicit_empty_run_id_without_persisting_specification(
    tmp_path,
) -> None:
    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()
    _create_video_with_asset_identity(repository)
    specification = _speech_specification()

    with pytest.raises(ValueError, match="run id"):
        repository.create_stage_run(
            video_id="video-1",
            specification=specification,
            run_id="",
        )

    assert repository.get_stage_specification(specification.specification_hash) is None


def test_repository_fails_closed_when_persisted_specification_is_corrupted(tmp_path) -> None:
    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()
    _create_video_with_asset_identity(repository)
    specification = _speech_specification()
    repository.create_stage_run(
        video_id="video-1",
        specification=specification,
        run_id="speech-run-1",
    )
    with repository._connect() as connection:
        connection.execute(
            """
            UPDATE stage_specifications
            SET canonical_json = ?
            WHERE specification_hash = ?
            """,
            ("{}", specification.specification_hash),
        )

    with pytest.raises(ValueError, match="specification"):
        repository.get_stage_run("speech-run-1")


def test_repository_fails_closed_when_stage_specification_link_is_dangling(tmp_path) -> None:
    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()
    _create_video_with_asset_identity(repository)
    specification = _speech_specification()
    repository.create_stage_run(
        video_id="video-1",
        specification=specification,
        run_id="speech-run-1",
    )
    with repository._connect() as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute(
            "DELETE FROM stage_specifications WHERE specification_hash = ?",
            (specification.specification_hash,),
        )

    with pytest.raises(ValueError, match="specification"):
        repository.get_stage_run("speech-run-1")
    with pytest.raises(ValueError, match="specification"):
        repository.list_stage_runs("video-1", StageKind.SPEECH)
    with pytest.raises(ValueError, match="specification"):
        repository.get_latest_stage_run("video-1", StageKind.SPEECH)


def test_repository_persists_sanitized_failure_and_retry_lineage(tmp_path) -> None:
    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()
    _create_video_with_asset_identity(repository)
    specification = _speech_specification()
    repository.create_stage_run(
        video_id="video-1",
        specification=specification,
        run_id="speech-run-1",
    )
    repository.transition_stage_run("speech-run-1", StageState.RUNNING)
    failed = repository.transition_stage_run(
        "speech-run-1",
        StageState.FAILED,
        error_code="provider_unavailable",
    )

    retry = repository.create_stage_run(
        video_id="video-1",
        specification=specification,
        run_id="speech-run-2",
        retry_of_run_id=failed.run_id,
    )

    assert failed.error_code == "provider_unavailable"
    assert retry.attempt == 2
    assert retry.retry_of_run_id == "speech-run-1"
    assert [run.run_id for run in repository.list_stage_runs("video-1")] == [
        "speech-run-1",
        "speech-run-2",
    ]
    assert repository.list_stage_runs("video-1", StageKind.SPEECH) == [failed, retry]


def test_concurrent_stage_attempts_receive_unique_monotonic_numbers(tmp_path) -> None:
    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()
    _create_video_with_asset_identity(repository)
    specification = _speech_specification()

    def create_attempt(index: int):
        return repository.create_stage_run(
            video_id="video-1",
            specification=specification,
            run_id=f"speech-run-{index}",
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        attempts = list(executor.map(create_attempt, range(1, 9)))

    assert sorted(run.attempt for run in attempts) == list(range(1, 9))
    assert len({run.run_id for run in attempts}) == 8


def test_repository_rejects_corrupted_retry_cycle_and_source_lineage(tmp_path) -> None:
    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()
    _create_video_with_asset_identity(repository)
    specification = _speech_specification()
    repository.create_stage_run(
        video_id="video-1",
        specification=specification,
        run_id="speech-run-1",
    )
    repository.transition_stage_run(
        "speech-run-1",
        StageState.FAILED,
        error_code="provider_unavailable",
    )
    repository.create_stage_run(
        video_id="video-1",
        specification=specification,
        run_id="speech-run-2",
        retry_of_run_id="speech-run-1",
    )
    repository.transition_stage_run(
        "speech-run-2",
        StageState.FAILED,
        error_code="provider_unavailable",
    )
    with repository._connect() as connection:
        connection.execute(
            "UPDATE stage_runs SET retry_of_run_id = ? WHERE run_id = ?",
            ("speech-run-2", "speech-run-1"),
        )

    with pytest.raises(ValueError, match="retry lineage"):
        repository.get_stage_run("speech-run-2")

    with repository._connect() as connection:
        connection.execute(
            "UPDATE stage_runs SET retry_of_run_id = NULL WHERE run_id = ?",
            ("speech-run-1",),
        )
        connection.execute(
            "UPDATE stage_runs SET source_sha256 = ? WHERE run_id = ?",
            ("b" * 64, "speech-run-2"),
        )

    with pytest.raises(ValueError, match="retry lineage"):
        repository.get_stage_run("speech-run-2")


def test_retry_parent_cannot_be_deleted_without_its_video(tmp_path) -> None:
    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()
    _create_video_with_asset_identity(repository)
    specification = _speech_specification()
    repository.create_stage_run(
        video_id="video-1",
        specification=specification,
        run_id="speech-run-1",
    )
    repository.transition_stage_run(
        "speech-run-1",
        StageState.FAILED,
        error_code="provider_unavailable",
    )
    repository.create_stage_run(
        video_id="video-1",
        specification=specification,
        run_id="speech-run-2",
        retry_of_run_id="speech-run-1",
    )

    with repository._connect() as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("DELETE FROM stage_runs WHERE run_id = ?", ("speech-run-1",))

    persisted_retry = repository.get_stage_run("speech-run-2")
    assert persisted_retry is not None
    assert persisted_retry.retry_of_run_id == "speech-run-1"

    with repository._connect() as connection:
        connection.execute("DELETE FROM videos WHERE id = ?", ("video-1",))
    assert repository.get_stage_run("speech-run-1") is None
    assert repository.get_stage_run("speech-run-2") is None


def test_repository_persists_not_configured_cancelled_and_stale_states(tmp_path) -> None:
    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()
    _create_video_with_asset_identity(repository)
    specification = _visual_specification()
    repository.create_stage_run(
        video_id="video-1",
        specification=specification,
        run_id="not-configured-run",
    )
    not_configured = repository.transition_stage_run(
        "not-configured-run",
        StageState.NOT_CONFIGURED,
    )
    repository.create_stage_run(
        video_id="video-1",
        specification=specification,
        run_id="cancelled-run",
    )
    cancelled = repository.transition_stage_run("cancelled-run", StageState.CANCELLED)
    repository.create_stage_run(
        video_id="video-1",
        specification=specification,
        run_id="complete-run",
    )
    repository.transition_stage_run("complete-run", StageState.RUNNING)
    complete = repository.transition_stage_run(
        "complete-run",
        StageState.COMPLETE,
        output_generation="generation-1",
    )
    stale = repository.transition_stage_run("complete-run", StageState.STALE)

    assert not_configured.state is StageState.NOT_CONFIGURED
    assert cancelled.state is StageState.CANCELLED
    assert stale.state is StageState.STALE
    assert stale.output_generation == complete.output_generation


def test_repository_rejects_raw_failure_details_and_invalid_transitions(tmp_path) -> None:
    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()
    _create_video_with_asset_identity(repository)
    repository.create_stage_run(
        video_id="video-1",
        specification=_speech_specification(),
        run_id="speech-run-1",
    )

    with pytest.raises(ValueError, match="error code"):
        repository.transition_stage_run(
            "speech-run-1",
            StageState.FAILED,
            error_code="failed at /Users/person/secret.mp4",
        )
    with pytest.raises(ValueError, match="transition"):
        repository.transition_stage_run(
            "speech-run-1",
            StageState.COMPLETE,
            output_generation="generation-1",
        )


def test_stage_run_rows_are_deleted_with_their_video(tmp_path) -> None:
    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()
    _create_video_with_asset_identity(repository)
    repository.create_stage_run(
        video_id="video-1",
        specification=_speech_specification(),
        run_id="speech-run-1",
    )

    with repository._connect() as connection:
        connection.execute("DELETE FROM videos WHERE id = ?", ("video-1",))

    assert repository.get_stage_run("speech-run-1") is None
