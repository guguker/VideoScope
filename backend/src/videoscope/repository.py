from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json
import logging
import math
from numbers import Real
import os
import re
import sqlite3
import stat
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Lock
from time import sleep
from typing import Any, Callable
from uuid import uuid4
from weakref import WeakValueDictionary

from videoscope.artifacts import (
    ArtifactGCJob,
    AssetIdentityError,
    AssetRecord,
    SEGMENT_STAGE_KINDS,
    SegmentGeneration,
    StageKind,
    StageRun,
    StageSpecification,
    StageState,
    TEXT_VECTOR_INPUT_STAGE_KINDS,
    TextVectorBuildPlan,
    TextVectorBuildReceipt,
    TextVectorGeneration,
    TextVectorGenerationInput,
    TextVectorIndexSpecification,
    TextVectorPointSource,
    TextVectorSearchBinding,
    asset_id_for_sha256,
    validate_artifact_identifier,
    validate_error_code,
    validate_stage_transition,
)
from videoscope.search.text_matching import lexical_match


logger = logging.getLogger(__name__)
SEGMENT_MODALITIES = frozenset({"scene", "speech", "ocr", "objects"})
SEGMENT_STAGE_MODALITIES = {
    StageKind.SCENES: "scene",
    StageKind.SPEECH: "speech",
    StageKind.OCR: "ocr",
    StageKind.OBJECTS: "objects",
}
LATEST_SCHEMA_VERSION = 6
_DATABASE_INITIALIZE_LOCK = Lock()
_ASSET_IDENTITY_LOCKS_GUARD = Lock()
_ASSET_IDENTITY_LOCKS: WeakValueDictionary[tuple[str, str], Any] = WeakValueDictionary()
_JOURNAL_MODE_RETRIES = 8
ASSET_HASH_CHUNK_SIZE = 1024 * 1024
_VIDEO_THUMBNAIL_UNCHANGED = object()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _reject_non_finite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _enable_wal(connection: sqlite3.Connection) -> None:
    journal_mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    if journal_mode == "wal":
        return
    for attempt in range(_JOURNAL_MODE_RETRIES):
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            return
        except sqlite3.OperationalError as error:
            if "locked" not in str(error).lower() or attempt == _JOURNAL_MODE_RETRIES - 1:
                raise
            sleep(0.01 * (2**attempt))


def _ensure_supported_schema_version(version: int) -> None:
    if version > LATEST_SCHEMA_VERSION:
        raise RuntimeError(
            "database schema is newer than this VideoScope build: "
            f"{version} > {LATEST_SCHEMA_VERSION}"
        )


def _asset_identity_lock(database_path: Path, video_id: str) -> Any:
    key = (str(database_path.resolve()), video_id)
    with _ASSET_IDENTITY_LOCKS_GUARD:
        lock = _ASSET_IDENTITY_LOCKS.get(key)
        if lock is None:
            lock = Lock()
            _ASSET_IDENTITY_LOCKS[key] = lock
        return lock


def _file_identity(
    file_stat: os.stat_result | object,
) -> tuple[int, int, int, int, int, int]:
    return (
        int(getattr(file_stat, "st_dev")),
        int(getattr(file_stat, "st_ino")),
        int(getattr(file_stat, "st_mode")),
        int(getattr(file_stat, "st_size")),
        int(getattr(file_stat, "st_mtime_ns")),
        int(getattr(file_stat, "st_ctime_ns")),
    )


def _hash_managed_regular_file(path: Path, *, expected_size: int) -> str:
    try:
        path_before = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise AssetIdentityError("asset media must be a managed regular file") from error
    if not stat.S_ISREG(path_before.st_mode):
        raise AssetIdentityError("asset media must be a managed regular file")
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        file_descriptor = os.open(path, flags)
    except OSError as error:
        raise AssetIdentityError("asset media must be a managed regular file") from error
    try:
        before = os.fstat(file_descriptor)
        if not stat.S_ISREG(before.st_mode) or _file_identity(before) != _file_identity(
            path_before
        ):
            raise AssetIdentityError("asset media must be a managed regular file")
        if before.st_size != expected_size or before.st_size <= 0:
            raise AssetIdentityError("asset media size does not match the video record")
        digest = hashlib.sha256()
        while chunk := os.read(file_descriptor, ASSET_HASH_CHUNK_SIZE):
            digest.update(chunk)
        after = os.fstat(file_descriptor)
    finally:
        os.close(file_descriptor)
    try:
        path_after = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise AssetIdentityError("asset media changed while hashing") from error
    if _file_identity(before) != _file_identity(after) or _file_identity(
        after
    ) != _file_identity(path_after):
        raise AssetIdentityError("asset media changed while hashing")
    return digest.hexdigest()


def _migration_1_create_library_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS videos (
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
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS segments (
            id TEXT PRIMARY KEY,
            video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
            start REAL NOT NULL,
            end REAL NOT NULL,
            modality TEXT NOT NULL,
            text TEXT NOT NULL,
            confidence REAL NOT NULL,
            metadata_json TEXT NOT NULL,
            thumbnail_path TEXT
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_segments_video_time
        ON segments(video_id, start, end)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_segments_modality
        ON segments(modality)
        """
    )


def _migration_2_add_video_display_name(connection: sqlite3.Connection) -> None:
    columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(videos)").fetchall()
    }
    if "display_name" not in columns:
        connection.execute("ALTER TABLE videos ADD COLUMN display_name TEXT")


def _sql_values(values: list[str]) -> str:
    return ", ".join(f"'{value}'" for value in values)


def _migration_3_add_stage_provenance(connection: sqlite3.Connection) -> None:
    stage_kinds = _sql_values([kind.value for kind in StageKind])
    stage_states = _sql_values([state.value for state in StageState])
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS stage_specifications (
            specification_hash TEXT PRIMARY KEY,
            stage_kind TEXT NOT NULL CHECK(stage_kind IN ({stage_kinds})),
            canonical_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS stage_runs (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL UNIQUE,
            video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
            specification_hash TEXT NOT NULL
                REFERENCES stage_specifications(specification_hash),
            source_sha256 TEXT NOT NULL,
            attempt INTEGER NOT NULL CHECK(attempt >= 1),
            state TEXT NOT NULL CHECK(state IN ({stage_states})),
            output_generation TEXT,
            error_code TEXT,
            retry_of_run_id TEXT REFERENCES stage_runs(run_id),
            created_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT,
            updated_at TEXT NOT NULL,
            UNIQUE(video_id, specification_hash, source_sha256, attempt),
            CHECK(retry_of_run_id IS NULL OR retry_of_run_id <> run_id)
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_stage_runs_video_spec
        ON stage_runs(video_id, specification_hash, sequence)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_stage_runs_state
        ON stage_runs(state)
        """
    )


def _migration_4_add_asset_identity(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS assets (
            asset_id TEXT PRIMARY KEY,
            sha256 TEXT NOT NULL UNIQUE,
            size_bytes INTEGER NOT NULL CHECK(size_bytes > 0),
            created_at TEXT NOT NULL,
            CHECK(asset_id = 'sha256:' || sha256)
        )
        """
    )
    columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(videos)").fetchall()
    }
    if "asset_id" not in columns:
        connection.execute(
            """
            ALTER TABLE videos
            ADD COLUMN asset_id TEXT REFERENCES assets(asset_id)
            """
        )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_videos_asset_id
        ON videos(asset_id)
        """
    )
    connection.execute(
        """
        CREATE TRIGGER IF NOT EXISTS videos_asset_id_immutable
        BEFORE UPDATE OF asset_id ON videos
        WHEN OLD.asset_id IS NOT NULL AND NEW.asset_id IS NOT OLD.asset_id
        BEGIN
            SELECT RAISE(ABORT, 'video asset identity is immutable');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER IF NOT EXISTS assets_immutable_update
        BEFORE UPDATE ON assets
        BEGIN
            SELECT RAISE(ABORT, 'asset identity is immutable');
        END
        """
    )


def _migration_5_add_segment_generations(connection: sqlite3.Connection) -> None:
    segment_stage_kinds = _sql_values(
        [kind.value for kind in sorted(SEGMENT_STAGE_KINDS, key=lambda item: item.value)]
    )
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS segment_generations (
            generation_id TEXT PRIMARY KEY,
            video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
            stage_kind TEXT NOT NULL CHECK(stage_kind IN ({segment_stage_kinds})),
            specification_hash TEXT NOT NULL
                REFERENCES stage_specifications(specification_hash),
            source_sha256 TEXT NOT NULL,
            run_id TEXT NOT NULL UNIQUE REFERENCES stage_runs(run_id),
            segment_count INTEGER NOT NULL CHECK(segment_count >= 0),
            completed_at TEXT NOT NULL,
            UNIQUE(video_id, stage_kind, generation_id)
        )
        """
    )
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS active_segment_generations (
            video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
            stage_kind TEXT NOT NULL CHECK(stage_kind IN ({segment_stage_kinds})),
            generation_id TEXT NOT NULL,
            activated_at TEXT NOT NULL,
            PRIMARY KEY(video_id, stage_kind),
            FOREIGN KEY(video_id, stage_kind, generation_id)
                REFERENCES segment_generations(video_id, stage_kind, generation_id)
        )
        """
    )
    columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(segments)").fetchall()
    }
    if "generation_id" not in columns:
        connection.execute(
            """
            ALTER TABLE segments
            ADD COLUMN generation_id TEXT REFERENCES segment_generations(generation_id)
            """
        )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_segment_generations_video_stage
        ON segment_generations(video_id, stage_kind, completed_at)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_segments_generation
        ON segments(generation_id, start, end)
        """
    )
    connection.execute(
        """
        CREATE TRIGGER IF NOT EXISTS segment_generation_run_identity_insert
        BEFORE INSERT ON segment_generations
        WHEN NOT EXISTS (
            SELECT 1
            FROM stage_runs AS runs
            JOIN stage_specifications AS specifications
              ON specifications.specification_hash = runs.specification_hash
            WHERE runs.run_id = NEW.run_id
              AND runs.video_id = NEW.video_id
              AND runs.specification_hash = NEW.specification_hash
              AND runs.source_sha256 = NEW.source_sha256
              AND runs.state = 'running'
              AND specifications.stage_kind = NEW.stage_kind
        )
        BEGIN
            SELECT RAISE(ABORT, 'segment generation run identity mismatch');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER IF NOT EXISTS segment_generations_immutable_update
        BEFORE UPDATE ON segment_generations
        BEGIN
            SELECT RAISE(ABORT, 'segment generation is immutable');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER IF NOT EXISTS segment_generations_immutable_delete
        BEFORE DELETE ON segment_generations
        WHEN EXISTS (SELECT 1 FROM videos WHERE id = OLD.video_id)
        BEGIN
            SELECT RAISE(ABORT, 'segment generation is immutable');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER IF NOT EXISTS generated_segments_validate_insert
        BEFORE INSERT ON segments
        WHEN NEW.generation_id IS NOT NULL AND NOT EXISTS (
            SELECT 1
            FROM segment_generations AS generations
            WHERE generations.generation_id = NEW.generation_id
              AND generations.video_id = NEW.video_id
              AND (
                    (generations.stage_kind = 'scenes' AND NEW.modality = 'scene')
                 OR (generations.stage_kind = 'speech' AND NEW.modality = 'speech')
                 OR (generations.stage_kind = 'ocr' AND NEW.modality = 'ocr')
                 OR (generations.stage_kind = 'objects' AND NEW.modality = 'objects')
              )
        )
        BEGIN
            SELECT RAISE(ABORT, 'generated segment identity mismatch');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER IF NOT EXISTS generated_segments_generation_immutable
        BEFORE UPDATE OF generation_id ON segments
        WHEN NEW.generation_id IS NOT OLD.generation_id
        BEGIN
            SELECT RAISE(ABORT, 'generated segment identity is immutable');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER IF NOT EXISTS generated_segments_immutable_update
        BEFORE UPDATE ON segments
        WHEN OLD.generation_id IS NOT NULL
        BEGIN
            SELECT RAISE(ABORT, 'generated segment is immutable');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER IF NOT EXISTS generated_segments_immutable_delete
        BEFORE DELETE ON segments
        WHEN OLD.generation_id IS NOT NULL
          AND EXISTS (SELECT 1 FROM videos WHERE id = OLD.video_id)
        BEGIN
            SELECT RAISE(ABORT, 'generated segment is immutable');
        END
        """
    )


def _migration_6_add_text_vector_generations(connection: sqlite3.Connection) -> None:
    input_stage_kinds = _sql_values([kind.value for kind in TEXT_VECTOR_INPUT_STAGE_KINDS])
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS text_vector_index_specifications (
            specification_hash TEXT PRIMARY KEY,
            canonical_json TEXT NOT NULL,
            collection_name TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS text_vector_builds (
            generation_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL UNIQUE REFERENCES stage_runs(run_id),
            video_id TEXT NOT NULL UNIQUE REFERENCES videos(id) ON DELETE CASCADE,
            stage_specification_hash TEXT NOT NULL
                REFERENCES stage_specifications(specification_hash),
            source_sha256 TEXT NOT NULL,
            index_specification_hash TEXT NOT NULL
                REFERENCES text_vector_index_specifications(specification_hash),
            collection_name TEXT NOT NULL,
            expected_previous_generation_id TEXT
                REFERENCES text_vector_generations(generation_id),
            input_manifest_sha256 TEXT NOT NULL,
            point_manifest_sha256 TEXT NOT NULL,
            point_count INTEGER NOT NULL CHECK(point_count >= 0),
            reserved_at TEXT NOT NULL,
            heartbeat_at TEXT NOT NULL,
            lease_expires_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS text_vector_build_inputs (
            generation_id TEXT NOT NULL
                REFERENCES text_vector_builds(generation_id) ON DELETE CASCADE,
            stage_kind TEXT NOT NULL CHECK(stage_kind IN ({input_stage_kinds})),
            specification_hash TEXT NOT NULL
                REFERENCES stage_specifications(specification_hash),
            segment_generation_id TEXT REFERENCES segment_generations(generation_id),
            segment_run_id TEXT REFERENCES stage_runs(run_id),
            source_sha256 TEXT,
            segment_count INTEGER NOT NULL CHECK(segment_count >= 0),
            content_manifest_sha256 TEXT NOT NULL,
            PRIMARY KEY(generation_id, stage_kind),
            CHECK(
                (segment_generation_id IS NULL AND segment_run_id IS NULL
                 AND source_sha256 IS NULL AND segment_count = 0)
                OR
                (segment_generation_id IS NOT NULL AND segment_run_id IS NOT NULL
                 AND source_sha256 IS NOT NULL)
            )
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS text_vector_generations (
            generation_id TEXT PRIMARY KEY,
            video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
            specification_hash TEXT NOT NULL
                REFERENCES stage_specifications(specification_hash),
            source_sha256 TEXT NOT NULL,
            run_id TEXT NOT NULL UNIQUE REFERENCES stage_runs(run_id),
            index_specification_hash TEXT NOT NULL
                REFERENCES text_vector_index_specifications(specification_hash),
            collection_name TEXT NOT NULL,
            input_manifest_sha256 TEXT NOT NULL,
            point_manifest_sha256 TEXT NOT NULL,
            vector_manifest_sha256 TEXT NOT NULL,
            point_count INTEGER NOT NULL CHECK(point_count >= 0),
            completed_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS text_vector_generation_inputs (
            generation_id TEXT NOT NULL
                REFERENCES text_vector_generations(generation_id) ON DELETE CASCADE,
            stage_kind TEXT NOT NULL CHECK(stage_kind IN ({input_stage_kinds})),
            specification_hash TEXT NOT NULL
                REFERENCES stage_specifications(specification_hash),
            segment_generation_id TEXT REFERENCES segment_generations(generation_id),
            segment_run_id TEXT REFERENCES stage_runs(run_id),
            source_sha256 TEXT,
            segment_count INTEGER NOT NULL CHECK(segment_count >= 0),
            content_manifest_sha256 TEXT NOT NULL,
            PRIMARY KEY(generation_id, stage_kind),
            CHECK(
                (segment_generation_id IS NULL AND segment_run_id IS NULL
                 AND source_sha256 IS NULL AND segment_count = 0)
                OR
                (segment_generation_id IS NOT NULL AND segment_run_id IS NOT NULL
                 AND source_sha256 IS NOT NULL)
            )
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS active_text_vector_generations (
            video_id TEXT PRIMARY KEY REFERENCES videos(id) ON DELETE CASCADE,
            generation_id TEXT NOT NULL UNIQUE
                REFERENCES text_vector_generations(generation_id),
            activated_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS artifact_gc_jobs (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL UNIQUE,
            artifact_kind TEXT NOT NULL CHECK(artifact_kind = 'text_vectors'),
            generation_id TEXT NOT NULL,
            index_specification_hash TEXT NOT NULL,
            collection_name TEXT NOT NULL,
            reason TEXT NOT NULL,
            state TEXT NOT NULL CHECK(state IN ('pending', 'running', 'complete', 'failed')),
            attempt INTEGER NOT NULL CHECK(attempt >= 0),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(artifact_kind, generation_id, index_specification_hash, reason)
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_text_vector_generations_video_completed
        ON text_vector_generations(video_id, completed_at)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_text_vector_builds_expiry
        ON text_vector_builds(lease_expires_at)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_artifact_gc_jobs_state
        ON artifact_gc_jobs(state, sequence)
        """
    )
    connection.execute(
        """
        CREATE TRIGGER IF NOT EXISTS text_vector_index_specifications_immutable_update
        BEFORE UPDATE ON text_vector_index_specifications
        BEGIN
            SELECT RAISE(ABORT, 'text vector index specification is immutable');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER IF NOT EXISTS text_vector_index_specifications_immutable_delete
        BEFORE DELETE ON text_vector_index_specifications
        BEGIN
            SELECT RAISE(ABORT, 'text vector index specification is immutable');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER IF NOT EXISTS text_vector_build_run_identity_insert
        BEFORE INSERT ON text_vector_builds
        WHEN NOT EXISTS (
            SELECT 1
            FROM stage_runs AS runs
            JOIN stage_specifications AS specifications
              ON specifications.specification_hash = runs.specification_hash
            WHERE runs.run_id = NEW.run_id
              AND runs.video_id = NEW.video_id
              AND runs.specification_hash = NEW.stage_specification_hash
              AND runs.source_sha256 = NEW.source_sha256
              AND runs.state = 'running'
              AND specifications.stage_kind = 'text_vectors'
        )
        BEGIN
            SELECT RAISE(ABORT, 'text vector build run identity mismatch');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER IF NOT EXISTS text_vector_generation_run_identity_insert
        BEFORE INSERT ON text_vector_generations
        WHEN NOT EXISTS (
            SELECT 1
            FROM stage_runs AS runs
            JOIN stage_specifications AS specifications
              ON specifications.specification_hash = runs.specification_hash
            WHERE runs.run_id = NEW.run_id
              AND runs.video_id = NEW.video_id
              AND runs.specification_hash = NEW.specification_hash
              AND runs.source_sha256 = NEW.source_sha256
              AND runs.state = 'running'
              AND specifications.stage_kind = 'text_vectors'
        )
        BEGIN
            SELECT RAISE(ABORT, 'text vector generation run identity mismatch');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER IF NOT EXISTS text_vector_generations_immutable_update
        BEFORE UPDATE ON text_vector_generations
        BEGIN
            SELECT RAISE(ABORT, 'text vector generation is immutable');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER IF NOT EXISTS text_vector_generations_immutable_delete
        BEFORE DELETE ON text_vector_generations
        WHEN EXISTS (SELECT 1 FROM videos WHERE id = OLD.video_id)
        BEGIN
            SELECT RAISE(ABORT, 'text vector generation is immutable');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER IF NOT EXISTS text_vector_generation_inputs_immutable_update
        BEFORE UPDATE ON text_vector_generation_inputs
        BEGIN
            SELECT RAISE(ABORT, 'text vector generation input is immutable');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER IF NOT EXISTS text_vector_generation_inputs_immutable_delete
        BEFORE DELETE ON text_vector_generation_inputs
        WHEN EXISTS (
            SELECT 1 FROM text_vector_generations
            WHERE generation_id = OLD.generation_id
        )
        BEGIN
            SELECT RAISE(ABORT, 'text vector generation input is immutable');
        END
        """
    )


_SCHEMA_MIGRATIONS: tuple[tuple[int, Callable[[sqlite3.Connection], None]], ...] = (
    (1, _migration_1_create_library_schema),
    (2, _migration_2_add_video_display_name),
    (3, _migration_3_add_stage_provenance),
    (4, _migration_4_add_asset_identity),
    (5, _migration_5_add_segment_generations),
    (6, _migration_6_add_text_vector_generations),
)

_STAGE_RUN_SELECT = """
    SELECT
        runs.run_id,
        runs.video_id,
        specifications.stage_kind,
        specifications.canonical_json AS specification_json,
        runs.state,
        runs.specification_hash,
        runs.source_sha256,
        runs.attempt,
        runs.output_generation,
        runs.error_code,
        runs.retry_of_run_id,
        runs.created_at,
        runs.started_at,
        runs.finished_at,
        runs.updated_at
    FROM stage_runs AS runs
    LEFT JOIN stage_specifications AS specifications
      ON specifications.specification_hash = runs.specification_hash
"""

_VIDEO_ASSET_SELECT = """
    SELECT
        videos.id AS video_id,
        videos.asset_id AS linked_asset_id,
        videos.size_bytes AS video_size_bytes,
        assets.asset_id,
        assets.sha256,
        assets.size_bytes,
        assets.created_at
    FROM videos
    LEFT JOIN assets ON assets.asset_id = videos.asset_id
"""

_SEGMENT_GENERATION_SELECT = """
    SELECT
        generation_id,
        video_id,
        stage_kind,
        specification_hash,
        source_sha256,
        run_id,
        segment_count,
        completed_at
    FROM segment_generations
"""

_TEXT_VECTOR_GENERATION_SELECT = """
    SELECT
        generation_id,
        video_id,
        specification_hash,
        source_sha256,
        run_id,
        index_specification_hash,
        collection_name,
        input_manifest_sha256,
        point_manifest_sha256,
        vector_manifest_sha256,
        point_count,
        completed_at
    FROM text_vector_generations
"""

_TEXT_VECTOR_INPUT_COLUMNS = """
    stage_kind,
    specification_hash,
    segment_generation_id,
    segment_run_id,
    source_sha256,
    segment_count,
    content_manifest_sha256
"""

_ARTIFACT_GC_SELECT = """
    SELECT
        job_id,
        artifact_kind,
        generation_id,
        index_specification_hash,
        collection_name,
        reason,
        state,
        attempt,
        created_at,
        updated_at
    FROM artifact_gc_jobs
"""


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text_vector_input_manifest(
    inputs: Iterable[TextVectorGenerationInput],
) -> str:
    return _canonical_sha256(
        [
            {
                "content_manifest_sha256": item.content_manifest_sha256,
                "segment_count": item.segment_count,
                "segment_generation_id": item.segment_generation_id,
                "segment_run_id": item.segment_run_id,
                "source_sha256": item.source_sha256,
                "specification_hash": item.specification_hash,
                "stage_kind": item.stage_kind.value,
            }
            for item in inputs
        ]
    )


def _text_vector_point_manifest(points: Iterable[TextVectorPointSource]) -> str:
    return _canonical_sha256(
        [
            {
                "modality": item.modality,
                "segment_generation_id": item.segment_generation_id,
                "segment_id": item.segment_id,
                "text_sha256": item.text_sha256,
                "video_id": item.video_id,
            }
            for item in points
        ]
    )


def _is_searchable_text_segment(segment: SegmentRecord) -> bool:
    if not segment.text.strip() or segment.modality not in {"speech", "ocr", "objects"}:
        return False
    return segment.modality != "speech" or len(
        re.findall(r"[\w]+", segment.text, flags=re.UNICODE)
    ) >= 2


@dataclass(frozen=True, slots=True)
class VideoRecord:
    id: str
    original_name: str
    display_name: str | None
    stored_name: str
    media_path: str
    size_bytes: int
    asset_id: str | None
    status: str
    progress: float
    stage: str
    duration: float | None
    width: int | None
    height: int | None
    fps: float | None
    thumbnail_path: str | None
    error: str | None
    created_at: str
    updated_at: str

    @property
    def name(self) -> str:
        return self.display_name or self.original_name


@dataclass(frozen=True, slots=True)
class SegmentRecord:
    id: str
    video_id: str
    start: float
    end: float
    modality: str
    text: str
    confidence: float
    metadata: dict[str, Any]
    thumbnail_path: str | None


@dataclass(frozen=True, slots=True)
class RepositoryAssetRecord:
    id: str
    sha256: str
    byte_size: int
    duration_seconds: float
    video_id: str

    def __post_init__(self) -> None:
        if self.id != asset_id_for_sha256(self.sha256):
            raise AssetIdentityError("repository asset id is corrupt")
        if (
            isinstance(self.byte_size, bool)
            or not isinstance(self.byte_size, int)
            or self.byte_size <= 0
        ):
            raise AssetIdentityError("repository asset size is corrupt")
        if (
            isinstance(self.duration_seconds, bool)
            or not isinstance(self.duration_seconds, Real)
            or not math.isfinite(float(self.duration_seconds))
            or self.duration_seconds <= 0
        ):
            raise AssetIdentityError("repository asset duration is corrupt")
        if type(self.video_id) is not str or not self.video_id:
            raise AssetIdentityError("repository asset video link is corrupt")


class Repository:
    _VIDEO_UPDATE_FIELDS = {
        "display_name",
        "status",
        "progress",
        "stage",
        "duration",
        "width",
        "height",
        "fps",
        "thumbnail_path",
        "error",
    }

    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with _DATABASE_INITIALIZE_LOCK:
            with self._connect() as connection:
                current_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
                _ensure_supported_schema_version(current_version)
                _enable_wal(connection)
                connection.execute("BEGIN IMMEDIATE")
                try:
                    current_version = int(
                        connection.execute("PRAGMA user_version").fetchone()[0]
                    )
                    _ensure_supported_schema_version(current_version)
                    for version, migration in _SCHEMA_MIGRATIONS:
                        if version <= current_version:
                            continue
                        migration(connection)
                        connection.execute(f"PRAGMA user_version = {version}")
                    connection.commit()
                except Exception:
                    connection.rollback()
                    raise

    def schema_version(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("PRAGMA user_version").fetchone()[0])

    @staticmethod
    def _insert_video(
        connection: sqlite3.Connection,
        *,
        video_id: str,
        original_name: str,
        stored_name: str,
        media_path: str,
        size_bytes: int,
        asset_id: str | None,
        timestamp: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO videos (
                id, original_name, stored_name, media_path, size_bytes,
                asset_id, status, progress, stage, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                video_id,
                original_name,
                stored_name,
                media_path,
                size_bytes,
                asset_id,
                "queued",
                0.0,
                "queued",
                timestamp,
                timestamp,
            ),
        )

    @staticmethod
    def _persist_asset(
        connection: sqlite3.Connection,
        asset: AssetRecord,
    ) -> AssetRecord:
        connection.execute(
            """
            INSERT OR IGNORE INTO assets (
                asset_id, sha256, size_bytes, created_at
            ) VALUES (?, ?, ?, ?)
            """,
            (asset.asset_id, asset.sha256, asset.size_bytes, asset.created_at),
        )
        row = connection.execute(
            """
            SELECT asset_id, sha256, size_bytes, created_at
            FROM assets
            WHERE asset_id = ?
            """,
            (asset.asset_id,),
        ).fetchone()
        if row is None:
            raise AssetIdentityError("asset identity was not persisted")
        try:
            persisted = AssetRecord(**dict(row))
        except (TypeError, ValueError) as error:
            raise AssetIdentityError("persisted asset identity is corrupt") from error
        if persisted.sha256 != asset.sha256 or persisted.size_bytes != asset.size_bytes:
            raise AssetIdentityError("asset digest already exists with a different size")
        return persisted

    @staticmethod
    def _asset_from_video_row(row: sqlite3.Row) -> AssetRecord | None:
        linked_asset_id = row["linked_asset_id"]
        if linked_asset_id is None:
            return None
        if row["asset_id"] is None:
            raise AssetIdentityError("persisted video asset link is corrupt")
        try:
            asset = AssetRecord(
                asset_id=row["asset_id"],
                sha256=row["sha256"],
                size_bytes=row["size_bytes"],
                created_at=row["created_at"],
            )
        except (TypeError, ValueError) as error:
            raise AssetIdentityError("persisted asset identity is corrupt") from error
        if linked_asset_id != asset.asset_id or row["video_size_bytes"] != asset.size_bytes:
            raise AssetIdentityError("persisted video asset link is corrupt")
        return asset

    @classmethod
    def _get_video_asset(
        cls,
        connection: sqlite3.Connection,
        video_id: str,
    ) -> AssetRecord | None:
        row = connection.execute(
            f"{_VIDEO_ASSET_SELECT} WHERE videos.id = ?",
            (video_id,),
        ).fetchone()
        if row is None:
            raise KeyError(video_id)
        return cls._asset_from_video_row(row)

    def create_video(
        self,
        *,
        video_id: str,
        original_name: str,
        stored_name: str,
        media_path: str,
        size_bytes: int,
    ) -> VideoRecord:
        timestamp = _now()
        with self._connect() as connection:
            self._insert_video(
                connection,
                video_id=video_id,
                original_name=original_name,
                stored_name=stored_name,
                media_path=media_path,
                size_bytes=size_bytes,
                asset_id=None,
                timestamp=timestamp,
            )
        record = self.get_video(video_id)
        if record is None:
            raise RuntimeError("video was not persisted")
        return record

    def create_video_with_asset(
        self,
        *,
        video_id: str,
        original_name: str,
        stored_name: str,
        media_path: str,
        size_bytes: int,
        source_sha256: str,
    ) -> VideoRecord:
        timestamp = _now()
        asset = AssetRecord.from_digest(
            sha256=source_sha256,
            size_bytes=size_bytes,
            created_at=timestamp,
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            persisted_asset = self._persist_asset(connection, asset)
            self._insert_video(
                connection,
                video_id=video_id,
                original_name=original_name,
                stored_name=stored_name,
                media_path=media_path,
                size_bytes=size_bytes,
                asset_id=persisted_asset.asset_id,
                timestamp=timestamp,
            )
            row = connection.execute(
                "SELECT * FROM videos WHERE id = ?",
                (video_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError("video with asset was not persisted")
            try:
                record = VideoRecord(**dict(row))
            except (TypeError, ValueError) as error:
                raise RuntimeError("persisted video with asset is corrupt") from error
        return record

    def get_video_asset(self, video_id: str) -> AssetRecord | None:
        with self._connect() as connection:
            try:
                return self._get_video_asset(connection, video_id)
            except KeyError:
                return None

    def find_assets_by_sha256(self, digest: str) -> tuple[RepositoryAssetRecord, ...]:
        asset_id_for_sha256(digest)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    assets.asset_id,
                    assets.sha256,
                    assets.size_bytes AS asset_size_bytes,
                    assets.created_at AS asset_created_at,
                    videos.id AS video_id,
                    videos.size_bytes AS video_size_bytes,
                    videos.duration AS duration_seconds
                FROM assets
                JOIN videos ON videos.asset_id = assets.asset_id
                WHERE assets.sha256 = ? AND videos.duration IS NOT NULL
                ORDER BY videos.created_at, videos.id
                """,
                (digest,),
            ).fetchall()
        matches: list[RepositoryAssetRecord] = []
        for row in rows:
            try:
                asset = AssetRecord(
                    asset_id=row["asset_id"],
                    sha256=row["sha256"],
                    size_bytes=row["asset_size_bytes"],
                    created_at=row["asset_created_at"],
                )
                if row["video_size_bytes"] != asset.size_bytes:
                    raise AssetIdentityError("repository asset video link is corrupt")
                matches.append(
                    RepositoryAssetRecord(
                        id=asset.asset_id,
                        sha256=asset.sha256,
                        byte_size=asset.size_bytes,
                        duration_seconds=row["duration_seconds"],
                        video_id=row["video_id"],
                    )
                )
            except (TypeError, ValueError) as error:
                raise AssetIdentityError("persisted repository asset binding is corrupt") from error
        return tuple(matches)

    def _link_video_asset(
        self,
        *,
        expected_video: VideoRecord,
        source_sha256: str,
    ) -> AssetRecord:
        asset = AssetRecord.from_digest(
            sha256=source_sha256,
            size_bytes=expected_video.size_bytes,
            created_at=_now(),
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT stored_name, media_path, size_bytes, created_at
                FROM videos
                WHERE id = ?
                """,
                (expected_video.id,),
            ).fetchone()
            if row is None:
                raise KeyError(expected_video.id)
            if (
                row["stored_name"] != expected_video.stored_name
                or row["media_path"] != expected_video.media_path
                or row["size_bytes"] != expected_video.size_bytes
                or row["created_at"] != expected_video.created_at
            ):
                raise AssetIdentityError("video record changed while hashing asset identity")
            existing = self._get_video_asset(connection, expected_video.id)
            if existing is not None:
                if (
                    existing.sha256 != source_sha256
                    or existing.size_bytes != expected_video.size_bytes
                ):
                    raise AssetIdentityError("video asset identity is immutable")
                return existing
            persisted = self._persist_asset(connection, asset)
            cursor = connection.execute(
                """
                UPDATE videos
                SET asset_id = ?
                WHERE id = ? AND asset_id IS NULL
                """,
                (persisted.asset_id, expected_video.id),
            )
            if cursor.rowcount != 1:
                raise AssetIdentityError("video asset identity changed concurrently")
        return persisted

    def ensure_asset_identity(
        self,
        video_id: str,
        *,
        media_root: Path,
    ) -> AssetRecord:
        existing = self.get_video_asset(video_id)
        if existing is not None:
            return existing
        with _asset_identity_lock(self.database_path, video_id):
            existing = self.get_video_asset(video_id)
            if existing is not None:
                return existing
            video = self.get_video(video_id)
            if video is None:
                raise KeyError(video_id)
            source_sha256 = self._hash_video_source(video, media_root=media_root)
            return self._link_video_asset(
                expected_video=video,
                source_sha256=source_sha256,
            )

    @staticmethod
    def _hash_video_source(
        video: VideoRecord,
        *,
        media_root: Path,
    ) -> str:
        try:
            resolved_root = Path(media_root).resolve(strict=True)
            media_path = Path(video.media_path)
            resolved_media = media_path.resolve(strict=True)
            expected_media = (resolved_root / video.stored_name).resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise AssetIdentityError("asset media must be a managed regular file") from error
        if (
            not resolved_root.is_dir()
            or Path(video.stored_name).name != video.stored_name
            or media_path.is_symlink()
            or resolved_media != expected_media
            or resolved_media.parent != resolved_root
        ):
            raise AssetIdentityError("asset media must be a managed regular file")
        return _hash_managed_regular_file(
            resolved_media,
            expected_size=video.size_bytes,
        )

    def verify_asset_identity(
        self,
        video_id: str,
        *,
        media_root: Path,
    ) -> AssetRecord:
        """Hash managed media once and prove it still matches its immutable Asset."""
        with _asset_identity_lock(self.database_path, video_id):
            video = self.get_video(video_id)
            if video is None:
                raise KeyError(video_id)
            source_sha256 = self._hash_video_source(video, media_root=media_root)
            existing = self.get_video_asset(video_id)
            if existing is None:
                return self._link_video_asset(
                    expected_video=video,
                    source_sha256=source_sha256,
                )
            if (
                existing.sha256 != source_sha256
                or existing.size_bytes != video.size_bytes
            ):
                raise AssetIdentityError(
                    "managed media content does not match its immutable asset identity"
                )
            return existing

    def update_video(self, video_id: str, **changes: object) -> VideoRecord:
        unknown = set(changes) - self._VIDEO_UPDATE_FIELDS
        if unknown:
            raise ValueError(f"unsupported video fields: {sorted(unknown)}")
        if not changes:
            record = self.get_video(video_id)
            if record is None:
                raise KeyError(video_id)
            return record

        assignments = [f"{field} = ?" for field in changes]
        assignments.append("updated_at = ?")
        values = [*changes.values(), _now(), video_id]
        with self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE videos SET {', '.join(assignments)} WHERE id = ?",
                values,
            )
            if cursor.rowcount != 1:
                raise KeyError(video_id)
        record = self.get_video(video_id)
        if record is None:
            raise KeyError(video_id)
        return record

    def get_video(self, video_id: str) -> VideoRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
        return VideoRecord(**dict(row)) if row else None

    def list_videos(self) -> list[VideoRecord]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM videos ORDER BY created_at DESC").fetchall()
        return [VideoRecord(**dict(row)) for row in rows]

    @staticmethod
    def _stage_run_from_row(row: sqlite3.Row) -> StageRun:
        specification = StageSpecification.from_canonical_json(row["specification_json"])
        if (
            specification.specification_hash != row["specification_hash"]
            or specification.kind.value != row["stage_kind"]
        ):
            raise ValueError("persisted stage specification failed integrity validation")
        return StageRun(
            run_id=row["run_id"],
            video_id=row["video_id"],
            stage_kind=row["stage_kind"],
            state=row["state"],
            specification_hash=row["specification_hash"],
            source_sha256=row["source_sha256"],
            attempt=row["attempt"],
            output_generation=row["output_generation"],
            error_code=row["error_code"],
            retry_of_run_id=row["retry_of_run_id"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _validate_stage_specification_links(
        connection: sqlite3.Connection,
        video_id: str,
    ) -> None:
        dangling = connection.execute(
            """
            SELECT runs.run_id
            FROM stage_runs AS runs
            LEFT JOIN stage_specifications AS specifications
              ON specifications.specification_hash = runs.specification_hash
            WHERE runs.video_id = ? AND specifications.specification_hash IS NULL
            LIMIT 1
            """,
            (video_id,),
        ).fetchone()
        if dangling is not None:
            raise ValueError("persisted stage specification link is missing")

    @classmethod
    def _get_stage_run_unchecked(
        cls,
        connection: sqlite3.Connection,
        run_id: str,
    ) -> StageRun | None:
        row = connection.execute(
            f"{_STAGE_RUN_SELECT} WHERE runs.run_id = ?",
            (run_id,),
        ).fetchone()
        return cls._stage_run_from_row(row) if row else None

    @classmethod
    def _validate_retry_lineage(
        cls,
        connection: sqlite3.Connection,
        stage_run: StageRun,
    ) -> None:
        current = stage_run
        visited = {current.run_id}
        while current.retry_of_run_id is not None:
            try:
                parent = cls._get_stage_run_unchecked(connection, current.retry_of_run_id)
            except ValueError as error:
                raise ValueError("invalid stage retry lineage: parent run is corrupt") from error
            if parent is None:
                raise ValueError("invalid stage retry lineage: parent run is missing")
            if parent.run_id in visited:
                raise ValueError("invalid stage retry lineage: cycle detected")
            if (
                parent.video_id != current.video_id
                or parent.specification_hash != current.specification_hash
                or parent.source_sha256 != current.source_sha256
            ):
                raise ValueError("invalid stage retry lineage: work identity changed")
            if parent.attempt >= current.attempt:
                raise ValueError("invalid stage retry lineage: attempts are not monotonic")
            if parent.state not in {StageState.FAILED, StageState.CANCELLED}:
                raise ValueError("invalid stage retry lineage: parent is not retryable")
            visited.add(parent.run_id)
            current = parent

    @classmethod
    def _get_stage_run(
        cls,
        connection: sqlite3.Connection,
        run_id: str,
    ) -> StageRun | None:
        stage_run = cls._get_stage_run_unchecked(connection, run_id)
        if stage_run is not None:
            cls._validate_retry_lineage(connection, stage_run)
            cls._validate_stage_source_identity(connection, stage_run)
        return stage_run

    @classmethod
    def _validate_stage_source_identity(
        cls,
        connection: sqlite3.Connection,
        stage_run: StageRun,
    ) -> None:
        try:
            asset = cls._get_video_asset(connection, stage_run.video_id)
        except KeyError as error:
            raise AssetIdentityError("stage video asset identity is unavailable") from error
        if asset is None:
            raise AssetIdentityError("stage video asset identity is unavailable")
        if asset.sha256 != stage_run.source_sha256:
            raise AssetIdentityError("persisted stage source identity is corrupt")

    def get_stage_specification(
        self,
        specification_hash: str,
    ) -> StageSpecification | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT stage_kind, canonical_json
                FROM stage_specifications
                WHERE specification_hash = ?
                """,
                (specification_hash,),
            ).fetchone()
        if row is None:
            return None
        specification = StageSpecification.from_canonical_json(row["canonical_json"])
        if (
            specification.specification_hash != specification_hash
            or specification.kind.value != row["stage_kind"]
        ):
            raise ValueError("persisted stage specification failed integrity validation")
        return specification

    def create_stage_run(
        self,
        *,
        video_id: str,
        specification: StageSpecification,
        run_id: str | None = None,
        retry_of_run_id: str | None = None,
    ) -> StageRun:
        if not isinstance(specification, StageSpecification):
            raise ValueError("stage specification must be validated")
        resolved_run_id = uuid4().hex if run_id is None else run_id
        timestamp = _now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            asset = self._get_video_asset(connection, video_id)
            if asset is None:
                raise AssetIdentityError("video asset identity is unavailable")
            source_sha256 = asset.sha256
            connection.execute(
                """
                INSERT OR IGNORE INTO stage_specifications (
                    specification_hash, stage_kind, canonical_json, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    specification.specification_hash,
                    specification.kind.value,
                    specification.canonical_json,
                    timestamp,
                ),
            )
            persisted_specification = connection.execute(
                """
                SELECT stage_kind, canonical_json
                FROM stage_specifications
                WHERE specification_hash = ?
                """,
                (specification.specification_hash,),
            ).fetchone()
            if persisted_specification is None or (
                persisted_specification["stage_kind"] != specification.kind.value
                or persisted_specification["canonical_json"] != specification.canonical_json
            ):
                raise ValueError("stage specification hash collision or corrupted registry")

            retry: StageRun | None = None
            if retry_of_run_id is not None:
                retry = self._get_stage_run(connection, retry_of_run_id)
                if retry is None:
                    raise KeyError(retry_of_run_id)
                if retry.state not in {StageState.FAILED, StageState.CANCELLED}:
                    raise ValueError("only a failed or cancelled stage run can be retried")
                if (
                    retry.video_id != video_id
                    or retry.specification_hash != specification.specification_hash
                    or retry.source_sha256 != source_sha256
                ):
                    raise ValueError("stage retry must preserve source and specification identity")

            attempt = int(
                connection.execute(
                    """
                    SELECT COALESCE(MAX(attempt), 0) + 1
                    FROM stage_runs
                    WHERE video_id = ?
                      AND specification_hash = ?
                      AND source_sha256 = ?
                    """,
                    (video_id, specification.specification_hash, source_sha256),
                ).fetchone()[0]
            )
            candidate = StageRun(
                run_id=resolved_run_id,
                video_id=video_id,
                stage_kind=specification.kind,
                state=StageState.QUEUED,
                specification_hash=specification.specification_hash,
                source_sha256=source_sha256,
                attempt=attempt,
                output_generation=None,
                error_code=None,
                retry_of_run_id=retry.run_id if retry else None,
                created_at=timestamp,
                started_at=None,
                finished_at=None,
                updated_at=timestamp,
            )
            try:
                connection.execute(
                    """
                    INSERT INTO stage_runs (
                        run_id, video_id, specification_hash, source_sha256,
                        attempt, state, output_generation, error_code,
                        retry_of_run_id, created_at, started_at, finished_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        candidate.run_id,
                        candidate.video_id,
                        candidate.specification_hash,
                        candidate.source_sha256,
                        candidate.attempt,
                        candidate.state.value,
                        candidate.output_generation,
                        candidate.error_code,
                        candidate.retry_of_run_id,
                        candidate.created_at,
                        candidate.started_at,
                        candidate.finished_at,
                        candidate.updated_at,
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise ValueError("stage run identity already exists or is invalid") from error
        return candidate

    def get_stage_run(self, run_id: str) -> StageRun | None:
        with self._connect() as connection:
            return self._get_stage_run(connection, run_id)

    def list_stage_runs(
        self,
        video_id: str,
        stage_kind: StageKind | None = None,
    ) -> list[StageRun]:
        parameters: tuple[object, ...]
        query = f"{_STAGE_RUN_SELECT} WHERE runs.video_id = ?"
        if stage_kind is None:
            parameters = (video_id,)
        else:
            try:
                resolved_kind = StageKind(stage_kind)
            except (TypeError, ValueError) as error:
                raise ValueError("unsupported stage kind") from error
            query += " AND specifications.stage_kind = ?"
            parameters = (video_id, resolved_kind.value)
        query += " ORDER BY runs.sequence"
        with self._connect() as connection:
            self._validate_stage_specification_links(connection, video_id)
            rows = connection.execute(query, parameters).fetchall()
            stage_runs = [self._stage_run_from_row(row) for row in rows]
            for stage_run in stage_runs:
                self._validate_retry_lineage(connection, stage_run)
                self._validate_stage_source_identity(connection, stage_run)
        return stage_runs

    def get_latest_stage_run(
        self,
        video_id: str,
        stage_kind: StageKind,
    ) -> StageRun | None:
        try:
            resolved_kind = StageKind(stage_kind)
        except (TypeError, ValueError) as error:
            raise ValueError("unsupported stage kind") from error
        with self._connect() as connection:
            self._validate_stage_specification_links(connection, video_id)
            row = connection.execute(
                f"""
                {_STAGE_RUN_SELECT}
                WHERE runs.video_id = ? AND specifications.stage_kind = ?
                ORDER BY runs.sequence DESC
                LIMIT 1
                """,
                (video_id, resolved_kind.value),
            ).fetchone()
            stage_run = self._stage_run_from_row(row) if row else None
            if stage_run is not None:
                self._validate_retry_lineage(connection, stage_run)
                self._validate_stage_source_identity(connection, stage_run)
        return stage_run

    def transition_stage_run(
        self,
        run_id: str,
        state: StageState,
        *,
        output_generation: str | None = None,
        error_code: str | None = None,
    ) -> StageRun:
        try:
            target_state = StageState(state)
        except (TypeError, ValueError) as error:
            raise ValueError("unsupported stage state") from error
        timestamp = _now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._get_stage_run(connection, run_id)
            if current is None:
                raise KeyError(run_id)
            validate_stage_transition(current.state, target_state)
            if (
                target_state is StageState.COMPLETE
                and current.stage_kind in {*SEGMENT_STAGE_KINDS, StageKind.TEXT_VECTORS}
            ):
                if current.stage_kind is StageKind.TEXT_VECTORS:
                    raise ValueError(
                        "text vector stages must complete through an atomic text vector generation commit"
                    )
                raise ValueError(
                    "segment stages must complete through an atomic segment generation commit"
                )

            started_at = current.started_at
            finished_at = current.finished_at
            resolved_generation = current.output_generation
            resolved_error_code = current.error_code
            if target_state is StageState.RUNNING:
                if output_generation is not None or error_code is not None:
                    raise ValueError("running transition cannot contain outcome details")
                started_at = timestamp
            elif target_state is StageState.COMPLETE:
                resolved_generation = output_generation
                resolved_error_code = error_code
                finished_at = timestamp
            elif target_state is StageState.FAILED:
                resolved_generation = output_generation
                resolved_error_code = error_code
                finished_at = timestamp
            elif target_state in {
                StageState.CANCELLED,
                StageState.NOT_CONFIGURED,
            }:
                if output_generation is not None or error_code is not None:
                    raise ValueError("terminal transition cannot contain outcome details")
                resolved_generation = None
                resolved_error_code = None
                finished_at = timestamp
            elif target_state is StageState.STALE:
                if output_generation is not None or error_code is not None:
                    raise ValueError("stale transition cannot replace outcome identity")

            candidate = StageRun(
                run_id=current.run_id,
                video_id=current.video_id,
                stage_kind=current.stage_kind,
                state=target_state,
                specification_hash=current.specification_hash,
                source_sha256=current.source_sha256,
                attempt=current.attempt,
                output_generation=resolved_generation,
                error_code=resolved_error_code,
                retry_of_run_id=current.retry_of_run_id,
                created_at=current.created_at,
                started_at=started_at,
                finished_at=finished_at,
                updated_at=timestamp,
            )
            cursor = connection.execute(
                """
                UPDATE stage_runs
                SET state = ?, output_generation = ?, error_code = ?,
                    started_at = ?, finished_at = ?, updated_at = ?
                WHERE run_id = ? AND state = ?
                """,
                (
                    candidate.state.value,
                    candidate.output_generation,
                    candidate.error_code,
                    candidate.started_at,
                    candidate.finished_at,
                    candidate.updated_at,
                    candidate.run_id,
                    current.state.value,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("stage run changed during state transition")
        return candidate

    @staticmethod
    def _resolve_segment_stage_kind(stage_kind: StageKind) -> StageKind:
        try:
            resolved = StageKind(stage_kind)
        except (TypeError, ValueError) as error:
            raise ValueError("unsupported segment generation stage") from error
        if resolved not in SEGMENT_STAGE_KINDS:
            raise ValueError("stage does not produce SQLite segments")
        return resolved

    @staticmethod
    def _validate_generated_thumbnail_path(thumbnail_path: str | None) -> None:
        if thumbnail_path is None:
            return
        if type(thumbnail_path) is not str or not thumbnail_path or "\x00" in thumbnail_path:
            raise ValueError("generated segment thumbnail path must be a safe path")
        path = Path(thumbnail_path)
        if path.name == "" or ".." in path.parts or str(path) != thumbnail_path:
            raise ValueError("generated segment thumbnail path must be a safe path")

    @classmethod
    def _validated_generation_segment(
        cls,
        candidate: object,
        *,
        video_id: str,
        modality: str,
    ) -> tuple[SegmentRecord, str]:
        if not isinstance(candidate, SegmentRecord):
            raise ValueError("generated segments must use the validated segment contract")
        validate_artifact_identifier(candidate.id, field_name="generated segment id")
        record = cls._validated_segment(
            segment_id=candidate.id,
            video_id=candidate.video_id,
            start=candidate.start,
            end=candidate.end,
            modality=candidate.modality,
            text=candidate.text,
            confidence=candidate.confidence,
            metadata=candidate.metadata,
            thumbnail_path=candidate.thumbnail_path,
        )
        if record.video_id != video_id:
            raise ValueError("generated segment video does not match its stage run")
        if record.modality != modality:
            raise ValueError("generated segment modality does not match its stage")
        cls._validate_generated_thumbnail_path(record.thumbnail_path)
        try:
            metadata_json = json.dumps(
                record.metadata,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("generated segment metadata must be finite JSON") from error
        return record, metadata_json

    @classmethod
    def _get_segment_generation_with_run(
        cls,
        connection: sqlite3.Connection,
        generation_id: str,
    ) -> tuple[SegmentGeneration, StageRun] | None:
        row = connection.execute(
            f"{_SEGMENT_GENERATION_SELECT} WHERE generation_id = ?",
            (generation_id,),
        ).fetchone()
        if row is None:
            return None
        try:
            generation = SegmentGeneration(**dict(row))
        except (TypeError, ValueError) as error:
            raise ValueError("persisted segment generation is corrupt") from error
        try:
            stage_run = cls._get_stage_run(connection, generation.run_id)
        except (TypeError, ValueError) as error:
            raise ValueError("persisted segment generation run is corrupt") from error
        if stage_run is None:
            raise ValueError("persisted segment generation run is missing")
        if (
            stage_run.video_id != generation.video_id
            or stage_run.stage_kind is not generation.stage_kind
            or stage_run.specification_hash != generation.specification_hash
            or stage_run.source_sha256 != generation.source_sha256
            or stage_run.output_generation != generation.generation_id
            or stage_run.state not in {StageState.COMPLETE, StageState.STALE}
            or stage_run.finished_at != generation.completed_at
        ):
            raise ValueError("persisted segment generation identity is corrupt")
        actual_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM segments WHERE generation_id = ?",
                (generation.generation_id,),
            ).fetchone()[0]
        )
        if actual_count != generation.segment_count:
            raise ValueError("persisted segment generation count is corrupt")
        expected_modality = SEGMENT_STAGE_MODALITIES[generation.stage_kind]
        invalid_segment = connection.execute(
            """
            SELECT id
            FROM segments
            WHERE generation_id = ?
              AND (video_id <> ? OR modality <> ?)
            LIMIT 1
            """,
            (generation.generation_id, generation.video_id, expected_modality),
        ).fetchone()
        if invalid_segment is not None:
            raise ValueError("persisted segment generation modality is corrupt")
        return generation, stage_run

    @classmethod
    def _get_active_segment_generation(
        cls,
        connection: sqlite3.Connection,
        video_id: str,
        stage_kind: StageKind,
    ) -> SegmentGeneration | None:
        pointer = connection.execute(
            """
            SELECT generation_id, activated_at
            FROM active_segment_generations
            WHERE video_id = ? AND stage_kind = ?
            """,
            (video_id, stage_kind.value),
        ).fetchone()
        if pointer is None:
            return None
        resolved = cls._get_segment_generation_with_run(
            connection,
            pointer["generation_id"],
        )
        if resolved is None:
            raise ValueError("active segment generation pointer is dangling")
        generation, stage_run = resolved
        if (
            generation.video_id != video_id
            or generation.stage_kind is not stage_kind
            or stage_run.state is not StageState.COMPLETE
            or pointer["activated_at"] != generation.completed_at
        ):
            raise ValueError("active segment generation pointer is corrupt")
        return generation

    def commit_segment_generation(
        self,
        run_id: str,
        *,
        segments: Iterable[SegmentRecord],
        generation_id: str | None = None,
        video_thumbnail_path: str | None | object = _VIDEO_THUMBNAIL_UNCHANGED,
    ) -> SegmentGeneration:
        try:
            candidates = tuple(segments)
        except TypeError as error:
            raise ValueError("generated segments must be a finite collection") from error
        resolved_generation_id = uuid4().hex if generation_id is None else generation_id
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            stage_run = self._get_stage_run(connection, run_id)
            if stage_run is None:
                raise KeyError(run_id)
            if stage_run.state is not StageState.RUNNING:
                raise ValueError("only a running stage run can publish a segment generation")
            stage_kind = self._resolve_segment_stage_kind(stage_run.stage_kind)
            modality = SEGMENT_STAGE_MODALITIES[stage_kind]
            validated_segments: list[tuple[SegmentRecord, str]] = []
            segment_ids: set[str] = set()
            for candidate in candidates:
                record, metadata_json = self._validated_generation_segment(
                    candidate,
                    video_id=stage_run.video_id,
                    modality=modality,
                )
                if record.id in segment_ids:
                    raise ValueError("generated segment ids must be unique")
                segment_ids.add(record.id)
                validated_segments.append((record, metadata_json))
            updates_video_thumbnail = (
                video_thumbnail_path is not _VIDEO_THUMBNAIL_UNCHANGED
            )
            if updates_video_thumbnail:
                if stage_kind is not StageKind.SCENES:
                    raise ValueError("only a scene generation can update the video thumbnail")
                if video_thumbnail_path is not None:
                    self._validate_generated_thumbnail_path(video_thumbnail_path)
                    if not any(
                        record.thumbnail_path == video_thumbnail_path
                        for record, _metadata_json in validated_segments
                    ):
                        raise ValueError(
                            "video thumbnail must belong to the published scene generation"
                        )
            completed_at = _now()
            generation = SegmentGeneration(
                generation_id=resolved_generation_id,
                video_id=stage_run.video_id,
                stage_kind=stage_kind,
                specification_hash=stage_run.specification_hash,
                source_sha256=stage_run.source_sha256,
                run_id=stage_run.run_id,
                segment_count=len(validated_segments),
                completed_at=completed_at,
            )
            connection.execute(
                """
                INSERT INTO segment_generations (
                    generation_id, video_id, stage_kind, specification_hash,
                    source_sha256, run_id, segment_count, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    generation.generation_id,
                    generation.video_id,
                    generation.stage_kind.value,
                    generation.specification_hash,
                    generation.source_sha256,
                    generation.run_id,
                    generation.segment_count,
                    generation.completed_at,
                ),
            )
            for record, metadata_json in validated_segments:
                connection.execute(
                    """
                    INSERT INTO segments (
                        id, video_id, start, end, modality, text,
                        confidence, metadata_json, thumbnail_path, generation_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.id,
                        record.video_id,
                        record.start,
                        record.end,
                        record.modality,
                        record.text,
                        record.confidence,
                        metadata_json,
                        record.thumbnail_path,
                        generation.generation_id,
                    ),
                )
            connection.execute(
                """
                INSERT INTO active_segment_generations (
                    video_id, stage_kind, generation_id, activated_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(video_id, stage_kind) DO UPDATE SET
                    generation_id = excluded.generation_id,
                    activated_at = excluded.activated_at
                """,
                (
                    generation.video_id,
                    generation.stage_kind.value,
                    generation.generation_id,
                    generation.completed_at,
                ),
            )
            completed_run = StageRun(
                run_id=stage_run.run_id,
                video_id=stage_run.video_id,
                stage_kind=stage_run.stage_kind,
                state=StageState.COMPLETE,
                specification_hash=stage_run.specification_hash,
                source_sha256=stage_run.source_sha256,
                attempt=stage_run.attempt,
                output_generation=generation.generation_id,
                error_code=None,
                retry_of_run_id=stage_run.retry_of_run_id,
                created_at=stage_run.created_at,
                started_at=stage_run.started_at,
                finished_at=generation.completed_at,
                updated_at=generation.completed_at,
            )
            cursor = connection.execute(
                """
                UPDATE stage_runs
                SET state = ?, output_generation = ?, error_code = NULL,
                    finished_at = ?, updated_at = ?
                WHERE run_id = ? AND state = ?
                """,
                (
                    completed_run.state.value,
                    completed_run.output_generation,
                    completed_run.finished_at,
                    completed_run.updated_at,
                    completed_run.run_id,
                    StageState.RUNNING.value,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("stage run changed during generation activation")
            if updates_video_thumbnail:
                cursor = connection.execute(
                    """
                    UPDATE videos
                    SET thumbnail_path = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        video_thumbnail_path,
                        generation.completed_at,
                        generation.video_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError(
                        "scene generation video changed during activation"
                    )
        return generation

    def get_segment_generation(self, generation_id: str) -> SegmentGeneration | None:
        with self._connect() as connection:
            resolved = self._get_segment_generation_with_run(connection, generation_id)
        return resolved[0] if resolved is not None else None

    def get_active_segment_generation(
        self,
        video_id: str,
        stage_kind: StageKind,
    ) -> SegmentGeneration | None:
        resolved_kind = self._resolve_segment_stage_kind(stage_kind)
        with self._connect() as connection:
            return self._get_active_segment_generation(connection, video_id, resolved_kind)

    def is_active_segment_generation_current(
        self,
        video_id: str,
        specification: StageSpecification,
    ) -> bool:
        if not isinstance(specification, StageSpecification):
            raise ValueError("stage specification must be validated")
        stage_kind = self._resolve_segment_stage_kind(specification.kind)
        with self._connect() as connection:
            generation = self._get_active_segment_generation(
                connection,
                video_id,
                stage_kind,
            )
        return (
            generation is not None
            and generation.specification_hash == specification.specification_hash
        )

    def list_active_segments(
        self,
        video_id: str,
        stage_kind: StageKind,
    ) -> list[SegmentRecord]:
        resolved_kind = self._resolve_segment_stage_kind(stage_kind)
        with self._connect() as connection:
            generation = self._get_active_segment_generation(
                connection,
                video_id,
                resolved_kind,
            )
            if generation is None:
                return []
            return self._list_generation_segments(connection, generation)

    @classmethod
    def _list_generation_segments(
        cls,
        connection: sqlite3.Connection,
        generation: SegmentGeneration,
    ) -> list[SegmentRecord]:
        rows = connection.execute(
            """
            SELECT *
            FROM segments
            WHERE generation_id = ?
            ORDER BY start, end, id
            """,
            (generation.generation_id,),
        ).fetchall()
        segments: list[SegmentRecord] = []
        expected_modality = SEGMENT_STAGE_MODALITIES[generation.stage_kind]
        for row in rows:
            try:
                metadata = json.loads(
                    row["metadata_json"],
                    parse_constant=_reject_non_finite_json,
                )
                record = cls._validated_segment(
                    segment_id=row["id"],
                    video_id=row["video_id"],
                    start=row["start"],
                    end=row["end"],
                    modality=row["modality"],
                    text=row["text"],
                    confidence=row["confidence"],
                    metadata=metadata,
                    thumbnail_path=row["thumbnail_path"],
                )
                validated, _ = cls._validated_generation_segment(
                    record,
                    video_id=generation.video_id,
                    modality=expected_modality,
                )
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError("persisted active segment generation is corrupt") from error
            segments.append(validated)
        return segments

    def list_current_active_segments(
        self,
        specifications: Iterable[StageSpecification],
        *,
        video_ids: Iterable[str] | None = None,
    ) -> list[SegmentRecord]:
        """Return one trusted, current SQLite generation per video and stage."""
        try:
            expected = tuple(specifications)
        except TypeError as error:
            raise ValueError("current segment specifications must be a collection") from error
        resolved: list[StageSpecification] = []
        kinds: set[StageKind] = set()
        for specification in expected:
            if not isinstance(specification, StageSpecification):
                raise ValueError("current segment specifications must be validated")
            kind = self._resolve_segment_stage_kind(specification.kind)
            if kind in kinds:
                raise ValueError("current segment specifications must have unique stages")
            kinds.add(kind)
            resolved.append(specification)

        selected_ids: tuple[str, ...] | None = None
        if video_ids is not None:
            try:
                selected_ids = tuple(dict.fromkeys(video_ids))
            except TypeError as error:
                raise ValueError("video ids must be a collection") from error
            if any(type(video_id) is not str or not video_id for video_id in selected_ids):
                raise ValueError("video ids must be non-empty strings")
            if not selected_ids:
                return []
        if not resolved:
            return []

        with self._connect() as connection:
            if selected_ids is None:
                rows = connection.execute(
                    """
                    SELECT DISTINCT video_id
                    FROM active_segment_generations
                    ORDER BY video_id
                    """
                ).fetchall()
                candidate_video_ids = tuple(str(row["video_id"]) for row in rows)
            else:
                candidate_video_ids = selected_ids

            segments: list[SegmentRecord] = []
            for video_id in candidate_video_ids:
                for specification in resolved:
                    generation = self._get_active_segment_generation(
                        connection,
                        video_id,
                        specification.kind,
                    )
                    if (
                        generation is None
                        or generation.specification_hash
                        != specification.specification_hash
                    ):
                        continue
                    segments.extend(
                        self._list_generation_segments(connection, generation)
                    )
        return segments

    @staticmethod
    def _resolve_semantic_specifications(
        specifications: Iterable[StageSpecification],
    ) -> tuple[StageSpecification, ...]:
        try:
            candidates = tuple(specifications)
        except TypeError as error:
            raise ValueError("semantic specifications must be a collection") from error
        by_kind: dict[StageKind, StageSpecification] = {}
        for specification in candidates:
            if not isinstance(specification, StageSpecification):
                raise ValueError("semantic specifications must be validated")
            if specification.kind not in TEXT_VECTOR_INPUT_STAGE_KINDS:
                raise ValueError("text vector inputs must be semantic segment stages")
            if specification.kind in by_kind:
                raise ValueError("semantic specifications must have unique stages")
            by_kind[specification.kind] = specification
        if set(by_kind) != set(TEXT_VECTOR_INPUT_STAGE_KINDS):
            raise ValueError("semantic specifications must cover speech, OCR and objects")
        return tuple(by_kind[kind] for kind in TEXT_VECTOR_INPUT_STAGE_KINDS)

    @staticmethod
    def _persist_stage_specification(
        connection: sqlite3.Connection,
        specification: StageSpecification,
        *,
        timestamp: str,
    ) -> None:
        connection.execute(
            """
            INSERT OR IGNORE INTO stage_specifications (
                specification_hash, stage_kind, canonical_json, created_at
            ) VALUES (?, ?, ?, ?)
            """,
            (
                specification.specification_hash,
                specification.kind.value,
                specification.canonical_json,
                timestamp,
            ),
        )
        row = connection.execute(
            """
            SELECT stage_kind, canonical_json
            FROM stage_specifications
            WHERE specification_hash = ?
            """,
            (specification.specification_hash,),
        ).fetchone()
        if row is None or (
            row["stage_kind"] != specification.kind.value
            or row["canonical_json"] != specification.canonical_json
        ):
            raise ValueError("stage specification hash collision or corrupted registry")

    @staticmethod
    def _get_stage_specification_from_connection(
        connection: sqlite3.Connection,
        specification_hash: str,
    ) -> StageSpecification:
        row = connection.execute(
            """
            SELECT stage_kind, canonical_json
            FROM stage_specifications
            WHERE specification_hash = ?
            """,
            (specification_hash,),
        ).fetchone()
        if row is None:
            raise ValueError("persisted stage specification is missing")
        try:
            specification = StageSpecification.from_canonical_json(row["canonical_json"])
        except (TypeError, ValueError) as error:
            raise ValueError("persisted stage specification is corrupt") from error
        if (
            specification.specification_hash != specification_hash
            or specification.kind.value != row["stage_kind"]
        ):
            raise ValueError("persisted stage specification is corrupt")
        return specification

    @staticmethod
    def _persist_text_vector_index_specification(
        connection: sqlite3.Connection,
        specification: TextVectorIndexSpecification,
        *,
        timestamp: str,
    ) -> None:
        connection.execute(
            """
            INSERT OR IGNORE INTO text_vector_index_specifications (
                specification_hash, canonical_json, collection_name, created_at
            ) VALUES (?, ?, ?, ?)
            """,
            (
                specification.specification_hash,
                specification.canonical_json,
                specification.collection_name,
                timestamp,
            ),
        )
        row = connection.execute(
            """
            SELECT canonical_json, collection_name
            FROM text_vector_index_specifications
            WHERE specification_hash = ?
            """,
            (specification.specification_hash,),
        ).fetchone()
        if row is None or (
            row["canonical_json"] != specification.canonical_json
            or row["collection_name"] != specification.collection_name
        ):
            raise ValueError("text vector index specification registry is corrupt")

    @staticmethod
    def _get_text_vector_index_specification(
        connection: sqlite3.Connection,
        specification_hash: str,
    ) -> TextVectorIndexSpecification:
        row = connection.execute(
            """
            SELECT canonical_json, collection_name
            FROM text_vector_index_specifications
            WHERE specification_hash = ?
            """,
            (specification_hash,),
        ).fetchone()
        if row is None:
            raise ValueError("persisted text vector index specification is missing")
        try:
            specification = TextVectorIndexSpecification.from_canonical_json(
                row["canonical_json"]
            )
        except (TypeError, ValueError) as error:
            raise ValueError("persisted text vector index specification is corrupt") from error
        if (
            specification.specification_hash != specification_hash
            or specification.collection_name != row["collection_name"]
        ):
            raise ValueError("persisted text vector index specification is corrupt")
        return specification

    @staticmethod
    def _segment_content_manifest(
        specification: StageSpecification,
        generation: SegmentGeneration | None,
        segments: Iterable[SegmentRecord],
    ) -> str:
        return _canonical_sha256(
            {
                "generation_id": generation.generation_id if generation is not None else None,
                "segments": [
                    {
                        "confidence": segment.confidence,
                        "end": segment.end,
                        "id": segment.id,
                        "metadata": segment.metadata,
                        "modality": segment.modality,
                        "start": segment.start,
                        "text": segment.text,
                        "thumbnail_path": segment.thumbnail_path,
                    }
                    for segment in segments
                ],
                "specification_hash": specification.specification_hash,
                "stage_kind": specification.kind.value,
            }
        )

    @classmethod
    def _snapshot_text_vector_inputs(
        cls,
        connection: sqlite3.Connection,
        *,
        video_id: str,
        source_sha256: str,
        semantic_specifications: tuple[StageSpecification, ...],
    ) -> tuple[tuple[TextVectorGenerationInput, ...], tuple[TextVectorPointSource, ...]]:
        inputs: list[TextVectorGenerationInput] = []
        points: list[TextVectorPointSource] = []
        for specification in semantic_specifications:
            active = cls._get_active_segment_generation(
                connection,
                video_id,
                specification.kind,
            )
            if (
                active is None
                or active.specification_hash != specification.specification_hash
                or active.source_sha256 != source_sha256
            ):
                inputs.append(
                    TextVectorGenerationInput(
                        stage_kind=specification.kind,
                        specification_hash=specification.specification_hash,
                        segment_generation_id=None,
                        segment_run_id=None,
                        source_sha256=None,
                        segment_count=0,
                        content_manifest_sha256=cls._segment_content_manifest(
                            specification,
                            None,
                            (),
                        ),
                    )
                )
                continue
            segments = cls._list_generation_segments(connection, active)
            inputs.append(
                TextVectorGenerationInput(
                    stage_kind=specification.kind,
                    specification_hash=specification.specification_hash,
                    segment_generation_id=active.generation_id,
                    segment_run_id=active.run_id,
                    source_sha256=active.source_sha256,
                    segment_count=active.segment_count,
                    content_manifest_sha256=cls._segment_content_manifest(
                        specification,
                        active,
                        segments,
                    ),
                )
            )
            points.extend(
                TextVectorPointSource(
                    video_id=segment.video_id,
                    segment_id=segment.id,
                    modality=segment.modality,
                    text=segment.text,
                    segment_generation_id=active.generation_id,
                    text_sha256=hashlib.sha256(segment.text.encode("utf-8")).hexdigest(),
                )
                for segment in segments
                if _is_searchable_text_segment(segment)
            )
        return tuple(inputs), tuple(points)

    @staticmethod
    def _validate_text_vector_stage_dependencies(
        text_specification: StageSpecification,
        semantic_specifications: tuple[StageSpecification, ...],
    ) -> None:
        if text_specification.kind is not StageKind.TEXT_VECTORS:
            raise ValueError("text vector build requires a text vector stage run")
        expected = {
            f"{specification.kind.value}_specification": specification.specification_hash
            for specification in semantic_specifications
        }
        for name, value in expected.items():
            persisted = text_specification.dependencies.get(name)
            if persisted is not None and persisted != value:
                raise ValueError("text vector stage dependency identity is inconsistent")

    def reserve_text_vector_generation(
        self,
        run_id: str,
        *,
        index_specification: TextVectorIndexSpecification,
        semantic_specifications: Iterable[StageSpecification],
        generation_id: str | None = None,
        lease_seconds: int = 900,
    ) -> TextVectorBuildPlan:
        if not isinstance(index_specification, TextVectorIndexSpecification):
            raise ValueError("text vector index specification must be validated")
        resolved_semantic = self._resolve_semantic_specifications(semantic_specifications)
        if (
            isinstance(lease_seconds, bool)
            or not isinstance(lease_seconds, int)
            or not 1 <= lease_seconds <= 86_400
        ):
            raise ValueError("text vector build lease must be between 1 and 86400 seconds")
        resolved_generation_id = uuid4().hex if generation_id is None else generation_id
        validate_artifact_identifier(
            resolved_generation_id,
            field_name="text vector generation id",
        )
        reserved_at = _now()
        lease_expires_at = (
            datetime.fromisoformat(reserved_at) + timedelta(seconds=lease_seconds)
        ).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            run = self._get_stage_run(connection, run_id)
            if run is None:
                raise KeyError(run_id)
            if run.state is not StageState.RUNNING or run.stage_kind is not StageKind.TEXT_VECTORS:
                raise ValueError("only a running text vector stage can reserve a build")
            if connection.execute(
                "SELECT 1 FROM text_vector_builds WHERE video_id = ?",
                (run.video_id,),
            ).fetchone() is not None:
                raise ValueError("a text vector build already active for this video")
            text_specification = self._get_stage_specification_from_connection(
                connection,
                run.specification_hash,
            )
            self._validate_text_vector_stage_dependencies(
                text_specification,
                resolved_semantic,
            )
            for specification in resolved_semantic:
                self._persist_stage_specification(
                    connection,
                    specification,
                    timestamp=reserved_at,
                )
            self._persist_text_vector_index_specification(
                connection,
                index_specification,
                timestamp=reserved_at,
            )
            inputs, points = self._snapshot_text_vector_inputs(
                connection,
                video_id=run.video_id,
                source_sha256=run.source_sha256,
                semantic_specifications=resolved_semantic,
            )
            previous = self._get_active_text_vector_generation(connection, run.video_id)
            plan = TextVectorBuildPlan(
                generation_id=resolved_generation_id,
                run_id=run.run_id,
                video_id=run.video_id,
                stage_specification_hash=run.specification_hash,
                source_sha256=run.source_sha256,
                index_specification=index_specification,
                expected_previous_generation_id=(
                    previous.generation_id if previous is not None else None
                ),
                inputs=inputs,
                points=points,
                input_manifest_sha256=_text_vector_input_manifest(inputs),
                point_manifest_sha256=_text_vector_point_manifest(points),
                reserved_at=reserved_at,
                lease_expires_at=lease_expires_at,
            )
            try:
                connection.execute(
                    """
                    INSERT INTO text_vector_builds (
                        generation_id, run_id, video_id, stage_specification_hash,
                        source_sha256, index_specification_hash, collection_name,
                        expected_previous_generation_id, input_manifest_sha256,
                        point_manifest_sha256, point_count, reserved_at,
                        heartbeat_at, lease_expires_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        plan.generation_id,
                        plan.run_id,
                        plan.video_id,
                        plan.stage_specification_hash,
                        plan.source_sha256,
                        plan.index_specification.specification_hash,
                        plan.index_specification.collection_name,
                        plan.expected_previous_generation_id,
                        plan.input_manifest_sha256,
                        plan.point_manifest_sha256,
                        len(plan.points),
                        plan.reserved_at,
                        plan.reserved_at,
                        plan.lease_expires_at,
                    ),
                )
                for item in plan.inputs:
                    connection.execute(
                        f"""
                        INSERT INTO text_vector_build_inputs (
                            generation_id, {_TEXT_VECTOR_INPUT_COLUMNS}
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            plan.generation_id,
                            item.stage_kind.value,
                            item.specification_hash,
                            item.segment_generation_id,
                            item.segment_run_id,
                            item.source_sha256,
                            item.segment_count,
                            item.content_manifest_sha256,
                        ),
                    )
            except sqlite3.IntegrityError as error:
                raise ValueError("text vector generation identity already exists or is invalid") from error
        return plan

    @staticmethod
    def _text_vector_input_from_row(row: sqlite3.Row) -> TextVectorGenerationInput:
        try:
            return TextVectorGenerationInput(
                stage_kind=row["stage_kind"],
                specification_hash=row["specification_hash"],
                segment_generation_id=row["segment_generation_id"],
                segment_run_id=row["segment_run_id"],
                source_sha256=row["source_sha256"],
                segment_count=row["segment_count"],
                content_manifest_sha256=row["content_manifest_sha256"],
            )
        except (TypeError, ValueError) as error:
            raise ValueError("persisted text vector input is corrupt") from error

    @classmethod
    def _load_text_vector_inputs(
        cls,
        connection: sqlite3.Connection,
        *,
        table: str,
        generation_id: str,
    ) -> tuple[TextVectorGenerationInput, ...]:
        if table not in {"text_vector_build_inputs", "text_vector_generation_inputs"}:
            raise ValueError("unsupported text vector input table")
        rows = connection.execute(
            f"SELECT {_TEXT_VECTOR_INPUT_COLUMNS} FROM {table} WHERE generation_id = ?",
            (generation_id,),
        ).fetchall()
        by_kind: dict[StageKind, TextVectorGenerationInput] = {}
        for row in rows:
            item = cls._text_vector_input_from_row(row)
            if item.stage_kind in by_kind:
                raise ValueError("persisted text vector inputs are corrupt")
            by_kind[item.stage_kind] = item
        if set(by_kind) != set(TEXT_VECTOR_INPUT_STAGE_KINDS):
            raise ValueError("persisted text vector inputs are incomplete")
        return tuple(by_kind[kind] for kind in TEXT_VECTOR_INPUT_STAGE_KINDS)

    @classmethod
    def _validate_referenced_text_vector_inputs(
        cls,
        connection: sqlite3.Connection,
        inputs: tuple[TextVectorGenerationInput, ...],
        *,
        video_id: str,
        source_sha256: str,
    ) -> tuple[TextVectorPointSource, ...]:
        points: list[TextVectorPointSource] = []
        for item in inputs:
            specification = cls._get_stage_specification_from_connection(
                connection,
                item.specification_hash,
            )
            if specification.kind is not item.stage_kind:
                raise ValueError("persisted text vector input specification is corrupt")
            if not item.present:
                expected_manifest = cls._segment_content_manifest(
                    specification,
                    None,
                    (),
                )
                if item.content_manifest_sha256 != expected_manifest:
                    raise ValueError("persisted text vector input manifest is corrupt")
                continue
            assert item.segment_generation_id is not None
            resolved = cls._get_segment_generation_with_run(
                connection,
                item.segment_generation_id,
            )
            if resolved is None:
                raise ValueError("persisted text vector input generation is missing")
            generation, _run = resolved
            if (
                generation.video_id != video_id
                or generation.stage_kind is not item.stage_kind
                or generation.specification_hash != item.specification_hash
                or generation.source_sha256 != source_sha256
                or generation.source_sha256 != item.source_sha256
                or generation.run_id != item.segment_run_id
                or generation.segment_count != item.segment_count
            ):
                raise ValueError("persisted text vector input lineage is corrupt")
            segments = cls._list_generation_segments(connection, generation)
            expected_manifest = cls._segment_content_manifest(
                specification,
                generation,
                segments,
            )
            if item.content_manifest_sha256 != expected_manifest:
                raise ValueError("persisted text vector input manifest is corrupt")
            points.extend(
                TextVectorPointSource(
                    video_id=segment.video_id,
                    segment_id=segment.id,
                    modality=segment.modality,
                    text=segment.text,
                    segment_generation_id=generation.generation_id,
                    text_sha256=hashlib.sha256(segment.text.encode("utf-8")).hexdigest(),
                )
                for segment in segments
                if _is_searchable_text_segment(segment)
            )
        return tuple(points)

    @classmethod
    def _get_text_vector_generation_with_run(
        cls,
        connection: sqlite3.Connection,
        generation_id: str,
    ) -> tuple[
        TextVectorGeneration,
        StageRun,
        TextVectorIndexSpecification,
        tuple[TextVectorGenerationInput, ...],
        tuple[TextVectorPointSource, ...],
    ] | None:
        row = connection.execute(
            f"{_TEXT_VECTOR_GENERATION_SELECT} WHERE generation_id = ?",
            (generation_id,),
        ).fetchone()
        if row is None:
            return None
        try:
            generation = TextVectorGeneration(**dict(row))
        except (TypeError, ValueError) as error:
            raise ValueError("persisted text vector generation is corrupt") from error
        run = cls._get_stage_run(connection, generation.run_id)
        if run is None:
            raise ValueError("persisted text vector generation run is missing")
        if (
            run.video_id != generation.video_id
            or run.stage_kind is not StageKind.TEXT_VECTORS
            or run.specification_hash != generation.specification_hash
            or run.source_sha256 != generation.source_sha256
            or run.output_generation != generation.generation_id
            or run.state not in {StageState.COMPLETE, StageState.STALE}
            or run.finished_at != generation.completed_at
        ):
            raise ValueError("persisted text vector generation identity is corrupt")
        index_specification = cls._get_text_vector_index_specification(
            connection,
            generation.index_specification_hash,
        )
        if generation.collection_name != index_specification.collection_name:
            raise ValueError("persisted text vector generation collection is corrupt")
        inputs = cls._load_text_vector_inputs(
            connection,
            table="text_vector_generation_inputs",
            generation_id=generation.generation_id,
        )
        points = cls._validate_referenced_text_vector_inputs(
            connection,
            inputs,
            video_id=generation.video_id,
            source_sha256=generation.source_sha256,
        )
        if (
            generation.input_manifest_sha256 != _text_vector_input_manifest(inputs)
            or generation.point_manifest_sha256 != _text_vector_point_manifest(points)
            or generation.point_count != len(points)
        ):
            raise ValueError("persisted text vector generation manifest is corrupt")
        return generation, run, index_specification, inputs, points

    @classmethod
    def _get_active_text_vector_generation(
        cls,
        connection: sqlite3.Connection,
        video_id: str,
    ) -> TextVectorGeneration | None:
        pointer = connection.execute(
            """
            SELECT generation_id, activated_at
            FROM active_text_vector_generations
            WHERE video_id = ?
            """,
            (video_id,),
        ).fetchone()
        if pointer is None:
            return None
        resolved = cls._get_text_vector_generation_with_run(
            connection,
            pointer["generation_id"],
        )
        if resolved is None:
            raise ValueError("active text vector generation pointer is dangling")
        generation, run, _index, _inputs, _points = resolved
        if (
            generation.video_id != video_id
            or run.state is not StageState.COMPLETE
            or pointer["activated_at"] != generation.completed_at
        ):
            raise ValueError("active text vector generation pointer is corrupt")
        return generation

    def get_text_vector_generation(
        self,
        generation_id: str,
    ) -> TextVectorGeneration | None:
        with self._connect() as connection:
            resolved = self._get_text_vector_generation_with_run(
                connection,
                generation_id,
            )
        return resolved[0] if resolved is not None else None

    def get_active_text_vector_generation(
        self,
        video_id: str,
    ) -> TextVectorGeneration | None:
        with self._connect() as connection:
            return self._get_active_text_vector_generation(connection, video_id)

    @classmethod
    def _validate_text_vector_build_row(
        cls,
        connection: sqlite3.Connection,
        build: sqlite3.Row,
        run: StageRun,
    ) -> tuple[
        TextVectorIndexSpecification,
        tuple[TextVectorGenerationInput, ...],
        tuple[TextVectorPointSource, ...],
    ]:
        try:
            validate_artifact_identifier(
                build["generation_id"],
                field_name="text vector build generation id",
            )
            validate_artifact_identifier(
                build["run_id"],
                field_name="text vector build run id",
            )
            validate_artifact_identifier(
                build["video_id"],
                field_name="text vector build video id",
            )
            validate_artifact_identifier(
                build["collection_name"],
                field_name="text vector build collection name",
            )
            if any(
                type(build[name]) is not str
                or re.fullmatch(r"[0-9a-f]{64}", build[name]) is None
                for name in (
                    "stage_specification_hash",
                    "source_sha256",
                    "index_specification_hash",
                    "input_manifest_sha256",
                    "point_manifest_sha256",
                )
            ):
                raise ValueError("text vector build digest is invalid")
            if (
                isinstance(build["point_count"], bool)
                or not isinstance(build["point_count"], int)
                or build["point_count"] < 0
            ):
                raise ValueError("text vector build point count is invalid")
            if build["expected_previous_generation_id"] is not None:
                validate_artifact_identifier(
                    build["expected_previous_generation_id"],
                    field_name="expected previous text vector generation id",
                )
            reserved_at = datetime.fromisoformat(build["reserved_at"])
            heartbeat_at = datetime.fromisoformat(build["heartbeat_at"])
            lease_expires_at = datetime.fromisoformat(build["lease_expires_at"])
            if any(
                value.tzinfo is None or value.utcoffset() is None
                for value in (reserved_at, heartbeat_at, lease_expires_at)
            ):
                raise ValueError("text vector build timestamps require timezones")
            if not reserved_at <= heartbeat_at < lease_expires_at:
                raise ValueError("text vector build timestamps are inconsistent")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("persisted text vector build is corrupt") from error
        if (
            run.run_id != build["run_id"]
            or run.video_id != build["video_id"]
            or run.stage_kind is not StageKind.TEXT_VECTORS
            or run.specification_hash != build["stage_specification_hash"]
            or run.source_sha256 != build["source_sha256"]
        ):
            raise ValueError("persisted text vector build run identity is corrupt")
        index_specification = cls._get_text_vector_index_specification(
            connection,
            build["index_specification_hash"],
        )
        if index_specification.collection_name != build["collection_name"]:
            raise ValueError("persisted text vector build collection is corrupt")
        inputs = cls._load_text_vector_inputs(
            connection,
            table="text_vector_build_inputs",
            generation_id=build["generation_id"],
        )
        points = cls._validate_referenced_text_vector_inputs(
            connection,
            inputs,
            video_id=run.video_id,
            source_sha256=run.source_sha256,
        )
        if (
            _text_vector_input_manifest(inputs) != build["input_manifest_sha256"]
            or _text_vector_point_manifest(points) != build["point_manifest_sha256"]
            or len(points) != build["point_count"]
        ):
            raise ValueError("persisted text vector build manifest is corrupt")
        return index_specification, inputs, points

    def heartbeat_text_vector_build(
        self,
        generation_id: str,
        *,
        lease_seconds: int = 900,
    ) -> str:
        validate_artifact_identifier(generation_id, field_name="text vector generation id")
        if (
            isinstance(lease_seconds, bool)
            or not isinstance(lease_seconds, int)
            or not 1 <= lease_seconds <= 86_400
        ):
            raise ValueError("text vector build lease must be between 1 and 86400 seconds")
        heartbeat_at = _now()
        lease_expires_at = (
            datetime.fromisoformat(heartbeat_at) + timedelta(seconds=lease_seconds)
        ).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM text_vector_builds WHERE generation_id = ?",
                (generation_id,),
            ).fetchone()
            if row is None:
                raise KeyError(generation_id)
            run = self._get_stage_run(connection, row["run_id"])
            if run is None or run.state is not StageState.RUNNING:
                raise ValueError("text vector build run is unavailable")
            self._validate_text_vector_build_row(connection, row, run)
            if datetime.fromisoformat(row["lease_expires_at"]) <= datetime.fromisoformat(
                heartbeat_at
            ):
                raise ValueError("text vector build lease has expired")
            connection.execute(
                """
                UPDATE text_vector_builds
                SET heartbeat_at = ?, lease_expires_at = ?
                WHERE generation_id = ?
                """,
                (heartbeat_at, lease_expires_at, generation_id),
            )
        return lease_expires_at

    def commit_text_vector_generation(
        self,
        run_id: str,
        *,
        receipt: TextVectorBuildReceipt,
    ) -> TextVectorGeneration:
        if not isinstance(receipt, TextVectorBuildReceipt):
            raise ValueError("text vector build receipt must be validated")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            build = connection.execute(
                "SELECT * FROM text_vector_builds WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if build is None:
                existing = connection.execute(
                    f"{_TEXT_VECTOR_GENERATION_SELECT} WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                if existing is None:
                    raise KeyError(run_id)
                resolved = self._get_text_vector_generation_with_run(
                    connection,
                    existing["generation_id"],
                )
                assert resolved is not None
                generation = resolved[0]
                if (
                    generation.generation_id != receipt.generation_id
                    or generation.index_specification_hash
                    != receipt.index_specification_hash
                    or generation.point_count != receipt.point_count
                    or generation.point_manifest_sha256
                    != receipt.point_manifest_sha256
                    or generation.vector_manifest_sha256
                    != receipt.vector_manifest_sha256
                ):
                    raise ValueError("text vector receipt does not match completed generation")
                return generation
            run = self._get_stage_run(connection, run_id)
            if run is None:
                raise KeyError(run_id)
            if run.state is not StageState.RUNNING or run.stage_kind is not StageKind.TEXT_VECTORS:
                raise ValueError("only a running text vector stage can publish a generation")
            _index_specification, validated_build_inputs, _build_points = (
                self._validate_text_vector_build_row(connection, build, run)
            )
            if datetime.fromisoformat(build["lease_expires_at"]) <= datetime.now(UTC):
                raise ValueError("text vector build lease has expired")
            if (
                receipt.generation_id != build["generation_id"]
                or receipt.index_specification_hash != build["index_specification_hash"]
                or receipt.point_count != build["point_count"]
                or receipt.point_manifest_sha256 != build["point_manifest_sha256"]
            ):
                raise ValueError("text vector build receipt does not match reservation")
            stored_inputs = validated_build_inputs
            semantic_specifications = tuple(
                self._get_stage_specification_from_connection(
                    connection,
                    item.specification_hash,
                )
                for item in stored_inputs
            )
            current_inputs, current_points = self._snapshot_text_vector_inputs(
                connection,
                video_id=run.video_id,
                source_sha256=run.source_sha256,
                semantic_specifications=semantic_specifications,
            )
            if (
                current_inputs != stored_inputs
                or _text_vector_input_manifest(current_inputs)
                != build["input_manifest_sha256"]
                or _text_vector_point_manifest(current_points)
                != build["point_manifest_sha256"]
                or len(current_points) != build["point_count"]
            ):
                raise ValueError("text vector inputs changed during build")
            active = self._get_active_text_vector_generation(connection, run.video_id)
            active_id = active.generation_id if active is not None else None
            if active_id != build["expected_previous_generation_id"]:
                raise ValueError("active text vector generation changed during build")
            completed_at = _now()
            generation = TextVectorGeneration(
                generation_id=build["generation_id"],
                video_id=run.video_id,
                specification_hash=run.specification_hash,
                source_sha256=run.source_sha256,
                run_id=run.run_id,
                index_specification_hash=build["index_specification_hash"],
                collection_name=build["collection_name"],
                input_manifest_sha256=build["input_manifest_sha256"],
                point_manifest_sha256=build["point_manifest_sha256"],
                vector_manifest_sha256=receipt.vector_manifest_sha256,
                point_count=build["point_count"],
                completed_at=completed_at,
            )
            connection.execute(
                """
                INSERT INTO text_vector_generations (
                    generation_id, video_id, specification_hash, source_sha256,
                    run_id, index_specification_hash, collection_name,
                    input_manifest_sha256, point_manifest_sha256,
                    vector_manifest_sha256, point_count, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    generation.generation_id,
                    generation.video_id,
                    generation.specification_hash,
                    generation.source_sha256,
                    generation.run_id,
                    generation.index_specification_hash,
                    generation.collection_name,
                    generation.input_manifest_sha256,
                    generation.point_manifest_sha256,
                    generation.vector_manifest_sha256,
                    generation.point_count,
                    generation.completed_at,
                ),
            )
            for item in stored_inputs:
                connection.execute(
                    f"""
                    INSERT INTO text_vector_generation_inputs (
                        generation_id, {_TEXT_VECTOR_INPUT_COLUMNS}
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        generation.generation_id,
                        item.stage_kind.value,
                        item.specification_hash,
                        item.segment_generation_id,
                        item.segment_run_id,
                        item.source_sha256,
                        item.segment_count,
                        item.content_manifest_sha256,
                    ),
                )
            if active_id is None:
                try:
                    connection.execute(
                        """
                        INSERT INTO active_text_vector_generations (
                            video_id, generation_id, activated_at
                        ) VALUES (?, ?, ?)
                        """,
                        (generation.video_id, generation.generation_id, completed_at),
                    )
                except sqlite3.IntegrityError as error:
                    raise RuntimeError(
                        "active text vector generation changed during activation"
                    ) from error
            else:
                cursor = connection.execute(
                    """
                    UPDATE active_text_vector_generations
                    SET generation_id = ?, activated_at = ?
                    WHERE video_id = ? AND generation_id = ?
                    """,
                    (
                        generation.generation_id,
                        completed_at,
                        generation.video_id,
                        active_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError(
                        "active text vector generation changed during activation"
                    )
            cursor = connection.execute(
                """
                UPDATE stage_runs
                SET state = ?, output_generation = ?, error_code = NULL,
                    finished_at = ?, updated_at = ?
                WHERE run_id = ? AND state = ?
                """,
                (
                    StageState.COMPLETE.value,
                    generation.generation_id,
                    completed_at,
                    completed_at,
                    run.run_id,
                    StageState.RUNNING.value,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("text vector stage changed during generation activation")
            connection.execute(
                "DELETE FROM text_vector_builds WHERE generation_id = ?",
                (generation.generation_id,),
            )
        return generation

    @staticmethod
    def _enqueue_artifact_gc(
        connection: sqlite3.Connection,
        *,
        generation_id: str,
        index_specification_hash: str,
        collection_name: str,
        reason: str,
        timestamp: str,
    ) -> None:
        validate_artifact_identifier(reason, field_name="artifact GC reason")
        connection.execute(
            """
            INSERT OR IGNORE INTO artifact_gc_jobs (
                job_id, artifact_kind, generation_id, index_specification_hash,
                collection_name, reason, state, attempt, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                uuid4().hex,
                StageKind.TEXT_VECTORS.value,
                generation_id,
                index_specification_hash,
                collection_name,
                reason,
                "pending",
                0,
                timestamp,
                timestamp,
            ),
        )

    def fail_text_vector_build(self, run_id: str, *, error_code: str) -> StageRun:
        validate_error_code(error_code)
        timestamp = _now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            build = connection.execute(
                "SELECT * FROM text_vector_builds WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if build is None:
                raise KeyError(run_id)
            run = self._get_stage_run(connection, run_id)
            if run is None or run.state is not StageState.RUNNING:
                raise ValueError("only a running text vector build can fail")
            self._validate_text_vector_build_row(connection, build, run)
            self._enqueue_artifact_gc(
                connection,
                generation_id=build["generation_id"],
                index_specification_hash=build["index_specification_hash"],
                collection_name=build["collection_name"],
                reason=error_code,
                timestamp=timestamp,
            )
            cursor = connection.execute(
                """
                UPDATE stage_runs
                SET state = ?, output_generation = NULL, error_code = ?,
                    finished_at = ?, updated_at = ?
                WHERE run_id = ? AND state = ?
                """,
                (
                    StageState.FAILED.value,
                    error_code,
                    timestamp,
                    timestamp,
                    run.run_id,
                    StageState.RUNNING.value,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("text vector stage changed while failing build")
            connection.execute(
                "DELETE FROM text_vector_builds WHERE generation_id = ?",
                (build["generation_id"],),
            )
        failed = self.get_stage_run(run_id)
        assert failed is not None
        return failed

    def expire_text_vector_builds(self, *, before: str | None = None) -> tuple[str, ...]:
        cutoff = _now() if before is None else before
        if type(cutoff) is not str:
            raise ValueError("text vector build expiry cutoff must be an ISO timestamp")
        try:
            resolved_cutoff = datetime.fromisoformat(cutoff)
        except ValueError as error:
            raise ValueError("text vector build expiry cutoff must be an ISO timestamp") from error
        if resolved_cutoff.tzinfo is None or resolved_cutoff.utcoffset() is None:
            raise ValueError("text vector build expiry cutoff must include a timezone")
        expired: list[str] = []
        timestamp = _now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT * FROM text_vector_builds
                WHERE lease_expires_at <= ?
                ORDER BY reserved_at, generation_id
                """,
                (resolved_cutoff.astimezone(UTC).isoformat(),),
            ).fetchall()
            for build in rows:
                run = self._get_stage_run(connection, build["run_id"])
                if run is None:
                    raise ValueError("expired text vector build run is corrupt")
                self._validate_text_vector_build_row(connection, build, run)
                self._enqueue_artifact_gc(
                    connection,
                    generation_id=build["generation_id"],
                    index_specification_hash=build["index_specification_hash"],
                    collection_name=build["collection_name"],
                    reason="text_vector_build_interrupted",
                    timestamp=timestamp,
                )
                if run.state is StageState.RUNNING:
                    cursor = connection.execute(
                        """
                        UPDATE stage_runs
                        SET state = ?, output_generation = NULL, error_code = ?,
                            finished_at = ?, updated_at = ?
                        WHERE run_id = ? AND state = ?
                        """,
                        (
                            StageState.FAILED.value,
                            "text_vector_build_interrupted",
                            timestamp,
                            timestamp,
                            run.run_id,
                            StageState.RUNNING.value,
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise RuntimeError("expired text vector stage changed concurrently")
                elif run.state not in {
                    StageState.FAILED,
                    StageState.CANCELLED,
                }:
                    raise ValueError("expired text vector build run is corrupt")
                connection.execute(
                    "DELETE FROM text_vector_builds WHERE generation_id = ?",
                    (build["generation_id"],),
                )
                expired.append(build["generation_id"])
        return tuple(expired)

    @classmethod
    def _artifact_gc_job_from_row(
        cls,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> ArtifactGCJob:
        try:
            job = ArtifactGCJob(**dict(row))
        except (TypeError, ValueError) as error:
            raise ValueError("persisted artifact GC job is corrupt") from error
        specification = cls._get_text_vector_index_specification(
            connection,
            job.index_specification_hash,
        )
        if specification.collection_name != job.collection_name:
            raise ValueError("persisted artifact GC collection is corrupt")
        if connection.execute(
            """
            SELECT 1
            FROM active_text_vector_generations
            WHERE generation_id = ?
            """,
            (job.generation_id,),
        ).fetchone() is not None:
            raise ValueError("artifact GC job targets an active generation")
        return job

    def list_pending_artifact_gc_jobs(self) -> tuple[ArtifactGCJob, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                f"{_ARTIFACT_GC_SELECT} WHERE state = ? ORDER BY created_at, job_id",
                ("pending",),
            ).fetchall()
            jobs = tuple(self._artifact_gc_job_from_row(connection, row) for row in rows)
        return jobs

    def claim_artifact_gc_job(self, job_id: str) -> ArtifactGCJob:
        validate_artifact_identifier(job_id, field_name="artifact GC job id")
        timestamp = _now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            pending = connection.execute(
                f"{_ARTIFACT_GC_SELECT} WHERE job_id = ? AND state = 'pending'",
                (job_id,),
            ).fetchone()
            if pending is None:
                raise ValueError("artifact GC job is not pending")
            self._artifact_gc_job_from_row(connection, pending)
            cursor = connection.execute(
                """
                UPDATE artifact_gc_jobs
                SET state = 'running', attempt = attempt + 1, updated_at = ?
                WHERE job_id = ? AND state = 'pending'
                """,
                (timestamp, job_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("artifact GC job is not pending")
            row = connection.execute(
                f"{_ARTIFACT_GC_SELECT} WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            assert row is not None
            job = self._artifact_gc_job_from_row(connection, row)
        return job

    def complete_artifact_gc_job(self, job_id: str, *, succeeded: bool) -> ArtifactGCJob:
        validate_artifact_identifier(job_id, field_name="artifact GC job id")
        if type(succeeded) is not bool:
            raise ValueError("artifact GC outcome must be boolean")
        timestamp = _now()
        state = "complete" if succeeded else "failed"
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            running = connection.execute(
                f"{_ARTIFACT_GC_SELECT} WHERE job_id = ? AND state = 'running'",
                (job_id,),
            ).fetchone()
            if running is None:
                raise ValueError("artifact GC job is not running")
            self._artifact_gc_job_from_row(connection, running)
            cursor = connection.execute(
                """
                UPDATE artifact_gc_jobs
                SET state = ?, updated_at = ?
                WHERE job_id = ? AND state = 'running'
                """,
                (state, timestamp, job_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("artifact GC job is not running")
            row = connection.execute(
                f"{_ARTIFACT_GC_SELECT} WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            assert row is not None
            job = self._artifact_gc_job_from_row(connection, row)
        return job

    def list_current_text_vector_bindings(
        self,
        *,
        video_ids: Iterable[str],
        text_specification: StageSpecification,
        semantic_specifications: Iterable[StageSpecification],
        required_modalities: set[str] | None = None,
    ) -> tuple[TextVectorSearchBinding, ...]:
        if (
            not isinstance(text_specification, StageSpecification)
            or text_specification.kind is not StageKind.TEXT_VECTORS
        ):
            raise ValueError("current text vector specification must be validated")
        semantic = self._resolve_semantic_specifications(semantic_specifications)
        self._validate_text_vector_stage_dependencies(text_specification, semantic)
        try:
            selected_ids = tuple(dict.fromkeys(video_ids))
        except TypeError as error:
            raise ValueError("video ids must be a collection") from error
        if any(type(video_id) is not str or not video_id for video_id in selected_ids):
            raise ValueError("video ids must be non-empty strings")
        required = set() if required_modalities is None else set(required_modalities)
        if not required <= {"speech", "ocr", "objects"}:
            raise ValueError("unsupported required text vector modality")
        bindings: list[TextVectorSearchBinding] = []
        expected_by_kind = {item.kind: item for item in semantic}
        with self._connect() as connection:
            for video_id in selected_ids:
                generation = self._get_active_text_vector_generation(connection, video_id)
                if generation is None or generation.specification_hash != text_specification.specification_hash:
                    continue
                resolved = self._get_text_vector_generation_with_run(
                    connection,
                    generation.generation_id,
                )
                assert resolved is not None
                _generation, _run, index_specification, inputs, points = resolved
                stale = False
                for item in inputs:
                    expected = expected_by_kind[item.stage_kind]
                    if item.specification_hash != expected.specification_hash:
                        stale = True
                        break
                    active = self._get_active_segment_generation(
                        connection,
                        video_id,
                        item.stage_kind,
                    )
                    current_id = None
                    if (
                        active is not None
                        and active.specification_hash == expected.specification_hash
                        and active.source_sha256 == generation.source_sha256
                    ):
                        current_id = active.generation_id
                    if current_id != item.segment_generation_id:
                        stale = True
                        break
                if stale:
                    continue
                binding = TextVectorSearchBinding(
                    generation=generation,
                    index_specification=index_specification,
                    inputs=inputs,
                    points=points,
                )
                if any(not binding.supports_modality(modality) for modality in required):
                    continue
                bindings.append(binding)
        return tuple(bindings)

    def add_segment(
        self,
        *,
        segment_id: str,
        video_id: str,
        start: float,
        end: float,
        modality: str,
        text: str,
        confidence: float,
        metadata: dict[str, Any] | None = None,
        thumbnail_path: str | None = None,
    ) -> SegmentRecord:
        record = self._validated_segment(
            segment_id=segment_id,
            video_id=video_id,
            start=start,
            end=end,
            modality=modality,
            text=text,
            confidence=confidence,
            metadata={} if metadata is None else metadata,
            thumbnail_path=thumbnail_path,
        )
        try:
            payload = json.dumps(
                record.metadata,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("segment metadata must be finite JSON") from error
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO segments (
                    id, video_id, start, end, modality, text,
                    confidence, metadata_json, thumbnail_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.video_id,
                    record.start,
                    record.end,
                    record.modality,
                    record.text,
                    record.confidence,
                    payload,
                    record.thumbnail_path,
                ),
            )
        return record

    @staticmethod
    def _validated_segment(
        *,
        segment_id: object,
        video_id: object,
        start: object,
        end: object,
        modality: object,
        text: object,
        confidence: object,
        metadata: object,
        thumbnail_path: object,
    ) -> SegmentRecord:
        if type(segment_id) is not str or not segment_id:
            raise ValueError("segment id must not be empty")
        if type(video_id) is not str or not video_id:
            raise ValueError("segment video id must not be empty")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, Real)
            or not isinstance(end, Real)
        ):
            raise ValueError("segment interval must be numeric")
        resolved_start = float(start)
        resolved_end = float(end)
        if (
            not math.isfinite(resolved_start)
            or not math.isfinite(resolved_end)
            or resolved_start < 0
            or resolved_end <= resolved_start
        ):
            raise ValueError("segment interval must be finite, non-negative and ordered")
        if modality not in SEGMENT_MODALITIES:
            raise ValueError("unsupported segment modality")
        if type(text) is not str:
            raise ValueError("segment text must be a string")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, Real)
            or not math.isfinite(resolved_confidence := float(confidence))
            or not 0 <= resolved_confidence <= 1
        ):
            raise ValueError("segment confidence must be finite and between zero and one")
        if not isinstance(metadata, dict):
            raise ValueError("segment metadata must be an object")
        if thumbnail_path is not None and type(thumbnail_path) is not str:
            raise ValueError("segment thumbnail path must be a string")
        return SegmentRecord(
            id=segment_id,
            video_id=video_id,
            start=resolved_start,
            end=resolved_end,
            modality=modality,
            text=text,
            confidence=resolved_confidence,
            metadata=dict(metadata),
            thumbnail_path=thumbnail_path,
        )

    def list_segments(self, video_id: str | None = None) -> list[SegmentRecord]:
        query = "SELECT * FROM segments"
        parameters: tuple[object, ...] = ()
        if video_id is not None:
            query += " WHERE video_id = ?"
            parameters = (video_id,)
        query += " ORDER BY video_id, start, end"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        segments: list[SegmentRecord] = []
        for row in rows:
            try:
                metadata = json.loads(
                    row["metadata_json"],
                    parse_constant=_reject_non_finite_json,
                )
                segments.append(
                    self._validated_segment(
                        segment_id=row["id"],
                        video_id=row["video_id"],
                        start=row["start"],
                        end=row["end"],
                        modality=row["modality"],
                        text=row["text"],
                        confidence=row["confidence"],
                        metadata=metadata,
                        thumbnail_path=row["thumbnail_path"],
                    )
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                logger.warning("Ignoring invalid persisted segment %s", row["id"])
        return segments

    def clear_segments(self, video_id: str, *, modality: str | None = None) -> None:
        with self._connect() as connection:
            if modality is None:
                connection.execute(
                    "DELETE FROM segments WHERE video_id = ? AND generation_id IS NULL",
                    (video_id,),
                )
            else:
                connection.execute(
                    """
                    DELETE FROM segments
                    WHERE video_id = ? AND modality = ? AND generation_id IS NULL
                    """,
                    (video_id, modality),
                )

    def search_segments_lexical(
        self,
        query: str,
        *,
        video_ids: list[str] | None = None,
        limit: int = 50,
        query_variants: list[str] | None = None,
        specifications: Iterable[StageSpecification] = (),
    ) -> list[tuple[SegmentRecord, float]]:
        variants = query_variants or [query]
        if not any(value.strip() for value in variants):
            return []
        candidates = self.list_current_active_segments(
            specifications,
            video_ids=video_ids,
        )
        ranked: list[tuple[SegmentRecord, float]] = []
        for segment in candidates:
            if video_ids and segment.video_id not in video_ids:
                continue
            score = max(lexical_match(variant, segment.text).score for variant in variants)
            if score <= 0:
                continue
            ranked.append((segment, score))
        return sorted(ranked, key=lambda item: item[1], reverse=True)[:limit]
