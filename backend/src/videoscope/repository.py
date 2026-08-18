from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json
import logging
import math
from numbers import Real
import os
import sqlite3
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from time import sleep
from typing import Any, Callable
from uuid import uuid4
from weakref import WeakValueDictionary

from videoscope.artifacts import (
    AssetIdentityError,
    AssetRecord,
    SEGMENT_STAGE_KINDS,
    SegmentGeneration,
    StageKind,
    StageRun,
    StageSpecification,
    StageState,
    asset_id_for_sha256,
    validate_artifact_identifier,
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
LATEST_SCHEMA_VERSION = 5
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


_SCHEMA_MIGRATIONS: tuple[tuple[int, Callable[[sqlite3.Connection], None]], ...] = (
    (1, _migration_1_create_library_schema),
    (2, _migration_2_add_video_display_name),
    (3, _migration_3_add_stage_provenance),
    (4, _migration_4_add_asset_identity),
    (5, _migration_5_add_segment_generations),
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
                and current.stage_kind in SEGMENT_STAGE_KINDS
            ):
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
