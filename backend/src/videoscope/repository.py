from __future__ import annotations

from collections.abc import Iterable, Sequence
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
from types import TracebackType
from typing import Any, Callable
from uuid import uuid4
from weakref import WeakValueDictionary

from videoscope.artifacts import (
    ArtifactGCAttempt,
    ArtifactGCJob,
    ArtifactGCOutcome,
    ArtifactGCQuarantine,
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
    TextVectorBuildQuarantine,
    TextVectorBuildRecoveryReport,
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
from videoscope.jobs import (
    JobFenceError,
    JobKind,
    JobState,
    JobTransitionError,
    PriorVideoState,
    VideoIndexIntent,
    VideoIndexJob,
    VideoIndexPlanSnapshot,
    advance_job,
    cancel_job,
    complete_job,
    fail_job,
    request_job_cancellation,
    retry_job,
    start_job,
    video_index_idempotency_hash,
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
EXTERNAL_INDEX_STAGE_KINDS = (
    StageKind.VISUAL_DENSE,
    StageKind.LIGHTHOUSE,
)
_VIDEO_INDEX_STAGE_KINDS = (
    StageKind.SCENES,
    StageKind.SPEECH,
    StageKind.OCR,
    StageKind.OBJECTS,
    StageKind.TEXT_VECTORS,
    StageKind.VISUAL_DENSE,
    StageKind.LIGHTHOUSE,
)
_VIDEO_INDEX_COMPLETION_STAGE_STATES = frozenset(
    {
        StageState.COMPLETE,
        StageState.FAILED,
        StageState.NOT_CONFIGURED,
    }
)
LATEST_SCHEMA_VERSION = 9
_DATABASE_INITIALIZE_LOCK = Lock()
_ASSET_IDENTITY_LOCKS_GUARD = Lock()
_ASSET_IDENTITY_LOCKS: WeakValueDictionary[tuple[str, str], Any] = WeakValueDictionary()
_JOURNAL_MODE_RETRIES = 8
ASSET_HASH_CHUNK_SIZE = 1024 * 1024
_VIDEO_THUMBNAIL_UNCHANGED = object()
ARTIFACT_GC_MAX_LEASE_SECONDS = 300
ARTIFACT_GC_V7_MAX_ATTEMPTS = 5
ARTIFACT_GC_RETRY_BASE_SECONDS = 5
ARTIFACT_GC_RETRY_MAX_SECONDS = 300
ARTIFACT_GC_BACKOFF_LEVEL_MAX = 7
ARTIFACT_GC_SCAN_LIMIT_MAX = 64
ARTIFACT_GC_AUDIT_RETENTION_ATTEMPTS = 256
REPOSITORY_BATCH_LIMIT_MAX = 1_000
VIDEO_INDEX_JOB_SCAN_LIMIT_MAX = 64
VIDEO_INDEX_PLAN_JSON_MAX_BYTES = 1024 * 1024
EXTERNAL_INDEX_DESCRIPTOR_JSON_MAX_BYTES = 16 * 1024
_EXTERNAL_GENERATION_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ARTIFACT_GC_TRANSIENT_LEGACY_ERROR_CODES = (
    "artifact_gc_storage_unavailable",
    "artifact_gc_lease_expired",
    "artifact_gc_attempts_exhausted",
)
_READ_ONLY_REQUIRED_TABLES = frozenset(
    {
        "active_segment_generations",
        "active_external_index_generations",
        "active_text_vector_generations",
        "artifact_gc_attempts",
        "artifact_gc_jobs",
        "artifact_gc_quarantines",
        "assets",
        "external_index_generations",
        "segment_generations",
        "segments",
        "stage_runs",
        "stage_specifications",
        "text_vector_build_inputs",
        "text_vector_builds",
        "text_vector_generation_inputs",
        "text_vector_generation_tombstones",
        "text_vector_generations",
        "text_vector_index_specifications",
        "videos",
        "video_index_jobs",
        "video_index_job_quarantines",
    }
)
_READ_ONLY_REQUIRED_INDEXES = frozenset(
    {
        "idx_artifact_gc_jobs_ready",
        "idx_external_index_generations_video_stage",
        "idx_segment_generations_video_stage",
        "idx_segments_generation",
        "idx_segments_modality",
        "idx_segments_video_time",
        "idx_stage_runs_state",
        "idx_stage_runs_video_spec",
        "idx_text_vector_builds_expiry",
        "idx_text_vector_generations_video_completed",
        "idx_videos_asset_id",
        "idx_stage_runs_job_id",
        "idx_video_index_jobs_active_idempotency",
        "idx_video_index_jobs_active_video",
        "idx_video_index_jobs_ready",
        "idx_video_index_jobs_retry_child",
        "idx_video_index_jobs_video_history",
        "idx_video_index_job_quarantines_job_sequence",
    }
)
_READ_ONLY_REQUIRED_TRIGGERS = frozenset(
    {
        "artifact_gc_attempts_bounded_delete",
        "artifact_gc_attempts_identity_immutable_update",
        "artifact_gc_attempts_terminal_immutable_update",
        "artifact_gc_jobs_attempt_fence",
        "artifact_gc_jobs_backoff_monotonic",
        "artifact_gc_jobs_immutable_delete",
        "artifact_gc_jobs_reject_committed_insert",
        "artifact_gc_jobs_reject_committed_update",
        "artifact_gc_jobs_state_transition",
        "artifact_gc_jobs_target_immutable_update",
        "artifact_gc_jobs_terminal_immutable_update",
        "artifact_gc_quarantines_immutable_delete",
        "artifact_gc_quarantines_immutable_update",
        "assets_immutable_update",
        "active_segment_generations_job_owner_insert",
        "active_segment_generations_job_owner_update",
        "active_text_vector_generations_job_owner_insert",
        "active_text_vector_generations_job_owner_update",
        "active_external_index_generations_job_owner_insert",
        "active_external_index_generations_job_owner_update",
        "external_index_generation_run_identity_insert",
        "external_index_generations_immutable_delete",
        "external_index_generations_immutable_update",
        "generated_segments_generation_immutable",
        "generated_segments_immutable_delete",
        "generated_segments_immutable_update",
        "generated_segments_validate_insert",
        "segment_generation_run_identity_insert",
        "segment_generations_immutable_delete",
        "segment_generations_immutable_update",
        "text_vector_build_run_identity_insert",
        "text_vector_builds_identity_immutable_update",
        "text_vector_builds_quarantine_immutable_update",
        "text_vector_builds_recovery_transition",
        "text_vector_builds_require_generation_tombstone",
        "text_vector_builds_require_terminal_tombstone_delete",
        "text_vector_generation_inputs_immutable_delete",
        "text_vector_generation_inputs_immutable_update",
        "text_vector_generation_run_identity_insert",
        "text_vector_generation_tombstones_immutable_delete",
        "text_vector_generation_tombstones_target_immutable",
        "text_vector_generation_tombstones_valid_transition",
        "text_vector_generations_immutable_delete",
        "text_vector_generations_immutable_update",
        "text_vector_generations_require_building_tombstone",
        "text_vector_index_specifications_immutable_delete",
        "text_vector_index_specifications_immutable_update",
        "videos_asset_id_immutable",
        "stage_runs_job_id_immutable_update",
        "stage_runs_video_index_job_ownership_insert",
        "stage_runs_video_index_job_active_update",
        "stage_runs_video_index_job_identity_insert",
        "video_index_jobs_cancel_request_immutable",
        "video_index_jobs_execution_fence",
        "video_index_jobs_identity_immutable_update",
        "video_index_jobs_immutable_delete",
        "video_index_jobs_progress_monotonic",
        "video_index_jobs_reject_active_stage_completion",
        "video_index_jobs_require_exact_stage_receipts",
        "video_index_jobs_retry_identity_insert",
        "video_index_jobs_source_identity_insert",
        "video_index_jobs_state_transition",
        "video_index_jobs_terminal_immutable_update",
        "video_index_jobs_updated_at_monotonic",
        "video_index_job_quarantines_immutable_delete",
        "video_index_job_quarantines_immutable_update",
    }
)
_READ_ONLY_REQUIRED_COLUMNS = {
    "videos": frozenset(
        {
            "asset_id",
            "created_at",
            "display_name",
            "id",
            "media_path",
            "size_bytes",
            "stored_name",
            "updated_at",
        }
    ),
    "segments": frozenset(
        {
            "confidence",
            "end",
            "generation_id",
            "id",
            "metadata_json",
            "modality",
            "start",
            "text",
            "video_id",
        }
    ),
    "assets": frozenset({"asset_id", "created_at", "sha256", "size_bytes"}),
    "segment_generations": frozenset(
        {
            "completed_at",
            "generation_id",
            "run_id",
            "source_sha256",
            "specification_hash",
            "stage_kind",
            "updates_video_thumbnail",
            "video_thumbnail_path",
            "video_id",
        }
    ),
    "text_vector_generations": frozenset(
        {
            "collection_name",
            "generation_id",
            "index_specification_hash",
            "run_id",
            "source_sha256",
            "specification_hash",
            "video_id",
        }
    ),
    "external_index_generations": frozenset(
        {
            "completed_at",
            "descriptor_json",
            "descriptor_sha256",
            "generation_id",
            "artifact_specification_hash",
            "run_id",
            "source_sha256",
            "specification_hash",
            "stage_kind",
            "video_id",
        }
    ),
    "active_external_index_generations": frozenset(
        {"activated_at", "generation_id", "stage_kind", "video_id"}
    ),
    "text_vector_builds": frozenset(
        {
            "generation_id",
            "quarantined_at",
            "recovery_error_code",
            "recovery_state",
            "run_id",
            "source_sha256",
            "video_id",
        }
    ),
    "artifact_gc_jobs": frozenset(
        {
            "attempt",
            "available_at",
            "backoff_level",
            "generation_id",
            "job_id",
            "lease_expires_at",
            "lease_token",
            "state",
            "worker_id",
        }
    ),
    "artifact_gc_attempts": frozenset(
        {
            "attempt",
            "claimed_at",
            "job_id",
            "lease_expires_at",
            "lease_token",
            "worker_id",
        }
    ),
    "artifact_gc_quarantines": frozenset(
        {"error_code", "quarantined_at", "sequence"}
    ),
    "stage_runs": frozenset({"job_id"}),
    "video_index_jobs": frozenset(
        {
            "attempt",
            "cancel_requested_at",
            "created_at",
            "error_code",
            "execution_token",
            "finished_at",
            "idempotency_hash",
            "intent",
            "job_id",
            "kind",
            "plan_hash",
            "plan_json",
            "prior_video_error_code",
            "prior_video_progress",
            "prior_video_stage",
            "prior_video_status",
            "progress",
            "retry_of_job_id",
            "source_sha256",
            "stage",
            "started_at",
            "state",
            "updated_at",
            "video_id",
        }
    ),
    "video_index_job_quarantines": frozenset(
        {"error_code", "job_sequence", "quarantined_at", "sequence"}
    ),
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _validate_repository_batch_limit(limit: object, *, field_name: str) -> int:
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= REPOSITORY_BATCH_LIMIT_MAX
    ):
        raise ValueError(
            f"{field_name} must be between 1 and {REPOSITORY_BATCH_LIMIT_MAX}"
        )
    return limit


def _validate_gc_lease_seconds(lease_seconds: object) -> int:
    if (
        isinstance(lease_seconds, bool)
        or not isinstance(lease_seconds, int)
        or not 1 <= lease_seconds <= ARTIFACT_GC_MAX_LEASE_SECONDS
    ):
        raise ValueError(
            "artifact GC lease must be between "
            f"1 and {ARTIFACT_GC_MAX_LEASE_SECONDS} seconds"
        )
    return lease_seconds


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


def _validate_latest_read_only_schema(connection: sqlite3.Connection) -> None:
    version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if version < LATEST_SCHEMA_VERSION:
        raise RuntimeError(
            "database schema is older than this VideoScope build: "
            f"{version} < {LATEST_SCHEMA_VERSION}"
        )
    if version > LATEST_SCHEMA_VERSION:
        raise RuntimeError(
            "database schema is newer than this VideoScope build: "
            f"{version} > {LATEST_SCHEMA_VERSION}"
        )
    integrity = connection.execute("PRAGMA quick_check(1)").fetchone()
    if integrity is None or integrity[0] != "ok":
        raise RuntimeError("database integrity validation failed")
    objects = {
        kind: {
            str(row["name"])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = ?",
                (kind,),
            ).fetchall()
        }
        for kind in ("table", "index", "trigger")
    }
    if (
        not _READ_ONLY_REQUIRED_TABLES <= objects["table"]
        or not _READ_ONLY_REQUIRED_INDEXES <= objects["index"]
        or not _READ_ONLY_REQUIRED_TRIGGERS <= objects["trigger"]
    ):
        raise RuntimeError("database schema is incomplete or corrupt")
    for table_name, required_columns in _READ_ONLY_REQUIRED_COLUMNS.items():
        columns = {
            str(row["name"])
            for row in connection.execute(
                f"PRAGMA table_info({table_name})"
            ).fetchall()
        }
        if not required_columns <= columns:
            raise RuntimeError("database schema is incomplete or corrupt")


def _validate_read_only_database_path(path: Path) -> Path:
    absolute = Path(os.path.abspath(path))
    try:
        final = _stat_retained_volume_path(absolute)
        if final is None:
            candidates = tuple(reversed((absolute, *absolute.parents)))
            for candidate in candidates:
                metadata = os.lstat(candidate)
                if stat.S_ISLNK(metadata.st_mode):
                    raise RuntimeError(
                        "read-only database path must not contain a symbolic link"
                    )
            final = os.stat(absolute, follow_symlinks=False)
    except RuntimeError:
        raise
    except OSError as error:
        raise RuntimeError("read-only database is unavailable") from error
    if not stat.S_ISREG(final.st_mode) or final.st_nlink != 1:
        raise RuntimeError("read-only database must be a managed regular file")
    return absolute


def _read_stable_sqlite_header(path: Path) -> bytes:
    try:
        observed = os.stat(path, follow_symlinks=False)
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as error:
        raise RuntimeError("read-only database is unavailable") from error
    try:
        opened = os.fstat(descriptor)
        header = os.read(descriptor, 100)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        current = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise RuntimeError("read-only database changed while opening") from error
    if (
        not stat.S_ISREG(opened.st_mode)
        or len(header) != 100
        or header[:16] != b"SQLite format 3\x00"
    ):
        raise RuntimeError("read-only database is unavailable or corrupt")
    identities = tuple(
        _file_identity(value)
        for value in (observed, opened, after, current)
    )
    if len(set(identities)) != 1:
        raise RuntimeError("read-only database changed while opening")
    return header


def _validate_existing_sqlite_sidecar(path: Path, *, label: str) -> None:
    try:
        sidecar_stat = _stat_retained_volume_path(path)
        if sidecar_stat is None:
            for candidate in reversed((path, *path.parents)):
                metadata = os.lstat(candidate)
                if stat.S_ISLNK(metadata.st_mode):
                    raise RuntimeError(
                        f"read-only database {label} must be a managed regular file"
                    )
            sidecar_stat = os.stat(path, follow_symlinks=False)
    except FileNotFoundError as error:
        raise RuntimeError(f"read-only database {label} is unavailable") from error
    except OSError as error:
        raise RuntimeError(f"read-only database {label} is unavailable") from error
    if not stat.S_ISREG(sidecar_stat.st_mode) or sidecar_stat.st_nlink != 1:
        raise RuntimeError(
            f"read-only database {label} must be a managed regular file"
        )


def _sqlite_sidecar_exists(path: Path, *, label: str) -> bool:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError as error:
        raise RuntimeError(
            f"read-only database {label} could not be inspected"
        ) from error
    return True


def _require_existing_wal_sidecars(database_path: Path) -> bool:
    """Validate SQLite journal state and return whether immutable open is required."""

    header = _read_stable_sqlite_header(database_path)
    if header[18] not in {1, 2} or header[19] not in {1, 2}:
        raise RuntimeError("read-only database is unavailable or corrupt")
    wal_path = Path(f"{database_path}-wal")
    shm_path = Path(f"{database_path}-shm")
    wal_exists = _sqlite_sidecar_exists(wal_path, label="WAL")
    shm_exists = _sqlite_sidecar_exists(shm_path, label="SHM")
    is_wal = 2 in {header[18], header[19]}
    if not is_wal:
        if _sqlite_sidecar_exists(
            Path(f"{database_path}-journal"),
            label="rollback journal",
        ):
            raise RuntimeError(
                "read-only DELETE database has an existing rollback journal"
            )
        if wal_exists or shm_exists:
            raise RuntimeError(
                "read-only DELETE database has unexpected WAL sidecars"
            )
        return False
    if wal_exists != shm_exists:
        raise RuntimeError(
            "read-only database WAL and SHM sidecars must either both exist or both be absent"
        )
    if not wal_exists:
        # A cleanly closed WAL database is fully checkpointed into the main file.
        # Immutable mode prevents SQLite from recreating sidecars while reading it.
        return True
    _validate_existing_sqlite_sidecar(
        wal_path,
        label="WAL",
    )
    _validate_existing_sqlite_sidecar(
        shm_path,
        label="SHM",
    )
    return False


def _reject_symbolic_link_components(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    try:
        retained = _stat_retained_volume_path(absolute)
    except (OSError, RuntimeError, ValueError) as error:
        raise AssetIdentityError(
            "asset media must be a managed regular file"
        ) from error
    if retained is not None:
        return
    for candidate in reversed((absolute, *absolute.parents)):
        try:
            metadata = os.lstat(candidate)
        except OSError as error:
            raise AssetIdentityError(
                "asset media must be a managed regular file"
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise AssetIdentityError("asset media must be a managed regular file")


def _retained_volume_components(
    path: Path,
) -> tuple[int, int, tuple[str, ...]] | None:
    absolute = Path(os.path.abspath(path))
    parts = absolute.parts
    if len(parts) < 4 or parts[0] != os.sep or parts[1] != ".vol":
        return None
    device_text, inode_text = parts[2], parts[3]
    if (
        not device_text.isdecimal()
        or not inode_text.isdecimal()
        or str(int(device_text)) != device_text
        or str(int(inode_text)) != inode_text
        or int(device_text) <= 0
        or int(inode_text) <= 0
    ):
        raise RuntimeError("retained volume path is invalid")
    descendants = tuple(parts[4:])
    if any(
        not component
        or component in {".", ".."}
        or "/" in component
        or "\x00" in component
        for component in descendants
    ):
        raise RuntimeError("retained volume path is invalid")
    return int(device_text), int(inode_text), descendants


def _stat_retained_volume_path(path: Path) -> os.stat_result | None:
    retained = _retained_volume_components(path)
    if retained is None:
        return None
    device, inode, descendants = retained
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    if not getattr(os, "O_DIRECTORY", 0) or not getattr(os, "O_NOFOLLOW", 0):
        raise RuntimeError("retained volume paths are unsupported")
    descriptor = os.open(f"/.vol/{device}/{inode}", flags)
    try:
        root = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(root.st_mode)
            or root.st_dev != device
            or root.st_ino != inode
        ):
            raise RuntimeError("retained volume root identity changed")
        if not descendants:
            return root
        for index, component in enumerate(descendants):
            observed = os.stat(
                component,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
            if index == len(descendants) - 1:
                if stat.S_ISLNK(observed.st_mode):
                    raise RuntimeError("retained volume path contains a symbolic link")
                return observed
            child = os.open(component, flags, dir_fd=descriptor)
            try:
                opened = os.fstat(child)
                current = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                if not (
                    stat.S_ISDIR(observed.st_mode)
                    and _file_identity(observed)
                    == _file_identity(opened)
                    == _file_identity(current)
                ):
                    raise RuntimeError("retained volume path changed while opening")
            except BaseException:
                os.close(child)
                raise
            os.close(descriptor)
            descriptor = child
    finally:
        os.close(descriptor)
    raise RuntimeError("retained volume path is invalid")


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
            JOIN videos AS videos ON videos.id = runs.video_id
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


def _migration_7_add_crash_safe_artifact_gc(connection: sqlite3.Connection) -> None:
    tombstone_exists = connection.execute(
        """
        SELECT 1 FROM sqlite_master
        WHERE type = 'table' AND name = 'text_vector_generation_tombstones'
        """
    ).fetchone() is not None
    gc_columns = {
        row[1]
        for row in connection.execute("PRAGMA table_info(artifact_gc_jobs)").fetchall()
    }
    v7_gc_columns = {
        "max_attempts",
        "available_at",
        "worker_id",
        "lease_token",
        "lease_expires_at",
        "error_code",
    }
    gc_is_v7 = v7_gc_columns <= gc_columns
    gc_is_later = "backoff_level" in gc_columns
    if tombstone_exists or gc_is_v7 or gc_is_later:
        required_tombstone_columns = {
            "generation_id",
            "index_specification_hash",
            "collection_name",
            "lifecycle_state",
            "created_at",
            "updated_at",
        }
        tombstone_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(text_vector_generation_tombstones)"
            ).fetchall()
        }
        required_triggers = {
            "text_vector_generation_tombstones_immutable_delete",
            "text_vector_generation_tombstones_valid_transition",
            "text_vector_generation_tombstones_target_immutable",
            "text_vector_builds_require_generation_tombstone",
            "text_vector_builds_require_terminal_tombstone_delete",
            "text_vector_builds_identity_immutable_update",
            "text_vector_generations_require_building_tombstone",
            "artifact_gc_jobs_reject_committed_insert",
            "artifact_gc_jobs_reject_committed_update",
            "artifact_gc_jobs_target_immutable_update",
            "artifact_gc_jobs_immutable_delete",
        }
        persisted_triggers = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            ).fetchall()
        }
        if (
            not tombstone_exists
            or not (gc_is_v7 or gc_is_later)
            or not required_tombstone_columns <= tombstone_columns
            or not required_triggers <= persisted_triggers
        ):
            raise ValueError("artifact GC schema migration is incomplete or corrupt")
        return

    collision = connection.execute(
        """
        WITH identities(generation_id, source) AS (
            SELECT generation_id, 'build' FROM text_vector_builds
            UNION ALL
            SELECT generation_id, 'generation' FROM text_vector_generations
            UNION ALL
            SELECT DISTINCT generation_id, 'gc' FROM artifact_gc_jobs
        )
        SELECT generation_id
        FROM identities
        GROUP BY generation_id
        HAVING COUNT(DISTINCT source) > 1
        LIMIT 1
        """
    ).fetchone()
    if collision is not None:
        raise ValueError("text vector generation identity history is corrupt")
    invalid_attempt = connection.execute(
        "SELECT job_id FROM artifact_gc_jobs WHERE attempt > 16 LIMIT 1"
    ).fetchone()
    if invalid_attempt is not None:
        raise ValueError("artifact GC attempt history is corrupt")
    duplicate_gc = connection.execute(
        """
        SELECT generation_id
        FROM artifact_gc_jobs
        GROUP BY generation_id
        HAVING COUNT(*) > 1
        LIMIT 1
        """
    ).fetchone()
    if duplicate_gc is not None:
        raise ValueError("artifact GC generation identity history is corrupt")
    conflicting_gc_state = connection.execute(
        """
        SELECT generation_id
        FROM artifact_gc_jobs
        GROUP BY generation_id
        HAVING COUNT(
            DISTINCT CASE
                WHEN state IN ('pending', 'running') THEN 'gc_pending'
                WHEN state = 'complete' THEN 'gc_complete'
                ELSE 'gc_failed'
            END
        ) > 1
        LIMIT 1
        """
    ).fetchone()
    if conflicting_gc_state is not None:
        raise ValueError("artifact GC lifecycle history is corrupt")

    connection.execute(
        """
        CREATE TABLE text_vector_generation_tombstones (
            generation_id TEXT PRIMARY KEY,
            index_specification_hash TEXT NOT NULL
                REFERENCES text_vector_index_specifications(specification_hash),
            collection_name TEXT NOT NULL,
            lifecycle_state TEXT NOT NULL CHECK(
                lifecycle_state IN (
                    'building', 'committed', 'gc_pending', 'gc_complete', 'gc_failed'
                )
            ),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        INSERT INTO text_vector_generation_tombstones (
            generation_id, index_specification_hash, collection_name,
            lifecycle_state, created_at, updated_at
        )
        SELECT generation_id, index_specification_hash, collection_name,
            'building', reserved_at, heartbeat_at
        FROM text_vector_builds
        """
    )
    connection.execute(
        """
        INSERT INTO text_vector_generation_tombstones (
            generation_id, index_specification_hash, collection_name,
            lifecycle_state, created_at, updated_at
        )
        SELECT generation_id, index_specification_hash, collection_name,
            'committed', completed_at, completed_at
        FROM text_vector_generations
        """
    )
    connection.execute(
        """
        INSERT INTO text_vector_generation_tombstones (
            generation_id, index_specification_hash, collection_name,
            lifecycle_state, created_at, updated_at
        )
        SELECT
            generation_id,
            MIN(index_specification_hash),
            MIN(collection_name),
            CASE
                WHEN MIN(state) = 'complete' AND MAX(state) = 'complete'
                    THEN 'gc_complete'
                WHEN MIN(state) = 'failed' AND MAX(state) = 'failed'
                    THEN 'gc_failed'
                ELSE 'gc_pending'
            END,
            MIN(created_at),
            MAX(updated_at)
        FROM artifact_gc_jobs
        GROUP BY generation_id
        """
    )

    connection.execute("DROP INDEX IF EXISTS idx_artifact_gc_jobs_state")
    connection.execute("ALTER TABLE artifact_gc_jobs RENAME TO artifact_gc_jobs_v6")
    connection.execute(
        """
        CREATE TABLE artifact_gc_jobs (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL UNIQUE,
            artifact_kind TEXT NOT NULL CHECK(artifact_kind = 'text_vectors'),
            generation_id TEXT NOT NULL UNIQUE
                REFERENCES text_vector_generation_tombstones(generation_id),
            index_specification_hash TEXT NOT NULL,
            collection_name TEXT NOT NULL,
            reason TEXT NOT NULL,
            state TEXT NOT NULL CHECK(state IN ('pending', 'running', 'complete', 'failed')),
            attempt INTEGER NOT NULL CHECK(attempt >= 0),
            max_attempts INTEGER NOT NULL CHECK(max_attempts BETWEEN 1 AND 16),
            available_at TEXT NOT NULL,
            worker_id TEXT,
            lease_token TEXT,
            lease_expires_at TEXT,
            error_code TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            CHECK(attempt <= max_attempts),
            CHECK(
                (state = 'running' AND worker_id IS NOT NULL
                    AND lease_token IS NOT NULL AND lease_expires_at IS NOT NULL
                    AND error_code IS NULL)
                OR
                (state != 'running' AND worker_id IS NULL
                    AND lease_token IS NULL AND lease_expires_at IS NULL)
            ),
            CHECK(state != 'complete' OR error_code IS NULL),
            CHECK(state != 'failed' OR error_code IS NOT NULL),
            UNIQUE(artifact_kind, generation_id, index_specification_hash, reason)
        )
        """
    )
    connection.execute(
        """
        INSERT INTO artifact_gc_jobs (
            sequence, job_id, artifact_kind, generation_id,
            index_specification_hash, collection_name, reason, state, attempt,
            max_attempts, available_at, worker_id, lease_token, lease_expires_at,
            error_code, created_at, updated_at
        )
        SELECT
            sequence, job_id, artifact_kind, generation_id,
            index_specification_hash, collection_name, reason,
            CASE WHEN state = 'running' THEN 'pending' ELSE state END,
            attempt,
            CASE
                WHEN attempt > ? THEN attempt
                ELSE ?
            END,
            updated_at,
            NULL, NULL, NULL,
            CASE WHEN state = 'failed' THEN 'legacy_gc_failed' ELSE NULL END,
            created_at, updated_at
        FROM artifact_gc_jobs_v6
        ORDER BY sequence
        """,
        (ARTIFACT_GC_V7_MAX_ATTEMPTS, ARTIFACT_GC_V7_MAX_ATTEMPTS),
    )
    connection.execute("DROP TABLE artifact_gc_jobs_v6")
    connection.execute(
        """
        CREATE INDEX idx_artifact_gc_jobs_ready
        ON artifact_gc_jobs(state, available_at, lease_expires_at, sequence)
        """
    )
    connection.execute(
        """
        CREATE TRIGGER text_vector_generation_tombstones_immutable_delete
        BEFORE DELETE ON text_vector_generation_tombstones
        BEGIN
            SELECT RAISE(ABORT, 'text vector generation tombstone is immutable');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER text_vector_generation_tombstones_valid_transition
        BEFORE UPDATE OF lifecycle_state ON text_vector_generation_tombstones
        WHEN NOT (
            NEW.lifecycle_state = OLD.lifecycle_state
            OR (OLD.lifecycle_state = 'building'
                AND NEW.lifecycle_state IN ('committed', 'gc_pending'))
            OR (OLD.lifecycle_state = 'gc_pending'
                AND NEW.lifecycle_state IN ('gc_complete', 'gc_failed'))
        )
        BEGIN
            SELECT RAISE(ABORT, 'invalid text vector generation lifecycle transition');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER text_vector_generation_tombstones_target_immutable
        BEFORE UPDATE OF
            generation_id, index_specification_hash, collection_name, created_at
        ON text_vector_generation_tombstones
        BEGIN
            SELECT RAISE(ABORT, 'text vector generation tombstone target is immutable');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER text_vector_builds_require_generation_tombstone
        BEFORE INSERT ON text_vector_builds
        WHEN NOT EXISTS (
            SELECT 1 FROM text_vector_generation_tombstones
            WHERE generation_id = NEW.generation_id
              AND lifecycle_state = 'building'
              AND index_specification_hash = NEW.index_specification_hash
              AND collection_name = NEW.collection_name
        )
        BEGIN
            SELECT RAISE(ABORT, 'text vector build generation identity is not reserved');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER text_vector_generations_require_building_tombstone
        BEFORE INSERT ON text_vector_generations
        WHEN NOT EXISTS (
            SELECT 1 FROM text_vector_generation_tombstones
            WHERE generation_id = NEW.generation_id
              AND lifecycle_state = 'building'
              AND index_specification_hash = NEW.index_specification_hash
              AND collection_name = NEW.collection_name
        )
        BEGIN
            SELECT RAISE(ABORT, 'text vector generation identity is not building');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER text_vector_builds_require_terminal_tombstone_delete
        BEFORE DELETE ON text_vector_builds
        WHEN EXISTS (
            SELECT 1 FROM text_vector_generation_tombstones
            WHERE generation_id = OLD.generation_id AND lifecycle_state = 'building'
        )
        BEGIN
            SELECT RAISE(ABORT, 'text vector build must be terminal before deletion');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER text_vector_builds_identity_immutable_update
        BEFORE UPDATE OF
            generation_id, run_id, video_id, stage_specification_hash,
            source_sha256, index_specification_hash, collection_name,
            expected_previous_generation_id, input_manifest_sha256,
            point_manifest_sha256, point_count, reserved_at
        ON text_vector_builds
        BEGIN
            SELECT RAISE(ABORT, 'text vector build identity is immutable');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER artifact_gc_jobs_reject_committed_insert
        BEFORE INSERT ON artifact_gc_jobs
        WHEN EXISTS (
            SELECT 1 FROM text_vector_generations
            WHERE generation_id = NEW.generation_id
        ) OR NOT EXISTS (
            SELECT 1 FROM text_vector_generation_tombstones
            WHERE generation_id = NEW.generation_id
              AND lifecycle_state = 'gc_pending'
              AND index_specification_hash = NEW.index_specification_hash
              AND collection_name = NEW.collection_name
        )
        BEGIN
            SELECT RAISE(ABORT, 'artifact GC cannot target a committed generation');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER artifact_gc_jobs_reject_committed_update
        BEFORE UPDATE ON artifact_gc_jobs
        WHEN EXISTS (
            SELECT 1 FROM text_vector_generations
            WHERE generation_id = NEW.generation_id
        )
        BEGIN
            SELECT RAISE(ABORT, 'artifact GC cannot target a committed generation');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER artifact_gc_jobs_target_immutable_update
        BEFORE UPDATE OF
            artifact_kind, generation_id, index_specification_hash,
            collection_name, reason, max_attempts, created_at
        ON artifact_gc_jobs
        BEGIN
            SELECT RAISE(ABORT, 'artifact GC target is immutable');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER artifact_gc_jobs_immutable_delete
        BEFORE DELETE ON artifact_gc_jobs
        BEGIN
            SELECT RAISE(ABORT, 'artifact GC job is immutable');
        END
        """
    )


def _create_artifact_gc_attempts_bounded_delete_trigger(
    connection: sqlite3.Connection,
) -> None:
    connection.execute(
        f"""
        CREATE TRIGGER artifact_gc_attempts_bounded_delete
        BEFORE DELETE ON artifact_gc_attempts
        WHEN OLD.finished_at IS NULL
          OR OLD.outcome IS NULL
          OR NOT EXISTS (
              SELECT 1 FROM artifact_gc_jobs AS jobs
              WHERE jobs.job_id = OLD.job_id
          )
          OR OLD.attempt > (
              SELECT jobs.attempt - {ARTIFACT_GC_AUDIT_RETENTION_ATTEMPTS}
              FROM artifact_gc_jobs AS jobs
              WHERE jobs.job_id = OLD.job_id
          )
        BEGIN
            SELECT RAISE(ABORT, 'artifact GC recent or open attempt is immutable');
        END
        """
    )


def _prune_artifact_gc_attempt_audit(
    connection: sqlite3.Connection,
    *,
    job_id: str | None = None,
) -> None:
    scope = "" if job_id is None else "AND job_id = ?"
    parameters: tuple[object, ...] = () if job_id is None else (job_id,)
    connection.execute(
        f"""
        DELETE FROM artifact_gc_attempts
        WHERE finished_at IS NOT NULL
          AND outcome IS NOT NULL
          AND attempt <= COALESCE((
              SELECT jobs.attempt - {ARTIFACT_GC_AUDIT_RETENTION_ATTEMPTS}
              FROM artifact_gc_jobs AS jobs
              WHERE jobs.job_id = artifact_gc_attempts.job_id
          ), -1)
          {scope}
        """,
        parameters,
    )


def _migration_8_add_recoverable_artifact_gc(connection: sqlite3.Connection) -> None:
    gc_columns = {
        row[1]
        for row in connection.execute("PRAGMA table_info(artifact_gc_jobs)").fetchall()
    }
    build_columns = {
        row[1]
        for row in connection.execute("PRAGMA table_info(text_vector_builds)").fetchall()
    }
    table_names = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    v8_markers_present = any(
        (
            "backoff_level" in gc_columns,
            "recovery_state" in build_columns,
            "artifact_gc_attempts" in table_names,
            "artifact_gc_quarantines" in table_names,
        )
    )
    if v8_markers_present:
        required_build_columns = {
            "recovery_state",
            "recovery_error_code",
            "quarantined_at",
        }
        required_tables = {"artifact_gc_attempts", "artifact_gc_quarantines"}
        required_triggers = {
            "text_vector_generation_tombstones_immutable_delete",
            "text_vector_generation_tombstones_valid_transition",
            "text_vector_generation_tombstones_target_immutable",
            "text_vector_builds_require_generation_tombstone",
            "text_vector_generations_require_building_tombstone",
            "text_vector_builds_require_terminal_tombstone_delete",
            "text_vector_builds_identity_immutable_update",
            "artifact_gc_jobs_reject_committed_insert",
            "artifact_gc_jobs_reject_committed_update",
            "artifact_gc_jobs_target_immutable_update",
            "artifact_gc_jobs_immutable_delete",
            "text_vector_builds_recovery_transition",
            "text_vector_builds_quarantine_immutable_update",
            "artifact_gc_attempts_bounded_delete",
            "artifact_gc_attempts_identity_immutable_update",
            "artifact_gc_attempts_terminal_immutable_update",
            "artifact_gc_quarantines_immutable_update",
            "artifact_gc_quarantines_immutable_delete",
            "artifact_gc_jobs_state_transition",
            "artifact_gc_jobs_attempt_fence",
            "artifact_gc_jobs_backoff_monotonic",
            "artifact_gc_jobs_terminal_immutable_update",
        }
        persisted_triggers = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            ).fetchall()
        }
        if (
            "backoff_level" not in gc_columns
            or "max_attempts" in gc_columns
            or not required_build_columns <= build_columns
            or not required_tables <= table_names
        ):
            raise ValueError("recoverable artifact GC migration is incomplete or corrupt")
        legacy_delete_trigger = "artifact_gc_attempts_immutable_delete"
        bounded_delete_trigger = "artifact_gc_attempts_bounded_delete"
        if legacy_delete_trigger in persisted_triggers:
            connection.execute(f"DROP TRIGGER {legacy_delete_trigger}")
            persisted_triggers.remove(legacy_delete_trigger)
            if bounded_delete_trigger not in persisted_triggers:
                _create_artifact_gc_attempts_bounded_delete_trigger(connection)
                persisted_triggers.add(bounded_delete_trigger)
            _prune_artifact_gc_attempt_audit(connection)
        if not required_triggers <= persisted_triggers:
            raise ValueError("recoverable artifact GC migration is incomplete or corrupt")
        return

    invalid_running = connection.execute(
        """
        SELECT job_id FROM artifact_gc_jobs
        WHERE state = 'running' AND attempt < 1
        LIMIT 1
        """
    ).fetchone()
    if invalid_running is not None:
        raise ValueError("running artifact GC attempt history is corrupt")

    connection.execute(
        """
        ALTER TABLE text_vector_builds
        ADD COLUMN recovery_state TEXT NOT NULL DEFAULT 'active'
            CHECK(recovery_state IN ('active', 'quarantined'))
        """
    )
    connection.execute(
        "ALTER TABLE text_vector_builds ADD COLUMN recovery_error_code TEXT"
    )
    connection.execute(
        "ALTER TABLE text_vector_builds ADD COLUMN quarantined_at TEXT"
    )

    for trigger_name in (
        "artifact_gc_jobs_reject_committed_insert",
        "artifact_gc_jobs_reject_committed_update",
        "artifact_gc_jobs_target_immutable_update",
        "artifact_gc_jobs_immutable_delete",
        "text_vector_generation_tombstones_valid_transition",
    ):
        connection.execute(f"DROP TRIGGER {trigger_name}")
    connection.execute("DROP INDEX idx_artifact_gc_jobs_ready")
    connection.execute("ALTER TABLE artifact_gc_jobs RENAME TO artifact_gc_jobs_v7")
    connection.execute(
        """
        CREATE TABLE artifact_gc_jobs (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL UNIQUE,
            artifact_kind TEXT NOT NULL CHECK(artifact_kind = 'text_vectors'),
            generation_id TEXT NOT NULL UNIQUE
                REFERENCES text_vector_generation_tombstones(generation_id),
            index_specification_hash TEXT NOT NULL,
            collection_name TEXT NOT NULL,
            reason TEXT NOT NULL,
            state TEXT NOT NULL CHECK(state IN ('pending', 'running', 'complete', 'failed')),
            attempt INTEGER NOT NULL CHECK(attempt >= 0),
            backoff_level INTEGER NOT NULL CHECK(backoff_level BETWEEN 0 AND 7),
            available_at TEXT NOT NULL,
            worker_id TEXT,
            lease_token TEXT,
            lease_expires_at TEXT,
            error_code TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            CHECK(
                (state = 'running' AND worker_id IS NOT NULL
                    AND lease_token IS NOT NULL AND lease_expires_at IS NOT NULL
                    AND error_code IS NULL)
                OR
                (state != 'running' AND worker_id IS NULL
                    AND lease_token IS NULL AND lease_expires_at IS NULL)
            ),
            CHECK(state != 'complete' OR error_code IS NULL),
            CHECK(state != 'failed' OR error_code IS NOT NULL),
            UNIQUE(artifact_kind, generation_id, index_specification_hash, reason)
        )
        """
    )
    transient_placeholders = _sql_values(ARTIFACT_GC_TRANSIENT_LEGACY_ERROR_CODES)
    connection.execute(
        f"""
        INSERT INTO artifact_gc_jobs (
            sequence, job_id, artifact_kind, generation_id,
            index_specification_hash, collection_name, reason, state, attempt,
            backoff_level, available_at, worker_id, lease_token,
            lease_expires_at, error_code, created_at, updated_at
        )
        SELECT
            sequence, job_id, artifact_kind, generation_id,
            index_specification_hash, collection_name, reason,
            CASE
                WHEN state = 'failed' AND error_code IN ({transient_placeholders})
                    THEN 'pending'
                ELSE state
            END,
            attempt,
            MIN(attempt, ?),
            available_at,
            CASE
                WHEN state = 'failed' AND error_code IN ({transient_placeholders})
                    THEN NULL
                ELSE worker_id
            END,
            CASE
                WHEN state = 'failed' AND error_code IN ({transient_placeholders})
                    THEN NULL
                ELSE lease_token
            END,
            CASE
                WHEN state = 'failed' AND error_code IN ({transient_placeholders})
                    THEN NULL
                ELSE lease_expires_at
            END,
            error_code, created_at, updated_at
        FROM artifact_gc_jobs_v7
        ORDER BY sequence
        """,
        (ARTIFACT_GC_BACKOFF_LEVEL_MAX,),
    )
    connection.execute(
        f"""
        UPDATE text_vector_generation_tombstones
        SET lifecycle_state = 'gc_pending', updated_at = (
            SELECT jobs.updated_at FROM artifact_gc_jobs AS jobs
            WHERE jobs.generation_id = text_vector_generation_tombstones.generation_id
        )
        WHERE lifecycle_state = 'gc_failed'
          AND generation_id IN (
            SELECT generation_id FROM artifact_gc_jobs
            WHERE state = 'pending' AND error_code IN ({transient_placeholders})
          )
        """
    )
    connection.execute("DROP TABLE artifact_gc_jobs_v7")
    connection.execute(
        """
        CREATE INDEX idx_artifact_gc_jobs_ready
        ON artifact_gc_jobs(state, available_at, lease_expires_at, sequence)
        """
    )
    connection.execute(
        """
        CREATE TABLE artifact_gc_attempts (
            job_id TEXT NOT NULL REFERENCES artifact_gc_jobs(job_id),
            attempt INTEGER NOT NULL CHECK(attempt >= 1),
            worker_id TEXT NOT NULL,
            lease_token TEXT NOT NULL UNIQUE,
            claimed_at TEXT NOT NULL,
            lease_expires_at TEXT NOT NULL,
            finished_at TEXT,
            outcome TEXT CHECK(
                outcome IS NULL OR outcome IN (
                    'success', 'transient_failure',
                    'permanent_failure', 'lease_expired'
                )
            ),
            error_code TEXT,
            PRIMARY KEY(job_id, attempt),
            CHECK(
                (outcome IS NULL AND finished_at IS NULL AND error_code IS NULL)
                OR
                (outcome = 'success' AND finished_at IS NOT NULL AND error_code IS NULL)
                OR
                (outcome IN ('transient_failure', 'permanent_failure', 'lease_expired')
                    AND finished_at IS NOT NULL AND error_code IS NOT NULL)
            )
        )
        """
    )
    connection.execute(
        """
        INSERT INTO artifact_gc_attempts (
            job_id, attempt, worker_id, lease_token, claimed_at,
            lease_expires_at, finished_at, outcome, error_code
        )
        SELECT
            job_id, attempt, worker_id, lease_token, updated_at,
            lease_expires_at, NULL, NULL, NULL
        FROM artifact_gc_jobs
        WHERE state = 'running'
        """
    )
    connection.execute(
        """
        CREATE TABLE artifact_gc_quarantines (
            sequence INTEGER PRIMARY KEY REFERENCES artifact_gc_jobs(sequence),
            error_code TEXT NOT NULL,
            quarantined_at TEXT NOT NULL
        )
        """
    )

    connection.execute(
        """
        CREATE TRIGGER text_vector_generation_tombstones_valid_transition
        BEFORE UPDATE OF lifecycle_state ON text_vector_generation_tombstones
        WHEN NOT (
            NEW.lifecycle_state = OLD.lifecycle_state
            OR (OLD.lifecycle_state = 'building'
                AND NEW.lifecycle_state IN ('committed', 'gc_pending'))
            OR (OLD.lifecycle_state = 'gc_pending'
                AND NEW.lifecycle_state IN ('gc_complete', 'gc_failed'))
        )
        BEGIN
            SELECT RAISE(ABORT, 'invalid text vector generation lifecycle transition');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER artifact_gc_jobs_reject_committed_insert
        BEFORE INSERT ON artifact_gc_jobs
        WHEN EXISTS (
            SELECT 1 FROM text_vector_generations
            WHERE generation_id = NEW.generation_id
        ) OR NOT EXISTS (
            SELECT 1 FROM text_vector_generation_tombstones
            WHERE generation_id = NEW.generation_id
              AND lifecycle_state = 'gc_pending'
              AND index_specification_hash = NEW.index_specification_hash
              AND collection_name = NEW.collection_name
        )
        BEGIN
            SELECT RAISE(ABORT, 'artifact GC cannot target a committed generation');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER artifact_gc_jobs_reject_committed_update
        BEFORE UPDATE ON artifact_gc_jobs
        WHEN EXISTS (
            SELECT 1 FROM text_vector_generations
            WHERE generation_id = NEW.generation_id
        )
        BEGIN
            SELECT RAISE(ABORT, 'artifact GC cannot target a committed generation');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER artifact_gc_jobs_target_immutable_update
        BEFORE UPDATE OF
            artifact_kind, generation_id, index_specification_hash,
            collection_name, reason, created_at
        ON artifact_gc_jobs
        BEGIN
            SELECT RAISE(ABORT, 'artifact GC target is immutable');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER artifact_gc_jobs_immutable_delete
        BEFORE DELETE ON artifact_gc_jobs
        BEGIN
            SELECT RAISE(ABORT, 'artifact GC job is immutable');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER artifact_gc_jobs_state_transition
        BEFORE UPDATE OF state ON artifact_gc_jobs
        WHEN NOT (
            NEW.state = OLD.state
            OR (OLD.state = 'pending' AND NEW.state = 'running')
            OR (OLD.state = 'running'
                AND NEW.state IN ('pending', 'complete', 'failed'))
        )
        BEGIN
            SELECT RAISE(ABORT, 'invalid artifact GC state transition');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER artifact_gc_jobs_attempt_fence
        BEFORE UPDATE OF attempt ON artifact_gc_jobs
        WHEN NOT (
            NEW.attempt = OLD.attempt
            OR (
                OLD.state = 'pending' AND NEW.state = 'running'
                AND NEW.attempt = OLD.attempt + 1
            )
        )
        BEGIN
            SELECT RAISE(ABORT, 'invalid artifact GC attempt transition');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER artifact_gc_jobs_backoff_monotonic
        BEFORE UPDATE OF backoff_level ON artifact_gc_jobs
        WHEN NEW.backoff_level < OLD.backoff_level
          OR NEW.backoff_level > OLD.backoff_level + 1
        BEGIN
            SELECT RAISE(ABORT, 'invalid artifact GC backoff transition');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER artifact_gc_jobs_terminal_immutable_update
        BEFORE UPDATE ON artifact_gc_jobs
        WHEN OLD.state IN ('complete', 'failed')
        BEGIN
            SELECT RAISE(ABORT, 'terminal artifact GC job is immutable');
        END
        """
    )
    _create_artifact_gc_attempts_bounded_delete_trigger(connection)
    connection.execute(
        """
        CREATE TRIGGER artifact_gc_attempts_identity_immutable_update
        BEFORE UPDATE OF job_id, attempt, worker_id, lease_token, claimed_at
        ON artifact_gc_attempts
        BEGIN
            SELECT RAISE(ABORT, 'artifact GC attempt identity is immutable');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER artifact_gc_attempts_terminal_immutable_update
        BEFORE UPDATE ON artifact_gc_attempts
        WHEN OLD.finished_at IS NOT NULL
        BEGIN
            SELECT RAISE(ABORT, 'finished artifact GC attempt is immutable');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER artifact_gc_quarantines_immutable_update
        BEFORE UPDATE ON artifact_gc_quarantines
        BEGIN
            SELECT RAISE(ABORT, 'artifact GC quarantine is immutable');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER artifact_gc_quarantines_immutable_delete
        BEFORE DELETE ON artifact_gc_quarantines
        BEGIN
            SELECT RAISE(ABORT, 'artifact GC quarantine is immutable');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER text_vector_builds_recovery_transition
        BEFORE UPDATE OF recovery_state ON text_vector_builds
        WHEN NOT (
            NEW.recovery_state = OLD.recovery_state
            OR (OLD.recovery_state = 'active' AND NEW.recovery_state = 'quarantined')
        )
        BEGIN
            SELECT RAISE(ABORT, 'invalid text vector build recovery transition');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER text_vector_builds_quarantine_immutable_update
        BEFORE UPDATE ON text_vector_builds
        WHEN OLD.recovery_state = 'quarantined'
        BEGIN
            SELECT RAISE(ABORT, 'quarantined text vector build is immutable');
        END
        """
    )


def _ensure_stage_run_video_index_job_linkage(
    connection: sqlite3.Connection,
) -> None:
    stage_columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(stage_runs)").fetchall()
    }
    if "job_id" not in stage_columns:
        connection.execute(
            """
            ALTER TABLE stage_runs
            ADD COLUMN job_id TEXT REFERENCES video_index_jobs(job_id)
            """
        )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_stage_runs_job_id ON stage_runs(job_id)"
    )
    connection.execute(
        """
        CREATE TRIGGER IF NOT EXISTS stage_runs_video_index_job_ownership_insert
        BEFORE INSERT ON stage_runs
        WHEN NEW.job_id IS NULL AND EXISTS (
            SELECT 1 FROM video_index_jobs AS jobs
            WHERE jobs.video_id = NEW.video_id
        )
        BEGIN
            SELECT RAISE(ABORT, 'durable video index job must own stage run');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER IF NOT EXISTS stage_runs_video_index_job_identity_insert
        BEFORE INSERT ON stage_runs
        WHEN NEW.job_id IS NOT NULL AND NOT EXISTS (
            SELECT 1 FROM video_index_jobs AS jobs
            WHERE jobs.job_id = NEW.job_id
              AND jobs.video_id = NEW.video_id
              AND jobs.source_sha256 = NEW.source_sha256
              AND jobs.state = 'running'
              AND jobs.cancel_requested_at IS NULL
        )
        BEGIN
            SELECT RAISE(ABORT, 'stage run video index job identity mismatch');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER IF NOT EXISTS stage_runs_job_id_immutable_update
        BEFORE UPDATE OF job_id ON stage_runs
        WHEN NEW.job_id IS NOT OLD.job_id
        BEGIN
            SELECT RAISE(ABORT, 'stage run video index job link is immutable');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER IF NOT EXISTS stage_runs_video_index_job_active_update
        BEFORE UPDATE OF state ON stage_runs
        WHEN OLD.job_id IS NOT NULL
          AND NEW.state NOT IN ('failed', 'cancelled')
          AND NOT EXISTS (
              SELECT 1 FROM video_index_jobs AS jobs
              WHERE jobs.job_id = OLD.job_id
                AND jobs.state = 'running'
                AND jobs.cancel_requested_at IS NULL
          )
        BEGIN
            SELECT RAISE(ABORT, 'stage run video index job is no longer active');
        END
        """
    )


def _ensure_video_index_job_completion_receipts(
    connection: sqlite3.Connection,
) -> None:
    """Reject raw-SQL success without one exact terminal receipt per plan stage."""
    connection.execute(
        """
        CREATE TRIGGER IF NOT EXISTS video_index_jobs_require_exact_stage_receipts
        BEFORE UPDATE OF state ON video_index_jobs
        WHEN NEW.state = 'complete'
          AND NOT EXISTS (
              SELECT 1 FROM stage_runs AS active_runs
              WHERE active_runs.job_id = OLD.job_id
                AND active_runs.state IN ('queued', 'running')
          )
          AND (
              (SELECT COUNT(*) FROM stage_runs AS runs
               WHERE runs.job_id = OLD.job_id) != 7
              OR
              (SELECT COUNT(DISTINCT specifications.stage_kind)
               FROM stage_runs AS runs
               JOIN stage_specifications AS specifications
                 ON specifications.specification_hash = runs.specification_hash
               WHERE runs.job_id = OLD.job_id) != 7
              OR EXISTS (
                  SELECT 1
                  FROM stage_runs AS runs
                  JOIN stage_specifications AS specifications
                    ON specifications.specification_hash = runs.specification_hash
                  WHERE runs.job_id = OLD.job_id
                    AND (
                        runs.video_id IS NOT OLD.video_id
                        OR runs.source_sha256 IS NOT OLD.source_sha256
                        OR runs.state NOT IN ('complete', 'failed', 'not_configured')
                        OR specifications.stage_kind NOT IN (
                            'scenes', 'speech', 'ocr', 'objects', 'text_vectors',
                            'visual_dense', 'lighthouse'
                        )
                        OR specifications.canonical_json IS NOT json_extract(
                            OLD.plan_json,
                            '$.stages.' || specifications.stage_kind
                        )
                    )
              )
          )
        BEGIN
            SELECT RAISE(
                ABORT,
                'video index job requires exact planned stage receipts'
            );
        END
        """
    )


def _ensure_video_index_staged_activation(
    connection: sqlite3.Connection,
) -> None:
    generation_columns = {
        str(row[1])
        for row in connection.execute(
            "PRAGMA table_info(segment_generations)"
        ).fetchall()
    }
    if "video_thumbnail_path" not in generation_columns:
        connection.execute(
            "ALTER TABLE segment_generations ADD COLUMN video_thumbnail_path TEXT"
        )
    if "updates_video_thumbnail" not in generation_columns:
        connection.execute(
            """
            ALTER TABLE segment_generations
            ADD COLUMN updates_video_thumbnail INTEGER NOT NULL DEFAULT 0
                CHECK(updates_video_thumbnail IN (0, 1))
            """
        )
    connection.execute(
        """
        CREATE TRIGGER IF NOT EXISTS active_segment_generations_job_owner_insert
        BEFORE INSERT ON active_segment_generations
        WHEN EXISTS (
            SELECT 1
            FROM segment_generations AS generations
            JOIN stage_runs AS runs ON runs.run_id = generations.run_id
            WHERE generations.generation_id = NEW.generation_id
              AND runs.job_id IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM video_index_jobs AS jobs
                  WHERE jobs.job_id = runs.job_id AND jobs.state = 'complete'
              )
        )
        BEGIN
            SELECT RAISE(ABORT, 'video index job is not complete');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER IF NOT EXISTS active_segment_generations_job_owner_update
        BEFORE UPDATE OF generation_id, activated_at
        ON active_segment_generations
        WHEN EXISTS (
            SELECT 1
            FROM segment_generations AS generations
            JOIN stage_runs AS runs ON runs.run_id = generations.run_id
            WHERE generations.generation_id = NEW.generation_id
              AND runs.job_id IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM video_index_jobs AS jobs
                  WHERE jobs.job_id = runs.job_id AND jobs.state = 'complete'
              )
        )
        BEGIN
            SELECT RAISE(ABORT, 'video index job is not complete');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER IF NOT EXISTS active_text_vector_generations_job_owner_insert
        BEFORE INSERT ON active_text_vector_generations
        WHEN EXISTS (
            SELECT 1
            FROM text_vector_generations AS generations
            JOIN stage_runs AS runs ON runs.run_id = generations.run_id
            WHERE generations.generation_id = NEW.generation_id
              AND runs.job_id IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM video_index_jobs AS jobs
                  WHERE jobs.job_id = runs.job_id AND jobs.state = 'complete'
              )
        )
        BEGIN
            SELECT RAISE(ABORT, 'video index job is not complete');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER IF NOT EXISTS active_text_vector_generations_job_owner_update
        BEFORE UPDATE OF generation_id, activated_at
        ON active_text_vector_generations
        WHEN EXISTS (
            SELECT 1
            FROM text_vector_generations AS generations
            JOIN stage_runs AS runs ON runs.run_id = generations.run_id
            WHERE generations.generation_id = NEW.generation_id
              AND runs.job_id IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM video_index_jobs AS jobs
                  WHERE jobs.job_id = runs.job_id AND jobs.state = 'complete'
              )
        )
        BEGIN
            SELECT RAISE(ABORT, 'video index job is not complete');
        END
        """
    )


def _ensure_external_index_generations(connection: sqlite3.Connection) -> None:
    external_kinds = _sql_values(
        [kind.value for kind in EXTERNAL_INDEX_STAGE_KINDS]
    )
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS external_index_generations (
            video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
            stage_kind TEXT NOT NULL CHECK(stage_kind IN ({external_kinds})),
            generation_id TEXT NOT NULL CHECK(
                length(generation_id) = 32
                AND generation_id NOT GLOB '*[^0-9a-f]*'
            ),
            specification_hash TEXT NOT NULL
                REFERENCES stage_specifications(specification_hash),
            artifact_specification_hash TEXT NOT NULL CHECK(
                length(artifact_specification_hash) = 64
                AND artifact_specification_hash NOT GLOB '*[^0-9a-f]*'
            ),
            source_sha256 TEXT NOT NULL CHECK(
                length(source_sha256) = 64
                AND source_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            run_id TEXT NOT NULL UNIQUE REFERENCES stage_runs(run_id),
            descriptor_json TEXT NOT NULL CHECK(
                typeof(descriptor_json) = 'text'
                AND length(CAST(descriptor_json AS BLOB)) BETWEEN 1
                    AND {EXTERNAL_INDEX_DESCRIPTOR_JSON_MAX_BYTES}
                AND json_valid(descriptor_json)
                AND json_type(descriptor_json, '$') = 'object'
            ),
            descriptor_sha256 TEXT NOT NULL CHECK(
                length(descriptor_sha256) = 64
                AND descriptor_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            completed_at TEXT NOT NULL,
            PRIMARY KEY(video_id, stage_kind, generation_id)
        )
        """
    )
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS active_external_index_generations (
            video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
            stage_kind TEXT NOT NULL CHECK(stage_kind IN ({external_kinds})),
            generation_id TEXT NOT NULL,
            activated_at TEXT NOT NULL,
            PRIMARY KEY(video_id, stage_kind),
            FOREIGN KEY(video_id, stage_kind, generation_id)
                REFERENCES external_index_generations(
                    video_id, stage_kind, generation_id
                )
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_external_index_generations_video_stage
        ON external_index_generations(video_id, stage_kind, completed_at)
        """
    )
    connection.execute(
        """
        CREATE TRIGGER IF NOT EXISTS external_index_generation_run_identity_insert
        BEFORE INSERT ON external_index_generations
        WHEN NOT EXISTS (
            SELECT 1
            FROM stage_runs AS runs
            JOIN stage_specifications AS specifications
              ON specifications.specification_hash = runs.specification_hash
            JOIN videos ON videos.id = runs.video_id
            JOIN assets ON assets.asset_id = videos.asset_id
            WHERE runs.run_id = NEW.run_id
              AND runs.video_id = NEW.video_id
              AND runs.specification_hash = NEW.specification_hash
              AND runs.source_sha256 = NEW.source_sha256
              AND runs.state = 'running'
              AND runs.output_generation IS NULL
              AND specifications.stage_kind = NEW.stage_kind
              AND (
                    (NEW.stage_kind = 'visual_dense'
                     AND json_extract(
                         specifications.canonical_json,
                         '$.parameters.visual_specification_hash'
                     ) = NEW.artifact_specification_hash)
                 OR (NEW.stage_kind = 'lighthouse'
                     AND json_extract(
                         specifications.canonical_json,
                         '$.parameters.specification_hash'
                     ) = NEW.artifact_specification_hash)
              )
              AND json_extract(NEW.descriptor_json, '$.video_id') = NEW.video_id
              AND json_extract(NEW.descriptor_json, '$.generation_id') = NEW.generation_id
              AND json_extract(NEW.descriptor_json, '$.specification_hash')
                    = NEW.artifact_specification_hash
              AND json_extract(NEW.descriptor_json, '$.source_sha256')
                    = NEW.source_sha256
              AND json_type(NEW.descriptor_json, '$.source_size_bytes') = 'integer'
              AND json_extract(NEW.descriptor_json, '$.source_size_bytes')
                    = assets.size_bytes
              AND videos.size_bytes = assets.size_bytes
              AND json_type(NEW.descriptor_json, '$.duration_seconds')
                    IN ('integer', 'real')
              AND videos.duration IS NOT NULL
              AND json_extract(NEW.descriptor_json, '$.duration_seconds')
                    = videos.duration
              AND (
                    (NEW.stage_kind = 'visual_dense'
                     AND json_type(NEW.descriptor_json, '$.content_sha256') = 'text'
                     AND json_type(NEW.descriptor_json, '$.manifest_sha256') IS NULL)
                 OR (NEW.stage_kind = 'lighthouse'
                     AND json_type(NEW.descriptor_json, '$.manifest_sha256') = 'text'
                     AND json_type(NEW.descriptor_json, '$.content_sha256') IS NULL)
              )
              AND (SELECT COUNT(*) FROM json_each(NEW.descriptor_json)) = 7
        )
        BEGIN
            SELECT RAISE(ABORT, 'external index generation run identity mismatch');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER IF NOT EXISTS external_index_generations_immutable_update
        BEFORE UPDATE ON external_index_generations
        BEGIN
            SELECT RAISE(ABORT, 'external index generation is immutable');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER IF NOT EXISTS external_index_generations_immutable_delete
        BEFORE DELETE ON external_index_generations
        WHEN EXISTS (SELECT 1 FROM videos WHERE id = OLD.video_id)
        BEGIN
            SELECT RAISE(ABORT, 'external index generation is immutable');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER IF NOT EXISTS active_external_index_generations_job_owner_insert
        BEFORE INSERT ON active_external_index_generations
        WHEN NOT EXISTS (
            SELECT 1
            FROM external_index_generations AS generations
            JOIN stage_runs AS runs ON runs.run_id = generations.run_id
            LEFT JOIN video_index_jobs AS jobs ON jobs.job_id = runs.job_id
            WHERE generations.video_id = NEW.video_id
              AND generations.stage_kind = NEW.stage_kind
              AND generations.generation_id = NEW.generation_id
              AND generations.completed_at = NEW.activated_at
              AND (runs.job_id IS NULL OR jobs.state = 'complete')
        )
        BEGIN
            SELECT RAISE(ABORT, 'video index job is not complete');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER IF NOT EXISTS active_external_index_generations_job_owner_update
        BEFORE UPDATE OF generation_id, activated_at
        ON active_external_index_generations
        WHEN NOT EXISTS (
            SELECT 1
            FROM external_index_generations AS generations
            JOIN stage_runs AS runs ON runs.run_id = generations.run_id
            LEFT JOIN video_index_jobs AS jobs ON jobs.job_id = runs.job_id
            WHERE generations.video_id = NEW.video_id
              AND generations.stage_kind = NEW.stage_kind
              AND generations.generation_id = NEW.generation_id
              AND generations.completed_at = NEW.activated_at
              AND (runs.job_id IS NULL OR jobs.state = 'complete')
        )
        BEGIN
            SELECT RAISE(ABORT, 'video index job is not complete');
        END
        """
    )


def _ensure_video_index_job_quarantines(
    connection: sqlite3.Connection,
) -> None:
    existing = connection.execute(
        """
        SELECT 1 FROM sqlite_master
        WHERE type = 'table' AND name = 'video_index_job_quarantines'
        """
    ).fetchone()
    if existing is not None:
        columns = {
            str(row[1])
            for row in connection.execute(
                "PRAGMA table_info(video_index_job_quarantines)"
            ).fetchall()
        }
        required = set(_READ_ONLY_REQUIRED_COLUMNS["video_index_job_quarantines"])
        if not required <= columns:
            raise ValueError("video index job quarantine schema is incomplete")
    else:
        connection.execute(
            """
            CREATE TABLE video_index_job_quarantines (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                job_sequence INTEGER NOT NULL REFERENCES video_index_jobs(sequence),
                error_code TEXT NOT NULL CHECK(error_code = 'job_record_corrupt'),
                quarantined_at TEXT NOT NULL
            )
            """
        )
    connection.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS
        idx_video_index_job_quarantines_job_sequence
        ON video_index_job_quarantines(job_sequence)
        """
    )
    connection.execute(
        """
        CREATE TRIGGER IF NOT EXISTS video_index_job_quarantines_immutable_update
        BEFORE UPDATE ON video_index_job_quarantines
        BEGIN
            SELECT RAISE(ABORT, 'video index job quarantine is immutable');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER IF NOT EXISTS video_index_job_quarantines_immutable_delete
        BEFORE DELETE ON video_index_job_quarantines
        BEGIN
            SELECT RAISE(ABORT, 'video index job quarantine is immutable');
        END
        """
    )


def _migration_9_add_durable_video_index_jobs(
    connection: sqlite3.Connection,
) -> None:
    """Add append-only, fenced video-index work without inferring legacy history."""
    _migration_8_add_recoverable_artifact_gc(connection)
    table_names = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    index_names = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        ).fetchall()
    }
    trigger_names = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger'"
        ).fetchall()
    }
    stage_columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(stage_runs)").fetchall()
    }
    required_indexes = {
        "idx_stage_runs_job_id",
        "idx_video_index_jobs_active_idempotency",
        "idx_video_index_jobs_active_video",
        "idx_video_index_jobs_ready",
        "idx_video_index_jobs_retry_child",
        "idx_video_index_jobs_video_history",
    }
    required_triggers = {
        "stage_runs_job_id_immutable_update",
        "stage_runs_video_index_job_ownership_insert",
        "stage_runs_video_index_job_active_update",
        "stage_runs_video_index_job_identity_insert",
        "video_index_jobs_cancel_request_immutable",
        "video_index_jobs_execution_fence",
        "video_index_jobs_identity_immutable_update",
        "video_index_jobs_immutable_delete",
        "video_index_jobs_progress_monotonic",
        "video_index_jobs_reject_active_stage_completion",
        "video_index_jobs_retry_identity_insert",
        "video_index_jobs_source_identity_insert",
        "video_index_jobs_state_transition",
        "video_index_jobs_terminal_immutable_update",
        "video_index_jobs_updated_at_monotonic",
    }
    markers_present = any(
        (
            "video_index_jobs" in table_names,
            "job_id" in stage_columns,
            bool(required_indexes & index_names),
            bool(required_triggers & trigger_names),
        )
    )
    if markers_present:
        job_columns = {
            str(row[1])
            for row in connection.execute(
                "PRAGMA table_info(video_index_jobs)"
            ).fetchall()
        }
        required_columns = set(_READ_ONLY_REQUIRED_COLUMNS["video_index_jobs"])
        job_indexes = required_indexes - {"idx_stage_runs_job_id"}
        job_triggers = {
            name for name in required_triggers if not name.startswith("stage_runs_")
        }
        if (
            "video_index_jobs" not in table_names
            or not required_columns <= job_columns
            or not job_indexes <= index_names
            or not job_triggers <= trigger_names
        ):
            raise ValueError("durable video index job migration is incomplete or corrupt")
        _ensure_video_index_job_quarantines(connection)
        _ensure_stage_run_video_index_job_linkage(connection)
        _ensure_video_index_job_completion_receipts(connection)
        _ensure_video_index_staged_activation(connection)
        _ensure_external_index_generations(connection)
        return

    connection.execute(
        f"""
        CREATE TABLE video_index_jobs (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL UNIQUE,
            kind TEXT NOT NULL CHECK(kind = 'video_index'),
            intent TEXT NOT NULL CHECK(intent IN ('ingest', 'reindex')),
            video_id TEXT NOT NULL REFERENCES videos(id),
            source_sha256 TEXT NOT NULL CHECK(
                length(source_sha256) = 64
                AND source_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            plan_hash TEXT NOT NULL CHECK(
                length(plan_hash) = 64
                AND plan_hash NOT GLOB '*[^0-9a-f]*'
            ),
            plan_json TEXT NOT NULL CHECK(
                typeof(plan_json) = 'text'
                AND length(CAST(plan_json AS BLOB)) BETWEEN 1 AND {VIDEO_INDEX_PLAN_JSON_MAX_BYTES}
            ),
            idempotency_hash TEXT NOT NULL CHECK(
                length(idempotency_hash) = 64
                AND idempotency_hash NOT GLOB '*[^0-9a-f]*'
            ),
            state TEXT NOT NULL CHECK(
                state IN ('queued', 'running', 'complete', 'failed', 'cancelled')
            ),
            progress REAL NOT NULL CHECK(
                typeof(progress) IN ('real', 'integer')
                AND progress >= 0.0 AND progress <= 1.0
            ),
            stage TEXT NOT NULL CHECK(
                length(stage) BETWEEN 1 AND 64
                AND stage GLOB '[a-z]*'
                AND stage NOT GLOB '*[^a-z0-9_-]*'
            ),
            attempt INTEGER NOT NULL CHECK(attempt >= 1),
            retry_of_job_id TEXT REFERENCES video_index_jobs(job_id),
            prior_video_status TEXT,
            prior_video_progress REAL,
            prior_video_stage TEXT,
            prior_video_error_code TEXT,
            execution_token TEXT UNIQUE CHECK(
                execution_token IS NULL OR (
                    length(execution_token) BETWEEN 16 AND 256
                    AND execution_token GLOB '[A-Za-z0-9]*'
                    AND execution_token NOT GLOB '*[^A-Za-z0-9._~-]*'
                )
            ),
            cancel_requested_at TEXT,
            error_code TEXT CHECK(
                error_code IS NULL OR (
                    length(error_code) BETWEEN 1 AND 64
                    AND error_code GLOB '[a-z]*'
                    AND error_code NOT GLOB '*[^a-z0-9_-]*'
                )
            ),
            created_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT,
            updated_at TEXT NOT NULL,
            CHECK(retry_of_job_id IS NULL OR retry_of_job_id <> job_id),
            CHECK(
                (retry_of_job_id IS NULL AND attempt = 1)
                OR (retry_of_job_id IS NOT NULL AND attempt >= 2)
            ),
            CHECK(
                (intent = 'ingest'
                    AND prior_video_status IS NULL
                    AND prior_video_progress IS NULL
                    AND prior_video_stage IS NULL
                    AND prior_video_error_code IS NULL)
                OR
                (intent = 'reindex'
                    AND prior_video_status IS NOT NULL
                    AND prior_video_progress BETWEEN 0.0 AND 1.0
                    AND prior_video_stage IS NOT NULL)
            ),
            CHECK(
                (state = 'queued'
                    AND progress = 0.0 AND stage = 'queued'
                    AND execution_token IS NULL
                    AND cancel_requested_at IS NULL
                    AND error_code IS NULL
                    AND started_at IS NULL AND finished_at IS NULL
                    AND updated_at = created_at)
                OR
                (state = 'running'
                    AND progress < 1.0
                    AND stage NOT IN ('queued', 'complete', 'failed', 'cancelled')
                    AND execution_token IS NOT NULL
                    AND started_at IS NOT NULL
                    AND finished_at IS NULL
                    AND error_code IS NULL)
                OR
                (state = 'complete'
                    AND progress = 1.0 AND stage = 'complete'
                    AND execution_token IS NULL
                    AND cancel_requested_at IS NULL
                    AND error_code IS NULL
                    AND started_at IS NOT NULL AND finished_at IS NOT NULL)
                OR
                (state = 'failed'
                    AND stage = 'failed'
                    AND execution_token IS NULL
                    AND cancel_requested_at IS NULL
                    AND error_code IS NOT NULL
                    AND started_at IS NOT NULL AND finished_at IS NOT NULL)
                OR
                (state = 'cancelled'
                    AND stage = 'cancelled'
                    AND execution_token IS NULL
                    AND cancel_requested_at IS NOT NULL
                    AND error_code IS NULL
                    AND finished_at IS NOT NULL
                    AND (started_at IS NOT NULL OR progress = 0.0))
            )
        )
        """
    )
    connection.execute(
        """
        CREATE UNIQUE INDEX idx_video_index_jobs_active_video
        ON video_index_jobs(video_id)
        WHERE state IN ('queued', 'running')
        """
    )
    connection.execute(
        """
        CREATE UNIQUE INDEX idx_video_index_jobs_active_idempotency
        ON video_index_jobs(idempotency_hash)
        WHERE state IN ('queued', 'running')
        """
    )
    connection.execute(
        """
        CREATE UNIQUE INDEX idx_video_index_jobs_retry_child
        ON video_index_jobs(retry_of_job_id)
        WHERE retry_of_job_id IS NOT NULL
        """
    )
    connection.execute(
        """
        CREATE INDEX idx_video_index_jobs_ready
        ON video_index_jobs(state, sequence)
        """
    )
    connection.execute(
        """
        CREATE INDEX idx_video_index_jobs_video_history
        ON video_index_jobs(video_id, sequence DESC)
        """
    )
    connection.execute(
        """
        CREATE TRIGGER video_index_jobs_source_identity_insert
        BEFORE INSERT ON video_index_jobs
        WHEN NOT EXISTS (
            SELECT 1
            FROM videos
            JOIN assets ON assets.asset_id = videos.asset_id
            WHERE videos.id = NEW.video_id
              AND assets.sha256 = NEW.source_sha256
              AND videos.size_bytes = assets.size_bytes
        )
        BEGIN
            SELECT RAISE(ABORT, 'video index job source identity mismatch');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER video_index_jobs_retry_identity_insert
        BEFORE INSERT ON video_index_jobs
        WHEN NEW.retry_of_job_id IS NOT NULL AND NOT EXISTS (
            SELECT 1 FROM video_index_jobs AS parent
            WHERE parent.job_id = NEW.retry_of_job_id
              AND parent.state IN ('failed', 'cancelled')
              AND NEW.attempt = parent.attempt + 1
              AND NEW.kind = parent.kind
              AND NEW.intent = parent.intent
              AND NEW.video_id = parent.video_id
              AND NEW.source_sha256 = parent.source_sha256
              AND NEW.plan_hash = parent.plan_hash
              AND NEW.plan_json = parent.plan_json
              AND NEW.idempotency_hash = parent.idempotency_hash
              AND NEW.prior_video_status IS parent.prior_video_status
              AND NEW.prior_video_progress IS parent.prior_video_progress
              AND NEW.prior_video_stage IS parent.prior_video_stage
              AND NEW.prior_video_error_code IS parent.prior_video_error_code
        )
        BEGIN
            SELECT RAISE(ABORT, 'video index retry identity mismatch');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER video_index_jobs_identity_immutable_update
        BEFORE UPDATE OF
            job_id, kind, intent, video_id, source_sha256, plan_hash,
            plan_json, idempotency_hash, attempt, retry_of_job_id,
            prior_video_status, prior_video_progress, prior_video_stage,
            prior_video_error_code, created_at
        ON video_index_jobs
        BEGIN
            SELECT RAISE(ABORT, 'video index job identity is immutable');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER video_index_jobs_state_transition
        BEFORE UPDATE OF state ON video_index_jobs
        WHEN NOT (
            NEW.state = OLD.state
            OR (OLD.state = 'queued' AND NEW.state IN ('running', 'cancelled'))
            OR (OLD.state = 'running'
                AND NEW.state IN ('complete', 'failed', 'cancelled'))
        )
        BEGIN
            SELECT RAISE(ABORT, 'invalid video index job state transition');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER video_index_jobs_terminal_immutable_update
        BEFORE UPDATE ON video_index_jobs
        WHEN OLD.state IN ('complete', 'failed', 'cancelled')
        BEGIN
            SELECT RAISE(ABORT, 'terminal video index job is immutable');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER video_index_jobs_reject_active_stage_completion
        BEFORE UPDATE OF state ON video_index_jobs
        WHEN NEW.state = 'complete' AND EXISTS (
            SELECT 1 FROM stage_runs AS runs
            WHERE runs.job_id = OLD.job_id
              AND runs.state IN ('queued', 'running')
        )
        BEGIN
            SELECT RAISE(ABORT, 'video index job has active stage runs');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER video_index_jobs_progress_monotonic
        BEFORE UPDATE OF progress ON video_index_jobs
        WHEN NEW.progress < OLD.progress
        BEGIN
            SELECT RAISE(ABORT, 'video index job progress must be monotonic');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER video_index_jobs_execution_fence
        BEFORE UPDATE OF execution_token ON video_index_jobs
        WHEN OLD.execution_token IS NOT NEW.execution_token AND NOT (
            (OLD.state = 'queued' AND NEW.state = 'running'
                AND OLD.execution_token IS NULL AND NEW.execution_token IS NOT NULL)
            OR
            (OLD.state = 'running'
                AND NEW.state IN ('complete', 'failed', 'cancelled')
                AND OLD.execution_token IS NOT NULL AND NEW.execution_token IS NULL)
        )
        BEGIN
            SELECT RAISE(ABORT, 'invalid video index execution fence transition');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER video_index_jobs_cancel_request_immutable
        BEFORE UPDATE OF cancel_requested_at ON video_index_jobs
        WHEN OLD.cancel_requested_at IS NOT NULL
          AND NEW.cancel_requested_at IS NOT OLD.cancel_requested_at
        BEGIN
            SELECT RAISE(ABORT, 'video index cancellation request is immutable');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER video_index_jobs_updated_at_monotonic
        BEFORE UPDATE OF updated_at ON video_index_jobs
        WHEN NEW.updated_at < OLD.updated_at
        BEGIN
            SELECT RAISE(ABORT, 'video index updated timestamp must be monotonic');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER video_index_jobs_immutable_delete
        BEFORE DELETE ON video_index_jobs
        BEGIN
            SELECT RAISE(ABORT, 'video index job is immutable');
        END
        """
    )

    _ensure_video_index_job_quarantines(connection)
    _ensure_stage_run_video_index_job_linkage(connection)
    _ensure_video_index_job_completion_receipts(connection)
    _ensure_video_index_staged_activation(connection)
    _ensure_external_index_generations(connection)


_SCHEMA_MIGRATIONS: tuple[tuple[int, Callable[[sqlite3.Connection], None]], ...] = (
    (1, _migration_1_create_library_schema),
    (2, _migration_2_add_video_display_name),
    (3, _migration_3_add_stage_provenance),
    (4, _migration_4_add_asset_identity),
    (5, _migration_5_add_segment_generations),
    (6, _migration_6_add_text_vector_generations),
    (7, _migration_7_add_crash_safe_artifact_gc),
    (8, _migration_8_add_recoverable_artifact_gc),
    (9, _migration_9_add_durable_video_index_jobs),
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
        sequence,
        job_id,
        artifact_kind,
        generation_id,
        index_specification_hash,
        collection_name,
        reason,
        state,
        attempt,
        backoff_level,
        available_at,
        worker_id,
        lease_token,
        lease_expires_at,
        error_code,
        created_at,
        updated_at
    FROM artifact_gc_jobs
"""

_VIDEO_INDEX_JOB_SELECT = """
    SELECT
        sequence,
        job_id,
        kind,
        intent,
        video_id,
        source_sha256,
        plan_hash,
        plan_json,
        idempotency_hash,
        state,
        progress,
        stage,
        attempt,
        retry_of_job_id,
        prior_video_status,
        prior_video_progress,
        prior_video_stage,
        prior_video_error_code,
        execution_token,
        cancel_requested_at,
        error_code,
        created_at,
        started_at,
        finished_at,
        updated_at
    FROM video_index_jobs
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


def _validate_external_index_descriptor(
    descriptor: object,
    *,
    stage_kind: StageKind,
    video_id: str,
    generation_id: str,
    artifact_specification_hash: str,
    source_sha256: str,
    source_size_bytes: int,
    duration_seconds: float,
) -> tuple[dict[str, object], str, str]:
    digest_field = (
        "content_sha256"
        if stage_kind is StageKind.VISUAL_DENSE
        else "manifest_sha256"
        if stage_kind is StageKind.LIGHTHOUSE
        else None
    )
    expected_keys = {
        "duration_seconds",
        "generation_id",
        "source_sha256",
        "source_size_bytes",
        "specification_hash",
        "video_id",
    }
    if digest_field is not None:
        expected_keys.add(digest_field)
    invalid = (
        digest_field is None
        or not isinstance(descriptor, dict)
        or set(descriptor) != expected_keys
        or descriptor.get("video_id") != video_id
        or descriptor.get("generation_id") != generation_id
        or descriptor.get("specification_hash") != artifact_specification_hash
        or descriptor.get("source_sha256") != source_sha256
        or descriptor.get("source_size_bytes") != source_size_bytes
        or type(descriptor.get("source_size_bytes")) is not int
        or type(descriptor.get(digest_field)) is not str
        or _SHA256_RE.fullmatch(str(descriptor.get(digest_field))) is None
    )
    duration = descriptor.get("duration_seconds") if isinstance(descriptor, dict) else None
    if (
        invalid
        or isinstance(duration, bool)
        or not isinstance(duration, Real)
        or not math.isfinite(float(duration))
        or not 0 < float(duration) <= 24 * 60 * 60
        or isinstance(duration_seconds, bool)
        or not isinstance(duration_seconds, Real)
        or not math.isfinite(float(duration_seconds))
        or not math.isclose(
            float(duration),
            float(duration_seconds),
            rel_tol=1e-9,
            abs_tol=1e-6,
        )
    ):
        raise ValueError("external index canonical descriptor is invalid")
    normalized = dict(descriptor)
    normalized["duration_seconds"] = float(duration)
    try:
        canonical = json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (RecursionError, TypeError, ValueError) as error:
        raise ValueError("external index canonical descriptor is invalid") from error
    encoded = canonical.encode("utf-8")
    if not 1 <= len(encoded) <= EXTERNAL_INDEX_DESCRIPTOR_JSON_MAX_BYTES:
        raise ValueError("external index canonical descriptor is invalid")
    return normalized, canonical, hashlib.sha256(encoded).hexdigest()


def _external_video_duration(
    connection: sqlite3.Connection,
    video_id: str,
) -> float:
    row = connection.execute(
        "SELECT duration FROM videos WHERE id = ?",
        (video_id,),
    ).fetchone()
    duration = row["duration"] if row is not None else None
    if (
        isinstance(duration, bool)
        or not isinstance(duration, Real)
        or not math.isfinite(float(duration))
        or not 0 < float(duration) <= 24 * 60 * 60
    ):
        raise ValueError("external index video duration is invalid")
    return float(duration)


def _external_artifact_specification_hash(
    specification: StageSpecification,
) -> str:
    parameter_name = (
        "visual_specification_hash"
        if specification.kind is StageKind.VISUAL_DENSE
        else "specification_hash"
        if specification.kind is StageKind.LIGHTHOUSE
        else None
    )
    value = (
        specification.parameters.get(parameter_name)
        if parameter_name is not None
        else None
    )
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError("external stage artifact specification is invalid")
    return value


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
class VideoIndexRecoveryReport:
    examined_count: int
    retry_job_ids: tuple[str, ...]
    cancelled_job_ids: tuple[str, ...]
    quarantined_count: int
    has_more: bool


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
class ExternalIndexGeneration:
    video_id: str
    stage_kind: StageKind
    generation_id: str
    specification_hash: str
    artifact_specification_hash: str
    source_sha256: str
    run_id: str
    descriptor: dict[str, object]
    descriptor_sha256: str
    completed_at: str


@dataclass(frozen=True, slots=True)
class ExternalIndexReleaseSnapshot:
    """One SQLite snapshot of durable external bindings for product search."""

    job_backed_video_ids: frozenset[str]
    visual_dense: dict[str, dict[str, object]]
    lighthouse: dict[str, dict[str, object]]

    def bindings_for(self, kind: StageKind) -> dict[str, dict[str, object]]:
        if kind is StageKind.VISUAL_DENSE:
            return {key: dict(value) for key, value in self.visual_dense.items()}
        if kind is StageKind.LIGHTHOUSE:
            return {key: dict(value) for key, value in self.lighthouse.items()}
        raise ValueError("external release snapshot requires an external stage")


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


class _RepositoryConnection(sqlite3.Connection):
    """Finish each transaction and release its SQLite resources in one scope."""

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            # sqlite3's context manager commits/rolls back but does not close;
            # deferred WAL cleanup must not mutate storage during later work.
            self.close()


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

    def __init__(self, database_path: Path, *, read_only: bool = False) -> None:
        self.database_path = Path(database_path)
        self._read_only = read_only
        self._read_only_uri: str | None = None
        if read_only:
            absolute = _validate_read_only_database_path(self.database_path)
            immutable = _require_existing_wal_sidecars(absolute)
            self.database_path = absolute
            immutable_parameter = "&immutable=1" if immutable else ""
            self._read_only_uri = (
                f"{absolute.as_uri()}?mode=ro{immutable_parameter}"
            )
            try:
                connection = self._connect()
                try:
                    _validate_latest_read_only_schema(connection)
                finally:
                    connection.close()
            except RuntimeError:
                raise
            except (OSError, sqlite3.Error) as error:
                raise RuntimeError("read-only database is unavailable or corrupt") from error

    @classmethod
    def open_read_only(cls, database_path: Path) -> Repository:
        """Open and validate the exact current schema without database writes.

        SQLite may update lock bytes in an already-existing WAL ``-shm`` file.
        A caller requiring a byte-identical source tree must first copy the
        database and its WAL under the persistent runtime lock, then open that
        private snapshot here.
        """
        return cls(database_path, read_only=True)

    @property
    def is_read_only(self) -> bool:
        return self._read_only

    def _connect(self) -> sqlite3.Connection:
        if self._read_only:
            assert self._read_only_uri is not None
            connection = sqlite3.connect(
                self._read_only_uri,
                timeout=30,
                uri=True,
                factory=_RepositoryConnection,
            )
        else:
            connection = sqlite3.connect(
                self.database_path, timeout=30, factory=_RepositoryConnection
            )
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            if self._read_only:
                connection.execute("PRAGMA query_only = ON")
        except BaseException:
            connection.close()
            raise
        return connection

    def initialize(self) -> None:
        if self._read_only:
            raise RuntimeError("read-only repository cannot be initialized")
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
                    if current_version == LATEST_SCHEMA_VERSION:
                        _migration_9_add_durable_video_index_jobs(connection)
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

    @staticmethod
    def _insert_video_index_job(
        connection: sqlite3.Connection,
        *,
        job: VideoIndexJob,
        plan: VideoIndexPlanSnapshot,
    ) -> None:
        if not isinstance(job, VideoIndexJob):
            raise ValueError("video index job must be validated")
        if not isinstance(plan, VideoIndexPlanSnapshot):
            raise ValueError("video index plan must be validated")
        if plan.plan_hash != job.plan_hash:
            raise ValueError("video index job plan identity mismatch")
        prior = job.prior_video_state
        connection.execute(
            """
            INSERT INTO video_index_jobs (
                job_id, kind, intent, video_id, source_sha256,
                plan_hash, plan_json, idempotency_hash, state, progress,
                stage, attempt, retry_of_job_id, prior_video_status,
                prior_video_progress, prior_video_stage,
                prior_video_error_code, execution_token,
                cancel_requested_at, error_code, created_at, started_at,
                finished_at, updated_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?
            )
            """,
            (
                job.job_id,
                job.kind.value,
                job.intent.value,
                job.video_id,
                job.source_sha256,
                job.plan_hash,
                plan.canonical_json,
                job.idempotency_hash,
                job.state.value,
                job.progress,
                job.stage,
                job.attempt,
                job.retry_of_job_id,
                prior.status if prior is not None else None,
                prior.progress if prior is not None else None,
                prior.stage if prior is not None else None,
                prior.error_code if prior is not None else None,
                job.execution_token,
                job.cancel_requested_at,
                job.error_code,
                job.created_at,
                job.started_at,
                job.finished_at,
                job.updated_at,
            ),
        )

    @classmethod
    def _video_index_job_from_row(
        cls,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        validate_retry: bool = True,
    ) -> tuple[VideoIndexJob, VideoIndexPlanSnapshot]:
        try:
            plan = VideoIndexPlanSnapshot.from_canonical_json(row["plan_json"])
            if plan.plan_hash != row["plan_hash"]:
                raise ValueError("video index plan hash mismatch")
            intent = VideoIndexIntent(row["intent"])
            prior: PriorVideoState | None = None
            if intent is VideoIndexIntent.REINDEX:
                prior = PriorVideoState(
                    status=row["prior_video_status"],
                    progress=row["prior_video_progress"],
                    stage=row["prior_video_stage"],
                    error_code=row["prior_video_error_code"],
                )
            job = VideoIndexJob(
                job_id=row["job_id"],
                kind=row["kind"],
                intent=intent,
                video_id=row["video_id"],
                source_sha256=row["source_sha256"],
                plan_hash=row["plan_hash"],
                idempotency_hash=row["idempotency_hash"],
                state=row["state"],
                progress=row["progress"],
                stage=row["stage"],
                attempt=row["attempt"],
                retry_of_job_id=row["retry_of_job_id"],
                prior_video_state=prior,
                execution_token=row["execution_token"],
                cancel_requested_at=row["cancel_requested_at"],
                error_code=row["error_code"],
                created_at=row["created_at"],
                started_at=row["started_at"],
                finished_at=row["finished_at"],
                updated_at=row["updated_at"],
            )
            source = connection.execute(
                """
                SELECT
                    assets.sha256,
                    assets.size_bytes AS asset_size_bytes,
                    videos.size_bytes AS video_size_bytes
                FROM videos
                JOIN assets ON assets.asset_id = videos.asset_id
                WHERE videos.id = ?
                """,
                (job.video_id,),
            ).fetchone()
            if (
                source is None
                or source["sha256"] != job.source_sha256
                or source["asset_size_bytes"] != source["video_size_bytes"]
            ):
                raise ValueError("video index source identity mismatch")
            if validate_retry and job.retry_of_job_id is not None:
                parent_row = connection.execute(
                    f"{_VIDEO_INDEX_JOB_SELECT} WHERE job_id = ?",
                    (job.retry_of_job_id,),
                ).fetchone()
                if parent_row is None:
                    raise ValueError("video index retry parent is missing")
                parent, parent_plan = cls._video_index_job_from_row(
                    connection,
                    parent_row,
                    validate_retry=False,
                )
                if (
                    parent.state not in {JobState.FAILED, JobState.CANCELLED}
                    or job.attempt != parent.attempt + 1
                    or job.kind is not parent.kind
                    or job.intent is not parent.intent
                    or job.video_id != parent.video_id
                    or job.source_sha256 != parent.source_sha256
                    or job.plan_hash != parent.plan_hash
                    or plan.canonical_json != parent_plan.canonical_json
                    or job.idempotency_hash != parent.idempotency_hash
                    or job.prior_video_state != parent.prior_video_state
                ):
                    raise ValueError("video index retry lineage mismatch")
            return job, plan
        except (KeyError, TypeError, ValueError, RecursionError) as error:
            raise ValueError("persisted video index job is corrupt") from error

    @classmethod
    def _get_video_index_job(
        cls,
        connection: sqlite3.Connection,
        job_id: str,
    ) -> tuple[VideoIndexJob, VideoIndexPlanSnapshot] | None:
        row = connection.execute(
            f"{_VIDEO_INDEX_JOB_SELECT} WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if row is None:
            return None
        return cls._video_index_job_from_row(connection, row)

    @staticmethod
    def _new_queued_video_index_job(
        *,
        job_id: str,
        intent: VideoIndexIntent,
        video_id: str,
        source_sha256: str,
        plan: VideoIndexPlanSnapshot,
        prior_video_state: PriorVideoState | None,
        timestamp: str,
    ) -> VideoIndexJob:
        return VideoIndexJob(
            job_id=job_id,
            kind=JobKind.VIDEO_INDEX,
            intent=intent,
            video_id=video_id,
            source_sha256=source_sha256,
            plan_hash=plan.plan_hash,
            idempotency_hash=video_index_idempotency_hash(
                kind=JobKind.VIDEO_INDEX,
                intent=intent,
                video_id=video_id,
                source_sha256=source_sha256,
                plan_hash=plan.plan_hash,
            ),
            state=JobState.QUEUED,
            progress=0.0,
            stage=JobState.QUEUED.value,
            attempt=1,
            retry_of_job_id=None,
            prior_video_state=prior_video_state,
            execution_token=None,
            cancel_requested_at=None,
            error_code=None,
            created_at=timestamp,
            started_at=None,
            finished_at=None,
            updated_at=timestamp,
        )

    def create_video_with_asset_and_index_job(
        self,
        *,
        video_id: str,
        original_name: str,
        stored_name: str,
        media_path: str,
        size_bytes: int,
        source_sha256: str,
        plan: VideoIndexPlanSnapshot,
        job_id: str | None = None,
    ) -> tuple[VideoRecord, VideoIndexJob]:
        if not isinstance(plan, VideoIndexPlanSnapshot):
            raise ValueError("video index plan must be validated")
        resolved_job_id = uuid4().hex if job_id is None else job_id
        validate_artifact_identifier(resolved_job_id, field_name="video index job id")
        timestamp = _now()
        asset = AssetRecord.from_digest(
            sha256=source_sha256,
            size_bytes=size_bytes,
            created_at=timestamp,
        )
        candidate = self._new_queued_video_index_job(
            job_id=resolved_job_id,
            intent=VideoIndexIntent.INGEST,
            video_id=video_id,
            source_sha256=source_sha256,
            plan=plan,
            prior_video_state=None,
            timestamp=timestamp,
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
            self._insert_video_index_job(
                connection,
                job=candidate,
                plan=plan,
            )
            video_row = connection.execute(
                "SELECT * FROM videos WHERE id = ?",
                (video_id,),
            ).fetchone()
            job_row = connection.execute(
                f"{_VIDEO_INDEX_JOB_SELECT} WHERE job_id = ?",
                (candidate.job_id,),
            ).fetchone()
            if video_row is None or job_row is None:
                raise RuntimeError("video index upload transaction was not persisted")
            video = VideoRecord(**dict(video_row))
            job, _persisted_plan = self._video_index_job_from_row(
                connection,
                job_row,
            )
        return video, job

    @staticmethod
    def _normalised_prior_video_state(video: VideoRecord) -> PriorVideoState:
        error_code = video.error
        try:
            return PriorVideoState(
                status=video.status,
                progress=video.progress,
                stage=video.stage,
                error_code=error_code,
            )
        except ValueError:
            if error_code is None:
                raise
            return PriorVideoState(
                status=video.status,
                progress=video.progress,
                stage=video.stage,
                error_code=None,
            )

    def enqueue_video_reindex_job(
        self,
        video_id: str,
        *,
        plan: VideoIndexPlanSnapshot,
        job_id: str | None = None,
    ) -> VideoIndexJob:
        if not isinstance(plan, VideoIndexPlanSnapshot):
            raise ValueError("video index plan must be validated")
        resolved_job_id = uuid4().hex if job_id is None else job_id
        validate_artifact_identifier(resolved_job_id, field_name="video index job id")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            video_row = connection.execute(
                "SELECT * FROM videos WHERE id = ?",
                (video_id,),
            ).fetchone()
            if video_row is None:
                raise KeyError(video_id)
            video = VideoRecord(**dict(video_row))
            asset = self._get_video_asset(connection, video_id)
            if asset is None:
                raise AssetIdentityError("video asset identity is unavailable")
            requested_hash = video_index_idempotency_hash(
                kind=JobKind.VIDEO_INDEX,
                intent=VideoIndexIntent.REINDEX,
                video_id=video_id,
                source_sha256=asset.sha256,
                plan_hash=plan.plan_hash,
            )
            active_row = connection.execute(
                f"""
                {_VIDEO_INDEX_JOB_SELECT}
                WHERE video_id = ? AND state IN ('queued', 'running')
                ORDER BY sequence DESC LIMIT 1
                """,
                (video_id,),
            ).fetchone()
            if active_row is not None:
                active, active_plan = self._video_index_job_from_row(
                    connection,
                    active_row,
                )
                if (
                    active.idempotency_hash == requested_hash
                    and active_plan.canonical_json == plan.canonical_json
                ):
                    return active
                raise JobTransitionError("video already has active indexing work")
            timestamp = _now()
            candidate = self._new_queued_video_index_job(
                job_id=resolved_job_id,
                intent=VideoIndexIntent.REINDEX,
                video_id=video_id,
                source_sha256=asset.sha256,
                plan=plan,
                prior_video_state=self._normalised_prior_video_state(video),
                timestamp=timestamp,
            )
            self._insert_video_index_job(
                connection,
                job=candidate,
                plan=plan,
            )
            cursor = connection.execute(
                """
                UPDATE videos
                SET status = 'queued', progress = 0.0, stage = 'queued',
                    error = NULL, updated_at = ?
                WHERE id = ?
                """,
                (timestamp, video_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("video changed during reindex enqueue")
            row = connection.execute(
                f"{_VIDEO_INDEX_JOB_SELECT} WHERE job_id = ?",
                (candidate.job_id,),
            ).fetchone()
            assert row is not None
            persisted, _persisted_plan = self._video_index_job_from_row(
                connection,
                row,
            )
            return persisted

    def get_video_index_job(self, job_id: str) -> VideoIndexJob | None:
        validate_artifact_identifier(job_id, field_name="video index job id")
        with self._connect() as connection:
            resolved = self._get_video_index_job(connection, job_id)
        return resolved[0] if resolved is not None else None

    def get_video_index_job_plan(
        self,
        job_id: str,
    ) -> VideoIndexPlanSnapshot | None:
        validate_artifact_identifier(job_id, field_name="video index job id")
        with self._connect() as connection:
            resolved = self._get_video_index_job(connection, job_id)
        return resolved[1] if resolved is not None else None

    def count_video_index_job_quarantines(self) -> int:
        with self._connect() as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM video_index_job_quarantines"
                ).fetchone()[0]
            )

    def list_video_index_jobs(
        self,
        *,
        video_id: str | None = None,
        limit: int = 100,
    ) -> tuple[VideoIndexJob, ...]:
        resolved_limit = _validate_repository_batch_limit(
            limit,
            field_name="video index job listing limit",
        )
        predicate = "WHERE video_id = ?" if video_id is not None else ""
        parameters: tuple[object, ...]
        if video_id is not None:
            validate_artifact_identifier(video_id, field_name="video index video id")
            parameters = (video_id, resolved_limit)
        else:
            parameters = (resolved_limit,)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                {_VIDEO_INDEX_JOB_SELECT}
                {predicate}
                ORDER BY sequence
                LIMIT ?
                """,
                parameters,
            ).fetchall()
            return tuple(
                self._video_index_job_from_row(connection, row)[0]
                for row in rows
            )

    def get_latest_video_index_jobs(
        self,
        video_ids: Sequence[str],
    ) -> dict[str, VideoIndexJob]:
        if isinstance(video_ids, (str, bytes)) or not isinstance(
            video_ids,
            Sequence,
        ):
            raise ValueError("video ids must be a bounded sequence")
        selected_ids = tuple(video_ids)
        if len(selected_ids) > REPOSITORY_BATCH_LIMIT_MAX:
            raise ValueError(
                "latest video index jobs are limited to "
                f"{REPOSITORY_BATCH_LIMIT_MAX} video ids"
            )
        for video_id in selected_ids:
            validate_artifact_identifier(
                video_id,
                field_name="latest video index job video id",
            )
        if len(set(selected_ids)) != len(selected_ids):
            raise ValueError("latest video index job video ids must be unique")
        if not selected_ids:
            return {}
        requested_values = ", ".join("(?)" for _video_id in selected_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                WITH requested(video_id) AS (VALUES {requested_values})
                SELECT selected.*,
                       quarantines.error_code AS quarantine_error_code
                FROM (
                    {_VIDEO_INDEX_JOB_SELECT}
                ) AS selected
                JOIN requested
                  ON requested.video_id = selected.video_id
                LEFT JOIN video_index_job_quarantines AS quarantines
                  ON quarantines.job_sequence = selected.sequence
                WHERE NOT EXISTS (
                    SELECT 1 FROM video_index_jobs AS newer
                    WHERE newer.video_id = selected.video_id
                      AND newer.sequence > selected.sequence
                )
                ORDER BY selected.sequence
                """,
                selected_ids,
            ).fetchall()
            latest: dict[str, VideoIndexJob] = {}
            for row in rows:
                if row["quarantine_error_code"] is not None:
                    raise ValueError("latest video index job is quarantined")
                job, _plan = self._video_index_job_from_row(connection, row)
                if job.video_id in latest:
                    raise ValueError("latest video index job selection is corrupt")
                latest[job.video_id] = job
            return latest

    @classmethod
    def _persist_video_index_transition(
        cls,
        connection: sqlite3.Connection,
        *,
        current: VideoIndexJob,
        candidate: VideoIndexJob,
    ) -> VideoIndexJob:
        cursor = connection.execute(
            """
            UPDATE video_index_jobs
            SET state = ?, progress = ?, stage = ?, execution_token = ?,
                cancel_requested_at = ?, error_code = ?, started_at = ?,
                finished_at = ?, updated_at = ?
            WHERE job_id = ? AND state = ?
              AND execution_token IS ? AND updated_at = ?
            """,
            (
                candidate.state.value,
                candidate.progress,
                candidate.stage,
                candidate.execution_token,
                candidate.cancel_requested_at,
                candidate.error_code,
                candidate.started_at,
                candidate.finished_at,
                candidate.updated_at,
                current.job_id,
                current.state.value,
                current.execution_token,
                current.updated_at,
            ),
        )
        if cursor.rowcount != 1:
            raise JobFenceError("job execution token no longer owns this attempt")
        row = connection.execute(
            f"{_VIDEO_INDEX_JOB_SELECT} WHERE job_id = ?",
            (current.job_id,),
        ).fetchone()
        assert row is not None
        return cls._video_index_job_from_row(connection, row)[0]

    @staticmethod
    def _sync_video_for_index_job(
        connection: sqlite3.Connection,
        job: VideoIndexJob,
    ) -> None:
        if job.state is JobState.QUEUED:
            values = ("queued", 0.0, "queued", None)
        elif job.state is JobState.RUNNING:
            values = ("processing", job.progress, job.stage, None)
        elif job.state is JobState.COMPLETE:
            values = ("ready", 1.0, "ready", None)
        elif (
            job.state in {JobState.FAILED, JobState.CANCELLED}
            and job.intent is VideoIndexIntent.REINDEX
            and job.prior_video_state is not None
        ):
            prior = job.prior_video_state
            values = (
                prior.status,
                prior.progress,
                prior.stage,
                prior.error_code,
            )
        elif job.state is JobState.FAILED:
            values = ("failed", job.progress, "failed", job.error_code)
        elif job.state is JobState.CANCELLED:
            values = ("failed", job.progress, "cancelled", "job_cancelled")
        else:  # pragma: no cover - exhaustive domain enum
            raise ValueError("unsupported video index job state")
        cursor = connection.execute(
            """
            UPDATE videos
            SET status = ?, progress = ?, stage = ?, error = ?, updated_at = ?
            WHERE id = ?
            """,
            (*values, job.updated_at, job.video_id),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("video index job lost its video")

    def claim_next_video_index_job(
        self,
        *,
        execution_token: str,
        stage: str = "starting",
        scan_limit: int = 16,
    ) -> VideoIndexJob | None:
        if (
            isinstance(scan_limit, bool)
            or not isinstance(scan_limit, int)
            or not 1 <= scan_limit <= VIDEO_INDEX_JOB_SCAN_LIMIT_MAX
        ):
            raise ValueError(
                "video index scan limit must be between "
                f"1 and {VIDEO_INDEX_JOB_SCAN_LIMIT_MAX}"
            )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                f"""
                {_VIDEO_INDEX_JOB_SELECT}
                WHERE state = 'queued'
                  AND NOT EXISTS (
                      SELECT 1 FROM video_index_job_quarantines AS quarantines
                      WHERE quarantines.job_sequence = video_index_jobs.sequence
                  )
                ORDER BY sequence
                LIMIT ?
                """,
                (scan_limit,),
            ).fetchall()
            if not rows:
                return None
            for row in rows:
                try:
                    current, _plan = self._video_index_job_from_row(connection, row)
                except ValueError:
                    self._quarantine_video_index_job_row(
                        connection,
                        row=row,
                        timestamp=_now(),
                    )
                    continue
                candidate = start_job(
                    current,
                    execution_token=execution_token,
                    stage=stage,
                    now=_now(),
                )
                persisted = self._persist_video_index_transition(
                    connection,
                    current=current,
                    candidate=candidate,
                )
                self._sync_video_for_index_job(connection, persisted)
                return persisted
            return None

    def checkpoint_video_index_job(
        self,
        job_id: str,
        *,
        execution_token: str,
        progress: float,
        stage: str,
    ) -> VideoIndexJob:
        validate_artifact_identifier(job_id, field_name="video index job id")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            resolved = self._get_video_index_job(connection, job_id)
            if resolved is None:
                raise KeyError(job_id)
            current, _plan = resolved
            candidate = advance_job(
                current,
                execution_token=execution_token,
                progress=progress,
                stage=stage,
                now=_now(),
            )
            persisted = self._persist_video_index_transition(
                connection,
                current=current,
                candidate=candidate,
            )
            self._sync_video_for_index_job(connection, persisted)
            return persisted

    def request_video_index_job_cancellation(self, job_id: str) -> VideoIndexJob:
        validate_artifact_identifier(job_id, field_name="video index job id")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            resolved = self._get_video_index_job(connection, job_id)
            if resolved is None:
                raise KeyError(job_id)
            current, _plan = resolved
            candidate = request_job_cancellation(current, now=_now())
            if candidate is current:
                return current
            persisted = self._persist_video_index_transition(
                connection,
                current=current,
                candidate=candidate,
            )
            if persisted.state is JobState.CANCELLED:
                self._terminalize_linked_stage_runs(
                    connection,
                    job_id=current.job_id,
                    timestamp=persisted.updated_at,
                    failure_error_code=None,
                )
                self._sync_video_for_index_job(connection, persisted)
            return persisted

    @staticmethod
    def _require_exact_video_index_job_stage_receipts(
        connection: sqlite3.Connection,
        *,
        job: VideoIndexJob,
        plan: VideoIndexPlanSnapshot,
    ) -> None:
        expected = {
            kind: plan.for_kind(kind).specification_hash
            for kind in _VIDEO_INDEX_STAGE_KINDS
        }
        rows = connection.execute(
            """
            SELECT
                runs.video_id,
                runs.source_sha256,
                runs.specification_hash,
                runs.state,
                specifications.stage_kind
            FROM stage_runs AS runs
            JOIN stage_specifications AS specifications
              ON specifications.specification_hash = runs.specification_hash
            WHERE runs.job_id = ?
            ORDER BY runs.sequence
            """,
            (job.job_id,),
        ).fetchall()
        if len(rows) != len(expected):
            raise JobTransitionError(
                "video index job requires exact planned stage receipts"
            )
        seen: set[StageKind] = set()
        try:
            for row in rows:
                stage_kind = StageKind(row["stage_kind"])
                state = StageState(row["state"])
                if (
                    stage_kind in seen
                    or stage_kind not in expected
                    or row["video_id"] != job.video_id
                    or row["source_sha256"] != job.source_sha256
                    or row["specification_hash"] != expected[stage_kind]
                    or state not in _VIDEO_INDEX_COMPLETION_STAGE_STATES
                ):
                    raise JobTransitionError(
                        "video index job requires exact planned stage receipts"
                    )
                seen.add(stage_kind)
        except (TypeError, ValueError) as error:
            raise JobTransitionError(
                "video index job requires exact planned stage receipts"
            ) from error
        if seen != set(expected):
            raise JobTransitionError(
                "video index job requires exact planned stage receipts"
            )

    @classmethod
    def _activate_video_index_job_outputs(
        cls,
        connection: sqlite3.Connection,
        *,
        job: VideoIndexJob,
        plan: VideoIndexPlanSnapshot,
    ) -> None:
        if job.state is not JobState.COMPLETE:
            raise JobTransitionError(
                "video index outputs require a complete owning job"
            )
        rows = connection.execute(
            """
            SELECT
                runs.run_id,
                runs.output_generation,
                runs.specification_hash,
                runs.source_sha256,
                specifications.stage_kind
            FROM stage_runs AS runs
            JOIN stage_specifications AS specifications
              ON specifications.specification_hash = runs.specification_hash
            WHERE runs.job_id = ? AND runs.state = 'complete'
            ORDER BY runs.sequence
            """,
            (job.job_id,),
        ).fetchall()
        segment_outputs: dict[StageKind, SegmentGeneration] = {}
        external_outputs: dict[StageKind, ExternalIndexGeneration] = {}
        text_output: TextVectorGeneration | None = None
        text_inputs: tuple[TextVectorGenerationInput, ...] | None = None
        for row in rows:
            try:
                stage_kind = StageKind(row["stage_kind"])
                planned = plan.for_kind(stage_kind)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "video index job output stage is corrupt"
                ) from error
            if (
                row["specification_hash"] != planned.specification_hash
                or row["source_sha256"] != job.source_sha256
            ):
                raise ValueError("video index job output identity is corrupt")
            if stage_kind in SEGMENT_STAGE_KINDS:
                if stage_kind in segment_outputs:
                    raise ValueError(
                        "video index job has multiple completed segment outputs"
                    )
                generation_id = row["output_generation"]
                if type(generation_id) is not str or not generation_id:
                    raise ValueError("video index job segment output is missing")
                resolved = cls._get_segment_generation_with_run(
                    connection,
                    generation_id,
                )
                if resolved is None:
                    raise ValueError("video index job segment output is missing")
                generation, run = resolved
                if (
                    run.run_id != row["run_id"]
                    or run.state is not StageState.COMPLETE
                    or generation.video_id != job.video_id
                ):
                    raise ValueError("video index job segment output is corrupt")
                segment_outputs[stage_kind] = generation
            elif stage_kind is StageKind.TEXT_VECTORS:
                if text_output is not None:
                    raise ValueError(
                        "video index job has multiple completed text vector outputs"
                    )
                generation_id = row["output_generation"]
                if type(generation_id) is not str or not generation_id:
                    raise ValueError("video index job text vector output is missing")
                resolved_text = cls._get_text_vector_generation_with_run(
                    connection,
                    generation_id,
                )
                if resolved_text is None:
                    raise ValueError("video index job text vector output is missing")
                generation, run, _index, inputs, _points = resolved_text
                if (
                    run.run_id != row["run_id"]
                    or run.state is not StageState.COMPLETE
                    or generation.video_id != job.video_id
                ):
                    raise ValueError("video index job text vector output is corrupt")
                text_output = generation
                text_inputs = inputs
            elif stage_kind in EXTERNAL_INDEX_STAGE_KINDS:
                if stage_kind in external_outputs:
                    raise ValueError(
                        "video index job has multiple completed external outputs"
                    )
                generation_id = row["output_generation"]
                if type(generation_id) is not str or not generation_id:
                    raise ValueError("video index job external output is missing")
                resolved_external = cls._get_external_index_generation_with_run(
                    connection,
                    video_id=job.video_id,
                    stage_kind=stage_kind,
                    generation_id=generation_id,
                )
                if resolved_external is None:
                    raise ValueError("video index job external output is missing")
                external_generation, run = resolved_external
                if (
                    run.run_id != row["run_id"]
                    or run.state is not StageState.COMPLETE
                    or external_generation.video_id != job.video_id
                ):
                    raise ValueError("video index job external output is corrupt")
                external_outputs[stage_kind] = external_generation

        for planned in plan.specifications.segment_specifications:
            stage_kind = planned.kind
            staged = segment_outputs.get(stage_kind)
            current = cls._get_active_segment_generation(
                connection,
                job.video_id,
                stage_kind,
                require_released_owner=False,
            )
            retained = (
                current
                if current is not None
                and current.specification_hash == planned.specification_hash
                and current.source_sha256 == job.source_sha256
                and cls._stage_run_owner_is_released(
                    connection,
                    current.run_id,
                )
                else None
            )
            selected = staged if staged is not None else retained
            if selected is None:
                connection.execute(
                    """
                    DELETE FROM active_segment_generations
                    WHERE video_id = ? AND stage_kind = ?
                    """,
                    (job.video_id, stage_kind.value),
                )
                if stage_kind is StageKind.SCENES:
                    cursor = connection.execute(
                        "UPDATE videos SET thumbnail_path = NULL WHERE id = ?",
                        (job.video_id,),
                    )
                    if cursor.rowcount != 1:
                        raise RuntimeError("video index job lost its video")
                continue
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
                    selected.video_id,
                    stage_kind.value,
                    selected.generation_id,
                    selected.completed_at,
                ),
            )
            if stage_kind is StageKind.SCENES and staged is not None:
                thumbnail = connection.execute(
                    """
                    SELECT video_thumbnail_path, updates_video_thumbnail
                    FROM segment_generations
                    WHERE generation_id = ?
                    """,
                    (selected.generation_id,),
                ).fetchone()
                if thumbnail is None or thumbnail["updates_video_thumbnail"] not in {
                    0,
                    1,
                }:
                    raise ValueError("staged scene thumbnail intent is corrupt")
                if thumbnail["updates_video_thumbnail"] == 1:
                    cls._validate_generated_thumbnail_path(
                        thumbnail["video_thumbnail_path"]
                    )
                    cursor = connection.execute(
                        "UPDATE videos SET thumbnail_path = ? WHERE id = ?",
                        (thumbnail["video_thumbnail_path"], selected.video_id),
                    )
                    if cursor.rowcount != 1:
                        raise RuntimeError(
                            "staged scene generation lost its video"
                        )

        expected_inputs, _expected_points = cls._snapshot_text_vector_inputs(
            connection,
            video_id=job.video_id,
            source_sha256=job.source_sha256,
            semantic_specifications=(
                plan.specifications.semantic_segment_specifications
            ),
            job_id=job.job_id,
        )
        current_text = cls._get_active_text_vector_generation(
            connection,
            job.video_id,
            require_released_owner=False,
        )
        retained_text: TextVectorGeneration | None = None
        if (
            current_text is not None
            and current_text.specification_hash
            == plan.specifications.text_vectors.specification_hash
            and current_text.source_sha256 == job.source_sha256
            and cls._stage_run_owner_is_released(
                connection,
                current_text.run_id,
            )
        ):
            resolved_current_text = cls._get_text_vector_generation_with_run(
                connection,
                current_text.generation_id,
            )
            assert resolved_current_text is not None
            if resolved_current_text[3] == expected_inputs:
                retained_text = current_text
        if text_output is not None:
            assert text_inputs is not None
            if text_inputs != expected_inputs:
                raise ValueError(
                    "video index job text vector inputs do not match staged segments"
                )
            selected_text = text_output
        else:
            selected_text = retained_text
        if selected_text is None:
            connection.execute(
                "DELETE FROM active_text_vector_generations WHERE video_id = ?",
                (job.video_id,),
            )
        else:
            connection.execute(
                """
                INSERT INTO active_text_vector_generations (
                    video_id, generation_id, activated_at
                ) VALUES (?, ?, ?)
                ON CONFLICT(video_id) DO UPDATE SET
                    generation_id = excluded.generation_id,
                    activated_at = excluded.activated_at
                """,
                (
                    selected_text.video_id,
                    selected_text.generation_id,
                    selected_text.completed_at,
                ),
            )

        for planned in (
            plan.visual_dense_specification,
            plan.lighthouse_specification,
        ):
            stage_kind = planned.kind
            staged_external = external_outputs.get(stage_kind)
            current_external = cls._get_active_external_index_generation(
                connection,
                job.video_id,
                stage_kind,
                require_released_owner=False,
            )
            retained_external = (
                current_external
                if current_external is not None
                and current_external.specification_hash
                == planned.specification_hash
                and current_external.source_sha256 == job.source_sha256
                and cls._stage_run_owner_is_released(
                    connection,
                    current_external.run_id,
                )
                else None
            )
            selected_external = staged_external or retained_external
            if selected_external is None:
                connection.execute(
                    """
                    DELETE FROM active_external_index_generations
                    WHERE video_id = ? AND stage_kind = ?
                    """,
                    (job.video_id, stage_kind.value),
                )
                continue
            connection.execute(
                """
                INSERT INTO active_external_index_generations (
                    video_id, stage_kind, generation_id, activated_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(video_id, stage_kind) DO UPDATE SET
                    generation_id = excluded.generation_id,
                    activated_at = excluded.activated_at
                """,
                (
                    selected_external.video_id,
                    selected_external.stage_kind.value,
                    selected_external.generation_id,
                    selected_external.completed_at,
                ),
            )

    def complete_video_index_job(
        self,
        job_id: str,
        *,
        execution_token: str,
    ) -> VideoIndexJob:
        validate_artifact_identifier(job_id, field_name="video index job id")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            resolved = self._get_video_index_job(connection, job_id)
            if resolved is None:
                raise KeyError(job_id)
            current, plan = resolved
            if connection.execute(
                """
                SELECT 1 FROM stage_runs
                WHERE job_id = ? AND state IN ('queued', 'running')
                LIMIT 1
                """,
                (current.job_id,),
            ).fetchone() is not None:
                raise JobTransitionError(
                    "video index job still has active stage runs"
                )
            self._require_exact_video_index_job_stage_receipts(
                connection,
                job=current,
                plan=plan,
            )
            candidate = complete_job(
                current,
                execution_token=execution_token,
                now=_now(),
            )
            persisted = self._persist_video_index_transition(
                connection,
                current=current,
                candidate=candidate,
            )
            self._activate_video_index_job_outputs(
                connection,
                job=persisted,
                plan=plan,
            )
            self._sync_video_for_index_job(connection, persisted)
            return persisted

    def fail_video_index_job(
        self,
        job_id: str,
        *,
        execution_token: str,
        error_code: str,
    ) -> VideoIndexJob:
        validate_artifact_identifier(job_id, field_name="video index job id")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            resolved = self._get_video_index_job(connection, job_id)
            if resolved is None:
                raise KeyError(job_id)
            current, _plan = resolved
            candidate = fail_job(
                current,
                execution_token=execution_token,
                error_code=error_code,
                now=_now(),
            )
            persisted = self._persist_video_index_transition(
                connection,
                current=current,
                candidate=candidate,
            )
            self._terminalize_linked_stage_runs(
                connection,
                job_id=current.job_id,
                timestamp=persisted.updated_at,
                failure_error_code=error_code,
            )
            self._sync_video_for_index_job(connection, persisted)
            return persisted

    def cancel_video_index_job(
        self,
        job_id: str,
        *,
        execution_token: str,
    ) -> VideoIndexJob:
        validate_artifact_identifier(job_id, field_name="video index job id")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            resolved = self._get_video_index_job(connection, job_id)
            if resolved is None:
                raise KeyError(job_id)
            current, _plan = resolved
            candidate = cancel_job(
                current,
                execution_token=execution_token,
                now=_now(),
            )
            persisted = self._persist_video_index_transition(
                connection,
                current=current,
                candidate=candidate,
            )
            self._terminalize_linked_stage_runs(
                connection,
                job_id=current.job_id,
                timestamp=persisted.updated_at,
                failure_error_code=None,
            )
            self._sync_video_for_index_job(connection, persisted)
            return persisted

    def retry_video_index_job(
        self,
        job_id: str,
        *,
        retry_job_id: str | None = None,
    ) -> VideoIndexJob:
        validate_artifact_identifier(job_id, field_name="video index job id")
        resolved_retry_id = uuid4().hex if retry_job_id is None else retry_job_id
        validate_artifact_identifier(
            resolved_retry_id,
            field_name="video index retry job id",
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            resolved = self._get_video_index_job(connection, job_id)
            if resolved is None:
                raise KeyError(job_id)
            current, plan = resolved
            child_row = connection.execute(
                f"{_VIDEO_INDEX_JOB_SELECT} WHERE retry_of_job_id = ?",
                (job_id,),
            ).fetchone()
            if child_row is not None:
                return self._video_index_job_from_row(connection, child_row)[0]
            candidate = retry_job(
                current,
                job_id=resolved_retry_id,
                now=_now(),
            )
            self._insert_video_index_job(
                connection,
                job=candidate,
                plan=plan,
            )
            self._sync_video_for_index_job(connection, candidate)
            row = connection.execute(
                f"{_VIDEO_INDEX_JOB_SELECT} WHERE job_id = ?",
                (candidate.job_id,),
            ).fetchone()
            assert row is not None
            return self._video_index_job_from_row(connection, row)[0]

    @staticmethod
    def _quarantine_video_index_job_row(
        connection: sqlite3.Connection,
        *,
        row: sqlite3.Row,
        timestamp: str,
    ) -> None:
        sequence = row["sequence"]
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
            raise ValueError("persisted video index job sequence is corrupt")
        connection.execute(
            """
            INSERT OR IGNORE INTO video_index_job_quarantines (
                job_sequence, error_code, quarantined_at
            ) VALUES (?, 'job_record_corrupt', ?)
            """,
            (sequence, timestamp),
        )
        persisted = connection.execute(
            """
            SELECT error_code FROM video_index_job_quarantines
            WHERE job_sequence = ?
            """,
            (sequence,),
        ).fetchone()
        if persisted is None or persisted["error_code"] != "job_record_corrupt":
            raise ValueError("video index job quarantine could not be persisted")

    @classmethod
    def _terminalize_linked_stage_runs(
        cls,
        connection: sqlite3.Connection,
        *,
        job_id: str,
        timestamp: str,
        failure_error_code: str | None,
    ) -> None:
        builds = connection.execute(
            """
            SELECT builds.*
            FROM text_vector_builds AS builds
            JOIN stage_runs AS runs ON runs.run_id = builds.run_id
            WHERE runs.job_id = ? AND builds.recovery_state = 'active'
            ORDER BY builds.reserved_at, builds.generation_id
            """,
            (job_id,),
        ).fetchall()
        cleanup_error_code = failure_error_code or "job_cancelled"
        for build in builds:
            run = cls._get_stage_run(connection, build["run_id"])
            if run is None:
                raise ValueError("linked text vector build run is corrupt")
            cls._validate_text_vector_build_row(connection, build, run)
            cls._enqueue_artifact_gc(
                connection,
                generation_id=build["generation_id"],
                index_specification_hash=build["index_specification_hash"],
                collection_name=build["collection_name"],
                reason=cleanup_error_code,
                timestamp=timestamp,
            )
            connection.execute(
                "DELETE FROM text_vector_builds WHERE generation_id = ?",
                (build["generation_id"],),
            )
        if failure_error_code is not None:
            connection.execute(
                """
                UPDATE stage_runs
                SET state = 'failed', output_generation = NULL,
                    error_code = ?,
                    finished_at = ?, updated_at = ?
                WHERE job_id = ? AND state = 'running'
                """,
                (failure_error_code, timestamp, timestamp, job_id),
            )
        else:
            connection.execute(
                """
                UPDATE stage_runs
                SET state = 'cancelled', output_generation = NULL,
                    error_code = NULL, finished_at = ?, updated_at = ?
                WHERE job_id = ? AND state = 'running'
                """,
                (timestamp, timestamp, job_id),
            )
        connection.execute(
            """
            UPDATE stage_runs
            SET state = 'cancelled', output_generation = NULL,
                error_code = NULL, finished_at = ?, updated_at = ?
            WHERE job_id = ? AND state = 'queued'
            """,
            (timestamp, timestamp, job_id),
        )

    def recover_abandoned_video_index_jobs(
        self,
        *,
        limit: int = 100,
    ) -> VideoIndexRecoveryReport:
        resolved_limit = _validate_repository_batch_limit(
            limit,
            field_name="video index recovery limit",
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                f"""
                {_VIDEO_INDEX_JOB_SELECT}
                WHERE state = 'running'
                  AND NOT EXISTS (
                      SELECT 1 FROM video_index_job_quarantines AS quarantines
                      WHERE quarantines.job_sequence = video_index_jobs.sequence
                  )
                ORDER BY sequence
                LIMIT ?
                """,
                (resolved_limit + 1,),
            ).fetchall()
            has_more = len(rows) > resolved_limit
            retry_job_ids: list[str] = []
            cancelled_job_ids: list[str] = []
            quarantined_count = 0
            for row in rows[:resolved_limit]:
                try:
                    current, plan = self._video_index_job_from_row(connection, row)
                except ValueError:
                    self._quarantine_video_index_job_row(
                        connection,
                        row=row,
                        timestamp=_now(),
                    )
                    quarantined_count += 1
                    continue
                assert current.execution_token is not None
                timestamp = _now()
                if current.cancel_requested_at is not None:
                    candidate = cancel_job(
                        current,
                        execution_token=current.execution_token,
                        now=timestamp,
                    )
                    persisted = self._persist_video_index_transition(
                        connection,
                        current=current,
                        candidate=candidate,
                    )
                    self._terminalize_linked_stage_runs(
                        connection,
                        job_id=current.job_id,
                        timestamp=timestamp,
                        failure_error_code=None,
                    )
                    self._sync_video_for_index_job(connection, persisted)
                    cancelled_job_ids.append(current.job_id)
                    continue

                failed_candidate = fail_job(
                    current,
                    execution_token=current.execution_token,
                    error_code="job_runtime_abandoned",
                    now=timestamp,
                )
                failed = self._persist_video_index_transition(
                    connection,
                    current=current,
                    candidate=failed_candidate,
                )
                self._terminalize_linked_stage_runs(
                    connection,
                    job_id=current.job_id,
                    timestamp=timestamp,
                    failure_error_code="job_runtime_abandoned",
                )
                child = retry_job(
                    failed,
                    job_id=uuid4().hex,
                    now=timestamp,
                )
                self._insert_video_index_job(
                    connection,
                    job=child,
                    plan=plan,
                )
                self._sync_video_for_index_job(connection, child)
                retry_job_ids.append(child.job_id)
            return VideoIndexRecoveryReport(
                examined_count=min(len(rows), resolved_limit),
                retry_job_ids=tuple(retry_job_ids),
                cancelled_job_ids=tuple(cancelled_job_ids),
                quarantined_count=quarantined_count,
                has_more=has_more,
            )

    def list_legacy_video_index_candidates(
        self,
        *,
        limit: int = 100,
    ) -> tuple[VideoRecord, ...]:
        resolved_limit = _validate_repository_batch_limit(
            limit,
            field_name="legacy video index candidate limit",
        )
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT videos.*
                FROM videos
                WHERE videos.status IN ('queued', 'processing')
                  AND NOT EXISTS (
                      SELECT 1 FROM video_index_jobs AS jobs
                      WHERE jobs.video_id = videos.id
                  )
                ORDER BY videos.created_at, videos.id
                LIMIT ?
                """,
                (resolved_limit,),
            ).fetchall()
        return tuple(VideoRecord(**dict(row)) for row in rows)

    def adopt_legacy_video_index_job(
        self,
        video_id: str,
        *,
        plan: VideoIndexPlanSnapshot,
        job_id: str | None = None,
    ) -> VideoIndexJob:
        if not isinstance(plan, VideoIndexPlanSnapshot):
            raise ValueError("video index plan must be validated")
        validate_artifact_identifier(video_id, field_name="legacy video index video id")
        resolved_job_id = uuid4().hex if job_id is None else job_id
        validate_artifact_identifier(
            resolved_job_id,
            field_name="legacy video index job id",
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            video_row = connection.execute(
                "SELECT * FROM videos WHERE id = ?",
                (video_id,),
            ).fetchone()
            if video_row is None:
                raise KeyError(video_id)
            video = VideoRecord(**dict(video_row))
            asset = self._get_video_asset(connection, video_id)
            if asset is None:
                raise AssetIdentityError("legacy video asset identity is unavailable")
            requested_hash = video_index_idempotency_hash(
                kind=JobKind.VIDEO_INDEX,
                intent=VideoIndexIntent.INGEST,
                video_id=video_id,
                source_sha256=asset.sha256,
                plan_hash=plan.plan_hash,
            )
            existing_row = connection.execute(
                f"""
                {_VIDEO_INDEX_JOB_SELECT}
                WHERE video_id = ?
                ORDER BY sequence DESC LIMIT 1
                """,
                (video_id,),
            ).fetchone()
            if existing_row is not None:
                existing, existing_plan = self._video_index_job_from_row(
                    connection,
                    existing_row,
                )
                if (
                    existing.state in {JobState.QUEUED, JobState.RUNNING}
                    and existing.idempotency_hash == requested_hash
                    and existing_plan.canonical_json == plan.canonical_json
                ):
                    return existing
                raise JobTransitionError("video already has durable indexing history")
            if video.status not in {"queued", "processing"}:
                raise JobTransitionError("video is not a legacy active indexing candidate")
            timestamp = _now()
            candidate = self._new_queued_video_index_job(
                job_id=resolved_job_id,
                intent=VideoIndexIntent.INGEST,
                video_id=video_id,
                source_sha256=asset.sha256,
                plan=plan,
                prior_video_state=None,
                timestamp=timestamp,
            )
            self._insert_video_index_job(
                connection,
                job=candidate,
                plan=plan,
            )
            self._sync_video_for_index_job(connection, candidate)
            row = connection.execute(
                f"{_VIDEO_INDEX_JOB_SELECT} WHERE job_id = ?",
                (candidate.job_id,),
            ).fetchone()
            assert row is not None
            return self._video_index_job_from_row(connection, row)[0]

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
        return self._repository_assets_from_rows(rows)

    def find_assets_by_sha256_bounded(
        self,
        digest: str,
        *,
        limit: int,
        video_id: str | None = None,
    ) -> tuple[RepositoryAssetRecord, ...]:
        """Return at most ``limit + 1`` rows for bounded benchmark resolution."""
        asset_id_for_sha256(digest)
        if type(limit) is not int or not 1 <= limit <= 1_000:
            raise ValueError("asset lookup limit is invalid")
        if video_id is not None and (
            type(video_id) is not str
            or not video_id
            or len(video_id) > 200
            or "\x00" in video_id
        ):
            raise ValueError("asset lookup video id is invalid")
        predicate = "AND videos.id = ?" if video_id is not None else ""
        parameters: tuple[object, ...] = (
            (digest, video_id, limit + 1)
            if video_id is not None
            else (digest, limit + 1)
        )
        with self._connect() as connection:
            rows = connection.execute(
                f"""
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
                {predicate}
                ORDER BY videos.created_at, videos.id
                LIMIT ?
                """,
                parameters,
            ).fetchall()
        return self._repository_assets_from_rows(rows)

    @staticmethod
    def _repository_assets_from_rows(
        rows: Iterable[sqlite3.Row],
    ) -> tuple[RepositoryAssetRecord, ...]:
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
        media_access_root: Path | None = None,
    ) -> str:
        try:
            canonical_root = Path(os.path.abspath(media_root))
            if media_access_root is None:
                _reject_symbolic_link_components(canonical_root)
                resolved_root = canonical_root.resolve(strict=True)
            else:
                access_root = Path(os.path.abspath(media_access_root))
                _reject_symbolic_link_components(access_root)
                access_stat = _stat_retained_volume_path(access_root)
                if access_stat is None or not stat.S_ISDIR(access_stat.st_mode):
                    raise AssetIdentityError(
                        "asset media must use a retained managed directory"
                    )
                resolved_root = access_root
            if Path(video.stored_name).name != video.stored_name:
                raise AssetIdentityError(
                    "asset media must be a managed regular file"
                )
            media_path = Path(video.media_path)
            if (
                type(video.media_path) is not str
                or not video.media_path
                or "\x00" in video.media_path
                or media_path.as_posix() != video.media_path
            ):
                raise AssetIdentityError(
                    "asset media must be a managed regular file"
                )
            expected_path = resolved_root / video.stored_name
            _reject_symbolic_link_components(expected_path)
            expected_media = (
                expected_path.resolve(strict=True)
                if media_access_root is None
                else expected_path
            )
            if media_path.is_absolute():
                if media_access_root is None:
                    _reject_symbolic_link_components(media_path)
                    persisted_matches = media_path.resolve(strict=True) == expected_media
                else:
                    persisted_matches = Path(os.path.abspath(media_path)) == (
                        canonical_root / video.stored_name
                    )
            else:
                parts = media_path.parts
                parent_parts = parts[:-1]
                root_parts = canonical_root.parts
                persisted_matches = (
                    bool(parts)
                    and parts[-1] == video.stored_name
                    and all(part not in {"", ".", ".."} for part in parts)
                    and len(parent_parts) <= len(root_parts)
                    and (
                        not parent_parts
                        or tuple(root_parts[-len(parent_parts) :]) == parent_parts
                    )
                )
            resolved_media = expected_media
        except (OSError, RuntimeError, ValueError) as error:
            raise AssetIdentityError("asset media must be a managed regular file") from error
        if (
            not stat.S_ISDIR(os.stat(resolved_root, follow_symlinks=False).st_mode)
            or not persisted_matches
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

    def verify_existing_asset_identity(
        self,
        video_id: str,
        *,
        media_root: Path,
        media_access_root: Path | None = None,
    ) -> AssetRecord:
        """Re-hash managed media and require its already-persisted Asset identity."""
        with _asset_identity_lock(self.database_path, video_id):
            video = self.get_video(video_id)
            if video is None:
                raise KeyError(video_id)
            existing = self.get_video_asset(video_id)
            if existing is None:
                raise AssetIdentityError("video asset identity is unavailable")
            source_sha256 = self._hash_video_source(
                video,
                media_root=media_root,
                media_access_root=media_access_root,
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
        job_id: str | None = None,
        execution_token: str | None = None,
    ) -> StageRun:
        if not isinstance(specification, StageSpecification):
            raise ValueError("stage specification must be validated")
        resolved_run_id = uuid4().hex if run_id is None else run_id
        timestamp = _now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            stage_run_columns = {
                str(row[1])
                for row in connection.execute(
                    "PRAGMA table_info(stage_runs)"
                ).fetchall()
            }
            supports_job_link = "job_id" in stage_run_columns
            if job_id is not None and not supports_job_link:
                raise RuntimeError(
                    "stage run job ownership requires the durable job schema"
                )
            asset = self._get_video_asset(connection, video_id)
            if asset is None:
                raise AssetIdentityError("video asset identity is unavailable")
            source_sha256 = asset.sha256
            if job_id is None:
                if execution_token is not None:
                    raise ValueError("unowned stage run cannot contain an execution token")
                if supports_job_link and connection.execute(
                    "SELECT 1 FROM video_index_jobs WHERE video_id = ? LIMIT 1",
                    (video_id,),
                ).fetchone() is not None:
                    raise JobFenceError(
                        "durable video index job must own this stage run"
                    )
            else:
                validate_artifact_identifier(
                    job_id,
                    field_name="stage run video index job id",
                )
                resolved_job = self._get_video_index_job(connection, job_id)
                if resolved_job is None:
                    raise KeyError(job_id)
                owning_job, job_plan = resolved_job
                if (
                    owning_job.state is not JobState.RUNNING
                    or owning_job.execution_token != execution_token
                    or owning_job.cancel_requested_at is not None
                ):
                    raise JobFenceError(
                        "job execution token no longer owns this stage run"
                    )
                if (
                    owning_job.video_id != video_id
                    or owning_job.source_sha256 != source_sha256
                ):
                    raise ValueError("stage run video index job identity mismatch")
                try:
                    planned_specification = job_plan.for_kind(specification.kind)
                except ValueError as error:
                    raise ValueError(
                        "stage run is not part of the video index plan"
                    ) from error
                if (
                    planned_specification.specification_hash
                    != specification.specification_hash
                ):
                    raise ValueError(
                        "stage run specification is not part of the video index plan"
                    )
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
                if supports_job_link:
                    connection.execute(
                        """
                        INSERT INTO stage_runs (
                            run_id, video_id, specification_hash, source_sha256,
                            attempt, state, output_generation, error_code,
                            retry_of_run_id, created_at, started_at, finished_at,
                            updated_at, job_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                            job_id,
                        ),
                    )
                else:
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

    @staticmethod
    def _require_stage_run_job_fence(
        connection: sqlite3.Connection,
        *,
        run_id: str,
        execution_token: str | None,
    ) -> None:
        jobs_table = connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table' AND name = 'video_index_jobs'
            """
        ).fetchone()
        stage_columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(stage_runs)").fetchall()
        }
        if jobs_table is None or "job_id" not in stage_columns:
            if execution_token is not None:
                raise JobFenceError(
                    "legacy stage run cannot use a job execution token"
                )
            return
        owner = connection.execute(
            """
            SELECT
                runs.job_id,
                jobs.state AS job_state,
                jobs.execution_token AS job_execution_token,
                jobs.cancel_requested_at
            FROM stage_runs AS runs
            LEFT JOIN video_index_jobs AS jobs ON jobs.job_id = runs.job_id
            WHERE runs.run_id = ?
            """,
            (run_id,),
        ).fetchone()
        if owner is None:
            raise KeyError(run_id)
        if owner["job_id"] is None:
            if execution_token is not None:
                raise JobFenceError("unowned stage run cannot use a job execution token")
            return
        if (
            type(execution_token) is not str
            or owner["job_state"] != JobState.RUNNING.value
            or owner["job_execution_token"] != execution_token
            or owner["cancel_requested_at"] is not None
        ):
            raise JobFenceError(
                "job execution token no longer owns this stage run"
            )

    @staticmethod
    def _stage_run_job_id(
        connection: sqlite3.Connection,
        run_id: str,
    ) -> str | None:
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(stage_runs)").fetchall()
        }
        if "job_id" not in columns:
            return None
        row = connection.execute(
            "SELECT job_id FROM stage_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise KeyError(run_id)
        value = row["job_id"]
        if value is not None and (type(value) is not str or not value):
            raise ValueError("persisted stage run job link is corrupt")
        return value

    @classmethod
    def _stage_run_owner_is_released(
        cls,
        connection: sqlite3.Connection,
        run_id: str,
    ) -> bool:
        owner_job_id = cls._stage_run_job_id(connection, run_id)
        if owner_job_id is None:
            return True
        row = connection.execute(
            "SELECT state FROM video_index_jobs WHERE job_id = ?",
            (owner_job_id,),
        ).fetchone()
        return row is not None and row["state"] == JobState.COMPLETE.value

    def transition_stage_run(
        self,
        run_id: str,
        state: StageState,
        *,
        output_generation: str | None = None,
        error_code: str | None = None,
        execution_token: str | None = None,
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
            self._require_stage_run_job_fence(
                connection,
                run_id=current.run_id,
                execution_token=execution_token,
            )
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
        *,
        require_released_owner: bool = True,
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
        if require_released_owner and not cls._stage_run_owner_is_released(
            connection,
            stage_run.run_id,
        ):
            raise ValueError(
                "active segment generation owner job is not complete"
            )
        return generation

    def commit_segment_generation(
        self,
        run_id: str,
        *,
        segments: Iterable[SegmentRecord],
        generation_id: str | None = None,
        video_thumbnail_path: str | None | object = _VIDEO_THUMBNAIL_UNCHANGED,
        execution_token: str | None = None,
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
            self._require_stage_run_job_fence(
                connection,
                run_id=stage_run.run_id,
                execution_token=execution_token,
            )
            owner_job_id = self._stage_run_job_id(
                connection,
                stage_run.run_id,
            )
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
            generation_columns = {
                str(row[1])
                for row in connection.execute(
                    "PRAGMA table_info(segment_generations)"
                ).fetchall()
            }
            if {
                "updates_video_thumbnail",
                "video_thumbnail_path",
            } <= generation_columns:
                connection.execute(
                    """
                    INSERT INTO segment_generations (
                        generation_id, video_id, stage_kind, specification_hash,
                        source_sha256, run_id, segment_count, completed_at,
                        video_thumbnail_path, updates_video_thumbnail
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        video_thumbnail_path if updates_video_thumbnail else None,
                        int(updates_video_thumbnail),
                    ),
                )
            else:
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
            if owner_job_id is None:
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
            if owner_job_id is None and updates_video_thumbnail:
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

    @classmethod
    def _get_external_index_generation_with_run(
        cls,
        connection: sqlite3.Connection,
        *,
        video_id: str,
        stage_kind: StageKind,
        generation_id: str,
    ) -> tuple[ExternalIndexGeneration, StageRun] | None:
        if stage_kind not in EXTERNAL_INDEX_STAGE_KINDS:
            raise ValueError("external index generation requires an external stage")
        row = connection.execute(
            """
            SELECT video_id, stage_kind, generation_id, specification_hash,
                   artifact_specification_hash, source_sha256, run_id,
                   descriptor_json, descriptor_sha256, completed_at
            FROM external_index_generations
            WHERE video_id = ? AND stage_kind = ? AND generation_id = ?
            """,
            (video_id, stage_kind.value, generation_id),
        ).fetchone()
        if row is None:
            return None
        try:
            persisted_kind = StageKind(row["stage_kind"])
            raw_descriptor = json.loads(
                row["descriptor_json"],
                parse_constant=_reject_non_finite_json,
            )
            asset = cls._get_video_asset(connection, row["video_id"])
            if asset is None:
                raise ValueError("external generation asset is missing")
            descriptor, canonical, descriptor_sha256 = (
                _validate_external_index_descriptor(
                    raw_descriptor,
                    stage_kind=persisted_kind,
                    video_id=row["video_id"],
                    generation_id=row["generation_id"],
                    artifact_specification_hash=row[
                        "artifact_specification_hash"
                    ],
                    source_sha256=row["source_sha256"],
                    source_size_bytes=asset.size_bytes,
                    duration_seconds=_external_video_duration(
                        connection,
                        row["video_id"],
                    ),
                )
            )
        except (KeyError, RecursionError, TypeError, ValueError) as error:
            raise ValueError("persisted external index generation is corrupt") from error
        if (
            persisted_kind is not stage_kind
            or row["descriptor_json"] != canonical
            or row["descriptor_sha256"] != descriptor_sha256
        ):
            raise ValueError("persisted external index generation is corrupt")
        try:
            stage_run = cls._get_stage_run(connection, row["run_id"])
        except (TypeError, ValueError) as error:
            raise ValueError("persisted external index generation run is corrupt") from error
        if stage_run is None:
            raise ValueError("persisted external index generation run is missing")
        if (
            stage_run.video_id != row["video_id"]
            or stage_run.stage_kind is not persisted_kind
            or stage_run.specification_hash != row["specification_hash"]
            or stage_run.source_sha256 != row["source_sha256"]
            or stage_run.output_generation != row["generation_id"]
            or stage_run.state not in {StageState.COMPLETE, StageState.STALE}
            or stage_run.finished_at != row["completed_at"]
        ):
            raise ValueError("persisted external index generation identity is corrupt")
        stage_specification = cls._get_stage_specification_from_connection(
            connection,
            stage_run.specification_hash,
        )
        if (
            _external_artifact_specification_hash(stage_specification)
            != row["artifact_specification_hash"]
        ):
            raise ValueError("persisted external index generation identity is corrupt")
        return (
            ExternalIndexGeneration(
                video_id=row["video_id"],
                stage_kind=persisted_kind,
                generation_id=row["generation_id"],
                specification_hash=row["specification_hash"],
                artifact_specification_hash=row[
                    "artifact_specification_hash"
                ],
                source_sha256=row["source_sha256"],
                run_id=row["run_id"],
                descriptor=descriptor,
                descriptor_sha256=descriptor_sha256,
                completed_at=row["completed_at"],
            ),
            stage_run,
        )

    @classmethod
    def _get_active_external_index_generation(
        cls,
        connection: sqlite3.Connection,
        video_id: str,
        stage_kind: StageKind,
        *,
        require_released_owner: bool = True,
    ) -> ExternalIndexGeneration | None:
        if stage_kind not in EXTERNAL_INDEX_STAGE_KINDS:
            raise ValueError("active external index requires an external stage")
        pointer = connection.execute(
            """
            SELECT generation_id, activated_at
            FROM active_external_index_generations
            WHERE video_id = ? AND stage_kind = ?
            """,
            (video_id, stage_kind.value),
        ).fetchone()
        if pointer is None:
            return None
        resolved = cls._get_external_index_generation_with_run(
            connection,
            video_id=video_id,
            stage_kind=stage_kind,
            generation_id=pointer["generation_id"],
        )
        if resolved is None:
            raise ValueError("active external index generation pointer is dangling")
        generation, run = resolved
        if pointer["activated_at"] != generation.completed_at:
            raise ValueError("active external index generation pointer is corrupt")
        if require_released_owner and not cls._stage_run_owner_is_released(
            connection,
            run.run_id,
        ):
            raise ValueError("active external generation owner job is not complete")
        return generation

    def commit_external_index_generation(
        self,
        run_id: str,
        *,
        descriptor: object,
        execution_token: str | None = None,
    ) -> ExternalIndexGeneration:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            stage_run = self._get_stage_run(connection, run_id)
            if stage_run is None:
                raise KeyError(run_id)
            self._require_stage_run_job_fence(
                connection,
                run_id=stage_run.run_id,
                execution_token=execution_token,
            )
            if (
                stage_run.state is not StageState.RUNNING
                or stage_run.stage_kind not in EXTERNAL_INDEX_STAGE_KINDS
            ):
                raise ValueError(
                    "only a running external stage can publish a generation"
                )
            owner_job_id = self._stage_run_job_id(connection, stage_run.run_id)
            asset = self._get_video_asset(connection, stage_run.video_id)
            if asset is None:
                raise AssetIdentityError("external generation asset is unavailable")
            generation_id = (
                descriptor.get("generation_id")
                if isinstance(descriptor, dict)
                else None
            )
            if (
                type(generation_id) is not str
                or _EXTERNAL_GENERATION_ID_RE.fullmatch(generation_id) is None
            ):
                raise ValueError("external index canonical descriptor is invalid")
            stage_specification = self._get_stage_specification_from_connection(
                connection,
                stage_run.specification_hash,
            )
            artifact_specification_hash = _external_artifact_specification_hash(
                stage_specification
            )
            normalized, canonical, descriptor_sha256 = (
                _validate_external_index_descriptor(
                    descriptor,
                    stage_kind=stage_run.stage_kind,
                    video_id=stage_run.video_id,
                    generation_id=generation_id,
                    artifact_specification_hash=artifact_specification_hash,
                    source_sha256=stage_run.source_sha256,
                    source_size_bytes=asset.size_bytes,
                    duration_seconds=_external_video_duration(
                        connection,
                        stage_run.video_id,
                    ),
                )
            )
            completed_at = _now()
            generation = ExternalIndexGeneration(
                video_id=stage_run.video_id,
                stage_kind=stage_run.stage_kind,
                generation_id=generation_id,
                specification_hash=stage_run.specification_hash,
                artifact_specification_hash=artifact_specification_hash,
                source_sha256=stage_run.source_sha256,
                run_id=stage_run.run_id,
                descriptor=normalized,
                descriptor_sha256=descriptor_sha256,
                completed_at=completed_at,
            )
            connection.execute(
                """
                INSERT INTO external_index_generations (
                    video_id, stage_kind, generation_id, specification_hash,
                    artifact_specification_hash, source_sha256, run_id,
                    descriptor_json, descriptor_sha256, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    generation.video_id,
                    generation.stage_kind.value,
                    generation.generation_id,
                    generation.specification_hash,
                    generation.artifact_specification_hash,
                    generation.source_sha256,
                    generation.run_id,
                    canonical,
                    generation.descriptor_sha256,
                    generation.completed_at,
                ),
            )
            if owner_job_id is None:
                connection.execute(
                    """
                    INSERT INTO active_external_index_generations (
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
            cursor = connection.execute(
                """
                UPDATE stage_runs
                SET state = 'complete', output_generation = ?, error_code = NULL,
                    finished_at = ?, updated_at = ?
                WHERE run_id = ? AND state = 'running'
                """,
                (
                    generation.generation_id,
                    generation.completed_at,
                    generation.completed_at,
                    generation.run_id,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("external stage run changed during generation commit")
        return generation

    def get_active_external_index_generation(
        self,
        video_id: str,
        stage_kind: StageKind,
    ) -> ExternalIndexGeneration | None:
        try:
            resolved_kind = StageKind(stage_kind)
        except (TypeError, ValueError) as error:
            raise ValueError("unsupported external index stage") from error
        with self._connect() as connection:
            return self._get_active_external_index_generation(
                connection,
                video_id,
                resolved_kind,
            )

    def get_external_index_release_snapshot(
        self,
        video_ids: Iterable[str],
    ) -> ExternalIndexReleaseSnapshot:
        try:
            selected = tuple(dict.fromkeys(video_ids))
        except TypeError as error:
            raise ValueError("video ids must be a finite collection") from error
        if (
            len(selected) > REPOSITORY_BATCH_LIMIT_MAX
            or any(type(video_id) is not str or not video_id for video_id in selected)
        ):
            raise ValueError("external release video ids are invalid")
        if not selected:
            return ExternalIndexReleaseSnapshot(frozenset(), {}, {})
        placeholders = ",".join("?" for _ in selected)
        visual: dict[str, dict[str, object]] = {}
        lighthouse: dict[str, dict[str, object]] = {}
        with self._connect() as connection:
            connection.execute("BEGIN")
            job_rows = connection.execute(
                f"""
                SELECT DISTINCT video_id FROM video_index_jobs
                WHERE video_id IN ({placeholders})
                """,
                selected,
            ).fetchall()
            job_backed = frozenset(row["video_id"] for row in job_rows)
            for video_id in selected:
                if video_id not in job_backed:
                    continue
                for kind, target in (
                    (StageKind.VISUAL_DENSE, visual),
                    (StageKind.LIGHTHOUSE, lighthouse),
                ):
                    generation = self._get_active_external_index_generation(
                        connection,
                        video_id,
                        kind,
                    )
                    if generation is not None:
                        target[video_id] = dict(generation.descriptor)
        return ExternalIndexReleaseSnapshot(job_backed, visual, lighthouse)

    def get_segment_generation(self, generation_id: str) -> SegmentGeneration | None:
        with self._connect() as connection:
            resolved = self._get_segment_generation_with_run(connection, generation_id)
        return resolved[0] if resolved is not None else None

    def get_segment_generation_snapshot(
        self,
        generation_id: str,
    ) -> tuple[SegmentGeneration, tuple[SegmentRecord, ...]] | None:
        """Load one immutable generation without consulting its active pointer."""
        with self._connect() as connection:
            resolved = self._get_segment_generation_with_run(connection, generation_id)
            if resolved is None:
                return None
            generation, _run = resolved
            segments = tuple(self._list_generation_segments(connection, generation))
        return generation, segments

    @staticmethod
    def _validate_benchmark_read_limits(
        *,
        max_segments: int,
        max_text_bytes: int,
        max_metadata_bytes: int,
    ) -> None:
        for name, value, maximum in (
            ("max_segments", max_segments, 1_000_000),
            ("max_text_bytes", max_text_bytes, 1_073_741_824),
            ("max_metadata_bytes", max_metadata_bytes, 1_073_741_824),
        ):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"benchmark {name} is invalid")

    @classmethod
    def _assert_benchmark_segment_generation_bounded(
        cls,
        connection: sqlite3.Connection,
        generation_id: str,
        *,
        max_segments: int,
        max_text_bytes: int,
        max_metadata_bytes: int,
    ) -> tuple[int, int, int] | None:
        row = connection.execute(
            """
            SELECT generation.segment_count,
                   length(CAST(specification.canonical_json AS BLOB)) AS specification_bytes
            FROM segment_generations AS generation
            JOIN stage_specifications AS specification
              ON specification.specification_hash = generation.specification_hash
            WHERE generation.generation_id = ?
            """,
            (generation_id,),
        ).fetchone()
        if row is None:
            return None
        declared_count = row["segment_count"]
        specification_bytes = row["specification_bytes"]
        if (
            isinstance(declared_count, bool)
            or not isinstance(declared_count, int)
            or declared_count < 0
            or declared_count > max_segments
        ):
            raise ValueError("persisted segment generation exceeds benchmark segment limit")
        if (
            isinstance(specification_bytes, bool)
            or not isinstance(specification_bytes, int)
            or specification_bytes < 0
            or specification_bytes > max_metadata_bytes
        ):
            raise ValueError("persisted stage specification exceeds benchmark metadata limit")
        cursor = connection.execute(
            """
            SELECT length(CAST(text AS BLOB)) AS text_bytes,
                   length(CAST(metadata_json AS BLOB))
                       + length(CAST(id AS BLOB))
                       + length(CAST(video_id AS BLOB))
                       + length(CAST(modality AS BLOB))
                       + COALESCE(length(CAST(thumbnail_path AS BLOB)), 0)
                       AS metadata_bytes
            FROM segments
            WHERE generation_id = ?
            LIMIT ?
            """,
            (generation_id, max_segments + 1),
        )
        actual_count = 0
        text_bytes = 0
        metadata_bytes = 0
        for segment_row in cursor:
            actual_count += 1
            if actual_count > max_segments:
                raise ValueError(
                    "persisted segment generation exceeds benchmark segment limit"
                )
            row_text_bytes = segment_row["text_bytes"]
            row_metadata_bytes = segment_row["metadata_bytes"]
            if (
                isinstance(row_text_bytes, bool)
                or not isinstance(row_text_bytes, int)
                or row_text_bytes < 0
                or isinstance(row_metadata_bytes, bool)
                or not isinstance(row_metadata_bytes, int)
                or row_metadata_bytes < 0
            ):
                raise ValueError("persisted segment generation size is corrupt")
            text_bytes += row_text_bytes
            metadata_bytes += row_metadata_bytes
            if text_bytes > max_text_bytes or metadata_bytes > max_metadata_bytes:
                raise ValueError(
                    "persisted segment generation exceeds benchmark read limits"
                )
        if actual_count != declared_count:
            raise ValueError("persisted segment generation exceeds benchmark read limits")
        return declared_count, text_bytes, metadata_bytes

    def get_segment_generation_snapshot_bounded(
        self,
        generation_id: str,
        *,
        max_segments: int,
        max_text_bytes: int,
        max_metadata_bytes: int,
    ) -> tuple[SegmentGeneration, tuple[SegmentRecord, ...]] | None:
        """Load one generation only after scalar SQL proves bounded materialization."""
        self._validate_benchmark_read_limits(
            max_segments=max_segments,
            max_text_bytes=max_text_bytes,
            max_metadata_bytes=max_metadata_bytes,
        )
        with self._connect() as connection:
            bounded = self._assert_benchmark_segment_generation_bounded(
                connection,
                generation_id,
                max_segments=max_segments,
                max_text_bytes=max_text_bytes,
                max_metadata_bytes=max_metadata_bytes,
            )
            if bounded is None:
                return None
            count, _text_bytes, _metadata_bytes = bounded
            resolved = self._get_segment_generation_with_run(connection, generation_id)
            if resolved is None:
                return None
            generation, _run = resolved
            segments = tuple(self._list_generation_segments(connection, generation))
        if len(segments) != count:
            raise ValueError("persisted segment generation changed during benchmark read")
        return generation, segments

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
    def _get_job_segment_generation(
        cls,
        connection: sqlite3.Connection,
        *,
        job_id: str,
        video_id: str,
        source_sha256: str,
        specification: StageSpecification,
    ) -> SegmentGeneration | None:
        rows = connection.execute(
            """
            SELECT generations.generation_id
            FROM segment_generations AS generations
            JOIN stage_runs AS runs ON runs.run_id = generations.run_id
            WHERE runs.job_id = ?
              AND runs.state = 'complete'
              AND runs.output_generation = generations.generation_id
              AND generations.video_id = ?
              AND generations.stage_kind = ?
              AND generations.specification_hash = ?
              AND generations.source_sha256 = ?
            ORDER BY runs.sequence
            LIMIT 2
            """,
            (
                job_id,
                video_id,
                specification.kind.value,
                specification.specification_hash,
                source_sha256,
            ),
        ).fetchall()
        if len(rows) > 1:
            raise ValueError(
                "video index job has multiple staged segment generations"
            )
        if not rows:
            return None
        resolved = cls._get_segment_generation_with_run(
            connection,
            rows[0]["generation_id"],
        )
        if resolved is None:
            raise ValueError("staged segment generation is missing")
        generation, run = resolved
        if run.state is not StageState.COMPLETE:
            raise ValueError("staged segment generation is not complete")
        return generation

    @classmethod
    def _snapshot_text_vector_inputs(
        cls,
        connection: sqlite3.Connection,
        *,
        video_id: str,
        source_sha256: str,
        semantic_specifications: tuple[StageSpecification, ...],
        job_id: str | None = None,
    ) -> tuple[tuple[TextVectorGenerationInput, ...], tuple[TextVectorPointSource, ...]]:
        inputs: list[TextVectorGenerationInput] = []
        points: list[TextVectorPointSource] = []
        for specification in semantic_specifications:
            active = (
                cls._get_job_segment_generation(
                    connection,
                    job_id=job_id,
                    video_id=video_id,
                    source_sha256=source_sha256,
                    specification=specification,
                )
                if job_id is not None
                else None
            )
            if active is None:
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
        execution_token: str | None = None,
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
            self._require_stage_run_job_fence(
                connection,
                run_id=run.run_id,
                execution_token=execution_token,
            )
            owner_job_id = self._stage_run_job_id(connection, run.run_id)
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
                job_id=owner_job_id,
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
                    INSERT INTO text_vector_generation_tombstones (
                        generation_id, index_specification_hash, collection_name,
                        lifecycle_state, created_at, updated_at
                    ) VALUES (?, ?, ?, 'building', ?, ?)
                    """,
                    (
                        plan.generation_id,
                        plan.index_specification.specification_hash,
                        plan.index_specification.collection_name,
                        plan.reserved_at,
                        plan.reserved_at,
                    ),
                )
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
        *,
        require_released_owner: bool = True,
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
        if require_released_owner and not cls._stage_run_owner_is_released(
            connection,
            run.run_id,
        ):
            raise ValueError(
                "active text vector generation owner job is not complete"
            )
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

    def get_text_vector_search_binding(
        self,
        generation_id: str,
    ) -> TextVectorSearchBinding | None:
        """Load one immutable vector generation without following active state."""
        with self._connect() as connection:
            resolved = self._get_text_vector_generation_with_run(
                connection,
                generation_id,
            )
        if resolved is None:
            return None
        generation, _run, index_specification, inputs, points = resolved
        return TextVectorSearchBinding(
            generation=generation,
            index_specification=index_specification,
            inputs=inputs,
            points=points,
        )

    def get_text_vector_search_binding_bounded(
        self,
        generation_id: str,
        *,
        max_points: int,
        max_segments: int,
        max_text_bytes: int,
        max_metadata_bytes: int,
    ) -> TextVectorSearchBinding | None:
        """Load a benchmark binding only after bounded scalar preflight checks."""
        if type(max_points) is not int or not 1 <= max_points <= 1_000_000:
            raise ValueError("benchmark max_points is invalid")
        self._validate_benchmark_read_limits(
            max_segments=max_segments,
            max_text_bytes=max_text_bytes,
            max_metadata_bytes=max_metadata_bytes,
        )
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT generation.point_count,
                       length(CAST(stage.canonical_json AS BLOB))
                           AS stage_specification_bytes,
                       length(CAST(index_specification.canonical_json AS BLOB))
                           AS index_specification_bytes
                FROM text_vector_generations AS generation
                JOIN stage_specifications AS stage
                  ON stage.specification_hash = generation.specification_hash
                JOIN text_vector_index_specifications AS index_specification
                  ON index_specification.specification_hash =
                     generation.index_specification_hash
                WHERE generation.generation_id = ?
                """,
                (generation_id,),
            ).fetchone()
            if row is None:
                return None
            point_count = row["point_count"]
            if (
                isinstance(point_count, bool)
                or not isinstance(point_count, int)
                or point_count < 0
                or point_count > max_points
            ):
                raise ValueError(
                    "persisted text vector generation exceeds benchmark point limit"
                )
            specification_bytes = (
                row["stage_specification_bytes"]
                + row["index_specification_bytes"]
            )
            if (
                isinstance(specification_bytes, bool)
                or not isinstance(specification_bytes, int)
                or specification_bytes < 0
                or specification_bytes > max_metadata_bytes
            ):
                raise ValueError(
                    "persisted text vector specifications exceed benchmark metadata limit"
                )
            input_rows = connection.execute(
                """
                SELECT input.segment_generation_id,
                       length(CAST(specification.canonical_json AS BLOB))
                           AS specification_bytes
                FROM text_vector_generation_inputs AS input
                JOIN stage_specifications AS specification
                  ON specification.specification_hash = input.specification_hash
                WHERE input.generation_id = ?
                ORDER BY input.stage_kind
                LIMIT 4
                """,
                (generation_id,),
            ).fetchall()
            if len(input_rows) > len(TEXT_VECTOR_INPUT_STAGE_KINDS):
                raise ValueError("persisted text vector inputs exceed benchmark limits")
            remaining_segments = max_segments
            remaining_text_bytes = max_text_bytes
            remaining_metadata_bytes = max_metadata_bytes - specification_bytes
            for input_row in input_rows:
                input_specification_bytes = input_row["specification_bytes"]
                if (
                    isinstance(input_specification_bytes, bool)
                    or not isinstance(input_specification_bytes, int)
                    or input_specification_bytes < 0
                    or input_specification_bytes > remaining_metadata_bytes
                ):
                    raise ValueError(
                        "persisted text vector specifications exceed benchmark metadata limit"
                    )
                remaining_metadata_bytes -= input_specification_bytes
                segment_generation_id = input_row["segment_generation_id"]
                if segment_generation_id is None:
                    continue
                bounded = self._assert_benchmark_segment_generation_bounded(
                    connection,
                    segment_generation_id,
                    max_segments=remaining_segments,
                    max_text_bytes=remaining_text_bytes,
                    max_metadata_bytes=remaining_metadata_bytes,
                )
                if bounded is None:
                    raise ValueError(
                        "persisted text vector input generation is missing"
                    )
                count, text_bytes, metadata_bytes = bounded
                remaining_segments -= count
                remaining_text_bytes -= text_bytes
                remaining_metadata_bytes -= metadata_bytes
            resolved = self._get_text_vector_generation_with_run(
                connection,
                generation_id,
            )
        if resolved is None:
            return None
        generation, _run, index_specification, inputs, points = resolved
        return TextVectorSearchBinding(
            generation=generation,
            index_specification=index_specification,
            inputs=inputs,
            points=points,
        )

    def get_active_text_vector_generation(
        self,
        video_id: str,
    ) -> TextVectorGeneration | None:
        with self._connect() as connection:
            return self._get_active_text_vector_generation(connection, video_id)

    def get_active_text_vector_generation_id(self, video_id: str) -> str | None:
        """Read and validate the active pointer without materializing its corpus."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT pointer.generation_id,
                       pointer.activated_at,
                       generation.video_id,
                       generation.run_id,
                       generation.completed_at,
                       run.state,
                       run.output_generation,
                       run.job_id,
                       jobs.state AS job_state
                FROM active_text_vector_generations AS pointer
                LEFT JOIN text_vector_generations AS generation
                  ON generation.generation_id = pointer.generation_id
                LEFT JOIN stage_runs AS run
                  ON run.run_id = generation.run_id
                LEFT JOIN video_index_jobs AS jobs
                  ON jobs.job_id = run.job_id
                WHERE pointer.video_id = ?
                """,
                (video_id,),
            ).fetchone()
        if row is None:
            return None
        try:
            validate_artifact_identifier(
                row["generation_id"],
                field_name="active text vector generation id",
            )
        except ValueError as error:
            raise ValueError("active text vector generation pointer is corrupt") from error
        if (
            row["video_id"] != video_id
            or row["run_id"] is None
            or row["state"] != StageState.COMPLETE.value
            or row["output_generation"] != row["generation_id"]
            or row["activated_at"] != row["completed_at"]
        ):
            raise ValueError("active text vector generation pointer is corrupt")
        if (
            row["job_id"] is not None
            and row["job_state"] != JobState.COMPLETE.value
        ):
            raise ValueError(
                "active text vector generation owner job is not complete"
            )
        return str(row["generation_id"])

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
            if (
                build["recovery_state"] != "active"
                or build["recovery_error_code"] is not None
                or build["quarantined_at"] is not None
            ):
                raise ValueError("text vector build is not active")
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
        tombstone = connection.execute(
            """
            SELECT lifecycle_state, index_specification_hash, collection_name
            FROM text_vector_generation_tombstones
            WHERE generation_id = ?
            """,
            (build["generation_id"],),
        ).fetchone()
        if (
            tombstone is None
            or tombstone["lifecycle_state"] != "building"
            or tombstone["index_specification_hash"]
            != build["index_specification_hash"]
            or tombstone["collection_name"] != build["collection_name"]
        ):
            raise ValueError("persisted text vector build tombstone is corrupt")
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
        execution_token: str | None = None,
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
            self._require_stage_run_job_fence(
                connection,
                run_id=run.run_id,
                execution_token=execution_token,
            )
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
        execution_token: str | None = None,
    ) -> TextVectorGeneration:
        if not isinstance(receipt, TextVectorBuildReceipt):
            raise ValueError("text vector build receipt must be validated")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            run = self._get_stage_run(connection, run_id)
            if run is None:
                raise KeyError(run_id)
            self._require_stage_run_job_fence(
                connection,
                run_id=run.run_id,
                execution_token=execution_token,
            )
            owner_job_id = self._stage_run_job_id(connection, run.run_id)
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
                job_id=owner_job_id,
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
            if owner_job_id is None:
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
            cursor = connection.execute(
                """
                UPDATE text_vector_generation_tombstones
                SET lifecycle_state = 'committed', updated_at = ?
                WHERE generation_id = ? AND lifecycle_state = 'building'
                """,
                (completed_at, generation.generation_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("text vector generation tombstone changed during commit")
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
        if connection.execute(
            "SELECT 1 FROM text_vector_generations WHERE generation_id = ?",
            (generation_id,),
        ).fetchone() is not None:
            raise ValueError("artifact GC cannot target a committed generation")
        tombstone = connection.execute(
            """
            SELECT lifecycle_state, index_specification_hash, collection_name
            FROM text_vector_generation_tombstones
            WHERE generation_id = ?
            """,
            (generation_id,),
        ).fetchone()
        if (
            tombstone is None
            or tombstone["lifecycle_state"] not in {"building", "gc_pending"}
            or tombstone["index_specification_hash"] != index_specification_hash
            or tombstone["collection_name"] != collection_name
        ):
            raise ValueError("artifact GC generation tombstone is corrupt")
        if tombstone["lifecycle_state"] == "building":
            cursor = connection.execute(
                """
                UPDATE text_vector_generation_tombstones
                SET lifecycle_state = 'gc_pending', updated_at = ?
                WHERE generation_id = ? AND lifecycle_state = 'building'
                """,
                (timestamp, generation_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("artifact GC generation tombstone changed concurrently")
        connection.execute(
            """
            INSERT OR IGNORE INTO artifact_gc_jobs (
                job_id, artifact_kind, generation_id, index_specification_hash,
                collection_name, reason, state, attempt, backoff_level,
                available_at, worker_id, lease_token, lease_expires_at, error_code,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, ?, ?)
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
                0,
                timestamp,
                timestamp,
                timestamp,
            ),
        )

    def fail_text_vector_build(
        self,
        run_id: str,
        *,
        error_code: str,
        execution_token: str | None = None,
    ) -> StageRun:
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
            self._require_stage_run_job_fence(
                connection,
                run_id=run.run_id,
                execution_token=execution_token,
            )
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

    @classmethod
    def _terminalize_text_vector_build_rows(
        cls,
        connection: sqlite3.Connection,
        rows: Iterable[sqlite3.Row],
        *,
        error_code: str,
        timestamp: str,
    ) -> tuple[str, ...]:
        validate_error_code(error_code)
        terminalized: list[str] = []
        for build in rows:
            run = cls._get_stage_run(connection, build["run_id"])
            if run is None:
                raise ValueError("abandoned text vector build run is corrupt")
            cls._validate_text_vector_build_row(connection, build, run)
            cls._enqueue_artifact_gc(
                connection,
                generation_id=build["generation_id"],
                index_specification_hash=build["index_specification_hash"],
                collection_name=build["collection_name"],
                reason=error_code,
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
                        error_code,
                        timestamp,
                        timestamp,
                        run.run_id,
                        StageState.RUNNING.value,
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("text vector stage changed during recovery")
            elif run.state not in {StageState.FAILED, StageState.CANCELLED}:
                raise ValueError("abandoned text vector build run is corrupt")
            connection.execute(
                "DELETE FROM text_vector_builds WHERE generation_id = ?",
                (build["generation_id"],),
            )
            terminalized.append(build["generation_id"])
        return tuple(terminalized)

    @classmethod
    def _recover_text_vector_build_rows(
        cls,
        connection: sqlite3.Connection,
        rows: Iterable[sqlite3.Row],
        *,
        error_code: str,
        timestamp: str,
    ) -> tuple[tuple[str, ...], int]:
        terminalized: list[str] = []
        quarantined_count = 0
        for build in rows:
            connection.execute("SAVEPOINT recover_text_vector_build")
            try:
                recovered = cls._terminalize_text_vector_build_rows(
                    connection,
                    (build,),
                    error_code=error_code,
                    timestamp=timestamp,
                )
            except (KeyError, ValueError, sqlite3.IntegrityError):
                connection.execute("ROLLBACK TO recover_text_vector_build")
                connection.execute("RELEASE recover_text_vector_build")
                cursor = connection.execute(
                    """
                    UPDATE text_vector_builds
                    SET recovery_state = 'quarantined',
                        recovery_error_code = ?, quarantined_at = ?
                    WHERE rowid = ? AND recovery_state = 'active'
                    """,
                    (
                        "text_vector_build_metadata_corrupt",
                        timestamp,
                        build["build_row_id"],
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError(
                        "text vector build quarantine changed concurrently"
                    )
                quarantined_count += 1
            else:
                connection.execute("RELEASE recover_text_vector_build")
                terminalized.extend(recovered)
        return tuple(terminalized), quarantined_count

    def expire_text_vector_builds(
        self,
        *,
        before: str | None = None,
        limit: int = 100,
    ) -> tuple[str, ...]:
        report = self.recover_expired_text_vector_builds(
            before=before,
            limit=limit,
        )
        if report.quarantined_count:
            raise RuntimeError(
                "expired text vector build metadata was quarantined; "
                "inspect repository degraded health"
            )
        return report.terminalized_generation_ids

    def recover_expired_text_vector_builds(
        self,
        *,
        before: str | None = None,
        limit: int = 100,
    ) -> TextVectorBuildRecoveryReport:
        resolved_limit = _validate_repository_batch_limit(
            limit,
            field_name="text vector build expiry limit",
        )
        cutoff = _now() if before is None else before
        if type(cutoff) is not str:
            raise ValueError("text vector build expiry cutoff must be an ISO timestamp")
        try:
            resolved_cutoff = datetime.fromisoformat(cutoff)
        except ValueError as error:
            raise ValueError("text vector build expiry cutoff must be an ISO timestamp") from error
        if resolved_cutoff.tzinfo is None or resolved_cutoff.utcoffset() is None:
            raise ValueError("text vector build expiry cutoff must include a timezone")
        timestamp = _now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT rowid AS build_row_id, * FROM text_vector_builds
                WHERE recovery_state = 'active' AND lease_expires_at <= ?
                ORDER BY reserved_at, generation_id
                LIMIT ?
                """,
                (resolved_cutoff.astimezone(UTC).isoformat(), resolved_limit),
            ).fetchall()
            terminalized, quarantined_count = self._recover_text_vector_build_rows(
                connection,
                rows,
                error_code="text_vector_build_interrupted",
                timestamp=timestamp,
            )
            has_more = connection.execute(
                """
                SELECT 1 FROM text_vector_builds
                WHERE recovery_state = 'active' AND lease_expires_at <= ?
                LIMIT 1
                """,
                (resolved_cutoff.astimezone(UTC).isoformat(),),
            ).fetchone() is not None
            return TextVectorBuildRecoveryReport(
                examined_count=len(rows),
                terminalized_generation_ids=terminalized,
                quarantined_count=quarantined_count,
                has_more=has_more,
            )

    def list_quarantined_text_vector_builds(
        self,
        *,
        limit: int = 100,
    ) -> tuple[TextVectorBuildQuarantine, ...]:
        resolved_limit = _validate_repository_batch_limit(
            limit,
            field_name="text vector build quarantine listing limit",
        )
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    rowid AS row_id,
                    recovery_error_code AS error_code,
                    quarantined_at
                FROM text_vector_builds
                WHERE recovery_state = 'quarantined'
                ORDER BY quarantined_at, rowid
                LIMIT ?
                """,
                (resolved_limit,),
            ).fetchall()
        try:
            return tuple(TextVectorBuildQuarantine(**dict(row)) for row in rows)
        except (TypeError, ValueError) as error:
            raise ValueError("persisted text vector build quarantine is corrupt") from error

    def recover_abandoned_text_vector_builds(
        self,
        *,
        limit: int = 100,
    ) -> TextVectorBuildRecoveryReport:
        """Recover prior-runtime builds after the caller acquires the exclusive data lock."""

        resolved_limit = _validate_repository_batch_limit(
            limit,
            field_name="text vector build abandonment limit",
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            timestamp = _now()
            rows = connection.execute(
                """
                SELECT rowid AS build_row_id, * FROM text_vector_builds
                WHERE recovery_state = 'active'
                ORDER BY reserved_at, generation_id
                LIMIT ?
                """,
                (resolved_limit,),
            ).fetchall()
            terminalized, quarantined_count = self._recover_text_vector_build_rows(
                connection,
                rows,
                error_code="text_vector_runtime_abandoned",
                timestamp=timestamp,
            )
            has_more = connection.execute(
                """
                SELECT 1 FROM text_vector_builds
                WHERE recovery_state = 'active'
                LIMIT 1
                """
            ).fetchone() is not None
            return TextVectorBuildRecoveryReport(
                examined_count=len(rows),
                terminalized_generation_ids=terminalized,
                quarantined_count=quarantined_count,
                has_more=has_more,
            )

    def abandon_text_vector_builds(self, *, limit: int = 100) -> tuple[str, ...]:
        """Compatibility wrapper for exclusive-lock startup recovery."""

        report = self.recover_abandoned_text_vector_builds(limit=limit)
        if report.quarantined_count:
            raise RuntimeError(
                "abandoned text vector build metadata was quarantined; "
                "inspect repository degraded health"
            )
        return report.terminalized_generation_ids

    @classmethod
    def _artifact_gc_job_from_row(
        cls,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> ArtifactGCJob:
        try:
            values = dict(row)
            values.pop("sequence", None)
            job = ArtifactGCJob(**values)
        except (TypeError, ValueError) as error:
            raise ValueError("persisted artifact GC job is corrupt") from error
        specification = cls._get_text_vector_index_specification(
            connection,
            job.index_specification_hash,
        )
        if specification.collection_name != job.collection_name:
            raise ValueError("persisted artifact GC collection is corrupt")
        if connection.execute(
            "SELECT 1 FROM text_vector_generations WHERE generation_id = ?",
            (job.generation_id,),
        ).fetchone() is not None:
            raise ValueError("artifact GC job targets a committed generation")
        tombstone = connection.execute(
            """
            SELECT lifecycle_state, index_specification_hash, collection_name
            FROM text_vector_generation_tombstones
            WHERE generation_id = ?
            """,
            (job.generation_id,),
        ).fetchone()
        expected_lifecycle = {
            "pending": "gc_pending",
            "running": "gc_pending",
            "complete": "gc_complete",
            "failed": "gc_failed",
        }[job.state]
        if (
            tombstone is None
            or tombstone["lifecycle_state"] != expected_lifecycle
            or tombstone["index_specification_hash"]
            != job.index_specification_hash
            or tombstone["collection_name"] != job.collection_name
        ):
            raise ValueError("persisted artifact GC tombstone is corrupt")
        return job

    @staticmethod
    def _artifact_gc_attempt_from_row(row: sqlite3.Row) -> ArtifactGCAttempt:
        try:
            return ArtifactGCAttempt(**dict(row))
        except (TypeError, ValueError) as error:
            raise ValueError("persisted artifact GC attempt audit is corrupt") from error

    def list_pending_artifact_gc_jobs(
        self,
        *,
        limit: int = 100,
    ) -> tuple[ArtifactGCJob, ...]:
        resolved_limit = _validate_repository_batch_limit(
            limit,
            field_name="artifact GC listing limit",
        )
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                {_ARTIFACT_GC_SELECT}
                WHERE state = 'pending'
                  AND NOT EXISTS (
                    SELECT 1 FROM artifact_gc_quarantines AS quarantines
                    WHERE quarantines.sequence = artifact_gc_jobs.sequence
                  )
                ORDER BY available_at, sequence
                LIMIT ?
                """,
                (resolved_limit,),
            ).fetchall()
            jobs = tuple(self._artifact_gc_job_from_row(connection, row) for row in rows)
        return jobs

    def claim_next_artifact_gc_job(
        self,
        *,
        worker_id: str,
        lease_seconds: int = 30,
        scan_limit: int = 16,
    ) -> ArtifactGCJob | None:
        validate_artifact_identifier(worker_id, field_name="artifact GC worker id")
        resolved_lease_seconds = _validate_gc_lease_seconds(lease_seconds)
        if (
            isinstance(scan_limit, bool)
            or not isinstance(scan_limit, int)
            or not 1 <= scan_limit <= ARTIFACT_GC_SCAN_LIMIT_MAX
        ):
            raise ValueError(
                "artifact GC scan limit must be between "
                f"1 and {ARTIFACT_GC_SCAN_LIMIT_MAX}"
            )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            timestamp = _now()
            lease_expires_at = (
                datetime.fromisoformat(timestamp)
                + timedelta(seconds=resolved_lease_seconds)
            ).isoformat()
            rows = connection.execute(
                f"""
                {_ARTIFACT_GC_SELECT}
                WHERE (
                    (state = 'pending' AND available_at <= ?)
                    OR (state = 'running' AND lease_expires_at <= ?)
                )
                  AND NOT EXISTS (
                    SELECT 1 FROM artifact_gc_quarantines AS quarantines
                    WHERE quarantines.sequence = artifact_gc_jobs.sequence
                  )
                ORDER BY
                    CASE
                        WHEN state = 'pending' THEN available_at
                        ELSE lease_expires_at
                    END,
                    sequence
                LIMIT ?
                """,
                (timestamp, timestamp, scan_limit),
            ).fetchall()
            for row in rows:
                sequence = row["sequence"]
                connection.execute("SAVEPOINT claim_artifact_gc")
                try:
                    current = self._artifact_gc_job_from_row(connection, row)
                    if current.state == "running":
                        assert current.lease_token is not None
                        audit_row = connection.execute(
                            """
                            SELECT
                                job_id, attempt, worker_id, lease_token,
                                claimed_at, lease_expires_at, finished_at,
                                outcome, error_code
                            FROM artifact_gc_attempts
                            WHERE job_id = ? AND attempt = ?
                            """,
                            (current.job_id, current.attempt),
                        ).fetchone()
                        if audit_row is None:
                            raise ValueError("artifact GC attempt audit is missing")
                        audit = self._artifact_gc_attempt_from_row(audit_row)
                        if (
                            audit.worker_id != current.worker_id
                            or audit.lease_token != current.lease_token
                            or audit.lease_expires_at != current.lease_expires_at
                            or audit.outcome is not None
                        ):
                            raise ValueError("artifact GC attempt audit is inconsistent")
                        cursor = connection.execute(
                            """
                            UPDATE artifact_gc_attempts
                            SET finished_at = ?, outcome = 'lease_expired',
                                error_code = 'artifact_gc_lease_expired'
                            WHERE job_id = ? AND attempt = ? AND lease_token = ?
                              AND outcome IS NULL
                            """,
                            (
                                timestamp,
                                current.job_id,
                                current.attempt,
                                current.lease_token,
                            ),
                        )
                        if cursor.rowcount != 1:
                            raise ValueError("artifact GC attempt audit is corrupt")
                        next_backoff_level = min(
                            current.backoff_level + 1,
                            ARTIFACT_GC_BACKOFF_LEVEL_MAX,
                        )
                        delay_seconds = min(
                            ARTIFACT_GC_RETRY_BASE_SECONDS
                            * (2**current.backoff_level),
                            ARTIFACT_GC_RETRY_MAX_SECONDS,
                        )
                        available_at = (
                            datetime.fromisoformat(timestamp)
                            + timedelta(seconds=delay_seconds)
                        ).isoformat()
                        cursor = connection.execute(
                            """
                            UPDATE artifact_gc_jobs
                            SET state = 'pending', backoff_level = ?,
                                available_at = ?, worker_id = NULL,
                                lease_token = NULL, lease_expires_at = NULL,
                                error_code = 'artifact_gc_lease_expired', updated_at = ?
                            WHERE job_id = ? AND state = 'running'
                              AND attempt = ? AND lease_token = ?
                              AND lease_expires_at <= ?
                            """,
                            (
                                next_backoff_level,
                                available_at,
                                timestamp,
                                current.job_id,
                                current.attempt,
                                current.lease_token,
                                timestamp,
                            ),
                        )
                        if cursor.rowcount != 1:
                            raise ValueError("artifact GC expired lease changed")
                        connection.execute("RELEASE claim_artifact_gc")
                        continue

                    lease_token = uuid4().hex
                    next_attempt = current.attempt + 1
                    cursor = connection.execute(
                        """
                        UPDATE artifact_gc_jobs
                        SET state = 'running', attempt = ?,
                            worker_id = ?, lease_token = ?, lease_expires_at = ?,
                            error_code = NULL, updated_at = ?
                        WHERE job_id = ? AND state = 'pending' AND attempt = ?
                          AND available_at <= ?
                        """,
                        (
                            next_attempt,
                            worker_id,
                            lease_token,
                            lease_expires_at,
                            timestamp,
                            current.job_id,
                            current.attempt,
                            timestamp,
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise ValueError("artifact GC claim changed concurrently")
                    connection.execute(
                        """
                        INSERT INTO artifact_gc_attempts (
                            job_id, attempt, worker_id, lease_token,
                            claimed_at, lease_expires_at,
                            finished_at, outcome, error_code
                        ) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL)
                        """,
                        (
                            current.job_id,
                            next_attempt,
                            worker_id,
                            lease_token,
                            timestamp,
                            lease_expires_at,
                        ),
                    )
                    _prune_artifact_gc_attempt_audit(
                        connection,
                        job_id=current.job_id,
                    )
                    claimed_row = connection.execute(
                        f"{_ARTIFACT_GC_SELECT} WHERE job_id = ?",
                        (current.job_id,),
                    ).fetchone()
                    assert claimed_row is not None
                    claimed = self._artifact_gc_job_from_row(connection, claimed_row)
                except (KeyError, ValueError, sqlite3.IntegrityError):
                    connection.execute("ROLLBACK TO claim_artifact_gc")
                    connection.execute("RELEASE claim_artifact_gc")
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO artifact_gc_quarantines (
                            sequence, error_code, quarantined_at
                        ) VALUES (?, 'artifact_gc_metadata_corrupt', ?)
                        """,
                        (sequence, timestamp),
                    )
                    continue
                connection.execute("RELEASE claim_artifact_gc")
                return claimed
            return None

    def list_artifact_gc_attempts(
        self,
        job_id: str,
        *,
        limit: int = 100,
    ) -> tuple[ArtifactGCAttempt, ...]:
        validate_artifact_identifier(job_id, field_name="artifact GC audit job id")
        resolved_limit = _validate_repository_batch_limit(
            limit,
            field_name="artifact GC audit listing limit",
        )
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    job_id, attempt, worker_id, lease_token,
                    claimed_at, lease_expires_at, finished_at, outcome, error_code
                FROM artifact_gc_attempts
                WHERE job_id = ?
                ORDER BY attempt DESC
                LIMIT ?
                """,
                (job_id, resolved_limit),
            ).fetchall()
        return tuple(
            self._artifact_gc_attempt_from_row(row) for row in reversed(rows)
        )

    def list_artifact_gc_quarantines(
        self,
        *,
        limit: int = 100,
    ) -> tuple[ArtifactGCQuarantine, ...]:
        resolved_limit = _validate_repository_batch_limit(
            limit,
            field_name="artifact GC quarantine listing limit",
        )
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT sequence, error_code, quarantined_at
                FROM artifact_gc_quarantines
                ORDER BY quarantined_at, sequence
                LIMIT ?
                """,
                (resolved_limit,),
            ).fetchall()
        try:
            return tuple(ArtifactGCQuarantine(**dict(row)) for row in rows)
        except (TypeError, ValueError) as error:
            raise ValueError("persisted artifact GC quarantine is corrupt") from error

    def heartbeat_artifact_gc_job(
        self,
        job_id: str,
        *,
        worker_id: str,
        lease_token: str,
        attempt: int,
        lease_seconds: int = 30,
    ) -> ArtifactGCJob:
        validate_artifact_identifier(job_id, field_name="artifact GC job id")
        validate_artifact_identifier(worker_id, field_name="artifact GC worker id")
        validate_artifact_identifier(lease_token, field_name="artifact GC lease token")
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
            raise ValueError("artifact GC attempt fence must be positive")
        resolved_lease_seconds = _validate_gc_lease_seconds(lease_seconds)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            timestamp = _now()
            lease_expires_at = (
                datetime.fromisoformat(timestamp)
                + timedelta(seconds=resolved_lease_seconds)
            ).isoformat()
            row = connection.execute(
                f"{_ARTIFACT_GC_SELECT} WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                raise KeyError(job_id)
            current = self._artifact_gc_job_from_row(connection, row)
            if (
                current.state != "running"
                or current.worker_id != worker_id
                or current.lease_token != lease_token
                or current.attempt != attempt
                or current.lease_expires_at is None
                or current.lease_expires_at <= timestamp
            ):
                raise ValueError("artifact GC claim changed or expired")
            cursor = connection.execute(
                """
                UPDATE artifact_gc_jobs
                SET lease_expires_at = ?, updated_at = ?
                WHERE job_id = ? AND state = 'running'
                  AND worker_id = ? AND lease_token = ? AND attempt = ?
                  AND lease_expires_at > ?
                """,
                (
                    lease_expires_at,
                    timestamp,
                    job_id,
                    worker_id,
                    lease_token,
                    attempt,
                    timestamp,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("artifact GC claim changed or expired")
            cursor = connection.execute(
                """
                UPDATE artifact_gc_attempts
                SET lease_expires_at = ?
                WHERE job_id = ? AND attempt = ? AND worker_id = ?
                  AND lease_token = ? AND outcome IS NULL
                """,
                (
                    lease_expires_at,
                    job_id,
                    attempt,
                    worker_id,
                    lease_token,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("artifact GC attempt audit changed or is corrupt")
            heartbeat_row = connection.execute(
                f"{_ARTIFACT_GC_SELECT} WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            assert heartbeat_row is not None
            return self._artifact_gc_job_from_row(connection, heartbeat_row)

    def finish_artifact_gc_job(
        self,
        job_id: str,
        *,
        worker_id: str,
        lease_token: str,
        attempt: int,
        outcome: ArtifactGCOutcome,
        error_code: str | None = None,
    ) -> ArtifactGCJob:
        validate_artifact_identifier(job_id, field_name="artifact GC job id")
        validate_artifact_identifier(worker_id, field_name="artifact GC worker id")
        validate_artifact_identifier(lease_token, field_name="artifact GC lease token")
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
            raise ValueError("artifact GC attempt fence must be positive")
        try:
            resolved_outcome = ArtifactGCOutcome(outcome)
        except (TypeError, ValueError) as error:
            raise ValueError("unsupported artifact GC outcome") from error
        validate_error_code(error_code)
        if resolved_outcome is ArtifactGCOutcome.SUCCESS:
            if error_code is not None:
                raise ValueError("successful artifact GC cannot contain an error")
        elif error_code is None:
            raise ValueError("failed artifact GC requires a sanitized error code")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            timestamp = _now()
            row = connection.execute(
                f"{_ARTIFACT_GC_SELECT} WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                raise KeyError(job_id)
            current = self._artifact_gc_job_from_row(connection, row)
            if (
                current.state != "running"
                or current.worker_id != worker_id
                or current.lease_token != lease_token
                or current.attempt != attempt
                or current.lease_expires_at is None
                or current.lease_expires_at <= timestamp
            ):
                raise ValueError("artifact GC claim changed or expired")

            if resolved_outcome is ArtifactGCOutcome.SUCCESS:
                next_state = "complete"
                next_lifecycle = "gc_complete"
                next_backoff_level = current.backoff_level
                available_at = timestamp
                persisted_error = None
            elif resolved_outcome is ArtifactGCOutcome.PERMANENT_FAILURE:
                next_state = "failed"
                next_lifecycle = "gc_failed"
                next_backoff_level = current.backoff_level
                available_at = timestamp
                persisted_error = error_code
            else:
                next_state = "pending"
                next_lifecycle = "gc_pending"
                next_backoff_level = min(
                    current.backoff_level + 1,
                    ARTIFACT_GC_BACKOFF_LEVEL_MAX,
                )
                delay_seconds = min(
                    ARTIFACT_GC_RETRY_BASE_SECONDS * (2**current.backoff_level),
                    ARTIFACT_GC_RETRY_MAX_SECONDS,
                )
                available_at = (
                    datetime.fromisoformat(timestamp) + timedelta(seconds=delay_seconds)
                ).isoformat()
                persisted_error = error_code

            cursor = connection.execute(
                """
                UPDATE artifact_gc_jobs
                SET state = ?, backoff_level = ?, available_at = ?, worker_id = NULL,
                    lease_token = NULL, lease_expires_at = NULL,
                    error_code = ?, updated_at = ?
                WHERE job_id = ? AND state = 'running'
                  AND worker_id = ? AND lease_token = ? AND attempt = ?
                  AND lease_expires_at > ?
                """,
                (
                    next_state,
                    next_backoff_level,
                    available_at,
                    persisted_error,
                    timestamp,
                    job_id,
                    worker_id,
                    lease_token,
                    attempt,
                    timestamp,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("artifact GC claim changed or expired")
            cursor = connection.execute(
                """
                UPDATE text_vector_generation_tombstones
                SET lifecycle_state = ?, updated_at = ?
                WHERE generation_id = ? AND lifecycle_state = 'gc_pending'
                """,
                (next_lifecycle, timestamp, current.generation_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("artifact GC tombstone changed during finish")
            cursor = connection.execute(
                """
                UPDATE artifact_gc_attempts
                SET finished_at = ?, outcome = ?, error_code = ?
                WHERE job_id = ? AND attempt = ? AND worker_id = ?
                  AND lease_token = ? AND outcome IS NULL
                """,
                (
                    timestamp,
                    resolved_outcome.value,
                    persisted_error,
                    job_id,
                    attempt,
                    worker_id,
                    lease_token,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("artifact GC attempt audit changed during finish")
            finished_row = connection.execute(
                f"{_ARTIFACT_GC_SELECT} WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            assert finished_row is not None
            return self._artifact_gc_job_from_row(connection, finished_row)

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
