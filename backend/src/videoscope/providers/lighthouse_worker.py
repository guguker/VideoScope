from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import hmac
from io import BytesIO
from importlib import metadata as importlib_metadata
import importlib.util
from ipaddress import ip_address
import json
import logging
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shutil
import stat
import sys
from threading import BoundedSemaphore, Lock
from time import monotonic
from typing import Any, Callable, Literal, Protocol
from urllib.parse import urlsplit
from uuid import uuid4
import zipfile

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
import numpy as np
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from videoscope.media.ffmpeg import FFmpeg
from videoscope.model_manifest import (
    LIGHTHOUSE_CHECKPOINT_SHA256,
    LIGHTHOUSE_CLIP_CHECKPOINT_SHA256,
    LIGHTHOUSE_CLIP_REVISION,
    LIGHTHOUSE_SOURCE_REVISION,
)
from videoscope.providers.base import ProviderState, ProviderStatus
from videoscope.search.fusion import EvidenceHit
from videoscope.storage import atomic_write_json


logger = logging.getLogger(__name__)

LIGHTHOUSE_WORKER_SCHEMA_VERSION = "lighthouse-worker-v1"
LIGHTHOUSE_CACHE_SCHEMA_VERSION = 2
LIGHTHOUSE_MODEL_IDENTITY = f"lighthouse/qd-detr-qvhighlight@sha256:{LIGHTHOUSE_CHECKPOINT_SHA256}"
LIGHTHOUSE_WORKER_LOCK_SHA256 = (
    "c0fb00e3f7b106bf063b2b31e1dacda66a36ad51153258d2a5cdf0407fa25235"
)
LIGHTHOUSE_WORKER_RUNTIME_IDENTITY = (
    "videoscope-lighthouse-worker-v1|python==3.11.14|platform==aarch64-apple-darwin|"
    f"lock-sha256:{LIGHTHOUSE_WORKER_LOCK_SHA256}"
)
MAX_REQUEST_BYTES = 64 * 1024
MAX_RESPONSE_BYTES = 256 * 1024
MAX_QUERY_CHARS = 500
MAX_VIDEO_IDS = 100
MAX_HITS = 100
DEFAULT_MAX_INPUT_BYTES = 32 * 1024 * 1024 * 1024
LIGHTHOUSE_FEATURE_DIMENSIONS = 514
MAX_CACHE_WINDOWS = 576
MAX_FEATURE_ARCHIVE_BYTES = 512 * 1024
MAX_GENERATION_FEATURE_BYTES = MAX_CACHE_WINDOWS * MAX_FEATURE_ARCHIVE_BYTES
MAX_MANIFEST_BYTES = 256 * 1024
MAX_ACTIVE_POINTER_BYTES = 4 * 1024
LIGHTHOUSE_WORKER_PYTHON_VERSION = (3, 11, 14)
LIGHTHOUSE_WORKER_CORE_DISTRIBUTIONS = {
    "fastapi": "0.141.1",
    "numpy": "1.23.5",
    "opencv-python": "4.11.0.86",
    "pydantic": "2.13.4",
    "pydantic-settings": "2.15.0",
    "torch": "2.11.0",
    "torchvision": "0.26.0",
    "uvicorn": "0.52.3",
}
_TOKEN_RE = re.compile(r"^[A-Za-z0-9._~-]{32,256}$")
_REQUEST_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_GENERATION_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_VIDEO_SUFFIXES = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"})


@dataclass(frozen=True, slots=True)
class LighthouseSpecification:
    """Every input that changes Lighthouse features, predictions, or cache encoding."""

    cache_schema_version: int = LIGHTHOUSE_CACHE_SCHEMA_VERSION
    checkpoint_sha256: str = LIGHTHOUSE_CHECKPOINT_SHA256
    clip_checkpoint_sha256: str = LIGHTHOUSE_CLIP_CHECKPOINT_SHA256
    lighthouse_revision: str = LIGHTHOUSE_SOURCE_REVISION
    clip_revision: str = LIGHTHOUSE_CLIP_REVISION
    runtime_identity: str = LIGHTHOUSE_WORKER_RUNTIME_IDENTITY
    max_window_seconds: float = 150.0

    def to_dict(self) -> dict[str, object]:
        return {
            "cache_encoding": "numpy-npz-float32-v1",
            "cache_schema_version": self.cache_schema_version,
            "checkpoint_sha256": self.checkpoint_sha256,
            "clip_checkpoint_sha256": self.clip_checkpoint_sha256,
            "clip_model": "ViT-B/32",
            "clip_revision": self.clip_revision,
            "feature_name": "clip",
            "lighthouse_revision": self.lighthouse_revision,
            "max_window_seconds": self.max_window_seconds,
            "model_adapter": "videoscope.lighthouse-qd-detr-v2",
            "preprocessing": {
                "color": "rgb",
                "frame_sampling": "temporal-centers-by-checkpoint-clip-length",
                "image_size": 224,
                "max_frames": 75,
                "normalize": "openai-clip-vit-b-32",
                "resize": "short-edge-and-center-crop",
            },
            "runtime_identity": self.runtime_identity,
            "worker_contract": LIGHTHOUSE_WORKER_SCHEMA_VERSION,
        }

    @property
    def identity(self) -> str:
        canonical = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256(canonical.encode("utf-8")).hexdigest()


DEFAULT_LIGHTHOUSE_SPECIFICATION = LighthouseSpecification()


class _ContractModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


def _validate_video_id(value: str) -> str:
    if _VIDEO_ID_RE.fullmatch(value) is None:
        raise ValueError("invalid video ID")
    return value


def _validate_relative_path(value: str) -> str:
    if (
        "\\" in value
        or "\x00" in value
        or re.fullmatch(r"[A-Za-z0-9._/-]+", value) is None
    ):
        raise ValueError("relative_path must be a safe POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("relative_path must stay inside the worker input root")
    return path.as_posix()


class LighthousePrepareRequest(_ContractModel):
    schema_version: Literal[LIGHTHOUSE_WORKER_SCHEMA_VERSION]
    request_id: str = Field(pattern=_REQUEST_ID_RE.pattern)
    specification_hash: str = Field(pattern=_SHA256_RE.pattern)
    model_identity: str = Field(min_length=1, max_length=200)
    runtime_identity: str = Field(min_length=1, max_length=600)
    video_id: str = Field(min_length=1, max_length=64)
    relative_path: str = Field(min_length=1, max_length=240)
    duration_seconds: float = Field(gt=0, le=24 * 60 * 60)
    source_sha256: str = Field(pattern=_SHA256_RE.pattern)
    source_size_bytes: int = Field(gt=0)

    @field_validator("video_id")
    @classmethod
    def validate_video_id(cls, value: str) -> str:
        return _validate_video_id(value)

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        return _validate_relative_path(value)


class LighthouseSearchRequest(_ContractModel):
    schema_version: Literal[LIGHTHOUSE_WORKER_SCHEMA_VERSION]
    request_id: str = Field(pattern=_REQUEST_ID_RE.pattern)
    specification_hash: str = Field(pattern=_SHA256_RE.pattern)
    model_identity: str = Field(min_length=1, max_length=200)
    runtime_identity: str = Field(min_length=1, max_length=600)
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    video_ids: list[str] = Field(min_length=1, max_length=MAX_VIDEO_IDS)
    limit: int = Field(ge=1, le=MAX_HITS)

    @field_validator("video_ids")
    @classmethod
    def validate_video_ids(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("video IDs must be unique")
        return [_validate_video_id(video_id) for video_id in value]


class LighthouseHitPayload(_ContractModel):
    video_id: str = Field(min_length=1, max_length=64)
    generation_id: str = Field(pattern=_GENERATION_ID_RE.pattern)
    window_index: int = Field(ge=0, le=10_000)
    rank: int = Field(ge=0, le=1_000)
    start: float = Field(ge=0, le=24 * 60 * 60)
    end: float = Field(gt=0, le=24 * 60 * 60)
    score: float = Field(ge=0, le=1)

    @field_validator("video_id")
    @classmethod
    def validate_video_id(cls, value: str) -> str:
        return _validate_video_id(value)

    @model_validator(mode="after")
    def validate_interval(self) -> "LighthouseHitPayload":
        if self.end <= self.start:
            raise ValueError("hit end must be after start")
        return self


class LighthousePrepareResponse(_ContractModel):
    schema_version: Literal[LIGHTHOUSE_WORKER_SCHEMA_VERSION]
    request_id: str = Field(pattern=_REQUEST_ID_RE.pattern)
    specification_hash: str = Field(pattern=_SHA256_RE.pattern)
    model_identity: str = Field(min_length=1, max_length=200)
    runtime_identity: str = Field(min_length=1, max_length=600)
    video_id: str = Field(min_length=1, max_length=64)
    generation_id: str = Field(pattern=_GENERATION_ID_RE.pattern)


class LighthouseSearchResponse(_ContractModel):
    schema_version: Literal[LIGHTHOUSE_WORKER_SCHEMA_VERSION]
    request_id: str = Field(pattern=_REQUEST_ID_RE.pattern)
    specification_hash: str = Field(pattern=_SHA256_RE.pattern)
    model_identity: str = Field(min_length=1, max_length=200)
    runtime_identity: str = Field(min_length=1, max_length=600)
    hits: list[LighthouseHitPayload] = Field(max_length=MAX_HITS)


class LighthouseHealthResponse(_ContractModel):
    schema_version: Literal[LIGHTHOUSE_WORKER_SCHEMA_VERSION]
    status: Literal["ok", "unavailable"]
    model_identity: str = Field(min_length=1, max_length=200)
    runtime_identity: str = Field(min_length=1, max_length=600)
    specification: dict[str, object]
    specification_hash: str = Field(pattern=_SHA256_RE.pattern)
    loaded: bool
    operations: list[Literal["prepare", "search"]] = Field(min_length=2, max_length=2)
    max_input_bytes: int = Field(gt=0)
    max_query_chars: Literal[MAX_QUERY_CHARS]
    max_video_ids: Literal[MAX_VIDEO_IDS]
    max_hits: Literal[MAX_HITS]
    max_concurrency: Literal[1]

    @field_validator("operations")
    @classmethod
    def validate_operations(
        cls,
        value: list[Literal["prepare", "search"]],
    ) -> list[Literal["prepare", "search"]]:
        if value != ["prepare", "search"]:
            raise ValueError("operations do not match the contract")
        return value


@dataclass(frozen=True, slots=True)
class LighthouseEncodedWindow:
    offset: float
    end: float
    video_features: np.ndarray
    video_mask: np.ndarray


@dataclass(frozen=True, slots=True)
class _SourceFingerprint:
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _worker_environment_is_exact(
    *,
    python_version: tuple[int, int, int] | None = None,
    lock_path: Path | None = None,
    expected_lock_sha256: str = LIGHTHOUSE_WORKER_LOCK_SHA256,
    distribution_version: Callable[[str], str] | None = None,
) -> bool:
    current_python = python_version or tuple(sys.version_info[:3])
    current_lock = lock_path or (
        Path(__file__).resolve().parents[4]
        / "workers"
        / "lighthouse"
        / "requirements.lock"
    )
    resolve_version = distribution_version or importlib_metadata.version
    if (
        current_python != LIGHTHOUSE_WORKER_PYTHON_VERSION
        or _SHA256_RE.fullmatch(expected_lock_sha256) is None
        or not current_lock.is_file()
        or current_lock.is_symlink()
    ):
        return False
    try:
        if not hmac.compare_digest(_file_sha256(current_lock), expected_lock_sha256):
            return False
        return all(
            resolve_version(distribution) == expected
            for distribution, expected in LIGHTHOUSE_WORKER_CORE_DISTRIBUTIONS.items()
        )
    except (
        importlib_metadata.PackageNotFoundError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ):
        return False


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(Path(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fingerprint(metadata: os.stat_result) -> _SourceFingerprint:
    return _SourceFingerprint(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        size=metadata.st_size,
        modified_ns=metadata.st_mtime_ns,
        changed_ns=metadata.st_ctime_ns,
    )


def _read_bounded_regular_file(path: Path, max_bytes: int) -> bytes | None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return None
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size < 0
            or before.st_size > max_bytes
        ):
            return None
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read(max_bytes + 1)
        after = os.fstat(descriptor)
    except OSError:
        return None
    finally:
        os.close(descriptor)
    if (
        len(payload) > max_bytes
        or len(payload) != after.st_size
        or _fingerprint(before) != _fingerprint(after)
    ):
        return None
    return payload


def _source_snapshot(path: Path) -> tuple[str, int, _SourceFingerprint]:
    before = path.stat()
    digest = _file_sha256(path)
    after = path.stat()
    if _fingerprint(before) != _fingerprint(after):
        raise RuntimeError("source changed while it was being hashed")
    return digest, after.st_size, _fingerprint(after)


class LighthouseGenerationStore:
    """Immutable, safe-to-read feature generations with one atomic active pointer."""

    def __init__(
        self,
        root: Path,
        specification: LighthouseSpecification = DEFAULT_LIGHTHOUSE_SPECIFICATION,
    ) -> None:
        self.root = Path(os.path.abspath(root))
        self.specification = specification

    @staticmethod
    def _reject_unsafe_ancestors(path: Path) -> None:
        absolute = Path(os.path.abspath(path))
        current = Path(absolute.anchor)
        for part in absolute.parts[1:]:
            current = current / part
            if current.is_symlink():
                raise ValueError("unsafe symlink in Lighthouse cache path")
            if current.exists() and not current.is_dir():
                raise ValueError("unsafe non-directory in Lighthouse cache path")

    def _safe_root(self, *, create: bool = False) -> Path:
        self._reject_unsafe_ancestors(self.root)
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
            self._reject_unsafe_ancestors(self.root)
        if self.root.exists() and (self.root.is_symlink() or not self.root.is_dir()):
            raise ValueError("unsafe Lighthouse cache root")
        return self.root

    def _video_dir(self, video_id: str, *, create: bool = False) -> Path:
        root = self._safe_root(create=create)
        video_dir = root / _validate_video_id(video_id)
        if video_dir.is_symlink() or (video_dir.exists() and not video_dir.is_dir()):
            raise ValueError("unsafe Lighthouse video cache path")
        if create:
            video_dir.mkdir(exist_ok=True)
            if video_dir.is_symlink() or not video_dir.is_dir():
                raise ValueError("unsafe Lighthouse video cache path")
            try:
                video_dir.resolve(strict=True).relative_to(root.resolve(strict=True))
            except (OSError, ValueError) as error:
                raise ValueError("unsafe Lighthouse video cache path") from error
        return video_dir

    def _validate_window(
        self,
        window: LighthouseEncodedWindow,
        *,
        expected_offset: float,
        duration_seconds: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        offset = float(window.offset)
        end = float(window.end)
        features = np.asarray(window.video_features, dtype=np.float32)
        mask = np.asarray(window.video_mask, dtype=np.float32)
        expected_end = min(
            duration_seconds,
            expected_offset + self.specification.max_window_seconds,
        )
        if (
            not math.isfinite(offset)
            or not math.isfinite(end)
            or not math.isclose(offset, expected_offset, rel_tol=1e-9, abs_tol=1e-6)
            or not math.isclose(end, expected_end, rel_tol=1e-9, abs_tol=1e-6)
            or end <= offset
            or end > duration_seconds + 1e-6
            or end - offset > self.specification.max_window_seconds + 1e-6
        ):
            raise ValueError("Lighthouse window boundaries are invalid")
        if (
            features.ndim != 3
            or features.shape[0] != 1
            or not 1 <= features.shape[1] <= 75
            or features.shape[2] != LIGHTHOUSE_FEATURE_DIMENSIONS
            or mask.ndim != 2
            or mask.shape != features.shape[:2]
            or not bool(np.all(np.isfinite(features)))
            or not bool(np.all(np.isfinite(mask)))
            or not bool(np.all((mask == 0) | (mask == 1)))
            or not bool(np.any(mask == 1))
        ):
            raise ValueError("Lighthouse features must be finite and structurally valid")
        return features, mask

    def build_generation(
        self,
        *,
        video_id: str,
        source_sha256: str,
        source_size_bytes: int,
        duration_seconds: float,
        windows: list[LighthouseEncodedWindow],
    ) -> str:
        video_dir = self._video_dir(video_id, create=True)
        if _SHA256_RE.fullmatch(source_sha256) is None or source_size_bytes <= 0:
            raise ValueError("Lighthouse source identity is invalid")
        duration_seconds = float(duration_seconds)
        if not math.isfinite(duration_seconds) or duration_seconds <= 0:
            raise ValueError("Lighthouse duration must be positive")
        expected_window_count = math.ceil(
            duration_seconds / self.specification.max_window_seconds
        )
        if (
            not windows
            or expected_window_count > MAX_CACHE_WINDOWS
            or len(windows) != expected_window_count
        ):
            raise ValueError("Lighthouse generation must contain bounded windows")

        validated: list[tuple[LighthouseEncodedWindow, np.ndarray, np.ndarray]] = []
        expected_offset = 0.0
        for window in windows:
            features, mask = self._validate_window(
                window,
                expected_offset=expected_offset,
                duration_seconds=duration_seconds,
            )
            validated.append((window, features, mask))
            expected_offset = float(window.end)
        if not math.isclose(expected_offset, duration_seconds, rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError("Lighthouse windows must cover the complete source")

        generation_id = uuid4().hex
        generations = video_dir / "generations"
        if generations.is_symlink() or (
            generations.exists() and not generations.is_dir()
        ):
            raise ValueError("unsafe Lighthouse generations path")
        generations.mkdir(exist_ok=True)
        if generations.is_symlink() or not generations.is_dir():
            raise ValueError("unsafe Lighthouse generations path")
        try:
            generations.resolve(strict=True).relative_to(video_dir.resolve(strict=True))
        except (OSError, ValueError) as error:
            raise ValueError("unsafe Lighthouse generations path") from error
        staging = generations / f".{generation_id}.building"
        generation = generations / generation_id
        staging.mkdir(exist_ok=False)
        _fsync_directory(generations)
        try:
            records: list[dict[str, object]] = []
            for index, (window, features, mask) in enumerate(validated):
                name = f"window-{index:04d}.npz"
                destination = staging / name
                with destination.open("xb") as handle:
                    np.savez(handle, video_features=features, video_mask=mask)
                    handle.flush()
                    os.fsync(handle.fileno())
                records.append(
                    {
                        "end": float(window.end),
                        "feature_dimensions": int(features.shape[2]),
                        "feature_rows": int(features.shape[1]),
                        "file": name,
                        "offset": float(window.offset),
                        "sha256": _file_sha256(destination),
                    }
                )
            manifest = {
                "duration_seconds": duration_seconds,
                "generation_id": generation_id,
                "manifest_schema_version": LIGHTHOUSE_CACHE_SCHEMA_VERSION,
                "source_sha256": source_sha256,
                "source_size_bytes": source_size_bytes,
                "specification": self.specification.to_dict(),
                "specification_hash": self.specification.identity,
                "windows": records,
            }
            atomic_write_json(staging / "manifest.json", manifest, sort_keys=True)
            _fsync_directory(staging)
            if self._load_generation_path(staging, generation_id) is None:
                raise ValueError("persisted Lighthouse generation failed validation")
            staging.replace(generation)
            _fsync_directory(generation)
            _fsync_directory(generations)
            if self._load_generation_path(generation, generation_id) is None:
                raise ValueError("completed Lighthouse generation failed validation")
            return generation_id
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def _load_generation_path(
        self,
        generation: Path,
        generation_id: str,
    ) -> tuple[dict[str, object], list[LighthouseEncodedWindow]] | None:
        if (
            _GENERATION_ID_RE.fullmatch(generation_id) is None
            or not generation.is_dir()
            or generation.is_symlink()
        ):
            return None
        manifest_path = generation / "manifest.json"
        manifest_bytes = _read_bounded_regular_file(manifest_path, MAX_MANIFEST_BYTES)
        if manifest_bytes is None:
            return None
        try:
            payload: object = json.loads(manifest_bytes.decode("utf-8"))
        except (OSError, UnicodeError, ValueError, TypeError):
            return None
        expected_keys = {
            "duration_seconds",
            "generation_id",
            "manifest_schema_version",
            "source_sha256",
            "source_size_bytes",
            "specification",
            "specification_hash",
            "windows",
        }
        if not isinstance(payload, dict) or set(payload) != expected_keys:
            return None
        duration = payload.get("duration_seconds")
        if (
            payload.get("generation_id") != generation_id
            or payload.get("manifest_schema_version") != LIGHTHOUSE_CACHE_SCHEMA_VERSION
            or payload.get("specification") != self.specification.to_dict()
            or payload.get("specification_hash") != self.specification.identity
            or type(payload.get("source_sha256")) is not str
            or _SHA256_RE.fullmatch(str(payload["source_sha256"])) is None
            or type(payload.get("source_size_bytes")) is not int
            or int(payload["source_size_bytes"]) <= 0
            or isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or float(duration) <= 0
        ):
            return None
        raw_windows = payload.get("windows")
        expected_count = math.ceil(
            float(duration) / self.specification.max_window_seconds
        )
        if (
            not isinstance(raw_windows, list)
            or not raw_windows
            or len(raw_windows) != expected_count
            or len(raw_windows) > MAX_CACHE_WINDOWS
        ):
            return None
        windows: list[LighthouseEncodedWindow] = []
        expected_offset = 0.0
        aggregate_bytes = 0
        try:
            for index, raw in enumerate(raw_windows):
                if not isinstance(raw, dict) or set(raw) != {
                    "end",
                    "feature_dimensions",
                    "feature_rows",
                    "file",
                    "offset",
                    "sha256",
                }:
                    return None
                name = f"window-{index:04d}.npz"
                if (
                    raw.get("file") != name
                    or type(raw.get("sha256")) is not str
                    or _SHA256_RE.fullmatch(str(raw["sha256"])) is None
                    or isinstance(raw.get("offset"), bool)
                    or not isinstance(raw.get("offset"), (int, float))
                    or isinstance(raw.get("end"), bool)
                    or not isinstance(raw.get("end"), (int, float))
                ):
                    return None
                window_path = generation / name
                archive_bytes = _read_bounded_regular_file(
                    window_path,
                    MAX_FEATURE_ARCHIVE_BYTES,
                )
                if archive_bytes is None or not archive_bytes:
                    return None
                archive_size = len(archive_bytes)
                aggregate_bytes += archive_size
                if (
                    aggregate_bytes > MAX_GENERATION_FEATURE_BYTES
                ):
                    return None
                if not hmac.compare_digest(
                    sha256(archive_bytes).hexdigest(),
                    str(raw["sha256"]),
                ):
                    return None
                if not self._archive_layout_is_safe(
                    archive_bytes,
                    feature_rows=raw.get("feature_rows"),
                    feature_dimensions=raw.get("feature_dimensions"),
                ):
                    return None
                with np.load(BytesIO(archive_bytes), allow_pickle=False) as arrays:
                    if set(arrays.files) != {"video_features", "video_mask"}:
                        return None
                    features = np.asarray(arrays["video_features"], dtype=np.float32)
                    mask = np.asarray(arrays["video_mask"], dtype=np.float32)
                window = LighthouseEncodedWindow(
                    offset=float(raw["offset"]),
                    end=float(raw["end"]),
                    video_features=features,
                    video_mask=mask,
                )
                self._validate_window(
                    window,
                    expected_offset=expected_offset,
                    duration_seconds=float(duration),
                )
                if (
                    raw.get("feature_rows") != int(features.shape[1])
                    or raw.get("feature_dimensions") != int(features.shape[2])
                ):
                    return None
                windows.append(window)
                expected_offset = window.end
        except (OSError, TypeError, ValueError):
            return None
        if not math.isclose(expected_offset, float(duration), rel_tol=1e-9, abs_tol=1e-6):
            return None
        return payload, windows

    @staticmethod
    def _archive_layout_is_safe(
        payload: bytes,
        *,
        feature_rows: object,
        feature_dimensions: object,
    ) -> bool:
        if (
            type(feature_rows) is not int
            or not 1 <= feature_rows <= 75
            or feature_dimensions != LIGHTHOUSE_FEATURE_DIMENSIONS
        ):
            return False
        expected_shapes = {
            "video_features.npy": (1, feature_rows, LIGHTHOUSE_FEATURE_DIMENSIONS),
            "video_mask.npy": (1, feature_rows),
        }
        try:
            with zipfile.ZipFile(BytesIO(payload), "r") as archive:
                entries = archive.infolist()
                if [entry.filename for entry in entries] != list(expected_shapes):
                    return False
                for entry in entries:
                    if (
                        entry.compress_type != zipfile.ZIP_STORED
                        or entry.flag_bits & 0x1
                        or entry.compress_size != entry.file_size
                        or entry.file_size <= 0
                        or entry.file_size > MAX_FEATURE_ARCHIVE_BYTES
                    ):
                        return False
                    with archive.open(entry, "r") as member:
                        if np.lib.format.read_magic(member) != (1, 0):
                            return False
                        shape, fortran_order, dtype = (
                            np.lib.format.read_array_header_1_0(
                                member,
                                max_header_size=1024,
                            )
                        )
                        expected_shape = expected_shapes[entry.filename]
                        expected_bytes = math.prod(expected_shape) * 4
                        if (
                            tuple(shape) != expected_shape
                            or fortran_order
                            or dtype != np.dtype("float32")
                            or member.tell() + expected_bytes != entry.file_size
                        ):
                            return False
        except (
            EOFError,
            OSError,
            ValueError,
            zipfile.BadZipFile,
            zipfile.LargeZipFile,
        ):
            return False
        return True

    def _load_generation(
        self,
        video_id: str,
        generation_id: str,
    ) -> tuple[dict[str, object], list[LighthouseEncodedWindow]] | None:
        try:
            video_dir = self._video_dir(video_id)
            generations = video_dir / "generations"
            if generations.is_symlink() or not generations.is_dir():
                return None
            generation = generations / generation_id
            generation.resolve().relative_to(
                generations.resolve(strict=True)
            )
        except (OSError, ValueError):
            return None
        return self._load_generation_path(generation, generation_id)

    def activate(self, video_id: str, generation_id: str) -> None:
        video_dir = self._video_dir(video_id)
        if self._load_generation(video_id, generation_id) is None:
            raise ValueError("Lighthouse generation cannot be activated")
        atomic_write_json(
            video_dir / "active.json",
            {
                "generation_id": generation_id,
                "schema_version": LIGHTHOUSE_CACHE_SCHEMA_VERSION,
                "specification_hash": self.specification.identity,
            },
            sort_keys=True,
        )
        _fsync_directory(video_dir)

    def _active_pointer(self, video_id: str) -> tuple[str, bytes] | None:
        try:
            pointer = self._video_dir(video_id) / "active.json"
        except ValueError:
            return None
        snapshot = _read_bounded_regular_file(pointer, MAX_ACTIVE_POINTER_BYTES)
        if snapshot is None:
            return None
        try:
            payload: object = json.loads(snapshot.decode("utf-8"))
        except (OSError, UnicodeError, ValueError, TypeError):
            return None
        if (
            not isinstance(payload, dict)
            or set(payload) != {"generation_id", "schema_version", "specification_hash"}
            or payload.get("schema_version") != LIGHTHOUSE_CACHE_SCHEMA_VERSION
            or payload.get("specification_hash") != self.specification.identity
            or type(payload.get("generation_id")) is not str
            or _GENERATION_ID_RE.fullmatch(str(payload["generation_id"])) is None
        ):
            return None
        return str(payload["generation_id"]), snapshot

    def active_generation_id(self, video_id: str) -> str | None:
        pointer = self._active_pointer(video_id)
        return pointer[0] if pointer is not None else None

    def load_active(
        self,
        video_id: str,
    ) -> tuple[str, dict[str, object], list[LighthouseEncodedWindow]] | None:
        pointer = self._active_pointer(video_id)
        if pointer is None:
            return None
        generation_id, snapshot = pointer
        loaded = self._load_generation(video_id, generation_id)
        if loaded is None:
            return None
        try:
            current_pointer = _read_bounded_regular_file(
                self._video_dir(video_id) / "active.json",
                MAX_ACTIVE_POINTER_BYTES,
            )
        except ValueError:
            return None
        if current_pointer != snapshot:
            return None
        manifest, windows = loaded
        return generation_id, manifest, windows

    def cache_is_current(self, video_id: str) -> bool:
        return self.load_active(video_id) is not None

    def active_generation_descriptors(self) -> list[dict[str, object]]:
        try:
            root = self._safe_root()
        except ValueError:
            return [{"state": "unavailable"}]
        if not root.is_dir() or root.is_symlink():
            return []
        descriptors: list[dict[str, object]] = []
        try:
            children = sorted(root.iterdir(), key=lambda path: path.name)
        except OSError:
            return [{"state": "unavailable"}]
        for child in children:
            if (
                not child.is_dir()
                or child.is_symlink()
                or _VIDEO_ID_RE.fullmatch(child.name) is None
            ):
                continue
            pointer = self._active_pointer(child.name)
            if pointer is None:
                if (child / "active.json").exists():
                    descriptors.append({"state": "stale", "video_id": child.name})
                continue
            generation_id = pointer[0]
            loaded = self.load_active(child.name)
            if loaded is None:
                descriptors.append(
                    {
                        "generation_id": generation_id,
                        "state": "stale",
                        "video_id": child.name,
                    }
                )
                continue
            manifest = loaded[1]
            descriptors.append(
                {
                    "generation_id": generation_id,
                    "source_sha256": manifest["source_sha256"],
                    "specification_hash": manifest["specification_hash"],
                    "state": "current",
                    "video_id": child.name,
                }
            )
        return descriptors


def _ensure_safe_worker_directory(path: Path, *, purpose: str) -> Path:
    directory = Path(os.path.abspath(path))
    try:
        LighthouseGenerationStore._reject_unsafe_ancestors(directory)
        directory.mkdir(parents=True, exist_ok=True)
        LighthouseGenerationStore._reject_unsafe_ancestors(directory)
    except (OSError, ValueError) as error:
        raise ValueError(f"unsafe Lighthouse {purpose} path") from error
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"unsafe Lighthouse {purpose} path")
    return directory


class LighthouseWorkerRuntime(Protocol):
    specification: LighthouseSpecification
    model_identity: str
    runtime_identity: str

    @property
    def loaded(self) -> bool: ...

    @property
    def available(self) -> bool: ...

    def build_generation(self, request: LighthousePrepareRequest, source: Path) -> str: ...

    def activate_generation(self, video_id: str, generation_id: str) -> None: ...

    def search(self, request: LighthouseSearchRequest) -> list[LighthouseHitPayload]: ...


class LocalLighthouseWorkerRuntime:
    """Lazy Torch/QD-DETR implementation, imported only by the worker process."""

    specification = DEFAULT_LIGHTHOUSE_SPECIFICATION
    model_identity = LIGHTHOUSE_MODEL_IDENTITY
    runtime_identity = LIGHTHOUSE_WORKER_RUNTIME_IDENTITY

    def __init__(
        self,
        *,
        checkpoint: Path,
        clip_checkpoint: Path,
        cache_root: Path,
        work_root: Path,
        ffmpeg: FFmpeg | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.clip_checkpoint = Path(clip_checkpoint)
        self.store = LighthouseGenerationStore(cache_root, self.specification)
        self.work_root = Path(os.path.abspath(work_root))
        self.ffmpeg = ffmpeg or FFmpeg()
        self._model: Any | None = None
        self._load_lock = Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @staticmethod
    def _trusted_file(path: Path, expected_sha256: str) -> bool:
        if not path.is_file() or path.is_symlink():
            return False
        try:
            return hmac.compare_digest(_file_sha256(path), expected_sha256)
        except OSError:
            return False

    @property
    def available(self) -> bool:
        if not _worker_environment_is_exact():
            return False
        if not self._trusted_file(self.checkpoint, LIGHTHOUSE_CHECKPOINT_SHA256):
            return False
        if not self._trusted_file(self.clip_checkpoint, LIGHTHOUSE_CLIP_CHECKPOINT_SHA256):
            return False
        return all(
            importlib.util.find_spec(name) is not None
            for name in ("clip", "lighthouse", "torch")
        )

    def _safe_work_root(self) -> Path:
        return _ensure_safe_worker_directory(self.work_root, purpose="work")

    def _load(self):  # type: ignore[no-untyped-def]
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            if not self.available:
                raise RuntimeError("Lighthouse worker model artifacts are unavailable")
            from videoscope.providers.lighthouse_qdetr import QDDETRPredictor

            self._model = QDDETRPredictor(
                str(self.checkpoint),
                device="cpu",
                feature_name="clip",
                checkpoint_sha256=LIGHTHOUSE_CHECKPOINT_SHA256,
                clip_checkpoint_path=str(self.clip_checkpoint),
                clip_checkpoint_sha256=LIGHTHOUSE_CLIP_CHECKPOINT_SHA256,
            )
            return self._model

    @staticmethod
    def _to_numpy(value: object) -> np.ndarray:
        current: Any = value
        for method in ("detach", "cpu"):
            function = getattr(current, method, None)
            if callable(function):
                current = function()
        return np.asarray(current, dtype=np.float32)

    def build_generation(self, request: LighthousePrepareRequest, source: Path) -> str:
        work_root = self._safe_work_root()
        model = self._load()
        work = work_root / f"lighthouse-{request.request_id}"
        work.mkdir(exist_ok=False)
        if work.is_symlink() or not work.is_dir():
            raise ValueError("unsafe Lighthouse work path")
        windows: list[LighthouseEncodedWindow] = []
        try:
            offset = 0.0
            index = 0
            while offset < request.duration_seconds:
                end = min(
                    request.duration_seconds,
                    offset + self.specification.max_window_seconds,
                )
                clip_path = work / f"window-{index:04d}.mp4"
                try:
                    self.ffmpeg.export_clip(source, clip_path, offset, end)
                    encoded = model.encode_video(str(clip_path))
                finally:
                    clip_path.unlink(missing_ok=True)
                if not isinstance(encoded, dict) or set(encoded) != {
                    "audio_feats",
                    "video_feats",
                    "video_mask",
                } or encoded["audio_feats"] is not None:
                    raise ValueError("Lighthouse encoder returned an invalid feature payload")
                windows.append(
                    LighthouseEncodedWindow(
                        offset=offset,
                        end=end,
                        video_features=self._to_numpy(encoded["video_feats"]),
                        video_mask=self._to_numpy(encoded["video_mask"]),
                    )
                )
                offset = end
                index += 1
            return self.store.build_generation(
                video_id=request.video_id,
                source_sha256=request.source_sha256,
                source_size_bytes=request.source_size_bytes,
                duration_seconds=request.duration_seconds,
                windows=windows,
            )
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def activate_generation(self, video_id: str, generation_id: str) -> None:
        self.store.activate(video_id, generation_id)

    def search(self, request: LighthouseSearchRequest) -> list[LighthouseHitPayload]:
        cached: list[tuple[str, str, list[LighthouseEncodedWindow]]] = []
        for video_id in request.video_ids:
            active = self.store.load_active(video_id)
            if active is not None:
                cached.append((video_id, active[0], active[2]))
        if not cached:
            return []
        model = self._load()
        import torch

        hits: list[LighthouseHitPayload] = []
        for video_id, generation_id, windows in cached:
            for window_index, window in enumerate(windows):
                prediction = model.predict(
                    request.query,
                    {
                        "audio_feats": None,
                        "video_feats": torch.from_numpy(window.video_features.copy()),
                        "video_mask": torch.from_numpy(window.video_mask.copy()),
                    },
                )
                raw_windows = (
                    prediction.get("pred_relevant_windows")
                    if isinstance(prediction, dict)
                    else None
                )
                if not isinstance(raw_windows, list):
                    continue
                for rank, raw in enumerate(raw_windows[:100]):
                    try:
                        start, end, score = (float(value) for value in raw)
                    except (TypeError, ValueError):
                        continue
                    if not all(math.isfinite(value) for value in (start, end, score)):
                        continue
                    absolute_start = max(window.offset, window.offset + start)
                    absolute_end = min(window.end, window.offset + end)
                    if absolute_end <= absolute_start:
                        continue
                    hits.append(
                        LighthouseHitPayload(
                            video_id=video_id,
                            generation_id=generation_id,
                            window_index=window_index,
                            rank=rank,
                            start=absolute_start,
                            end=absolute_end,
                            score=max(0.0, min(1.0, score)),
                        )
                    )
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[: request.limit]


class _WorkerBoundaryMiddleware:
    def __init__(self, app: ASGIApp, *, api_key: str, max_request_bytes: int) -> None:
        self.app = app
        self.expected_authorization = f"Bearer {api_key}".encode("ascii")
        self.max_request_bytes = max_request_bytes

    @staticmethod
    async def _error(
        scope: Scope,
        receive: Receive,
        send: Send,
        status_code: int,
        detail: str,
    ) -> None:
        headers = {"WWW-Authenticate": "Bearer"} if status_code == 401 else None
        await JSONResponse({"detail": detail}, status_code=status_code, headers=headers)(
            scope, receive, send
        )

    @staticmethod
    def _is_loopback(scope: Scope) -> bool:
        client = scope.get("client")
        if not client:
            return False
        try:
            return ip_address(str(client[0])).is_loopback
        except ValueError:
            return False

    def _authorized(self, scope: Scope) -> bool:
        supplied = next(
            (
                value
                for name, value in scope.get("headers", [])
                if name.lower() == b"authorization"
            ),
            b"",
        )
        return secrets.compare_digest(supplied, self.expected_authorization)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        protected = scope["type"] == "http" and str(scope.get("path", "")).startswith("/v1/")
        if not protected:
            await self.app(scope, receive, send)
            return
        if not self._is_loopback(scope):
            await self._error(scope, receive, send, 403, "Loopback access only")
            return
        if not self._authorized(scope):
            await self._error(scope, receive, send, 401, "Unauthorized")
            return
        if scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return

        headers = {name.lower(): value for name, value in scope.get("headers", [])}
        raw_length = headers.get(b"content-length")
        if raw_length is not None:
            try:
                declared = int(raw_length)
            except ValueError:
                await self._error(scope, receive, send, 400, "Invalid Content-Length")
                return
            if declared < 0:
                await self._error(scope, receive, send, 400, "Invalid Content-Length")
                return
            if declared > self.max_request_bytes:
                await self._error(scope, receive, send, 413, "Request body too large")
                return

        chunks: list[bytes] = []
        received = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            chunk = message.get("body", b"")
            received += len(chunk)
            if received > self.max_request_bytes:
                await self._error(scope, receive, send, 413, "Request body too large")
                return
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        delivered = False

        async def replay() -> Message:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        await self.app(scope, replay, send)


def _validate_input(
    input_root: Path,
    request: LighthousePrepareRequest,
    max_input_bytes: int,
) -> Path:
    relative = PurePosixPath(request.relative_path)
    current = input_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("symlink inputs are not allowed")
    try:
        source = input_root.joinpath(*relative.parts).resolve(strict=True)
        source.relative_to(input_root)
        metadata = source.stat()
    except (OSError, ValueError) as error:
        raise ValueError("input is not under the configured root") from error
    if not stat.S_ISREG(metadata.st_mode) or source.is_symlink():
        raise ValueError("input is not a regular file")
    if metadata.st_size <= 0:
        raise ValueError("input must not be empty")
    if metadata.st_size > max_input_bytes:
        raise OverflowError("input is too large")
    if source.suffix.casefold() not in _VIDEO_SUFFIXES:
        raise ValueError("input extension is not supported")
    return source


def _request_identity_matches(runtime: LighthouseWorkerRuntime, request: object) -> bool:
    return (
        getattr(request, "specification_hash", None) == runtime.specification.identity
        and getattr(request, "model_identity", None) == runtime.model_identity
        and getattr(request, "runtime_identity", None) == runtime.runtime_identity
    )


def create_lighthouse_worker_app(
    *,
    runtime: LighthouseWorkerRuntime,
    input_root: Path,
    api_key: str,
    max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES,
    max_request_bytes: int = MAX_REQUEST_BYTES,
) -> FastAPI:
    if not _TOKEN_RE.fullmatch(api_key):
        raise ValueError("Lighthouse worker API key must be 32-256 URL-safe characters")
    if max_input_bytes <= 0 or max_request_bytes <= 0:
        raise ValueError("Lighthouse worker size limits must be positive")
    lexical_root = Path(os.path.abspath(input_root))
    try:
        LighthouseGenerationStore._reject_unsafe_ancestors(lexical_root)
        resolved_root = lexical_root.resolve(strict=True)
    except (OSError, ValueError) as error:
        raise ValueError("Lighthouse worker input root must be a safe directory") from error
    if lexical_root.is_symlink() or not resolved_root.is_dir():
        raise ValueError("Lighthouse worker input root must be a safe directory")
    if runtime.runtime_identity != runtime.specification.runtime_identity:
        raise ValueError("Lighthouse runtime and specification identities disagree")
    app = FastAPI(
        title="VideoScope Lighthouse Worker",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    capacity = BoundedSemaphore(1)

    @app.get("/v1/health", response_model=LighthouseHealthResponse)
    def health() -> LighthouseHealthResponse:
        return LighthouseHealthResponse(
            schema_version=LIGHTHOUSE_WORKER_SCHEMA_VERSION,
            status="ok" if runtime.available else "unavailable",
            model_identity=runtime.model_identity,
            runtime_identity=runtime.runtime_identity,
            specification=runtime.specification.to_dict(),
            specification_hash=runtime.specification.identity,
            loaded=runtime.loaded,
            operations=["prepare", "search"],
            max_input_bytes=max_input_bytes,
            max_query_chars=MAX_QUERY_CHARS,
            max_video_ids=MAX_VIDEO_IDS,
            max_hits=MAX_HITS,
            max_concurrency=1,
        )

    @app.post("/v1/prepare", response_model=LighthousePrepareResponse)
    async def prepare(request: LighthousePrepareRequest) -> LighthousePrepareResponse:
        if not _request_identity_matches(runtime, request):
            raise HTTPException(409, "Lighthouse worker identity does not match")
        try:
            source = _validate_input(resolved_root, request, max_input_bytes)
            before = _source_snapshot(source)
        except OverflowError as error:
            raise HTTPException(413, "Lighthouse worker input is too large") from error
        except (OSError, RuntimeError, ValueError) as error:
            raise HTTPException(400, "Invalid Lighthouse worker input") from error
        if before[0] != request.source_sha256 or before[1] != request.source_size_bytes:
            raise HTTPException(409, "Lighthouse source identity does not match")
        if not capacity.acquire(blocking=False):
            raise HTTPException(429, "Lighthouse worker is busy")
        try:
            try:
                generation_id = await run_in_threadpool(runtime.build_generation, request, source)
            except Exception as error:
                logger.exception("Lighthouse worker preparation failed")
                raise HTTPException(503, "Lighthouse worker preparation failed") from error
            if _GENERATION_ID_RE.fullmatch(generation_id) is None:
                logger.error("Lighthouse runtime returned an invalid generation ID")
                raise HTTPException(503, "Lighthouse worker preparation failed")
            try:
                after = _source_snapshot(source)
            except (OSError, RuntimeError) as error:
                raise HTTPException(409, "Lighthouse source changed during preparation") from error
            if after != before:
                raise HTTPException(409, "Lighthouse source changed during preparation")
            try:
                await run_in_threadpool(
                    runtime.activate_generation,
                    request.video_id,
                    generation_id,
                )
            except Exception as error:
                logger.exception("Lighthouse worker activation failed")
                raise HTTPException(503, "Lighthouse worker preparation failed") from error
        finally:
            capacity.release()
        return LighthousePrepareResponse(
            schema_version=LIGHTHOUSE_WORKER_SCHEMA_VERSION,
            request_id=request.request_id,
            specification_hash=runtime.specification.identity,
            model_identity=runtime.model_identity,
            runtime_identity=runtime.runtime_identity,
            video_id=request.video_id,
            generation_id=generation_id,
        )

    @app.post("/v1/search", response_model=LighthouseSearchResponse)
    async def search(request: LighthouseSearchRequest) -> LighthouseSearchResponse:
        if not _request_identity_matches(runtime, request):
            raise HTTPException(409, "Lighthouse worker identity does not match")
        if not capacity.acquire(blocking=False):
            raise HTTPException(429, "Lighthouse worker is busy")
        try:
            try:
                raw_hits = await run_in_threadpool(runtime.search, request)
                hits = [LighthouseHitPayload.model_validate(hit) for hit in raw_hits]
            except Exception as error:
                logger.exception("Lighthouse worker search failed")
                raise HTTPException(503, "Lighthouse worker search failed") from error
        finally:
            capacity.release()
        return LighthouseSearchResponse(
            schema_version=LIGHTHOUSE_WORKER_SCHEMA_VERSION,
            request_id=request.request_id,
            specification_hash=runtime.specification.identity,
            model_identity=runtime.model_identity,
            runtime_identity=runtime.runtime_identity,
            hits=hits[: request.limit],
        )

    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost", "[::1]"],
    )
    app.add_middleware(
        _WorkerBoundaryMiddleware,
        api_key=api_key,
        max_request_bytes=max_request_bytes,
    )
    return app


class HTTPClient(Protocol):
    def get(  # type: ignore[no-untyped-def]
        self,
        url: str,
        *,
        headers: dict[str, str],
        timeout: float,
    ): ...

    def post(  # type: ignore[no-untyped-def]
        self,
        url: str,
        *,
        json: object,
        headers: dict[str, str],
        timeout: float,
    ): ...


class _BufferedJSONResponse:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self.payload


class _BoundedHTTPClient:
    def __init__(self, max_response_bytes: int = MAX_RESPONSE_BYTES) -> None:
        self.max_response_bytes = max_response_bytes

    def _request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        timeout: float,
        json_payload: object | None = None,
    ) -> _BufferedJSONResponse:
        import httpx

        kwargs: dict[str, object] = {"headers": headers, "timeout": timeout}
        if json_payload is not None:
            kwargs["json"] = json_payload
        with httpx.Client(trust_env=False, follow_redirects=False) as client:
            with client.stream(method, url, **kwargs) as response:
                response.raise_for_status()
                raw_length = response.headers.get("Content-Length")
                if raw_length is not None:
                    try:
                        declared = int(raw_length)
                    except ValueError as error:
                        raise RuntimeError(
                            "Lighthouse worker returned invalid Content-Length"
                        ) from error
                    if declared < 0 or declared > self.max_response_bytes:
                        raise RuntimeError("Lighthouse worker response is too large")
                body = bytearray()
                for chunk in response.iter_bytes(chunk_size=8192):
                    body.extend(chunk)
                    if len(body) > self.max_response_bytes:
                        raise RuntimeError("Lighthouse worker response is too large")
        return _BufferedJSONResponse(json.loads(bytes(body)))

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str],
        timeout: float,
    ) -> _BufferedJSONResponse:
        return self._request("GET", url, headers=headers, timeout=timeout)

    def post(
        self,
        url: str,
        *,
        json: object,
        headers: dict[str, str],
        timeout: float,
    ) -> _BufferedJSONResponse:
        return self._request(
            "POST",
            url,
            headers=headers,
            timeout=timeout,
            json_payload=json,
        )


def _validate_endpoint(endpoint: str) -> str:
    parsed = urlsplit(endpoint.strip())
    if (
        parsed.scheme != "http"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("Lighthouse worker endpoint must be a plain loopback HTTP origin")
    try:
        address = ip_address(parsed.hostname.casefold())
    except ValueError as error:
        raise ValueError("Lighthouse worker endpoint must use 127.0.0.1") from error
    if str(address) != "127.0.0.1":
        raise ValueError("Lighthouse worker endpoint must use 127.0.0.1")
    try:
        parsed.port
    except ValueError as error:
        raise ValueError("Lighthouse worker endpoint contains an invalid port") from error
    return endpoint.strip().rstrip("/")


class LighthouseWorkerClient:
    """Backend adapter preserving the existing moment-retriever public API."""

    id = "lighthouse"

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        input_root: Path,
        cache_dir: Path,
        timeout: float = 300.0,
        client: HTTPClient | None = None,
        specification: LighthouseSpecification = DEFAULT_LIGHTHOUSE_SPECIFICATION,
    ) -> None:
        self.endpoint = _validate_endpoint(endpoint)
        if not _TOKEN_RE.fullmatch(api_key):
            raise ValueError("Lighthouse worker API key must be 32-256 URL-safe characters")
        if not 0 < timeout <= 3600:
            raise ValueError("Lighthouse worker timeout must be between 0 and 3600 seconds")
        self._api_key = api_key
        self.input_root = Path(input_root).resolve()
        self.timeout = timeout
        self.specification = specification
        self.store = LighthouseGenerationStore(Path(cache_dir) / "lighthouse", specification)
        self.client = client
        self._client_lock = Lock()
        self._status_lock = Lock()
        self._cached_status: tuple[float, ProviderStatus] | None = None

    @property
    def checkpoint_sha256(self) -> str:
        return self.specification.checkpoint_sha256

    @property
    def max_window_seconds(self) -> float:
        return self.specification.max_window_seconds

    @property
    def cache_identity(self) -> dict[str, object]:
        return self.specification.to_dict()

    @property
    def identity(self) -> dict[str, object]:
        return {
            "active_generations": self.store.active_generation_descriptors(),
            "endpoint": self.endpoint,
            "mode": "isolated-worker",
            "model_identity": LIGHTHOUSE_MODEL_IDENTITY,
            "runtime_identity": LIGHTHOUSE_WORKER_RUNTIME_IDENTITY,
            "specification": self.specification.to_dict(),
            "specification_hash": self.specification.identity,
            "worker_contract": LIGHTHOUSE_WORKER_SCHEMA_VERSION,
        }

    def _http_client(self) -> HTTPClient:
        if self.client is None:
            with self._client_lock:
                if self.client is None:
                    self.client = _BoundedHTTPClient()
        return self.client

    def _headers(self, *, json_request: bool = False) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        if json_request:
            headers["Content-Type"] = "application/json"
        return headers

    def status(self) -> ProviderStatus:
        with self._status_lock:
            if self._cached_status is not None and self._cached_status[0] > monotonic():
                return self._cached_status[1]
        try:
            response = self._http_client().get(
                f"{self.endpoint}/v1/health",
                headers=self._headers(),
                timeout=min(self.timeout, 5.0),
            )
            response.raise_for_status()
            health = LighthouseHealthResponse.model_validate(response.json())
            exact = (
                health.status == "ok"
                and health.model_identity == LIGHTHOUSE_MODEL_IDENTITY
                and health.runtime_identity == LIGHTHOUSE_WORKER_RUNTIME_IDENTITY
                and health.specification == self.specification.to_dict()
                and health.specification_hash == self.specification.identity
            )
        except Exception:
            exact = False
        status = ProviderStatus(
            self.id,
            "Lighthouse QD-DETR",
            ProviderState.READY if exact else ProviderState.UNAVAILABLE,
            "Lighthouse worker is ready"
            if exact
            else "Lighthouse worker is unreachable or incompatible",
            optional=True,
        )
        with self._status_lock:
            self._cached_status = (monotonic() + (5.0 if exact else 2.0), status)
        return status

    def _relative_source(self, source: Path) -> str:
        lexical = Path(os.path.abspath(source))
        try:
            relative = lexical.relative_to(self.input_root)
        except ValueError as error:
            raise RuntimeError("Lighthouse source is outside configured media root") from error
        return PurePosixPath(*relative.parts).as_posix()

    def _validate_response_identity(  # type: ignore[no-untyped-def]
        self,
        payload: object,
        request_id: str,
    ):
        if (
            getattr(payload, "request_id", None) != request_id
            or getattr(payload, "specification_hash", None) != self.specification.identity
            or getattr(payload, "model_identity", None) != LIGHTHOUSE_MODEL_IDENTITY
            or getattr(payload, "runtime_identity", None) != LIGHTHOUSE_WORKER_RUNTIME_IDENTITY
        ):
            raise ValueError("Lighthouse worker response identity mismatch")
        return payload

    def cache_is_current(self, video_id: str) -> bool:
        return self.store.cache_is_current(video_id)

    def prepare(self, video_id: str, source: Path, duration: float) -> None:
        try:
            source_sha, source_size, _ = _source_snapshot(Path(source))
            request_id = uuid4().hex
            request = LighthousePrepareRequest(
                schema_version=LIGHTHOUSE_WORKER_SCHEMA_VERSION,
                request_id=request_id,
                specification_hash=self.specification.identity,
                model_identity=LIGHTHOUSE_MODEL_IDENTITY,
                runtime_identity=LIGHTHOUSE_WORKER_RUNTIME_IDENTITY,
                video_id=video_id,
                relative_path=self._relative_source(source),
                duration_seconds=float(duration),
                source_sha256=source_sha,
                source_size_bytes=source_size,
            )
            response = self._http_client().post(
                f"{self.endpoint}/v1/prepare",
                json=request.model_dump(mode="json"),
                headers=self._headers(json_request=True),
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = LighthousePrepareResponse.model_validate(response.json())
            self._validate_response_identity(payload, request_id)
            if payload.video_id != video_id:
                raise ValueError("Lighthouse worker response video mismatch")
        except Exception:
            raise RuntimeError("Lighthouse worker preparation failed") from None

    def search(
        self,
        query: str,
        video_ids: list[str],
        *,
        limit: int = 30,
    ) -> list[EvidenceHit]:
        request_id = uuid4().hex
        try:
            request = LighthouseSearchRequest(
                schema_version=LIGHTHOUSE_WORKER_SCHEMA_VERSION,
                request_id=request_id,
                specification_hash=self.specification.identity,
                model_identity=LIGHTHOUSE_MODEL_IDENTITY,
                runtime_identity=LIGHTHOUSE_WORKER_RUNTIME_IDENTITY,
                query=query,
                video_ids=video_ids,
                limit=limit,
            )
            response = self._http_client().post(
                f"{self.endpoint}/v1/search",
                json=request.model_dump(mode="json"),
                headers=self._headers(json_request=True),
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = LighthouseSearchResponse.model_validate(response.json())
            self._validate_response_identity(payload, request_id)
        except Exception:
            raise RuntimeError("Lighthouse worker search failed") from None
        return [
            EvidenceHit(
                video_id=hit.video_id,
                segment_id=(
                    f"lighthouse:{hit.video_id}:{hit.generation_id}:"
                    f"{hit.window_index:04d}:{hit.rank:04d}"
                ),
                start=hit.start,
                end=hit.end,
                modality="lighthouse",
                score=hit.score,
                text=query,
                metadata={
                    "generation_id": hit.generation_id,
                    "source": "lighthouse",
                    "window": f"window-{hit.window_index:04d}",
                },
            )
            for hit in payload.hits
        ]


class LighthouseWorkerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        populate_by_name=True,
        extra="ignore",
    )

    api_key: str = Field(
        min_length=32,
        max_length=256,
        pattern=r"^[A-Za-z0-9._~-]+$",
        repr=False,
        validation_alias=AliasChoices(
            "LIGHTHOUSE_API_KEY",
            "VIDEOSCOPE_LIGHTHOUSE_API_KEY",
        ),
    )
    data_dir: Path = Field(default=Path("data"), validation_alias="VIDEOSCOPE_DATA_DIR")
    checkpoint: Path | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "LIGHTHOUSE_CHECKPOINT",
            "VIDEOSCOPE_LIGHTHOUSE_CHECKPOINT",
        ),
    )
    clip_checkpoint: Path | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "LIGHTHOUSE_CLIP_CHECKPOINT",
            "VIDEOSCOPE_LIGHTHOUSE_CLIP_CHECKPOINT",
        ),
    )
    port: int = Field(
        default=8782,
        ge=1,
        le=65_535,
        validation_alias=AliasChoices(
            "LIGHTHOUSE_WORKER_PORT",
            "VIDEOSCOPE_LIGHTHOUSE_WORKER_PORT",
        ),
    )

    @property
    def media_root(self) -> Path:
        return self.data_dir / "media"

    @property
    def cache_root(self) -> Path:
        return self.data_dir / "cache" / "lighthouse"

    @property
    def work_root(self) -> Path:
        return self.data_dir / "tmp" / "lighthouse-worker"

    @property
    def checkpoint_path(self) -> Path:
        return self.checkpoint or self.data_dir / "models/lighthouse/clip_qd_detr_qvhighlight.ckpt"

    @property
    def clip_checkpoint_path(self) -> Path:
        return self.clip_checkpoint or self.data_dir / "models/lighthouse/ViT-B-32.pt"


def main() -> None:
    try:
        settings = LighthouseWorkerSettings()
    except ValidationError:
        raise RuntimeError("Lighthouse worker configuration is invalid") from None
    try:
        media_root = _ensure_safe_worker_directory(settings.media_root, purpose="media")
        cache_root = _ensure_safe_worker_directory(settings.cache_root, purpose="cache")
        work_root = _ensure_safe_worker_directory(settings.work_root, purpose="work")
    except ValueError:
        raise RuntimeError("Lighthouse worker directory configuration is invalid") from None
    runtime = LocalLighthouseWorkerRuntime(
        checkpoint=settings.checkpoint_path,
        clip_checkpoint=settings.clip_checkpoint_path,
        cache_root=cache_root,
        work_root=work_root,
    )
    app = create_lighthouse_worker_app(
        runtime=runtime,
        input_root=media_root,
        api_key=settings.api_key,
    )
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=settings.port, workers=1)


if __name__ == "__main__":
    main()
