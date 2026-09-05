from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
from hashlib import sha256
import importlib.metadata
import importlib.util
from ipaddress import ip_address
import json
import logging
import os
import platform
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
import sys
import tempfile
from threading import BoundedSemaphore, Lock
from time import monotonic
import traceback
from typing import Any, Literal, Protocol, Self
from urllib.parse import urlsplit
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
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

from videoscope.model_manifest import model_identity, model_revision
from videoscope.providers.qwen_video import (
    QWEN_INFERENCE_RUNTIME_IDENTITY,
    QWEN_PROMPT_PROTOCOL_SHA256,
    QwenInferenceStatus,
    QwenVideoJudgement,
    QwenVideoReranker,
    parse_qwen_judgement,
)


logger = logging.getLogger(__name__)

QWEN_WORKER_SCHEMA_VERSION = "qwen-worker-v4"
MAX_REQUEST_BYTES = 64 * 1024
MAX_RESPONSE_BYTES = 64 * 1024
MAX_QUERY_CHARS = 500
MAX_TOKENS = 1024
DEFAULT_MAX_INPUT_BYTES = 128 * 1024 * 1024
_REQUEST_ID_RE = r"^[0-9a-f]{32}$"
_TOKEN_RE = re.compile(r"^[A-Za-z0-9._~-]{32,256}$")
_VIDEO_SUFFIXES = frozenset({".mp4"})
_STORYBOARD_SUFFIXES = frozenset({".jpg", ".jpeg"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _project_path(*parts: str) -> Path:
    source = Path(__file__).resolve()
    candidates = [Path.cwd().joinpath(*parts)]
    if len(source.parents) > 4:
        candidates.append(source.parents[4].joinpath(*parts))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[-1]


DEFAULT_QWEN_RUNTIME_MANIFEST = _project_path("workers", "qwen", "runtime.lock.json")
DEFAULT_QWEN_MODEL_MANIFEST = _project_path(
    "workers", "qwen", "model-artifacts.lock.json"
)
DEFAULT_BACKEND_LOCK = _project_path("backend", "uv.lock")
QWEN_RUNTIME_MANIFEST_SHA256 = (
    "0481dda4b0aef9927d5f9adb1a5f7292f997cac39a6d59ecef0087913d80335f"
)
QWEN_MODEL_MANIFEST_SHA256 = (
    "abaeb6d14ccbdc1741cccea700edd3385eb7cb3d215796b465c4d5a1a0504fe0"
)


def _canonical_json_sha256(payload: object) -> str:
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as error:
        raise RuntimeError("Qwen worker manifest is invalid") from error
    return sha256(encoded).hexdigest()


def _qwen_source_bundle_sha256() -> str:
    source_dir = Path(__file__).resolve().parent
    files: list[dict[str, str]] = []
    for name in ("qwen_video.py", "qwen_worker.py"):
        try:
            digest = sha256((source_dir / name).read_bytes()).hexdigest()
        except OSError as error:
            raise RuntimeError("Qwen worker source bundle is unavailable") from error
        files.append(
            {
                "logical_path": f"videoscope/providers/{name}",
                "sha256": digest,
            }
        )
    return _canonical_json_sha256({"files": files, "schema_version": 1})


QWEN_SOURCE_BUNDLE_SHA256 = _qwen_source_bundle_sha256()


def _input_root_identity(path: Path) -> str:
    try:
        resolved = Path(path).resolve(strict=True)
        descriptor = os.open(
            resolved,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except (OSError, RuntimeError) as error:
        raise RuntimeError("Qwen worker input root is unavailable or unsafe") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError("Qwen worker input root is not a directory")
        return _canonical_json_sha256(
            {
                "device": metadata.st_dev,
                "group": metadata.st_gid,
                "inode": metadata.st_ino,
                "mode": stat.S_IMODE(metadata.st_mode),
                "owner": metadata.st_uid,
                "schema_version": 1,
            }
        )
    finally:
        os.close(descriptor)


def _load_manifest(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Qwen worker manifest is unavailable or invalid") from error
    if not isinstance(payload, dict):
        raise RuntimeError("Qwen worker manifest is invalid")
    return payload


def _normalized_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).casefold()


def _installed_distributions() -> dict[str, str]:
    installed: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not isinstance(name, str) or not name:
            raise RuntimeError("Qwen worker installed distribution has no name")
        normalized = _normalized_distribution_name(name)
        if normalized in installed:
            raise RuntimeError("Qwen worker has duplicate installed distributions")
        installed[normalized] = distribution.version
    return installed


def _host_matches_qwen_contract(
    *,
    python_version: tuple[int, int, int],
    system: str,
    machine: str,
    macos_version: str,
) -> bool:
    try:
        macos_major = int(macos_version.split(".", 1)[0])
    except (TypeError, ValueError):
        return False
    return (
        python_version == (3, 12, 13)
        and system == "Darwin"
        and machine.casefold() in {"arm64", "aarch64"}
        and macos_major >= 14
    )


def _verify_model_artifact(path: Path, *, size: int, digest: str, root: Path) -> None:
    try:
        resolved_root = root.resolve(strict=True)
        resolved = path.resolve(strict=True)
        resolved.relative_to(resolved_root)
        descriptor = os.open(
            resolved,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise RuntimeError("Qwen model artifact is missing or unsafe") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != size:
            raise RuntimeError("Qwen model artifact size mismatch")
        checksum = sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, size - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > size:
                raise RuntimeError("Qwen model artifact size mismatch")
            checksum.update(chunk)
        after = os.fstat(descriptor)
        before_fingerprint = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_fingerprint = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if (
            total != size
            or before_fingerprint != after_fingerprint
            or checksum.hexdigest() != digest
        ):
            raise RuntimeError("Qwen model artifact SHA-256 mismatch")
    finally:
        os.close(descriptor)


def attest_qwen_worker_startup(
    model_name: str,
    revision: str,
    *,
    runtime_manifest_path: Path = DEFAULT_QWEN_RUNTIME_MANIFEST,
    model_manifest_path: Path = DEFAULT_QWEN_MODEL_MANIFEST,
    backend_lock_path: Path = DEFAULT_BACKEND_LOCK,
    expected_runtime_manifest_sha256: str = QWEN_RUNTIME_MANIFEST_SHA256,
    expected_model_manifest_sha256: str = QWEN_MODEL_MANIFEST_SHA256,
    expected_source_bundle_sha256: str = QWEN_SOURCE_BUNDLE_SHA256,
    expected_prompt_protocol_sha256: str = QWEN_PROMPT_PROTOCOL_SHA256,
    python_version: tuple[int, int, int] | None = None,
    system: str | None = None,
    machine: str | None = None,
    macos_version: str | None = None,
    installed_distributions: dict[str, str] | None = None,
    snapshot_resolver: Any | None = None,
) -> dict[str, str]:
    """Fail closed unless runtime, packages and every model file are reviewed."""
    if (
        expected_source_bundle_sha256 != QWEN_SOURCE_BUNDLE_SHA256
        or expected_prompt_protocol_sha256 != QWEN_PROMPT_PROTOCOL_SHA256
    ):
        raise RuntimeError("Qwen worker executable protocol identity mismatch")
    runtime = _load_manifest(runtime_manifest_path)
    model = _load_manifest(model_manifest_path)
    if (
        _canonical_json_sha256(runtime) != expected_runtime_manifest_sha256
        or _canonical_json_sha256(model) != expected_model_manifest_sha256
    ):
        raise RuntimeError("Qwen worker manifest identity mismatch")
    if (
        set(runtime)
        != {
            "backend_lock_sha256",
            "distributions",
            "platform",
            "python",
            "schema_version",
        }
        or runtime.get("schema_version") != 1
        or runtime.get("python") != "3.12.13"
        or runtime.get("platform") != "aarch64-apple-darwin-macos14plus"
        or not isinstance(runtime.get("distributions"), dict)
        or not runtime["distributions"]
        or set(model)
        != {
            "artifacts",
            "license",
            "model",
            "revision",
            "schema_version",
        }
        or model.get("schema_version") != 1
        or model.get("license") != "apache-2.0"
        or model.get("model") != model_name
        or model.get("revision") != revision
        or not isinstance(model.get("artifacts"), list)
        or not model["artifacts"]
    ):
        raise RuntimeError("Qwen worker manifest contract mismatch")
    lock_digest = runtime.get("backend_lock_sha256")
    try:
        actual_lock_digest = sha256(backend_lock_path.read_bytes()).hexdigest()
    except OSError as error:
        raise RuntimeError("Qwen worker dependency lock is unavailable") from error
    if (
        type(lock_digest) is not str
        or _SHA256_RE.fullmatch(lock_digest) is None
        or actual_lock_digest != lock_digest
    ):
        raise RuntimeError("Qwen worker dependency lock mismatch")

    actual_python = python_version or tuple(sys.version_info[:3])
    actual_system = system or platform.system()
    actual_machine = machine or platform.machine()
    actual_macos = macos_version if macos_version is not None else platform.mac_ver()[0]
    if not _host_matches_qwen_contract(
        python_version=actual_python,
        system=actual_system,
        machine=actual_machine,
        macos_version=actual_macos,
    ):
        raise RuntimeError("Qwen worker requires the reviewed Apple Silicon runtime")
    distributions = runtime["distributions"]
    assert isinstance(distributions, dict)
    expected_distributions = {
        _normalized_distribution_name(str(name)): str(version)
        for name, version in distributions.items()
    }
    actual_distributions = (
        {
            _normalized_distribution_name(str(name)): str(version)
            for name, version in installed_distributions.items()
        }
        if installed_distributions is not None
        else _installed_distributions()
    )
    if actual_distributions != expected_distributions:
        raise RuntimeError("Qwen worker installed distributions mismatch")

    if snapshot_resolver is None:
        from huggingface_hub import snapshot_download

        def snapshot_resolver(selected_model: str, selected_revision: str) -> Path:
            return Path(
                snapshot_download(
                    selected_model,
                    revision=selected_revision,
                    local_files_only=True,
                )
            )

    try:
        snapshot = Path(snapshot_resolver(model_name, revision))
        snapshot_root = snapshot.resolve(strict=True).parent.parent
    except Exception as error:
        raise RuntimeError("reviewed local Qwen snapshot is unavailable") from error
    artifacts = model["artifacts"]
    assert isinstance(artifacts, list)
    names: set[str] = set()
    for raw in artifacts:
        if not isinstance(raw, dict) or set(raw) != {"name", "sha256", "size"}:
            raise RuntimeError("Qwen model artifact contract is invalid")
        name = raw.get("name")
        digest = raw.get("sha256")
        size = raw.get("size")
        if (
            type(name) is not str
            or not name
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
            or name in names
            or type(digest) is not str
            or _SHA256_RE.fullmatch(digest) is None
            or type(size) is not int
            or size <= 0
        ):
            raise RuntimeError("Qwen model artifact contract is invalid")
        names.add(name)
        _verify_model_artifact(
            snapshot / name,
            size=size,
            digest=digest,
            root=snapshot_root,
        )
    try:
        actual_names = {path.name for path in snapshot.iterdir()}
    except OSError as error:
        raise RuntimeError("reviewed local Qwen snapshot is unavailable") from error
    if actual_names != names:
        raise RuntimeError("Qwen model artifact set mismatch")
    return {
        "model_identity": model_identity(model_name, revision),
        "prompt_protocol_sha256": QWEN_PROMPT_PROTOCOL_SHA256,
        "runtime_identity": QWEN_INFERENCE_RUNTIME_IDENTITY,
        "source_bundle_sha256": QWEN_SOURCE_BUNDLE_SHA256,
    }


class _ContractModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


class QwenJudgeRequest(_ContractModel):
    schema_version: Literal[QWEN_WORKER_SCHEMA_VERSION]
    request_id: str = Field(pattern=_REQUEST_ID_RE)
    model_identity: str = Field(min_length=1, max_length=300)
    runtime_identity: Literal[QWEN_INFERENCE_RUNTIME_IDENTITY]
    source_bundle_sha256: str = Field(pattern=_SHA256_RE.pattern)
    prompt_protocol_sha256: str = Field(pattern=_SHA256_RE.pattern)
    input_root_sha256: str = Field(pattern=_SHA256_RE.pattern)
    input_kind: Literal["video", "storyboard"]
    relative_path: str = Field(min_length=1, max_length=240)
    expected_sha256: str = Field(pattern=_SHA256_RE.pattern)
    expected_byte_size: int = Field(gt=0, le=DEFAULT_MAX_INPUT_BYTES)
    prompt_kind: Literal["basketball_facts", "generic_query"]
    query: str | None = Field(default=None, max_length=MAX_QUERY_CHARS)
    fps: float | None = Field(default=None, ge=0.5, le=8)
    max_tokens: int = Field(ge=64, le=MAX_TOKENS)

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
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

    @model_validator(mode="after")
    def validate_prompt(self) -> Self:
        if self.input_kind == "video":
            if self.fps is None:
                raise ValueError("video input requires fps")
            if self.prompt_kind == "basketball_facts":
                if self.query is not None:
                    raise ValueError(
                        "basketball-facts video requires the fixed prompt"
                    )
            elif not self.query:
                raise ValueError("generic-query video requires a non-empty query")
        elif self.prompt_kind != "generic_query" or not self.query or self.fps is not None:
            raise ValueError("storyboard input requires a non-empty generic query")
        return self


class QwenJudgementPayload(_ContractModel):
    matches_query: bool | None = None
    confidence: float = Field(default=0.0, ge=0, le=1)
    event_start: float | None = Field(default=None, ge=0)
    event_end: float | None = Field(default=None, ge=0)
    shot_attempt: bool | None = None
    ball_through_hoop: bool | None = None
    shooter_outside_arc: bool | None = None
    three_point_signal: bool | None = None
    shooter_jersey: str | None = Field(default=None, pattern=r"^\d{1,3}$")
    evidence: str = Field(default="", max_length=500)
    made: bool | None = None
    three_point: bool | None = None

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if (
            self.event_start is not None
            and self.event_end is not None
            and self.event_end <= self.event_start
        ):
            raise ValueError("event_end must be greater than event_start")
        return self


class QwenJudgeResponse(_ContractModel):
    schema_version: Literal[QWEN_WORKER_SCHEMA_VERSION]
    request_id: str = Field(pattern=_REQUEST_ID_RE)
    model_identity: str = Field(min_length=1, max_length=300)
    runtime_identity: Literal[QWEN_INFERENCE_RUNTIME_IDENTITY]
    source_bundle_sha256: str = Field(pattern=_SHA256_RE.pattern)
    prompt_protocol_sha256: str = Field(pattern=_SHA256_RE.pattern)
    input_root_sha256: str = Field(pattern=_SHA256_RE.pattern)
    source_sha256: str = Field(pattern=_SHA256_RE.pattern)
    byte_size: int = Field(gt=0, le=DEFAULT_MAX_INPUT_BYTES)
    judgement: QwenJudgementPayload


class QwenHealthResponse(_ContractModel):
    schema_version: Literal[QWEN_WORKER_SCHEMA_VERSION]
    status: Literal["ok", "unavailable"]
    model_identity: str = Field(min_length=1, max_length=300)
    runtime_identity: Literal[QWEN_INFERENCE_RUNTIME_IDENTITY]
    source_bundle_sha256: str = Field(pattern=_SHA256_RE.pattern)
    prompt_protocol_sha256: str = Field(pattern=_SHA256_RE.pattern)
    input_root_sha256: str = Field(pattern=_SHA256_RE.pattern)
    loaded: bool
    input_kinds: list[Literal["video", "storyboard"]] = Field(
        min_length=2,
        max_length=2,
    )
    max_input_bytes: int = Field(gt=0)
    max_query_chars: Literal[MAX_QUERY_CHARS]
    max_tokens: Literal[MAX_TOKENS]
    max_concurrency: int = Field(ge=1, le=8)

    @field_validator("input_kinds")
    @classmethod
    def validate_input_kinds(
        cls,
        value: list[Literal["video", "storyboard"]],
    ) -> list[Literal["video", "storyboard"]]:
        if value != ["video", "storyboard"]:
            raise ValueError("input_kinds do not match the contract")
        return value


class QwenProbeRequest(_ContractModel):
    schema_version: Literal[QWEN_WORKER_SCHEMA_VERSION]
    request_id: str = Field(pattern=_REQUEST_ID_RE)
    model_identity: str = Field(min_length=1, max_length=300)
    runtime_identity: Literal[QWEN_INFERENCE_RUNTIME_IDENTITY]
    source_bundle_sha256: str = Field(pattern=_SHA256_RE.pattern)
    prompt_protocol_sha256: str = Field(pattern=_SHA256_RE.pattern)
    input_root_sha256: str = Field(pattern=_SHA256_RE.pattern)
    relative_path: str = Field(min_length=1, max_length=240)
    expected_sha256: str = Field(pattern=_SHA256_RE.pattern)
    expected_byte_size: int = Field(gt=0, le=DEFAULT_MAX_INPUT_BYTES)

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        return QwenJudgeRequest.validate_relative_path(value)


class QwenProbeResponse(_ContractModel):
    schema_version: Literal[QWEN_WORKER_SCHEMA_VERSION]
    request_id: str = Field(pattern=_REQUEST_ID_RE)
    model_identity: str = Field(min_length=1, max_length=300)
    runtime_identity: Literal[QWEN_INFERENCE_RUNTIME_IDENTITY]
    source_bundle_sha256: str = Field(pattern=_SHA256_RE.pattern)
    prompt_protocol_sha256: str = Field(pattern=_SHA256_RE.pattern)
    input_root_sha256: str = Field(pattern=_SHA256_RE.pattern)
    source_sha256: str = Field(pattern=_SHA256_RE.pattern)
    byte_size: int = Field(gt=0, le=DEFAULT_MAX_INPUT_BYTES)
    verified: Literal[True]


class WorkerRuntime(Protocol):
    @property
    def model_identity(self) -> str: ...

    @property
    def loaded(self) -> bool: ...

    @property
    def available(self) -> bool: ...

    def judge(
        self,
        request: QwenJudgeRequest,
        source: Path,
    ) -> QwenVideoJudgement: ...


@dataclass(frozen=True, slots=True)
class _SourceFingerprint:
    device: int
    inode: int
    mode: int
    size: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True, slots=True)
class _DirectoryFingerprint:
    device: int
    inode: int
    mode: int
    size: int
    modified_ns: int
    changed_ns: int


class _WorkerInputValidationError(ValueError):
    pass


class _WorkerInputIdentityError(ValueError):
    pass


@dataclass(slots=True)
class _OpenedWorkerInput:
    descriptor: int
    directory_descriptors: tuple[int, ...]
    directory_fingerprints: tuple[_DirectoryFingerprint, ...]
    source_fingerprint: _SourceFingerprint
    entry_name: str
    expected_sha256: str
    expected_byte_size: int
    _closed: bool = False

    def verify_current(self) -> None:
        if self._closed:
            raise _WorkerInputIdentityError("input descriptor is closed")
        try:
            source_metadata = os.fstat(self.descriptor)
            directory_metadata = tuple(
                os.fstat(descriptor) for descriptor in self.directory_descriptors
            )
            namespace_metadata = os.stat(
                self.entry_name,
                dir_fd=self.directory_descriptors[-1],
                follow_symlinks=False,
            )
        except OSError as error:
            raise _WorkerInputIdentityError("input namespace changed") from error
        if (
            _source_fingerprint(source_metadata) != self.source_fingerprint
            or _source_fingerprint(namespace_metadata) != self.source_fingerprint
            or tuple(_directory_fingerprint(item) for item in directory_metadata)
            != self.directory_fingerprints
        ):
            raise _WorkerInputIdentityError("input namespace changed")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[OSError] = []
        try:
            os.close(self.descriptor)
        except OSError as error:
            errors.append(error)
        for descriptor in reversed(self.directory_descriptors):
            try:
                os.close(descriptor)
            except OSError as error:
                errors.append(error)
        if errors:
            raise errors[0]


@dataclass(slots=True)
class _MaterializedWorkerInput:
    source: _OpenedWorkerInput
    directory: tempfile.TemporaryDirectory[str]
    directory_descriptor: int
    directory_fingerprint: _DirectoryFingerprint
    descriptor: int
    fingerprint: _SourceFingerprint
    path: Path
    _closed: bool = False

    def verify_current(self) -> None:
        self.source.verify_current()
        try:
            metadata = os.fstat(self.descriptor)
            directory_metadata = os.fstat(self.directory_descriptor)
            namespace_metadata = os.stat(
                self.path.name,
                dir_fd=self.directory_descriptor,
                follow_symlinks=False,
            )
            digest, byte_size = _descriptor_identity(
                self.descriptor,
                maximum=self.source.expected_byte_size,
            )
        except (OSError, _WorkerInputValidationError) as error:
            raise _WorkerInputIdentityError("materialized input changed") from error
        if (
            _source_fingerprint(metadata) != self.fingerprint
            or _source_fingerprint(namespace_metadata) != self.fingerprint
            or _directory_fingerprint(directory_metadata) != self.directory_fingerprint
            or digest != self.source.expected_sha256
            or byte_size != self.source.expected_byte_size
        ):
            raise _WorkerInputIdentityError("materialized input changed")

    def close(self) -> bool:
        if self._closed:
            return True
        self._closed = True
        success = True
        for descriptor in (self.descriptor, self.directory_descriptor):
            try:
                os.close(descriptor)
            except OSError:
                success = False
        try:
            self.source.close()
        except OSError:
            success = False
        try:
            self.directory.cleanup()
        except OSError:
            success = False
        return success


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
    """HTTPX transport that cannot use proxy env or buffer unbounded responses."""

    def __init__(self, *, max_response_bytes: int) -> None:
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

        kwargs: dict[str, object] = {
            "headers": headers,
            "timeout": timeout,
        }
        if json_payload is not None:
            kwargs["json"] = json_payload
        with httpx.Client(trust_env=False, follow_redirects=False) as client:
            with client.stream(method, url, **kwargs) as response:
                response.raise_for_status()
                raw_length = response.headers.get("Content-Length")
                if raw_length is not None:
                    try:
                        declared_length = int(raw_length)
                    except ValueError as error:
                        raise RuntimeError(
                            "Qwen worker returned invalid Content-Length"
                        ) from error
                    if (
                        declared_length < 0
                        or declared_length > self.max_response_bytes
                    ):
                        raise RuntimeError("Qwen worker response is too large")
                body = bytearray()
                for chunk in response.iter_bytes(chunk_size=8192):
                    body.extend(chunk)
                    if len(body) > self.max_response_bytes:
                        raise RuntimeError("Qwen worker response is too large")
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


class _WorkerBoundaryMiddleware:
    """Reject remote peers, unauthenticated calls, and oversized JSON before parsing."""

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
        await JSONResponse(
            {"detail": detail},
            status_code=status_code,
            headers=headers,
        )(scope, receive, send)

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
        protected = scope["type"] == "http" and str(
            scope.get("path", "")
        ).startswith("/v1/")
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
                declared_length = int(raw_length)
            except ValueError:
                await self._error(scope, receive, send, 400, "Invalid Content-Length")
                return
            if declared_length < 0:
                await self._error(scope, receive, send, 400, "Invalid Content-Length")
                return
            if declared_length > self.max_request_bytes:
                await self._error(scope, receive, send, 413, "Request body too large")
                return

        chunks: list[bytes] = []
        received_bytes = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            chunk = message.get("body", b"")
            received_bytes += len(chunk)
            if received_bytes > self.max_request_bytes:
                await self._error(scope, receive, send, 413, "Request body too large")
                return
            chunks.append(chunk)
            if not message.get("more_body", False):
                break

        body = b"".join(chunks)
        delivered = False

        async def replay_receive() -> Message:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        await self.app(scope, replay_receive, send)


def _source_fingerprint(metadata: os.stat_result) -> _SourceFingerprint:
    return _SourceFingerprint(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        mode=metadata.st_mode,
        size=metadata.st_size,
        modified_ns=metadata.st_mtime_ns,
        changed_ns=metadata.st_ctime_ns,
    )


def _directory_fingerprint(metadata: os.stat_result) -> _DirectoryFingerprint:
    return _DirectoryFingerprint(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        mode=metadata.st_mode,
        size=metadata.st_size,
        modified_ns=metadata.st_mtime_ns,
        changed_ns=metadata.st_ctime_ns,
    )


def _descriptor_identity(descriptor: int, *, maximum: int) -> tuple[str, int]:
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
    except OSError as error:
        raise _WorkerInputValidationError("input cannot be read") from error
    checksum = sha256()
    total = 0
    while True:
        try:
            chunk = os.read(descriptor, min(1024 * 1024, maximum - total + 1))
        except OSError as error:
            raise _WorkerInputValidationError("input cannot be read") from error
        if not chunk:
            break
        total += len(chunk)
        if total > maximum:
            raise OverflowError("input exceeds the configured size limit")
        checksum.update(chunk)
    return checksum.hexdigest(), total


def _open_worker_input(
    *,
    input_root: Path,
    relative_path: str,
    allowed_suffixes: frozenset[str] | None,
    max_input_bytes: int,
    expected_sha256: str | None = None,
    expected_byte_size: int | None = None,
) -> _OpenedWorkerInput:
    relative = PurePosixPath(relative_path)
    if (
        allowed_suffixes is not None
        and relative.suffix.casefold() not in allowed_suffixes
    ):
        raise _WorkerInputValidationError("input extension is not supported")
    directory_descriptors: list[int] = []
    source_descriptor: int | None = None
    try:
        root_descriptor = os.open(
            input_root,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        directory_descriptors.append(root_descriptor)
        for part in relative.parts[:-1]:
            descriptor = os.open(
                part,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_descriptors[-1],
            )
            directory_descriptors.append(descriptor)
            metadata = os.fstat(descriptor)
            if not stat.S_ISDIR(metadata.st_mode):
                raise _WorkerInputValidationError("input parent is not a directory")
        source_descriptor = os.open(
            relative.parts[-1],
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_descriptors[-1],
        )
        before = os.fstat(source_descriptor)
    except (OSError, IndexError, _WorkerInputValidationError) as error:
        if source_descriptor is not None:
            try:
                os.close(source_descriptor)
            except OSError:
                pass
        for descriptor in reversed(directory_descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise _WorkerInputValidationError(
            "input is not a file under the configured root"
        ) from error

    try:
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise _WorkerInputValidationError(
                "input must be a non-empty regular file"
            )
        if before.st_size > max_input_bytes:
            raise OverflowError("input exceeds the configured size limit")
        if expected_byte_size is not None and before.st_size != expected_byte_size:
            raise _WorkerInputIdentityError("input size does not match request")
        digest, byte_size = _descriptor_identity(
            source_descriptor,
            maximum=max_input_bytes,
        )
        after = os.fstat(source_descriptor)
        if _source_fingerprint(after) != _source_fingerprint(before):
            raise _WorkerInputIdentityError("input changed while being read")
        if expected_sha256 is not None and digest != expected_sha256:
            raise _WorkerInputIdentityError("input digest does not match request")
        directory_fingerprints = tuple(
            _directory_fingerprint(os.fstat(descriptor))
            for descriptor in directory_descriptors
        )
        opened = _OpenedWorkerInput(
            descriptor=source_descriptor,
            directory_descriptors=tuple(directory_descriptors),
            directory_fingerprints=directory_fingerprints,
            source_fingerprint=_source_fingerprint(before),
            entry_name=relative.parts[-1],
            expected_sha256=digest,
            expected_byte_size=byte_size,
        )
        opened.verify_current()
        return opened
    except BaseException:
        try:
            os.close(source_descriptor)
        except OSError:
            pass
        for descriptor in reversed(directory_descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        try:
            written = os.write(descriptor, payload[offset:])
        except OSError as error:
            raise _WorkerInputValidationError(
                "materialized input cannot be written"
            ) from error
        if written <= 0:
            raise _WorkerInputValidationError("materialized input cannot be written")
        offset += written


def _materialize_worker_input(
    *,
    input_root: Path,
    request: QwenJudgeRequest,
    max_input_bytes: int,
) -> _MaterializedWorkerInput:
    allowed = (
        _VIDEO_SUFFIXES
        if request.input_kind == "video"
        else _STORYBOARD_SUFFIXES
    )
    opened = _open_worker_input(
        input_root=input_root,
        relative_path=request.relative_path,
        allowed_suffixes=allowed,
        max_input_bytes=max_input_bytes,
        expected_sha256=request.expected_sha256,
        expected_byte_size=request.expected_byte_size,
    )
    private_directory: tempfile.TemporaryDirectory[str] | None = None
    private_directory_descriptor: int | None = None
    materialized_descriptor: int | None = None
    try:
        private_directory = tempfile.TemporaryDirectory(
            prefix="videoscope-qwen-input-",
        )
        private_root = Path(private_directory.name)
        private_directory_descriptor = os.open(
            private_root,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        destination_name = (
            "input" + PurePosixPath(request.relative_path).suffix.casefold()
        )
        writable_descriptor = os.open(
            destination_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=private_directory_descriptor,
        )
        try:
            os.lseek(opened.descriptor, 0, os.SEEK_SET)
            copied = 0
            while copied < opened.expected_byte_size:
                chunk = os.read(
                    opened.descriptor,
                    min(1024 * 1024, opened.expected_byte_size - copied),
                )
                if not chunk:
                    break
                _write_all(writable_descriptor, chunk)
                copied += len(chunk)
            if copied != opened.expected_byte_size or os.read(opened.descriptor, 1):
                raise _WorkerInputIdentityError("input changed during materialization")
            os.fsync(writable_descriptor)
            os.fchmod(writable_descriptor, 0o400)
        finally:
            os.close(writable_descriptor)
        opened.verify_current()
        materialized_descriptor = os.open(
            destination_name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=private_directory_descriptor,
        )
        materialized_metadata = os.fstat(materialized_descriptor)
        materialized_fingerprint = _source_fingerprint(materialized_metadata)
        if (
            not stat.S_ISREG(materialized_metadata.st_mode)
            or stat.S_IMODE(materialized_metadata.st_mode) != 0o400
            or materialized_metadata.st_size != opened.expected_byte_size
        ):
            raise _WorkerInputIdentityError("materialized input is invalid")
        digest, byte_size = _descriptor_identity(
            materialized_descriptor,
            maximum=opened.expected_byte_size,
        )
        if (
            digest != opened.expected_sha256
            or byte_size != opened.expected_byte_size
        ):
            raise _WorkerInputIdentityError("materialized input identity mismatch")
        return _MaterializedWorkerInput(
            source=opened,
            directory=private_directory,
            directory_descriptor=private_directory_descriptor,
            directory_fingerprint=_directory_fingerprint(
                os.fstat(private_directory_descriptor)
            ),
            descriptor=materialized_descriptor,
            fingerprint=materialized_fingerprint,
            path=private_root / destination_name,
        )
    except BaseException:
        if materialized_descriptor is not None:
            try:
                os.close(materialized_descriptor)
            except OSError:
                pass
        if private_directory_descriptor is not None:
            try:
                os.close(private_directory_descriptor)
            except OSError:
                pass
        try:
            opened.close()
        except OSError:
            pass
        if private_directory is not None:
            private_directory.cleanup()
        raise


def _probe_worker_input(
    *,
    input_root: Path,
    request: QwenProbeRequest,
    max_input_bytes: int,
) -> None:
    opened = _open_worker_input(
        input_root=input_root,
        relative_path=request.relative_path,
        allowed_suffixes=None,
        max_input_bytes=max_input_bytes,
        expected_sha256=request.expected_sha256,
        expected_byte_size=request.expected_byte_size,
    )
    try:
        opened.verify_current()
    finally:
        opened.close()


def create_qwen_worker_app(
    *,
    runtime: WorkerRuntime,
    input_root: Path,
    api_key: str,
    max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES,
    max_concurrency: int = 1,
    max_request_bytes: int = MAX_REQUEST_BYTES,
) -> FastAPI:
    if not _TOKEN_RE.fullmatch(api_key):
        raise ValueError("Qwen worker API key must be 32-256 URL-safe characters")
    if max_input_bytes <= 0 or max_request_bytes <= 0:
        raise ValueError("Qwen worker size limits must be positive")
    if not 1 <= max_concurrency <= 8:
        raise ValueError("Qwen worker concurrency must be between 1 and 8")
    resolved_root = Path(input_root).resolve(strict=True)
    if not resolved_root.is_dir():
        raise ValueError("Qwen worker input root must be a directory")
    input_root_sha256 = _input_root_identity(resolved_root)
    if not runtime.model_identity.strip():
        raise ValueError("Qwen worker model identity must not be empty")

    app = FastAPI(
        title="VideoScope Qwen Worker",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    capacity = BoundedSemaphore(max_concurrency)

    def input_root_is_current() -> bool:
        try:
            return _input_root_identity(resolved_root) == input_root_sha256
        except RuntimeError:
            return False

    def validate_execution_identity(request: object) -> None:
        if (
            getattr(request, "model_identity", None) != runtime.model_identity
            or getattr(request, "source_bundle_sha256", None)
            != QWEN_SOURCE_BUNDLE_SHA256
            or getattr(request, "prompt_protocol_sha256", None)
            != QWEN_PROMPT_PROTOCOL_SHA256
            or getattr(request, "input_root_sha256", None) != input_root_sha256
            or not input_root_is_current()
        ):
            raise HTTPException(409, "Qwen worker execution identity does not match")

    @app.get("/v1/health", response_model=QwenHealthResponse)
    def health() -> QwenHealthResponse:
        return QwenHealthResponse(
            schema_version=QWEN_WORKER_SCHEMA_VERSION,
            status="ok" if runtime.available and input_root_is_current() else "unavailable",
            model_identity=runtime.model_identity,
            runtime_identity=QWEN_INFERENCE_RUNTIME_IDENTITY,
            source_bundle_sha256=QWEN_SOURCE_BUNDLE_SHA256,
            prompt_protocol_sha256=QWEN_PROMPT_PROTOCOL_SHA256,
            input_root_sha256=input_root_sha256,
            loaded=runtime.loaded,
            input_kinds=["video", "storyboard"],
            max_input_bytes=max_input_bytes,
            max_query_chars=MAX_QUERY_CHARS,
            max_tokens=MAX_TOKENS,
            max_concurrency=max_concurrency,
        )

    @app.post("/v1/judge", response_model=QwenJudgeResponse)
    async def judge(request: QwenJudgeRequest) -> QwenJudgeResponse:
        validate_execution_identity(request)
        if not capacity.acquire(blocking=False):
            raise HTTPException(429, "Qwen worker is busy")
        materialized: _MaterializedWorkerInput | None = None
        cleanup_ok = True
        try:
            try:
                materialized = _materialize_worker_input(
                    input_root=resolved_root,
                    request=request,
                    max_input_bytes=max_input_bytes,
                )
            except OverflowError as error:
                raise HTTPException(413, "Qwen worker input is too large") from error
            except _WorkerInputIdentityError as error:
                raise HTTPException(
                    409,
                    "Qwen worker input identity mismatch",
                ) from error
            except _WorkerInputValidationError as error:
                raise HTTPException(400, "Invalid Qwen worker input") from error
            try:
                judgement = await run_in_threadpool(
                    runtime.judge,
                    request,
                    materialized.path,
                )
                payload = QwenJudgementPayload.model_validate(asdict(judgement))
            except Exception as error:
                logger.exception("Qwen worker inference failed")
                raise HTTPException(503, "Qwen worker inference failed") from error
            try:
                materialized.verify_current()
            except (_WorkerInputIdentityError, OverflowError) as error:
                raise HTTPException(409, "Qwen worker input changed during inference") from error
        finally:
            if materialized is not None:
                cleanup_ok = materialized.close()
            capacity.release()
        if not cleanup_ok:
            raise HTTPException(503, "Qwen worker input cleanup failed")
        return QwenJudgeResponse(
            schema_version=QWEN_WORKER_SCHEMA_VERSION,
            request_id=request.request_id,
            model_identity=runtime.model_identity,
            runtime_identity=QWEN_INFERENCE_RUNTIME_IDENTITY,
            source_bundle_sha256=QWEN_SOURCE_BUNDLE_SHA256,
            prompt_protocol_sha256=QWEN_PROMPT_PROTOCOL_SHA256,
            input_root_sha256=input_root_sha256,
            source_sha256=request.expected_sha256,
            byte_size=request.expected_byte_size,
            judgement=payload,
        )

    @app.post("/v1/probe", response_model=QwenProbeResponse)
    def probe(request: QwenProbeRequest) -> QwenProbeResponse:
        validate_execution_identity(request)
        try:
            _probe_worker_input(
                input_root=resolved_root,
                request=request,
                max_input_bytes=max_input_bytes,
            )
        except OverflowError as error:
            raise HTTPException(413, "Qwen worker probe input is too large") from error
        except (_WorkerInputValidationError, _WorkerInputIdentityError) as error:
            raise HTTPException(409, "Qwen worker probe input identity mismatch") from error
        return QwenProbeResponse(
            schema_version=QWEN_WORKER_SCHEMA_VERSION,
            request_id=request.request_id,
            model_identity=runtime.model_identity,
            runtime_identity=QWEN_INFERENCE_RUNTIME_IDENTITY,
            source_bundle_sha256=QWEN_SOURCE_BUNDLE_SHA256,
            prompt_protocol_sha256=QWEN_PROMPT_PROTOCOL_SHA256,
            input_root_sha256=input_root_sha256,
            source_sha256=request.expected_sha256,
            byte_size=request.expected_byte_size,
            verified=True,
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
        raise ValueError("Qwen worker endpoint must be a plain loopback HTTP origin")
    hostname = parsed.hostname.casefold()
    try:
        address = ip_address(hostname)
    except ValueError as error:
        raise ValueError("Qwen worker endpoint must use 127.0.0.1") from error
    if str(address) != "127.0.0.1":
        raise ValueError("Qwen worker endpoint must use 127.0.0.1")
    try:
        parsed.port
    except ValueError as error:
        raise ValueError("Qwen worker endpoint contains an invalid port") from error
    return endpoint.strip().rstrip("/")


class QwenWorkerClient:
    """Authenticated adapter from CandidateReranker to the isolated local worker."""

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        input_root: Path,
        expected_model_identity: str,
        expected_source_bundle_sha256: str = QWEN_SOURCE_BUNDLE_SHA256,
        expected_prompt_protocol_sha256: str = QWEN_PROMPT_PROTOCOL_SHA256,
        timeout: float = 180.0,
        client: HTTPClient | None = None,
    ) -> None:
        self.endpoint = _validate_endpoint(endpoint)
        if not _TOKEN_RE.fullmatch(api_key):
            raise ValueError("Qwen worker API key must be 32-256 URL-safe characters")
        if not expected_model_identity.strip():
            raise ValueError("expected Qwen model identity must not be empty")
        if (
            _SHA256_RE.fullmatch(expected_source_bundle_sha256) is None
            or _SHA256_RE.fullmatch(expected_prompt_protocol_sha256) is None
        ):
            raise ValueError("expected Qwen executable identity is invalid")
        if not 0 < timeout <= 600:
            raise ValueError("Qwen worker timeout must be between 0 and 600 seconds")
        self._api_key = api_key
        try:
            self.input_root = Path(input_root).resolve(strict=True)
            self.input_root_sha256 = _input_root_identity(self.input_root)
        except (OSError, RuntimeError):
            raise ValueError("Qwen worker input root is unavailable or unsafe") from None
        if not self.input_root.is_dir():
            raise ValueError("Qwen worker input root must be a directory")
        self.expected_model_identity = expected_model_identity
        self.expected_source_bundle_sha256 = expected_source_bundle_sha256
        self.expected_prompt_protocol_sha256 = expected_prompt_protocol_sha256
        self.timeout = timeout
        self.client = client
        self._client_lock = Lock()
        self._status_lock = Lock()
        self._cached_status: tuple[float, QwenInferenceStatus] | None = None

    @property
    def identity(self) -> dict[str, object]:
        return {
            "mode": "isolated-worker",
            "contract": QWEN_WORKER_SCHEMA_VERSION,
            "endpoint": self.endpoint,
            "model": self.expected_model_identity,
            "runtime_identity": QWEN_INFERENCE_RUNTIME_IDENTITY,
            "source_bundle_sha256": self.expected_source_bundle_sha256,
            "prompt_protocol_sha256": self.expected_prompt_protocol_sha256,
            "input_root_sha256": self.input_root_sha256,
        }

    def _input_root_is_current(self) -> bool:
        try:
            return _input_root_identity(self.input_root) == self.input_root_sha256
        except RuntimeError:
            return False

    def _response_identity_matches(self, payload: object) -> bool:
        return (
            getattr(payload, "model_identity", None) == self.expected_model_identity
            and getattr(payload, "runtime_identity", None)
            == QWEN_INFERENCE_RUNTIME_IDENTITY
            and getattr(payload, "source_bundle_sha256", None)
            == self.expected_source_bundle_sha256
            and getattr(payload, "prompt_protocol_sha256", None)
            == self.expected_prompt_protocol_sha256
            and getattr(payload, "input_root_sha256", None)
            == self.input_root_sha256
            and self._input_root_is_current()
        )

    def _http_client(self) -> HTTPClient:
        if self.client is None:
            with self._client_lock:
                if self.client is None:
                    self.client = _BoundedHTTPClient(
                        max_response_bytes=MAX_RESPONSE_BYTES
                    )
        return self.client

    def _headers(self, *, json_request: bool = False) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        if json_request:
            headers["Content-Type"] = "application/json"
        return headers

    def status(self) -> QwenInferenceStatus:
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
            capability = QwenHealthResponse.model_validate(response.json())
        except Exception:
            status = QwenInferenceStatus(False, "Qwen worker is unreachable")
            with self._status_lock:
                self._cached_status = (monotonic() + 2.0, status)
            return status
        if capability.status != "ok":
            status = QwenInferenceStatus(False, "Qwen worker model is unavailable")
        elif not self._response_identity_matches(capability):
            status = QwenInferenceStatus(False, "Qwen worker execution identity does not match")
        else:
            status = QwenInferenceStatus(True, "Qwen worker is ready")
        with self._status_lock:
            self._cached_status = (monotonic() + 5.0, status)
        return status

    def _relative_source(self, source: Path) -> str:
        lexical = Path(os.path.abspath(source))
        try:
            relative = lexical.relative_to(self.input_root)
        except ValueError as error:
            raise RuntimeError("Qwen worker input is outside configured input root") from error
        return PurePosixPath(*relative.parts).as_posix()

    def _judge(
        self,
        *,
        source: Path,
        input_kind: Literal["video", "storyboard"],
        prompt_kind: Literal["basketball_facts", "generic_query"],
        query: str | None,
        fps: float | None,
        max_tokens: int,
        expected_sha256: str | None,
        expected_byte_size: int | None,
    ) -> QwenVideoJudgement:
        relative_path, source_sha256, source_byte_size = self._source_contract(
            source,
            input_kind=input_kind,
            expected_sha256=expected_sha256,
            expected_byte_size=expected_byte_size,
        )
        request_id = uuid4().hex
        request = QwenJudgeRequest(
            schema_version=QWEN_WORKER_SCHEMA_VERSION,
            request_id=request_id,
            model_identity=self.expected_model_identity,
            runtime_identity=QWEN_INFERENCE_RUNTIME_IDENTITY,
            source_bundle_sha256=self.expected_source_bundle_sha256,
            prompt_protocol_sha256=self.expected_prompt_protocol_sha256,
            input_root_sha256=self.input_root_sha256,
            input_kind=input_kind,
            relative_path=relative_path,
            expected_sha256=source_sha256,
            expected_byte_size=source_byte_size,
            prompt_kind=prompt_kind,
            query=query,
            fps=fps,
            max_tokens=max_tokens,
        )
        try:
            response = self._http_client().post(
                f"{self.endpoint}/v1/judge",
                json=request.model_dump(mode="json"),
                headers=self._headers(json_request=True),
                timeout=self.timeout,
            )
            response.raise_for_status()
            raw_payload = response.json()
        except Exception:
            raise RuntimeError("Qwen worker request failed") from None
        try:
            payload = QwenJudgeResponse.model_validate(raw_payload)
            if (
                payload.request_id != request_id
                or payload.source_sha256 != source_sha256
                or payload.byte_size != source_byte_size
                or not self._response_identity_matches(payload)
            ):
                raise ValueError("Qwen worker response identity mismatch")
            return QwenVideoJudgement(**payload.judgement.model_dump())
        except (TypeError, ValueError, ValidationError):
            raise RuntimeError("Qwen worker response contract violation") from None

    def _source_contract(
        self,
        source: Path,
        *,
        input_kind: Literal["video", "storyboard"] | None,
        expected_sha256: str | None = None,
        expected_byte_size: int | None = None,
    ) -> tuple[str, str, int]:
        relative_path = self._relative_source(source)
        if (expected_sha256 is None) != (expected_byte_size is None):
            raise RuntimeError("Qwen worker source identity is incomplete")
        if expected_sha256 is not None and (
            _SHA256_RE.fullmatch(expected_sha256) is None
            or type(expected_byte_size) is not int
            or expected_byte_size <= 0
            or expected_byte_size > DEFAULT_MAX_INPUT_BYTES
        ):
            raise RuntimeError("Qwen worker source identity is invalid")
        allowed = (
            _VIDEO_SUFFIXES
            if input_kind == "video"
            else _STORYBOARD_SUFFIXES
            if input_kind == "storyboard"
            else None
        )
        try:
            opened = _open_worker_input(
                input_root=self.input_root,
                relative_path=relative_path,
                allowed_suffixes=allowed,
                max_input_bytes=DEFAULT_MAX_INPUT_BYTES,
                expected_sha256=expected_sha256,
                expected_byte_size=expected_byte_size,
            )
        except (OSError, ValueError, OverflowError):
            raise RuntimeError("Qwen worker source is unavailable or changed") from None
        try:
            opened.verify_current()
            result = (
                relative_path,
                opened.expected_sha256,
                opened.expected_byte_size,
            )
        finally:
            try:
                opened.close()
            except OSError:
                raise RuntimeError("Qwen worker source cleanup failed") from None
        return result

    def probe_source(self, source: Path) -> bool:
        try:
            relative_path, expected_sha256, expected_byte_size = self._source_contract(
                source,
                input_kind=None,
            )
        except RuntimeError:
            raise RuntimeError("Qwen worker probe source is unavailable") from None
        request_id = uuid4().hex
        request = QwenProbeRequest(
            schema_version=QWEN_WORKER_SCHEMA_VERSION,
            request_id=request_id,
            model_identity=self.expected_model_identity,
            runtime_identity=QWEN_INFERENCE_RUNTIME_IDENTITY,
            source_bundle_sha256=self.expected_source_bundle_sha256,
            prompt_protocol_sha256=self.expected_prompt_protocol_sha256,
            input_root_sha256=self.input_root_sha256,
            relative_path=relative_path,
            expected_sha256=expected_sha256,
            expected_byte_size=expected_byte_size,
        )
        try:
            response = self._http_client().post(
                f"{self.endpoint}/v1/probe",
                json=request.model_dump(mode="json"),
                headers=self._headers(json_request=True),
                timeout=min(self.timeout, 10.0),
            )
            response.raise_for_status()
            payload = QwenProbeResponse.model_validate(response.json())
            if (
                payload.request_id != request_id
                or payload.verified is not True
                or payload.source_sha256 != expected_sha256
                or payload.byte_size != expected_byte_size
                or not self._response_identity_matches(payload)
            ):
                raise ValueError("Qwen worker probe response identity mismatch")
        except Exception:
            raise RuntimeError("Qwen worker probe failed") from None
        return True

    def judge_video(
        self,
        source: Path,
        *,
        fps: float,
        max_tokens: int,
        expected_sha256: str | None = None,
        expected_byte_size: int | None = None,
    ) -> QwenVideoJudgement:
        return self._judge(
            source=source,
            input_kind="video",
            prompt_kind="basketball_facts",
            query=None,
            fps=fps,
            max_tokens=max_tokens,
            expected_sha256=expected_sha256,
            expected_byte_size=expected_byte_size,
        )

    def judge_video_query(
        self,
        source: Path,
        query: str,
        *,
        fps: float,
        max_tokens: int,
        expected_sha256: str | None = None,
        expected_byte_size: int | None = None,
    ) -> QwenVideoJudgement:
        return self._judge(
            source=source,
            input_kind="video",
            prompt_kind="generic_query",
            query=query,
            fps=fps,
            max_tokens=max_tokens,
            expected_sha256=expected_sha256,
            expected_byte_size=expected_byte_size,
        )

    def judge_storyboard(
        self,
        source: Path,
        query: str,
        *,
        max_tokens: int,
        expected_sha256: str | None = None,
        expected_byte_size: int | None = None,
    ) -> QwenVideoJudgement:
        return self._judge(
            source=source,
            input_kind="storyboard",
            prompt_kind="generic_query",
            query=query,
            fps=None,
            max_tokens=max_tokens,
            expected_sha256=expected_sha256,
            expected_byte_size=expected_byte_size,
        )


class MLXQwenWorkerRuntime:
    """Lazy MLX runtime. Importing the worker contract never imports MLX-VLM."""

    def __init__(self, model_name: str, revision: str | None) -> None:
        if not model_name.strip():
            raise ValueError("Qwen worker model name must not be empty")
        self.model_name = model_name.strip()
        self.model_revision = revision
        self._model: Any | None = None
        self._processor: Any | None = None
        self._model_reference: str | None = None
        self._load_lock = Lock()
        self._inference_lock = Lock()
        self._healthy = True

    @property
    def model_identity(self) -> str:
        return model_identity(self.model_name, self.model_revision)

    @property
    def loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    def _resolve_local_reference(self) -> str:
        if self._model_reference is not None:
            return self._model_reference
        model_path = Path(self.model_name).expanduser()
        if self.model_revision is None and model_path.exists():
            reference = model_path.resolve(strict=True)
        else:
            from huggingface_hub import snapshot_download

            reference = Path(
                snapshot_download(
                    self.model_name,
                    revision=self.model_revision,
                    local_files_only=True,
                )
            )
        config = reference / "config.json"
        weights = next(reference.glob("*.safetensors"), None)
        if not config.is_file() or weights is None or not weights.is_file():
            raise RuntimeError("local Qwen snapshot is incomplete")
        self._model_reference = str(reference)
        return self._model_reference

    @property
    def available(self) -> bool:
        if not self._healthy:
            return False
        if self.loaded:
            return True
        if importlib.util.find_spec("mlx_vlm") is None:
            return False
        try:
            self._resolve_local_reference()
        except Exception:
            return False
        return True

    def _load(self):  # type: ignore[no-untyped-def]
        if not self._healthy:
            raise RuntimeError("Qwen worker runtime is unavailable")
        if self.loaded:
            return self._model, self._processor
        with self._load_lock:
            if self.loaded:
                return self._model, self._processor
            from mlx_vlm import load

            reference = self._resolve_local_reference()
            self._model, self._processor = load(reference)
            return self._model, self._processor

    @staticmethod
    def _mlx_runtime_module() -> Any:
        mlx = importlib.import_module("mlx.core")
        if (
            not callable(getattr(mlx, "synchronize", None))
            or not callable(getattr(mlx, "clear_cache", None))
        ):
            raise RuntimeError("Qwen MLX cache controls do not match the lock")
        return mlx

    @staticmethod
    def _request_state_owner(model: Any) -> Any:
        language_model = getattr(model, "language_model", None)
        if (
            language_model is None
            or not hasattr(language_model, "_position_ids")
            or not hasattr(language_model, "_rope_deltas")
        ):
            raise RuntimeError("Qwen request state controls do not match the lock")
        return language_model

    def _release_mlx_after_request(
        self,
        *,
        language_model: Any,
        mlx: Any,
        primary_error: BaseException | None,
    ) -> None:
        try:
            mlx.synchronize()
            language_model._position_ids = None
            language_model._rope_deltas = None
            gc.collect()
            mlx.clear_cache()
        except BaseException as cleanup_error:
            self._healthy = False
            if primary_error is not None:
                logger.error(
                    "Qwen inference also failed before MLX cleanup",
                    exc_info=(
                        type(primary_error),
                        primary_error,
                        primary_error.__traceback__,
                    ),
                )
            raise RuntimeError(
                "Qwen worker could not release MLX request memory"
            ) from cleanup_error

    def _generate(
        self,
        *,
        model: Any,
        processor: Any,
        prompt: object,
        media: dict[str, object],
        request: QwenJudgeRequest,
    ) -> str:
        from mlx_vlm import generate

        mlx = self._mlx_runtime_module()
        language_model = self._request_state_owner(model)
        result: object | None = None
        text: str | None = None
        with self._inference_lock:
            if not self._healthy:
                raise RuntimeError("Qwen worker runtime is unavailable")
            try:
                result = generate(
                    model,
                    processor,
                    prompt,
                    **media,
                    max_tokens=request.max_tokens,
                    temperature=0.0,
                    enable_thinking=False,
                    verbose=False,
                )
                text = str(result.text)
            except BaseException as primary_error:
                result = None
                traceback.clear_frames(primary_error.__traceback__)
                self._release_mlx_after_request(
                    language_model=language_model,
                    mlx=mlx,
                    primary_error=primary_error,
                )
                raise
            else:
                result = None
                self._release_mlx_after_request(
                    language_model=language_model,
                    mlx=mlx,
                    primary_error=None,
                )
        if text is None:
            raise RuntimeError("Qwen worker returned no text")
        return text

    def _generate_video(self, source: Path, request: QwenJudgeRequest) -> str:
        from mlx_vlm import apply_chat_template

        if request.prompt_kind == "basketball_facts":
            prompt_text = QwenVideoReranker._fact_prompt()
        else:
            if request.query is None:
                raise ValueError("generic-query video requires a query")
            prompt_text = QwenVideoReranker._generic_prompt(request.query)
        model, processor = self._load()
        prompt = apply_chat_template(
            processor,
            model.config,
            [prompt_text],
            video=str(source),
            fps=request.fps,
            enable_thinking=False,
        )
        return self._generate(
            model=model,
            processor=processor,
            prompt=prompt,
            media={"video": [str(source)], "fps": request.fps},
            request=request,
        )

    def _generate_storyboard(self, source: Path, request: QwenJudgeRequest) -> str:
        from mlx_vlm import apply_chat_template

        if request.query is None:
            raise ValueError("storyboard query is required")
        model, processor = self._load()
        prompt = apply_chat_template(
            processor,
            model.config,
            [QwenVideoReranker._generic_prompt(request.query)],
            num_images=1,
            enable_thinking=False,
        )
        return self._generate(
            model=model,
            processor=processor,
            prompt=prompt,
            media={"image": [str(source)]},
            request=request,
        )

    def judge(self, request: QwenJudgeRequest, source: Path) -> QwenVideoJudgement:
        text = (
            self._generate_video(source, request)
            if request.input_kind == "video"
            else self._generate_storyboard(source, request)
        )
        return parse_qwen_judgement(text)


class QwenWorkerSettings(BaseSettings):
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
            "QWEN_VIDEO_API_KEY",
            "VIDEOSCOPE_QWEN_VIDEO_API_KEY",
        ),
    )
    model_name: str = Field(
        min_length=1,
        validation_alias=AliasChoices(
            "QWEN_VIDEO_MODEL",
            "VIDEOSCOPE_QWEN_VIDEO_MODEL",
        ),
    )
    data_dir: Path = Field(
        default=Path("data"),
        validation_alias="VIDEOSCOPE_DATA_DIR",
    )
    worker_input_root: Path | None = Field(
        default=None,
        validation_alias="VIDEOSCOPE_QWEN_WORKER_INPUT_ROOT",
    )
    port: int = Field(
        default=8781,
        ge=1,
        le=65_535,
        validation_alias=AliasChoices(
            "QWEN_WORKER_PORT",
            "VIDEOSCOPE_QWEN_WORKER_PORT",
        ),
    )
    max_concurrency: int = Field(
        default=1,
        ge=1,
        le=8,
        validation_alias=AliasChoices(
            "QWEN_WORKER_MAX_CONCURRENCY",
            "VIDEOSCOPE_QWEN_WORKER_MAX_CONCURRENCY",
        ),
    )

    @property
    def input_root(self) -> Path:
        return (
            self.worker_input_root
            if self.worker_input_root is not None
            else self.data_dir / "tmp"
        )


def main() -> None:
    try:
        settings = QwenWorkerSettings()
    except ValidationError:
        raise RuntimeError("Qwen worker configuration is invalid") from None
    revision = model_revision(settings.model_name)
    if revision is None:
        raise RuntimeError("Qwen worker model must have a pinned revision")
    attest_qwen_worker_startup(settings.model_name, revision)
    try:
        settings.input_root.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise RuntimeError("Qwen worker input root cannot be created") from None
    if not settings.input_root.is_dir():
        raise RuntimeError("Qwen worker input root must be a directory")
    runtime = MLXQwenWorkerRuntime(
        settings.model_name,
        revision,
    )
    app = create_qwen_worker_app(
        runtime=runtime,
        input_root=settings.input_root,
        api_key=settings.api_key,
        max_concurrency=settings.max_concurrency,
    )
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=settings.port, workers=1)


if __name__ == "__main__":
    main()
