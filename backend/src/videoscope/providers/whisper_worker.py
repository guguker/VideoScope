from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
import errno
import fcntl
import hashlib
import importlib
import importlib.metadata
from ipaddress import ip_address
import json
import logging
import math
import os
import platform
from pathlib import Path, PurePosixPath
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
from threading import BoundedSemaphore, Lock
from time import monotonic
from typing import Any, Callable, Literal, Protocol, Self
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
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from videoscope.model_manifest import MODEL_REVISIONS, WHISPER_MODEL, model_identity
from videoscope.providers.types import TimedText
from videoscope.providers.whisper import (
    MAX_WHISPER_EFFECTIVE_PROMPT_CHARS,
    WhisperInferenceStatus,
    WhisperPromptSnapshot,
)


logger = logging.getLogger(__name__)

WHISPER_WORKER_SCHEMA_VERSION = "whisper-worker-v1"
WHISPER_WORKER_PYTHON_VERSION = (3, 12, 13)
WHISPER_WORKER_SYSTEM = "Darwin"
WHISPER_WORKER_MACHINE = "arm64"
WHISPER_INFERENCE_RUNTIME_IDENTITY = (
    "mlx-whisper-worker-v2:mlx-whisper==0.4.3:python==3.12.13:"
    "platform=darwin-arm64-macos14plus:model-artifacts=v1"
)
# Updated together with workers/whisper/requirements.lock. The lock digest is
# part of every health/request/response identity and is never inferred at run time.
WHISPER_WORKER_LOCK_SHA256 = (
    "23000652063f4e3e9fbbb66cfa80f2ec252b3bfb7df517f89ede1283c7af4212"
)
WHISPER_DEPENDENCY_IDENTITY = (
    f"whisper-worker-lock-v1:sha256:{WHISPER_WORKER_LOCK_SHA256}"
)

WHISPER_REVIEWED_MODEL_ARTIFACTS: dict[
    tuple[str, str], tuple[tuple[str, int, str], ...]
] = {
    (WHISPER_MODEL, MODEL_REVISIONS[WHISPER_MODEL]): (
        (
            "config.json",
            268,
            "b34fc29e4e11e0a25e812775dd67f4dd16fc2c8eb43d28ae25ff7d660ecb6379",
        ),
        (
            "weights.safetensors",
            1_613_977_612,
            "951ed3fc1203e6a62467abb2144a96ce7eafca8fa77e3704fdb8635ff3e7f8a6",
        ),
    ),
}

WHISPER_EXACT_DEPENDENCIES: tuple[tuple[str, str], ...] = (
    ("annotated-doc", "0.0.5"),
    ("annotated-types", "0.8.0"),
    ("anyio", "4.14.2"),
    ("certifi", "2026.7.22"),
    ("charset-normalizer", "3.5.1"),
    ("click", "8.4.2"),
    ("fastapi", "0.141.1"),
    ("filelock", "3.32.3"),
    ("fsspec", "2026.7.0"),
    ("h11", "0.16.0"),
    ("hf-xet", "1.6.0"),
    ("httpcore", "1.0.9"),
    ("httpx", "0.28.1"),
    ("huggingface-hub", "1.27.0"),
    ("idna", "3.19"),
    ("jinja2", "3.1.6"),
    ("llvmlite", "0.49.0"),
    ("markupsafe", "3.0.3"),
    ("mlx", "0.32.0"),
    ("mlx-metal", "0.32.0"),
    ("mlx-whisper", "0.4.3"),
    ("more-itertools", "11.1.0"),
    ("mpmath", "1.3.0"),
    ("networkx", "3.6.1"),
    ("numba", "0.67.0"),
    ("numpy", "2.5.2"),
    ("packaging", "26.3"),
    ("pydantic", "2.13.4"),
    ("pydantic-core", "2.46.4"),
    ("pydantic-settings", "2.15.0"),
    ("python-dotenv", "1.2.3"),
    ("pyyaml", "6.0.3"),
    ("regex", "2026.7.19"),
    ("requests", "2.34.2"),
    ("scipy", "1.18.0"),
    ("setuptools", "84.0.0"),
    ("starlette", "1.6.0"),
    ("sympy", "1.14.0"),
    ("tiktoken", "0.14.0"),
    ("torch", "2.13.0"),
    ("tqdm", "4.70.0"),
    ("typing-extensions", "4.16.0"),
    ("typing-inspection", "0.4.4"),
    ("urllib3", "2.7.0"),
    ("uvicorn", "0.52.1"),
)

MAX_REQUEST_BYTES = 64 * 1024
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_EFFECTIVE_PROMPT_CHARS = MAX_WHISPER_EFFECTIVE_PROMPT_CHARS
DEFAULT_MAX_INPUT_BYTES = 8 * 1024 * 1024 * 1024
DEFAULT_MAX_DURATION_SECONDS = 6 * 60 * 60
MAX_OUTPUT_SEGMENTS = 8192
MAX_WORDS_PER_SEGMENT = 512
MAX_OUTPUT_WORDS = 131_072
MAX_SEGMENT_TEXT_CHARS = 4096
MAX_WORD_TEXT_CHARS = 256
MAX_OUTPUT_TEXT_CHARS = 4 * 1024 * 1024
WORK_ROOT_FREE_RESERVE_BYTES = 256 * 1024 * 1024

_WORK_ROOT_LOCK_FILE = ".videoscope-whisper-worker.lock"
_WORKER_OWNER_FILE = ".videoscope-whisper-owner"
_WORKER_LEASE_FILE = ".videoscope-whisper-lease"
_WORKER_OWNER_PREFIX = "videoscope-whisper-worker-owned-v1:"
_WORKER_DIRECTORY_RE = re.compile(r"^videoscope-whisper-worker-[0-9a-f]{32}$")
_WORKER_SNAPSHOT_RE = re.compile(
    r"^input-[A-Za-z0-9_-]{1,64}\.(?:avi|m4v|mkv|mov|mp4|webm)$"
)
_MAX_WORK_ROOT_ENTRIES = 1024
_MAX_WORKER_DIRECTORY_ENTRIES = 1024

_REQUEST_ID_RE = r"^[0-9a-f]{32}$"
_SHA256_RE = r"^[0-9a-f]{64}$"
_PINNED_MODEL_RE = re.compile(r"^.{1,240}@[0-9a-f]{40,64}$")
_TOKEN_RE = re.compile(r"^[A-Za-z0-9._~-]{32,256}$")
_LANGUAGE_RE = re.compile(r"^(?:auto|[a-z]{2,3}(?:-[a-z0-9]{2,8})?)$")
_HOST_RE = re.compile(r"^(?:127\.0\.0\.1|localhost|\[::1\])(?::[0-9]{1,5})?$")
_VIDEO_SUFFIXES = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"})
_GLOSSARY_STATES = frozenset(
    {
        "not_configured",
        "missing",
        "unsafe",
        "oversized",
        "changed",
        "ready",
        "invalid",
        "unreadable",
    }
)


class _ContractModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


class WhisperTranscribeRequest(_ContractModel):
    schema_version: Literal[WHISPER_WORKER_SCHEMA_VERSION]
    request_id: str = Field(pattern=_REQUEST_ID_RE)
    model_identity: str = Field(min_length=42, max_length=320)
    runtime_identity: str = Field(min_length=1, max_length=160)
    dependency_identity: str = Field(min_length=1, max_length=160)
    relative_path: str = Field(min_length=1, max_length=240)
    source_size: int = Field(gt=0, le=DEFAULT_MAX_INPUT_BYTES)
    source_sha256: str = Field(pattern=_SHA256_RE)
    language: str = Field(min_length=2, max_length=32)
    effective_prompt: str | None = Field(
        default=None,
        max_length=MAX_EFFECTIVE_PROMPT_CHARS,
    )
    effective_prompt_sha256: str = Field(pattern=_SHA256_RE)
    glossary_state: str = Field(min_length=1, max_length=32)

    @field_validator("model_identity")
    @classmethod
    def validate_model_identity(cls, value: str) -> str:
        if _PINNED_MODEL_RE.fullmatch(value) is None:
            raise ValueError("model identity must contain an immutable revision")
        return value

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
        return value

    @field_validator("language")
    @classmethod
    def validate_language(cls, value: str) -> str:
        if _LANGUAGE_RE.fullmatch(value) is None:
            raise ValueError("language is not supported by the worker contract")
        return value

    @field_validator("glossary_state")
    @classmethod
    def validate_glossary_state(cls, value: str) -> str:
        if value not in _GLOSSARY_STATES:
            raise ValueError("unknown glossary state")
        return value

    @model_validator(mode="after")
    def validate_prompt_identity(self) -> Self:
        digest = hashlib.sha256((self.effective_prompt or "").encode("utf-8")).hexdigest()
        if digest != self.effective_prompt_sha256:
            raise ValueError("effective prompt identity does not match")
        return self


class WhisperWordPayload(_ContractModel):
    text: str = Field(min_length=1, max_length=MAX_WORD_TEXT_CHARS)
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if self.end <= self.start:
            raise ValueError("word end must be greater than start")
        return self


class WhisperSegmentPayload(_ContractModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    text: str = Field(min_length=1, max_length=MAX_SEGMENT_TEXT_CHARS)
    confidence: float = Field(ge=0, le=1)
    words: list[WhisperWordPayload] = Field(
        default_factory=list,
        max_length=MAX_WORDS_PER_SEGMENT,
    )

    @model_validator(mode="after")
    def validate_timeline(self) -> Self:
        if self.end <= self.start:
            raise ValueError("segment end must be greater than start")
        previous_end: float | None = None
        for word in self.words:
            if (
                word.start < self.start
                or word.end > self.end
                or (previous_end is not None and word.start < previous_end)
            ):
                raise ValueError("word timestamps must be ordered inside the segment")
            previous_end = word.end
        return self


class WhisperTranscribeResponse(_ContractModel):
    schema_version: Literal[WHISPER_WORKER_SCHEMA_VERSION]
    request_id: str = Field(pattern=_REQUEST_ID_RE)
    model_identity: str = Field(min_length=42, max_length=320)
    runtime_identity: str = Field(min_length=1, max_length=160)
    dependency_identity: str = Field(min_length=1, max_length=160)
    source_size: int = Field(gt=0, le=DEFAULT_MAX_INPUT_BYTES)
    source_sha256: str = Field(pattern=_SHA256_RE)
    effective_prompt_sha256: str = Field(pattern=_SHA256_RE)
    glossary_state: str = Field(min_length=1, max_length=32)
    duration_seconds: float = Field(gt=0, le=DEFAULT_MAX_DURATION_SECONDS)
    detected_language: str = Field(min_length=2, max_length=32)
    segments: list[WhisperSegmentPayload] = Field(max_length=MAX_OUTPUT_SEGMENTS)

    @field_validator("glossary_state")
    @classmethod
    def validate_glossary_state(cls, value: str) -> str:
        if value not in _GLOSSARY_STATES:
            raise ValueError("unknown glossary state")
        return value

    @field_validator("detected_language")
    @classmethod
    def validate_detected_language(cls, value: str) -> str:
        if _LANGUAGE_RE.fullmatch(value) is None or value == "auto":
            raise ValueError("detected language is invalid")
        return value

    @model_validator(mode="after")
    def validate_output_bounds(self) -> Self:
        previous_end: float | None = None
        word_count = 0
        text_chars = 0
        for segment in self.segments:
            if (
                (previous_end is not None and segment.start < previous_end)
                or segment.end > self.duration_seconds + 1e-3
            ):
                raise ValueError("segment timestamps are outside the ordered media timeline")
            previous_end = segment.end
            word_count += len(segment.words)
            text_chars += len(segment.text) + sum(len(word.text) for word in segment.words)
        if word_count > MAX_OUTPUT_WORDS or text_chars > MAX_OUTPUT_TEXT_CHARS:
            raise ValueError("transcript output exceeds the aggregate contract")
        return self


class WhisperHealthResponse(_ContractModel):
    schema_version: Literal[WHISPER_WORKER_SCHEMA_VERSION]
    status: Literal["ok", "unavailable"]
    model_identity: str = Field(min_length=42, max_length=320)
    runtime_identity: str = Field(min_length=1, max_length=160)
    dependency_identity: str = Field(min_length=1, max_length=160)
    loaded: bool
    max_input_bytes: int = Field(gt=0, le=DEFAULT_MAX_INPUT_BYTES)
    max_duration_seconds: float = Field(gt=0, le=DEFAULT_MAX_DURATION_SECONDS)
    max_request_bytes: int = Field(gt=0, le=MAX_REQUEST_BYTES)
    max_response_bytes: Literal[MAX_RESPONSE_BYTES]
    max_concurrency: Literal[1]


@dataclass(frozen=True, slots=True)
class WhisperTranscript:
    language: str
    segments: tuple[TimedText, ...]


class WorkerRuntime(Protocol):
    @property
    def model_identity(self) -> str: ...

    @property
    def runtime_identity(self) -> str: ...

    @property
    def dependency_identity(self) -> str: ...

    @property
    def loaded(self) -> bool: ...

    @property
    def available(self) -> bool: ...

    def transcribe(
        self,
        request: WhisperTranscribeRequest,
        source: Path,
        duration_seconds: float,
    ) -> WhisperTranscript: ...


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


@dataclass(frozen=True, slots=True)
class _SourceFingerprint:
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True, slots=True)
class _DirectoryFingerprint:
    device: int
    inode: int


@dataclass(frozen=True, slots=True)
class _SourceContract:
    path: Path
    fingerprint: _SourceFingerprint
    size: int
    sha256: str


class _InputRejected(ValueError):
    pass


class _InputContractMismatch(ValueError):
    pass


class _InputTooLarge(ValueError):
    pass


class _WorkRootUnavailable(RuntimeError):
    pass


def _discard_snapshot(path: Path) -> None:
    """Best-effort per-request cleanup; lifespan cleanup is the final backstop."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.error("Whisper worker could not remove a private input snapshot")


class _BufferedJSONResponse:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self.payload


class _BoundedHTTPClient:
    """HTTP transport that ignores proxy env, redirects, and oversized responses."""

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

        kwargs: dict[str, object] = {"headers": headers, "timeout": timeout}
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
                            "Whisper worker returned invalid Content-Length"
                        ) from error
                    if declared_length < 0 or declared_length > self.max_response_bytes:
                        raise RuntimeError("Whisper worker response is too large")
                body = bytearray()
                for chunk in response.iter_bytes(chunk_size=8192):
                    body.extend(chunk)
                    if len(body) > self.max_response_bytes:
                        raise RuntimeError("Whisper worker response is too large")
        try:
            payload = json.loads(bytes(body))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeError("Whisper worker returned invalid JSON") from error
        return _BufferedJSONResponse(payload)

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
    """Authenticate exact local HTTP requests and bound JSON before parsing."""

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

    def _authorized(self, headers: list[tuple[bytes, bytes]]) -> bool:
        supplied = [
            value for name, value in headers if name.lower() == b"authorization"
        ]
        return len(supplied) == 1 and secrets.compare_digest(
            supplied[0], self.expected_authorization
        )

    @staticmethod
    def _valid_host(headers: list[tuple[bytes, bytes]]) -> bool:
        supplied = [value for name, value in headers if name.lower() == b"host"]
        if len(supplied) != 1:
            return False
        try:
            host = supplied[0].decode("ascii")
        except UnicodeError:
            return False
        if _HOST_RE.fullmatch(host) is None:
            return False
        try:
            port = int(host.rsplit(":", 1)[1]) if ":" in host and not host.endswith("]") else None
        except ValueError:
            return False
        return port is None or 1 <= port <= 65535

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        protected = scope["type"] == "http" and str(scope.get("path", "")).startswith(
            "/v1/"
        )
        if not protected:
            await self.app(scope, receive, send)
            return
        headers = list(scope.get("headers", []))
        if not self._is_loopback(scope):
            await self._error(scope, receive, send, 403, "Loopback access only")
            return
        if not self._valid_host(headers):
            await self._error(scope, receive, send, 400, "Invalid Host header")
            return
        if not self._authorized(headers):
            await self._error(scope, receive, send, 401, "Unauthorized")
            return
        if scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return

        lengths = [value for name, value in headers if name.lower() == b"content-length"]
        if len(lengths) > 1:
            await self._error(scope, receive, send, 400, "Invalid Content-Length")
            return
        if lengths:
            try:
                declared_length = int(lengths[0])
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
        size=metadata.st_size,
        modified_ns=metadata.st_mtime_ns,
        changed_ns=metadata.st_ctime_ns,
    )


def _directory_fingerprint(metadata: os.stat_result) -> _DirectoryFingerprint:
    return _DirectoryFingerprint(device=metadata.st_dev, inode=metadata.st_ino)


def _open_directory_without_following(path: Path) -> tuple[Path, int]:
    """Open an absolute directory through one all-components no-follow chain."""
    absolute = Path(os.path.abspath(path))
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(absolute.anchor, directory_flags)
    try:
        for part in absolute.parts[1:]:
            child = os.open(part, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise OSError(errno.ENOTDIR, "input root is not a directory")
    except BaseException:
        os.close(descriptor)
        raise
    return absolute, descriptor


class _InputRoot:
    """An immutable directory identity used as the anchor for relative input reads."""

    def __init__(
        self,
        path: Path,
        descriptor: int,
        fingerprint: _DirectoryFingerprint,
    ) -> None:
        self.path = path
        self._descriptor: int | None = descriptor
        self.fingerprint = fingerprint

    @classmethod
    def open(
        cls,
        path: Path,
        *,
        expected: _DirectoryFingerprint | None = None,
    ) -> _InputRoot:
        descriptor: int | None = None
        try:
            absolute, descriptor = _open_directory_without_following(path)
            metadata = os.fstat(descriptor)
        except OSError as error:
            if descriptor is not None:
                os.close(descriptor)
            raise _InputRejected("input root cannot be opened safely") from error
        fingerprint = _directory_fingerprint(metadata)
        if expected is not None and fingerprint != expected:
            os.close(descriptor)
            raise _InputRejected("input root identity changed")
        return cls(absolute, descriptor, fingerprint)

    @property
    def descriptor(self) -> int:
        if self._descriptor is None:
            raise RuntimeError("input root is closed")
        return self._descriptor

    def duplicate_descriptor(self) -> int:
        descriptor = self._descriptor
        if descriptor is None:
            raise _InputRejected("input root is closed")
        duplicate: int | None = None
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(before.st_mode)
                or _directory_fingerprint(before) != self.fingerprint
            ):
                raise _InputRejected("input root identity changed")
            duplicate = os.dup(descriptor)
            after = os.fstat(duplicate)
        except _InputRejected:
            raise
        except OSError as error:
            if duplicate is not None:
                os.close(duplicate)
            raise _InputRejected("input root cannot be opened safely") from error
        assert duplicate is not None
        if (
            not stat.S_ISDIR(after.st_mode)
            or _directory_fingerprint(after) != self.fingerprint
        ):
            os.close(duplicate)
            raise _InputRejected("input root identity changed")
        return duplicate

    def close(self) -> None:
        descriptor = self._descriptor
        if descriptor is None:
            return
        self._descriptor = None
        os.close(descriptor)

    def __del__(self) -> None:
        try:
            self.close()
        except OSError:
            pass


def _open_parent_without_following(path: Path) -> tuple[Path, int]:
    """Open an absolute lexical parent without traversing any symlink."""
    absolute = Path(os.path.abspath(path))
    if not absolute.name:
        raise OSError(errno.EINVAL, "source path has no file name")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(absolute.anchor, directory_flags)
    try:
        for part in absolute.parts[1:-1]:
            child = os.open(part, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
    except BaseException:
        os.close(descriptor)
        raise
    return absolute, descriptor


def _open_regular_source(
    source: Path,
    *,
    max_input_bytes: int,
) -> tuple[Path, int, int, os.stat_result]:
    """Open a bounded source through one all-components no-follow chain."""
    parent_descriptor: int | None = None
    source_descriptor: int | None = None
    try:
        absolute, parent_descriptor = _open_parent_without_following(source)
        lexical = os.stat(
            absolute.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if stat.S_ISLNK(lexical.st_mode) or not stat.S_ISREG(lexical.st_mode):
            raise _InputRejected("input is not a regular file")
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        source_descriptor = os.open(
            absolute.name,
            flags,
            dir_fd=parent_descriptor,
        )
        before = os.fstat(source_descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or _source_fingerprint(before) != _source_fingerprint(lexical)
        ):
            raise _InputRejected("input is not a stable regular file")
        if before.st_size <= 0:
            raise _InputRejected("input file is empty")
        if before.st_size > max_input_bytes:
            raise _InputTooLarge("input exceeds configured size")
        return absolute, parent_descriptor, source_descriptor, before
    except (_InputRejected, _InputTooLarge):
        if source_descriptor is not None:
            os.close(source_descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)
        raise
    except OSError as error:
        if source_descriptor is not None:
            os.close(source_descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)
        raise _InputRejected("input cannot be opened safely") from error


def _rooted_relative_path(relative_path: str) -> PurePosixPath:
    if (
        "\\" in relative_path
        or "\x00" in relative_path
        or re.fullmatch(r"[A-Za-z0-9._/-]+", relative_path) is None
    ):
        raise _InputRejected("input path is unsafe")
    relative = PurePosixPath(relative_path)
    if (
        relative.is_absolute()
        or relative_path != relative.as_posix()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.suffix.casefold() not in _VIDEO_SUFFIXES
    ):
        raise _InputRejected("input path is unsafe")
    return relative


def _open_rooted_regular_source(
    input_root: _InputRoot,
    relative_path: str,
    *,
    max_input_bytes: int,
) -> tuple[Path, int, int, os.stat_result]:
    """Open a bounded source relative to one retained input-root identity."""
    relative = _rooted_relative_path(relative_path)
    parent_descriptor: int | None = None
    source_descriptor: int | None = None
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        parent_descriptor = input_root.duplicate_descriptor()
        for part in relative.parts[:-1]:
            child = os.open(part, directory_flags, dir_fd=parent_descriptor)
            os.close(parent_descriptor)
            parent_descriptor = child
        lexical = os.stat(
            relative.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if stat.S_ISLNK(lexical.st_mode) or not stat.S_ISREG(lexical.st_mode):
            raise _InputRejected("input is not a regular file")
        source_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        source_descriptor = os.open(
            relative.name,
            source_flags,
            dir_fd=parent_descriptor,
        )
        before = os.fstat(source_descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or _source_fingerprint(before) != _source_fingerprint(lexical)
        ):
            raise _InputRejected("input is not a stable regular file")
        if before.st_size <= 0:
            raise _InputRejected("input file is empty")
        if before.st_size > max_input_bytes:
            raise _InputTooLarge("input exceeds configured size")
        return (
            input_root.path.joinpath(*relative.parts),
            parent_descriptor,
            source_descriptor,
            before,
        )
    except (_InputRejected, _InputTooLarge):
        if source_descriptor is not None:
            os.close(source_descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)
        raise
    except OSError as error:
        if source_descriptor is not None:
            os.close(source_descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)
        raise _InputRejected("input cannot be opened safely") from error


def _current_rooted_source_metadata(
    input_root: _InputRoot,
    relative_path: str,
    parent_descriptor: int,
) -> tuple[os.stat_result, os.stat_result]:
    """Verify an opened source against a fresh walk from the retained root fd."""
    relative = _rooted_relative_path(relative_path)
    verification_parent: int | None = None
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        current = os.stat(
            relative.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        verification_parent = input_root.duplicate_descriptor()
        for part in relative.parts[:-1]:
            child = os.open(part, directory_flags, dir_fd=verification_parent)
            os.close(verification_parent)
            verification_parent = child
        verification = os.stat(
            relative.name,
            dir_fd=verification_parent,
            follow_symlinks=False,
        )
    except (_InputRejected, OSError) as error:
        raise _InputContractMismatch("input path changed") from error
    finally:
        if verification_parent is not None:
            os.close(verification_parent)
    return current, verification


def _current_source_metadata(
    absolute: Path,
    parent_descriptor: int,
) -> tuple[os.stat_result, os.stat_result]:
    """Verify both the opened parent and the current lexical namespace."""
    try:
        current = os.stat(
            absolute.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        _, verification_parent = _open_parent_without_following(absolute)
    except OSError as error:
        raise _InputContractMismatch("input path changed") from error
    try:
        verification = os.stat(
            absolute.name,
            dir_fd=verification_parent,
            follow_symlinks=False,
        )
    except OSError as error:
        raise _InputContractMismatch("input path changed") from error
    finally:
        os.close(verification_parent)
    return current, verification


def _validate_real_directory(
    path: Path,
    *,
    label: str,
    require_private_owner: bool = False,
) -> Path:
    """Resolve a configured directory only after rejecting every lexical symlink."""
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    try:
        for part in absolute.parts[1:]:
            current = current / part
            metadata = current.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"{label} must not contain symlink ancestors")
        final = absolute.lstat()
    except ValueError:
        raise
    except OSError as error:
        raise ValueError(f"{label} is unavailable") from error
    if not stat.S_ISDIR(final.st_mode):
        raise ValueError(f"{label} must be a real directory")
    if require_private_owner:
        permission_bits = stat.S_IMODE(final.st_mode)
        if (
            final.st_uid != os.geteuid()
            or permission_bits & (stat.S_IWGRP | stat.S_IWOTH)
            or not permission_bits & stat.S_IWUSR
            or not permission_bits & stat.S_IXUSR
        ):
            raise ValueError(f"{label} has unsafe ownership or permissions")
    return absolute.resolve(strict=True)


def _ensure_real_directory(
    path: Path,
    *,
    label: str,
    require_private_owner: bool,
) -> Path:
    """Create missing directory components without ever traversing a symlink."""
    absolute = Path(os.path.abspath(path))
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(absolute.anchor, directory_flags)
    try:
        for part in absolute.parts[1:]:
            try:
                child = os.open(part, directory_flags, dir_fd=descriptor)
            except FileNotFoundError:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child = os.open(part, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
    except OSError as error:
        raise ValueError(f"{label} cannot be created safely") from error
    finally:
        os.close(descriptor)
    return _validate_real_directory(
        absolute,
        label=label,
        require_private_owner=require_private_owner,
    )


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _is_private_owned_regular(metadata: os.stat_result) -> bool:
    return (
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_uid == os.geteuid()
        and stat.S_IMODE(metadata.st_mode) == 0o600
        and metadata.st_nlink == 1
    )


def _is_private_owned_directory(metadata: os.stat_result) -> bool:
    return (
        stat.S_ISDIR(metadata.st_mode)
        and metadata.st_uid == os.geteuid()
        and stat.S_IMODE(metadata.st_mode) == 0o700
    )


def _open_owned_regular_at(
    parent_descriptor: int,
    name: str,
    *,
    writable: bool,
) -> int:
    """Open one private single-link file and bind it to its directory entry."""
    before = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    if not _is_private_owned_regular(before):
        raise OSError(errno.EPERM, "worker ownership file is unsafe")
    flags = (
        (os.O_RDWR if writable else os.O_RDONLY)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    try:
        opened = os.fstat(descriptor)
        current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (
            not _is_private_owned_regular(opened)
            or not _same_file_identity(before, opened)
            or not _same_file_identity(opened, current)
        ):
            raise OSError(errno.EPERM, "worker ownership file identity changed")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _descriptor_still_names_owned_file(
    parent_descriptor: int,
    name: str,
    descriptor: int,
) -> bool:
    try:
        opened = os.fstat(descriptor)
        current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError:
        return False
    return (
        _is_private_owned_regular(opened)
        and _is_private_owned_regular(current)
        and _same_file_identity(opened, current)
    )


def _read_small_descriptor(descriptor: int, *, limit: int = 512) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    payload = os.read(descriptor, limit + 1)
    if len(payload) > limit:
        raise OSError(errno.EFBIG, "worker ownership record is oversized")
    return payload


def _worker_owner_record(directory_name: str) -> bytes:
    return f"{_WORKER_OWNER_PREFIX}{directory_name}\n".encode("ascii")


def _bounded_directory_names(
    directory_descriptor: int,
    *,
    maximum: int,
) -> tuple[str, ...]:
    names: list[str] = []
    with os.scandir(directory_descriptor) as entries:
        for entry in entries:
            if len(names) >= maximum:
                raise OSError(errno.E2BIG, "worker directory has too many entries")
            names.append(entry.name)
    return tuple(names)


def _owned_snapshot_names(directory_descriptor: int) -> tuple[str, ...]:
    names = _bounded_directory_names(
        directory_descriptor,
        maximum=_MAX_WORKER_DIRECTORY_ENTRIES,
    )
    if _WORKER_OWNER_FILE not in names or _WORKER_LEASE_FILE not in names:
        raise OSError(errno.EPERM, "worker directory ownership files are missing")
    snapshots: list[str] = []
    for name in names:
        if name in {_WORKER_OWNER_FILE, _WORKER_LEASE_FILE}:
            continue
        if _WORKER_SNAPSHOT_RE.fullmatch(name) is None:
            raise OSError(errno.EPERM, "worker directory contains foreign entries")
        metadata = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if not _is_private_owned_regular(metadata):
            raise OSError(errno.EPERM, "worker snapshot entry is unsafe")
        snapshots.append(name)
    return tuple(snapshots)


def _cleanup_owned_worker_directory(
    root_descriptor: int,
    directory_name: str,
) -> bool:
    """Remove one proven owned, unlocked orphan without following any link."""
    if _WORKER_DIRECTORY_RE.fullmatch(directory_name) is None:
        return False
    directory_descriptor: int | None = None
    owner_descriptor: int | None = None
    lease_descriptor: int | None = None
    lease_locked = False
    try:
        before = os.stat(
            directory_name,
            dir_fd=root_descriptor,
            follow_symlinks=False,
        )
        if not _is_private_owned_directory(before):
            return False
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        directory_descriptor = os.open(
            directory_name,
            directory_flags,
            dir_fd=root_descriptor,
        )
        opened = os.fstat(directory_descriptor)
        current = os.stat(
            directory_name,
            dir_fd=root_descriptor,
            follow_symlinks=False,
        )
        if (
            not _is_private_owned_directory(opened)
            or not _same_file_identity(before, opened)
            or not _same_file_identity(opened, current)
        ):
            return False

        owner_descriptor = _open_owned_regular_at(
            directory_descriptor,
            _WORKER_OWNER_FILE,
            writable=False,
        )
        if _read_small_descriptor(owner_descriptor) != _worker_owner_record(
            directory_name
        ):
            return False
        lease_descriptor = _open_owned_regular_at(
            directory_descriptor,
            _WORKER_LEASE_FILE,
            writable=True,
        )
        try:
            fcntl.flock(lease_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno in {errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK}:
                return False
            raise
        lease_locked = True

        snapshots = _owned_snapshot_names(directory_descriptor)
        current = os.stat(
            directory_name,
            dir_fd=root_descriptor,
            follow_symlinks=False,
        )
        if (
            not _same_file_identity(opened, current)
            or not _descriptor_still_names_owned_file(
                directory_descriptor,
                _WORKER_OWNER_FILE,
                owner_descriptor,
            )
            or not _descriptor_still_names_owned_file(
                directory_descriptor,
                _WORKER_LEASE_FILE,
                lease_descriptor,
            )
            or _read_small_descriptor(owner_descriptor)
            != _worker_owner_record(directory_name)
        ):
            return False

        for name in snapshots:
            metadata = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if not _is_private_owned_regular(metadata):
                return False
        for name in snapshots:
            os.unlink(name, dir_fd=directory_descriptor)
        os.unlink(_WORKER_OWNER_FILE, dir_fd=directory_descriptor)
        os.unlink(_WORKER_LEASE_FILE, dir_fd=directory_descriptor)
        os.fsync(directory_descriptor)
        os.rmdir(directory_name, dir_fd=root_descriptor)
        os.fsync(root_descriptor)
        return True
    except OSError:
        return False
    finally:
        if lease_descriptor is not None:
            if lease_locked:
                try:
                    fcntl.flock(lease_descriptor, fcntl.LOCK_UN)
                except OSError:
                    pass
            os.close(lease_descriptor)
        if owner_descriptor is not None:
            os.close(owner_descriptor)
        if directory_descriptor is not None:
            os.close(directory_descriptor)


class _WhisperWorkRootLease:
    """Exclusive process ownership for one dedicated Whisper work root."""

    def __init__(self, path: Path, root_descriptor: int, descriptor: int) -> None:
        self.path = path
        self._root_descriptor: int | None = root_descriptor
        self._descriptor: int | None = descriptor

    @classmethod
    def acquire(cls, path: Path) -> _WhisperWorkRootLease:
        absolute, root_descriptor = _open_directory_without_following(path)
        descriptor: int | None = None
        created = False
        try:
            flags = (
                os.O_RDWR
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                descriptor = os.open(
                    _WORK_ROOT_LOCK_FILE,
                    flags,
                    0o600,
                    dir_fd=root_descriptor,
                )
                created = True
                os.fchmod(descriptor, 0o600)
                os.fsync(descriptor)
                os.fsync(root_descriptor)
            except FileExistsError:
                descriptor = _open_owned_regular_at(
                    root_descriptor,
                    _WORK_ROOT_LOCK_FILE,
                    writable=True,
                )
            assert descriptor is not None
            if created and not _descriptor_still_names_owned_file(
                root_descriptor,
                _WORK_ROOT_LOCK_FILE,
                descriptor,
            ):
                raise OSError(errno.EPERM, "work-root lease identity changed")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                if error.errno in {errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK}:
                    raise ValueError(
                        "Whisper worker work root is already in use"
                    ) from error
                raise
            if not _descriptor_still_names_owned_file(
                root_descriptor,
                _WORK_ROOT_LOCK_FILE,
                descriptor,
            ):
                raise OSError(errno.EPERM, "work-root lease identity changed")
        except ValueError:
            if descriptor is not None:
                os.close(descriptor)
            os.close(root_descriptor)
            raise
        except OSError as error:
            if descriptor is not None:
                os.close(descriptor)
            os.close(root_descriptor)
            raise ValueError(
                "Whisper worker work root cannot be owned safely"
            ) from error
        return cls(absolute, root_descriptor, descriptor)

    @property
    def descriptor(self) -> int:
        if self._descriptor is None:
            raise RuntimeError("Whisper worker work-root lease is closed")
        return self._descriptor

    @property
    def root_descriptor(self) -> int:
        if self._root_descriptor is None:
            raise RuntimeError("Whisper worker work-root lease is closed")
        return self._root_descriptor

    def cleanup_stale_directories(self) -> None:
        try:
            names = _bounded_directory_names(
                self.root_descriptor,
                maximum=_MAX_WORK_ROOT_ENTRIES,
            )
        except OSError as error:
            raise ValueError("Whisper worker work root cannot be inspected") from error
        for name in names:
            if _WORKER_DIRECTORY_RE.fullmatch(name) is not None:
                _cleanup_owned_worker_directory(self.root_descriptor, name)

    def create_temporary_root(self) -> _WhisperTemporaryRoot:
        for _attempt in range(32):
            directory_name = f"videoscope-whisper-worker-{uuid4().hex}"
            try:
                os.mkdir(directory_name, mode=0o700, dir_fd=self.root_descriptor)
            except FileExistsError:
                continue
            break
        else:
            raise ValueError("Whisper worker cannot reserve a private directory")

        directory_descriptor: int | None = None
        owner_descriptor: int | None = None
        lease_descriptor: int | None = None
        try:
            directory_flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            directory_descriptor = os.open(
                directory_name,
                directory_flags,
                dir_fd=self.root_descriptor,
            )
            os.fchmod(directory_descriptor, 0o700)
            opened_directory = os.fstat(directory_descriptor)
            named_directory = os.stat(
                directory_name,
                dir_fd=self.root_descriptor,
                follow_symlinks=False,
            )
            if (
                not _is_private_owned_directory(opened_directory)
                or not _same_file_identity(opened_directory, named_directory)
            ):
                raise OSError(errno.EPERM, "worker directory identity changed")
            file_flags = (
                os.O_RDWR
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            owner_descriptor = os.open(
                _WORKER_OWNER_FILE,
                file_flags,
                0o600,
                dir_fd=directory_descriptor,
            )
            os.fchmod(owner_descriptor, 0o600)
            payload = _worker_owner_record(directory_name)
            view = memoryview(payload)
            while view:
                written = os.write(owner_descriptor, view)
                if written <= 0:
                    raise OSError("worker owner record write did not make progress")
                view = view[written:]
            os.fsync(owner_descriptor)
            lease_descriptor = os.open(
                _WORKER_LEASE_FILE,
                file_flags,
                0o600,
                dir_fd=directory_descriptor,
            )
            os.fchmod(lease_descriptor, 0o600)
            os.fsync(lease_descriptor)
            fcntl.flock(lease_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.fsync(directory_descriptor)
            os.fsync(self.root_descriptor)
            if (
                not _descriptor_still_names_owned_file(
                    directory_descriptor,
                    _WORKER_OWNER_FILE,
                    owner_descriptor,
                )
                or not _descriptor_still_names_owned_file(
                    directory_descriptor,
                    _WORKER_LEASE_FILE,
                    lease_descriptor,
                )
                or _read_small_descriptor(owner_descriptor)
                != _worker_owner_record(directory_name)
            ):
                raise OSError(errno.EPERM, "worker ownership identity changed")
        except OSError as error:
            if lease_descriptor is not None:
                os.close(lease_descriptor)
            if owner_descriptor is not None:
                os.close(owner_descriptor)
            if directory_descriptor is not None:
                for name in (_WORKER_LEASE_FILE, _WORKER_OWNER_FILE):
                    try:
                        os.unlink(name, dir_fd=directory_descriptor)
                    except OSError:
                        pass
                os.close(directory_descriptor)
            try:
                os.rmdir(directory_name, dir_fd=self.root_descriptor)
            except OSError:
                pass
            raise ValueError(
                "Whisper worker work root cannot create a private directory"
            ) from error
        assert directory_descriptor is not None
        assert owner_descriptor is not None
        assert lease_descriptor is not None
        os.close(owner_descriptor)
        return _WhisperTemporaryRoot(
            root_lease=self,
            directory_name=directory_name,
            directory_descriptor=directory_descriptor,
            lease_descriptor=lease_descriptor,
        )

    def close(self) -> None:
        descriptor = self._descriptor
        root_descriptor = self._root_descriptor
        self._descriptor = None
        self._root_descriptor = None
        if descriptor is not None:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(descriptor)
        if root_descriptor is not None:
            os.close(root_descriptor)

    def __del__(self) -> None:
        try:
            self.close()
        except OSError:
            pass


class _WhisperTemporaryRoot:
    """One marked request directory whose lease lives with the ASGI app."""

    def __init__(
        self,
        *,
        root_lease: _WhisperWorkRootLease,
        directory_name: str,
        directory_descriptor: int,
        lease_descriptor: int,
    ) -> None:
        self._root_lease = root_lease
        self._directory_name = directory_name
        self._directory_descriptor: int | None = directory_descriptor
        self._lease_descriptor: int | None = lease_descriptor
        self.name = str(root_lease.path / directory_name)

    @property
    def lease_descriptor(self) -> int:
        if self._lease_descriptor is None:
            raise RuntimeError("Whisper worker temporary-root lease is closed")
        return self._lease_descriptor

    def close(self) -> None:
        lease_descriptor = self._lease_descriptor
        directory_descriptor = self._directory_descriptor
        self._lease_descriptor = None
        self._directory_descriptor = None
        if lease_descriptor is None and directory_descriptor is None:
            return
        if lease_descriptor is not None:
            try:
                fcntl.flock(lease_descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(lease_descriptor)
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        try:
            _cleanup_owned_worker_directory(
                self._root_lease.root_descriptor,
                self._directory_name,
            )
        except (OSError, RuntimeError):
            logger.error("Whisper worker could not clean its private directory")

    def __del__(self) -> None:
        try:
            self.close()
        except OSError:
            pass


def _inspect_regular_source(
    source: Path,
    *,
    max_input_bytes: int,
    input_root: _InputRoot | None = None,
    relative_path: str | None = None,
) -> _SourceContract:
    if (input_root is None) != (relative_path is None):
        raise ValueError(
            "rooted source inspection requires both root and relative path"
        )
    if input_root is None:
        absolute, parent_descriptor, descriptor, before = _open_regular_source(
            source,
            max_input_bytes=max_input_bytes,
        )
    else:
        assert relative_path is not None
        absolute, parent_descriptor, descriptor, before = _open_rooted_regular_source(
            input_root,
            relative_path,
            max_input_bytes=max_input_bytes,
        )
    try:
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_input_bytes:
                raise _InputTooLarge("input exceeds configured size")
            digest.update(chunk)
        after = os.fstat(descriptor)
        if input_root is None:
            current, verification = _current_source_metadata(
                absolute,
                parent_descriptor,
            )
        else:
            assert relative_path is not None
            current, verification = _current_rooted_source_metadata(
                input_root,
                relative_path,
                parent_descriptor,
            )
    except OSError as error:
        raise _InputRejected("input could not be read safely") from error
    finally:
        os.close(descriptor)
        os.close(parent_descriptor)
    fingerprint = _source_fingerprint(before)
    if (
        total != before.st_size
        or _source_fingerprint(after) != fingerprint
        or _source_fingerprint(current) != fingerprint
        or _source_fingerprint(verification) != fingerprint
        or stat.S_ISLNK(current.st_mode)
        or stat.S_ISLNK(verification.st_mode)
    ):
        raise _InputContractMismatch("input changed while hashing")
    return _SourceContract(
        path=absolute,
        fingerprint=fingerprint,
        size=total,
        sha256=digest.hexdigest(),
    )


def _copy_source_snapshot(
    source: Path,
    *,
    max_input_bytes: int,
    temp_root: Path,
    input_root: _InputRoot | None = None,
    relative_path: str | None = None,
) -> tuple[_SourceContract, Path]:
    """Copy/hash from one no-follow fd so ML sees immutable worker-owned bytes."""
    if (input_root is None) != (relative_path is None):
        raise ValueError("rooted source copy requires both root and relative path")
    if input_root is None:
        absolute, parent_descriptor, source_fd, before = _open_regular_source(
            source,
            max_input_bytes=max_input_bytes,
        )
    else:
        assert relative_path is not None
        absolute, parent_descriptor, source_fd, before = _open_rooted_regular_source(
            input_root,
            relative_path,
            max_input_bytes=max_input_bytes,
        )
    destination_fd: int | None = None
    destination: Path | None = None
    try:
        try:
            free_bytes = shutil.disk_usage(temp_root).free
        except OSError as error:
            raise _WorkRootUnavailable("worker work root cannot be measured") from error
        if free_bytes < before.st_size + WORK_ROOT_FREE_RESERVE_BYTES:
            raise _WorkRootUnavailable("worker work root has insufficient free space")
        try:
            destination_fd, destination_name = tempfile.mkstemp(
                prefix="input-",
                suffix=source.suffix.casefold(),
                dir=temp_root,
            )
        except OSError as error:
            raise _WorkRootUnavailable("worker snapshot cannot be created") from error
        destination = Path(destination_name)
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_input_bytes:
                raise _InputTooLarge("input exceeds configured size")
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_fd, view)
                if written <= 0:
                    raise OSError("worker snapshot write did not make progress")
                view = view[written:]
        os.fsync(destination_fd)
        after = os.fstat(source_fd)
        destination_stat = os.fstat(destination_fd)
        if input_root is None:
            current, verification = _current_source_metadata(
                absolute,
                parent_descriptor,
            )
        else:
            assert relative_path is not None
            current, verification = _current_rooted_source_metadata(
                input_root,
                relative_path,
                parent_descriptor,
            )
    except OSError as error:
        if destination is not None:
            _discard_snapshot(destination)
        raise _WorkRootUnavailable("worker snapshot write failed") from error
    except Exception:
        if destination is not None:
            _discard_snapshot(destination)
        raise
    finally:
        os.close(source_fd)
        os.close(parent_descriptor)
        if destination_fd is not None:
            os.close(destination_fd)
    if destination is None:
        raise _WorkRootUnavailable("worker snapshot was not created")
    fingerprint = _source_fingerprint(before)
    if (
        total != before.st_size
        or destination_stat.st_size != total
        or _source_fingerprint(after) != fingerprint
        or _source_fingerprint(current) != fingerprint
        or _source_fingerprint(verification) != fingerprint
        or stat.S_ISLNK(current.st_mode)
        or stat.S_ISLNK(verification.st_mode)
    ):
        _discard_snapshot(destination)
        raise _InputContractMismatch("input changed while snapshotting")
    try:
        os.chmod(destination, 0o600)
    except OSError as error:
        _discard_snapshot(destination)
        raise _WorkRootUnavailable("worker snapshot permissions could not be set") from error
    return (
        _SourceContract(absolute, fingerprint, total, digest.hexdigest()),
        destination,
    )


def _default_duration_probe(source: Path) -> float:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(source),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError("media duration probe failed") from error
    if result.returncode != 0 or len(result.stdout) > 1024:
        raise ValueError("media duration probe failed")
    try:
        duration = float(result.stdout.decode("ascii").strip())
    except (UnicodeError, ValueError) as error:
        raise ValueError("media duration probe returned invalid output") from error
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("media duration is invalid")
    return duration


def _timed_text_payload(item: TimedText) -> WhisperSegmentPayload:
    if not isinstance(item, TimedText) or not isinstance(item.metadata, dict):
        raise ValueError("runtime segment has the wrong type")
    words_payload: list[WhisperWordPayload] = []
    raw_words = item.metadata.get("words", [])
    if not isinstance(raw_words, list):
        raise ValueError("segment words must be a list")
    for raw_word in raw_words:
        if not isinstance(raw_word, dict) or set(raw_word) != {
            "word",
            "start",
            "end",
            "probability",
        }:
            raise ValueError("word payload is invalid")
        words_payload.append(
            WhisperWordPayload(
                text=raw_word["word"],
                start=raw_word["start"],
                end=raw_word["end"],
                confidence=raw_word["probability"],
            )
        )
    return WhisperSegmentPayload(
        start=item.start,
        end=item.end,
        text=item.text,
        confidence=item.confidence,
        words=words_payload,
    )


def _transcript_payloads(transcript: WhisperTranscript) -> list[WhisperSegmentPayload]:
    if not isinstance(transcript, WhisperTranscript):
        raise ValueError("runtime transcript has the wrong type")
    if not isinstance(transcript.language, str):
        raise ValueError("runtime transcript language has the wrong type")
    if not isinstance(transcript.segments, tuple):
        raise ValueError("runtime transcript segments must be an immutable tuple")
    if len(transcript.segments) > MAX_OUTPUT_SEGMENTS:
        raise ValueError("runtime transcript has too many segments")
    return [_timed_text_payload(item) for item in transcript.segments]


def create_whisper_worker_app(
    *,
    runtime: WorkerRuntime,
    input_root: Path,
    work_root: Path,
    api_key: str,
    duration_probe: Callable[[Path], float] = _default_duration_probe,
    max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES,
    max_duration_seconds: float = DEFAULT_MAX_DURATION_SECONDS,
    max_concurrency: int = 1,
    max_request_bytes: int = MAX_REQUEST_BYTES,
) -> FastAPI:
    if not _TOKEN_RE.fullmatch(api_key):
        raise ValueError("Whisper worker API key must be 32-256 URL-safe characters")
    if not 0 < max_input_bytes <= DEFAULT_MAX_INPUT_BYTES:
        raise ValueError("Whisper worker input size limit is invalid")
    if not 0 < max_duration_seconds <= DEFAULT_MAX_DURATION_SECONDS:
        raise ValueError("Whisper worker duration limit is invalid")
    if not 0 < max_request_bytes <= MAX_REQUEST_BYTES:
        raise ValueError("Whisper worker request size limit is invalid")
    if max_concurrency != 1:
        raise ValueError("Whisper worker concurrency must be exactly one")
    resolved_root = _validate_real_directory(
        Path(input_root),
        label="Whisper worker input root",
    )
    resolved_work_root = _validate_real_directory(
        Path(work_root),
        label="Whisper worker work root",
        require_private_owner=True,
    )
    if _PINNED_MODEL_RE.fullmatch(runtime.model_identity) is None:
        raise ValueError("Whisper worker model identity must be immutable")
    if runtime.runtime_identity != WHISPER_INFERENCE_RUNTIME_IDENTITY:
        raise ValueError("Whisper worker runtime identity does not match the contract")
    if runtime.dependency_identity != WHISPER_DEPENDENCY_IDENTITY:
        raise ValueError("Whisper worker dependency identity does not match the lock")

    try:
        retained_input_root = _InputRoot.open(resolved_root)
    except _InputRejected as error:
        raise ValueError("Whisper worker input root is unavailable") from error
    try:
        work_root_lease = _WhisperWorkRootLease.acquire(resolved_work_root)
        work_root_lease.cleanup_stale_directories()
        temporary_root = work_root_lease.create_temporary_root()
    except (OSError, ValueError) as error:
        retained_input_root.close()
        if "work_root_lease" in locals():
            work_root_lease.close()
        if isinstance(error, ValueError):
            raise
        raise ValueError("Whisper worker work root is unavailable") from error
    temp_root = Path(temporary_root.name)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            try:
                temporary_root.close()
            finally:
                try:
                    retained_input_root.close()
                finally:
                    work_root_lease.close()

    app = FastAPI(
        title="VideoScope Whisper Worker",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    capacity = BoundedSemaphore(max_concurrency)
    # Keep both process and per-directory leases alive for the ASGI app lifetime.
    app.state.whisper_temporary_root = temporary_root
    app.state.whisper_work_root_lease = work_root_lease
    app.state.whisper_input_root = retained_input_root

    @app.get("/v1/health", response_model=WhisperHealthResponse)
    def health() -> WhisperHealthResponse:
        return WhisperHealthResponse(
            schema_version=WHISPER_WORKER_SCHEMA_VERSION,
            status="ok" if runtime.available else "unavailable",
            model_identity=runtime.model_identity,
            runtime_identity=runtime.runtime_identity,
            dependency_identity=runtime.dependency_identity,
            loaded=runtime.loaded,
            max_input_bytes=max_input_bytes,
            max_duration_seconds=max_duration_seconds,
            max_request_bytes=max_request_bytes,
            max_response_bytes=MAX_RESPONSE_BYTES,
            max_concurrency=max_concurrency,
        )

    @app.post("/v1/transcribe", response_model=WhisperTranscribeResponse)
    async def transcribe(request: WhisperTranscribeRequest) -> WhisperTranscribeResponse:
        if (
            request.model_identity != runtime.model_identity
            or request.runtime_identity != runtime.runtime_identity
            or request.dependency_identity != runtime.dependency_identity
        ):
            raise HTTPException(409, "Whisper worker identity does not match")
        if request.source_size > max_input_bytes:
            raise HTTPException(413, "Whisper worker input is too large")
        if not runtime.available:
            raise HTTPException(503, "Whisper worker runtime is unavailable")
        if not capacity.acquire(blocking=False):
            raise HTTPException(429, "Whisper worker is busy")

        snapshot_path: Path | None = None
        try:
            try:
                relative = _rooted_relative_path(request.relative_path)
                source = retained_input_root.path.joinpath(*relative.parts)
                before, snapshot_path = await run_in_threadpool(
                    _copy_source_snapshot,
                    source,
                    max_input_bytes=max_input_bytes,
                    temp_root=temp_root,
                    input_root=retained_input_root,
                    relative_path=request.relative_path,
                )
            except _InputTooLarge as error:
                raise HTTPException(413, "Whisper worker input is too large") from error
            except _InputContractMismatch as error:
                raise HTTPException(409, "Whisper worker input changed") from error
            except _WorkRootUnavailable as error:
                raise HTTPException(503, "Whisper worker storage is unavailable") from error
            except _InputRejected as error:
                raise HTTPException(400, "Invalid Whisper worker input") from error
            if before.size != request.source_size or before.sha256 != request.source_sha256:
                raise HTTPException(409, "Whisper worker source identity does not match")
            try:
                duration = await run_in_threadpool(duration_probe, snapshot_path)
            except Exception as error:
                raise HTTPException(400, "Invalid Whisper worker media") from error
            if not isinstance(duration, (int, float)) or isinstance(duration, bool):
                raise HTTPException(400, "Invalid Whisper worker media")
            duration = float(duration)
            if not math.isfinite(duration) or duration <= 0:
                raise HTTPException(400, "Invalid Whisper worker media")
            if duration > max_duration_seconds:
                raise HTTPException(413, "Whisper worker input duration is too large")
            try:
                transcript = await run_in_threadpool(
                    runtime.transcribe,
                    request,
                    snapshot_path,
                    duration,
                )
            except Exception as error:
                logger.exception("Whisper worker inference failed")
                raise HTTPException(503, "Whisper worker inference failed") from error
            try:
                after = await run_in_threadpool(
                    _inspect_regular_source,
                    source,
                    max_input_bytes=max_input_bytes,
                    input_root=retained_input_root,
                    relative_path=request.relative_path,
                )
            except (_InputRejected, _InputTooLarge, _InputContractMismatch) as error:
                raise HTTPException(
                    409, "Whisper worker input changed during inference"
                ) from error
            if after != before:
                raise HTTPException(409, "Whisper worker input changed during inference")
            try:
                segments = _transcript_payloads(transcript)
                response = WhisperTranscribeResponse(
                    schema_version=WHISPER_WORKER_SCHEMA_VERSION,
                    request_id=request.request_id,
                    model_identity=runtime.model_identity,
                    runtime_identity=runtime.runtime_identity,
                    dependency_identity=runtime.dependency_identity,
                    source_size=before.size,
                    source_sha256=before.sha256,
                    effective_prompt_sha256=request.effective_prompt_sha256,
                    glossary_state=request.glossary_state,
                    duration_seconds=duration,
                    detected_language=transcript.language,
                    segments=segments,
                )
                if len(response.model_dump_json().encode("utf-8")) > MAX_RESPONSE_BYTES:
                    raise ValueError("serialized transcript exceeds the response limit")
            except (TypeError, ValueError, ValidationError) as error:
                logger.warning("Whisper worker rejected invalid inference output")
                raise HTTPException(
                    503, "Whisper worker produced invalid output"
                ) from error
            return response
        finally:
            if snapshot_path is not None:
                _discard_snapshot(snapshot_path)
            capacity.release()

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
        or parsed.hostname != "127.0.0.1"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("Whisper worker endpoint must use plain HTTP on 127.0.0.1")
    try:
        address = ip_address(parsed.hostname)
        port = parsed.port
    except ValueError as error:
        raise ValueError("Whisper worker endpoint must use 127.0.0.1") from error
    if str(address) != "127.0.0.1" or port is None or not 1 <= port <= 65535:
        raise ValueError("Whisper worker endpoint must use 127.0.0.1 with a port")
    return endpoint.strip().rstrip("/")


def _client_source_contract(
    input_root: Path,
    input_root_fingerprint: _DirectoryFingerprint,
    source: Path,
) -> tuple[str, _SourceContract]:
    lexical = Path(os.path.abspath(source))
    try:
        relative = lexical.relative_to(input_root)
    except ValueError as error:
        raise RuntimeError("Whisper worker input is outside configured input root") from error
    relative_path = PurePosixPath(*relative.parts).as_posix()
    rooted_input: _InputRoot | None = None
    try:
        rooted_input = _InputRoot.open(
            input_root,
            expected=input_root_fingerprint,
        )
        relative = _rooted_relative_path(relative_path)
        safe_source = rooted_input.path.joinpath(*relative.parts)
        contract = _inspect_regular_source(
            safe_source,
            max_input_bytes=DEFAULT_MAX_INPUT_BYTES,
            input_root=rooted_input,
            relative_path=relative_path,
        )
    except (_InputRejected, _InputTooLarge, _InputContractMismatch):
        raise RuntimeError("Whisper worker input is unsafe or unavailable") from None
    finally:
        if rooted_input is not None:
            rooted_input.close()
    return relative_path, contract


class WhisperWorkerClient:
    """Authenticated adapter to a pinned isolated mlx-whisper process."""

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        input_root: Path,
        expected_model_identity: str,
        timeout: float = 600.0,
        client: HTTPClient | None = None,
        request_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.endpoint = _validate_endpoint(endpoint)
        if not _TOKEN_RE.fullmatch(api_key):
            raise ValueError("Whisper worker API key must be 32-256 URL-safe characters")
        if _PINNED_MODEL_RE.fullmatch(expected_model_identity) is None:
            raise ValueError("expected Whisper model identity must be immutable")
        if not 0 < timeout <= 3600:
            raise ValueError("Whisper worker timeout must be between 0 and 3600 seconds")
        resolved_root = _validate_real_directory(
            Path(input_root),
            label="Whisper worker input root",
        )
        try:
            initial_root = _InputRoot.open(resolved_root)
        except _InputRejected as error:
            raise ValueError("Whisper worker input root is unavailable") from error
        self._api_key = api_key
        self.input_root = resolved_root
        self._input_root_fingerprint = initial_root.fingerprint
        initial_root.close()
        self.expected_model_identity = expected_model_identity
        self.timeout = timeout
        self.client = client
        self.request_id_factory = request_id_factory or (lambda: uuid4().hex)
        self._client_lock = Lock()
        self._status_lock = Lock()
        self._cached_status: tuple[float, WhisperInferenceStatus] | None = None

    @property
    def identity(self) -> dict[str, object]:
        return {
            "mode": "isolated-worker",
            "contract": WHISPER_WORKER_SCHEMA_VERSION,
            "endpoint": self.endpoint,
            "model": self.expected_model_identity,
            "runtime_identity": WHISPER_INFERENCE_RUNTIME_IDENTITY,
            "dependency_identity": WHISPER_DEPENDENCY_IDENTITY,
        }

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

    def status(self) -> WhisperInferenceStatus:
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
            capability = WhisperHealthResponse.model_validate(response.json())
        except Exception:
            status = WhisperInferenceStatus(False, "Whisper worker is unreachable")
        else:
            identity_matches = (
                capability.model_identity == self.expected_model_identity
                and capability.runtime_identity == WHISPER_INFERENCE_RUNTIME_IDENTITY
                and capability.dependency_identity == WHISPER_DEPENDENCY_IDENTITY
            )
            if not identity_matches:
                status = WhisperInferenceStatus(
                    False, "Whisper worker identity does not match"
                )
            elif capability.status != "ok":
                status = WhisperInferenceStatus(
                    False, "Whisper worker model is unavailable"
                )
            else:
                status = WhisperInferenceStatus(True, "Whisper worker is ready")
        with self._status_lock:
            self._cached_status = (
                monotonic() + (5.0 if status.ready else 2.0),
                status,
            )
        return status

    def transcribe(
        self,
        source: Path,
        *,
        language: str,
        prompt_snapshot: WhisperPromptSnapshot,
    ) -> list[TimedText]:
        relative_path, source_contract = _client_source_contract(
            self.input_root,
            self._input_root_fingerprint,
            source,
        )
        request_id = self.request_id_factory()
        try:
            request = WhisperTranscribeRequest(
                schema_version=WHISPER_WORKER_SCHEMA_VERSION,
                request_id=request_id,
                model_identity=self.expected_model_identity,
                runtime_identity=WHISPER_INFERENCE_RUNTIME_IDENTITY,
                dependency_identity=WHISPER_DEPENDENCY_IDENTITY,
                relative_path=relative_path,
                source_size=source_contract.size,
                source_sha256=source_contract.sha256,
                language=language,
                effective_prompt=prompt_snapshot.effective_prompt,
                effective_prompt_sha256=prompt_snapshot.effective_prompt_sha256,
                glossary_state=prompt_snapshot.glossary_state,
            )
        except (TypeError, ValueError, ValidationError):
            raise RuntimeError("Whisper worker request contract violation") from None
        try:
            response = self._http_client().post(
                f"{self.endpoint}/v1/transcribe",
                json=request.model_dump(mode="json"),
                headers=self._headers(json_request=True),
                timeout=self.timeout,
            )
            response.raise_for_status()
            raw_payload = response.json()
        except Exception:
            raise RuntimeError("Whisper worker request failed") from None
        try:
            payload = WhisperTranscribeResponse.model_validate(raw_payload)
            if (
                payload.request_id != request_id
                or payload.model_identity != self.expected_model_identity
                or payload.runtime_identity != WHISPER_INFERENCE_RUNTIME_IDENTITY
                or payload.dependency_identity != WHISPER_DEPENDENCY_IDENTITY
                or payload.source_size != source_contract.size
                or payload.source_sha256 != source_contract.sha256
                or payload.effective_prompt_sha256
                != prompt_snapshot.effective_prompt_sha256
                or payload.glossary_state != prompt_snapshot.glossary_state
            ):
                raise ValueError("Whisper worker response identity mismatch")
            return [
                TimedText(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                    confidence=segment.confidence,
                    metadata={
                        "language": payload.detected_language,
                        "engine": "mlx-whisper",
                        **(
                            {
                                "words": [
                                    {
                                        "word": word.text,
                                        "start": word.start,
                                        "end": word.end,
                                        "probability": word.confidence,
                                    }
                                    for word in segment.words
                                ]
                            }
                            if segment.words
                            else {}
                        ),
                    },
                )
                for segment in payload.segments
            ]
        except (TypeError, ValueError, ValidationError):
            raise RuntimeError("Whisper worker response contract violation") from None


def _installed_dependency_identity() -> str:
    for distribution, expected_version in WHISPER_EXACT_DEPENDENCIES:
        try:
            installed = importlib.metadata.version(distribution)
        except Exception:
            return "unavailable"
        if installed != expected_version:
            return "unavailable"
    return WHISPER_DEPENDENCY_IDENTITY


def _runtime_platform_is_exact() -> bool:
    try:
        macos_major = int(platform.mac_ver()[0].split(".", 1)[0])
    except (TypeError, ValueError):
        return False
    return (
        tuple(sys.version_info[:3]) == WHISPER_WORKER_PYTHON_VERSION
        and platform.system() == WHISPER_WORKER_SYSTEM
        and platform.machine() == WHISPER_WORKER_MACHINE
        and macos_major >= 14
    )


def _validated_model_artifact_state(
    reference: Path,
    *,
    model_name: str,
    revision: str,
    cached_state: tuple[tuple[object, ...], ...] | None,
    force_hash: bool,
) -> tuple[tuple[object, ...], ...]:
    artifacts = WHISPER_REVIEWED_MODEL_ARTIFACTS.get((model_name, revision))
    if artifacts is None:
        raise RuntimeError("Whisper model identity has no reviewed artifact manifest")
    if not reference.is_dir() or reference.is_symlink():
        raise RuntimeError("local Whisper snapshot directory is unsafe")

    resolved_artifacts: list[tuple[Path, int, str]] = []
    state: list[tuple[object, ...]] = []
    for filename, expected_size, expected_sha256 in artifacts:
        lexical = reference / filename
        try:
            resolved = lexical.resolve(strict=True)
            metadata = resolved.stat()
        except OSError as error:
            raise RuntimeError("local Whisper snapshot is incomplete") from error
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != expected_size:
            raise RuntimeError("local Whisper model artifact size does not match")
        resolved_artifacts.append((resolved, expected_size, expected_sha256))
        state.append(
            (
                filename,
                str(resolved),
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
            )
        )
    frozen_state = tuple(state)
    if not force_hash and cached_state == frozen_state:
        return frozen_state

    for resolved, expected_size, expected_sha256 in resolved_artifacts:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(resolved, flags)
        except OSError as error:
            raise RuntimeError("local Whisper model artifact cannot be opened") from error
        try:
            before = os.fstat(descriptor)
            digest = hashlib.sha256()
            total = 0
            while chunk := os.read(descriptor, 1024 * 1024):
                total += len(chunk)
                if total > expected_size:
                    raise RuntimeError("local Whisper model artifact grew while hashing")
                digest.update(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        if (
            total != expected_size
            or digest.hexdigest() != expected_sha256
            or _source_fingerprint(before) != _source_fingerprint(after)
        ):
            raise RuntimeError("local Whisper model artifact hash does not match")
    return frozen_state


def _normalize_language(value: object, _fallback: str) -> str:
    if type(value) is not str:
        raise ValueError("mlx-whisper language is invalid")
    language = value.strip()
    if (
        language != language.casefold()
        or language == "auto"
        or _LANGUAGE_RE.fullmatch(language) is None
    ):
        raise ValueError("mlx-whisper language is invalid")
    return language


def _strict_model_number(raw: dict[str, object], key: str) -> float:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"mlx-whisper {key} is invalid")
    return float(value)


def _normalize_mlx_transcript(
    raw_result: object,
    *,
    requested_language: str,
    duration_seconds: float,
) -> WhisperTranscript:
    if not isinstance(raw_result, dict):
        raise ValueError("mlx-whisper result must be an object")
    raw_segments = raw_result.get("segments")
    if not isinstance(raw_segments, list):
        raise ValueError("mlx-whisper segments must be a list")
    if len(raw_segments) > MAX_OUTPUT_SEGMENTS:
        raise ValueError("mlx-whisper returned too many segments")
    detected_language = _normalize_language(
        raw_result.get("language"),
        requested_language,
    )
    segments: list[TimedText] = []
    total_words = 0
    total_text = 0
    previous_segment_end: float | None = None
    for raw in raw_segments:
        if not isinstance(raw, dict):
            raise ValueError("mlx-whisper segment has the wrong type")
        try:
            raw_text = raw.get("text")
            if type(raw_text) is not str:
                raise TypeError("segment text has the wrong type")
            text = raw_text.strip()
            start = _strict_model_number(raw, "start")
            end = _strict_model_number(raw, "end")
            average_log_probability = _strict_model_number(raw, "avg_logprob")
        except (TypeError, ValueError) as error:
            raise ValueError("mlx-whisper segment is invalid") from error
        if (
            not text
            or not all(
                math.isfinite(value)
                for value in (start, end, average_log_probability)
            )
            or (
                previous_segment_end is not None
                and start < previous_segment_end
            )
            or start < 0
            or end <= start
            or end > duration_seconds + 1e-3
        ):
            raise ValueError("mlx-whisper segment timeline is invalid")
        if len(text) > MAX_SEGMENT_TEXT_CHARS:
            raise ValueError("mlx-whisper segment text is too long")
        confidence = max(0.0, min(1.0, math.exp(min(0.0, average_log_probability))))
        words: list[dict[str, object]] = []
        raw_words = raw.get("words")
        if not isinstance(raw_words, list):
            raise ValueError("mlx-whisper words must be a list")
        if len(raw_words) > MAX_WORDS_PER_SEGMENT:
            raise ValueError("mlx-whisper returned too many words in one segment")
        previous_word_end: float | None = None
        for raw_word in raw_words:
            if total_words >= MAX_OUTPUT_WORDS:
                raise ValueError("mlx-whisper returned too many words")
            if not isinstance(raw_word, dict):
                raise ValueError("mlx-whisper word has the wrong type")
            try:
                raw_word_text = raw_word.get("word")
                if type(raw_word_text) is not str:
                    raise TypeError("word text has the wrong type")
                word_text = raw_word_text.strip()
                word_start = _strict_model_number(raw_word, "start")
                word_end = _strict_model_number(raw_word, "end")
                probability = _strict_model_number(raw_word, "probability")
            except (TypeError, ValueError) as error:
                raise ValueError("mlx-whisper word is invalid") from error
            if (
                not word_text
                or not all(
                    math.isfinite(value)
                    for value in (word_start, word_end, probability)
                )
                or (
                    previous_word_end is not None
                    and word_start < previous_word_end
                )
                or word_start < start
                or word_end < word_start
                or word_end > end
                or not 0 <= probability <= 1
            ):
                raise ValueError("mlx-whisper word timeline is invalid")
            if len(word_text) > MAX_WORD_TEXT_CHARS:
                raise ValueError("mlx-whisper word text is too long")
            previous_word_end = word_end
            total_words += 1
            total_text += len(word_text)
            if word_end == word_start:
                # MLX timestamp-token quantization can collapse a valid word to a
                # point. Preserve the strict segment and ordering contracts, but
                # omit unusable word-level metadata instead of inventing bounds.
                continue
            words.append(
                {
                    "word": word_text,
                    "start": word_start,
                    "end": word_end,
                    "probability": probability,
                }
            )
        total_text += len(text)
        if total_text > MAX_OUTPUT_TEXT_CHARS:
            raise ValueError("mlx-whisper returned too much text")
        segments.append(
            TimedText(
                start=start,
                end=end,
                text=text,
                confidence=confidence,
                metadata={
                    "language": detected_language,
                    "engine": "mlx-whisper",
                    **({"words": words} if words else {}),
                },
            )
        )
        previous_segment_end = end
    return WhisperTranscript(detected_language, tuple(segments))


class MLXWhisperWorkerRuntime:
    """Pinned lazy mlx-whisper adapter with separate load and inference locks."""

    runtime_identity = WHISPER_INFERENCE_RUNTIME_IDENTITY
    dependency_identity = WHISPER_DEPENDENCY_IDENTITY

    def __init__(self, model_name: str, revision: str) -> None:
        if not model_name.strip():
            raise ValueError("Whisper worker model name must not be empty")
        if re.fullmatch(r"[0-9a-f]{40,64}", revision) is None:
            raise ValueError("Whisper worker model revision must be an immutable SHA")
        self.model_name = model_name.strip()
        self.model_revision = revision
        self._model_reference: str | None = None
        self._model_artifact_state: tuple[tuple[object, ...], ...] | None = None
        self._load_lock = Lock()
        self._inference_lock = Lock()

    @property
    def model_identity(self) -> str:
        return model_identity(self.model_name, self.model_revision)

    @property
    def loaded(self) -> bool:
        # mlx-whisper owns model allocation inside each call; health validation
        # only resolves and inspects the pinned local snapshot.
        return False

    def _resolve_local_reference(self, *, force_hash: bool = False) -> str:
        with self._load_lock:
            if self._model_reference is None:
                from huggingface_hub import snapshot_download

                reference = Path(
                    snapshot_download(
                        self.model_name,
                        revision=self.model_revision,
                        local_files_only=True,
                    )
                ).resolve(strict=True)
                self._model_reference = str(reference)
            reference = Path(self._model_reference)
            self._model_artifact_state = _validated_model_artifact_state(
                reference,
                model_name=self.model_name,
                revision=self.model_revision,
                cached_state=self._model_artifact_state,
                force_hash=force_hash,
            )
            return self._model_reference

    @property
    def available(self) -> bool:
        if (
            not _runtime_platform_is_exact()
            or _installed_dependency_identity() != WHISPER_DEPENDENCY_IDENTITY
        ):
            return False
        try:
            self._resolve_local_reference()
        except Exception:
            return False
        return True

    def transcribe(
        self,
        request: WhisperTranscribeRequest,
        source: Path,
        duration_seconds: float,
    ) -> WhisperTranscript:
        if (
            not _runtime_platform_is_exact()
            or _installed_dependency_identity() != WHISPER_DEPENDENCY_IDENTITY
        ):
            raise RuntimeError("Whisper worker runtime identity is unavailable")
        reference = self._resolve_local_reference(force_hash=True)
        mlx_whisper = importlib.import_module("mlx_whisper")
        options: dict[str, object] = {
            "path_or_hf_repo": reference,
            "word_timestamps": True,
            "verbose": False,
            "temperature": 0.0,
            "condition_on_previous_text": True,
            "hallucination_silence_threshold": 2.0,
        }
        if request.language != "auto":
            options["language"] = request.language
        if request.effective_prompt:
            options["initial_prompt"] = request.effective_prompt
        with self._inference_lock:
            raw_result = mlx_whisper.transcribe(str(source), **options)
            self._resolve_local_reference(force_hash=True)
        return _normalize_mlx_transcript(
            raw_result,
            requested_language=request.language,
            duration_seconds=duration_seconds,
        )


class WhisperWorkerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VIDEOSCOPE_WHISPER_WORKER_",
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )

    host: Literal["127.0.0.1"] = "127.0.0.1"
    port: int = Field(default=8784, ge=1, le=65535)
    api_key: str = Field(
        min_length=32,
        max_length=256,
        pattern=r"^[A-Za-z0-9._~-]+$",
        repr=False,
    )
    input_root: Path
    work_root: Path
    model_name: str = Field(
        validation_alias=AliasChoices(
            "WHISPER_MODEL",
            "VIDEOSCOPE_WHISPER_MODEL",
            "VIDEOSCOPE_WHISPER_WORKER_MODEL_NAME",
        )
    )
    model_revision: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    max_input_bytes: int = Field(default=DEFAULT_MAX_INPUT_BYTES, gt=0)
    max_duration_seconds: float = Field(
        default=DEFAULT_MAX_DURATION_SECONDS,
        gt=0,
    )
    log_level: Literal["critical", "error", "warning", "info"] = "info"


def main() -> None:
    import uvicorn

    settings = WhisperWorkerSettings()  # type: ignore[call-arg]
    input_root = _ensure_real_directory(
        settings.input_root,
        label="Whisper worker input root",
        require_private_owner=False,
    )
    work_root = _ensure_real_directory(
        settings.work_root,
        label="Whisper worker work root",
        require_private_owner=True,
    )
    runtime = MLXWhisperWorkerRuntime(
        settings.model_name,
        settings.model_revision,
    )
    app = create_whisper_worker_app(
        runtime=runtime,
        input_root=input_root,
        work_root=work_root,
        api_key=settings.api_key,
        max_input_bytes=settings.max_input_bytes,
        max_duration_seconds=settings.max_duration_seconds,
    )
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        access_log=False,
        proxy_headers=False,
        server_header=False,
    )


if __name__ == "__main__":
    main()
