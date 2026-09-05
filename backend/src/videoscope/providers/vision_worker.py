from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha1, sha256
import hmac
from importlib import metadata as importlib_metadata
from ipaddress import ip_address
import json
import logging
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import secrets
import stat
import sys
from tempfile import TemporaryDirectory
from threading import BoundedSemaphore, Lock
from typing import Any, Literal, Protocol
from uuid import uuid4
import warnings

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from pydantic import AliasChoices, Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from videoscope.model_manifest import (
    MODEL_REVISIONS,
    SIGLIP_224_MODEL,
    SIGLIP_384_MODEL,
)
from videoscope.providers.vision_worker_contract import (
    MAX_BATCH_IMAGE_BYTES,
    MAX_BATCH_IMAGE_PIXELS,
    MAX_DETECTIONS,
    MAX_IMAGE_BYTES,
    MAX_IMAGE_DIMENSION,
    MAX_IMAGE_PIXELS,
    MAX_IMAGES,
    MAX_PROBE_BYTES,
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    MAX_TEXTS,
    REVIEWED_SIGLIP_PROFILES,
    RFDETR_COMPUTE_BACKEND,
    RFDETR_COMPUTE_DTYPE,
    RFDETR_SMALL_CHECKPOINT_SHA256,
    SIGLIP_ARTIFACT_MANIFEST,
    SIGLIP_COMPUTE_BACKEND,
    SIGLIP_COMPUTE_DTYPE,
    SIGLIP_MAX_TEXT_TOKENS,
    VISION_WORKER_CORE_DISTRIBUTIONS,
    VISION_WORKER_LOCK_SHA256,
    VISION_WORKER_MINIMUM_MACOS_MAJOR,
    VISION_WORKER_PYTHON_VERSION,
    VISION_WORKER_RUNTIME_IDENTITY,
    VisionDetectRequest,
    VisionDetectionPayload,
    VisionDetectionResponse,
    VisionEmbedImagesRequest,
    VisionEmbeddingResponse,
    VisionEmbedTextsRequest,
    VisionHealthResponse,
    VisionImageItem,
    VisionSourceProbeRequest,
    VisionSourceProbeResponse,
    VisionVectorItem,
    VisionWorkerSpecification,
    identity_fields,
    siglip_profile_is_reviewed,
    worker_input_root_identity_from_metadata,
)


logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"^[A-Za-z0-9._~-]{32,256}$")
_HOST_RE = re.compile(r"^127\.0\.0\.1(?::([0-9]{1,5}))?$")
_SAFE_SUBDIRECTORY_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_HF_BLOB_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_LOCK_REQUIREMENT_RE = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9][A-Za-z0-9._+!-]*)(?: \\)?$"
)
_IMAGE_FORMATS = {
    ".jpeg": frozenset({"JPEG"}),
    ".jpg": frozenset({"JPEG"}),
    ".png": frozenset({"PNG"}),
    ".webp": frozenset({"WEBP"}),
}
_RFDETR_CLASSES = {
    "rfdetr-nano": "RFDETRNano",
    "rfdetr-small": "RFDETRSmall",
    "rfdetr-medium": "RFDETRMedium",
    "rfdetr-large": "RFDETRLarge",
}
_SIGLIP_IMAGE_INFERENCE_BATCH_SIZE = 8
_SIGLIP_TEXT_INFERENCE_BATCH_SIZE = 32
_DERIVED_INPUT_SUBDIRECTORIES = ("visual-index", "thumbnails", "tmp")
_RFDETR_CONSTRUCTION_LOCK = Lock()


@dataclass(frozen=True, slots=True)
class _SourceFingerprint:
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True, slots=True)
class _PreparedImage:
    item_id: str
    relative_path: str
    inference_copy: Path
    fingerprint: _SourceFingerprint
    sha256: str
    size_bytes: int
    width: int
    height: int


class _SourceIdentityMismatch(ValueError):
    pass


class _SourceChanged(ValueError):
    pass


class _RetainedInputRoot:
    """A process-lifetime root descriptor opened without following any symlink."""

    def __init__(self, descriptor: int) -> None:
        self._descriptor: int | None = descriptor
        self._lock = Lock()

    @classmethod
    def open(cls, path: Path) -> _RetainedInputRoot:
        nofollow = getattr(os, "O_NOFOLLOW", None)
        if nofollow is None:
            raise ValueError("Vision worker requires O_NOFOLLOW input containment")
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | nofollow
        )
        absolute = Path(os.path.abspath(path))
        descriptor: int | None = None
        try:
            descriptor = os.open(absolute.anchor, flags)
            for part in absolute.parts[1:]:
                next_descriptor = os.open(part, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = next_descriptor
            metadata = os.fstat(descriptor)
            if not stat.S_ISDIR(metadata.st_mode):
                raise ValueError("Vision worker input root must be a directory")
            retained = cls(descriptor)
            descriptor = None
            return retained
        except (OSError, ValueError) as error:
            raise ValueError("Vision worker input root must be a safe directory") from error
        finally:
            if descriptor is not None:
                os.close(descriptor)

    @property
    def descriptor(self) -> int:
        with self._lock:
            if self._descriptor is None:
                raise RuntimeError("Vision worker input root is closed")
            return self._descriptor

    def duplicate(self) -> int:
        with self._lock:
            if self._descriptor is None:
                raise OSError("Vision worker input root is closed")
            return os.dup(self._descriptor)

    def close(self) -> None:
        with self._lock:
            descriptor = self._descriptor
            self._descriptor = None
        if descriptor is not None:
            os.close(descriptor)


class VisionWorkerRuntime(Protocol):
    specification: VisionWorkerSpecification

    @property
    def available(self) -> bool: ...

    @property
    def siglip_loaded(self) -> bool: ...

    @property
    def detector_loaded(self) -> bool: ...

    def embed_images(self, sources: tuple[Path, ...]) -> list[list[float]]: ...

    def embed_texts(self, texts: tuple[str, ...]) -> list[list[float]]: ...

    def detect(
        self,
        source: Path,
        minimum_confidence: float,
    ) -> list[dict[str, object]]: ...


def _fingerprint(metadata: os.stat_result) -> _SourceFingerprint:
    return _SourceFingerprint(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        size=metadata.st_size,
        modified_ns=metadata.st_mtime_ns,
        changed_ns=metadata.st_ctime_ns,
    )


def _open_contained_source(input_root: _RetainedInputRoot, relative_path: str) -> int:
    """Open a file beneath input_root without following any path-component symlink."""
    relative = PurePosixPath(relative_path)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os,
        "O_NOFOLLOW",
        0,
    )
    current_descriptor = input_root.duplicate()
    try:
        for part in relative.parts[:-1]:
            next_descriptor = os.open(
                part,
                directory_flags,
                dir_fd=current_descriptor,
            )
            os.close(current_descriptor)
            current_descriptor = next_descriptor
        return os.open(
            relative.parts[-1],
            file_flags,
            dir_fd=current_descriptor,
        )
    finally:
        os.close(current_descriptor)


def _read_probe_source(
    input_root: _RetainedInputRoot,
    request: VisionSourceProbeRequest,
    *,
    allowed_subdirectories: frozenset[str] | None,
    allowed_subdirectory_prefixes: tuple[str, ...],
) -> tuple[str, int]:
    if not _input_subdirectory_is_allowed(
        request.relative_path,
        allowed_subdirectories=allowed_subdirectories,
        allowed_prefixes=allowed_subdirectory_prefixes,
    ):
        raise ValueError("input is outside allowlisted worker subdirectories")
    try:
        descriptor = _open_contained_source(input_root, request.relative_path)
    except OSError as error:
        raise ValueError("probe source cannot be opened safely") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > MAX_PROBE_BYTES
        ):
            raise ValueError("probe source must be a bounded single-link file")
        digest = sha256()
        size_bytes = 0
        while True:
            chunk = os.read(descriptor, min(4096, MAX_PROBE_BYTES + 1 - size_bytes))
            if not chunk:
                break
            size_bytes += len(chunk)
            if size_bytes > MAX_PROBE_BYTES:
                raise ValueError("probe source exceeds its bound")
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    value = digest.hexdigest()
    if _fingerprint(before) != _fingerprint(after) or size_bytes != after.st_size:
        raise ValueError("probe source changed while it was read")
    if (
        size_bytes != request.expected_size_bytes
        or not hmac.compare_digest(value, request.expected_sha256)
    ):
        raise _SourceIdentityMismatch("probe source identity does not match")
    return value, size_bytes


def _copy_and_hash_regular_file(
    source_descriptor: int,
    destination: Path,
    *,
    max_bytes: int,
) -> tuple[str, int, _SourceFingerprint]:
    try:
        before = os.fstat(source_descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("input image must be a single-link regular file")
        if before.st_size <= 0:
            raise ValueError("input image must not be empty")
        if before.st_size > max_bytes:
            raise OverflowError("input image is too large")
        digest = sha256()
        size_bytes = 0
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        destination_flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
        )
        destination_descriptor = os.open(destination, destination_flags, 0o600)
        try:
            with os.fdopen(destination_descriptor, "wb", closefd=False) as output:
                while True:
                    chunk = os.read(source_descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    size_bytes += len(chunk)
                    if size_bytes > max_bytes:
                        raise OverflowError("input image is too large")
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
        finally:
            os.close(destination_descriptor)
        after = os.fstat(source_descriptor)
    finally:
        os.close(source_descriptor)
    if _fingerprint(before) != _fingerprint(after) or size_bytes != after.st_size:
        raise _SourceChanged("input changed while it was copied")
    return digest.hexdigest(), size_bytes, _fingerprint(after)


def _hash_regular_file(
    input_root: _RetainedInputRoot,
    relative_path: str,
    *,
    max_bytes: int,
) -> tuple[str, int, _SourceFingerprint]:
    try:
        descriptor = _open_contained_source(input_root, relative_path)
    except OSError as error:
        raise _SourceChanged("input cannot be reopened safely") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > max_bytes
        ):
            raise _SourceChanged("input metadata changed")
        digest = sha256()
        size_bytes = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            size_bytes += len(chunk)
            if size_bytes > max_bytes:
                raise _SourceChanged("input size changed")
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _fingerprint(before) != _fingerprint(after) or size_bytes != after.st_size:
        raise _SourceChanged("input changed while it was checked")
    return digest.hexdigest(), size_bytes, _fingerprint(after)


def _inspect_image(
    path: Path,
    *,
    source_suffix: str,
    max_dimension: int,
    max_pixels: int,
) -> tuple[int, int]:
    try:
        from PIL import Image

        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                width, height = image.size
                image_format = image.format
                if (
                    type(width) is not int
                    or type(height) is not int
                    or width <= 0
                    or height <= 0
                    or width > max_dimension
                    or height > max_dimension
                    or width * height > max_pixels
                ):
                    raise OverflowError("image dimensions exceed configured limits")
                if image_format not in _IMAGE_FORMATS[source_suffix.casefold()]:
                    raise ValueError("image content does not match its extension")
                image.verify()
    except OverflowError:
        raise
    except Exception as error:
        raise ValueError("input is not a valid supported image") from error
    return width, height


def _prepare_image(
    *,
    input_root: _RetainedInputRoot,
    item: VisionImageItem,
    destination: Path,
    allowed_subdirectories: frozenset[str] | None,
    allowed_subdirectory_prefixes: tuple[str, ...],
    max_image_bytes: int,
    max_image_dimension: int,
    max_image_pixels: int,
) -> _PreparedImage:
    relative = PurePosixPath(item.relative_path)
    if not _input_subdirectory_is_allowed(
        item.relative_path,
        allowed_subdirectories=allowed_subdirectories,
        allowed_prefixes=allowed_subdirectory_prefixes,
    ):
        raise ValueError("input is outside allowlisted worker subdirectories")
    source_suffix = relative.suffix.casefold()
    if source_suffix not in _IMAGE_FORMATS:
        raise ValueError("input image extension is not supported")
    try:
        descriptor = _open_contained_source(input_root, item.relative_path)
    except OSError as error:
        raise ValueError("input image cannot be opened safely") from error
    digest, size_bytes, fingerprint = _copy_and_hash_regular_file(
        descriptor,
        destination,
        max_bytes=max_image_bytes,
    )
    if size_bytes != item.expected_size_bytes or not hmac.compare_digest(
        digest,
        item.expected_sha256,
    ):
        raise _SourceIdentityMismatch("source content identity does not match")
    width, height = _inspect_image(
        destination,
        source_suffix=source_suffix,
        max_dimension=max_image_dimension,
        max_pixels=max_image_pixels,
    )
    return _PreparedImage(
        item_id=item.item_id,
        relative_path=item.relative_path,
        inference_copy=destination,
        fingerprint=fingerprint,
        sha256=digest,
        size_bytes=size_bytes,
        width=width,
        height=height,
    )


def _verify_source_after_inference(
    prepared: _PreparedImage,
    *,
    input_root: _RetainedInputRoot,
    max_image_bytes: int,
) -> None:
    digest, size_bytes, fingerprint = _hash_regular_file(
        input_root,
        prepared.relative_path,
        max_bytes=max_image_bytes,
    )
    if (
        fingerprint != prepared.fingerprint
        or size_bytes != prepared.size_bytes
        or not hmac.compare_digest(digest, prepared.sha256)
    ):
        raise _SourceChanged("input changed during inference")


class _WorkerBoundaryMiddleware:
    """Authenticate and reject invalid Host/body size before JSON parsing."""

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
    def _loopback(scope: Scope) -> bool:
        client = scope.get("client")
        if not client:
            return False
        try:
            return ip_address(str(client[0])).is_loopback
        except ValueError:
            return False

    @staticmethod
    def _valid_host(scope: Scope) -> bool:
        values = [
            value
            for name, value in scope.get("headers", [])
            if name.lower() == b"host"
        ]
        if len(values) != 1:
            return False
        try:
            host = values[0].decode("ascii")
        except UnicodeError:
            return False
        match = _HOST_RE.fullmatch(host)
        if match is None:
            return False
        return match.group(1) is None or 1 <= int(match.group(1)) <= 65_535

    def _authorized(self, scope: Scope) -> bool:
        values = [
            value
            for name, value in scope.get("headers", [])
            if name.lower() == b"authorization"
        ]
        return len(values) == 1 and secrets.compare_digest(
            values[0],
            self.expected_authorization,
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        protected = scope["type"] == "http" and str(scope.get("path", "")).startswith(
            "/v1/"
        )
        if not protected:
            await self.app(scope, receive, send)
            return
        if not self._valid_host(scope):
            await self._error(scope, receive, send, 400, "Invalid Host")
            return
        if not self._loopback(scope):
            await self._error(scope, receive, send, 403, "Loopback access only")
            return
        if not self._authorized(scope):
            await self._error(scope, receive, send, 401, "Unauthorized")
            return
        if scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return

        header_pairs = scope.get("headers", [])
        content_types = [
            value for name, value in header_pairs if name.lower() == b"content-type"
        ]
        if len(content_types) != 1 or content_types[0].split(b";", 1)[0].strip().lower() != b"application/json":
            await self._error(scope, receive, send, 415, "JSON body required")
            return
        raw_lengths = [
            value for name, value in header_pairs if name.lower() == b"content-length"
        ]
        if len(raw_lengths) > 1:
            await self._error(scope, receive, send, 400, "Invalid Content-Length")
            return
        if raw_lengths:
            try:
                declared = int(raw_lengths[0])
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
        try:
            def reject_constant(_value: str) -> None:
                raise ValueError("non-finite JSON value")

            def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
                payload: dict[str, object] = {}
                for name, value in pairs:
                    if name in payload:
                        raise ValueError("duplicate JSON object key")
                    payload[name] = value
                return payload

            json.loads(
                body,
                parse_constant=reject_constant,
                object_pairs_hook=unique_object,
            )
        except (UnicodeError, ValueError, TypeError):
            await self._error(scope, receive, send, 400, "Invalid JSON body")
            return
        delivered = False

        async def replay_receive() -> Message:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        await self.app(scope, replay_receive, send)


def _request_identity_matches(
    runtime: VisionWorkerRuntime,
    request: object,
    *,
    input_root_identity: str,
) -> bool:
    specification = runtime.specification
    return all(
        (
            getattr(request, "runtime_identity", None) == specification.runtime_identity,
            getattr(request, "specification_hash", None) == specification.identity,
            getattr(request, "siglip_specification_hash", None)
            == specification.siglip_identity,
            getattr(request, "detector_specification_hash", None)
            == specification.detector_identity,
            getattr(request, "siglip_model_identity", None)
            == specification.siglip_model_identity,
            getattr(request, "detector_model_identity", None)
            == specification.detector_model_identity,
            getattr(request, "input_root_identity", None) == input_root_identity,
        )
    )


def _validated_allowed_subdirectories(
    values: tuple[str, ...] | None,
) -> frozenset[str] | None:
    if values is None:
        return None
    if not values or len(values) > 16:
        raise ValueError("Vision worker input subdirectory allowlist is invalid")
    normalized = frozenset(values)
    if len(normalized) != len(values) or any(
        type(value) is not str
        or len(value) > 256
        or PurePosixPath(value).is_absolute()
        or PurePosixPath(value).as_posix() != value
        or not 1 <= len(PurePosixPath(value).parts) <= 4
        or any(
            _SAFE_SUBDIRECTORY_RE.fullmatch(part) is None
            or part in {".", ".."}
            for part in PurePosixPath(value).parts
        )
        for value in values
    ):
        raise ValueError("Vision worker input subdirectory allowlist is invalid")
    return normalized


def _validated_allowed_subdirectory_prefixes(
    values: tuple[str, ...] | None,
) -> tuple[str, ...]:
    if values is None:
        return ()
    if not values or len(values) > 8 or len(set(values)) != len(values):
        raise ValueError("Vision worker input subdirectory prefix allowlist is invalid")
    if any(
        not value
        or len(value) > 48
        or _SAFE_SUBDIRECTORY_RE.fullmatch(value) is None
        or value in {".", ".."}
        for value in values
    ):
        raise ValueError("Vision worker input subdirectory prefix allowlist is invalid")
    return values


def _input_subdirectory_is_allowed(
    relative_path: str,
    *,
    allowed_subdirectories: frozenset[str] | None,
    allowed_prefixes: tuple[str, ...],
) -> bool:
    if allowed_subdirectories is None and not allowed_prefixes:
        return True
    first = PurePosixPath(relative_path).parts[0]
    return (
        allowed_subdirectories is not None
        and any(
            relative_path == allowed
            or relative_path.startswith(f"{allowed}/")
            for allowed in allowed_subdirectories
        )
    ) or any(first.startswith(prefix) for prefix in allowed_prefixes)


def create_vision_worker_app(
    *,
    runtime: VisionWorkerRuntime,
    input_root: Path,
    api_key: str,
    allowed_input_subdirectories: tuple[str, ...] | None = None,
    allowed_input_subdirectory_prefixes: tuple[str, ...] | None = None,
    max_request_bytes: int = MAX_REQUEST_BYTES,
    max_image_bytes: int = MAX_IMAGE_BYTES,
    max_image_dimension: int = MAX_IMAGE_DIMENSION,
    max_image_pixels: int = MAX_IMAGE_PIXELS,
    max_batch_image_bytes: int = MAX_BATCH_IMAGE_BYTES,
    max_batch_image_pixels: int = MAX_BATCH_IMAGE_PIXELS,
    max_concurrency: int = 1,
) -> FastAPI:
    if _TOKEN_RE.fullmatch(api_key) is None:
        raise ValueError("Vision worker API key must be 32-256 URL-safe characters")
    if not 0 < max_request_bytes <= MAX_REQUEST_BYTES:
        raise ValueError("Vision worker request limit is invalid")
    if not 0 < max_image_bytes <= MAX_IMAGE_BYTES:
        raise ValueError("Vision worker image byte limit is invalid")
    if not 0 < max_image_dimension <= MAX_IMAGE_DIMENSION:
        raise ValueError("Vision worker image dimension limit is invalid")
    if not 0 < max_image_pixels <= MAX_IMAGE_PIXELS:
        raise ValueError("Vision worker image pixel limit is invalid")
    if not max_image_bytes <= max_batch_image_bytes <= MAX_BATCH_IMAGE_BYTES:
        raise ValueError("Vision worker batch byte limit is invalid")
    if not max_image_pixels <= max_batch_image_pixels <= MAX_BATCH_IMAGE_PIXELS:
        raise ValueError("Vision worker batch pixel limit is invalid")
    if not 1 <= max_concurrency <= 4:
        raise ValueError("Vision worker concurrency must be between 1 and 4")
    allowed = _validated_allowed_subdirectories(allowed_input_subdirectories)
    allowed_prefixes = _validated_allowed_subdirectory_prefixes(
        allowed_input_subdirectory_prefixes
    )
    specification = runtime.specification
    if specification.runtime_identity != VISION_WORKER_RUNTIME_IDENTITY:
        raise ValueError("Vision worker runtime and specification identities disagree")
    retained_root = _RetainedInputRoot.open(input_root)
    input_root_identity = worker_input_root_identity_from_metadata(
        os.fstat(retained_root.descriptor)
    )
    exact_identity = identity_fields(
        specification,
        input_root_identity=input_root_identity,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            retained_root.close()

    app = FastAPI(
        title="VideoScope Vision Worker",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.vision_input_root = retained_root

    @app.exception_handler(RequestValidationError)
    async def invalid_request(
        _request: object,
        _error: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            {"detail": "Invalid Vision worker request"},
            status_code=422,
        )

    capacity = BoundedSemaphore(max_concurrency)

    @app.get("/v1/health", response_model=VisionHealthResponse)
    def health() -> VisionHealthResponse:
        return VisionHealthResponse(
            **exact_identity,
            status="ok" if runtime.available else "unavailable",
            siglip_loaded=runtime.siglip_loaded,
            detector_loaded=runtime.detector_loaded,
            operations=["probe", "embed_images", "embed_texts", "detect"],
            embedding_dimensions=specification.embedding_dimensions,
            max_images=MAX_IMAGES,
            max_texts=MAX_TEXTS,
            max_image_bytes=max_image_bytes,
            max_image_dimension=max_image_dimension,
            max_image_pixels=max_image_pixels,
            max_batch_image_bytes=max_batch_image_bytes,
            max_batch_image_pixels=max_batch_image_pixels,
            max_detections=MAX_DETECTIONS,
            max_concurrency=max_concurrency,
        )

    @app.post("/v1/probe", response_model=VisionSourceProbeResponse)
    def probe(request: VisionSourceProbeRequest) -> VisionSourceProbeResponse:
        if not _request_identity_matches(
            runtime,
            request,
            input_root_identity=input_root_identity,
        ):
            raise HTTPException(409, "Vision worker identity does not match")
        try:
            digest, size_bytes = _read_probe_source(
                retained_root,
                request,
                allowed_subdirectories=allowed,
                allowed_subdirectory_prefixes=allowed_prefixes,
            )
        except _SourceIdentityMismatch as error:
            raise HTTPException(
                409,
                "Vision worker source identity does not match",
            ) from error
        except (OSError, ValueError) as error:
            raise HTTPException(400, "Invalid Vision worker input") from error
        return VisionSourceProbeResponse(
            **exact_identity,
            request_id=request.request_id,
            source_sha256=digest,
            source_size_bytes=size_bytes,
        )

    async def embed_images_impl(
        request: VisionEmbedImagesRequest,
    ) -> VisionEmbeddingResponse:
        if not _request_identity_matches(
            runtime,
            request,
            input_root_identity=input_root_identity,
        ):
            raise HTTPException(409, "Vision worker identity does not match")
        if sum(item.expected_size_bytes for item in request.items) > max_batch_image_bytes:
            raise HTTPException(413, "Vision worker input exceeds configured limits")
        if not capacity.acquire(blocking=False):
            raise HTTPException(429, "Vision worker is busy")
        try:
            with TemporaryDirectory(prefix="videoscope-vision-") as temporary:
                temporary_root = Path(temporary)
                prepared: list[_PreparedImage] = []
                try:
                    for index, item in enumerate(request.items):
                        prepared.append(
                            _prepare_image(
                                input_root=retained_root,
                                item=item,
                                destination=(
                                    temporary_root
                                    / f"image-{index:04d}{Path(item.relative_path).suffix.casefold()}"
                                ),
                                allowed_subdirectories=allowed,
                                allowed_subdirectory_prefixes=allowed_prefixes,
                                max_image_bytes=max_image_bytes,
                                max_image_dimension=max_image_dimension,
                                max_image_pixels=max_image_pixels,
                            )
                        )
                        if sum(image.width * image.height for image in prepared) > max_batch_image_pixels:
                            raise OverflowError("batch pixels exceed configured limit")
                except _SourceIdentityMismatch as error:
                    raise HTTPException(409, "Vision worker source identity does not match") from error
                except OverflowError as error:
                    raise HTTPException(413, "Vision worker input exceeds configured limits") from error
                except (_SourceChanged, OSError, ValueError) as error:
                    raise HTTPException(400, "Invalid Vision worker input") from error
                try:
                    raw_vectors = await run_in_threadpool(
                        runtime.embed_images,
                        tuple(image.inference_copy for image in prepared),
                    )
                    if len(raw_vectors) != len(prepared):
                        raise ValueError("runtime returned wrong vector count")
                    response = VisionEmbeddingResponse(
                        **exact_identity,
                        request_id=request.request_id,
                        embedding_dimensions=specification.embedding_dimensions,
                        items=[
                            VisionVectorItem(item_id=item.item_id, vector=vector)
                            for item, vector in zip(request.items, raw_vectors, strict=True)
                        ],
                    )
                    if len(response.model_dump_json().encode("utf-8")) > MAX_RESPONSE_BYTES:
                        raise ValueError("response exceeds configured limit")
                except Exception as error:
                    logger.exception("Vision worker image embedding failed")
                    raise HTTPException(503, "Vision worker image embedding failed") from error
                try:
                    for image in prepared:
                        _verify_source_after_inference(
                            image,
                            input_root=retained_root,
                            max_image_bytes=max_image_bytes,
                        )
                except (_SourceChanged, OSError, ValueError) as error:
                    raise HTTPException(
                        409,
                        "Vision worker input changed during inference",
                    ) from error
                return response
        finally:
            capacity.release()

    @app.post("/v1/embed/images", response_model=VisionEmbeddingResponse)
    async def embed_images(request: VisionEmbedImagesRequest) -> VisionEmbeddingResponse:
        return await embed_images_impl(request)

    @app.post("/v1/embed/texts", response_model=VisionEmbeddingResponse)
    async def embed_texts(request: VisionEmbedTextsRequest) -> VisionEmbeddingResponse:
        if not _request_identity_matches(
            runtime,
            request,
            input_root_identity=input_root_identity,
        ):
            raise HTTPException(409, "Vision worker identity does not match")
        if not capacity.acquire(blocking=False):
            raise HTTPException(429, "Vision worker is busy")
        try:
            try:
                raw_vectors = await run_in_threadpool(
                    runtime.embed_texts,
                    tuple(item.text for item in request.items),
                )
                if len(raw_vectors) != len(request.items):
                    raise ValueError("runtime returned wrong vector count")
                response = VisionEmbeddingResponse(
                    **exact_identity,
                    request_id=request.request_id,
                    embedding_dimensions=specification.embedding_dimensions,
                    items=[
                        VisionVectorItem(item_id=item.item_id, vector=vector)
                        for item, vector in zip(request.items, raw_vectors, strict=True)
                    ],
                )
                if len(response.model_dump_json().encode("utf-8")) > MAX_RESPONSE_BYTES:
                    raise ValueError("response exceeds configured limit")
                return response
            except Exception as error:
                logger.exception("Vision worker text embedding failed")
                raise HTTPException(503, "Vision worker text embedding failed") from error
        finally:
            capacity.release()

    @app.post("/v1/detect", response_model=VisionDetectionResponse)
    async def detect(request: VisionDetectRequest) -> VisionDetectionResponse:
        if not _request_identity_matches(
            runtime,
            request,
            input_root_identity=input_root_identity,
        ):
            raise HTTPException(409, "Vision worker identity does not match")
        if not math.isclose(
            request.minimum_confidence,
            specification.minimum_confidence,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise HTTPException(409, "Vision worker detector threshold does not match")
        if not capacity.acquire(blocking=False):
            raise HTTPException(429, "Vision worker is busy")
        try:
            with TemporaryDirectory(prefix="videoscope-vision-") as temporary:
                try:
                    prepared = _prepare_image(
                        input_root=retained_root,
                        item=request.source,
                        destination=(
                            Path(temporary)
                            / f"detect{Path(request.source.relative_path).suffix.casefold()}"
                        ),
                        allowed_subdirectories=allowed,
                        allowed_subdirectory_prefixes=allowed_prefixes,
                        max_image_bytes=max_image_bytes,
                        max_image_dimension=max_image_dimension,
                        max_image_pixels=max_image_pixels,
                    )
                except _SourceIdentityMismatch as error:
                    raise HTTPException(409, "Vision worker source identity does not match") from error
                except OverflowError as error:
                    raise HTTPException(413, "Vision worker input exceeds configured limits") from error
                except (_SourceChanged, OSError, ValueError) as error:
                    raise HTTPException(400, "Invalid Vision worker input") from error
                try:
                    raw_detections = await run_in_threadpool(
                        runtime.detect,
                        prepared.inference_copy,
                        request.minimum_confidence,
                    )
                    if len(raw_detections) > MAX_DETECTIONS:
                        raise ValueError("runtime returned too many detections")
                    response = VisionDetectionResponse(
                        **exact_identity,
                        request_id=request.request_id,
                        source_id=request.source.item_id,
                        image_width=prepared.width,
                        image_height=prepared.height,
                        detections=[
                            VisionDetectionPayload.model_validate(detection)
                            for detection in raw_detections
                        ],
                    )
                except Exception as error:
                    logger.exception("Vision worker detection failed")
                    raise HTTPException(503, "Vision worker detection failed") from error
                try:
                    _verify_source_after_inference(
                        prepared,
                        input_root=retained_root,
                        max_image_bytes=max_image_bytes,
                    )
                except (_SourceChanged, OSError, ValueError) as error:
                    raise HTTPException(
                        409,
                        "Vision worker input changed during inference",
                    ) from error
                return response
        finally:
            capacity.release()

    app.add_middleware(
        _WorkerBoundaryMiddleware,
        api_key=api_key,
        max_request_bytes=max_request_bytes,
    )
    return app


def _file_sha256(path: Path) -> str:
    digest = sha256()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError("artifact is not a single-link regular file")
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _fingerprint(metadata) != _fingerprint(after):
        raise ValueError("artifact changed while hashing")
    return digest.hexdigest()


def _read_bounded_regular_file(path: Path, *, max_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os,
        "O_NOFOLLOW",
        0,
    )
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > max_bytes
        ):
            raise ValueError("artifact is not a bounded single-link regular file")
        body = bytearray()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            body.extend(chunk)
            if len(body) > max_bytes:
                raise ValueError("artifact exceeds its configured size limit")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _fingerprint(before) != _fingerprint(after) or len(body) != after.st_size:
        raise ValueError("artifact changed while it was read")
    return bytes(body)


def _locked_distribution_versions(lock_text: str) -> dict[str, str]:
    versions: dict[str, str] = {}
    for line in lock_text.splitlines():
        if not line or line[0].isspace() or line.startswith("#"):
            continue
        match = _LOCK_REQUIREMENT_RE.fullmatch(line)
        if match is None:
            raise ValueError("vision worker lock contains an unsupported requirement")
        name, version = match.groups()
        normalized = re.sub(r"[-_.]+", "-", name).casefold()
        if normalized in versions:
            raise ValueError("vision worker lock contains duplicate requirements")
        versions[normalized] = version
    if not versions:
        raise ValueError("vision worker lock contains no requirements")
    return versions


def vision_worker_environment_is_exact(
    *,
    lock_path: Path | None = None,
    expected_lock_sha256: str = VISION_WORKER_LOCK_SHA256,
    python_version: tuple[int, int, int] | None = None,
    system: str | None = None,
    machine: str | None = None,
    macos_version: tuple[int, ...] | None = None,
    distribution_version: Any | None = None,
) -> bool:
    current_lock = lock_path or (
        Path(__file__).resolve().parents[4] / "workers" / "vision" / "requirements.lock"
    )
    resolve_version = distribution_version or importlib_metadata.version
    if macos_version is None:
        try:
            macos_version = tuple(
                int(part) for part in platform.mac_ver()[0].split(".") if part
            )
        except ValueError:
            return False
    if (
        (python_version or tuple(sys.version_info[:3])) != VISION_WORKER_PYTHON_VERSION
        or (system or platform.system()).casefold() != "darwin"
        or (machine or platform.machine()).casefold() not in {"arm64", "aarch64"}
        or not macos_version
        or type(macos_version[0]) is not int
        or macos_version[0] < VISION_WORKER_MINIMUM_MACOS_MAJOR
        or not current_lock.is_file()
        or current_lock.is_symlink()
    ):
        return False
    try:
        lock_bytes = _read_bounded_regular_file(
            current_lock,
            max_bytes=4 * 1024 * 1024,
        )
        if not hmac.compare_digest(
            sha256(lock_bytes).hexdigest(),
            expected_lock_sha256,
        ):
            return False
        locked_versions = _locked_distribution_versions(lock_bytes.decode("utf-8"))
        if any(
            locked_versions.get(re.sub(r"[-_.]+", "-", distribution).casefold())
            != expected
            for distribution, expected in VISION_WORKER_CORE_DISTRIBUTIONS.items()
        ):
            return False
        return all(
            resolve_version(distribution) == expected
            for distribution, expected in locked_versions.items()
        )
    except (
        importlib_metadata.PackageNotFoundError,
        UnicodeError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ):
        return False


def _mps_float32_environment_is_exact(*, torch_module: Any | None = None) -> bool:
    try:
        if torch_module is None:
            import torch as torch_module

        mps = torch_module.backends.mps
        if not mps.is_built() or not mps.is_available():
            return False
        probe = torch_module.empty(
            (1,),
            device="mps",
            dtype=torch_module.float32,
        )
        return (
            probe.device.type == "mps"
            and probe.dtype == torch_module.float32
        )
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
        return False


def _siglip_compute_environment_is_exact(*, torch_module: Any | None = None) -> bool:
    return (
        SIGLIP_COMPUTE_BACKEND == "mps"
        and SIGLIP_COMPUTE_DTYPE == "float32"
        and _mps_float32_environment_is_exact(torch_module=torch_module)
    )


def _rfdetr_compute_environment_is_exact(*, torch_module: Any | None = None) -> bool:
    return (
        RFDETR_COMPUTE_BACKEND == "mps"
        and RFDETR_COMPUTE_DTYPE == "float32"
        and _mps_float32_environment_is_exact(torch_module=torch_module)
    )


@lru_cache(maxsize=64)
def _hf_blob_content_is_exact(
    path_value: str,
    oid: str,
    fingerprint: _SourceFingerprint,
) -> bool:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os,
        "O_NOFOLLOW",
        0,
    )
    try:
        descriptor = os.open(path_value, flags)
    except OSError:
        return False
    try:
        before = os.fstat(descriptor)
        if (
            _fingerprint(before) != fingerprint
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
        ):
            return False
        if len(oid) == 64:
            digest = sha256()
        else:
            # Hugging Face stores small Git-managed files under their Git blob OID.
            digest = sha1(usedforsecurity=False)
            digest.update(f"blob {before.st_size}\0".encode("ascii"))
        size_bytes = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            size_bytes += len(chunk)
            if size_bytes > fingerprint.size:
                return False
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    return (
        _fingerprint(after) == fingerprint
        and size_bytes == fingerprint.size
        and hmac.compare_digest(digest.hexdigest(), oid)
    )


def _siglip_snapshot_is_available(model: str, revision: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        artifacts = SIGLIP_ARTIFACT_MANIFEST.get((model, revision))
        if artifacts is None:
            return False
        expected_repo_name = f"models--{model.replace('/', '--')}"
        for filename, oid, size_bytes in artifacts:
            cached = try_to_load_from_cache(model, filename, revision=revision)
            if not isinstance(cached, str):
                return False
            path = Path(cached)
            resolved = path.resolve(strict=True)
            snapshot_root = path.parent.parent
            repo_root = snapshot_root.parent
            expected_blobs = (repo_root / "blobs").resolve(strict=True)
            metadata = resolved.stat()
            if (
                not path.is_symlink()
                or path.parent.name != revision
                or snapshot_root.name != "snapshots"
                or repo_root.name != expected_repo_name
                or path.name != filename
                or resolved.parent != expected_blobs
                or resolved.name != oid
                or _HF_BLOB_RE.fullmatch(oid) is None
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_size != size_bytes
                or not _hf_blob_content_is_exact(
                    str(resolved),
                    oid,
                    _fingerprint(metadata),
                )
            ):
                return False
        return True
    except (OSError, TypeError, ValueError):
        return False


class LocalVisionWorkerRuntime:
    """Lazy, offline-only SigLIP/RF-DETR runtime with exact artifact identity."""

    def __init__(
        self,
        *,
        specification: VisionWorkerSpecification,
        detector_checkpoint: Path,
    ) -> None:
        if not siglip_profile_is_reviewed(specification):
            raise ValueError("SigLIP profile is not a reviewed immutable profile")
        checkpoint = Path(os.path.abspath(detector_checkpoint))
        if checkpoint.is_symlink():
            raise ValueError("RF-DETR checkpoint must not be a symlink")
        self.specification = specification
        self.detector_checkpoint = checkpoint
        self._siglip_model: Any | None = None
        self._siglip_processor: Any | None = None
        self._detector: Any | None = None
        self._siglip_load_lock = Lock()
        self._detector_load_lock = Lock()
        self._inference_lock = Lock()
        self._checkpoint_lock = Lock()
        self._checkpoint_cache: tuple[_SourceFingerprint, bool] | None = None

    @property
    def siglip_loaded(self) -> bool:
        return self._siglip_model is not None and self._siglip_processor is not None

    @property
    def detector_loaded(self) -> bool:
        return self._detector is not None

    def _checkpoint_is_exact(self) -> bool:
        try:
            metadata = self.detector_checkpoint.stat()
            fingerprint = _fingerprint(metadata)
        except OSError:
            return False
        if (
            self.detector_checkpoint.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size <= 0
        ):
            return False
        with self._checkpoint_lock:
            if self._checkpoint_cache is not None and self._checkpoint_cache[0] == fingerprint:
                return self._checkpoint_cache[1]
            try:
                exact = hmac.compare_digest(
                    _file_sha256(self.detector_checkpoint),
                    self.specification.detector_checkpoint_sha256,
                )
            except (OSError, ValueError):
                exact = False
            self._checkpoint_cache = (fingerprint, exact)
            return exact

    @property
    def available(self) -> bool:
        return (
            vision_worker_environment_is_exact()
            and _siglip_compute_environment_is_exact()
            and _rfdetr_compute_environment_is_exact()
            and siglip_profile_is_reviewed(self.specification)
            and _siglip_snapshot_is_available(
                self.specification.siglip_model,
                self.specification.siglip_revision,
            )
            and self._checkpoint_is_exact()
        )

    @staticmethod
    def _pooled(output: object) -> Any:
        pooled = getattr(output, "pooler_output", None)
        if pooled is not None:
            return pooled
        if isinstance(output, tuple) and len(output) > 1:
            return output[1]
        return output

    @staticmethod
    def _normalize(features: Any) -> Any:
        return features / features.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)

    @staticmethod
    def _to_siglip_compute(value: Any, torch_module: Any) -> Any:
        try:
            floating = value.is_floating_point()
            if type(floating) is not bool:
                raise TypeError("tensor floating-point state is invalid")
            options: dict[str, object] = {"device": SIGLIP_COMPUTE_BACKEND}
            if floating:
                options["dtype"] = torch_module.float32
            moved = value.to(**options)
            if moved.device.type != SIGLIP_COMPUTE_BACKEND:
                raise RuntimeError("SigLIP input did not move to exact compute backend")
            if floating and moved.dtype != torch_module.float32:
                raise RuntimeError("SigLIP input did not move to exact compute dtype")
            return moved
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            raise RuntimeError("SigLIP input compute identity does not match") from error

    def _load_siglip(self) -> tuple[Any, Any]:
        if self.siglip_loaded:
            return self._siglip_model, self._siglip_processor
        with self._siglip_load_lock:
            if self.siglip_loaded:
                return self._siglip_model, self._siglip_processor
            if not self.available:
                raise RuntimeError("Vision worker artifacts or runtime are unavailable")
            import torch
            from transformers import AutoModel, AutoProcessor

            if not _siglip_compute_environment_is_exact(torch_module=torch):
                raise RuntimeError("SigLIP requires exact MPS float32 compute")

            options: dict[str, object] = {
                "revision": self.specification.siglip_revision,
                "local_files_only": True,
                "trust_remote_code": False,
            }
            processor = AutoProcessor.from_pretrained(
                self.specification.siglip_model,
                **options,
            )
            model = AutoModel.from_pretrained(
                self.specification.siglip_model,
                **options,
            )
            raw_scale = float(model.logit_scale.detach().cpu().item())
            actual_scale = math.exp(raw_scale)
            actual_bias = float(model.logit_bias.detach().cpu().item())
            if not (
                math.isclose(
                    actual_scale,
                    self.specification.siglip_logit_scale,
                    rel_tol=0,
                    abs_tol=1e-6,
                )
                and math.isclose(
                    actual_bias,
                    self.specification.siglip_logit_bias,
                    rel_tol=0,
                    abs_tol=1e-6,
                )
            ):
                raise RuntimeError("SigLIP calibration identity does not match")
            if not _siglip_snapshot_is_available(
                self.specification.siglip_model,
                self.specification.siglip_revision,
            ):
                raise RuntimeError("SigLIP artifacts changed during loading")
            model.eval().to(
                device=SIGLIP_COMPUTE_BACKEND,
                dtype=torch.float32,
            )
            self._siglip_processor = processor
            self._siglip_model = model
            return model, processor

    def embed_images(self, sources: tuple[Path, ...]) -> list[list[float]]:
        import torch
        from PIL import Image

        model, processor = self._load_siglip()
        rows: list[list[float]] = []
        with self._inference_lock:
            for start in range(0, len(sources), _SIGLIP_IMAGE_INFERENCE_BATCH_SIZE):
                batch_sources = sources[start : start + _SIGLIP_IMAGE_INFERENCE_BATCH_SIZE]
                images: list[Any] = []
                try:
                    for source in batch_sources:
                        with Image.open(source) as image:
                            images.append(image.convert("RGB"))
                    inputs = processor(images=images, return_tensors="pt")
                    inputs = {
                        name: self._to_siglip_compute(value, torch)
                        for name, value in inputs.items()
                    }
                    with torch.inference_mode():
                        encoded = self._normalize(
                            self._pooled(model.get_image_features(**inputs))
                        )
                    if encoded.ndim != 2 or encoded.shape != (
                        len(batch_sources),
                        self.specification.embedding_dimensions,
                    ):
                        raise RuntimeError("SigLIP image vector shape does not match")
                    rows.extend(encoded.detach().float().cpu().tolist())
                finally:
                    for image in images:
                        image.close()
        return rows

    def embed_texts(self, texts: tuple[str, ...]) -> list[list[float]]:
        import torch

        model, processor = self._load_siglip()
        rows: list[list[float]] = []
        with self._inference_lock:
            for start in range(0, len(texts), _SIGLIP_TEXT_INFERENCE_BATCH_SIZE):
                batch_texts = texts[start : start + _SIGLIP_TEXT_INFERENCE_BATCH_SIZE]
                inputs = processor(
                    text=list(batch_texts),
                    max_length=SIGLIP_MAX_TEXT_TOKENS,
                    padding="max_length",
                    return_tensors="pt",
                    truncation=True,
                )
                inputs = {
                    name: self._to_siglip_compute(value, torch)
                    for name, value in inputs.items()
                }
                with torch.inference_mode():
                    encoded = self._normalize(
                        self._pooled(model.get_text_features(**inputs))
                    )
                if encoded.ndim != 2 or encoded.shape != (
                    len(batch_texts),
                    self.specification.embedding_dimensions,
                ):
                    raise RuntimeError("SigLIP text vector shape does not match")
                rows.extend(encoded.detach().float().cpu().tolist())
        return rows

    @staticmethod
    def _place_detector_on_exact_compute(
        detector: Any,
        torch_module: Any,
        *,
        move: bool,
    ) -> None:
        try:
            context = detector.model
            if context.device.type != RFDETR_COMPUTE_BACKEND:
                raise RuntimeError("RF-DETR context compute backend does not match")
            model = context.model
            if move:
                model = model.to(
                    device=RFDETR_COMPUTE_BACKEND,
                    dtype=torch_module.float32,
                )
                context.model = model
            parameter_count = 0
            for tensor in model.parameters():
                parameter_count += 1
                if tensor.device.type != RFDETR_COMPUTE_BACKEND:
                    raise RuntimeError("RF-DETR parameter backend does not match")
                if tensor.is_floating_point() and tensor.dtype != torch_module.float32:
                    raise RuntimeError("RF-DETR parameter dtype does not match")
            if parameter_count == 0:
                raise RuntimeError("RF-DETR model has no parameters")
            for tensor in model.buffers():
                if tensor.device.type != RFDETR_COMPUTE_BACKEND:
                    raise RuntimeError("RF-DETR buffer backend does not match")
                if tensor.is_floating_point() and tensor.dtype != torch_module.float32:
                    raise RuntimeError("RF-DETR buffer dtype does not match")
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            raise RuntimeError("RF-DETR compute identity does not match") from error

    def _load_detector(self) -> Any:
        import torch

        if self._detector is not None:
            if not _rfdetr_compute_environment_is_exact(torch_module=torch):
                raise RuntimeError("RF-DETR requires exact MPS float32 compute")
            self._place_detector_on_exact_compute(
                self._detector,
                torch,
                move=False,
            )
            return self._detector
        with self._detector_load_lock:
            if self._detector is not None:
                if not _rfdetr_compute_environment_is_exact(torch_module=torch):
                    raise RuntimeError("RF-DETR requires exact MPS float32 compute")
                self._place_detector_on_exact_compute(
                    self._detector,
                    torch,
                    move=False,
                )
                return self._detector
            if not self.available:
                raise RuntimeError("Vision worker artifacts or runtime are unavailable")
            if not _rfdetr_compute_environment_is_exact(torch_module=torch):
                raise RuntimeError("RF-DETR requires exact MPS float32 compute")
            with _RFDETR_CONSTRUCTION_LOCK:
                if not self._checkpoint_is_exact():
                    raise RuntimeError("RF-DETR checkpoint identity does not match")
                try:
                    before = _fingerprint(self.detector_checkpoint.stat())
                except OSError as error:
                    raise RuntimeError("RF-DETR checkpoint is unavailable") from error

                import rfdetr
                from rfdetr import detr as rfdetr_detr
                from rfdetr.models import weights as rfdetr_weights

                detector_class = getattr(
                    rfdetr,
                    _RFDETR_CLASSES[self.specification.detector_model_id],
                )
                original_detr_downloader = rfdetr_detr.download_pretrain_weights
                original_weights_downloader = rfdetr_weights.download_pretrain_weights
                with TemporaryDirectory(prefix="videoscope-rfdetr-") as temporary:
                    temporary_root = Path(temporary).resolve(strict=True)
                    private_checkpoint = temporary_root / f"{uuid4().hex}.pth"
                    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
                        os,
                        "O_NOFOLLOW",
                        0,
                    )
                    try:
                        descriptor = os.open(self.detector_checkpoint, flags)
                    except OSError as error:
                        raise RuntimeError(
                            "RF-DETR checkpoint cannot be copied safely"
                        ) from error
                    digest, size_bytes, copied_fingerprint = (
                        _copy_and_hash_regular_file(
                            descriptor,
                            private_checkpoint,
                            max_bytes=before.size,
                        )
                    )
                    if (
                        copied_fingerprint != before
                        or size_bytes != before.size
                        or not hmac.compare_digest(
                            digest,
                            self.specification.detector_checkpoint_sha256,
                        )
                    ):
                        raise RuntimeError("RF-DETR checkpoint changed while copying")
                    private_checkpoint.chmod(0o400)
                    private_before = _fingerprint(private_checkpoint.stat())

                    def reject_dependency_download(
                        pretrain_weights: str,
                        redownload: bool = False,
                        validate_md5: bool = True,
                    ) -> None:
                        del validate_md5
                        requested = Path(os.path.abspath(pretrain_weights))
                        if redownload or requested != private_checkpoint:
                            raise RuntimeError("RF-DETR model download is disabled")
                        if _fingerprint(requested.stat()) != private_before:
                            raise RuntimeError(
                                "RF-DETR private checkpoint identity does not match"
                            )

                    try:
                        rfdetr_detr.download_pretrain_weights = reject_dependency_download
                        rfdetr_weights.download_pretrain_weights = reject_dependency_download
                        detector = detector_class(
                            pretrain_weights=str(private_checkpoint),
                            device=RFDETR_COMPUTE_BACKEND,
                        )
                    finally:
                        rfdetr_detr.download_pretrain_weights = original_detr_downloader
                        rfdetr_weights.download_pretrain_weights = original_weights_downloader
                    private_after = _fingerprint(private_checkpoint.stat())
                    if (
                        private_before != private_after
                        or not hmac.compare_digest(
                            _file_sha256(private_checkpoint),
                            self.specification.detector_checkpoint_sha256,
                        )
                    ):
                        raise RuntimeError(
                            "RF-DETR private checkpoint changed during loading"
                        )

                try:
                    after = _fingerprint(self.detector_checkpoint.stat())
                except OSError as error:
                    raise RuntimeError("RF-DETR checkpoint changed during loading") from error
                with self._checkpoint_lock:
                    self._checkpoint_cache = None
                if before != after or not self._checkpoint_is_exact():
                    raise RuntimeError("RF-DETR checkpoint changed during loading")
                self._place_detector_on_exact_compute(
                    detector,
                    torch,
                    move=True,
                )
                self._detector = detector
                return detector

    def detect(
        self,
        source: Path,
        minimum_confidence: float,
    ) -> list[dict[str, object]]:
        if not math.isclose(
            minimum_confidence,
            self.specification.minimum_confidence,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise RuntimeError("RF-DETR threshold identity does not match")
        from rfdetr.assets.coco_classes import COCO_CLASSES
        from PIL import Image

        detector = self._load_detector()
        with self._inference_lock:
            with Image.open(source) as image:
                rgb_image = image.convert("RGB")
            try:
                detections = detector.predict(
                    rgb_image,
                    threshold=minimum_confidence,
                    include_source_image=False,
                )
            finally:
                rgb_image.close()
        if isinstance(detections, list):
            detections = detections[0] if detections else None
        if detections is None:
            return []
        class_names = getattr(detections, "data", {}).get("class_name")
        output: list[dict[str, object]] = []
        for index in range(min(len(detections), MAX_DETECTIONS + 1)):
            confidence = float(detections.confidence[index])
            class_id = int(detections.class_id[index])
            x1, y1, x2, y2 = map(float, detections.xyxy[index])
            label = class_names[index] if class_names is not None else None
            output.append(
                {
                    "label": str(label or COCO_CLASSES.get(class_id, f"class-{class_id}")),
                    "confidence": confidence,
                    "x": (x1 + x2) / 2,
                    "y": (y1 + y2) / 2,
                    "width": x2 - x1,
                    "height": y2 - y1,
                }
            )
        return output


class VisionWorkerSettings(BaseSettings):
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
            "VIDEOSCOPE_VISION_WORKER_API_KEY",
            "VISION_WORKER_API_KEY",
        ),
    )
    host: Literal["127.0.0.1"] = Field(
        default="127.0.0.1",
        validation_alias="VIDEOSCOPE_VISION_WORKER_HOST",
    )
    port: int = Field(
        default=8783,
        ge=1,
        le=65_535,
        validation_alias="VIDEOSCOPE_VISION_WORKER_PORT",
    )
    input_root: Path = Field(
        default=Path("data"),
        validation_alias="VIDEOSCOPE_VISION_WORKER_INPUT_ROOT",
    )
    product_data_subdirectory: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9._-]+$",
        validation_alias="VIDEOSCOPE_VISION_WORKER_PRODUCT_DATA_SUBDIRECTORY",
    )
    siglip_model: str = Field(
        default=SIGLIP_224_MODEL,
        validation_alias="VIDEOSCOPE_SIGLIP_MODEL",
    )
    detector_checkpoint: Path = Field(
        default=Path("data/models/rfdetr/rf-detr-small.pth"),
        validation_alias="VIDEOSCOPE_VISION_WORKER_RFDETR_CHECKPOINT",
    )
    detector_model_id: Literal[
        "rfdetr-nano",
        "rfdetr-small",
        "rfdetr-medium",
        "rfdetr-large",
    ] = Field(
        default="rfdetr-small",
        validation_alias=AliasChoices(
            "VIDEOSCOPE_VISION_DETECTOR_MODEL_ID",
            "VIDEOSCOPE_VISION_WORKER_DETECTOR_MODEL_ID",
        ),
    )
    detector_checkpoint_sha256: str = Field(
        default=RFDETR_SMALL_CHECKPOINT_SHA256,
        pattern=r"^[0-9a-f]{64}$",
        validation_alias=AliasChoices(
            "VIDEOSCOPE_VISION_DETECTOR_CHECKPOINT_SHA256",
            "VIDEOSCOPE_VISION_WORKER_DETECTOR_CHECKPOINT_SHA256",
            "VIDEOSCOPE_VISION_WORKER_RFDETR_SHA256",
        ),
    )
    minimum_confidence: float = Field(
        default=0.25,
        ge=0,
        le=1,
        validation_alias="VIDEOSCOPE_VISION_WORKER_MINIMUM_CONFIDENCE",
    )
    max_concurrency: int = Field(
        default=1,
        ge=1,
        le=4,
        validation_alias="VIDEOSCOPE_VISION_WORKER_MAX_CONCURRENCY",
    )

    @field_validator("siglip_model")
    @classmethod
    def validate_siglip_model(cls, value: str) -> str:
        revision = MODEL_REVISIONS.get(value)
        if (
            value not in {SIGLIP_224_MODEL, SIGLIP_384_MODEL}
            or revision is None
            or (value, revision) not in REVIEWED_SIGLIP_PROFILES
        ):
            raise ValueError("SigLIP model is not a reviewed local profile")
        return value

    @field_validator("product_data_subdirectory")
    @classmethod
    def validate_product_data_subdirectory(cls, value: str | None) -> str | None:
        if value in {".", ".."}:
            raise ValueError("Product data subdirectory is invalid")
        return value

    def allowed_input_subdirectories(self) -> tuple[str, ...]:
        if self.product_data_subdirectory is None:
            return _DERIVED_INPUT_SUBDIRECTORIES
        return tuple(
            f"{self.product_data_subdirectory}/{name}"
            for name in _DERIVED_INPUT_SUBDIRECTORIES
        )

    def specification(self) -> VisionWorkerSpecification:
        revision = MODEL_REVISIONS[self.siglip_model]
        dimensions, logit_scale, logit_bias = REVIEWED_SIGLIP_PROFILES[
            (self.siglip_model, revision)
        ]
        return VisionWorkerSpecification(
            siglip_model=self.siglip_model,
            siglip_revision=revision,
            embedding_dimensions=dimensions,
            detector_model_id=self.detector_model_id,
            detector_checkpoint_sha256=self.detector_checkpoint_sha256,
            siglip_logit_scale=logit_scale,
            siglip_logit_bias=logit_bias,
            minimum_confidence=self.minimum_confidence,
        )


def main() -> None:
    settings = VisionWorkerSettings()
    runtime = LocalVisionWorkerRuntime(
        specification=settings.specification(),
        detector_checkpoint=settings.detector_checkpoint,
    )
    app = create_vision_worker_app(
        runtime=runtime,
        input_root=settings.input_root,
        api_key=settings.api_key,
        allowed_input_subdirectories=settings.allowed_input_subdirectories(),
        allowed_input_subdirectory_prefixes=("benchmark-environment-",),
        max_concurrency=settings.max_concurrency,
    )
    import uvicorn

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        access_log=False,
    )


if __name__ == "__main__":
    main()
