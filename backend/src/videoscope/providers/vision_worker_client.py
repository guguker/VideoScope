from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from ipaddress import ip_address
import json
import os
from pathlib import Path, PurePosixPath
import stat
from threading import Lock
from time import monotonic
from typing import Protocol
from urllib.parse import urlsplit
from uuid import uuid4

from pydantic import ValidationError

from videoscope.providers.base import ProviderState, ProviderStatus
from videoscope.providers.types import ObjectTag
from videoscope.providers.vision_worker_contract import (
    MAX_IMAGE_BYTES,
    MAX_IMAGES,
    MAX_RESPONSE_BYTES,
    MAX_TEXTS,
    VisionDetectRequest,
    VisionDetectionResponse,
    VisionEmbedImagesRequest,
    VisionEmbeddingResponse,
    VisionEmbedTextsRequest,
    VisionHealthResponse,
    VisionImageItem,
    VisionTextItem,
    VisionWorkerSpecification,
    identity_fields,
)


@dataclass(frozen=True, slots=True)
class VisionCapabilityStatus:
    ready: bool
    detail: str


@dataclass(frozen=True, slots=True)
class _SourceFingerprint:
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int


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


def _reject_non_finite_json(value: str) -> None:
    raise ValueError(f"non-finite Vision worker JSON: {value}")


class _BoundedHTTPClient:
    """HTTP transport with no proxy environment, redirects, or unbounded response."""

    def __init__(self, *, max_response_bytes: int = MAX_RESPONSE_BYTES) -> None:
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
                            "Vision worker returned invalid Content-Length"
                        ) from error
                    if declared < 0 or declared > self.max_response_bytes:
                        raise RuntimeError("Vision worker response is too large")
                content_type = response.headers.get("Content-Type")
                if content_type is not None and content_type.split(";", 1)[0].strip().casefold() != "application/json":
                    raise RuntimeError("Vision worker returned non-JSON content")
                body = bytearray()
                for chunk in response.iter_bytes(chunk_size=8192):
                    body.extend(chunk)
                    if len(body) > self.max_response_bytes:
                        raise RuntimeError("Vision worker response is too large")
        try:
            payload = json.loads(bytes(body), parse_constant=_reject_non_finite_json)
        except (UnicodeError, ValueError, TypeError) as error:
            raise RuntimeError("Vision worker returned invalid JSON") from error
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
        raise ValueError("Vision worker endpoint must be a plain literal loopback HTTP origin")
    try:
        address = ip_address(parsed.hostname)
        port = parsed.port
    except ValueError as error:
        raise ValueError(
            "Vision worker endpoint must be a plain literal loopback HTTP origin"
        ) from error
    if str(address) != "127.0.0.1" or port is None or not 1 <= port <= 65_535:
        raise ValueError("Vision worker endpoint must be a plain literal loopback HTTP origin")
    return endpoint.strip().rstrip("/")


def _fingerprint(metadata: os.stat_result) -> _SourceFingerprint:
    return _SourceFingerprint(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        size=metadata.st_size,
        modified_ns=metadata.st_mtime_ns,
        changed_ns=metadata.st_ctime_ns,
    )


def _source_identity(path: Path) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeError("Vision worker input cannot be opened safely") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > MAX_IMAGE_BYTES
        ):
            raise RuntimeError("Vision worker input is not a bounded regular file")
        digest = sha256()
        size_bytes = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            size_bytes += len(chunk)
            if size_bytes > MAX_IMAGE_BYTES:
                raise RuntimeError("Vision worker input is too large")
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _fingerprint(before) != _fingerprint(after) or size_bytes != after.st_size:
        raise RuntimeError("Vision worker input changed while it was hashed")
    return digest.hexdigest(), size_bytes


class VisionWorkerClient:
    """Fail-closed encoder and ObjectProvider adapter for the isolated worker."""

    id = "vision-worker"

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        input_root: Path,
        specification: VisionWorkerSpecification,
        timeout: float = 120.0,
        client: HTTPClient | None = None,
    ) -> None:
        self.endpoint = _validate_endpoint(endpoint)
        if not 32 <= len(api_key) <= 256 or any(
            character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._~-"
            for character in api_key
        ):
            raise ValueError("Vision worker API key must be 32-256 URL-safe characters")
        if not 0 < timeout <= 600:
            raise ValueError("Vision worker timeout must be between 0 and 600 seconds")
        lexical_root = Path(os.path.abspath(input_root))
        if lexical_root.is_symlink():
            raise ValueError("Vision worker input root must not be a symlink")
        self._api_key = api_key
        self.input_root = lexical_root
        self.specification = specification
        self.timeout = timeout
        self.client = client
        self._client_lock = Lock()
        self._status_lock = Lock()
        self._cached_status: tuple[float, VisionCapabilityStatus] | None = None

    @property
    def identity(self) -> dict[str, object]:
        return {
            "mode": "isolated-worker",
            "endpoint": self.endpoint,
            "specification": self.specification.to_dict(),
            "specification_hash": self.specification.identity,
            "siglip_specification_hash": self.specification.siglip_identity,
            "detector_specification_hash": self.specification.detector_identity,
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

    def capability(self) -> VisionCapabilityStatus:
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
            health = VisionHealthResponse.model_validate(response.json())
            expected = identity_fields(self.specification)
            exact = (
                health.status == "ok"
                and all(getattr(health, field) == value for field, value in expected.items())
                and health.embedding_dimensions == self.specification.embedding_dimensions
            )
        except Exception:
            exact = False
        status = VisionCapabilityStatus(
            ready=exact,
            detail=(
                "Vision worker is ready"
                if exact
                else "Vision worker is unreachable, unavailable, or incompatible"
            ),
        )
        with self._status_lock:
            self._cached_status = (monotonic() + (5.0 if exact else 2.0), status)
        return status

    def status(self) -> ProviderStatus:
        capability = self.capability()
        return ProviderStatus(
            id=self.id,
            label="Vision worker (SigLIP + RF-DETR)",
            state=(
                ProviderState.READY
                if capability.ready
                else ProviderState.UNAVAILABLE
            ),
            detail=capability.detail,
            optional=True,
        )

    def _relative_source(self, source: Path) -> str:
        lexical = Path(os.path.abspath(source))
        current = self.input_root
        try:
            relative = lexical.relative_to(self.input_root)
        except ValueError as error:
            raise RuntimeError("Vision worker input is outside configured root") from error
        if not relative.parts:
            raise RuntimeError("Vision worker input must be a file")
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise RuntimeError("Vision worker input must not contain symlinks")
        return PurePosixPath(*relative.parts).as_posix()

    def _image_item(self, source: Path, *, item_id: str) -> VisionImageItem:
        path = Path(source)
        relative_path = self._relative_source(path)
        digest, size_bytes = _source_identity(path)
        return VisionImageItem(
            item_id=item_id,
            relative_path=relative_path,
            expected_sha256=digest,
            expected_size_bytes=size_bytes,
        )

    def _validate_response_identity(
        self,
        payload: object,
        request_id: str,
    ) -> None:
        expected = identity_fields(self.specification)
        if getattr(payload, "request_id", None) != request_id or any(
            getattr(payload, field, None) != value for field, value in expected.items()
        ):
            raise ValueError("Vision worker response identity mismatch")

    def _embedding_request(
        self,
        *,
        path: str,
        request: VisionEmbedImagesRequest | VisionEmbedTextsRequest,
        expected_ids: list[str],
    ) -> list[list[float]]:
        try:
            response = self._http_client().post(
                f"{self.endpoint}{path}",
                json=request.model_dump(mode="json"),
                headers=self._headers(json_request=True),
                timeout=self.timeout,
            )
            response.raise_for_status()
            raw_payload = response.json()
        except Exception:
            raise RuntimeError("Vision worker embedding request failed") from None
        try:
            payload = VisionEmbeddingResponse.model_validate(raw_payload)
            self._validate_response_identity(payload, request.request_id)
            if payload.embedding_dimensions != self.specification.embedding_dimensions:
                raise ValueError("Vision worker vector dimensions changed")
            if [item.item_id for item in payload.items] != expected_ids:
                raise ValueError("Vision worker returned unknown or reordered item IDs")
            return [item.vector for item in payload.items]
        except (TypeError, ValueError, ValidationError):
            raise RuntimeError("Vision worker response contract violation") from None

    def image_vectors(self, paths: list[Path]):  # type: ignore[no-untyped-def]
        import numpy as np

        if not paths:
            return np.empty(
                (0, self.specification.embedding_dimensions),
                dtype=np.float32,
            )
        if len(paths) > MAX_IMAGES:
            raise RuntimeError("Vision worker image batch is too large")
        request_id = uuid4().hex
        items = [
            self._image_item(path, item_id=f"image-{index:04d}")
            for index, path in enumerate(paths)
        ]
        request = VisionEmbedImagesRequest(
            **identity_fields(self.specification),
            request_id=request_id,
            items=items,
        )
        vectors = self._embedding_request(
            path="/v1/embed/images",
            request=request,
            expected_ids=[item.item_id for item in items],
        )
        return np.asarray(vectors, dtype=np.float32)

    def text_vectors(self, texts: list[str]):  # type: ignore[no-untyped-def]
        import numpy as np

        if not texts:
            return np.empty(
                (0, self.specification.embedding_dimensions),
                dtype=np.float32,
            )
        if len(texts) > MAX_TEXTS:
            raise RuntimeError("Vision worker text batch is too large")
        request_id = uuid4().hex
        items = [
            VisionTextItem(item_id=f"text-{index:04d}", text=text)
            for index, text in enumerate(texts)
        ]
        request = VisionEmbedTextsRequest(
            **identity_fields(self.specification),
            request_id=request_id,
            items=items,
        )
        vectors = self._embedding_request(
            path="/v1/embed/texts",
            request=request,
            expected_ids=[item.item_id for item in items],
        )
        return np.asarray(vectors, dtype=np.float32)

    def detect(self, image: Path) -> list[ObjectTag]:
        request_id = uuid4().hex
        item = self._image_item(image, item_id="detect-source")
        request = VisionDetectRequest(
            **identity_fields(self.specification),
            request_id=request_id,
            source=item,
            minimum_confidence=self.specification.minimum_confidence,
        )
        try:
            response = self._http_client().post(
                f"{self.endpoint}/v1/detect",
                json=request.model_dump(mode="json"),
                headers=self._headers(json_request=True),
                timeout=self.timeout,
            )
            response.raise_for_status()
            raw_payload = response.json()
        except Exception:
            raise RuntimeError("Vision worker detection request failed") from None
        try:
            payload = VisionDetectionResponse.model_validate(raw_payload)
            self._validate_response_identity(payload, request_id)
            if payload.source_id != item.item_id:
                raise ValueError("Vision worker returned an unknown source ID")
        except (TypeError, ValueError, ValidationError):
            raise RuntimeError("Vision worker response contract violation") from None
        return [
            ObjectTag(
                label=detection.label,
                confidence=detection.confidence,
                metadata={
                    "x": detection.x,
                    "y": detection.y,
                    "width": detection.width,
                    "height": detection.height,
                },
            )
            for detection in payload.detections
        ]
