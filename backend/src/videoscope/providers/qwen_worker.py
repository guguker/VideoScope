from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
from ipaddress import ip_address
import json
import logging
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
from threading import BoundedSemaphore, Lock
from time import monotonic
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
    QwenInferenceStatus,
    QwenVideoJudgement,
    QwenVideoReranker,
    parse_qwen_judgement,
)


logger = logging.getLogger(__name__)

QWEN_WORKER_SCHEMA_VERSION = "qwen-worker-v1"
MAX_REQUEST_BYTES = 64 * 1024
MAX_RESPONSE_BYTES = 64 * 1024
MAX_QUERY_CHARS = 500
MAX_TOKENS = 1024
DEFAULT_MAX_INPUT_BYTES = 128 * 1024 * 1024
_REQUEST_ID_RE = r"^[0-9a-f]{32}$"
_TOKEN_RE = re.compile(r"^[A-Za-z0-9._~-]{32,256}$")
_VIDEO_SUFFIXES = frozenset({".mp4"})
_STORYBOARD_SUFFIXES = frozenset({".jpg", ".jpeg"})


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
    input_kind: Literal["video", "storyboard"]
    relative_path: str = Field(min_length=1, max_length=240)
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
            if (
                self.prompt_kind != "basketball_facts"
                or self.query is not None
                or self.fps is None
            ):
                raise ValueError("video input requires the fixed basketball-facts prompt")
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
    judgement: QwenJudgementPayload


class QwenHealthResponse(_ContractModel):
    schema_version: Literal[QWEN_WORKER_SCHEMA_VERSION]
    status: Literal["ok", "unavailable"]
    model_identity: str = Field(min_length=1, max_length=300)
    runtime_identity: Literal[QWEN_INFERENCE_RUNTIME_IDENTITY]
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


def _validate_worker_input(
    *,
    input_root: Path,
    request: QwenJudgeRequest,
    max_input_bytes: int,
) -> tuple[Path, _SourceFingerprint]:
    relative = PurePosixPath(request.relative_path)
    lexical_path = input_root.joinpath(*relative.parts)
    current = input_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("symlink inputs are not allowed")
    try:
        source = lexical_path.resolve(strict=True)
        source.relative_to(input_root)
        metadata = source.stat()
    except (OSError, ValueError) as error:
        raise ValueError("input is not a file under the configured root") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("input is not a regular file")
    if metadata.st_size <= 0:
        raise ValueError("input file must not be empty")
    allowed = (
        _VIDEO_SUFFIXES
        if request.input_kind == "video"
        else _STORYBOARD_SUFFIXES
    )
    if source.suffix.casefold() not in allowed:
        raise ValueError("input extension is not supported")
    if metadata.st_size > max_input_bytes:
        raise OverflowError("input exceeds the configured size limit")
    return source, _SourceFingerprint(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        size=metadata.st_size,
        modified_ns=metadata.st_mtime_ns,
        changed_ns=metadata.st_ctime_ns,
    )


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
    if not runtime.model_identity.strip():
        raise ValueError("Qwen worker model identity must not be empty")

    app = FastAPI(
        title="VideoScope Qwen Worker",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    capacity = BoundedSemaphore(max_concurrency)

    @app.get("/v1/health", response_model=QwenHealthResponse)
    def health() -> QwenHealthResponse:
        return QwenHealthResponse(
            schema_version=QWEN_WORKER_SCHEMA_VERSION,
            status="ok" if runtime.available else "unavailable",
            model_identity=runtime.model_identity,
            runtime_identity=QWEN_INFERENCE_RUNTIME_IDENTITY,
            loaded=runtime.loaded,
            input_kinds=["video", "storyboard"],
            max_input_bytes=max_input_bytes,
            max_query_chars=MAX_QUERY_CHARS,
            max_tokens=MAX_TOKENS,
            max_concurrency=max_concurrency,
        )

    @app.post("/v1/judge", response_model=QwenJudgeResponse)
    async def judge(request: QwenJudgeRequest) -> QwenJudgeResponse:
        if request.model_identity != runtime.model_identity:
            raise HTTPException(409, "Qwen worker model identity does not match")
        try:
            source, before = _validate_worker_input(
                input_root=resolved_root,
                request=request,
                max_input_bytes=max_input_bytes,
            )
        except OverflowError as error:
            raise HTTPException(413, "Qwen worker input is too large") from error
        except ValueError as error:
            raise HTTPException(400, "Invalid Qwen worker input") from error
        if not capacity.acquire(blocking=False):
            raise HTTPException(429, "Qwen worker is busy")
        try:
            try:
                judgement = await run_in_threadpool(runtime.judge, request, source)
                payload = QwenJudgementPayload.model_validate(asdict(judgement))
            except Exception as error:
                logger.exception("Qwen worker inference failed")
                raise HTTPException(503, "Qwen worker inference failed") from error
            try:
                verified_source, after = _validate_worker_input(
                    input_root=resolved_root,
                    request=request,
                    max_input_bytes=max_input_bytes,
                )
            except (ValueError, OverflowError) as error:
                raise HTTPException(409, "Qwen worker input changed during inference") from error
            if verified_source != source or after != before:
                raise HTTPException(409, "Qwen worker input changed during inference")
        finally:
            capacity.release()
        return QwenJudgeResponse(
            schema_version=QWEN_WORKER_SCHEMA_VERSION,
            request_id=request.request_id,
            model_identity=runtime.model_identity,
            runtime_identity=QWEN_INFERENCE_RUNTIME_IDENTITY,
            judgement=payload,
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
        timeout: float = 180.0,
        client: HTTPClient | None = None,
    ) -> None:
        self.endpoint = _validate_endpoint(endpoint)
        if not _TOKEN_RE.fullmatch(api_key):
            raise ValueError("Qwen worker API key must be 32-256 URL-safe characters")
        if not expected_model_identity.strip():
            raise ValueError("expected Qwen model identity must not be empty")
        if not 0 < timeout <= 600:
            raise ValueError("Qwen worker timeout must be between 0 and 600 seconds")
        self._api_key = api_key
        self.input_root = Path(input_root).resolve()
        self.expected_model_identity = expected_model_identity
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
        elif capability.model_identity != self.expected_model_identity:
            status = QwenInferenceStatus(False, "Qwen worker model identity does not match")
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
    ) -> QwenVideoJudgement:
        request_id = uuid4().hex
        request = QwenJudgeRequest(
            schema_version=QWEN_WORKER_SCHEMA_VERSION,
            request_id=request_id,
            model_identity=self.expected_model_identity,
            runtime_identity=QWEN_INFERENCE_RUNTIME_IDENTITY,
            input_kind=input_kind,
            relative_path=self._relative_source(source),
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
                or payload.model_identity != self.expected_model_identity
                or payload.runtime_identity != QWEN_INFERENCE_RUNTIME_IDENTITY
            ):
                raise ValueError("Qwen worker response identity mismatch")
            return QwenVideoJudgement(**payload.judgement.model_dump())
        except (TypeError, ValueError, ValidationError):
            raise RuntimeError("Qwen worker response contract violation") from None

    def judge_video(
        self,
        source: Path,
        *,
        fps: float,
        max_tokens: int,
    ) -> QwenVideoJudgement:
        return self._judge(
            source=source,
            input_kind="video",
            prompt_kind="basketball_facts",
            query=None,
            fps=fps,
            max_tokens=max_tokens,
        )

    def judge_storyboard(
        self,
        source: Path,
        query: str,
        *,
        max_tokens: int,
    ) -> QwenVideoJudgement:
        return self._judge(
            source=source,
            input_kind="storyboard",
            prompt_kind="generic_query",
            query=query,
            fps=None,
            max_tokens=max_tokens,
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
        if self.loaded:
            return self._model, self._processor
        with self._load_lock:
            if self.loaded:
                return self._model, self._processor
            from mlx_vlm import load

            reference = self._resolve_local_reference()
            self._model, self._processor = load(reference)
            return self._model, self._processor

    def _generate_video(self, source: Path, request: QwenJudgeRequest) -> str:
        from mlx_vlm import apply_chat_template, generate

        model, processor = self._load()
        prompt = apply_chat_template(
            processor,
            model.config,
            [QwenVideoReranker._fact_prompt()],
            video=str(source),
            fps=request.fps,
            enable_thinking=False,
        )
        with self._inference_lock:
            result = generate(
                model,
                processor,
                prompt,
                video=[str(source)],
                fps=request.fps,
                max_tokens=request.max_tokens,
                temperature=0.0,
                enable_thinking=False,
                verbose=False,
            )
        return str(result.text)

    def _generate_storyboard(self, source: Path, request: QwenJudgeRequest) -> str:
        from mlx_vlm import apply_chat_template, generate

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
        with self._inference_lock:
            result = generate(
                model,
                processor,
                prompt,
                image=[str(source)],
                max_tokens=request.max_tokens,
                temperature=0.0,
                enable_thinking=False,
                verbose=False,
            )
        return str(result.text)

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
        return self.data_dir / "tmp"


def main() -> None:
    try:
        settings = QwenWorkerSettings()
    except ValidationError:
        raise RuntimeError("Qwen worker configuration is invalid") from None
    revision = model_revision(settings.model_name)
    if revision is None:
        raise RuntimeError("Qwen worker model must have a pinned revision")
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
