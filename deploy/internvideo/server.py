from __future__ import annotations

import base64
import binascii
from collections.abc import Callable
from io import BytesIO
import json
import logging
import math
import os
import re
import secrets
from threading import Lock
from typing import Any, Literal, Self

from fastapi import FastAPI, HTTPException, Security
from fastapi.exceptions import RequestValidationError
from fastapi.security import HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


logger = logging.getLogger("videoscope.internvideo")

MODEL_NAME = os.getenv("INTERNVIDEO_MODEL", "OpenGVLab/InternVideo2_5_Chat_8B").strip()
MAX_CANDIDATES = 4
MAX_FRAMES_PER_CANDIDATE = 8
MAX_FRAME_BASE64_CHARS = 2_000_000
MAX_REQUEST_BYTES = 64 * 1024 * 1024
MAX_IMAGE_PIXELS = 4_194_304
_COMMIT_REVISION_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_VIDEO_ID_RE = r"^[A-Za-z0-9_-]{1,64}$"


def _required_api_key() -> str:
    value = os.getenv("INTERNVIDEO_API_KEY")
    if not value or not value.strip():
        raise RuntimeError("INTERNVIDEO_API_KEY must be configured")
    if len(value) < 32:
        raise RuntimeError("INTERNVIDEO_API_KEY must contain at least 32 characters")
    return value


def _required_model_revision() -> str:
    value = os.getenv("INTERNVIDEO_REVISION", "").strip()
    if not _COMMIT_REVISION_RE.fullmatch(value):
        raise RuntimeError("INTERNVIDEO_REVISION must be a full 40-character commit SHA")
    return value.lower()


if not MODEL_NAME:
    raise RuntimeError("INTERNVIDEO_MODEL must not be empty")
API_KEY = _required_api_key()
MODEL_REVISION = _required_model_revision()


class ContractModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


class FramePayload(ContractModel):
    timestamp: float = Field(ge=0)
    jpeg_base64: str = Field(min_length=4, max_length=MAX_FRAME_BASE64_CHARS)

    @field_validator("jpeg_base64")
    @classmethod
    def validate_base64(cls, value: str) -> str:
        try:
            base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError("frame must contain valid base64") from error
        return value


class CandidatePayload(ContractModel):
    id: str = Field(min_length=1, max_length=160)
    video_id: str = Field(pattern=_VIDEO_ID_RE)
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    frames: list[FramePayload] = Field(min_length=1, max_length=MAX_FRAMES_PER_CANDIDATE)

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        if any(frame.timestamp < self.start or frame.timestamp > self.end for frame in self.frames):
            raise ValueError("frame timestamps must fall inside the candidate interval")
        return self


class RerankRequest(ContractModel):
    model: str = Field(min_length=1, max_length=160)
    task: Literal["temporal_relevance"]
    query: str = Field(min_length=1, max_length=500)
    candidates: list[CandidatePayload] = Field(min_length=1, max_length=MAX_CANDIDATES)

    @model_validator(mode="after")
    def validate_candidate_ids(self) -> Self:
        ids = [candidate.id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate ids must be unique")
        return self


class ScorePayload(ContractModel):
    id: str = Field(min_length=1, max_length=160)
    score: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=300)


class RerankResponse(ContractModel):
    scores: list[ScorePayload] = Field(max_length=MAX_CANDIDATES)


class HealthResponse(ContractModel):
    status: Literal["ok"]
    model: str
    revision: str
    loaded: bool


class RerankGuardMiddleware:
    """Authenticate and bound `/rerank` before FastAPI reads or validates JSON."""

    def __init__(self, app: ASGIApp, *, api_key: str, max_request_bytes: int) -> None:
        self.app = app
        self.expected_authorization = f"Bearer {api_key}".encode("utf-8")
        self.max_request_bytes = max_request_bytes

    @staticmethod
    async def _error(scope: Scope, receive: Receive, send: Send, status_code: int, detail: str) -> None:
        headers = {"WWW-Authenticate": "Bearer"} if status_code == 401 else None
        await JSONResponse({"detail": detail}, status_code=status_code, headers=headers)(scope, receive, send)

    def _authorized(self, scope: Scope) -> bool:
        supplied = next(
            (value for name, value in scope.get("headers", []) if name.lower() == b"authorization"),
            b"",
        )
        return secrets.compare_digest(supplied, self.expected_authorization)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        is_rerank = (
            scope["type"] == "http"
            and scope.get("method") == "POST"
            and str(scope.get("path", "")).rstrip("/") == "/rerank"
        )
        if not is_rerank:
            await self.app(scope, receive, send)
            return

        if not self._authorized(scope):
            await self._error(scope, receive, send, 401, "Unauthorized")
            return

        header_map = {name.lower(): value for name, value in scope.get("headers", [])}
        content_length = header_map.get(b"content-length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
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
        body_delivered = False

        async def replay_receive() -> Message:
            nonlocal body_delivered
            if not body_delivered:
                body_delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        await self.app(scope, replay_receive, send)


class ModelRuntime:
    def __init__(self) -> None:
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.transform: Callable[[Any], Any] | None = None
        self.load_lock = Lock()
        self.inference_lock = Lock()

    def _build_components(self) -> tuple[Any, Any, Callable[[Any], Any]]:
        import torch
        import torchvision.transforms as transforms
        from transformers import AutoModel, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("InternVideo 2.5 endpoint requires a CUDA GPU")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
            trust_remote_code=True,
            use_fast=False,
        )
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            use_flash_attn=True,
        ).eval().cuda()
        transform = transforms.Compose([
            transforms.Resize((448, 448), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])
        return model, tokenizer, transform

    def load(self) -> tuple[Any, Any, Callable[[Any], Any]]:
        if self.model is None:
            with self.load_lock:
                if self.model is None:
                    model, tokenizer, transform = self._build_components()
                    self.model = model
                    self.tokenizer = tokenizer
                    self.transform = transform
        if self.model is None or self.tokenizer is None or self.transform is None:
            raise RuntimeError("InternVideo model initialization is incomplete")
        return self.model, self.tokenizer, self.transform

    @staticmethod
    def _decode_frame(encoded_frame: str):  # type: ignore[no-untyped-def]
        from PIL import Image

        try:
            raw = base64.b64decode(encoded_frame, validate=True)
            image = Image.open(BytesIO(raw))
            if image.format != "JPEG":
                raise ValueError("frame is not a JPEG image")
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise ValueError("JPEG dimensions exceed the configured limit")
            image.load()
            return image.convert("RGB")
        except Exception as error:
            if isinstance(error, ValueError) and str(error).startswith(("frame is", "JPEG dimensions")):
                raise
            raise ValueError("candidate contains an invalid JPEG frame") from error

    def score(self, query: str, candidate: CandidatePayload) -> tuple[float, str]:
        import torch

        decoded_images = [self._decode_frame(frame.jpeg_base64) for frame in candidate.frames]
        model, tokenizer, transform = self.load()
        images = [transform(image) for image in decoded_images]
        pixel_values = torch.stack(images).to(device=model.device, dtype=torch.bfloat16)
        frame_prefix = "".join(f"Frame{index + 1}: <image>\n" for index in range(len(images)))
        question = (
            frame_prefix
            + "Determine how well this video interval matches the search query: "
            + json.dumps(query, ensure_ascii=False)
            + '. Return only JSON in the form {"score": 0.0, "reason": "short evidence"}. '
            + "Score 1 means an exact visible temporal match and 0 means unrelated."
        )
        generation_config = {
            "do_sample": False,
            "temperature": 0.0,
            "max_new_tokens": 96,
            "top_p": 0.1,
            "num_beams": 1,
        }
        with self.inference_lock, torch.inference_mode():
            response = model.chat(
                tokenizer,
                pixel_values,
                question,
                generation_config,
                num_patches_list=[1] * len(images),
            )
        match = re.search(r"\{.*?\}", str(response), flags=re.DOTALL)
        if not match:
            raise ValueError("model did not return a JSON score")
        payload = json.loads(match.group(0))
        score = float(payload["score"])
        if not math.isfinite(score):
            raise ValueError("model returned a non-finite score")
        score = max(0.0, min(1.0, score))
        reason = str(payload.get("reason") or "").strip()[:300] or "InternVideo visual match"
        return score, reason


runtime = ModelRuntime()
app = FastAPI(title="VideoScope InternVideo 2.5 endpoint", version="0.2.0")
app.add_middleware(
    RerankGuardMiddleware,
    api_key=API_KEY,
    max_request_bytes=MAX_REQUEST_BYTES,
)
bearer_scheme = HTTPBearer(auto_error=False)


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(_request: object, error: RequestValidationError) -> JSONResponse:
    safe_errors = [
        {"type": item["type"], "loc": item["loc"], "msg": item["msg"]}
        for item in error.errors()
    ]
    return JSONResponse({"detail": safe_errors}, status_code=422)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model=MODEL_NAME,
        revision=MODEL_REVISION,
        loaded=runtime.model is not None,
    )


@app.post(
    "/rerank",
    response_model=RerankResponse,
    dependencies=[Security(bearer_scheme)],
    responses={
        401: {"description": "Missing or invalid bearer token"},
        413: {"description": "Request body exceeds the service limit"},
        503: {"description": "The GPU reranker is temporarily unavailable"},
    },
)
def rerank(request: RerankRequest) -> RerankResponse:
    if request.model != MODEL_NAME:
        raise HTTPException(status_code=422, detail="Unsupported model")
    scores: list[ScorePayload] = []
    for candidate in request.candidates:
        try:
            score, reason = runtime.score(request.query, candidate)
        except Exception as error:
            logger.exception("InternVideo rerank failed for candidate %s", candidate.id)
            raise HTTPException(status_code=503, detail="Reranker temporarily unavailable") from error
        scores.append(ScorePayload(id=candidate.id, score=score, reason=reason))
    return RerankResponse(scores=scores)
