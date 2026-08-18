from __future__ import annotations

from ipaddress import ip_address
from pathlib import Path
from typing import Self
from urllib.parse import urlsplit

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from videoscope.model_manifest import (
    SIGLIP_224_MODEL,
    TEXT_EMBEDDING_DIMENSIONS,
    TEXT_EMBEDDING_MODEL,
    WHISPER_MODEL,
    model_revision,
)
from videoscope.providers.vision_worker_contract import (
    MAX_IMAGES,
    MAX_IMAGE_DIMENSION,
    REVIEWED_SIGLIP_PROFILES,
    RFDETR_SMALL_CHECKPOINT_SHA256,
)
from videoscope.providers.whisper import MAX_WHISPER_EFFECTIVE_PROMPT_CHARS


def _validate_literal_loopback_origin(value: str, *, field_name: str) -> None:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "http"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError(f"{field_name} must be a plain literal loopback HTTP origin")
    try:
        address = ip_address(parsed.hostname.casefold())
        port = parsed.port
    except ValueError as error:
        raise ValueError(
            f"{field_name} must be a plain literal loopback HTTP origin"
        ) from error
    if str(address) != "127.0.0.1" or port is None:
        raise ValueError(f"{field_name} must use 127.0.0.1 and an explicit port")


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="VIDEOSCOPE_",
        env_ignore_empty=True,
        populate_by_name=True,
        extra="ignore",
    )

    data_dir: Path = Path("data")
    max_upload_bytes: int = Field(default=12 * 1024 * 1024 * 1024, gt=0)
    host: str = Field(default="127.0.0.1", min_length=1)
    port: int = Field(default=8765, ge=1, le=65_535)
    max_scene_seconds: float = Field(default=30.0, gt=0)
    scene_threshold: float = Field(default=3.0, gt=0)
    ocr_worker_python: Path = Path(".venv-ocr/bin/python")
    ocr_worker_script: Path = Path("scripts/paddle-ocr-worker.py")
    roboflow_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("ROBOFLOW_API_KEY", "VIDEOSCOPE_ROBOFLOW_API_KEY"),
    )
    roboflow_model_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("ROBOFLOW_MODEL_ID", "VIDEOSCOPE_ROBOFLOW_MODEL_ID"),
    )
    vision_worker_endpoint: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "VISION_WORKER_ENDPOINT",
            "VIDEOSCOPE_VISION_WORKER_ENDPOINT",
        ),
    )
    vision_worker_api_key: str | None = Field(
        default=None,
        min_length=32,
        max_length=256,
        pattern=r"^[A-Za-z0-9._~-]+$",
        repr=False,
        validation_alias=AliasChoices(
            "VISION_WORKER_API_KEY",
            "VIDEOSCOPE_VISION_WORKER_API_KEY",
        ),
    )
    vision_worker_timeout: float = Field(default=120.0, gt=0, le=600)
    vision_worker_minimum_confidence: float = Field(
        default=0.25,
        ge=0,
        le=1,
        validation_alias="VIDEOSCOPE_VISION_WORKER_MINIMUM_CONFIDENCE",
    )
    vision_detector_model_id: str = Field(
        default="rfdetr-small",
        pattern=r"^rfdetr-(?:nano|small|medium|large)$",
        validation_alias=AliasChoices(
            "VIDEOSCOPE_VISION_DETECTOR_MODEL_ID",
            "VIDEOSCOPE_VISION_WORKER_DETECTOR_MODEL_ID",
        ),
    )
    vision_detector_checkpoint_sha256: str = Field(
        default=RFDETR_SMALL_CHECKPOINT_SHA256,
        pattern=r"^[0-9a-f]{64}$",
        validation_alias=AliasChoices(
            "VIDEOSCOPE_VISION_DETECTOR_CHECKPOINT_SHA256",
            "VIDEOSCOPE_VISION_WORKER_DETECTOR_CHECKPOINT_SHA256",
            "VIDEOSCOPE_VISION_WORKER_RFDETR_SHA256",
        ),
    )
    internvideo_endpoint: str | None = Field(
        default=None,
        validation_alias=AliasChoices("INTERNVIDEO_ENDPOINT", "VIDEOSCOPE_INTERNVIDEO_ENDPOINT"),
    )
    internvideo_api_key: str | None = Field(
        default=None,
        min_length=32,
        validation_alias=AliasChoices("INTERNVIDEO_API_KEY", "VIDEOSCOPE_INTERNVIDEO_API_KEY"),
    )
    internvideo_timeout: float = Field(default=180.0, gt=0)
    internvideo_top_candidates: int = Field(default=4, ge=1, le=4)
    qwen_video_model: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "QWEN_VIDEO_MODEL",
            "VIDEOSCOPE_QWEN_VIDEO_MODEL",
        ),
    )
    qwen_video_endpoint: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "QWEN_VIDEO_ENDPOINT",
            "VIDEOSCOPE_QWEN_VIDEO_ENDPOINT",
        ),
    )
    qwen_video_api_key: str | None = Field(
        default=None,
        min_length=32,
        max_length=256,
        pattern=r"^[A-Za-z0-9._~-]+$",
        repr=False,
        validation_alias=AliasChoices(
            "QWEN_VIDEO_API_KEY",
            "VIDEOSCOPE_QWEN_VIDEO_API_KEY",
        ),
    )
    qwen_video_timeout: float = Field(default=180.0, gt=0, le=600)
    qwen_video_allow_in_process: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "QWEN_VIDEO_ALLOW_IN_PROCESS",
            "VIDEOSCOPE_QWEN_VIDEO_ALLOW_IN_PROCESS",
        ),
    )
    qwen_video_top_candidates: int = Field(default=12, ge=1, le=50)
    qwen_video_context_seconds: float = Field(default=4.0, ge=0)
    qwen_video_min_clip_seconds: float = Field(default=7.0, gt=0)
    qwen_video_max_clip_seconds: float = Field(default=12.0, gt=0)
    qwen_video_frame_count: int = Field(default=12, ge=1, le=120)
    qwen_video_fps: float = Field(default=2.0, gt=0)
    lighthouse_root: Path | None = Field(
        default=None,
        validation_alias=AliasChoices("LIGHTHOUSE_ROOT", "VIDEOSCOPE_LIGHTHOUSE_ROOT"),
    )
    lighthouse_checkpoint: Path | None = Field(
        default=Path("data/models/lighthouse/clip_qd_detr_qvhighlight.ckpt"),
        validation_alias=AliasChoices("LIGHTHOUSE_CHECKPOINT", "VIDEOSCOPE_LIGHTHOUSE_CHECKPOINT"),
    )
    lighthouse_clip_checkpoint: Path | None = Field(
        default=Path("data/models/lighthouse/ViT-B-32.pt"),
        validation_alias=AliasChoices(
            "LIGHTHOUSE_CLIP_CHECKPOINT",
            "VIDEOSCOPE_LIGHTHOUSE_CLIP_CHECKPOINT",
        ),
    )
    lighthouse_endpoint: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "LIGHTHOUSE_ENDPOINT",
            "VIDEOSCOPE_LIGHTHOUSE_ENDPOINT",
        ),
    )
    lighthouse_api_key: str | None = Field(
        default=None,
        min_length=32,
        max_length=256,
        pattern=r"^[A-Za-z0-9._~-]+$",
        repr=False,
        validation_alias=AliasChoices(
            "LIGHTHOUSE_API_KEY",
            "VIDEOSCOPE_LIGHTHOUSE_API_KEY",
        ),
    )
    lighthouse_timeout: float = Field(default=300.0, gt=0, le=3600)
    lighthouse_allow_in_process: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "LIGHTHOUSE_ALLOW_IN_PROCESS",
            "VIDEOSCOPE_LIGHTHOUSE_ALLOW_IN_PROCESS",
        ),
    )
    whisper_model: str = Field(
        default=WHISPER_MODEL,
        validation_alias=AliasChoices("WHISPER_MODEL", "VIDEOSCOPE_WHISPER_MODEL"),
    )
    whisper_language: str = Field(
        default="auto",
        pattern=r"^(?:auto|[a-z]{2,3}(?:-[a-z0-9]{2,8})?)$",
        validation_alias=AliasChoices("WHISPER_LANGUAGE", "VIDEOSCOPE_WHISPER_LANGUAGE"),
    )
    whisper_initial_prompt: str | None = Field(
        default=(
            "Точная русская речь. Сохраняй имена, фамилии, названия команд, "
            "организаций и географические названия."
        ),
        validation_alias=AliasChoices(
            "WHISPER_INITIAL_PROMPT",
            "VIDEOSCOPE_WHISPER_INITIAL_PROMPT",
        ),
        max_length=MAX_WHISPER_EFFECTIVE_PROMPT_CHARS,
    )
    whisper_worker_endpoint: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "WHISPER_WORKER_ENDPOINT",
            "VIDEOSCOPE_WHISPER_WORKER_ENDPOINT",
        ),
    )
    whisper_worker_api_key: str | None = Field(
        default=None,
        min_length=32,
        max_length=256,
        pattern=r"^[A-Za-z0-9._~-]+$",
        repr=False,
        validation_alias=AliasChoices(
            "WHISPER_WORKER_API_KEY",
            "VIDEOSCOPE_WHISPER_WORKER_API_KEY",
        ),
    )
    whisper_worker_timeout: float = Field(default=3600.0, gt=0, le=3600)
    text_embedding_model: str = TEXT_EMBEDDING_MODEL
    text_embedding_dimensions: int = Field(default=TEXT_EMBEDDING_DIMENSIONS, gt=0)
    siglip_model: str = SIGLIP_224_MODEL
    siglip_batch_size: int = Field(default=8, ge=1, le=MAX_IMAGES)
    visual_index_step: float = Field(default=1.0, gt=0)
    visual_index_max_width: int = Field(
        default=640,
        ge=64,
        le=MAX_IMAGE_DIMENSION,
    )
    semantic_text_min_score: float = Field(default=0.42, ge=0, le=1)
    visual_min_score: float = Field(default=0.18, ge=0, le=1)
    temporal_refinement_candidates: int = Field(default=12, ge=1, le=100)
    temporal_refinement_step: float = Field(default=0.75, gt=0)

    @field_validator("host")
    @classmethod
    def validate_loopback_host(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized == "localhost":
            return normalized
        try:
            address = ip_address(normalized)
        except ValueError as error:
            raise ValueError("host must be a loopback address") from error
        if not address.is_loopback or address.version != 4:
            raise ValueError("host must be localhost or an IPv4 loopback address")
        return normalized

    @model_validator(mode="after")
    def validate_qwen_clip_bounds(self) -> Self:
        if (
            self.text_embedding_model == TEXT_EMBEDDING_MODEL
            and self.text_embedding_dimensions != TEXT_EMBEDDING_DIMENSIONS
        ):
            raise ValueError(
                f"the default text embedding model requires {TEXT_EMBEDDING_DIMENSIONS} dimensions"
            )
        if self.qwen_video_max_clip_seconds < self.qwen_video_min_clip_seconds:
            raise ValueError(
                "qwen_video_max_clip_seconds must be greater than or equal to "
                "qwen_video_min_clip_seconds"
            )
        if self.qwen_video_endpoint:
            if not self.qwen_video_api_key:
                raise ValueError(
                    "qwen_video_api_key is required when qwen_video_endpoint is set"
                )
            if not self.qwen_video_model:
                raise ValueError(
                    "qwen_video_model is required when qwen_video_endpoint is set"
                )
            parsed_qwen = urlsplit(self.qwen_video_endpoint)
            if (
                parsed_qwen.scheme != "http"
                or not parsed_qwen.hostname
                or parsed_qwen.username is not None
                or parsed_qwen.password is not None
                or parsed_qwen.query
                or parsed_qwen.fragment
                or parsed_qwen.path not in {"", "/"}
            ):
                raise ValueError(
                    "qwen_video_endpoint must be a plain loopback HTTP origin"
                )
            qwen_hostname = parsed_qwen.hostname.casefold()
            try:
                qwen_address = ip_address(qwen_hostname)
            except ValueError:
                qwen_address = None
            if qwen_address is None or str(qwen_address) != "127.0.0.1":
                raise ValueError("qwen_video_endpoint must use 127.0.0.1")
            try:
                parsed_qwen.port
            except ValueError as error:
                raise ValueError("qwen_video_endpoint contains an invalid port") from error
            if model_revision(self.qwen_video_model) is None:
                raise ValueError(
                    "qwen_video_model must have a pinned revision for worker mode"
                )
        if self.qwen_video_endpoint and self.qwen_video_allow_in_process:
            raise ValueError(
                "qwen_video_endpoint cannot be combined with qwen_video_allow_in_process"
            )
        if bool(self.vision_worker_endpoint) != bool(self.vision_worker_api_key):
            missing = (
                "vision_worker_api_key"
                if self.vision_worker_endpoint
                else "vision_worker_endpoint"
            )
            raise ValueError(f"{missing} is required for the isolated vision worker")
        if self.vision_worker_endpoint:
            _validate_literal_loopback_origin(
                self.vision_worker_endpoint,
                field_name="vision_worker_endpoint",
            )
            siglip_revision = model_revision(self.siglip_model)
            if (
                siglip_revision is None
                or (self.siglip_model, siglip_revision)
                not in REVIEWED_SIGLIP_PROFILES
            ):
                raise ValueError(
                    "siglip_model must be a reviewed SigLIP worker profile"
                )
        if bool(self.whisper_worker_endpoint) != bool(self.whisper_worker_api_key):
            missing = (
                "whisper_worker_api_key"
                if self.whisper_worker_endpoint
                else "whisper_worker_endpoint"
            )
            raise ValueError(f"{missing} is required for the isolated Whisper worker")
        if self.whisper_worker_endpoint:
            _validate_literal_loopback_origin(
                self.whisper_worker_endpoint,
                field_name="whisper_worker_endpoint",
            )
            if self.whisper_model != WHISPER_MODEL:
                raise ValueError(
                    "whisper_model must be the reviewed Whisper worker model"
                )
        if (
            self.roboflow_api_key is None
            and self.roboflow_model_id
            in {"rfdetr-nano", "rfdetr-small", "rfdetr-medium", "rfdetr-large"}
        ):
            # Migrate the former in-process local detector setting without
            # making an existing .env prevent the isolated backend from starting.
            self.roboflow_model_id = None
        if self.roboflow_model_id is not None:
            if self.roboflow_api_key and "/" not in self.roboflow_model_id:
                raise ValueError("roboflow_model_id must have project/version form")
        if self.vision_worker_endpoint and (
            self.roboflow_api_key or self.roboflow_model_id
        ):
            raise ValueError(
                "hosted Roboflow cannot be combined with the local vision worker"
            )
        if self.lighthouse_endpoint:
            if not self.lighthouse_api_key:
                raise ValueError(
                    "lighthouse_api_key is required when lighthouse_endpoint is set"
                )
            parsed_lighthouse = urlsplit(self.lighthouse_endpoint)
            if (
                parsed_lighthouse.scheme != "http"
                or not parsed_lighthouse.hostname
                or parsed_lighthouse.username is not None
                or parsed_lighthouse.password is not None
                or parsed_lighthouse.query
                or parsed_lighthouse.fragment
                or parsed_lighthouse.path not in {"", "/"}
            ):
                raise ValueError(
                    "lighthouse_endpoint must be a plain loopback HTTP origin"
                )
            try:
                lighthouse_address = ip_address(
                    parsed_lighthouse.hostname.casefold()
                )
            except ValueError:
                lighthouse_address = None
            if (
                lighthouse_address is None
                or str(lighthouse_address) != "127.0.0.1"
            ):
                raise ValueError("lighthouse_endpoint must use 127.0.0.1")
            try:
                parsed_lighthouse.port
            except ValueError as error:
                raise ValueError(
                    "lighthouse_endpoint contains an invalid port"
                ) from error
        if self.lighthouse_endpoint and self.lighthouse_allow_in_process:
            raise ValueError(
                "lighthouse_endpoint cannot be combined with "
                "lighthouse_allow_in_process"
            )
        if self.internvideo_endpoint:
            if not self.internvideo_api_key:
                raise ValueError("internvideo_api_key is required when internvideo_endpoint is set")
            parsed = urlsplit(self.internvideo_endpoint)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError("internvideo_endpoint must be an absolute HTTP(S) URL")
            if parsed.scheme == "http":
                hostname = parsed.hostname.casefold()
                is_loopback = hostname == "localhost"
                if not is_loopback:
                    try:
                        is_loopback = ip_address(hostname).is_loopback
                    except ValueError:
                        is_loopback = False
                if not is_loopback:
                    raise ValueError("remote internvideo_endpoint must use HTTPS")
        return self

    @property
    def glossary_path(self) -> Path:
        return self.data_dir / "search-glossary.json"

    @property
    def evaluation_cases_path(self) -> Path:
        return self.data_dir / "evaluation" / "cases.json"

    @property
    def evaluation_report_path(self) -> Path:
        return self.data_dir / "evaluation" / "latest-report.json"

    @property
    def database_path(self) -> Path:
        return self.data_dir / "videoscope.sqlite3"

    @property
    def media_dir(self) -> Path:
        return self.data_dir / "media"

    @property
    def thumbnails_dir(self) -> Path:
        return self.data_dir / "thumbnails"

    @property
    def clips_dir(self) -> Path:
        return self.data_dir / "clips"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def qdrant_dir(self) -> Path:
        return self.data_dir / "qdrant"

    @property
    def visual_index_dir(self) -> Path:
        return self.data_dir / "visual-index"

    @property
    def temp_dir(self) -> Path:
        return self.data_dir / "tmp"

    def ensure_directories(self) -> None:
        for directory in (
            self.data_dir,
            self.media_dir,
            self.thumbnails_dir,
            self.clips_dir,
            self.cache_dir,
            self.models_dir,
            self.qdrant_dir,
            self.visual_index_dir,
            self.temp_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
