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
)


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
        default="rfdetr-small",
        validation_alias=AliasChoices("ROBOFLOW_MODEL_ID", "VIDEOSCOPE_ROBOFLOW_MODEL_ID"),
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
    whisper_model: str = Field(
        default=WHISPER_MODEL,
        validation_alias=AliasChoices("WHISPER_MODEL", "VIDEOSCOPE_WHISPER_MODEL"),
    )
    whisper_language: str = Field(
        default="auto",
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
    )
    text_embedding_model: str = TEXT_EMBEDDING_MODEL
    text_embedding_dimensions: int = Field(default=TEXT_EMBEDDING_DIMENSIONS, gt=0)
    siglip_model: str = SIGLIP_224_MODEL
    siglip_batch_size: int = Field(default=8, ge=1, le=256)
    visual_index_step: float = Field(default=1.0, gt=0)
    visual_index_max_width: int = Field(default=640, ge=64)
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
