from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="VIDEOSCOPE_",
        extra="ignore",
    )

    data_dir: Path = Path("data")
    max_upload_bytes: int = 12 * 1024 * 1024 * 1024
    host: str = "127.0.0.1"
    port: int = 8765
    search_limit: int = 30
    max_scene_seconds: float = 30.0
    scene_threshold: float = 3.0
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
        validation_alias=AliasChoices("INTERNVIDEO_API_KEY", "VIDEOSCOPE_INTERNVIDEO_API_KEY"),
    )
    internvideo_timeout: float = 180.0
    internvideo_top_candidates: int = 4
    lighthouse_root: Path | None = Field(
        default=None,
        validation_alias=AliasChoices("LIGHTHOUSE_ROOT", "VIDEOSCOPE_LIGHTHOUSE_ROOT"),
    )
    lighthouse_checkpoint: Path | None = Field(
        default=Path("data/models/lighthouse/clip_qd_detr_qvhighlight.ckpt"),
        validation_alias=AliasChoices("LIGHTHOUSE_CHECKPOINT", "VIDEOSCOPE_LIGHTHOUSE_CHECKPOINT"),
    )
    whisper_model: str = Field(
        default="mlx-community/whisper-large-v3-turbo",
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
    text_embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    text_embedding_dimensions: int = 768
    siglip_model: str = "google/siglip2-base-patch16-224"
    siglip_quality_model: str = "google/siglip2-base-patch16-384"
    siglip_batch_size: int = 8
    semantic_text_min_score: float = 0.42
    visual_min_score: float = 0.18
    temporal_refinement_candidates: int = 2
    temporal_refinement_step: float = 1.5

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
