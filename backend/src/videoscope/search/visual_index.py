from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Protocol
from uuid import uuid4

import numpy as np

from videoscope.providers.base import ProviderState, ProviderStatus
from videoscope.model_manifest import model_identity
from videoscope.repository import SegmentRecord
from videoscope.search.fusion import EvidenceHit
from videoscope.search.sports_events import (
    aggregate_stage_percentiles,
    aggregate_top_k_stage_percentiles,
    classify_basketball_event_query,
    event_prompt_stages,
    rank_made_free_throw_windows,
    rank_made_two_point_windows,
    rank_temporal_event_windows,
)
from videoscope.storage import atomic_write_json


class DenseFrameExtractor(Protocol):
    def extract_frames(
        self,
        source: Path,
        destination: Path,
        start: float,
        end: float,
        *,
        step: float,
        max_width: int = 640,
    ) -> list[object]: ...


VISUAL_INDEX_MANIFEST_SCHEMA_VERSION = 1
VISUAL_INDEX_SCHEMA_VERSION = 2
VISUAL_INDEX_SAMPLING_STRATEGY = "fixed_step_seconds"
VISUAL_INDEX_PREPROCESSING_REVISION = "siglip2-auto-processor-rgb-v1"
VISUAL_INDEX_EXTRACTOR_IDENTITY = "ffmpeg-fixed-step-v1"
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_GENERATION_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class VisualIndexSpecification:
    """Canonical compatibility boundary for a dense visual generation."""

    model_name: str
    model_revision: str | None
    sampling_strategy: str = VISUAL_INDEX_SAMPLING_STRATEGY
    sample_step: float = 1.0
    max_width: int = 640
    preprocessing_revision: str = VISUAL_INDEX_PREPROCESSING_REVISION
    schema_version: int = VISUAL_INDEX_SCHEMA_VERSION
    extractor_identity: str = VISUAL_INDEX_EXTRACTOR_IDENTITY

    def __post_init__(self) -> None:
        string_values = {
            "model_name": self.model_name,
            "sampling_strategy": self.sampling_strategy,
            "preprocessing_revision": self.preprocessing_revision,
            "extractor_identity": self.extractor_identity,
        }
        if any(type(value) is not str or not value.strip() for value in string_values.values()):
            raise ValueError("visual index specification strings must be non-empty")
        if self.model_revision is not None and (
            type(self.model_revision) is not str or not self.model_revision.strip()
        ):
            raise ValueError("visual model revision must be a non-empty string")
        if (
            isinstance(self.sample_step, bool)
            or not math.isfinite(float(self.sample_step))
            or float(self.sample_step) <= 0
        ):
            raise ValueError("visual index step must be positive")
        if type(self.max_width) is not int or self.max_width < 64:
            raise ValueError("visual index maximum width must be at least 64")
        if type(self.schema_version) is not int or self.schema_version < 1:
            raise ValueError("visual index schema version must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "extractor_identity": self.extractor_identity,
            "max_width": self.max_width,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "preprocessing_revision": self.preprocessing_revision,
            "sample_step": float(self.sample_step),
            "sampling_strategy": self.sampling_strategy,
            "schema_version": self.schema_version,
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


_JERSEY_NUMBER_PATTERN = re.compile(
    r"(?:под\s+номером|номер(?:ом)?|№|jersey(?:\s+number)?)\s*#?\s*(\d{1,2})",
    flags=re.IGNORECASE,
)


def visual_prompt_variants(query: str) -> list[str]:
    """Добавляет предметные англоязычные формулировки без изменения запроса пользователя."""
    normalized = " ".join(query.split()).strip()
    if not normalized:
        return []
    prompts = [normalized]

    event_specification = classify_basketball_event_query(normalized)
    if event_specification is not None:
        if event_specification.event_type == "made_three_point":
            prompts.extend(
                [
                    "basketball player shoots a three-point jump shot from behind the arc",
                    "basketball player makes a three-point shot from behind the arc",
                    "the basketball goes through the hoop after a three-point shot",
                    "a complete made three-point basketball shot from release to basket",
                ]
            )
        elif event_specification.event_type == "made_two_point":
            prompts.extend(
                [
                    "basketball player makes a two-point shot from inside the three-point line",
                    "the basketball goes through the hoop after a two-point shot",
                    "a complete made two-point basketball shot from release to basket",
                ]
            )
        elif event_specification.event_type == "made_free_throw":
            prompts.extend(
                [
                    "basketball player makes a free throw from the free throw line",
                    "the basketball goes through the hoop after a free throw",
                    "a complete made free throw from release to basket",
                ]
            )

    jersey_match = _JERSEY_NUMBER_PATTERN.search(normalized)
    if jersey_match:
        prompts.append(
            f"basketball player wearing jersey number {jersey_match.group(1)}"
        )

    return list(dict.fromkeys(prompts))


def made_three_point_prompt_stages(
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Возвращает формулировки выпуска, результата и контекста после броска."""
    specification = event_prompt_stages("made_three_point")
    return (
        specification.release_prompts,
        specification.outcome_prompts,
        specification.followup_prompts,
    )


class SiglipVisualIndex:
    """Многоязычный поиск соответствий между кадрами и текстом на базе SigLIP 2."""

    id = "siglip2"

    def __init__(
        self,
        path: Path,
        *,
        model_name: str,
        model_revision: str | None = None,
        batch_size: int = 8,
        sample_step: float = 1.0,
        max_width: int = 640,
        sampling_strategy: str = VISUAL_INDEX_SAMPLING_STRATEGY,
        preprocessing_revision: str = VISUAL_INDEX_PREPROCESSING_REVISION,
        schema_version: int = VISUAL_INDEX_SCHEMA_VERSION,
        extractor_identity: str = VISUAL_INDEX_EXTRACTOR_IDENTITY,
    ) -> None:
        self.path = Path(path)
        self.model_name = model_name
        self.model_revision = model_revision
        self.batch_size = max(1, batch_size)
        self.specification = VisualIndexSpecification(
            model_name=model_name,
            model_revision=model_revision,
            sampling_strategy=sampling_strategy,
            sample_step=sample_step,
            max_width=max_width,
            preprocessing_revision=preprocessing_revision,
            schema_version=schema_version,
            extractor_identity=extractor_identity,
        )
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._logit_scale = 1.0
        self._logit_bias = 0.0

    @property
    def model_identity(self) -> str:
        return model_identity(self.model_name, self.model_revision)

    @property
    def identity(self) -> str:
        """Expose the complete spec and active generations to evaluation freshness."""
        descriptor = {
            "active_generations": self._active_generation_descriptors(),
            "specification": self.specification.to_dict(),
            "specification_hash": self.specification.identity,
        }
        return json.dumps(
            descriptor,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def status(self, *, check_index: bool = True) -> ProviderStatus:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            return ProviderStatus(
                self.id,
                "SigLIP 2",
                ProviderState.UNAVAILABLE,
                "torch and transformers are required",
            )
        model_path = Path(self.model_name).expanduser()
        if not model_path.exists():
            try:
                from huggingface_hub import try_to_load_from_cache

                config = try_to_load_from_cache(
                    self.model_name,
                    "config.json",
                    revision=self.model_revision,
                )
                weights = try_to_load_from_cache(
                    self.model_name,
                    "model.safetensors",
                    revision=self.model_revision,
                )
            except Exception:
                config = None
                weights = None
            if not config or not weights:
                return ProviderStatus(
                    self.id,
                    "SigLIP 2",
                    ProviderState.NEEDS_CONFIGURATION,
                    f"Модель не загружена: {self.model_name}",
                )
        if check_index and self._has_incompatible_indexes():
            return ProviderStatus(
                self.id,
                "SigLIP 2",
                ProviderState.NEEDS_CONFIGURATION,
                "Визуальный индекс отсутствует, повреждён или создан другой спецификацией; "
                "выполните make index-visual",
            )
        return ProviderStatus(
            self.id,
            "SigLIP 2",
            ProviderState.READY,
            f"Многоязычный визуальный поиск: {self.model_name}",
        )

    def _has_incompatible_indexes(self) -> bool:
        if not self.path.is_dir():
            return False
        for video_path in self.path.iterdir():
            if (
                not video_path.is_dir()
                or video_path.is_symlink()
                or not self._valid_video_id(video_path.name)
            ):
                continue
            has_legacy_artifacts = any(
                (video_path / name).exists()
                for name in ("vectors.npy", "metadata.json", "model.txt")
            )
            has_generations = (video_path / "generations").is_dir()
            has_active_pointer = (video_path / "active.json").exists()
            if has_active_pointer and self.index_is_current(video_path.name):
                # A validated dense generation supersedes, but does not delete,
                # inert legacy files left for bounded migration rollback.
                continue
            if has_legacy_artifacts:
                return True
            if (has_generations or has_active_pointer) and not self.index_is_current(
                video_path.name
            ):
                return True
        return False

    @staticmethod
    def _pooled(output: object):  # type: ignore[no-untyped-def]
        pooled = getattr(output, "pooler_output", None)
        if pooled is not None:
            return pooled
        if isinstance(output, tuple) and len(output) > 1:
            return output[1]
        return output

    def _load(self):  # type: ignore[no-untyped-def]
        if self._model is not None and self._processor is not None:
            return self._model, self._processor

        import torch
        from transformers import AutoModel, AutoProcessor

        self._device = "mps" if torch.backends.mps.is_available() else "cpu"
        load_options = (
            {"revision": self.model_revision}
            if self.model_revision is not None
            else {}
        )
        self._processor = AutoProcessor.from_pretrained(self.model_name, **load_options)
        self._model = AutoModel.from_pretrained(self.model_name, **load_options)
        self._model.eval().to(self._device)
        self._logit_scale = math.exp(float(self._model.logit_scale.detach().cpu().item()))
        self._logit_bias = float(self._model.logit_bias.detach().cpu().item())
        return self._model, self._processor

    @staticmethod
    def _normalize(features):  # type: ignore[no-untyped-def]
        return features / features.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)

    def _image_vectors(self, paths: list[Path]) -> np.ndarray:
        import torch
        from PIL import Image

        model, processor = self._load()
        output: list[np.ndarray] = []
        for offset in range(0, len(paths), self.batch_size):
            images = []
            for path in paths[offset : offset + self.batch_size]:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            inputs = processor(images=images, return_tensors="pt")
            inputs = {name: value.to(self._device) for name, value in inputs.items()}
            with torch.inference_mode():
                encoded = self._pooled(model.get_image_features(**inputs))
                encoded = self._normalize(encoded)
            output.append(encoded.detach().float().cpu().numpy())
        return np.concatenate(output, axis=0) if output else np.empty((0, 0), dtype=np.float32)

    def _text_vector(self, query: str) -> np.ndarray:
        import torch

        model, processor = self._load()
        inputs = processor(text=[query], padding="max_length", return_tensors="pt")
        inputs = {name: value.to(self._device) for name, value in inputs.items()}
        with torch.inference_mode():
            encoded = self._pooled(model.get_text_features(**inputs))
            encoded = self._normalize(encoded)
        return encoded[0].detach().float().cpu().numpy()

    def _text_vectors(self, query: str) -> np.ndarray:
        prompts = visual_prompt_variants(query)
        normalized = query.casefold()
        if len(query.split()) <= 5 and not any(
            subject in normalized
            for subject in ("человек", "мужчина", "женщина", "игрок", "person", "man", "woman", "player")
        ):
            prompts.append(f"человек {query}")
        return np.stack([self._text_vector(prompt) for prompt in prompts])

    def score_images(self, query: str, paths: list[Path]) -> list[float]:
        if not query.strip() or not paths:
            return []
        image_vectors = self._image_vectors(paths)
        query_vectors = self._text_vectors(query)
        similarities = np.max(image_vectors @ query_vectors.T, axis=1)
        return [self._rank_score(float(value)) for value in similarities]

    @staticmethod
    def _valid_video_id(video_id: object) -> bool:
        return type(video_id) is str and _VIDEO_ID_RE.fullmatch(video_id) is not None

    def _video_dir(self, video_id: str) -> Path:
        if not self._valid_video_id(video_id):
            raise ValueError("invalid visual index video id")
        return self.path / video_id

    @staticmethod
    def _source_identity(source: Path) -> tuple[str, int]:
        source = Path(source)
        if not source.is_file() or source.is_symlink():
            raise ValueError("visual index source must be a regular file")
        digest = sha256()
        size_bytes = 0
        with source.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                size_bytes += len(chunk)
        if size_bytes <= 0:
            raise ValueError("visual index source must not be empty")
        return digest.hexdigest(), size_bytes

    @staticmethod
    def _read_json(path: Path) -> object:
        return json.loads(path.read_text(encoding="utf-8"))

    def _read_active_pointer(self, video_id: str) -> tuple[str, bytes] | None:
        try:
            video_dir = self._video_dir(video_id)
        except ValueError:
            return None
        pointer_path = video_dir / "active.json"
        if not pointer_path.is_file() or pointer_path.is_symlink():
            return None
        try:
            snapshot = pointer_path.read_bytes()
            payload: object = json.loads(snapshot.decode("utf-8"))
        except (OSError, UnicodeError, ValueError, TypeError):
            return None
        if (
            not isinstance(payload, dict)
            or set(payload) != {"generation_id", "schema_version"}
            or type(payload.get("schema_version")) is not int
            or payload.get("schema_version") != VISUAL_INDEX_MANIFEST_SCHEMA_VERSION
        ):
            return None
        generation_id = payload.get("generation_id")
        if type(generation_id) is not str or _GENERATION_ID_RE.fullmatch(generation_id) is None:
            return None
        return generation_id, snapshot

    def active_generation_id(self, video_id: str) -> str | None:
        pointer = self._read_active_pointer(video_id)
        return pointer[0] if pointer is not None else None

    def _generation_dir(self, video_id: str, generation_id: str) -> Path | None:
        if not self._valid_video_id(video_id) or _GENERATION_ID_RE.fullmatch(generation_id) is None:
            return None
        generation = self.path / video_id / "generations" / generation_id
        if not generation.is_dir() or generation.is_symlink():
            return None
        try:
            generation.resolve().relative_to((self.path / video_id / "generations").resolve())
        except (OSError, ValueError):
            return None
        return generation

    def _validate_manifest(
        self,
        raw_manifest: object,
        *,
        generation_id: str,
    ) -> dict[str, object] | None:
        expected_keys = {
            "duration_seconds",
            "frame_count",
            "generation_id",
            "manifest_schema_version",
            "source_sha256",
            "source_size_bytes",
            "specification",
            "specification_hash",
            "vector_count",
            "vector_dimensions",
        }
        if not isinstance(raw_manifest, dict) or set(raw_manifest) != expected_keys:
            return None
        integer_fields = (
            "frame_count",
            "source_size_bytes",
            "vector_count",
            "vector_dimensions",
        )
        if any(
            type(raw_manifest.get(name)) is not int or int(raw_manifest[name]) <= 0
            for name in integer_fields
        ):
            return None
        duration = raw_manifest.get("duration_seconds")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or float(duration) <= 0
        ):
            return None
        if (
            type(raw_manifest.get("manifest_schema_version")) is not int
            or raw_manifest.get("manifest_schema_version")
            != VISUAL_INDEX_MANIFEST_SCHEMA_VERSION
            or raw_manifest.get("generation_id") != generation_id
            or raw_manifest.get("specification") != self.specification.to_dict()
            or raw_manifest.get("specification_hash") != self.specification.identity
            or type(raw_manifest.get("source_sha256")) is not str
            or _SHA256_RE.fullmatch(str(raw_manifest["source_sha256"])) is None
            or raw_manifest["frame_count"] != raw_manifest["vector_count"]
        ):
            return None
        return raw_manifest

    def _validate_metadata(
        self,
        raw_metadata: object,
        *,
        generation: Path,
        duration: float,
        validate_thumbnail_paths: bool = True,
    ) -> list[dict[str, Any]] | None:
        if not isinstance(raw_metadata, list) or not raw_metadata:
            return None
        metadata: list[dict[str, Any]] = []
        segment_ids: set[str] = set()
        previous_start = -1.0
        try:
            generation_root = generation.resolve()
            for index, raw_item in enumerate(raw_metadata):
                if not isinstance(raw_item, dict) or set(raw_item) != {
                    "end",
                    "frame_path",
                    "segment_id",
                    "start",
                    "thumbnail_path",
                }:
                    raise ValueError("invalid visual metadata")
                raw_start = raw_item["start"]
                raw_end = raw_item["end"]
                if (
                    isinstance(raw_start, bool)
                    or not isinstance(raw_start, (int, float))
                    or isinstance(raw_end, bool)
                    or not isinstance(raw_end, (int, float))
                ):
                    raise ValueError("invalid visual metadata boundaries")
                start = float(raw_start)
                end = float(raw_end)
                segment_id = raw_item["segment_id"]
                frame_path = raw_item["frame_path"]
                thumbnail_path = raw_item["thumbnail_path"]
                if (
                    type(segment_id) is not str
                    or not segment_id
                    or segment_id in segment_ids
                    or type(frame_path) is not str
                    or not frame_path
                    or Path(frame_path).is_absolute()
                    or ".." in Path(frame_path).parts
                    or type(thumbnail_path) is not str
                    or not thumbnail_path
                    or not math.isfinite(start)
                    or not math.isfinite(end)
                    or start < 0
                    or start <= previous_start
                    or end <= start
                    or end > duration
                ):
                    raise ValueError("invalid visual metadata")
                resolved_frame = (generation / frame_path).resolve()
                resolved_frame.relative_to(generation_root)
                raw_frame = generation / frame_path
                if raw_frame.is_symlink() or not resolved_frame.is_file():
                    raise ValueError("visual generation frame is missing")
                if validate_thumbnail_paths:
                    raw_thumbnail = Path(thumbnail_path)
                    resolved_thumbnail = raw_thumbnail.resolve()
                    if raw_thumbnail.is_symlink() or not resolved_thumbnail.is_file():
                        raise ValueError("visual generation thumbnail is missing")
                    if resolved_thumbnail != resolved_frame and raw_thumbnail.name != (
                        f"visual-{generation.name}-{index:06d}.jpg"
                    ):
                        raise ValueError("visual generation thumbnail identity is invalid")
                segment_ids.add(segment_id)
                previous_start = start
                metadata.append(
                    {
                        "segment_id": segment_id,
                        "start": start,
                        "end": end,
                        "thumbnail_path": thumbnail_path,
                    }
                )
        except (KeyError, OSError, TypeError, ValueError):
            return None
        return metadata

    def _load_generation(
        self,
        video_id: str,
        generation_id: str,
        *,
        memory_map: bool = False,
    ) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, object]] | None:
        generation = self._generation_dir(video_id, generation_id)
        if generation is None:
            return None
        vectors_path = generation / "vectors.npy"
        metadata_path = generation / "metadata.json"
        manifest_path = generation / "manifest.json"
        if any(
            not path.is_file() or path.is_symlink()
            for path in (vectors_path, metadata_path, manifest_path)
        ):
            return None
        try:
            manifest = self._validate_manifest(
                self._read_json(manifest_path),
                generation_id=generation_id,
            )
            if manifest is None:
                return None
            metadata = self._validate_metadata(
                self._read_json(metadata_path),
                generation=generation,
                duration=float(manifest["duration_seconds"]),
            )
            vectors = np.load(
                vectors_path,
                allow_pickle=False,
                mmap_mode="r" if memory_map else None,
            )
        except (EOFError, OSError, UnicodeError, ValueError, TypeError):
            return None
        if metadata is None:
            return None
        try:
            vectors_are_valid = (
                vectors.ndim == 2
                and vectors.shape[0] == len(metadata)
                and vectors.shape[1] > 0
                and vectors.shape[0] == manifest["vector_count"]
                and vectors.shape[1] == manifest["vector_dimensions"]
                and bool(np.all(np.isfinite(vectors)))
            )
        except (TypeError, ValueError):
            return None
        return (vectors, metadata, manifest) if vectors_are_valid else None

    def _load_video_index(
        self,
        video_id: str,
        *,
        memory_map: bool = False,
    ) -> tuple[np.ndarray, list[dict[str, Any]]] | None:
        pointer = self._read_active_pointer(video_id)
        if pointer is None:
            return None
        generation_id, pointer_snapshot = pointer
        loaded = self._load_generation(
            video_id,
            generation_id,
            memory_map=memory_map,
        )
        if loaded is None:
            return None
        try:
            if (self._video_dir(video_id) / "active.json").read_bytes() != pointer_snapshot:
                return None
        except (OSError, ValueError):
            return None
        vectors, metadata, _manifest = loaded
        return vectors, metadata

    def index_is_current(self, video_id: str) -> bool:
        """Return whether the active dense generation matches the full specification."""
        return self._load_video_index(video_id, memory_map=True) is not None

    def _active_generation_descriptors(self) -> list[dict[str, object]]:
        if not self.path.is_dir():
            return []
        descriptors: list[dict[str, object]] = []
        for video_dir in sorted(self.path.iterdir(), key=lambda item: item.name):
            if (
                not video_dir.is_dir()
                or video_dir.is_symlink()
                or not self._valid_video_id(video_dir.name)
            ):
                continue
            pointer = self._read_active_pointer(video_dir.name)
            if pointer is None:
                if (video_dir / "active.json").exists():
                    descriptors.append({"state": "invalid", "video_id": video_dir.name})
                continue
            generation_id, _snapshot = pointer
            loaded_generation = self._load_generation(
                video_dir.name,
                generation_id,
                memory_map=True,
            )
            manifest: dict[str, object] | None = None
            if loaded_generation is not None:
                _vectors, _metadata, manifest = loaded_generation
            descriptor: dict[str, object] = {
                "generation_id": generation_id,
                "state": "current" if manifest is not None else "invalid",
                "video_id": video_dir.name,
            }
            if manifest is not None:
                descriptor["source_sha256"] = manifest["source_sha256"]
                descriptor["source_size_bytes"] = manifest["source_size_bytes"]
                descriptor["duration_seconds"] = manifest["duration_seconds"]
            descriptors.append(descriptor)
        return descriptors

    @staticmethod
    def _atomic_save_vectors(path: Path, vectors: np.ndarray) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.save(handle, vectors, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def replace_video(self, video_id: str, segments: list[SegmentRecord]) -> None:
        del video_id, segments
        raise RuntimeError(
            "scene-only visual indexes are legacy; use replace_video_source"
        )

    @staticmethod
    def _create_thumbnail_aliases(
        generation: Path,
        metadata: list[dict[str, object]],
    ) -> list[Path]:
        created: list[Path] = []
        try:
            for item in metadata:
                thumbnail_value = item["thumbnail_path"]
                if thumbnail_value is None:
                    continue
                thumbnail = Path(str(thumbnail_value))
                thumbnail.parent.mkdir(parents=True, exist_ok=True)
                source_frame = generation / str(item["frame_path"])
                if thumbnail.resolve() == source_frame.resolve():
                    continue
                if thumbnail.exists() or thumbnail.is_symlink():
                    raise FileExistsError("visual thumbnail alias already exists")
                try:
                    os.link(source_frame, thumbnail)
                except OSError:
                    shutil.copy2(source_frame, thumbnail)
                created.append(thumbnail)
        except Exception:
            for path in created:
                path.unlink(missing_ok=True)
            raise
        return created

    def replace_video_source(
        self,
        video_id: str,
        source: Path,
        duration: float,
        extractor: DenseFrameExtractor,
        *,
        frames_dir: Path | None = None,
    ) -> str:
        """Build and atomically activate an immutable dense visual generation."""
        duration = float(duration)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("video duration must be positive")
        video_dir = self._video_dir(video_id)
        source = Path(source)
        source_sha256, source_size_bytes = self._source_identity(source)
        generation_id = uuid4().hex
        generations_dir = video_dir / "generations"
        generations_dir.mkdir(parents=True, exist_ok=True)
        staging = generations_dir / f".{generation_id}.building"
        generation = generations_dir / generation_id
        staging.mkdir(exist_ok=False)
        created_aliases: list[Path] = []
        activated = False
        try:
            extracted_dir = staging / "frames"
            frames = extractor.extract_frames(
                source,
                extracted_dir,
                0.0,
                duration,
                step=float(self.specification.sample_step),
                max_width=self.specification.max_width,
            )
            if not frames:
                raise RuntimeError("dense frame extraction returned no frames")
            paths = [Path(getattr(frame, "path")) for frame in frames]
            timestamps = [float(getattr(frame, "timestamp")) for frame in frames]
            previous_timestamp = -1.0
            for timestamp, path in zip(timestamps, paths, strict=True):
                if (
                    not math.isfinite(timestamp)
                    or timestamp < 0
                    or timestamp <= previous_timestamp
                    or timestamp >= duration
                ):
                    raise ValueError(
                        "visual frame timestamps must be finite, ordered and inside the video"
                    )
                if path.is_symlink():
                    raise ValueError("visual extractor returned an unsafe frame")
                resolved_path = path.resolve()
                try:
                    resolved_path.relative_to(extracted_dir.resolve())
                except (OSError, ValueError) as error:
                    raise ValueError("visual extractor returned a frame outside its generation") from error
                if not resolved_path.is_file() or resolved_path.is_symlink():
                    raise ValueError("visual extractor returned a missing or unsafe frame")
                previous_timestamp = timestamp

            vectors = self._image_vectors(paths).astype(np.float32, copy=False)
            if (
                vectors.ndim != 2
                or vectors.shape[0] != len(paths)
                or vectors.shape[0] <= 0
                or vectors.shape[1] <= 0
                or not bool(np.all(np.isfinite(vectors)))
            ):
                raise ValueError("SigLIP index vectors do not match finite frame rows")
            if self._source_identity(source) != (source_sha256, source_size_bytes):
                raise RuntimeError("visual index source changed during generation build")

            alias_root = Path(frames_dir) if frames_dir is not None else None
            metadata: list[dict[str, object]] = []
            for index, (timestamp, path) in enumerate(zip(timestamps, paths, strict=True)):
                relative_frame = path.resolve().relative_to(staging.resolve()).as_posix()
                thumbnail_path = (
                    alias_root / f"visual-{generation_id}-{index:06d}.jpg"
                    if alias_root is not None
                    else generation / relative_frame
                )
                metadata.append(
                    {
                        "end": min(duration, timestamp + float(self.specification.sample_step)),
                        "frame_path": relative_frame,
                        "segment_id": f"dense-{index:06d}",
                        "start": timestamp,
                        "thumbnail_path": str(thumbnail_path),
                    }
                )
            manifest = {
                "duration_seconds": duration,
                "frame_count": len(metadata),
                "generation_id": generation_id,
                "manifest_schema_version": VISUAL_INDEX_MANIFEST_SCHEMA_VERSION,
                "source_sha256": source_sha256,
                "source_size_bytes": source_size_bytes,
                "specification": self.specification.to_dict(),
                "specification_hash": self.specification.identity,
                "vector_count": int(vectors.shape[0]),
                "vector_dimensions": int(vectors.shape[1]),
            }
            self._atomic_save_vectors(staging / "vectors.npy", vectors)
            atomic_write_json(staging / "metadata.json", metadata, sort_keys=True)
            atomic_write_json(staging / "manifest.json", manifest, sort_keys=True)
            staged = self._validate_manifest(
                self._read_json(staging / "manifest.json"),
                generation_id=generation_id,
            )
            staged_metadata = self._validate_metadata(
                self._read_json(staging / "metadata.json"),
                generation=staging,
                duration=duration,
                validate_thumbnail_paths=False,
            )
            if staged is None or staged_metadata is None:
                raise ValueError("visual generation validation failed")
            persisted_vectors = np.load(staging / "vectors.npy", allow_pickle=False)
            if (
                persisted_vectors.shape != vectors.shape
                or not bool(np.all(np.isfinite(persisted_vectors)))
            ):
                raise ValueError("persisted visual vectors failed validation")

            staging.replace(generation)
            created_aliases = self._create_thumbnail_aliases(generation, metadata)
            if self._load_generation(video_id, generation_id, memory_map=True) is None:
                raise ValueError("completed visual generation failed read validation")
            atomic_write_json(
                video_dir / "active.json",
                {
                    "generation_id": generation_id,
                    "schema_version": VISUAL_INDEX_MANIFEST_SCHEMA_VERSION,
                },
                sort_keys=True,
            )
            activated = True
            return generation_id
        finally:
            if staging.exists():
                shutil.rmtree(staging)
            if not activated:
                for alias in created_aliases:
                    alias.unlink(missing_ok=True)

    def _probability(self, similarity: float) -> float:
        logit = similarity * self._logit_scale + self._logit_bias
        if logit >= 0:
            return 1.0 / (1.0 + math.exp(-logit))
        exp_logit = math.exp(logit)
        return exp_logit / (1.0 + exp_logit)

    @staticmethod
    def _rank_score(similarity: float) -> float:
        """Преобразует близость SigLIP в шкалу ранжирования, не выдавая её за вероятность."""
        if not math.isfinite(similarity):
            raise ValueError("SigLIP returned a non-finite similarity")
        return max(0.0, min(1.0, (similarity + 1.0) / 2.0))

    def search(
        self,
        query: str,
        *,
        video_ids: list[str] | None = None,
        limit: int = 50,
    ) -> list[EvidenceHit]:
        if not query.strip() or limit <= 0 or not self.path.is_dir():
            return []
        event_specification = classify_basketball_event_query(query)
        event_stage_vectors: dict[str, np.ndarray] | None = None
        query_vectors: np.ndarray | None = None
        if event_specification is not None:
            prompt_groups = {
                "release": event_specification.release_prompts,
                "outcome": event_specification.outcome_prompts,
                "followup": event_specification.followup_prompts,
                "contrast": event_specification.contrast_prompts,
                "plus": event_specification.plus_prompts,
                "miss": event_specification.miss_prompts,
                "transition": event_specification.transition_prompts,
            }
            event_stage_vectors = {
                name: np.stack(
                    [self._text_vector(prompt) for prompt in prompts]
                )
                for name, prompts in prompt_groups.items()
                if prompts
            }
        else:
            query_vectors = self._text_vectors(query)
        encoded_queries = (
            list(event_stage_vectors.values())
            if event_stage_vectors is not None
            else [query_vectors]
        )
        if any(
            vector is None or not np.all(np.isfinite(vector))
            for vector in encoded_queries
        ):
            raise RuntimeError("SigLIP returned non-finite query vectors")
        candidates: list[EvidenceHit] = []
        selected = video_ids or [path.name for path in self.path.iterdir() if path.is_dir()]
        for video_id in selected:
            loaded_index = self._load_video_index(video_id)
            if loaded_index is None:
                continue
            vectors, metadata = loaded_index
            expected_dimensions = (
                event_stage_vectors["release"].shape[1]
                if event_stage_vectors is not None
                else query_vectors.shape[1]  # type: ignore[union-attr]
            )
            if vectors.shape[1] != expected_dimensions:
                continue
            if event_stage_vectors is not None:
                assert event_specification is not None
                release_scores = aggregate_stage_percentiles(
                    vectors @ event_stage_vectors["release"].T
                )
                outcome_scores = aggregate_stage_percentiles(
                    vectors @ event_stage_vectors["outcome"].T
                )
                timestamps = np.array(
                    [float(item["start"]) for item in metadata],
                    dtype=np.float32,
                )
                positive_steps = np.diff(timestamps)
                positive_steps = positive_steps[positive_steps > 0]
                step = (
                    float(np.median(positive_steps))
                    if len(positive_steps)
                    else 1.0
                )
                window_frames = max(
                    1,
                    int(math.ceil(event_specification.max_gap_seconds / step)),
                )
                followup_frames = max(
                    1,
                    int(
                        math.ceil(
                            event_specification.followup_gap_seconds / step
                        )
                    ),
                )
                suppression_frames = max(
                    1,
                    int(
                        math.ceil(
                            event_specification.suppression_seconds / step
                        )
                    ),
                )
                if (
                    event_specification.scoring_strategy
                    == "release_outcome_followup"
                ):
                    followup_scores = aggregate_stage_percentiles(
                        vectors @ event_stage_vectors["followup"].T
                    )
                    event_windows = rank_temporal_event_windows(
                        release_scores,
                        outcome_scores,
                        followup_scores=followup_scores,
                        max_gap=window_frames,
                        followup_gap=followup_frames,
                        suppression_radius=suppression_frames,
                        minimum_followup_score=(
                            event_specification.minimum_followup_score
                        ),
                        limit=limit,
                    )
                    stage_order = ["release", "outcome", "followup"]
                elif (
                    event_specification.scoring_strategy
                    == "made_two_fixed_lag"
                ):
                    contrast_scores = aggregate_stage_percentiles(
                        vectors @ event_stage_vectors["contrast"].T
                    )
                    event_windows = rank_made_two_point_windows(
                        release_scores,
                        outcome_scores,
                        contrast_scores,
                        release_lags=(
                            max(1, int(round(3.0 / step))),
                            max(1, int(round(2.0 / step))),
                        ),
                        contrast_lag=max(1, int(round(2.0 / step))),
                        suppression_radius=suppression_frames,
                        limit=limit,
                    )
                    stage_order = ["release", "contrast", "outcome"]
                else:
                    release_scores = aggregate_stage_percentiles(
                        vectors @ event_stage_vectors["release"].T,
                        quantile=0.75,
                    )
                    outcome_scores = aggregate_stage_percentiles(
                        vectors @ event_stage_vectors["outcome"].T,
                        quantile=0.75,
                    )
                    plus_scores = aggregate_stage_percentiles(
                        vectors @ event_stage_vectors["plus"].T
                    )
                    miss_scores = aggregate_top_k_stage_percentiles(
                        vectors @ event_stage_vectors["miss"].T,
                        top_k=2,
                    )
                    transition_scores = aggregate_stage_percentiles(
                        vectors @ event_stage_vectors["transition"].T
                    )
                    event_windows = rank_made_free_throw_windows(
                        release_scores,
                        outcome_scores,
                        plus_scores,
                        miss_scores,
                        transition_scores,
                        release_lag=max(1, int(round(2.0 / step))),
                        plus_window=max(0, int(round(2.0 / step))),
                        reset_window=(
                            max(0, int(round(2.0 / step))),
                            max(0, int(round(5.0 / step))),
                        ),
                        negative_window=max(0, int(round(3.0 / step))),
                        suppression_radius=suppression_frames,
                        limit=limit,
                    )
                    stage_order = [
                        "release",
                        "outcome",
                        "plus",
                        "reset",
                        "miss",
                        "transition",
                    ]
                for window in event_windows:
                    release_item = metadata[window.release_index]
                    outcome_item = metadata[window.outcome_index]
                    followup_item = (
                        metadata[window.followup_index]
                        if window.followup_index is not None
                        else None
                    )
                    component_scores = {
                        "raw_release_score": window.raw_release_score,
                        "contrast_score": window.contrast_score,
                        "plus_score": window.plus_score,
                        "reset_score": window.reset_score,
                        "miss_score": window.miss_score,
                        "transition_score": window.transition_score,
                    }
                    event_metadata = {
                        name: value
                        for name, value in component_scores.items()
                        if value is not None
                    }
                    candidates.append(
                        EvidenceHit(
                            video_id=video_id,
                            segment_id=(
                                "visual-event:"
                                f"{release_item['segment_id']}:"
                                f"{outcome_item['segment_id']}"
                            ),
                            start=float(release_item["start"]),
                            end=float(outcome_item["end"]),
                            modality="visual",
                            score=window.score,
                            text=query,
                            metadata={
                                "source": "siglip2-temporal-event",
                                "event_type": event_specification.event_type,
                                "event_candidate": True,
                                "requires_ball_through_hoop": (
                                    event_specification.requires_ball_through_hoop
                                ),
                                "release_cue": event_specification.release_cue,
                                "outcome_cue": event_specification.outcome_cue,
                                "stage_order": stage_order,
                                "core_score": window.core_score,
                                "release_score": window.release_score,
                                "outcome_score": window.outcome_score,
                                "followup_score": window.followup_score,
                                "release_timestamp": float(release_item["start"]),
                                "outcome_timestamp": float(outcome_item["start"]),
                                "thumbnail_path": release_item.get("thumbnail_path"),
                                "temporal_event": True,
                                "temporal_refinement": True,
                                **event_metadata,
                                **(
                                    {
                                        "followup_timestamp": float(
                                            followup_item["start"]
                                        )
                                    }
                                    if followup_item is not None
                                    else {}
                                ),
                            },
                        )
                    )
                continue

            assert query_vectors is not None
            similarities = np.max(vectors @ query_vectors.T, axis=1)
            for index in np.argsort(similarities)[::-1][:limit]:
                item = metadata[int(index)]
                raw_similarity = float(similarities[int(index)])
                try:
                    item_start = float(item["start"])
                    item_end = float(item["end"])
                except (KeyError, TypeError, ValueError):
                    continue
                if (
                    not math.isfinite(raw_similarity)
                    or not math.isfinite(item_start)
                    or not math.isfinite(item_end)
                    or item_end <= item_start
                ):
                    continue
                candidates.append(
                    EvidenceHit(
                        video_id=video_id,
                        segment_id=f"visual:{item['segment_id']}",
                        start=item_start,
                        end=item_end,
                        modality="visual",
                        score=self._rank_score(raw_similarity),
                        text=query,
                        metadata={
                            "source": "siglip2",
                            "raw_similarity": raw_similarity,
                            "pair_probability": self._probability(raw_similarity),
                            "thumbnail_path": item.get("thumbnail_path"),
                        },
                    )
                )
        return sorted(candidates, key=lambda hit: hit.score, reverse=True)[:limit]
