from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Protocol

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
from videoscope.storage import atomic_write_json, atomic_write_text


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
    ) -> None:
        self.path = Path(path)
        self.model_name = model_name
        self.model_revision = model_revision
        self.batch_size = max(1, batch_size)
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._logit_scale = 1.0
        self._logit_bias = 0.0

    @property
    def model_identity(self) -> str:
        return model_identity(self.model_name, self.model_revision)

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
                "Визуальный индекс создан другой ревизией модели; выполните make index-visual",
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
            if not video_path.is_dir() or not (video_path / "vectors.npy").is_file():
                continue
            model_path = video_path / "model.txt"
            try:
                stored_identity = model_path.read_text(encoding="utf-8")
            except OSError:
                return True
            if stored_identity != self.model_identity:
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

    def _video_paths(self, video_id: str) -> tuple[Path, Path]:
        target = self.path / video_id
        return target / "vectors.npy", target / "metadata.json"

    def _load_video_index(
        self,
        video_id: str,
        *,
        memory_map: bool = False,
    ) -> tuple[np.ndarray, list[dict[str, Any]]] | None:
        if (
            not video_id
            or video_id in {".", ".."}
            or Path(video_id).name != video_id
        ):
            return None
        vectors_path, metadata_path = self._video_paths(video_id)
        model_path = vectors_path.parent / "model.txt"
        if not vectors_path.is_file() or not metadata_path.is_file():
            return None
        try:
            if model_path.read_text(encoding="utf-8") != self.model_identity:
                return None
            vectors = np.load(
                vectors_path,
                allow_pickle=False,
                mmap_mode="r" if memory_map else None,
            )
            raw_metadata: object = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (EOFError, OSError, UnicodeError, ValueError, TypeError):
            return None
        if not isinstance(raw_metadata, list):
            return None
        metadata: list[dict[str, Any]] = []
        try:
            for raw_item in raw_metadata:
                if not isinstance(raw_item, dict):
                    raise ValueError("invalid visual metadata")
                start = float(raw_item["start"])
                end = float(raw_item["end"])
                segment_id = raw_item["segment_id"]
                thumbnail_path = raw_item.get("thumbnail_path")
                if (
                    type(segment_id) is not str
                    or not segment_id
                    or not math.isfinite(start)
                    or not math.isfinite(end)
                    or start < 0
                    or end <= start
                    or (thumbnail_path is not None and type(thumbnail_path) is not str)
                ):
                    raise ValueError("invalid visual metadata")
                metadata.append(
                    {
                        "segment_id": segment_id,
                        "start": start,
                        "end": end,
                        "thumbnail_path": thumbnail_path,
                    }
                )
        except (KeyError, TypeError, ValueError):
            return None
        try:
            vectors_are_valid = (
                vectors.ndim == 2
                and vectors.shape[0] > 0
                and vectors.shape[1] > 0
                and vectors.shape[0] == len(metadata)
                and bool(np.all(np.isfinite(vectors)))
            )
        except (TypeError, ValueError):
            return None
        if not vectors_are_valid:
            return None
        try:
            if model_path.read_text(encoding="utf-8") != self.model_identity:
                return None
        except (OSError, UnicodeError):
            return None
        return vectors, metadata

    def index_is_current(self, video_id: str) -> bool:
        """Return whether a complete index for this video matches this model revision."""
        return self._load_video_index(video_id, memory_map=True) is not None

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

    def _persist_video_index(
        self,
        vectors_path: Path,
        vectors: np.ndarray,
        metadata: list[dict[str, object]],
    ) -> None:
        vectors_path.parent.mkdir(parents=True, exist_ok=True)
        model_path = vectors_path.parent / "model.txt"
        resolved_vectors = vectors.astype(np.float32, copy=False)
        if (
            resolved_vectors.ndim != 2
            or resolved_vectors.shape[0] != len(metadata)
            or not np.all(np.isfinite(resolved_vectors))
        ):
            raise ValueError("SigLIP index vectors do not match finite metadata rows")

        # The identity marker is the commit record: an interrupted multi-file
        # replacement must never make mixed old/new files look current.
        model_path.unlink(missing_ok=True)
        self._atomic_save_vectors(vectors_path, resolved_vectors)
        atomic_write_json(vectors_path.parent / "metadata.json", metadata)
        atomic_write_text(model_path, self.model_identity)

    def replace_video(self, video_id: str, segments: list[SegmentRecord]) -> None:
        scenes = [
            segment
            for segment in segments
            if segment.modality == "scene"
            and segment.thumbnail_path
            and Path(segment.thumbnail_path).is_file()
        ]
        vectors_path, metadata_path = self._video_paths(video_id)
        model_path = vectors_path.parent / "model.txt"
        if not scenes:
            vectors_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            model_path.unlink(missing_ok=True)
            return

        vectors = self._image_vectors([Path(segment.thumbnail_path or "") for segment in scenes])
        metadata = [
            {
                "segment_id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "thumbnail_path": segment.thumbnail_path,
            }
            for segment in scenes
        ]
        self._persist_video_index(vectors_path, vectors, metadata)

    def replace_video_source(
        self,
        video_id: str,
        source: Path,
        duration: float,
        extractor: DenseFrameExtractor,
        *,
        step: float = 1.0,
        max_width: int = 640,
        frames_dir: Path | None = None,
    ) -> None:
        """Строит равномерный индекс, чтобы короткие действия не терялись между сценами."""
        duration = float(duration)
        step = float(step)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("video duration must be positive")
        if not math.isfinite(step) or step <= 0:
            raise ValueError("visual index step must be positive")
        step = max(0.25, step)
        target = self.path / video_id
        resolved_frames_dir = Path(frames_dir) if frames_dir else target / "frames"
        frames = extractor.extract_frames(
            Path(source),
            resolved_frames_dir,
            0.0,
            duration,
            step=step,
            max_width=max_width,
        )
        paths = [Path(getattr(frame, "path")) for frame in frames]
        timestamps = [float(getattr(frame, "timestamp")) for frame in frames]
        if any(not math.isfinite(timestamp) or timestamp < 0 for timestamp in timestamps):
            raise ValueError("visual frame timestamps must be finite and non-negative")
        if not paths:
            self.replace_video(video_id, [])
            return

        vectors_path, _ = self._video_paths(video_id)
        vectors = self._image_vectors(paths)
        metadata = [
            {
                "segment_id": f"dense-{index:06d}",
                "start": timestamp,
                "end": min(duration, timestamp + step),
                "thumbnail_path": str(path),
            }
            for index, (timestamp, path) in enumerate(zip(timestamps, paths, strict=True))
        ]
        self._persist_video_index(vectors_path, vectors, metadata)

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
