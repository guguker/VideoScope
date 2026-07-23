from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from videoscope.providers.base import ProviderState, ProviderStatus
from videoscope.repository import SegmentRecord
from videoscope.search.fusion import EvidenceHit


class SiglipVisualIndex:
    """Многоязычный поиск соответствий между кадрами и текстом на базе SigLIP 2."""

    id = "siglip2"

    def __init__(self, path: Path, *, model_name: str, batch_size: int = 8) -> None:
        self.path = Path(path)
        self.model_name = model_name
        self.batch_size = max(1, batch_size)
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._logit_scale = 1.0
        self._logit_bias = 0.0

    def status(self) -> ProviderStatus:
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

                config = try_to_load_from_cache(self.model_name, "config.json")
                weights = try_to_load_from_cache(self.model_name, "model.safetensors")
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
        return ProviderStatus(
            self.id,
            "SigLIP 2",
            ProviderState.READY,
            f"Многоязычный визуальный поиск: {self.model_name}",
        )

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
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
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
        prompts = [query]
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
        vectors_path.parent.mkdir(parents=True, exist_ok=True)
        if not scenes:
            vectors_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            model_path.unlink(missing_ok=True)
            return

        vectors = self._image_vectors([Path(segment.thumbnail_path or "") for segment in scenes])
        np.save(vectors_path, vectors.astype(np.float32, copy=False), allow_pickle=False)
        metadata = [
            {
                "segment_id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "thumbnail_path": segment.thumbnail_path,
            }
            for segment in scenes
        ]
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
        model_path.write_text(self.model_name, encoding="utf-8")

    def _probability(self, similarity: float) -> float:
        logit = similarity * self._logit_scale + self._logit_bias
        if logit >= 0:
            return 1.0 / (1.0 + math.exp(-logit))
        exp_logit = math.exp(logit)
        return exp_logit / (1.0 + exp_logit)

    @staticmethod
    def _rank_score(similarity: float) -> float:
        """Преобразует близость SigLIP в шкалу ранжирования, не выдавая её за вероятность."""
        logit = (similarity - 0.05) * 20.0
        if logit >= 0:
            return 1.0 / (1.0 + math.exp(-logit))
        exp_logit = math.exp(logit)
        return exp_logit / (1.0 + exp_logit)

    def search(
        self,
        query: str,
        *,
        video_ids: list[str] | None = None,
        limit: int = 50,
    ) -> list[EvidenceHit]:
        if not query.strip() or limit <= 0 or not self.path.is_dir():
            return []
        query_vectors = self._text_vectors(query)
        candidates: list[EvidenceHit] = []
        selected = video_ids or [path.name for path in self.path.iterdir() if path.is_dir()]
        for video_id in selected:
            vectors_path, metadata_path = self._video_paths(video_id)
            model_path = vectors_path.parent / "model.txt"
            if (
                not vectors_path.is_file()
                or not metadata_path.is_file()
                or (
                    model_path.is_file()
                    and model_path.read_text(encoding="utf-8") != self.model_name
                )
                or (
                    not model_path.is_file()
                    and self.model_name != "google/siglip2-base-patch16-224"
                )
            ):
                continue
            vectors = np.load(vectors_path, allow_pickle=False)
            metadata: list[dict[str, Any]] = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                vectors.ndim != 2
                or vectors.shape[0] != len(metadata)
                or vectors.shape[1] != query_vectors.shape[1]
            ):
                continue
            similarities = np.max(vectors @ query_vectors.T, axis=1)
            for index in np.argsort(similarities)[::-1][:limit]:
                item = metadata[int(index)]
                raw_similarity = float(similarities[int(index)])
                candidates.append(
                    EvidenceHit(
                        video_id=video_id,
                        segment_id=f"visual:{item['segment_id']}",
                        start=float(item["start"]),
                        end=float(item["end"]),
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
