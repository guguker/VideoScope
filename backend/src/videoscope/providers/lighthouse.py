from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from videoscope.media.ffmpeg import FFmpeg
from videoscope.providers.base import ProviderState, ProviderStatus
from videoscope.search.fusion import EvidenceHit


class LighthouseRetriever:
    id = "lighthouse"
    max_window_seconds = 150.0

    def __init__(
        self,
        *,
        checkpoint: Path | None,
        cache_dir: Path,
        ffmpeg: FFmpeg,
        source_root: Path | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.cache_dir = Path(cache_dir) / "lighthouse"
        self.ffmpeg = ffmpeg
        self.source_root = Path(source_root) if source_root else None
        self._model = None
        if self.source_root and str(self.source_root) not in sys.path:
            sys.path.insert(0, str(self.source_root))

    @staticmethod
    def _predictor_class():  # type: ignore[no-untyped-def]
        try:
            from lighthouse.models import QDDETRPredictor

            return QDDETRPredictor, "official Lighthouse API"
        except Exception:
            from videoscope.providers.lighthouse_qdetr import QDDETRPredictor

            return QDDETRPredictor, "адаптер совместимости VideoScope с CLIP"

    def status(self) -> ProviderStatus:
        if self.checkpoint is None or not self.checkpoint.is_file():
            return ProviderStatus(
                self.id,
                "Lighthouse QD-DETR",
                ProviderState.NEEDS_CONFIGURATION,
                "A QD-DETR checkpoint is required",
                optional=True,
            )
        try:
            _, implementation = self._predictor_class()
        except (ImportError, ModuleNotFoundError):
            return ProviderStatus(
                self.id,
                "Lighthouse QD-DETR",
                ProviderState.UNAVAILABLE,
                "Lighthouse не установлен",
                optional=True,
            )
        return ProviderStatus(
            self.id,
            "Lighthouse QD-DETR",
            ProviderState.READY,
            f"Обработка на CPU с признаками CLIP и окнами по 150 секунд "
            f"({implementation})",
            optional=True,
        )

    def _load(self):  # type: ignore[no-untyped-def]
        if self._model is None:
            if self.checkpoint is None:
                raise RuntimeError("Lighthouse checkpoint is not configured")
            predictor_class, _ = self._predictor_class()

            self._model = predictor_class(
                str(self.checkpoint),
                device="cpu",
                feature_name="clip",
            )
        return self._model

    def prepare(self, video_id: str, source: Path, duration: float) -> None:
        import torch

        target_dir = self.cache_dir / video_id
        target_dir.mkdir(parents=True, exist_ok=True)
        model = self._load()
        offset = 0.0
        window_index = 0
        while offset < duration:
            end = min(duration, offset + self.max_window_seconds)
            window_path = target_dir / f"window-{window_index:04d}.mp4"
            cache_path = target_dir / f"window-{window_index:04d}.pt"
            self.ffmpeg.export_clip(source, window_path, offset, end)
            encoded = model.encode_video(str(window_path))
            torch.save(
                {"offset": offset, "end": end, "features": encoded},
                cache_path,
            )
            window_path.unlink(missing_ok=True)
            offset = end
            window_index += 1

    def search(self, query: str, video_ids: list[str], *, limit: int = 30) -> list[EvidenceHit]:
        import torch

        model = self._load()
        hits: list[EvidenceHit] = []
        for video_id in video_ids:
            for cache_path in sorted((self.cache_dir / video_id).glob("window-*.pt")):
                payload: dict[str, Any] = torch.load(cache_path, map_location="cpu", weights_only=False)
                offset = float(payload["offset"])
                window_end = float(payload["end"])
                prediction = model.predict(query, payload["features"])
                for index, window in enumerate(prediction.get("pred_relevant_windows") or []):
                    start, end, score = map(float, window)
                    absolute_start = max(offset, offset + start)
                    absolute_end = min(window_end, offset + end)
                    if absolute_end <= absolute_start:
                        continue
                    hits.append(
                        EvidenceHit(
                            video_id=video_id,
                            segment_id=f"lighthouse:{video_id}:{cache_path.stem}:{index}",
                            start=absolute_start,
                            end=absolute_end,
                            modality="lighthouse",
                            score=max(0.0, min(1.0, score)),
                            text=query,
                            metadata={"source": "lighthouse", "window": cache_path.stem},
                        )
                    )
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:limit]
