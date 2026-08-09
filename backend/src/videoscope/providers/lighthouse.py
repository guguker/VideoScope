from __future__ import annotations

import hashlib
import hmac
import json
import math
import sys
from pathlib import Path
from typing import Any

from videoscope.media.ffmpeg import FFmpeg
from videoscope.model_manifest import (
    LIGHTHOUSE_CHECKPOINT_SHA256,
    LIGHTHOUSE_CLIP_REVISION,
    LIGHTHOUSE_SOURCE_REVISION,
)
from videoscope.providers.base import ProviderState, ProviderStatus
from videoscope.search.fusion import EvidenceHit
from videoscope.storage import atomic_write_json


class LighthouseRetriever:
    id = "lighthouse"
    max_window_seconds = 150.0
    cache_schema_version = 1

    def __init__(
        self,
        *,
        checkpoint: Path | None,
        checkpoint_sha256: str = LIGHTHOUSE_CHECKPOINT_SHA256,
        cache_dir: Path,
        ffmpeg: FFmpeg,
        source_root: Path | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.checkpoint_sha256 = checkpoint_sha256
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

    def _checkpoint_is_trusted(self) -> bool:
        if self.checkpoint is None or not self.checkpoint.is_file():
            return False
        digest = hashlib.sha256()
        try:
            with self.checkpoint.open("rb") as checkpoint_file:
                for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError:
            return False
        return hmac.compare_digest(digest.hexdigest(), self.checkpoint_sha256)

    @property
    def cache_identity(self) -> dict[str, object]:
        predictor_class, _ = self._predictor_class()
        return {
            "schema_version": self.cache_schema_version,
            "checkpoint_sha256": self.checkpoint_sha256,
            "lighthouse_revision": LIGHTHOUSE_SOURCE_REVISION,
            "clip_revision": LIGHTHOUSE_CLIP_REVISION,
            "predictor": f"{predictor_class.__module__}.{predictor_class.__qualname__}",
            "feature_name": "clip",
            "max_window_seconds": self.max_window_seconds,
        }

    def cache_is_current(self, video_id: str) -> bool:
        return self._load_current_cache(video_id) is not None

    @staticmethod
    def _features_are_usable(features: object, torch: Any) -> bool:
        if not isinstance(features, dict) or set(features) != {
            "video_feats",
            "video_mask",
            "audio_feats",
        }:
            return False
        video_features = features["video_feats"]
        video_mask = features["video_mask"]
        if (
            not isinstance(video_features, torch.Tensor)
            or not isinstance(video_mask, torch.Tensor)
            or features["audio_feats"] is not None
            or not video_features.is_floating_point()
            or video_features.ndim != 3
            or video_features.shape[0] != 1
            or not 1 <= video_features.shape[1] <= 75
            or video_features.shape[2] <= 2
            or video_mask.ndim != 2
            or video_mask.shape != video_features.shape[:2]
        ):
            return False
        try:
            return bool(torch.isfinite(video_features).all().item()) and bool(
                torch.isfinite(video_mask).all().item()
            ) and bool(
                ((video_mask == 0) | (video_mask == 1)).all().item()
            ) and bool(video_mask.bool().any().item())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False

    def _load_cache_window(
        self,
        cache_path: Path,
    ) -> tuple[float, float, object] | None:
        try:
            import torch

            payload: object = torch.load(
                cache_path,
                map_location="cpu",
                weights_only=True,
            )
            if not isinstance(payload, dict) or set(payload) != {
                "offset",
                "end",
                "features",
            }:
                return None
            raw_offset = payload["offset"]
            raw_end = payload["end"]
            if isinstance(raw_offset, bool) or isinstance(raw_end, bool):
                return None
            offset = float(raw_offset)
            end = float(raw_end)
            if (
                not math.isfinite(offset)
                or not math.isfinite(end)
                or offset < 0
                or end <= offset
                or end - offset > self.max_window_seconds + 1e-6
                or not self._features_are_usable(payload["features"], torch)
            ):
                return None
            return offset, end, payload["features"]
        except Exception:
            # Cache files are derived, untrusted state. A safe weights-only load
            # failure makes the cache stale; it must never become an eval miss.
            return None

    def _load_current_cache(
        self,
        video_id: str,
    ) -> list[tuple[Path, float, float, object]] | None:
        if (
            not video_id
            or video_id in {".", ".."}
            or Path(video_id).name != video_id
        ):
            return None
        target_dir = self.cache_dir / video_id
        manifest = target_dir / "manifest.json"
        try:
            expected_identity = self.cache_identity
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError, TypeError, ImportError):
            return None
        if payload != expected_identity:
            return None

        cache_paths = sorted(target_dir.glob("window-*.pt"))
        if not cache_paths:
            return None
        loaded: list[tuple[Path, float, float, object]] = []
        expected_offset = 0.0
        for index, cache_path in enumerate(cache_paths):
            if cache_path.name != f"window-{index:04d}.pt":
                return None
            window = self._load_cache_window(cache_path)
            if window is None:
                return None
            offset, end, features = window
            if not math.isclose(
                offset,
                expected_offset,
                rel_tol=1e-9,
                abs_tol=1e-6,
            ):
                return None
            loaded.append((cache_path, offset, end, features))
            expected_offset = end

        try:
            committed_identity = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError, TypeError):
            return None
        if committed_identity != expected_identity:
            return None
        return loaded

    def status(self) -> ProviderStatus:
        if self.checkpoint is None or not self.checkpoint.is_file():
            return ProviderStatus(
                self.id,
                "Lighthouse QD-DETR",
                ProviderState.NEEDS_CONFIGURATION,
                "A QD-DETR checkpoint is required",
                optional=True,
            )
        if not self._checkpoint_is_trusted():
            return ProviderStatus(
                self.id,
                "Lighthouse QD-DETR",
                ProviderState.NEEDS_CONFIGURATION,
                "QD-DETR checkpoint checksum mismatch; reinstall it with make install-lighthouse",
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
            if not self._checkpoint_is_trusted():
                raise RuntimeError("Lighthouse checkpoint checksum mismatch")
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
        for stale_path in target_dir.glob("window-*"):
            if stale_path.is_file() or stale_path.is_symlink():
                stale_path.unlink(missing_ok=True)
        (target_dir / "manifest.json").unlink(missing_ok=True)
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
        atomic_write_json(target_dir / "manifest.json", self.cache_identity, sort_keys=True)

    def search(self, query: str, video_ids: list[str], *, limit: int = 30) -> list[EvidenceHit]:
        model = None
        hits: list[EvidenceHit] = []
        for video_id in video_ids:
            cached_windows = self._load_current_cache(video_id)
            if cached_windows is None:
                continue
            if model is None:
                model = self._load()
            for cache_path, offset, window_end, features in cached_windows:
                prediction = model.predict(query, features)
                for index, window in enumerate(prediction.get("pred_relevant_windows") or []):
                    try:
                        start, end, score = map(float, window)
                    except (TypeError, ValueError):
                        continue
                    if not all(math.isfinite(value) for value in (start, end, score)):
                        continue
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
