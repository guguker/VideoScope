from __future__ import annotations

import base64
import json
import math
import os
from pathlib import Path
import re

from videoscope.providers.base import ProviderState, ProviderStatus
from videoscope.providers.types import ObjectTag


LOCAL_MODELS = {
    "rfdetr-nano": "RFDETRNano",
    "rfdetr-small": "RFDETRSmall",
    "rfdetr-medium": "RFDETRMedium",
    "rfdetr-large": "RFDETRLarge",
}
_HOSTED_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_MAX_HOSTED_IMAGE_BYTES = 20 * 1024 * 1024
_MAX_HOSTED_RESPONSE_BYTES = 10 * 1024 * 1024


def _reject_non_finite_json(value: str) -> None:
    raise ValueError(f"non-finite Roboflow JSON: {value}")


class RoboflowDetector:
    id = "roboflow"

    def __init__(
        self,
        *,
        api_key: str | None,
        model_id: str | None,
        api_url: str = "https://serverless.roboflow.com",
        minimum_confidence: float = 0.25,
        cache_dir: Path | None = None,
    ) -> None:
        self.api_key = api_key
        self.model_id = model_id
        self.api_url = api_url
        self.minimum_confidence = minimum_confidence
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._backend = None

    @property
    def is_local(self) -> bool:
        return bool(self.model_id in LOCAL_MODELS)

    def status(self) -> ProviderStatus:
        if self.is_local:
            try:
                import rfdetr  # noqa: F401
                import supervision  # noqa: F401
            except ImportError:
                return ProviderStatus(
                    self.id,
                    "Roboflow RF-DETR",
                    ProviderState.UNAVAILABLE,
                    "Установите пакеты rfdetr и supervision",
                    optional=True,
                )
            return ProviderStatus(
                self.id,
                "Roboflow RF-DETR",
                ProviderState.READY,
                f"Локальное обнаружение объектов: {self.model_id} "
                "(кадры не покидают устройство)",
                optional=True,
            )
        if not self.api_key:
            return ProviderStatus(
                self.id,
                "Roboflow",
                ProviderState.NEEDS_CONFIGURATION,
                f"Задайте ROBOFLOW_API_KEY "
                f"(модель: {self.model_id or 'не выбрана'})",
                optional=True,
            )
        if not self.model_id:
            return ProviderStatus(
                self.id,
                "Roboflow",
                ProviderState.NEEDS_CONFIGURATION,
                "Задайте ROBOFLOW_MODEL_ID",
                optional=True,
            )
        if _HOSTED_MODEL_ID_RE.fullmatch(self.model_id) is None:
            return ProviderStatus(
                self.id,
                "Roboflow",
                ProviderState.NEEDS_CONFIGURATION,
                "Идентификатор облачной модели должен иметь вид project/version",
                optional=True,
            )
        destination = "local server" if self.api_url.startswith("http://127.0.0.1") else "Roboflow API"
        return ProviderStatus(
            self.id,
            "Roboflow",
            ProviderState.READY,
            f"Object detection via {destination}: {self.model_id}",
            optional=True,
        )

    def _load(self):  # type: ignore[no-untyped-def]
        if self._backend is not None:
            return self._backend
        if self.is_local:
            if self.cache_dir is not None:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                os.environ.setdefault("RF_HOME", str(self.cache_dir.resolve()))
            import rfdetr

            model_class = getattr(rfdetr, LOCAL_MODELS[str(self.model_id)])
            self._backend = ("local", model_class())
            return self._backend
        if not self.api_key or not self.model_id:
            raise RuntimeError("Roboflow is not configured")
        import httpx

        client = httpx.Client(timeout=60.0, follow_redirects=False)
        self._backend = ("hosted", client)
        return self._backend

    def detect(self, image: Path) -> list[ObjectTag]:
        backend, detector = self._load()
        if backend == "local":
            return self._detect_local(detector, image)

        if _HOSTED_MODEL_ID_RE.fullmatch(str(self.model_id)) is None:
            raise RuntimeError("Roboflow model id is invalid")
        try:
            image_size = image.stat().st_size
        except OSError as error:
            raise RuntimeError("Roboflow frame cannot be read") from error
        if image_size <= 0 or image_size > _MAX_HOSTED_IMAGE_BYTES:
            raise RuntimeError("Roboflow frame size is invalid")
        encoded = base64.b64encode(image.read_bytes()).decode("ascii")
        result = detector.post(
            f"{self.api_url.rstrip('/')}/{self.model_id}",
            params={"api_key": self.api_key},
            content=encoded,
            headers={"content-type": "text/plain; charset=us-ascii"},
        )
        declared_size = result.headers.get("content-length")
        if declared_size is not None:
            try:
                if int(declared_size) > _MAX_HOSTED_RESPONSE_BYTES:
                    raise RuntimeError("Roboflow response is too large")
            except ValueError as error:
                raise RuntimeError("Roboflow response size is invalid") from error
        result.raise_for_status()
        if len(result.content) > _MAX_HOSTED_RESPONSE_BYTES:
            raise RuntimeError("Roboflow response is too large")
        try:
            response = json.loads(
                result.content,
                parse_constant=_reject_non_finite_json,
            )
        except (UnicodeError, ValueError, TypeError) as error:
            raise RuntimeError("Roboflow response is invalid") from error
        if isinstance(response, list):
            response = response[0] if response else {}
        if not isinstance(response, dict):
            return []
        predictions = response.get("predictions") or []
        if not isinstance(predictions, list):
            return []
        tags: list[ObjectTag] = []
        for prediction in predictions[:1_000]:
            if not isinstance(prediction, dict):
                continue
            try:
                confidence = float(prediction["confidence"])
            except (KeyError, TypeError, ValueError):
                continue
            if (
                not math.isfinite(confidence)
                or not 0 <= confidence <= 1
                or confidence < self.minimum_confidence
            ):
                continue
            label = str(prediction.get("class") or "unknown")[:120]
            metadata: dict[str, float | None] = {}
            valid_box = True
            for name in ("x", "y", "width", "height"):
                value = prediction.get(name)
                if value is None:
                    metadata[name] = None
                    continue
                try:
                    resolved = float(value)
                except (TypeError, ValueError):
                    valid_box = False
                    break
                if not math.isfinite(resolved):
                    valid_box = False
                    break
                metadata[name] = resolved
            if not valid_box:
                continue
            tags.append(
                ObjectTag(
                    label=label,
                    confidence=confidence,
                    metadata=metadata,
                )
            )
        return tags

    def _detect_local(self, model, image: Path) -> list[ObjectTag]:  # type: ignore[no-untyped-def]
        from rfdetr.assets.coco_classes import COCO_CLASSES

        detections = model.predict(
            str(image),
            threshold=self.minimum_confidence,
            include_source_image=False,
        )
        if isinstance(detections, list):
            detections = detections[0] if detections else None
        if detections is None:
            return []

        class_names = getattr(detections, "data", {}).get("class_name")
        tags: list[ObjectTag] = []
        for index in range(len(detections)):
            confidence = float(detections.confidence[index])
            class_id = int(detections.class_id[index])
            x1, y1, x2, y2 = map(float, detections.xyxy[index])
            label = class_names[index] if class_names is not None else None
            tags.append(
                ObjectTag(
                    label=str(label or COCO_CLASSES.get(class_id, f"class-{class_id}")),
                    confidence=confidence,
                    metadata={
                        "x": (x1 + x2) / 2,
                        "y": (y1 + y2) / 2,
                        "width": x2 - x1,
                        "height": y2 - y1,
                    },
                )
            )
        return tags
