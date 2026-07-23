from __future__ import annotations

import os
from pathlib import Path

from videoscope.providers.base import ProviderState, ProviderStatus
from videoscope.providers.types import ObjectTag


LOCAL_MODELS = {
    "rfdetr-nano": "RFDETRNano",
    "rfdetr-small": "RFDETRSmall",
    "rfdetr-medium": "RFDETRMedium",
    "rfdetr-large": "RFDETRLarge",
}


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
                    "Install rfdetr and supervision",
                    optional=True,
                )
            return ProviderStatus(
                self.id,
                "Roboflow RF-DETR",
                ProviderState.READY,
                f"Local object detection: {self.model_id} (frames stay on device)",
                optional=True,
            )
        if not self.api_key:
            return ProviderStatus(
                self.id,
                "Roboflow",
                ProviderState.NEEDS_CONFIGURATION,
                f"Set ROBOFLOW_API_KEY (model: {self.model_id or 'not selected'})",
                optional=True,
            )
        if not self.model_id:
            return ProviderStatus(
                self.id,
                "Roboflow",
                ProviderState.NEEDS_CONFIGURATION,
                "Set ROBOFLOW_MODEL_ID",
                optional=True,
            )
        try:
            import inference_sdk  # noqa: F401
            import supervision  # noqa: F401
        except ImportError:
            return ProviderStatus(
                self.id,
                "Roboflow",
                ProviderState.UNAVAILABLE,
                "inference-sdk or supervision is not installed",
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
        from inference_sdk import InferenceHTTPClient

        client = InferenceHTTPClient(api_url=self.api_url, api_key=self.api_key)
        self._backend = ("hosted", client)
        return self._backend

    def detect(self, image: Path) -> list[ObjectTag]:
        backend, detector = self._load()
        if backend == "local":
            return self._detect_local(detector, image)

        import supervision as sv

        response = detector.infer(str(image), model_id=self.model_id)
        if isinstance(response, list):
            response = response[0] if response else {}
        if not isinstance(response, dict):
            return []

        # Supervision validates the Roboflow response shape and applies class-aware NMS.
        detections = sv.Detections.from_inference(response).with_nms(threshold=0.5)
        predictions = response.get("predictions") or []
        tags: list[ObjectTag] = []
        for index in range(len(detections)):
            prediction = predictions[index] if index < len(predictions) else {}
            confidence = float(detections.confidence[index]) if detections.confidence is not None else 0.0
            if confidence < self.minimum_confidence:
                continue
            label = str(prediction.get("class") or f"class-{detections.class_id[index]}")
            tags.append(
                ObjectTag(
                    label=label,
                    confidence=confidence,
                    metadata={
                        "x": prediction.get("x"),
                        "y": prediction.get("y"),
                        "width": prediction.get("width"),
                        "height": prediction.get("height"),
                    },
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
