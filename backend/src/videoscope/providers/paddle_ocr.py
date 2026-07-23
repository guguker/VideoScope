from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from videoscope.providers.base import ProviderState, ProviderStatus


def _result_payload(result: object) -> dict[str, Any]:
    payload: object = getattr(result, "json", result)
    if callable(payload):
        payload = payload()
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        return {}
    nested = payload.get("res")
    return nested if isinstance(nested, dict) else payload


class PaddleOCRReader:
    id = "paddleocr"

    def __init__(self, *, minimum_confidence: float = 0.55) -> None:
        self.minimum_confidence = minimum_confidence
        self._model = None

    def status(self) -> ProviderStatus:
        try:
            import paddleocr  # noqa: F401
        except ImportError:
            return ProviderStatus(
                self.id,
                "PaddleOCR",
                ProviderState.UNAVAILABLE,
                "paddleocr is not installed",
                optional=True,
            )
        return ProviderStatus(
            self.id,
            "PaddleOCR",
            ProviderState.READY,
            "PP-OCR with the Transformers engine",
            optional=True,
        )

    def _load(self):  # type: ignore[no-untyped-def]
        if self._model is None:
            from paddleocr import PaddleOCR

            self._model = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                engine="transformers",
            )
        return self._model

    def read(self, image: Path) -> list[tuple[str, float]]:
        output: list[tuple[str, float]] = []
        for result in self._load().predict(str(image)):
            payload = _result_payload(result)
            texts = payload.get("rec_texts") or []
            scores = payload.get("rec_scores") or []
            for text, score in zip(texts, scores, strict=False):
                normalized = str(text).strip()
                confidence = float(score)
                if normalized and confidence >= self.minimum_confidence:
                    output.append((normalized, confidence))
        return output
