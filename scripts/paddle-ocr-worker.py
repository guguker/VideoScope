#!/usr/bin/env python3
from __future__ import annotations

from contextlib import redirect_stdout
import json
import logging
import math
from pathlib import Path
import sys
from typing import Any


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("videoscope.paddle_ocr_worker")


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


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def main() -> int:
    protocol_output = sys.stdout
    with redirect_stdout(sys.stderr):
        from paddleocr import PaddleOCR

        model = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            engine="transformers",
        )

    for line in sys.stdin:
        response: dict[str, object]
        try:
            request = json.loads(line, parse_constant=_reject_constant)
            if set(request) != {"path"} or type(request["path"]) is not str:
                raise ValueError("invalid request")
            image = Path(request["path"])
            if not image.is_file():
                raise ValueError("image does not exist")
            items: list[list[object]] = []
            with redirect_stdout(sys.stderr):
                results = model.predict(str(image))
            for result in results:
                payload = _result_payload(result)
                texts = payload.get("rec_texts") or []
                scores = payload.get("rec_scores") or []
                for text, score in zip(texts, scores, strict=False):
                    normalized = str(text).strip()
                    confidence = float(score)
                    if normalized and math.isfinite(confidence) and 0 <= confidence <= 1:
                        items.append([normalized, confidence])
            response = {"ok": True, "items": items}
        except Exception:
            logger.exception("PaddleOCR worker request failed")
            response = {"ok": False}
        protocol_output.write(
            json.dumps(response, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            + "\n"
        )
        protocol_output.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
