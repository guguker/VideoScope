from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import threading
from typing import Any

from videoscope.providers.base import ProviderState, ProviderStatus


def _reject_non_finite_json(value: str) -> None:
    raise ValueError(f"non-finite worker JSON: {value}")


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

    def __init__(
        self,
        *,
        minimum_confidence: float = 0.55,
        worker_python: Path | None = None,
        worker_script: Path | None = None,
    ) -> None:
        self.minimum_confidence = minimum_confidence
        self.worker_python = Path(worker_python) if worker_python else None
        self.worker_script = Path(worker_script) if worker_script else None
        self._model = None
        self._worker_process: subprocess.Popen[str] | None = None
        self._worker_lock = threading.Lock()

    @property
    def _uses_worker(self) -> bool:
        return self.worker_python is not None or self.worker_script is not None

    def status(self) -> ProviderStatus:
        if self._uses_worker:
            if (
                self.worker_python is None
                or self.worker_script is None
                or not self.worker_python.is_file()
                or not self.worker_script.is_file()
            ):
                return ProviderStatus(
                    self.id,
                    "PaddleOCR",
                    ProviderState.UNAVAILABLE,
                    "Изолированный OCR worker не установлен; выполните make install-ocr",
                    optional=True,
                )
            try:
                probe = subprocess.run(
                    [str(self.worker_python), "-c", "import paddleocr"],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=20,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError):
                probe = None
            if probe is None or probe.returncode != 0:
                return ProviderStatus(
                    self.id,
                    "PaddleOCR",
                    ProviderState.UNAVAILABLE,
                    "Изолированное OCR-окружение повреждено; повторите make install-ocr",
                    optional=True,
                )
            return ProviderStatus(
                self.id,
                "PaddleOCR",
                ProviderState.READY,
                "PP-OCR в отдельном совместимом процессе",
                optional=True,
            )
        try:
            import paddleocr  # noqa: F401
        except ImportError:
            return ProviderStatus(
                self.id,
                "PaddleOCR",
                ProviderState.UNAVAILABLE,
                "PaddleOCR не установлен",
                optional=True,
            )
        return ProviderStatus(
            self.id,
            "PaddleOCR",
            ProviderState.READY,
            "PP-OCR с модельным движком Transformers",
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

    def _load_worker(self) -> subprocess.Popen[str]:
        if self._worker_process is not None and self._worker_process.poll() is None:
            return self._worker_process
        if self.worker_python is None or self.worker_script is None:
            raise RuntimeError("PaddleOCR worker is not configured")
        process = subprocess.Popen(
            [str(self.worker_python), str(self.worker_script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if process.stdin is None or process.stdout is None:
            process.terminate()
            raise RuntimeError("PaddleOCR worker pipes are unavailable")
        self._worker_process = process
        return process

    def close(self) -> None:
        process = self._worker_process
        self._worker_process = None
        if process is not None and process.poll() is None:
            process.terminate()

    def _read_worker(self, image: Path) -> list[tuple[str, float]]:
        with self._worker_lock:
            process = self._load_worker()
            assert process.stdin is not None
            assert process.stdout is not None
            try:
                process.stdin.write(
                    json.dumps(
                        {"path": str(image)},
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                process.stdin.flush()
                line = process.stdout.readline()
            except (BrokenPipeError, OSError) as error:
                self._worker_process = None
                raise RuntimeError("PaddleOCR worker failed") from error
        if not line:
            self._worker_process = None
            raise RuntimeError("PaddleOCR worker stopped")
        try:
            payload = json.loads(
                line,
                parse_constant=_reject_non_finite_json,
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise RuntimeError("PaddleOCR worker returned invalid data") from error
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            raise RuntimeError("PaddleOCR worker could not process the frame")
        return self._normalized_items(payload.get("items"))

    def _normalized_items(self, items: object) -> list[tuple[str, float]]:
        if not isinstance(items, list):
            return []
        output: list[tuple[str, float]] = []
        for item in items:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            try:
                normalized = str(item[0]).strip()
                confidence = float(item[1])
            except (TypeError, ValueError):
                continue
            if (
                normalized
                and math.isfinite(confidence)
                and self.minimum_confidence <= confidence <= 1
            ):
                output.append((normalized, confidence))
        return output

    def read(self, image: Path) -> list[tuple[str, float]]:
        if self._uses_worker:
            return self._read_worker(image)
        items: list[list[object]] = []
        for result in self._load().predict(str(image)):
            payload = _result_payload(result)
            texts = payload.get("rec_texts") or []
            scores = payload.get("rec_scores") or []
            for text, score in zip(texts, scores, strict=False):
                items.append([text, score])
        return self._normalized_items(items)
