from __future__ import annotations

import math
import json
from pathlib import Path

from videoscope.providers.base import ProviderState, ProviderStatus
from videoscope.providers.types import TimedText


class WhisperTranscriber:
    id = "whisper"

    def __init__(
        self,
        model: str,
        language: str = "auto",
        initial_prompt: str | None = None,
        glossary_path: Path | None = None,
    ) -> None:
        self.model = model
        self.language = language
        self.initial_prompt = initial_prompt
        self.glossary_path = Path(glossary_path) if glossary_path else None

    def _resolved_prompt(self) -> str | None:
        parts = [self.initial_prompt.strip()] if self.initial_prompt else []
        if self.glossary_path and self.glossary_path.is_file():
            try:
                payload = json.loads(self.glossary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            terms: list[str] = []
            if isinstance(payload, dict):
                for canonical, aliases in payload.items():
                    terms.append(str(canonical))
                    if isinstance(aliases, list):
                        terms.extend(str(alias) for alias in aliases)
            terms = list(dict.fromkeys(term.strip() for term in terms if term.strip()))[:120]
            if terms:
                parts.append("Словарь имён и терминов: " + ", ".join(terms) + ".")
        return " ".join(parts) or None

    def status(self) -> ProviderStatus:
        try:
            import mlx_whisper  # noqa: F401
        except ImportError:
            return ProviderStatus(
                self.id,
                "Whisper MLX",
                ProviderState.UNAVAILABLE,
                "mlx-whisper is not installed",
            )
        return ProviderStatus(
            self.id,
            "Whisper MLX",
            ProviderState.READY,
            f"Local speech recognition: {self.model}",
        )

    def transcribe(self, source: Path) -> list[TimedText]:
        import mlx_whisper

        options: dict[str, object] = {
            "path_or_hf_repo": self.model,
            "word_timestamps": True,
            "verbose": False,
            "temperature": 0.0,
            "condition_on_previous_text": True,
            "hallucination_silence_threshold": 2.0,
        }
        if self.language != "auto":
            options["language"] = self.language
        prompt = self._resolved_prompt()
        if prompt:
            options["initial_prompt"] = prompt
        result = mlx_whisper.transcribe(str(source), **options)
        detected_language = str(result.get("language") or self.language)
        segments: list[TimedText] = []
        for raw in result.get("segments") or []:
            text = str(raw.get("text") or "").strip()
            start = float(raw.get("start") or 0.0)
            end = float(raw.get("end") or start)
            if not text or end <= start:
                continue
            average_log_probability = float(raw.get("avg_logprob") or 0.0)
            confidence = max(0.0, min(1.0, math.exp(average_log_probability)))
            words = []
            for word in raw.get("words") or []:
                word_text = str(word.get("word") or "").strip()
                word_start = float(word.get("start") or start)
                word_end = float(word.get("end") or word_start)
                if not word_text or word_end <= word_start:
                    continue
                words.append(
                    {
                        "word": word_text,
                        "start": word_start,
                        "end": word_end,
                        "probability": max(0.0, min(1.0, float(word.get("probability") or confidence))),
                    }
                )
            segments.append(
                TimedText(
                    start=start,
                    end=end,
                    text=text,
                    confidence=confidence,
                    metadata={
                        "language": detected_language,
                        "engine": "mlx-whisper",
                        **({"words": words} if words else {}),
                    },
                )
            )
        return segments
