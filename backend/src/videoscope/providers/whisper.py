from __future__ import annotations

import math
import json
from pathlib import Path
import stat

from videoscope.providers.base import ProviderState, ProviderStatus
from videoscope.providers.types import TimedText


MAX_GLOSSARY_BYTES = 1024 * 1024


def resolve_whisper_prompt(
    initial_prompt: str | None,
    glossary_path: Path | None,
) -> tuple[str | None, str]:
    """Resolve the exact bounded prompt and a stable glossary state identity."""
    parts = [initial_prompt.strip()] if initial_prompt else []
    state = "not_configured" if glossary_path is None else "missing"
    payload: object = {}
    if glossary_path is not None:
        path = Path(glossary_path)
        try:
            file_stat = path.lstat()
            if path.is_symlink() or not stat.S_ISREG(file_stat.st_mode):
                state = "unsafe"
            elif file_stat.st_size > MAX_GLOSSARY_BYTES:
                state = "oversized"
            else:
                raw = path.read_bytes()
                if len(raw) != file_stat.st_size:
                    state = "changed"
                else:
                    payload = json.loads(raw.decode("utf-8"))
                    state = "ready" if isinstance(payload, dict) else "invalid"
        except FileNotFoundError:
            state = "missing"
        except PermissionError:
            state = "unreadable"
        except (OSError, UnicodeError, json.JSONDecodeError):
            state = "invalid"
    terms: list[str] = []
    if state == "ready" and isinstance(payload, dict):
        for canonical, aliases in payload.items():
            terms.append(str(canonical))
            if isinstance(aliases, list):
                terms.extend(str(alias) for alias in aliases)
    terms = list(dict.fromkeys(term.strip() for term in terms if term.strip()))[:120]
    if terms:
        parts.append("Словарь имён и терминов: " + ", ".join(terms) + ".")
    return " ".join(parts) or None, state


class WhisperTranscriber:
    id = "whisper"

    def __init__(
        self,
        model: str,
        language: str = "auto",
        initial_prompt: str | None = None,
        glossary_path: Path | None = None,
        model_revision: str | None = None,
    ) -> None:
        self.model = model
        self.language = language
        self.initial_prompt = initial_prompt
        self.glossary_path = Path(glossary_path) if glossary_path else None
        self.model_revision = model_revision

    def _model_reference(self) -> str:
        model_path = Path(self.model).expanduser()
        if model_path.exists() or self.model_revision is None:
            return str(model_path) if model_path.exists() else self.model
        from huggingface_hub import snapshot_download

        return snapshot_download(
            self.model,
            revision=self.model_revision,
            local_files_only=True,
        )

    def _resolved_prompt(self) -> str | None:
        prompt, _state = resolve_whisper_prompt(
            self.initial_prompt,
            self.glossary_path,
        )
        return prompt

    def status(self) -> ProviderStatus:
        try:
            import mlx_whisper  # noqa: F401
        except ImportError:
            return ProviderStatus(
                self.id,
                "Whisper MLX",
                ProviderState.UNAVAILABLE,
                "mlx-whisper не установлен",
            )
        if self.model_revision is not None:
            try:
                self._model_reference()
            except Exception:
                return ProviderStatus(
                    self.id,
                    "Whisper MLX",
                    ProviderState.NEEDS_CONFIGURATION,
                    "Закреплённая ревизия Whisper ещё не загружена",
                )
        return ProviderStatus(
            self.id,
            "Whisper MLX",
            ProviderState.READY,
            f"Локальное распознавание речи: {self.model}",
        )

    def transcribe(self, source: Path) -> list[TimedText]:
        import mlx_whisper

        options: dict[str, object] = {
            "path_or_hf_repo": self._model_reference(),
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
            if not isinstance(raw, dict):
                continue
            try:
                text = str(raw.get("text") or "").strip()
                start = float(raw.get("start") or 0.0)
                end = float(raw.get("end") or start)
                average_log_probability = float(raw.get("avg_logprob") or 0.0)
            except (TypeError, ValueError):
                continue
            if (
                not text
                or not all(
                    math.isfinite(value)
                    for value in (start, end, average_log_probability)
                )
                or start < 0
                or end <= start
            ):
                continue
            confidence = max(
                0.0,
                min(1.0, math.exp(min(0.0, average_log_probability))),
            )
            words = []
            for word in raw.get("words") or []:
                if not isinstance(word, dict):
                    continue
                try:
                    word_text = str(word.get("word") or "").strip()
                    word_start = float(word.get("start") or start)
                    word_end = float(word.get("end") or word_start)
                    probability = float(word.get("probability") or confidence)
                except (TypeError, ValueError):
                    continue
                if (
                    not word_text
                    or not all(
                        math.isfinite(value)
                        for value in (word_start, word_end, probability)
                    )
                    or word_start < 0
                    or word_end <= word_start
                ):
                    continue
                words.append(
                    {
                        "word": word_text,
                        "start": word_start,
                        "end": word_end,
                        "probability": max(0.0, min(1.0, probability)),
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
