from __future__ import annotations

from dataclasses import dataclass
import errno
import hashlib
import math
import json
import os
from pathlib import Path
import stat
from typing import Protocol

from videoscope.model_manifest import model_identity
from videoscope.providers.base import ProviderState, ProviderStatus
from videoscope.providers.types import TimedText


MAX_GLOSSARY_BYTES = 1024 * 1024
MAX_GLOSSARY_TERMS = 120
MAX_GLOSSARY_TERM_CHARS = 160
MAX_WHISPER_EFFECTIVE_PROMPT_CHARS = 16_000


@dataclass(frozen=True, slots=True)
class WhisperPromptSnapshot:
    """One immutable prompt/glossary view shared by a speech stage run."""

    effective_prompt: str | None
    effective_prompt_sha256: str
    glossary_state: str


@dataclass(frozen=True, slots=True)
class WhisperInferenceStatus:
    ready: bool
    detail: str


class WhisperInferenceClient(Protocol):
    @property
    def identity(self) -> dict[str, object]: ...

    def status(self) -> WhisperInferenceStatus: ...

    def transcribe(
        self,
        source: Path,
        *,
        language: str,
        prompt_snapshot: WhisperPromptSnapshot,
    ) -> list[TimedText]: ...


@dataclass(frozen=True, slots=True)
class _GlossaryFingerprint:
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int


def _fingerprint(metadata: os.stat_result) -> _GlossaryFingerprint:
    return _GlossaryFingerprint(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        size=metadata.st_size,
        modified_ns=metadata.st_mtime_ns,
        changed_ns=metadata.st_ctime_ns,
    )


def _open_parent_without_following(path: Path) -> tuple[Path, int]:
    """Open a lexical parent by descriptor without traversing any symlink."""
    absolute = Path(os.path.abspath(path))
    if not absolute.name:
        raise OSError(errno.EINVAL, "glossary path has no file name")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(absolute.anchor, directory_flags)
    try:
        for part in absolute.parts[1:-1]:
            child = os.open(part, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
    except BaseException:
        os.close(descriptor)
        raise
    return absolute, descriptor


def _unsafe_path_error(error: OSError) -> bool:
    return error.errno in {errno.ELOOP, errno.ENOTDIR, errno.EINVAL}


def _read_regular_file_without_following(path: Path) -> tuple[bytes | None, str]:
    """Read one stable file through an all-components O_NOFOLLOW descriptor chain."""
    try:
        absolute, parent_descriptor = _open_parent_without_following(path)
    except FileNotFoundError:
        return None, "missing"
    except PermissionError:
        return None, "unreadable"
    except OSError as error:
        return None, "unsafe" if _unsafe_path_error(error) else "invalid"

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        try:
            lexical = os.stat(
                absolute.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return None, "missing"
        except PermissionError:
            return None, "unreadable"
        except OSError:
            return None, "invalid"
        if stat.S_ISLNK(lexical.st_mode) or not stat.S_ISREG(lexical.st_mode):
            return None, "unsafe"
        if lexical.st_size > MAX_GLOSSARY_BYTES:
            return None, "oversized"

        try:
            descriptor = os.open(absolute.name, flags, dir_fd=parent_descriptor)
        except FileNotFoundError:
            return None, "changed"
        except PermissionError:
            return None, "unreadable"
        except OSError:
            return None, "unsafe"
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_size > MAX_GLOSSARY_BYTES
                or _fingerprint(before) != _fingerprint(lexical)
            ):
                return None, "changed"
            chunks: list[bytes] = []
            remaining = MAX_GLOSSARY_BYTES + 1
            while remaining > 0:
                chunk = os.read(descriptor, min(64 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            after = os.fstat(descriptor)
        except OSError:
            return None, "unreadable"
        finally:
            os.close(descriptor)
        try:
            current = os.stat(
                absolute.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except OSError:
            return None, "changed"
    finally:
        os.close(parent_descriptor)
    if len(raw) > MAX_GLOSSARY_BYTES:
        return None, "oversized"

    # Re-open the configured lexical chain as a final namespace check. The
    # content itself was read only from the already-safe descriptor above.
    try:
        _, verification_parent = _open_parent_without_following(absolute)
    except OSError:
        return None, "changed"
    try:
        verification = os.stat(
            absolute.name,
            dir_fd=verification_parent,
            follow_symlinks=False,
        )
    except OSError:
        return None, "changed"
    finally:
        os.close(verification_parent)
    expected = _fingerprint(before)
    if (
        len(raw) != before.st_size
        or _fingerprint(after) != expected
        or _fingerprint(current) != expected
        or _fingerprint(verification) != expected
        or stat.S_ISLNK(current.st_mode)
        or stat.S_ISLNK(verification.st_mode)
    ):
        return None, "changed"
    return raw, "ready"


def snapshot_whisper_prompt(
    initial_prompt: str | None,
    glossary_path: Path | None,
) -> WhisperPromptSnapshot:
    """Capture the exact prompt used by one indexing run without symlink races."""
    normalized_initial = initial_prompt.strip() if initial_prompt else ""
    if len(normalized_initial) > MAX_WHISPER_EFFECTIVE_PROMPT_CHARS:
        raise ValueError("Whisper initial prompt exceeds the bounded worker contract")
    parts = [normalized_initial] if normalized_initial else []
    state = "not_configured" if glossary_path is None else "missing"
    payload: object = {}
    if glossary_path is not None:
        raw, state = _read_regular_file_without_following(Path(glossary_path))
        if raw is not None:
            try:
                payload = json.loads(raw.decode("utf-8"))
                state = "ready" if isinstance(payload, dict) else "invalid"
            except (UnicodeError, json.JSONDecodeError, RecursionError):
                state = "invalid"
    terms: list[str] = []
    if state == "ready" and isinstance(payload, dict):
        for canonical, aliases in payload.items():
            terms.append(str(canonical))
            if isinstance(aliases, list):
                terms.extend(str(alias) for alias in aliases)
    terms = list(
        dict.fromkeys(
            term.strip()[:MAX_GLOSSARY_TERM_CHARS]
            for term in terms
            if term.strip()
        )
    )[:MAX_GLOSSARY_TERMS]
    if terms:
        prefix = "Словарь имён и терминов: "
        available = MAX_WHISPER_EFFECTIVE_PROMPT_CHARS - len(" ".join(parts))
        accepted: list[str] = []
        for term in terms:
            candidate = prefix + ", ".join([*accepted, term]) + "."
            separator = 1 if parts else 0
            if len(candidate) + separator > available:
                break
            accepted.append(term)
        if accepted:
            parts.append(prefix + ", ".join(accepted) + ".")
    effective_prompt = " ".join(parts) or None
    digest = hashlib.sha256((effective_prompt or "").encode("utf-8")).hexdigest()
    return WhisperPromptSnapshot(
        effective_prompt=effective_prompt,
        effective_prompt_sha256=digest,
        glossary_state=state,
    )


def resolve_whisper_prompt(
    initial_prompt: str | None,
    glossary_path: Path | None,
) -> tuple[str | None, str]:
    """Resolve the exact bounded prompt and a stable glossary state identity."""
    snapshot = snapshot_whisper_prompt(initial_prompt, glossary_path)
    return snapshot.effective_prompt, snapshot.glossary_state


class WhisperTranscriber:
    id = "whisper"

    def __init__(
        self,
        model: str,
        language: str = "auto",
        initial_prompt: str | None = None,
        glossary_path: Path | None = None,
        model_revision: str | None = None,
        inference_client: WhisperInferenceClient | None = None,
    ) -> None:
        self.model = model
        self.language = language
        self.initial_prompt = initial_prompt
        self.glossary_path = Path(glossary_path) if glossary_path else None
        self.model_revision = model_revision
        self.inference_client = inference_client

    @property
    def identity(self) -> dict[str, object]:
        boundary = (
            self.inference_client.identity
            if self.inference_client is not None
            else {"mode": "in-process"}
        )
        return {
            "provider": self.id,
            "model": model_identity(self.model, self.model_revision),
            "language": self.language,
            "boundary": boundary,
        }

    def prompt_snapshot(self) -> WhisperPromptSnapshot:
        return snapshot_whisper_prompt(self.initial_prompt, self.glossary_path)

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
        return self.prompt_snapshot().effective_prompt

    def status(self) -> ProviderStatus:
        if self.inference_client is not None:
            worker = self.inference_client.status()
            return ProviderStatus(
                self.id,
                "Whisper MLX",
                ProviderState.READY if worker.ready else ProviderState.UNAVAILABLE,
                worker.detail,
            )
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
                    (
                        "Закреплённая ревизия Whisper "
                        "ещё не загружена"
                    ),
                )
        return ProviderStatus(
            self.id,
            "Whisper MLX",
            ProviderState.READY,
            f"Локальное распознавание речи: {self.model}",
        )

    def transcribe(
        self,
        source: Path,
        *,
        prompt_snapshot: WhisperPromptSnapshot | None = None,
    ) -> list[TimedText]:
        if self.inference_client is not None:
            return self.inference_client.transcribe(
                source,
                language=self.language,
                prompt_snapshot=prompt_snapshot or self.prompt_snapshot(),
            )
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
