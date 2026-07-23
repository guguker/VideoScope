from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


ALLOWED_VIDEO_EXTENSIONS = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"})


class UploadRejected(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ValidatedUpload:
    safe_name: str
    extension: str
    size_bytes: int


def _basename(filename: str) -> str:
    return filename.replace("\\", "/").rsplit("/", 1)[-1]


def _safe_stem(stem: str) -> str:
    normalized = unicodedata.normalize("NFKC", stem).strip()
    normalized = re.sub(r"\s+", "_", normalized)
    normalized = re.sub(r"[^\w.-]", "_", normalized, flags=re.UNICODE)
    normalized = re.sub(r"_+", "_", normalized).strip("._")
    return normalized[:120]


def validate_upload(filename: str, size_bytes: int, *, max_bytes: int) -> ValidatedUpload:
    if size_bytes <= 0:
        raise UploadRejected("video is empty")
    if size_bytes > max_bytes:
        raise UploadRejected("video is too large")

    basename = _basename(filename)
    extension = Path(basename).suffix.lower()
    stem = _safe_stem(Path(basename).stem)
    if extension not in ALLOWED_VIDEO_EXTENSIONS or not stem:
        raise UploadRejected("unsupported video filename or container")

    return ValidatedUpload(
        safe_name=f"{stem}{extension}",
        extension=extension,
        size_bytes=size_bytes,
    )

