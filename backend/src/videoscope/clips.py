from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
import os
from pathlib import Path

from videoscope.media.ffmpeg import FFmpeg
from videoscope.repository import Repository


MAX_EXPORT_STEM_BYTES = 180


@dataclass(frozen=True, slots=True)
class ClipSelection:
    video_id: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class ExportedClip:
    name: str
    path: Path
    duration: float
    created_at: str


def _safe_export_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKC", name).strip()
    normalized = re.sub(r"\s+", "_", normalized)
    normalized = re.sub(r"[^\w.-]", "_", normalized, flags=re.UNICODE)
    normalized = re.sub(r"_+", "_", normalized).strip("._")
    normalized = normalized or "videoscope-export"
    while len(normalized.encode("utf-8")) > MAX_EXPORT_STEM_BYTES:
        normalized = normalized[:-1]
    return normalized + ".mp4"


def _reserve_export_path(clips_dir: Path, output_name: str) -> Path:
    clips_dir.mkdir(parents=True, exist_ok=True)
    requested = clips_dir / output_name
    stem = requested.stem
    suffix = 1
    while True:
        candidate = requested if suffix == 1 else clips_dir / f"{stem}-{suffix}.mp4"
        try:
            descriptor = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            suffix += 1
            continue
        os.close(descriptor)
        return candidate


class ClipService:
    def __init__(
        self,
        repository: Repository,
        ffmpeg: FFmpeg,
        clips_dir: Path,
        temp_dir: Path,
    ) -> None:
        self.repository = repository
        self.ffmpeg = ffmpeg
        self.clips_dir = clips_dir
        self.temp_dir = temp_dir

    def export(self, name: str, selections: list[ClipSelection]) -> ExportedClip:
        if not 1 <= len(selections) <= 30:
            raise ValueError("an export must contain between 1 and 30 clips")
        resolved: list[tuple[Path, float, float]] = []
        total_duration = 0.0
        for selection in selections:
            video = self.repository.get_video(selection.video_id)
            if video is None or video.duration is None:
                raise ValueError("video is not ready for export")
            start = float(selection.start)
            end = float(selection.end)
            if start < 0 or end <= start or end > video.duration:
                raise ValueError("clip interval is outside video bounds")
            if end - start < 0.2 or end - start > 300:
                raise ValueError("clip duration must be between 0.2 and 300 seconds")
            resolved.append((Path(video.media_path), start, end))
            total_duration += end - start

        destination = _reserve_export_path(self.clips_dir, _safe_export_name(name))
        output_name = destination.name
        try:
            self.ffmpeg.export_montage(resolved, destination, self.temp_dir)
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        return ExportedClip(
            name=output_name,
            path=destination,
            duration=total_duration,
            created_at=datetime.now(UTC).isoformat(),
        )
