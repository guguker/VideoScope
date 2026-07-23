from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from videoscope.media.ffmpeg import FFmpeg
from videoscope.repository import Repository


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
    return (normalized[:100] or "videoscope-export") + ".mp4"


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

        output_name = _safe_export_name(name)
        destination = self.clips_dir / output_name
        if destination.exists():
            stem = destination.stem
            suffix = 2
            while destination.exists():
                destination = self.clips_dir / f"{stem}-{suffix}.mp4"
                suffix += 1
            output_name = destination.name
        self.ffmpeg.export_montage(resolved, destination, self.temp_dir)
        return ExportedClip(
            name=output_name,
            path=destination,
            duration=total_duration,
            created_at=datetime.now(UTC).isoformat(),
        )

