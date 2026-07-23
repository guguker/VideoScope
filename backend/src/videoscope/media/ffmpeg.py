from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class MediaProbe:
    duration: float
    width: int
    height: int
    fps: float


@dataclass(frozen=True, slots=True)
class SampledFrame:
    timestamp: float
    path: Path


def _fraction(value: str | None) -> float:
    if not value or value == "0/0":
        return 0.0
    numerator, separator, denominator = value.partition("/")
    if not separator:
        return float(value)
    divisor = float(denominator)
    return float(numerator) / divisor if divisor else 0.0


class FFmpeg:
    def __init__(self, ffmpeg_binary: str = "ffmpeg", ffprobe_binary: str = "ffprobe") -> None:
        self.ffmpeg_binary = ffmpeg_binary
        self.ffprobe_binary = ffprobe_binary

    def _run_json(self, arguments: list[str]) -> dict[str, Any]:
        completed = subprocess.run(
            arguments,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return json.loads(completed.stdout)

    def probe(self, source: Path) -> MediaProbe:
        payload = self._run_json(
            [
                self.ffprobe_binary,
                "-v",
                "error",
                "-show_entries",
                "format=duration:stream=codec_type,width,height,avg_frame_rate:stream_tags=rotate",
                "-of",
                "json",
                str(source),
            ]
        )
        stream = next(
            (candidate for candidate in payload.get("streams", []) if candidate.get("codec_type") == "video"),
            None,
        )
        if stream is None:
            raise ValueError("uploaded file has no video stream")

        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        rotation = int((stream.get("tags") or {}).get("rotate") or 0) % 360
        if rotation in {90, 270}:
            width, height = height, width

        return MediaProbe(
            duration=float((payload.get("format") or {}).get("duration") or 0.0),
            width=width,
            height=height,
            fps=_fraction(stream.get("avg_frame_rate")),
        )

    def build_clip_command(
        self,
        source: Path,
        destination: Path,
        start: float,
        end: float,
    ) -> list[str]:
        if end <= start or start < 0:
            raise ValueError("clip interval must be positive and ordered")
        return [
            self.ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(source),
            "-t",
            f"{end - start:.3f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(destination),
        ]

    def export_clip(self, source: Path, destination: Path, start: float, end: float) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            self.build_clip_command(source, destination, start, end),
            check=True,
            capture_output=True,
            timeout=max(120, int(end - start) * 4),
        )

    def extract_frame(
        self,
        source: Path,
        destination: Path,
        timestamp: float,
        *,
        max_width: int = 960,
    ) -> None:
        if timestamp < 0:
            raise ValueError("frame timestamp must be non-negative")
        destination.parent.mkdir(parents=True, exist_ok=True)
        command = [
            self.ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-vf",
            f"scale='min({max_width},iw)':-2",
            "-q:v",
            "2",
            str(destination),
        ]
        subprocess.run(command, check=True, capture_output=True, timeout=120)

    def build_frame_sample_command(
        self,
        source: Path,
        destination_pattern: Path,
        start: float,
        end: float,
        step: float,
        *,
        max_width: int = 640,
    ) -> list[str]:
        if start < 0 or end <= start:
            raise ValueError("frame sample interval must be positive and ordered")
        if step <= 0:
            raise ValueError("frame sample step must be positive")
        return [
            self.ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(source),
            "-t",
            f"{end - start:.3f}",
            "-vf",
            f"fps={1.0 / step:.6f},scale='min({max_width},iw)':-2",
            "-q:v",
            "3",
            "-start_number",
            "0",
            str(destination_pattern),
        ]

    def extract_frames(
        self,
        source: Path,
        destination: Path,
        start: float,
        end: float,
        *,
        step: float,
        max_width: int = 640,
    ) -> list[SampledFrame]:
        destination.mkdir(parents=True, exist_ok=True)
        for existing in destination.glob("sample-*.jpg"):
            existing.unlink(missing_ok=True)
        pattern = destination / "sample-%04d.jpg"
        subprocess.run(
            self.build_frame_sample_command(
                source,
                pattern,
                start,
                end,
                step,
                max_width=max_width,
            ),
            check=True,
            capture_output=True,
            timeout=max(120, int(end - start) * 3),
        )
        return [
            SampledFrame(min(end, start + index * step), path)
            for index, path in enumerate(sorted(destination.glob("sample-*.jpg")))
        ]

    def export_montage(
        self,
        selections: list[tuple[Path, float, float]],
        destination: Path,
        temp_dir: Path,
    ) -> None:
        if not selections:
            raise ValueError("at least one clip is required")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="videoscope-export-", dir=temp_dir) as directory:
            workspace = Path(directory)
            clip_paths: list[Path] = []
            for index, (source, start, end) in enumerate(selections):
                clip_path = workspace / f"part-{index:04d}.mp4"
                self.export_clip(source, clip_path, start, end)
                clip_paths.append(clip_path)

            if len(clip_paths) == 1:
                clip_paths[0].replace(destination)
                return

            concat_path = workspace / "concat.txt"
            concat_path.write_text(
                "".join(f"file '{path.as_posix()}'\n" for path in clip_paths),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    self.ffmpeg_binary,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(concat_path),
                    "-c",
                    "copy",
                    "-movflags",
                    "+faststart",
                    str(destination),
                ],
                check=True,
                capture_output=True,
                timeout=max(120, int(sum(end - start for _, start, end in selections)) * 4),
            )
