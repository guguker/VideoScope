from pathlib import Path

import pytest

from videoscope.clips import ClipSelection, ClipService
from videoscope.repository import Repository


class FakeFFmpeg:
    def __init__(self) -> None:
        self.calls: list[tuple[list[tuple[Path, float, float]], Path]] = []

    def export_montage(self, selections, destination, temp_dir) -> None:  # type: ignore[no-untyped-def]
        self.calls.append((selections, destination))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"mp4")


def test_clip_service_validates_and_exports_montage(tmp_path) -> None:
    repository = Repository(tmp_path / "db.sqlite3")
    repository.initialize()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    repository.create_video(
        video_id="video-1",
        original_name="match.mp4",
        stored_name="source.mp4",
        media_path=str(source),
        size_bytes=5,
    )
    repository.update_video("video-1", duration=120.0, status="ready")
    ffmpeg = FakeFFmpeg()
    service = ClipService(repository, ffmpeg, tmp_path / "clips", tmp_path / "tmp")  # type: ignore[arg-type]

    exported = service.export(
        "Best plays / final",
        [ClipSelection(video_id="video-1", start=5, end=9)],
    )

    assert exported.path.is_file()
    assert exported.name == "Best_plays_final.mp4"
    assert ffmpeg.calls[0][0] == [(source, 5.0, 9.0)]


def test_clip_service_rejects_out_of_bounds_selection(tmp_path) -> None:
    repository = Repository(tmp_path / "db.sqlite3")
    repository.initialize()
    repository.create_video(
        video_id="video-1",
        original_name="match.mp4",
        stored_name="source.mp4",
        media_path=str(tmp_path / "source.mp4"),
        size_bytes=5,
    )
    repository.update_video("video-1", duration=10.0, status="ready")
    service = ClipService(repository, FakeFFmpeg(), tmp_path / "clips", tmp_path / "tmp")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="bounds"):
        service.export("clip", [ClipSelection(video_id="video-1", start=9, end=12)])

