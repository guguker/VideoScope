from pathlib import Path

import pytest

from videoscope.media.uploads import UploadRejected, validate_upload


def test_validate_upload_accepts_known_video_container() -> None:
    result = validate_upload("Summer Game 01.MP4", 25_000, max_bytes=100_000)

    assert result.safe_name == "Summer_Game_01.mp4"
    assert result.extension == ".mp4"


@pytest.mark.parametrize("filename", ["notes.txt", "video.mp4.exe", ".mp4", "video"])
def test_validate_upload_rejects_unknown_or_ambiguous_names(filename: str) -> None:
    with pytest.raises(UploadRejected):
        validate_upload(filename, 100, max_bytes=100_000)


def test_validate_upload_rejects_oversized_file() -> None:
    with pytest.raises(UploadRejected, match="too large"):
        validate_upload("game.mov", 100_001, max_bytes=100_000)


def test_safe_name_never_contains_a_path() -> None:
    result = validate_upload("../../game final.mkv", 100, max_bytes=100_000)

    assert Path(result.safe_name).name == result.safe_name
    assert result.safe_name == "game_final.mkv"

