from pathlib import Path

import pytest

from videoscope.media.ffmpeg import FFmpeg, MediaProbe


def test_probe_parses_fractional_fps_and_rotation(monkeypatch, tmp_path) -> None:
    runner = FFmpeg()
    monkeypatch.setattr(
        runner,
        "_run_json",
        lambda _args: {
            "format": {"duration": "65.25"},
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1920,
                    "height": 1080,
                    "avg_frame_rate": "30000/1001",
                    "tags": {"rotate": "90"},
                }
            ],
        },
    )

    probe = runner.probe(tmp_path / "clip.mp4")

    assert probe == MediaProbe(
        duration=65.25,
        width=1080,
        height=1920,
        fps=pytest.approx(30_000 / 1_001),
    )


def test_clip_command_uses_argument_list_and_clamped_interval(tmp_path) -> None:
    runner = FFmpeg()
    command = runner.build_clip_command(
        source=Path("/media/source.mp4"),
        destination=tmp_path / "clip.mp4",
        start=5.0,
        end=8.25,
    )

    assert command[0] == "ffmpeg"
    assert command[command.index("-ss") + 1] == "5.000"
    assert command[command.index("-t") + 1] == "3.250"
    assert "/media/source.mp4" in command


def test_clip_command_rejects_empty_interval(tmp_path) -> None:
    runner = FFmpeg()

    with pytest.raises(ValueError):
        runner.build_clip_command(Path("in.mp4"), tmp_path / "out.mp4", 5, 5)


def test_frame_sample_command_extracts_an_interval_in_one_process(tmp_path) -> None:
    runner = FFmpeg()
    command = runner.build_frame_sample_command(
        source=Path("/media/source.mp4"),
        destination_pattern=tmp_path / "sample-%04d.jpg",
        start=10,
        end=18,
        step=2,
    )

    assert command[command.index("-ss") + 1] == "10.000"
    assert command[command.index("-t") + 1] == "8.000"
    assert "fps=0.500000" in command[command.index("-vf") + 1]


def test_frame_sample_command_rejects_invalid_step(tmp_path) -> None:
    runner = FFmpeg()

    with pytest.raises(ValueError):
        runner.build_frame_sample_command(
            Path("in.mp4"),
            tmp_path / "sample-%04d.jpg",
            0,
            10,
            0,
        )
