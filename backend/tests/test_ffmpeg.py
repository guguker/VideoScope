from pathlib import Path
import subprocess

import pytest

import videoscope.media.ffmpeg as ffmpeg_module
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


def test_attested_adapter_does_not_inherit_ambient_process_environment(
    tmp_path,
    monkeypatch,
) -> None:
    tool = tmp_path / "ffmpeg-fixture"
    tool.write_text(
        "#!/bin/sh\n"
        "if [ -n \"${VIDEOSCOPE_AMBIENT_MARKER-}\" ]; then exit 73; fi\n"
        "printf '%s' '{\"format\":{\"duration\":\"1\"},"
        "\"streams\":[{\"codec_type\":\"video\",\"width\":1,"
        "\"height\":1,\"avg_frame_rate\":\"1/1\"}]}'\n",
        encoding="utf-8",
    )
    tool.chmod(0o555)
    monkeypatch.setenv("VIDEOSCOPE_AMBIENT_MARKER", "hostile")

    with pytest.raises(subprocess.CalledProcessError):
        FFmpeg(str(tool), str(tool)).probe(tmp_path / "source.mp4")

    attested = FFmpeg.from_attested_paths(tool.resolve(), tool.resolve())
    assert attested.probe(tmp_path / "source.mp4").duration == 1.0


def test_every_attested_subprocess_path_uses_the_frozen_empty_environment(
    tmp_path,
    monkeypatch,
) -> None:
    binary = (tmp_path / "reviewed-ffmpeg").resolve()
    binary.touch()
    runner = FFmpeg.from_attested_paths(binary, binary)
    calls: list[dict[str, object]] = []

    def run(arguments, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        stdout: str | bytes = b""
        if kwargs.get("text"):
            stdout = (
                '{"format":{"duration":"1"},"streams":'
                '[{"codec_type":"video","width":1,"height":1,'
                '"avg_frame_rate":"1/1"}]}'
            )
        return subprocess.CompletedProcess(arguments, 0, stdout=stdout, stderr=b"")

    monkeypatch.setattr(ffmpeg_module.subprocess, "run", run)
    source = tmp_path / "source.mp4"
    runner.probe(source)
    runner.export_clip(source, tmp_path / "clip.mp4", 0, 1)
    runner.extract_frame(source, tmp_path / "frame.jpg", 0)
    runner.extract_frames(source, tmp_path / "frames", 0, 1, step=0.5)
    runner.export_montage(
        [(source, 0, 1), (source, 1, 2)],
        tmp_path / "montage.mp4",
        tmp_path / "temp",
    )

    assert len(calls) == 7
    assert all(call["env"] == {} for call in calls)
