import pytest

from videoscope.repository import Repository


def test_repository_round_trips_video_and_segments(tmp_path) -> None:
    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()
    video = repository.create_video(
        video_id="video-1",
        original_name="match.mp4",
        stored_name="video-1.mp4",
        media_path="/tmp/video-1.mp4",
        size_bytes=1024,
    )
    repository.add_segment(
        segment_id="segment-1",
        video_id=video.id,
        start=4.0,
        end=9.5,
        modality="speech",
        text="important moment",
        confidence=0.8,
        metadata={"language": "en"},
    )

    loaded = repository.get_video(video.id)
    segments = repository.list_segments(video.id)

    assert loaded is not None
    assert loaded.original_name == "match.mp4"
    assert segments[0].metadata == {"language": "en"}


def test_repository_updates_processing_state_atomically(tmp_path) -> None:
    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()
    repository.create_video(
        video_id="video-1",
        original_name="match.mp4",
        stored_name="video-1.mp4",
        media_path="/tmp/video-1.mp4",
        size_bytes=1024,
    )

    repository.update_video("video-1", status="processing", progress=0.45, stage="ocr")

    loaded = repository.get_video("video-1")
    assert loaded is not None
    assert (loaded.status, loaded.progress, loaded.stage) == ("processing", 0.45, "ocr")


@pytest.mark.parametrize(
    "changes",
    [
        {"start": -1.0},
        {"start": float("nan")},
        {"end": float("inf")},
        {"confidence": float("nan")},
        {"confidence": 1.1},
        {"modality": "qwen_video"},
        {"metadata": []},
        {"metadata": {"score": float("nan")}},
    ],
)
def test_repository_rejects_invalid_segment_contract(tmp_path, changes) -> None:  # type: ignore[no-untyped-def]
    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()
    repository.create_video(
        video_id="video-1",
        original_name="match.mp4",
        stored_name="video-1.mp4",
        media_path="/tmp/video-1.mp4",
        size_bytes=1024,
    )
    payload = {
        "segment_id": "segment-1",
        "video_id": "video-1",
        "start": 1.0,
        "end": 2.0,
        "modality": "speech",
        "text": "moment",
        "confidence": 0.8,
        "metadata": {},
    }
    payload.update(changes)

    with pytest.raises(ValueError):
        repository.add_segment(**payload)


def test_repository_ignores_invalid_legacy_segment_rows(tmp_path) -> None:
    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()
    repository.create_video(
        video_id="video-1",
        original_name="match.mp4",
        stored_name="video-1.mp4",
        media_path="/tmp/video-1.mp4",
        size_bytes=1024,
    )
    with repository._connect() as connection:
        connection.execute(
            """
            INSERT INTO segments (
                id, video_id, start, end, modality, text,
                confidence, metadata_json, thumbnail_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-bad",
                "video-1",
                -1.0,
                2.0,
                "speech",
                "bad",
                0.8,
                "{}",
                None,
            ),
        )

    assert repository.list_segments("video-1") == []
