from pathlib import Path

from videoscope.artifacts import StageKind, StageState
from videoscope.config import AppSettings
from videoscope.media.ffmpeg import MediaProbe
from videoscope.media.ffmpeg import SampledFrame
from videoscope.processing.indexer import Indexer, merge_timed_text
from videoscope.providers.types import ObjectTag, TimedText
from videoscope.repository import Repository
from videoscope.runtime import create_indexing_specifications


def _specification_resolver(tmp_path: Path):  # type: ignore[no-untyped-def]
    settings = AppSettings(data_dir=tmp_path / "data")
    return lambda: create_indexing_specifications(settings)


class FakeFFmpeg:
    def __init__(self) -> None:
        self.dense_calls: list[tuple[float, int]] = []

    def probe(self, _source: Path) -> MediaProbe:
        return MediaProbe(duration=20.0, width=1280, height=720, fps=25.0)

    def extract_frame(self, _source: Path, destination: Path, _timestamp: float) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"jpeg")

    def extract_frames(
        self,
        _source: Path,
        destination: Path,
        start: float,
        end: float,
        *,
        step: float,
        max_width: int = 640,
    ) -> list[SampledFrame]:
        self.dense_calls.append((step, max_width))
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / "sample-0000.jpg"
        path.write_bytes(b"dense")
        return [SampledFrame(start, path)]


class FakeScenes:
    def detect(self, _source: Path, _duration: float) -> list[tuple[float, float]]:
        return [(0.0, 8.0), (8.0, 20.0)]


class FakeSpeech:
    def transcribe(self, _source: Path) -> list[TimedText]:
        return [TimedText(2.0, 6.0, "three point shot", 0.91)]


class FakeOCR:
    def read(self, _image: Path) -> list[tuple[str, float]]:
        return [("HOME 87 GUEST 86", 0.88)]


class FakeObjects:
    def detect(self, _image: Path) -> list[ObjectTag]:
        return [ObjectTag("basketball", 0.93), ObjectTag("player", 0.90)]


class RecordingIndex:
    def __init__(self) -> None:
        self.segment_ids: list[str] = []

    def replace_video(self, video_id: str, segments) -> None:  # type: ignore[no-untyped-def]
        assert video_id == "video-1"
        self.segment_ids = [segment.id for segment in segments]


def test_merge_timed_text_builds_context_without_crossing_long_gaps() -> None:
    merged = merge_timed_text(
        [
            TimedText(0, 2, "игрок", 0.8),
            TimedText(2.2, 4, "забивает трёхочковый", 0.9),
            TimedText(9, 11, "другая сцена", 0.7),
        ]
    )

    assert [item.text for item in merged] == ["игрок забивает трёхочковый", "другая сцена"]
    assert merged[0].metadata["merged_segments"] == 2
    assert merged[0].confidence > 0.8


def test_merge_timed_text_preserves_word_alignment() -> None:
    merged = merge_timed_text([
        TimedText(
            0,
            2,
            "Сегодня",
            0.9,
            {"words": [{"word": "Сегодня", "start": 0.1, "end": 0.8, "probability": 0.95}]},
        ),
        TimedText(
            2.1,
            4,
            "Мозгов",
            0.9,
            {"words": [{"word": "Мозгов", "start": 2.2, "end": 3.0, "probability": 0.91}]},
        ),
    ])

    assert [word["word"] for word in merged[0].metadata["words"]] == ["Сегодня", "Мозгов"]


def test_indexer_collects_each_available_modality(tmp_path) -> None:
    repository = Repository(tmp_path / "db.sqlite3")
    repository.initialize()
    media = tmp_path / "video.mp4"
    media.write_bytes(b"video")
    repository.create_video(
        video_id="video-1",
        original_name="match.mp4",
        stored_name="video.mp4",
        media_path=str(media),
        size_bytes=5,
    )
    vector_index = RecordingIndex()
    indexer = Indexer(
        repository=repository,
        media_root=tmp_path,
        thumbnails_dir=tmp_path / "thumbs",
        specification_resolver=_specification_resolver(tmp_path),
        ffmpeg=FakeFFmpeg(),  # type: ignore[arg-type]
        scenes=FakeScenes(),  # type: ignore[arg-type]
        speech=FakeSpeech(),  # type: ignore[arg-type]
        ocr=FakeOCR(),  # type: ignore[arg-type]
        objects=FakeObjects(),  # type: ignore[arg-type]
        vector_index=vector_index,  # type: ignore[arg-type]
    )

    indexer.process("video-1")

    video = repository.get_video("video-1")
    segments = repository.list_segments("video-1")
    assert video is not None
    assert video.status == "ready"
    assert video.progress == 1.0
    assert {segment.modality for segment in segments} == {"scene", "speech", "ocr", "objects"}
    assert set(vector_index.segment_ids) == {
        segment.id for segment in segments if segment.modality != "scene"
    }
    assert Path(video.thumbnail_path or "").is_file()


def test_ingest_and_reindex_use_the_same_dense_visual_source_path(
    tmp_path,
    monkeypatch,
) -> None:
    from videoscope.search.visual_index import SiglipVisualIndex

    repository = Repository(tmp_path / "db.sqlite3")
    repository.initialize()
    media = tmp_path / "video.mp4"
    media.write_bytes(b"video")
    repository.create_video(
        video_id="video-1",
        original_name="match.mp4",
        stored_name="video.mp4",
        media_path=str(media),
        size_bytes=5,
    )
    ffmpeg = FakeFFmpeg()
    visual = SiglipVisualIndex(
        tmp_path / "visual",
        model_name="test",
        sample_step=2.5,
        max_width=384,
    )
    monkeypatch.setattr(
        visual,
        "_image_vectors",
        lambda paths: __import__("numpy").ones((len(paths), 2), dtype="float32"),
    )
    indexer = Indexer(
        repository=repository,
        media_root=tmp_path,
        thumbnails_dir=tmp_path / "thumbs",
        specification_resolver=_specification_resolver(tmp_path),
        ffmpeg=ffmpeg,  # type: ignore[arg-type]
        scenes=FakeScenes(),  # type: ignore[arg-type]
        vector_index=RecordingIndex(),  # type: ignore[arg-type]
        visual_index=visual,
    )

    indexer.process("video-1")
    indexer.process("video-1")

    assert ffmpeg.dense_calls == [(2.5, 384), (2.5, 384)]
    assert visual.index_is_current("video-1") is True


def test_indexer_keeps_working_when_optional_provider_fails(tmp_path) -> None:
    class BrokenOCR:
        def read(self, _image: Path):  # type: ignore[no-untyped-def]
            raise RuntimeError("model unavailable")

    repository = Repository(tmp_path / "db.sqlite3")
    repository.initialize()
    media = tmp_path / "video.mp4"
    media.write_bytes(b"video")
    repository.create_video(
        video_id="video-1",
        original_name="match.mp4",
        stored_name="video.mp4",
        media_path=str(media),
        size_bytes=5,
    )
    indexer = Indexer(
        repository=repository,
        media_root=tmp_path,
        thumbnails_dir=tmp_path / "thumbs",
        specification_resolver=_specification_resolver(tmp_path),
        ffmpeg=FakeFFmpeg(),  # type: ignore[arg-type]
        scenes=FakeScenes(),  # type: ignore[arg-type]
        speech=None,
        ocr=BrokenOCR(),  # type: ignore[arg-type]
        objects=None,
        vector_index=RecordingIndex(),  # type: ignore[arg-type]
    )

    indexer.process("video-1")

    video = repository.get_video("video-1")
    assert video is not None
    assert video.status == "ready"
    assert "ocr" in (video.error or "")


def test_indexer_does_not_publish_partial_ocr_after_transient_error(tmp_path) -> None:
    class TransientOCR:
        def __init__(self) -> None:
            self.calls = 0

        def read(self, _image: Path):  # type: ignore[no-untyped-def]
            self.calls += 1
            if self.calls == 1:
                raise OSError(5, "temporary input/output error")
            return [("HOME 90 GUEST 87", 0.92)]

    repository = Repository(tmp_path / "db.sqlite3")
    repository.initialize()
    media = tmp_path / "video.mp4"
    media.write_bytes(b"video")
    repository.create_video(
        video_id="video-1",
        original_name="match.mp4",
        stored_name="video.mp4",
        media_path=str(media),
        size_bytes=5,
    )
    indexer = Indexer(
        repository=repository,
        media_root=tmp_path,
        thumbnails_dir=tmp_path / "thumbs",
        specification_resolver=_specification_resolver(tmp_path),
        ffmpeg=FakeFFmpeg(),  # type: ignore[arg-type]
        scenes=FakeScenes(),  # type: ignore[arg-type]
        speech=None,
        ocr=TransientOCR(),  # type: ignore[arg-type]
        objects=None,
        vector_index=RecordingIndex(),  # type: ignore[arg-type]
    )

    indexer.process("video-1")

    assert repository.list_active_segments("video-1", StageKind.OCR) == []
    run = repository.get_latest_stage_run("video-1", StageKind.OCR)
    assert run is not None
    assert run.state is StageState.FAILED
    assert run.error_code == "ocr_provider_failed"
