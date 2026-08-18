from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from videoscope.artifacts import StageKind, StageState
from videoscope.config import AppSettings
from videoscope.media.ffmpeg import MediaProbe
from videoscope.processing.indexer import Indexer
from videoscope.providers.types import TimedText
from videoscope.repository import Repository
from videoscope.search.vector_index import EmptyVectorIndex
import videoscope.runtime as runtime_module


class SingleSceneDetector:
    def detect(self, _source: Path, duration: float) -> list[tuple[float, float]]:
        return [(0.0, duration)]


class FixtureFFmpeg:
    def probe(self, _source: Path) -> MediaProbe:
        return MediaProbe(duration=12.0, width=1280, height=720, fps=25.0)

    def extract_frame(
        self,
        _source: Path,
        destination: Path,
        _timestamp: float,
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"jpeg")

    def extract_frames(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        return []


class RunBoundSpeech:
    def __init__(self, glossary_path: Path) -> None:
        self.glossary_path = glossary_path
        self.received_snapshots: list[object] = []
        self.mutate_glossary = False
        self.fail: Exception | None = None

    def transcribe(
        self,
        _source: Path,
        *,
        prompt_snapshot,
    ) -> list[TimedText]:
        self.received_snapshots.append(prompt_snapshot)
        if self.mutate_glossary:
            self.glossary_path.write_text(
                '{"Мозгов":["Mozgov","Мозгова"]}',
                encoding="utf-8",
            )
        if self.fail is not None:
            raise self.fail
        return [TimedText(1.0, 4.0, "точная речь", 0.92)]


def _repository(tmp_path: Path) -> tuple[Repository, Path, Path]:
    media_root = tmp_path / "data" / "media"
    media_root.mkdir(parents=True, exist_ok=True)
    source = media_root / "source.mp4"
    payload = b"immutable-video-fixture"
    source.write_bytes(payload)
    repository = Repository(tmp_path / "data" / "videoscope.sqlite3")
    repository.initialize()
    repository.create_video_with_asset(
        video_id="video-1",
        original_name="source.mp4",
        stored_name=source.name,
        media_path=str(source),
        size_bytes=len(payload),
        source_sha256=sha256(payload).hexdigest(),
    )
    return repository, media_root, source


def _indexer(
    tmp_path: Path,
    settings: AppSettings,
    repository: Repository,
    media_root: Path,
    speech: RunBoundSpeech,
) -> Indexer:
    return Indexer(
        repository=repository,
        media_root=media_root,
        thumbnails_dir=settings.thumbnails_dir,
        specification_resolver=lambda: runtime_module.create_indexing_run_plan(
            settings
        ),
        ffmpeg=FixtureFFmpeg(),  # type: ignore[arg-type]
        scenes=SingleSceneDetector(),  # type: ignore[arg-type]
        vector_index=EmptyVectorIndex(),
        speech=speech,
        ocr=None,
        objects=None,
    )


def test_indexer_forwards_the_exact_plan_snapshot_to_whisper(tmp_path: Path) -> None:
    settings = AppSettings(
        _env_file=None,
        data_dir=tmp_path / "data",
        whisper_worker_endpoint="http://127.0.0.1:8784",
        whisper_worker_api_key="w" * 32,
    )
    settings.ensure_directories()
    settings.glossary_path.write_text('{"Мозгов":["Mozgov"]}', encoding="utf-8")
    repository, media_root, _source = _repository(tmp_path)
    speech = RunBoundSpeech(settings.glossary_path)
    expected = runtime_module.create_indexing_run_plan(settings)
    indexer = _indexer(tmp_path, settings, repository, media_root, speech)

    indexer.process("video-1")

    assert speech.received_snapshots == [expected.whisper_prompt_snapshot]
    generation = repository.get_active_segment_generation(
        "video-1",
        StageKind.SPEECH,
    )
    assert generation is not None
    assert generation.specification_hash == (
        expected.specifications.speech.specification_hash
    )


def test_glossary_drift_fails_the_run_and_preserves_active_speech_generation(
    tmp_path: Path,
) -> None:
    settings = AppSettings(
        _env_file=None,
        data_dir=tmp_path / "data",
        whisper_worker_endpoint="http://127.0.0.1:8784",
        whisper_worker_api_key="w" * 32,
    )
    settings.ensure_directories()
    settings.glossary_path.write_text('{"Мозгов":["Mozgov"]}', encoding="utf-8")
    repository, media_root, _source = _repository(tmp_path)
    speech = RunBoundSpeech(settings.glossary_path)
    indexer = _indexer(tmp_path, settings, repository, media_root, speech)
    indexer.process("video-1")
    active_before = repository.get_active_segment_generation(
        "video-1",
        StageKind.SPEECH,
    )
    speech.mutate_glossary = True

    indexer.process("video-1")

    assert repository.get_active_segment_generation(
        "video-1",
        StageKind.SPEECH,
    ) == active_before
    latest = repository.get_latest_stage_run("video-1", StageKind.SPEECH)
    assert latest is not None
    assert latest.state is StageState.FAILED
    assert latest.error_code == "speech_specification_changed"
    received = speech.received_snapshots[-1]
    current = runtime_module.create_indexing_run_plan(settings)
    assert received.effective_prompt_sha256 != (
        current.whisper_prompt_snapshot.effective_prompt_sha256
    )


def test_worker_failure_preserves_the_previous_speech_generation(
    tmp_path: Path,
) -> None:
    settings = AppSettings(
        _env_file=None,
        data_dir=tmp_path / "data",
        whisper_worker_endpoint="http://127.0.0.1:8784",
        whisper_worker_api_key="w" * 32,
    )
    settings.ensure_directories()
    repository, media_root, _source = _repository(tmp_path)
    speech = RunBoundSpeech(settings.glossary_path)
    indexer = _indexer(tmp_path, settings, repository, media_root, speech)
    indexer.process("video-1")
    active_before = repository.get_active_segment_generation(
        "video-1",
        StageKind.SPEECH,
    )
    speech.fail = RuntimeError("untrusted worker response detail")

    indexer.process("video-1")

    assert repository.get_active_segment_generation(
        "video-1",
        StageKind.SPEECH,
    ) == active_before
    latest = repository.get_latest_stage_run("video-1", StageKind.SPEECH)
    assert latest is not None
    assert latest.state is StageState.FAILED
    assert latest.error_code == "speech_provider_failed"
    assert "untrusted" not in latest.error_code
