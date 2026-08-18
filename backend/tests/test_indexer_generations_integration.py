from __future__ import annotations

from pathlib import Path

import pytest

from videoscope.artifacts import (
    AssetIdentityError,
    IndexingSpecifications,
    StageKind,
    StageSpecification,
    StageState,
)
from videoscope.config import AppSettings
from videoscope.media.ffmpeg import MediaProbe
from videoscope.processing.indexer import Indexer
from videoscope.providers.types import ObjectTag, TimedText
from videoscope.repository import Repository
from videoscope.runtime import create_indexing_specifications
from videoscope.search.service import SearchService
from videoscope.search.vector_index import EmptyVectorIndex, MemoryVectorIndex


def _specification(kind: StageKind) -> StageSpecification:
    return StageSpecification(
        kind=kind,
        schema_version=1,
        implementation_revision=f"tests.{kind.value}.v1",
        parameters={"contract": f"{kind.value}-segments-v1"},
        model_identity=(
            f"tests/{kind.value}@revision-1"
            if kind is not StageKind.SCENES
            else None
        ),
        dependencies={"provider": f"test-{kind.value}"},
    )


def _specifications() -> IndexingSpecifications:
    return IndexingSpecifications(
        scenes=_specification(StageKind.SCENES),
        speech=_specification(StageKind.SPEECH),
        ocr=_specification(StageKind.OCR),
        objects=_specification(StageKind.OBJECTS),
        text_vectors=_specification(StageKind.TEXT_VECTORS),
    )


class MutableFFmpeg:
    def __init__(self) -> None:
        self.fail_extract = False

    def probe(self, _source: Path) -> MediaProbe:
        return MediaProbe(duration=20.0, width=1280, height=720, fps=25.0)

    def extract_frame(self, _source: Path, destination: Path, _timestamp: float) -> None:
        if self.fail_extract:
            raise RuntimeError("raw frame extraction secret")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"jpeg")

    def extract_frames(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        return []


class MutableScenes:
    def __init__(self) -> None:
        self.intervals = [(0.0, 8.0), (8.0, 20.0)]

    def detect(self, _source: Path, _duration: float) -> list[tuple[float, float]]:
        return list(self.intervals)


class MutableSpeech:
    def __init__(self) -> None:
        self.text = "first spoken generation"
        self.error: Exception | None = None

    def transcribe(self, _source: Path) -> list[TimedText]:
        if self.error is not None:
            raise self.error
        return [TimedText(2.0, 6.0, self.text, 0.91)]


class MutableOCR:
    def __init__(self) -> None:
        self.text = "HOME 87 GUEST 86"
        self.error: Exception | None = None

    def read(self, _image: Path) -> list[tuple[str, float]]:
        if self.error is not None:
            raise self.error
        return [(self.text, 0.88)]


class FakeObjects:
    def detect(self, _image: Path) -> list[ObjectTag]:
        return [ObjectTag("basketball", 0.93)]


class RecordingIndex:
    def __init__(self) -> None:
        self.snapshots: list[list[str]] = []
        self.fail = False

    def replace_video(self, video_id: str, segments) -> None:  # type: ignore[no-untyped-def]
        assert video_id == "video-1"
        if self.fail:
            raise RuntimeError("vector backend leaked detail")
        self.snapshots.append([segment.id for segment in segments])


def _repository_with_legacy_video(tmp_path: Path) -> tuple[Repository, Path]:
    media_root = tmp_path / "media"
    media_root.mkdir()
    source = media_root / "video.mp4"
    source.write_bytes(b"video")
    repository = Repository(tmp_path / "db.sqlite3")
    repository.initialize()
    repository.create_video(
        video_id="video-1",
        original_name="match.mp4",
        stored_name=source.name,
        media_path=str(source),
        size_bytes=source.stat().st_size,
    )
    repository.add_segment(
        segment_id="legacy-speech",
        video_id="video-1",
        start=0,
        end=1,
        modality="speech",
        text="legacy untrusted words",
        confidence=1,
    )
    return repository, media_root


def _indexer(
    tmp_path: Path,
    repository: Repository,
    media_root: Path,
    *,
    ffmpeg: MutableFFmpeg,
    scenes: MutableScenes,
    vector_index: RecordingIndex,
    speech: MutableSpeech | None,
    ocr: MutableOCR | None,
    objects: FakeObjects | None,
) -> Indexer:
    return Indexer(
        repository=repository,
        media_root=media_root,
        thumbnails_dir=tmp_path / "thumbnails",
        specification_resolver=_specifications,
        ffmpeg=ffmpeg,  # type: ignore[arg-type]
        scenes=scenes,  # type: ignore[arg-type]
        speech=speech,
        ocr=ocr,
        objects=objects,
        vector_index=vector_index,  # type: ignore[arg-type]
    )


def test_reindex_activates_new_generations_without_deleting_previous_or_legacy(
    tmp_path,
) -> None:
    repository, media_root = _repository_with_legacy_video(tmp_path)
    ffmpeg = MutableFFmpeg()
    scenes = MutableScenes()
    speech = MutableSpeech()
    ocr = MutableOCR()
    vector_index = RecordingIndex()
    specifications = _specifications()
    indexer = _indexer(
        tmp_path,
        repository,
        media_root,
        ffmpeg=ffmpeg,
        scenes=scenes,
        vector_index=vector_index,
        speech=speech,
        ocr=ocr,
        objects=FakeObjects(),
    )

    indexer.process("video-1")
    first = {
        kind: repository.get_active_segment_generation("video-1", kind)
        for kind in (
            StageKind.SCENES,
            StageKind.SPEECH,
            StageKind.OCR,
            StageKind.OBJECTS,
        )
    }
    first_ids = {segment.id for segment in repository.list_segments("video-1")}
    speech.text = "second spoken generation"
    ocr.text = "HOME 90 GUEST 87"
    scenes.intervals = [(0.0, 20.0)]

    indexer.process("video-1")

    second = {
        kind: repository.get_active_segment_generation("video-1", kind)
        for kind in first
    }
    assert all(first[kind] is not None and second[kind] is not None for kind in first)
    assert all(
        first[kind].generation_id != second[kind].generation_id  # type: ignore[union-attr]
        for kind in first
    )
    assert first_ids <= {segment.id for segment in repository.list_segments("video-1")}
    current = repository.list_current_active_segments(
        specifications.semantic_segment_specifications,
        video_ids=["video-1"],
    )
    assert vector_index.snapshots[-1] == [segment.id for segment in current]
    assert all(segment.modality != "scene" for segment in current)
    assert "legacy-speech" not in vector_index.snapshots[-1]
    assert set(vector_index.snapshots[-1]).isdisjoint(first_ids - {"legacy-speech"})
    assert repository.get_video_asset("video-1") is not None

    old_scene = first[StageKind.SCENES]
    new_scene = second[StageKind.SCENES]
    assert old_scene is not None and new_scene is not None
    old_dir = tmp_path / "thumbnails" / "video-1" / "generations" / old_scene.generation_id
    new_dir = tmp_path / "thumbnails" / "video-1" / "generations" / new_scene.generation_id
    assert old_dir.is_dir()
    assert new_dir.is_dir()


def test_failed_optional_stage_preserves_prior_active_generation_and_sanitizes_error(
    tmp_path,
) -> None:
    repository, media_root = _repository_with_legacy_video(tmp_path)
    ffmpeg = MutableFFmpeg()
    scenes = MutableScenes()
    speech = MutableSpeech()
    ocr = MutableOCR()
    indexer = _indexer(
        tmp_path,
        repository,
        media_root,
        ffmpeg=ffmpeg,
        scenes=scenes,
        vector_index=RecordingIndex(),
        speech=speech,
        ocr=ocr,
        objects=None,
    )
    indexer.process("video-1")
    speech_generation = repository.get_active_segment_generation(
        "video-1", StageKind.SPEECH
    )
    ocr_generation = repository.get_active_segment_generation("video-1", StageKind.OCR)
    speech.error = RuntimeError("secret speech provider message")
    ocr.error = RuntimeError("secret OCR provider message")

    indexer.process("video-1")

    assert repository.get_active_segment_generation("video-1", StageKind.SPEECH) == speech_generation
    assert repository.get_active_segment_generation("video-1", StageKind.OCR) == ocr_generation
    latest_speech = repository.get_latest_stage_run("video-1", StageKind.SPEECH)
    latest_ocr = repository.get_latest_stage_run("video-1", StageKind.OCR)
    assert latest_speech is not None and latest_speech.state is StageState.FAILED
    assert latest_speech.error_code == "speech_provider_failed"
    assert latest_ocr is not None and latest_ocr.state is StageState.FAILED
    assert latest_ocr.error_code == "ocr_provider_failed"
    assert "secret" not in (latest_speech.error_code + latest_ocr.error_code)


def test_absent_optional_providers_are_recorded_as_not_configured(tmp_path) -> None:
    repository, media_root = _repository_with_legacy_video(tmp_path)
    indexer = _indexer(
        tmp_path,
        repository,
        media_root,
        ffmpeg=MutableFFmpeg(),
        scenes=MutableScenes(),
        vector_index=RecordingIndex(),
        speech=None,
        ocr=None,
        objects=None,
    )

    indexer.process("video-1")

    for kind in (StageKind.SPEECH, StageKind.OCR, StageKind.OBJECTS):
        run = repository.get_latest_stage_run("video-1", kind)
        assert run is not None and run.state is StageState.NOT_CONFIGURED
        assert repository.get_active_segment_generation("video-1", kind) is None


def test_failed_scene_build_keeps_thumbnail_active_generation_and_old_directory(
    tmp_path,
) -> None:
    repository, media_root = _repository_with_legacy_video(tmp_path)
    ffmpeg = MutableFFmpeg()
    indexer = _indexer(
        tmp_path,
        repository,
        media_root,
        ffmpeg=ffmpeg,
        scenes=MutableScenes(),
        vector_index=RecordingIndex(),
        speech=None,
        ocr=None,
        objects=None,
    )
    indexer.process("video-1")
    before_video = repository.get_video("video-1")
    before_generation = repository.get_active_segment_generation(
        "video-1", StageKind.SCENES
    )
    assert before_video is not None and before_generation is not None
    ffmpeg.fail_extract = True

    indexer.process("video-1")

    after_video = repository.get_video("video-1")
    assert after_video is not None
    assert after_video.thumbnail_path == before_video.thumbnail_path
    assert repository.get_active_segment_generation(
        "video-1", StageKind.SCENES
    ) == before_generation
    old_dir = (
        tmp_path
        / "thumbnails"
        / "video-1"
        / "generations"
        / before_generation.generation_id
    )
    assert old_dir.is_dir()
    generations = [path for path in old_dir.parent.iterdir() if path.is_dir()]
    assert generations == [old_dir]
    run = repository.get_latest_stage_run("video-1", StageKind.SCENES)
    assert run is not None and run.state is StageState.FAILED
    assert run.error_code == "scene_build_failed"


def test_database_activation_failure_removes_only_new_scene_generation(
    tmp_path,
    monkeypatch,
) -> None:
    repository, media_root = _repository_with_legacy_video(tmp_path)
    indexer = _indexer(
        tmp_path,
        repository,
        media_root,
        ffmpeg=MutableFFmpeg(),
        scenes=MutableScenes(),
        vector_index=RecordingIndex(),
        speech=None,
        ocr=None,
        objects=None,
    )
    indexer.process("video-1")
    active = repository.get_active_segment_generation("video-1", StageKind.SCENES)
    assert active is not None
    original_commit = repository.commit_segment_generation

    def reject_scene_commit(run_id, **kwargs):  # type: ignore[no-untyped-def]
        run = repository.get_stage_run(run_id)
        if run is not None and run.stage_kind is StageKind.SCENES:
            raise RuntimeError("database unavailable")
        return original_commit(run_id, **kwargs)

    monkeypatch.setattr(repository, "commit_segment_generation", reject_scene_commit)

    indexer.process("video-1")

    assert repository.get_active_segment_generation("video-1", StageKind.SCENES) == active
    generations_root = tmp_path / "thumbnails" / "video-1" / "generations"
    assert {path.name for path in generations_root.iterdir()} == {active.generation_id}


def test_vector_failure_is_a_failed_stage_run_not_false_complete(tmp_path) -> None:
    repository, media_root = _repository_with_legacy_video(tmp_path)
    vector_index = RecordingIndex()
    vector_index.fail = True
    indexer = _indexer(
        tmp_path,
        repository,
        media_root,
        ffmpeg=MutableFFmpeg(),
        scenes=MutableScenes(),
        vector_index=vector_index,
        speech=MutableSpeech(),
        ocr=MutableOCR(),
        objects=FakeObjects(),
    )

    indexer.process("video-1")

    run = repository.get_latest_stage_run("video-1", StageKind.TEXT_VECTORS)
    assert run is not None and run.state is StageState.FAILED
    assert run.error_code == "text_vector_index_failed"
    video = repository.get_video("video-1")
    assert video is not None and video.status == "ready"
    assert "vector" in (video.error or "")


def test_indexing_specifications_reject_wrong_stage_assignment() -> None:
    with pytest.raises(ValueError, match="speech"):
        IndexingSpecifications(
            scenes=_specification(StageKind.SCENES),
            speech=_specification(StageKind.OCR),
            ocr=_specification(StageKind.OCR),
            objects=_specification(StageKind.OBJECTS),
            text_vectors=_specification(StageKind.TEXT_VECTORS),
        )


def test_glossary_change_invalidates_speech_until_same_runtime_reindexes(
    tmp_path,
) -> None:
    repository, media_root = _repository_with_legacy_video(tmp_path)
    settings = AppSettings(data_dir=tmp_path / "data")
    settings.ensure_directories()
    settings.glossary_path.write_text('{"Мозгов":["Mozgov"]}', encoding="utf-8")
    resolver = lambda: create_indexing_specifications(settings)
    vector_index = MemoryVectorIndex()
    speech = MutableSpeech()
    speech.text = "Сегодня Мозгов забивает"
    indexer = Indexer(
        repository=repository,
        media_root=media_root,
        thumbnails_dir=tmp_path / "thumbnails",
        specification_resolver=resolver,
        ffmpeg=MutableFFmpeg(),  # type: ignore[arg-type]
        scenes=MutableScenes(),  # type: ignore[arg-type]
        speech=speech,
        ocr=None,
        objects=None,
        vector_index=vector_index,
    )
    search = SearchService(
        repository,
        vector_index,
        specification_resolver=resolver,
        thumbnails_dir=tmp_path / "thumbnails",
    )
    indexer.process("video-1")
    first_specification = resolver().speech

    assert search.search("Мозгов", mode="speech", use_lighthouse=False)
    settings.glossary_path.write_text('{"Мозгов":["Mozgov","Мозгова"]}', encoding="utf-8")
    changed_specification = resolver().speech
    assert changed_specification.specification_hash != first_specification.specification_hash
    assert repository.is_active_segment_generation_current(
        "video-1", changed_specification
    ) is False
    assert search.search("Мозгов", mode="speech", use_lighthouse=False) == []

    indexer.process("video-1")

    assert repository.is_active_segment_generation_current(
        "video-1", changed_specification
    ) is True
    assert search.search("Мозгов", mode="speech", use_lighthouse=False)


def test_successful_unversioned_text_vector_write_is_not_false_complete(tmp_path) -> None:
    repository, media_root = _repository_with_legacy_video(tmp_path)
    indexer = _indexer(
        tmp_path,
        repository,
        media_root,
        ffmpeg=MutableFFmpeg(),
        scenes=MutableScenes(),
        vector_index=RecordingIndex(),
        speech=MutableSpeech(),
        ocr=None,
        objects=None,
    )

    indexer.process("video-1")

    run = repository.get_latest_stage_run("video-1", StageKind.TEXT_VECTORS)
    assert run is None or run.state is not StageState.COMPLETE


def test_versioned_text_vector_build_atomically_activates_and_retains_previous(
    tmp_path,
) -> None:
    repository, media_root = _repository_with_legacy_video(tmp_path)
    vector_index = MemoryVectorIndex()
    speech = MutableSpeech()
    indexer = _indexer(
        tmp_path,
        repository,
        media_root,
        ffmpeg=MutableFFmpeg(),
        scenes=MutableScenes(),
        vector_index=vector_index,  # type: ignore[arg-type]
        speech=speech,
        ocr=MutableOCR(),
        objects=FakeObjects(),
    )

    indexer.process("video-1")
    first = repository.get_active_text_vector_generation("video-1")
    speech.text = "second spoken generation"
    indexer.process("video-1")
    second = repository.get_active_text_vector_generation("video-1")

    assert first is not None and second is not None
    assert second.generation_id != first.generation_id
    assert repository.get_text_vector_generation(first.generation_id) == first
    latest = repository.get_latest_stage_run("video-1", StageKind.TEXT_VECTORS)
    assert latest is not None and latest.state is StageState.COMPLETE
    assert latest.output_generation == second.generation_id


def test_failed_versioned_text_vector_build_preserves_previous_active(tmp_path) -> None:
    class FailingMemoryIndex(MemoryVectorIndex):
        fail = False

        def build_generation(self, plan):  # type: ignore[no-untyped-def]
            if self.fail:
                raise RuntimeError("provider secret must not be persisted")
            return super().build_generation(plan)

    repository, media_root = _repository_with_legacy_video(tmp_path)
    vector_index = FailingMemoryIndex()
    indexer = _indexer(
        tmp_path,
        repository,
        media_root,
        ffmpeg=MutableFFmpeg(),
        scenes=MutableScenes(),
        vector_index=vector_index,  # type: ignore[arg-type]
        speech=MutableSpeech(),
        ocr=None,
        objects=None,
    )
    indexer.process("video-1")
    active = repository.get_active_text_vector_generation("video-1")
    vector_index.fail = True

    indexer.process("video-1")

    assert repository.get_active_text_vector_generation("video-1") == active
    latest = repository.get_latest_stage_run("video-1", StageKind.TEXT_VECTORS)
    assert latest is not None and latest.state is StageState.FAILED
    assert latest.error_code == "text_vector_index_failed"
    assert repository.list_pending_artifact_gc_jobs()


def test_absent_vector_provider_is_not_configured_not_false_complete(tmp_path) -> None:
    repository, media_root = _repository_with_legacy_video(tmp_path)
    indexer = _indexer(
        tmp_path,
        repository,
        media_root,
        ffmpeg=MutableFFmpeg(),
        scenes=MutableScenes(),
        vector_index=EmptyVectorIndex(),  # type: ignore[arg-type]
        speech=MutableSpeech(),
        ocr=None,
        objects=None,
    )

    indexer.process("video-1")

    latest = repository.get_latest_stage_run("video-1", StageKind.TEXT_VECTORS)
    assert latest is not None and latest.state is StageState.NOT_CONFIGURED
    assert repository.get_active_text_vector_generation("video-1") is None


def test_same_size_media_replacement_aborts_before_any_new_stage_run(tmp_path) -> None:
    repository, media_root = _repository_with_legacy_video(tmp_path)
    indexer = _indexer(
        tmp_path,
        repository,
        media_root,
        ffmpeg=MutableFFmpeg(),
        scenes=MutableScenes(),
        vector_index=RecordingIndex(),
        speech=MutableSpeech(),
        ocr=None,
        objects=None,
    )
    indexer.process("video-1")
    before_runs = repository.list_stage_runs("video-1")
    before_scene = repository.get_active_segment_generation(
        "video-1", StageKind.SCENES
    )
    (media_root / "video.mp4").write_bytes(b"other")

    with pytest.raises(AssetIdentityError, match="immutable asset"):
        indexer.process("video-1")

    assert repository.list_stage_runs("video-1") == before_runs
    assert repository.get_active_segment_generation(
        "video-1", StageKind.SCENES
    ) == before_scene


def test_symlinked_generation_root_fails_without_touching_outside_files(tmp_path) -> None:
    repository, media_root = _repository_with_legacy_video(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    video_thumbnail_root = tmp_path / "thumbnails" / "video-1"
    video_thumbnail_root.mkdir(parents=True)
    (video_thumbnail_root / "generations").symlink_to(
        outside,
        target_is_directory=True,
    )
    indexer = _indexer(
        tmp_path,
        repository,
        media_root,
        ffmpeg=MutableFFmpeg(),
        scenes=MutableScenes(),
        vector_index=RecordingIndex(),
        speech=None,
        ocr=None,
        objects=None,
    )

    indexer.process("video-1")

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert list(outside.iterdir()) == [sentinel]
    assert repository.get_active_segment_generation(
        "video-1", StageKind.SCENES
    ) is None
    run = repository.get_latest_stage_run("video-1", StageKind.SCENES)
    assert run is not None and run.state is StageState.FAILED


def test_glossary_change_during_transcription_discards_candidate_generation(
    tmp_path,
) -> None:
    repository, media_root = _repository_with_legacy_video(tmp_path)
    settings = AppSettings(data_dir=tmp_path / "data")
    settings.ensure_directories()
    settings.glossary_path.write_text('{"Мозгов":["Mozgov"]}', encoding="utf-8")
    resolver = lambda: create_indexing_specifications(settings)
    speech = MutableSpeech()
    indexer = Indexer(
        repository=repository,
        media_root=media_root,
        thumbnails_dir=tmp_path / "thumbnails",
        specification_resolver=resolver,
        ffmpeg=MutableFFmpeg(),  # type: ignore[arg-type]
        scenes=MutableScenes(),  # type: ignore[arg-type]
        speech=speech,
        ocr=None,
        objects=None,
        vector_index=RecordingIndex(),  # type: ignore[arg-type]
    )
    indexer.process("video-1")
    active_before = repository.get_active_segment_generation(
        "video-1", StageKind.SPEECH
    )

    class MutatingSpeech:
        def transcribe(self, _source: Path) -> list[TimedText]:
            settings.glossary_path.write_text(
                '{"Мозгов":["Mozgov","Мозгова"]}',
                encoding="utf-8",
            )
            return [TimedText(2.0, 6.0, "new candidate", 0.9)]

    indexer.speech = MutatingSpeech()
    indexer.process("video-1")

    assert repository.get_active_segment_generation(
        "video-1", StageKind.SPEECH
    ) == active_before
    run = repository.get_latest_stage_run("video-1", StageKind.SPEECH)
    assert run is not None and run.state is StageState.FAILED
    assert run.error_code == "speech_specification_changed"


def test_scene_cleanup_oserror_does_not_leave_run_running_or_replace_active(
    tmp_path,
    monkeypatch,
) -> None:
    repository, media_root = _repository_with_legacy_video(tmp_path)
    ffmpeg = MutableFFmpeg()
    indexer = _indexer(
        tmp_path,
        repository,
        media_root,
        ffmpeg=ffmpeg,
        scenes=MutableScenes(),
        vector_index=RecordingIndex(),
        speech=None,
        ocr=None,
        objects=None,
    )
    indexer.process("video-1")
    active = repository.get_active_segment_generation("video-1", StageKind.SCENES)
    ffmpeg.fail_extract = True

    def failed_cleanup(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise OSError("permission denied")

    monkeypatch.setattr(indexer, "_remove_generated_directory", failed_cleanup)

    indexer.process("video-1")

    assert repository.get_active_segment_generation("video-1", StageKind.SCENES) == active
    run = repository.get_latest_stage_run("video-1", StageKind.SCENES)
    assert run is not None and run.state is StageState.FAILED
    assert run.error_code == "scene_build_failed"
