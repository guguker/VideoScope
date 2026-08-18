from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import videoscope.search.visual_index as visual_index_module
from videoscope.config import AppSettings
from videoscope.evaluation import evaluation_runtime_revision
from videoscope.media.ffmpeg import SampledFrame
from videoscope.runtime import create_visual_index
from videoscope.search.visual_index import (
    SiglipVisualIndex,
    VisualIndexSpecification,
)


class RecordingDenseExtractor:
    def __init__(self, *, fail: Exception | None = None) -> None:
        self.fail = fail
        self.calls: list[tuple[Path, float, float, float, int]] = []

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
        self.calls.append((source, start, end, step, max_width))
        if self.fail is not None:
            raise self.fail
        destination.mkdir(parents=True, exist_ok=True)
        frames: list[SampledFrame] = []
        timestamp = start
        while timestamp < end:
            path = destination / f"sample-{len(frames):04d}.jpg"
            path.write_bytes(f"frame-{timestamp}".encode())
            frames.append(SampledFrame(timestamp, path))
            timestamp += step
        return frames


def _source(tmp_path: Path, content: bytes = b"deterministic-video") -> Path:
    source = tmp_path / "video.mp4"
    source.write_bytes(content)
    return source


def _index(tmp_path: Path, **overrides: object) -> SiglipVisualIndex:
    options: dict[str, object] = {
        "model_name": "organization/model",
        "model_revision": "a" * 40,
        "sample_step": 2.0,
        "max_width": 512,
        "sampling_strategy": "fixed_step_seconds",
        "preprocessing_revision": "siglip-rgb-v1",
        "schema_version": 2,
        "extractor_identity": "ffmpeg-fixed-step-v1",
    }
    options.update(overrides)
    return SiglipVisualIndex(tmp_path / "index", **options)  # type: ignore[arg-type]


def _install_fake_embeddings(
    monkeypatch: pytest.MonkeyPatch,
    index: SiglipVisualIndex,
    *,
    value: float = 1.0,
) -> None:
    monkeypatch.setattr(
        index,
        "_image_vectors",
        lambda paths: np.full((len(paths), 2), value, dtype=np.float32),
    )
    monkeypatch.setattr(
        index,
        "_text_vectors",
        lambda _query: np.array([[1.0, 0.0]], dtype=np.float32),
    )


def test_visual_specification_identity_covers_every_compatibility_dimension() -> None:
    specification = VisualIndexSpecification(
        model_name="organization/model",
        model_revision="a" * 40,
        sampling_strategy="fixed_step_seconds",
        sample_step=1.0,
        max_width=640,
        preprocessing_revision="siglip-rgb-v1",
        schema_version=2,
        extractor_identity="ffmpeg-fixed-step-v1",
    )

    assert specification.identity == VisualIndexSpecification(
        **specification.to_dict()
    ).identity
    for changed in (
        replace(specification, model_revision="b" * 40),
        replace(specification, sample_step=0.5),
        replace(specification, max_width=384),
        replace(specification, preprocessing_revision="siglip-rgb-v2"),
        replace(specification, schema_version=3),
        replace(specification, extractor_identity="ffmpeg-fixed-step-v2"),
        replace(specification, sampling_strategy="fixed_step_frames"),
    ):
        assert changed.identity != specification.identity


def test_dense_build_persists_and_atomically_activates_a_valid_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    index = _index(tmp_path)
    extractor = RecordingDenseExtractor()
    _install_fake_embeddings(monkeypatch, index)

    generation_id = index.replace_video_source(
        "video-1",
        source,
        5.0,
        extractor,
        frames_dir=tmp_path / "thumbnails" / "video-1",
    )

    assert extractor.calls == [(source, 0.0, 5.0, 2.0, 512)]
    assert index.active_generation_id("video-1") == generation_id
    generation = tmp_path / "index" / "video-1" / "generations" / generation_id
    manifest = json.loads((generation / "manifest.json").read_text(encoding="utf-8"))
    metadata = json.loads((generation / "metadata.json").read_text(encoding="utf-8"))
    assert manifest["generation_id"] == generation_id
    assert manifest["specification"] == index.specification.to_dict()
    assert manifest["specification_hash"] == index.specification.identity
    assert manifest["source_sha256"] == sha256(source.read_bytes()).hexdigest()
    assert manifest["duration_seconds"] == 5.0
    assert manifest["frame_count"] == 3
    assert [item["start"] for item in metadata] == [0.0, 2.0, 4.0]
    assert index.index_is_current("video-1") is True
    assert index.search("query", video_ids=["video-1"])


@pytest.mark.parametrize(
    "override",
    [
        {"model_revision": "b" * 40},
        {"sample_step": 0.5},
        {"max_width": 384},
        {"preprocessing_revision": "siglip-rgb-v2"},
        {"schema_version": 3},
    ],
)
def test_changed_visual_specification_makes_active_generation_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    override: dict[str, object],
) -> None:
    source = _source(tmp_path)
    current = _index(tmp_path)
    _install_fake_embeddings(monkeypatch, current)
    current.replace_video_source("video-1", source, 3.0, RecordingDenseExtractor())

    changed = _index(tmp_path, **override)

    assert changed.index_is_current("video-1") is False


def test_scene_model_only_legacy_artifacts_are_stale_and_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = _index(tmp_path)
    legacy = tmp_path / "index" / "video-1"
    legacy.mkdir(parents=True)
    np.save(legacy / "vectors.npy", np.array([[1.0]], dtype=np.float32))
    (legacy / "metadata.json").write_text(
        '[{"segment_id":"scene","start":0,"end":1,"thumbnail_path":null}]',
        encoding="utf-8",
    )
    (legacy / "model.txt").write_text(index.model_identity, encoding="utf-8")

    assert index.index_is_current("video-1") is False
    assert (legacy / "vectors.npy").is_file()
    assert (legacy / "metadata.json").is_file()
    assert (legacy / "model.txt").is_file()

    source = _source(tmp_path)
    _install_fake_embeddings(monkeypatch, index)
    index.replace_video_source("video-1", source, 3.0, RecordingDenseExtractor())

    assert index.index_is_current("video-1") is True
    assert index._has_incompatible_indexes() is False
    assert (legacy / "vectors.npy").is_file()


@pytest.mark.parametrize("failure_stage", ["extraction", "embedding", "persistence", "validation"])
def test_failed_rebuild_preserves_previous_active_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    source = _source(tmp_path)
    index = _index(tmp_path)
    _install_fake_embeddings(monkeypatch, index)
    first_generation = index.replace_video_source(
        "video-1", source, 3.0, RecordingDenseExtractor()
    )

    extractor = RecordingDenseExtractor(
        fail=RuntimeError("extraction failed") if failure_stage == "extraction" else None
    )
    if failure_stage == "embedding":
        monkeypatch.setattr(
            index,
            "_image_vectors",
            lambda _paths: (_ for _ in ()).throw(RuntimeError("embedding failed")),
        )
    elif failure_stage == "persistence":
        original_write = visual_index_module.atomic_write_json

        def fail_metadata(path: Path, payload: object, **kwargs: object) -> None:
            if path.name == "metadata.json":
                raise OSError("disk full")
            original_write(path, payload, **kwargs)

        monkeypatch.setattr(visual_index_module, "atomic_write_json", fail_metadata)
    elif failure_stage == "validation":
        monkeypatch.setattr(
            index,
            "_image_vectors",
            lambda paths: np.full((len(paths), 2), np.nan, dtype=np.float32),
        )

    with pytest.raises((OSError, RuntimeError, ValueError)):
        index.replace_video_source("video-1", source, 3.0, extractor)

    assert index.active_generation_id("video-1") == first_generation
    assert index.index_is_current("video-1") is True


def test_failed_activation_preserves_previous_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    index = _index(tmp_path)
    _install_fake_embeddings(monkeypatch, index)
    first_generation = index.replace_video_source(
        "video-1", source, 3.0, RecordingDenseExtractor()
    )
    original_write = visual_index_module.atomic_write_json

    def fail_activation(path: Path, payload: object, **kwargs: object) -> None:
        if path.name == "active.json":
            raise OSError("activation interrupted")
        original_write(path, payload, **kwargs)

    monkeypatch.setattr(visual_index_module, "atomic_write_json", fail_activation)

    with pytest.raises(OSError, match="activation interrupted"):
        index.replace_video_source("video-1", source, 3.0, RecordingDenseExtractor())

    assert index.active_generation_id("video-1") == first_generation
    assert index.index_is_current("video-1") is True


def test_source_mutation_during_build_preserves_previous_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    index = _index(tmp_path)
    _install_fake_embeddings(monkeypatch, index)
    first_generation = index.replace_video_source(
        "video-1", source, 3.0, RecordingDenseExtractor()
    )

    class MutatingExtractor(RecordingDenseExtractor):
        def extract_frames(self, *args: object, **kwargs: object) -> list[SampledFrame]:
            frames = super().extract_frames(*args, **kwargs)  # type: ignore[arg-type]
            source.write_bytes(b"mutated-during-extraction")
            return frames

    with pytest.raises(RuntimeError, match="source changed"):
        index.replace_video_source("video-1", source, 3.0, MutatingExtractor())

    assert index.active_generation_id("video-1") == first_generation


def test_extractor_symlink_frame_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    index = _index(tmp_path)
    _install_fake_embeddings(monkeypatch, index)
    class SymlinkExtractor:
        def extract_frames(
            self,
            _source: Path,
            destination: Path,
            start: float,
            _end: float,
            *,
            step: float,
            max_width: int = 640,
        ) -> list[SampledFrame]:
            del step, max_width
            destination.mkdir(parents=True, exist_ok=True)
            target = destination / "real.jpg"
            target.write_bytes(b"inside")
            path = destination / "sample-0000.jpg"
            path.symlink_to(target)
            return [SampledFrame(start, path)]

    with pytest.raises(ValueError, match="unsafe frame"):
        index.replace_video_source("video-1", source, 3.0, SymlinkExtractor())

    assert index.active_generation_id("video-1") is None


@pytest.mark.parametrize(
    "corruption",
    [
        "active-json",
        "active-types",
        "manifest-json",
        "manifest-specification",
        "manifest-source",
        "metadata-order",
        "thumbnail-path",
        "metadata-bounds",
        "metadata-types",
    ],
)
def test_interrupted_or_malformed_generation_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    source = _source(tmp_path)
    index = _index(tmp_path)
    _install_fake_embeddings(monkeypatch, index)
    generation_id = index.replace_video_source(
        "video-1", source, 3.0, RecordingDenseExtractor()
    )
    video_dir = tmp_path / "index" / "video-1"
    generation = video_dir / "generations" / generation_id
    if corruption == "active-json":
        (video_dir / "active.json").write_text("{", encoding="utf-8")
    elif corruption == "active-types":
        (video_dir / "active.json").write_text(
            json.dumps({"generation_id": generation_id, "schema_version": True}),
            encoding="utf-8",
        )
    elif corruption == "manifest-json":
        (generation / "manifest.json").write_text("{", encoding="utf-8")
    elif corruption == "manifest-specification":
        manifest = json.loads((generation / "manifest.json").read_text(encoding="utf-8"))
        manifest["specification_hash"] = "0" * 64
        (generation / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    elif corruption == "manifest-source":
        manifest = json.loads((generation / "manifest.json").read_text(encoding="utf-8"))
        manifest["source_sha256"] = "not-a-hash"
        (generation / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    elif corruption == "metadata-order":
        metadata = json.loads((generation / "metadata.json").read_text(encoding="utf-8"))
        metadata.reverse()
        (generation / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    elif corruption == "thumbnail-path":
        metadata = json.loads((generation / "metadata.json").read_text(encoding="utf-8"))
        metadata[0]["thumbnail_path"] = "/does/not/exist.jpg"
        (generation / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    elif corruption == "metadata-bounds":
        metadata = json.loads((generation / "metadata.json").read_text(encoding="utf-8"))
        metadata[-1]["end"] = 10_000
        (generation / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    else:
        metadata = json.loads((generation / "metadata.json").read_text(encoding="utf-8"))
        metadata[0]["start"] = True
        (generation / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    assert index.index_is_current("video-1") is False
    assert index.search("query", video_ids=["video-1"]) == []


def test_runtime_identity_contains_full_specification_and_active_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    index = _index(tmp_path)
    _install_fake_embeddings(monkeypatch, index)
    before = evaluation_runtime_revision(SimpleNamespace(visual_search=index))
    generation_id = index.replace_video_source(
        "video-1", source, 3.0, RecordingDenseExtractor()
    )
    after = evaluation_runtime_revision(SimpleNamespace(visual_search=index))
    identity = json.loads(index.identity)

    assert before != after
    assert identity["specification"] == index.specification.to_dict()
    assert identity["active_generations"] == [
        {
            "duration_seconds": 3.0,
            "generation_id": generation_id,
            "source_sha256": identity["active_generations"][0]["source_sha256"],
            "source_size_bytes": len(source.read_bytes()),
            "state": "current",
            "video_id": "video-1",
        }
    ]


def test_runtime_identity_marks_corrupt_active_generation_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    index = _index(tmp_path)
    _install_fake_embeddings(monkeypatch, index)
    generation_id = index.replace_video_source(
        "video-1", source, 3.0, RecordingDenseExtractor()
    )
    before = evaluation_runtime_revision(SimpleNamespace(visual_search=index))
    generation = tmp_path / "index" / "video-1" / "generations" / generation_id
    np.save(
        generation / "vectors.npy",
        np.array([[np.nan, 0.0], [1.0, 0.0]], dtype=np.float32),
        allow_pickle=False,
    )

    identity = json.loads(index.identity)
    after = evaluation_runtime_revision(SimpleNamespace(visual_search=index))

    assert identity["active_generations"] == [
        {
            "generation_id": generation_id,
            "state": "invalid",
            "video_id": "video-1",
        }
    ]
    assert after != before


def test_runtime_and_maintenance_factory_share_visual_specification(tmp_path: Path) -> None:
    settings = AppSettings(
        data_dir=tmp_path / "data",
        siglip_model="organization/model",
        visual_index_step=0.75,
        visual_index_max_width=384,
    )

    runtime_index = create_visual_index(settings)
    maintenance_index = create_visual_index(settings)

    assert runtime_index.specification == maintenance_index.specification
    assert runtime_index.specification.sample_step == 0.75
    assert runtime_index.specification.max_width == 384
