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
from videoscope.providers.vision_worker_client import VisionCapabilityStatus
from videoscope.providers.vision_worker_contract import (
    RFDETR_SMALL_CHECKPOINT_SHA256,
    VisionWorkerSpecification,
)
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


class FakeVisionInferenceClient:
    def __init__(self, specification: VisionWorkerSpecification) -> None:
        self.specification = specification
        self.invalid_vectors = False
        self.text_batches: list[list[str]] = []

    @property
    def identity(self) -> dict[str, object]:
        return {
            "mode": "isolated-worker",
            "specification_hash": self.specification.identity,
            "siglip_specification_hash": self.specification.siglip_identity,
            "detector_specification_hash": self.specification.detector_identity,
        }

    def capability(self) -> VisionCapabilityStatus:
        return VisionCapabilityStatus(True, "fixture worker is ready")

    def image_vectors(self, paths: list[Path]) -> np.ndarray:
        if self.invalid_vectors:
            return np.full((len(paths), 2), np.nan, dtype=np.float32)
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(paths), 1))

    def text_vectors(self, texts: list[str]) -> np.ndarray:
        self.text_batches.append(list(texts))
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))


def _worker_specification(
    *,
    detector_checkpoint_sha256: str = RFDETR_SMALL_CHECKPOINT_SHA256,
    siglip_logit_scale: float = 10.0,
) -> VisionWorkerSpecification:
    return VisionWorkerSpecification(
        siglip_model="google/siglip2-base-patch16-224",
        siglip_revision="75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
        embedding_dimensions=2,
        detector_model_id="rfdetr-small",
        detector_checkpoint_sha256=detector_checkpoint_sha256,
        siglip_logit_scale=siglip_logit_scale,
        siglip_logit_bias=-1.0,
        minimum_confidence=0.25,
    )


def _worker_index(
    tmp_path: Path,
    client: FakeVisionInferenceClient,
) -> SiglipVisualIndex:
    return SiglipVisualIndex(
        tmp_path / "index",
        model_name=client.specification.siglip_model,
        model_revision=client.specification.siglip_revision,
        sample_step=2.0,
        max_width=512,
        inference_client=client,
    )


def test_event_search_routes_each_temporal_prompt_through_the_worker(
    tmp_path: Path,
) -> None:
    client = FakeVisionInferenceClient(_worker_specification())
    index = _worker_index(tmp_path, client)
    index.replace_video_source(
        "video-1",
        _source(tmp_path),
        6.0,
        RecordingDenseExtractor(),
    )

    index.search(
        "игрок реализует штрафной",
        video_ids=["video-1"],
        limit=5,
    )

    assert len(client.text_batches) > 1
    assert all(len(batch) == 1 for batch in client.text_batches)


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

    with pytest.raises(ValueError, match="max width"):
        replace(specification, max_width=8_193)


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


def test_dense_durable_build_is_inactive_until_explicit_legacy_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    index = _index(tmp_path)
    _install_fake_embeddings(monkeypatch, index)

    descriptor = index.build_video_source(
        "video-1",
        source,
        5.0,
        RecordingDenseExtractor(),
        frames_dir=tmp_path / "thumbnails" / "video-1",
    )
    generation_id = str(descriptor["generation_id"])

    assert index.active_generation_id("video-1") is None
    assert descriptor == {
        "content_sha256": descriptor["content_sha256"],
        "duration_seconds": 5.0,
        "generation_id": generation_id,
        "source_sha256": sha256(source.read_bytes()).hexdigest(),
        "source_size_bytes": len(source.read_bytes()),
        "specification_hash": index.specification.identity,
        "video_id": "video-1",
    }

    index.activate_generation("video-1", generation_id)

    assert index.active_generation_id("video-1") == generation_id


def test_dense_legacy_replacement_remains_build_plus_activate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = _index(tmp_path)
    _install_fake_embeddings(monkeypatch, index)

    generation_id = index.replace_video_source(
        "video-1",
        _source(tmp_path),
        5.0,
        RecordingDenseExtractor(),
    )

    assert index.active_generation_id("video-1") == generation_id


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


def test_oversized_active_pointer_is_rejected_without_unbounded_path_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = _index(tmp_path)
    _install_fake_embeddings(monkeypatch, index)
    generation_id = index.replace_video_source(
        "video-1",
        _source(tmp_path),
        3.0,
        RecordingDenseExtractor(),
    )
    assert index.active_generation_id("video-1") == generation_id
    pointer = tmp_path / "index" / "video-1" / "active.json"
    pointer.write_bytes(
        b"{" + b" " * visual_index_module._MAX_VISUAL_ACTIVE_POINTER_BYTES + b"}"
    )
    native_read_bytes = Path.read_bytes

    def reject_unbounded_read(path: Path) -> bytes:
        if path == pointer:
            raise AssertionError("active pointer used an unbounded path read")
        return native_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_unbounded_read)

    assert index.active_generation_id("video-1") is None


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


def test_malformed_worker_vectors_preserve_previous_active_generation(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    client = FakeVisionInferenceClient(_worker_specification())
    index = _worker_index(tmp_path, client)
    first_generation = index.replace_video_source(
        "video-1",
        source,
        3.0,
        RecordingDenseExtractor(),
    )
    client.invalid_vectors = True

    with pytest.raises(ValueError, match="finite frame rows"):
        index.replace_video_source(
            "video-1",
            source,
            3.0,
            RecordingDenseExtractor(),
        )

    assert index.active_generation_id("video-1") == first_generation
    assert index.index_is_current("video-1") is True


def test_worker_siglip_identity_change_stales_generation_but_detector_change_does_not(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    baseline = _worker_index(
        tmp_path,
        FakeVisionInferenceClient(_worker_specification()),
    )
    baseline.replace_video_source(
        "video-1",
        source,
        3.0,
        RecordingDenseExtractor(),
    )
    detector_changed = _worker_index(
        tmp_path,
        FakeVisionInferenceClient(
            _worker_specification(detector_checkpoint_sha256="a" * 64)
        ),
    )
    siglip_worker_changed = _worker_index(
        tmp_path,
        FakeVisionInferenceClient(
            _worker_specification(siglip_logit_scale=11.0)
        ),
    )

    assert baseline.specification.schema_version == 3
    assert detector_changed.specification.identity == baseline.specification.identity
    assert detector_changed.index_is_current("video-1") is True
    assert siglip_worker_changed.specification.identity != baseline.specification.identity
    assert siglip_worker_changed.index_is_current("video-1") is False


def test_generation_bound_search_does_not_follow_replaced_active_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    index = _index(tmp_path)
    monkeypatch.setattr(
        index,
        "_text_vectors",
        lambda _query: np.array([[1.0, 0.0]], dtype=np.float32),
    )
    monkeypatch.setattr(
        index,
        "_image_vectors",
        lambda paths: np.tile(
            np.array([[1.0, 0.0]], dtype=np.float32),
            (len(paths), 1),
        ),
    )
    first = index.replace_video_source(
        "video-1",
        source,
        3.0,
        RecordingDenseExtractor(),
    )
    binding = index.generation_descriptor("video-1", first)
    assert binding is not None
    monkeypatch.setattr(
        index,
        "_image_vectors",
        lambda paths: np.tile(
            np.array([[0.0, 1.0]], dtype=np.float32),
            (len(paths), 1),
        ),
    )
    second = index.replace_video_source(
        "video-1",
        source,
        3.0,
        RecordingDenseExtractor(),
    )

    pinned = index.search_generations(
        "query",
        generation_bindings={"video-1": binding},
        limit=5,
    )
    active = index.search("query", video_ids=["video-1"], limit=5)

    assert index.active_generation_id("video-1") == second
    assert first != second
    assert pinned[0].score > active[0].score


@pytest.mark.parametrize("load_error", [None, OSError("transient read failure")])
@pytest.mark.parametrize("descriptor_cached", [True, False])
def test_generation_bound_search_fails_closed_on_transient_generation_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_error: OSError | None,
    descriptor_cached: bool,
) -> None:
    source = _source(tmp_path)
    index = _index(tmp_path)
    _install_fake_embeddings(monkeypatch, index)
    generation_id = index.replace_video_source(
        "video-1",
        source,
        3.0,
        RecordingDenseExtractor(),
    )
    binding = index.generation_descriptor("video-1", generation_id)
    assert binding is not None
    if not descriptor_cached:
        index._generation_descriptor_cache.clear()
    native_load = index._load_generation
    fail_next_load = True

    def fail_once(
        video_id: str,
        requested_generation_id: str,
        *,
        memory_map: bool = False,
    ):  # type: ignore[no-untyped-def]
        nonlocal fail_next_load
        if fail_next_load:
            fail_next_load = False
            if load_error is not None:
                raise load_error
            return None
        return native_load(
            video_id,
            requested_generation_id,
            memory_map=memory_map,
        )

    monkeypatch.setattr(index, "_load_generation", fail_once)

    with pytest.raises(RuntimeError, match="bound visual generation is unavailable"):
        index.search_generations(
            "query",
            generation_bindings={"video-1": binding},
            limit=5,
        )

    assert index.search_generations(
        "query",
        generation_bindings={"video-1": binding},
        limit=5,
    )
    assert index.generation_descriptor("video-1", generation_id) == binding


def test_generation_binding_covers_mutated_ranking_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    index = _index(tmp_path)
    _install_fake_embeddings(monkeypatch, index)
    generation_id = index.replace_video_source(
        "video-1",
        source,
        3.0,
        RecordingDenseExtractor(),
    )
    binding = index.generation_descriptor("video-1", generation_id)
    assert binding is not None
    vectors_path = (
        tmp_path
        / "index"
        / "video-1"
        / "generations"
        / generation_id
        / "vectors.npy"
    )
    vectors = np.load(vectors_path, allow_pickle=False)
    mutated = vectors.copy()
    mutated[:, 0] = 0.123
    with vectors_path.open("wb") as handle:
        np.save(handle, mutated, allow_pickle=False)

    with pytest.raises(ValueError, match="stale or corrupt"):
        index.search_generations(
            "query",
            generation_bindings={"video-1": binding},
            limit=5,
        )


def test_generation_descriptor_reuses_content_digest_until_stat_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    index = _index(tmp_path)
    _install_fake_embeddings(monkeypatch, index)
    generation_id = index.replace_video_source(
        "video-1",
        source,
        3.0,
        RecordingDenseExtractor(),
    )
    native_identity = visual_index_module._stable_regular_file_identity
    content_reads = 0

    def recording_identity(path: Path) -> tuple[str, int]:
        nonlocal content_reads
        content_reads += 1
        return native_identity(path)

    monkeypatch.setattr(
        visual_index_module,
        "_stable_regular_file_identity",
        recording_identity,
    )

    first = index.generation_descriptor("video-1", generation_id)
    second = index.generation_descriptor("video-1", generation_id)
    third = index.generation_descriptor("video-1", generation_id)
    final = index.generation_descriptor(
        "video-1",
        generation_id,
        force_content_validation=True,
    )

    assert first == second == third == final
    assert content_reads == 6


@pytest.mark.parametrize(
    ("artifact_name", "limit_name", "json_prefix"),
    [
        ("manifest.json", "_MAX_VISUAL_MANIFEST_BYTES", b"{"),
        ("metadata.json", "_MAX_VISUAL_METADATA_BYTES", b"["),
    ],
)
def test_generation_descriptor_rejects_oversized_json_before_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_name: str,
    limit_name: str,
    json_prefix: bytes,
) -> None:
    index = _index(tmp_path)
    _install_fake_embeddings(monkeypatch, index)
    generation_id = index.replace_video_source(
        "video-1",
        _source(tmp_path),
        3.0,
        RecordingDenseExtractor(),
    )
    artifact = (
        tmp_path
        / "index"
        / "video-1"
        / "generations"
        / generation_id
        / artifact_name
    )
    artifact_bytes = artifact.read_bytes()
    byte_limit = len(artifact_bytes) - 1
    monkeypatch.setattr(visual_index_module, limit_name, byte_limit)
    native_loads = json.loads

    def reject_oversized_parse(raw, *args, **kwargs):  # type: ignore[no-untyped-def]
        encoded = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
        if encoded.lstrip().startswith(json_prefix) and len(encoded) > byte_limit:
            raise AssertionError("oversized JSON reached the parser")
        return native_loads(raw, *args, **kwargs)

    monkeypatch.setattr(visual_index_module.json, "loads", reject_oversized_parse)

    assert index.generation_descriptor("video-1", generation_id) is None


@pytest.mark.parametrize(
    ("mutations", "limit_name"),
    [
        (
            {"frame_count": "limit-plus-one", "vector_count": "limit-plus-one"},
            "_MAX_VISUAL_FRAME_COUNT",
        ),
        ({"vector_dimensions": "limit-plus-one"}, "_MAX_VISUAL_VECTOR_DIMENSIONS"),
    ],
)
def test_generation_descriptor_rejects_manifest_resource_claims_before_vector_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutations: dict[str, str],
    limit_name: str,
) -> None:
    index = _index(tmp_path)
    _install_fake_embeddings(monkeypatch, index)
    generation_id = index.replace_video_source(
        "video-1",
        _source(tmp_path),
        3.0,
        RecordingDenseExtractor(),
    )
    manifest_path = (
        tmp_path
        / "index"
        / "video-1"
        / "generations"
        / generation_id
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    claimed = getattr(visual_index_module, limit_name) + 1
    for field_name in mutations:
        manifest[field_name] = claimed
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        visual_index_module.np,
        "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("oversized vector claim reached numpy")
        ),
    )

    assert index.generation_descriptor("video-1", generation_id) is None


def test_generation_descriptor_rejects_vector_byte_budget_before_numpy_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = _index(tmp_path)
    _install_fake_embeddings(monkeypatch, index)
    generation_id = index.replace_video_source(
        "video-1",
        _source(tmp_path),
        3.0,
        RecordingDenseExtractor(),
    )
    monkeypatch.setattr(visual_index_module, "_MAX_VISUAL_VECTOR_BYTES", 1)
    monkeypatch.setattr(
        visual_index_module.np,
        "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("oversized vector file reached numpy")
        ),
    )

    assert index.generation_descriptor("video-1", generation_id) is None


def test_generation_descriptor_rejects_non_float32_vector_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = _index(tmp_path)
    _install_fake_embeddings(monkeypatch, index)
    generation_id = index.replace_video_source(
        "video-1",
        _source(tmp_path),
        3.0,
        RecordingDenseExtractor(),
    )
    vectors_path = (
        tmp_path
        / "index"
        / "video-1"
        / "generations"
        / generation_id
        / "vectors.npy"
    )
    vectors = np.load(vectors_path, allow_pickle=False)
    np.save(vectors_path, vectors.astype(np.float64), allow_pickle=False)

    assert index.generation_descriptor("video-1", generation_id) is None


def test_generation_descriptor_rejects_oversized_frame_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = _index(tmp_path)
    _install_fake_embeddings(monkeypatch, index)
    generation_id = index.replace_video_source(
        "video-1",
        _source(tmp_path),
        3.0,
        RecordingDenseExtractor(),
    )
    generation = (
        tmp_path / "index" / "video-1" / "generations" / generation_id
    )
    metadata = json.loads((generation / "metadata.json").read_text(encoding="utf-8"))
    frame = generation / metadata[0]["frame_path"]
    monkeypatch.setattr(
        visual_index_module,
        "_MAX_VISUAL_FRAME_BYTES",
        frame.stat().st_size - 1,
    )

    assert index.generation_descriptor("video-1", generation_id) is None
