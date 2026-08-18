import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

import videoscope.search.visual_index as visual_index_module
from videoscope.media.ffmpeg import SampledFrame
from videoscope.search.sports_events import classify_basketball_event_query
from videoscope.search.visual_index import (
    SiglipVisualIndex,
    made_three_point_prompt_stages,
    visual_prompt_variants,
)


def active_generation_path(tmp_path: Path, index: SiglipVisualIndex) -> Path:
    generation_id = index.active_generation_id("video-1")
    assert generation_id is not None
    return tmp_path / "index" / "video-1" / "generations" / generation_id


def test_visual_index_persists_frames_and_returns_closest_scene(tmp_path, monkeypatch) -> None:
    index = SiglipVisualIndex(
        tmp_path / "index", model_name="test", batch_size=2, sample_step=10.0
    )
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(
        index,
        "_image_vectors",
        lambda _paths: np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    monkeypatch.setattr(index, "_text_vector", lambda _query: np.array([0.1, 0.9], dtype=np.float32))

    assert index.index_is_current("video-1") is False

    index.replace_video_source(
        "video-1", source, 15.0, DenseFrameExtractor()
    )
    hits = index.search("right", video_ids=["video-1"], limit=2)

    assert hits[0].start == 10
    assert hits[0].modality == "visual"
    assert hits[0].metadata["source"] == "siglip2"
    generation = active_generation_path(tmp_path, index)
    metadata = json.loads((generation / "metadata.json").read_text())
    manifest = json.loads((generation / "manifest.json").read_text())
    assert len(metadata) == 2
    assert manifest["specification_hash"] == index.specification.identity
    assert index.index_is_current("video-1") is True


def test_visual_index_readiness_rejects_stale_or_malformed_artifacts(tmp_path) -> None:
    index = SiglipVisualIndex(tmp_path / "index", model_name="current")
    video_dir = tmp_path / "index" / "video-1"
    video_dir.mkdir(parents=True)
    np.save(
        video_dir / "vectors.npy",
        np.array([[1.0, 0.0]], dtype=np.float32),
        allow_pickle=False,
    )
    metadata_path = video_dir / "metadata.json"
    metadata_path.write_text(
        '[{"segment_id":"scene","start":0,"end":1,"thumbnail_path":null}]',
        encoding="utf-8",
    )
    model_path = video_dir / "model.txt"
    model_path.write_text("stale", encoding="utf-8")

    assert index.index_is_current("video-1") is False

    model_path.write_text("current", encoding="utf-8")
    assert index.index_is_current("video-1") is False

    metadata_path.write_text('{"not":"a-list"}', encoding="utf-8")
    assert index.index_is_current("video-1") is False

    metadata_path.write_text(
        '[{"segment_id":"scene","start":0,"end":1,"thumbnail_path":null}]',
        encoding="utf-8",
    )
    np.save(
        video_dir / "vectors.npy",
        np.array([["not", "numeric"]]),
        allow_pickle=False,
    )
    assert index.index_is_current("video-1") is False


def test_visual_index_ignores_vectors_from_another_model(tmp_path, monkeypatch) -> None:
    first = SiglipVisualIndex(tmp_path / "index", model_name="first")
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(first, "_image_vectors", lambda _paths: np.array([[1.0]], dtype=np.float32))
    first.replace_video_source("video-1", source, 1.0, DenseFrameExtractor())

    second = SiglipVisualIndex(tmp_path / "index", model_name="second")
    monkeypatch.setattr(second, "_text_vector", lambda _query: np.array([1.0], dtype=np.float32))

    assert second.search("query", video_ids=["video-1"]) == []


def test_visual_index_keeps_active_generation_after_partial_replacement(tmp_path, monkeypatch) -> None:
    index = SiglipVisualIndex(tmp_path / "index", model_name="test")
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(
        index,
        "_image_vectors",
        lambda _paths: np.array([[1.0]], dtype=np.float32),
    )
    first_generation = index.replace_video_source(
        "video-1", source, 1.0, DenseFrameExtractor()
    )

    monkeypatch.setattr(
        visual_index_module,
        "atomic_write_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    with pytest.raises(OSError, match="disk full"):
        index.replace_video_source("video-1", source, 1.0, DenseFrameExtractor())

    assert index.active_generation_id("video-1") == first_generation
    monkeypatch.setattr(
        index,
        "_text_vectors",
        lambda _query: np.array([[1.0]], dtype=np.float32),
    )
    assert index.search("query", video_ids=["video-1"])


def test_visual_index_requires_identity_marker_even_for_default_model(tmp_path, monkeypatch) -> None:
    index = SiglipVisualIndex(
        tmp_path / "index",
        model_name="google/siglip2-base-patch16-224",
    )
    video_dir = tmp_path / "index" / "video-1"
    video_dir.mkdir(parents=True)
    np.save(video_dir / "vectors.npy", np.array([[1.0]], dtype=np.float32))
    (video_dir / "metadata.json").write_text(
        '[{"segment_id":"scene","start":0,"end":1,"thumbnail_path":null}]',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        index,
        "_text_vectors",
        lambda _query: np.array([[1.0]], dtype=np.float32),
    )

    assert index.search("query", video_ids=["video-1"]) == []


def test_visual_index_revision_is_part_of_persisted_identity(tmp_path, monkeypatch) -> None:
    first = SiglipVisualIndex(
        tmp_path / "index",
        model_name="organization/model",
        model_revision="a" * 40,
    )
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(first, "_image_vectors", lambda _paths: np.array([[1.0]], dtype=np.float32))
    first.replace_video_source("video-1", source, 1.0, DenseFrameExtractor())

    manifest_file = active_generation_path(tmp_path, first) / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["specification"]["model_revision"] == "a" * 40

    second = SiglipVisualIndex(
        tmp_path / "index",
        model_name="organization/model",
        model_revision="b" * 40,
    )
    monkeypatch.setattr(second, "_text_vector", lambda _query: np.array([1.0], dtype=np.float32))

    assert second.search("query", video_ids=["video-1"]) == []


def test_visual_probability_is_stable_for_large_logits(tmp_path) -> None:
    index = SiglipVisualIndex(tmp_path, model_name="test")
    index._logit_scale = 100
    index._logit_bias = 0

    assert index._probability(1) == pytest.approx(1)
    assert index._probability(-1) == pytest.approx(0)
    assert index._rank_score(-1) == pytest.approx(0)
    assert index._rank_score(0) == pytest.approx(0.5)
    assert index._rank_score(1) == pytest.approx(1)
    assert index._rank_score(0.12) > index._rank_score(0.08)
    with pytest.raises(ValueError, match="non-finite"):
        index._rank_score(float("nan"))


def test_visual_index_drops_corrupt_non_finite_vectors(tmp_path, monkeypatch) -> None:
    index = SiglipVisualIndex(tmp_path / "index", model_name="test")
    video_dir = tmp_path / "index" / "video-1"
    video_dir.mkdir(parents=True)
    np.save(
        video_dir / "vectors.npy",
        np.array([[float("nan"), 0.0]], dtype=np.float32),
        allow_pickle=False,
    )
    (video_dir / "metadata.json").write_text(
        json.dumps(
            [
                {
                    "segment_id": "corrupt",
                    "start": 1.0,
                    "end": 2.0,
                    "thumbnail_path": None,
                }
            ]
        ),
        encoding="utf-8",
    )
    (video_dir / "model.txt").write_text("test", encoding="utf-8")
    monkeypatch.setattr(
        index,
        "_text_vectors",
        lambda _query: np.array([[1.0, 0.0]], dtype=np.float32),
    )

    assert index.search("query", video_ids=["video-1"]) == []


def test_visual_search_before_indexing_returns_empty(tmp_path, monkeypatch) -> None:
    index = SiglipVisualIndex(tmp_path / "missing", model_name="test")
    monkeypatch.setattr(index, "_text_vector", lambda _query: np.array([1.0], dtype=np.float32))

    assert index.search("query") == []


def test_visual_index_checks_the_configured_model_revision(tmp_path, monkeypatch) -> None:
    revision = "b" * 40
    calls: list[tuple[str, str, str | None]] = []
    import huggingface_hub

    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))

    def cached(model: str, filename: str, *, revision: str | None = None) -> str:
        calls.append((model, filename, revision))
        return str(tmp_path / filename)

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", cached)
    index = SiglipVisualIndex(
        tmp_path / "index",
        model_name="organization/model",
        model_revision=revision,
    )

    assert index.status().state.value == "ready"
    assert calls == [
        ("organization/model", "config.json", revision),
        ("organization/model", "model.safetensors", revision),
    ]


def test_visual_index_reports_legacy_vectors_as_requiring_reindex(tmp_path, monkeypatch) -> None:
    revision = "b" * 40
    video_dir = tmp_path / "index" / "video-1"
    video_dir.mkdir(parents=True)
    (video_dir / "vectors.npy").write_bytes(b"legacy")
    (video_dir / "metadata.json").write_text("[]", encoding="utf-8")
    (video_dir / "model.txt").write_text("organization/model", encoding="utf-8")

    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "try_to_load_from_cache",
        lambda *_args, **_kwargs: str(tmp_path / "cached"),
    )
    index = SiglipVisualIndex(
        tmp_path / "index",
        model_name="organization/model",
        model_revision=revision,
    )

    status = index.status()

    assert status.state.value == "needs_configuration"
    assert "index-visual" in status.detail
    assert index.status(check_index=False).state.value == "ready"


def test_basketball_three_pointer_query_gets_english_temporal_prompts() -> None:
    prompts = visual_prompt_variants("игрок забивает трехочковый")

    assert prompts[0] == "игрок забивает трехочковый"
    assert "basketball player makes a three-point shot from behind the arc" in prompts
    assert "the basketball goes through the hoop after a three-point shot" in prompts


@pytest.mark.parametrize(
    ("query", "expected_prompt"),
    [
        (
            "игрок забивает двухочковый",
            "basketball player makes a two-point shot from inside the three-point line",
        ),
        (
            "игрок реализует штрафной",
            "basketball player makes a free throw from the free throw line",
        ),
    ],
)
def test_other_made_basketball_queries_get_type_specific_prompts(
    query: str,
    expected_prompt: str,
) -> None:
    assert expected_prompt in visual_prompt_variants(query)


def test_jersey_number_query_gets_player_identity_prompt() -> None:
    prompts = visual_prompt_variants("найди игрока под номером 15")

    assert "basketball player wearing jersey number 15" in prompts


def test_unrelated_query_does_not_get_basketball_prompts() -> None:
    assert visual_prompt_variants("человек открывает дверь") == [
        "человек открывает дверь"
    ]


class DenseFrameExtractor:
    def extract_frames(
        self,
        _source,
        destination,
        start,
        end,
        *,
        step,
        max_width=640,
    ):
        destination.mkdir(parents=True, exist_ok=True)
        frames = []
        timestamp = start
        index = 0
        while timestamp < end:
            path = destination / f"sample-{index:04d}.jpg"
            path.write_bytes(b"frame")
            frames.append(SampledFrame(timestamp, path))
            timestamp += step
            index += 1
        return frames


def test_dense_visual_index_persists_fixed_step_frames(tmp_path, monkeypatch) -> None:
    index = SiglipVisualIndex(
        tmp_path / "index", model_name="test", batch_size=2, sample_step=2.0
    )
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(
        index,
        "_image_vectors",
        lambda paths: np.eye(len(paths), dtype=np.float32),
    )

    index.replace_video_source(
        "video-1",
        source,
        5.0,
        DenseFrameExtractor(),
    )

    generation = active_generation_path(tmp_path, index)
    metadata = json.loads(
        (generation / "metadata.json").read_text()
    )
    assert [item["start"] for item in metadata] == [0.0, 2.0, 4.0]
    assert [item["end"] for item in metadata] == [2.0, 4.0, 5.0]
    assert all(item["segment_id"].startswith("dense-") for item in metadata)


def test_made_three_point_search_returns_chronological_event_window(
    tmp_path,
    monkeypatch,
) -> None:
    index = SiglipVisualIndex(tmp_path / "index", model_name="test", batch_size=4)
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    release_prompts, outcome_prompts, followup_prompts = (
        made_three_point_prompt_stages()
    )
    frame_vectors = np.zeros((12, 3), dtype=np.float32)
    frame_vectors[1] = [1.0, 0.0, 0.0]
    frame_vectors[3] = [0.0, 1.0, 0.0]
    frame_vectors[5] = [0.0, 0.0, 1.0]
    frame_vectors[10] = [0.0, 1.0, 0.0]
    monkeypatch.setattr(index, "_image_vectors", lambda _paths: frame_vectors)

    def text_vector(prompt: str) -> np.ndarray:
        if prompt in release_prompts:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if prompt in outcome_prompts:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if prompt in followup_prompts:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(index, "_text_vector", text_vector)
    index.replace_video_source(
        "video-1",
        source,
        12.0,
        DenseFrameExtractor(),
    )

    hits = index.search(
        "игрок забивает трехочковый",
        video_ids=["video-1"],
        limit=5,
    )

    assert len(hits) == 1
    assert hits[0].start == 1.0
    assert hits[0].end == 4.0
    assert hits[0].metadata["source"] == "siglip2-temporal-event"
    assert hits[0].metadata["release_timestamp"] == 1.0
    assert hits[0].metadata["outcome_timestamp"] == 3.0
    assert hits[0].metadata["followup_timestamp"] == 5.0


@pytest.mark.parametrize(
    ("query", "event_type", "release_cue", "stage_order"),
    [
        (
            "игрок забивает трехочковый",
            "made_three_point",
            "outside_arc",
            ["release", "outcome", "followup"],
        ),
        (
            "игрок забивает двухочковый",
            "made_two_point",
            "inside_arc",
            ["release", "contrast", "outcome"],
        ),
        (
            "игрок реализует штрафной",
            "made_free_throw",
            "free_throw_line",
            ["release", "outcome", "plus", "reset", "miss", "transition"],
        ),
    ],
)
def test_made_basketball_search_exposes_candidate_stage_evidence(
    tmp_path,
    monkeypatch,
    query: str,
    event_type: str,
    release_cue: str,
    stage_order: list[str],
) -> None:
    index = SiglipVisualIndex(tmp_path / "index", model_name="test", batch_size=4)
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    specification = classify_basketball_event_query(query)
    assert specification is not None
    frame_vectors = np.zeros((12, 3), dtype=np.float32)
    frame_vectors[1] = [1.0, 0.0, 0.0]
    frame_vectors[3] = [0.0, 1.0, 0.0]
    frame_vectors[5] = [0.0, 0.0, 1.0]
    monkeypatch.setattr(index, "_image_vectors", lambda _paths: frame_vectors)

    def text_vector(prompt: str) -> np.ndarray:
        if prompt in specification.release_prompts:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if prompt in specification.outcome_prompts:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if prompt in specification.followup_prompts:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return np.zeros(3, dtype=np.float32)

    monkeypatch.setattr(index, "_text_vector", text_vector)
    index.replace_video_source(
        "video-1",
        source,
        12.0,
        DenseFrameExtractor(),
    )

    hits = index.search(query, video_ids=["video-1"], limit=5)

    assert len(hits) == 1
    assert hits[0].metadata["event_type"] == event_type
    assert hits[0].metadata["event_candidate"] is True
    assert hits[0].metadata["requires_ball_through_hoop"] is True
    assert hits[0].metadata["release_cue"] == release_cue
    assert hits[0].metadata["outcome_cue"] == "ball_through_hoop"
    assert hits[0].metadata["stage_order"] == stage_order
    if event_type == "made_free_throw":
        assert 0 <= hits[0].metadata["reset_score"] <= 1
