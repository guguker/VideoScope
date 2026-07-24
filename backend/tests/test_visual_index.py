import json

import numpy as np
import pytest

from videoscope.media.ffmpeg import SampledFrame
from videoscope.repository import SegmentRecord
from videoscope.search.sports_events import classify_basketball_event_query
from videoscope.search.visual_index import (
    SiglipVisualIndex,
    made_three_point_prompt_stages,
    visual_prompt_variants,
)


def scene(tmp_path, segment_id: str, start: float, vector: np.ndarray) -> SegmentRecord:
    thumbnail = tmp_path / f"{segment_id}.jpg"
    thumbnail.write_bytes(b"frame")
    return SegmentRecord(
        id=segment_id,
        video_id="video-1",
        start=start,
        end=start + 5,
        modality="scene",
        text="scene",
        confidence=1,
        metadata={"test_vector": vector.tolist()},
        thumbnail_path=str(thumbnail),
    )


def test_visual_index_persists_frames_and_returns_closest_scene(tmp_path, monkeypatch) -> None:
    index = SiglipVisualIndex(tmp_path / "index", model_name="test", batch_size=2)
    segments = [
        scene(tmp_path, "left", 0, np.array([1.0, 0.0], dtype=np.float32)),
        scene(tmp_path, "right", 10, np.array([0.0, 1.0], dtype=np.float32)),
    ]
    monkeypatch.setattr(
        index,
        "_image_vectors",
        lambda _paths: np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    monkeypatch.setattr(index, "_text_vector", lambda _query: np.array([0.1, 0.9], dtype=np.float32))

    index.replace_video("video-1", segments)
    hits = index.search("right", video_ids=["video-1"], limit=2)

    assert hits[0].start == 10
    assert hits[0].modality == "visual"
    assert hits[0].metadata["source"] == "siglip2"
    metadata = json.loads((tmp_path / "index" / "video-1" / "metadata.json").read_text())
    assert len(metadata) == 2
    assert (tmp_path / "index" / "video-1" / "model.txt").read_text() == "test"


def test_visual_index_ignores_vectors_from_another_model(tmp_path, monkeypatch) -> None:
    first = SiglipVisualIndex(tmp_path / "index", model_name="first")
    segments = [scene(tmp_path, "scene", 0, np.array([1.0], dtype=np.float32))]
    monkeypatch.setattr(first, "_image_vectors", lambda _paths: np.array([[1.0]], dtype=np.float32))
    first.replace_video("video-1", segments)

    second = SiglipVisualIndex(tmp_path / "index", model_name="second")
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


def test_visual_search_before_indexing_returns_empty(tmp_path, monkeypatch) -> None:
    index = SiglipVisualIndex(tmp_path / "missing", model_name="test")
    monkeypatch.setattr(index, "_text_vector", lambda _query: np.array([1.0], dtype=np.float32))

    assert index.search("query") == []


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
    index = SiglipVisualIndex(tmp_path / "index", model_name="test", batch_size=2)
    monkeypatch.setattr(
        index,
        "_image_vectors",
        lambda paths: np.eye(len(paths), dtype=np.float32),
    )

    index.replace_video_source(
        "video-1",
        tmp_path / "video.mp4",
        5.0,
        DenseFrameExtractor(),
        step=2.0,
    )

    metadata = json.loads(
        (tmp_path / "index" / "video-1" / "metadata.json").read_text()
    )
    assert [item["start"] for item in metadata] == [0.0, 2.0, 4.0]
    assert [item["end"] for item in metadata] == [2.0, 4.0, 5.0]
    assert all(item["segment_id"].startswith("dense-") for item in metadata)


def test_made_three_point_search_returns_chronological_event_window(
    tmp_path,
    monkeypatch,
) -> None:
    index = SiglipVisualIndex(tmp_path / "index", model_name="test", batch_size=4)
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
        tmp_path / "video.mp4",
        12.0,
        DenseFrameExtractor(),
        step=1.0,
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
        tmp_path / "video.mp4",
        12.0,
        DenseFrameExtractor(),
        step=1.0,
    )

    hits = index.search(query, video_ids=["video-1"], limit=5)

    assert len(hits) == 1
    assert hits[0].metadata["event_type"] == event_type
    assert hits[0].metadata["event_candidate"] is True
    assert hits[0].metadata["requires_ball_through_hoop"] is True
    assert hits[0].metadata["release_cue"] == release_cue
    assert hits[0].metadata["outcome_cue"] == "ball_through_hoop"
    assert hits[0].metadata["stage_order"] == stage_order
