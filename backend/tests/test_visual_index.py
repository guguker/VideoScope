import json

import numpy as np
import pytest

from videoscope.repository import SegmentRecord
from videoscope.search.visual_index import SiglipVisualIndex


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
    assert index._rank_score(0.05) == pytest.approx(0.5)
    assert index._rank_score(0.12) > index._rank_score(0.08)


def test_visual_search_before_indexing_returns_empty(tmp_path, monkeypatch) -> None:
    index = SiglipVisualIndex(tmp_path / "missing", model_name="test")
    monkeypatch.setattr(index, "_text_vector", lambda _query: np.array([1.0], dtype=np.float32))

    assert index.search("query") == []
