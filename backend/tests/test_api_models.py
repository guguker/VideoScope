import pytest
from pydantic import ValidationError

from videoscope.api_models import (
    ClipSelectionRequest,
    EvaluationCaseRequest,
    EvaluationRunRequest,
    SearchRequest,
)


@pytest.mark.parametrize("video_ids", [[], [""], ["../secret"], ["video-1", "video-1"]])
def test_search_scope_requires_unique_safe_video_ids(video_ids: list[str]) -> None:
    with pytest.raises(ValidationError):
        SearchRequest(query="moment", video_ids=video_ids)


def test_search_request_rejects_coerced_or_blank_values() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(query="moment", limit="3")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        SearchRequest(query="   ")


@pytest.mark.parametrize(
    "payload",
    [
        {"video_id": "../secret", "start": 1.0, "end": 2.0},
        {"video_id": "video-1", "start": 2.0, "end": 2.0},
        {"video_id": "video-1", "start": 2.0, "end": 1.0},
        {"video_id": "video-1", "start": 1.0, "end": float("inf")},
    ],
)
def test_clip_selection_requires_a_finite_ordered_interval(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ClipSelectionRequest(**payload)


def test_evaluation_case_uses_the_same_video_and_interval_boundaries() -> None:
    with pytest.raises(ValidationError):
        EvaluationCaseRequest(
            id="case-1",
            query="moment",
            video_id="../secret",
            start=1,
            end=2,
        )
    with pytest.raises(ValidationError):
        EvaluationCaseRequest(
            id="case-1",
            query="moment",
            video_id="video-1",
            start=2,
            end=1,
        )


def test_evaluation_variants_must_be_unique() -> None:
    with pytest.raises(ValidationError):
        EvaluationRunRequest(variants=["auto", "auto"])
