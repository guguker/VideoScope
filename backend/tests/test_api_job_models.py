from __future__ import annotations

import pytest
from pydantic import ValidationError

from videoscope.api_models import JobResponse, JobSummaryResponse, VideoResponse


def _job_values() -> dict[str, object]:
    return {
        "job_id": "job-1",
        "intent": "reindex",
        "state": "running",
        "progress": 0.4,
        "stage": "speech",
        "attempt": 2,
        "cancel_requested_at": None,
        "error_code": None,
        "created_at": "2026-08-21T09:00:00+00:00",
        "started_at": "2026-08-21T09:00:01+00:00",
        "finished_at": None,
        "updated_at": "2026-08-21T09:00:02+00:00",
    }


def test_job_response_is_public_bounded_and_contains_no_execution_secret() -> None:
    response = JobResponse(
        **_job_values(),
        video_id="video-1",
        retry_of_job_id="job-0",
    )

    assert response.state == "running"
    assert response.video_id == "video-1"
    assert set(response.model_dump()) == {
        "job_id",
        "intent",
        "state",
        "progress",
        "stage",
        "attempt",
        "cancel_requested_at",
        "error_code",
        "created_at",
        "started_at",
        "finished_at",
        "updated_at",
        "video_id",
        "retry_of_job_id",
    }

    with pytest.raises(ValidationError, match="execution_token"):
        JobResponse(
            **_job_values(),
            video_id="video-1",
            retry_of_job_id=None,
            execution_token="must-never-be-public",  # type: ignore[call-arg]
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("job_id", "../job"),
        ("state", "processing"),
        ("progress", float("nan")),
        ("progress", 1.01),
        ("stage", "x" * 65),
        ("attempt", 0),
        ("error_code", "raw exception: /private/path"),
    ],
)
def test_job_summary_rejects_unbounded_or_internal_values(field, value) -> None:  # type: ignore[no-untyped-def]
    values = _job_values()
    values[field] = value
    with pytest.raises(ValidationError):
        JobSummaryResponse(**values)


def test_video_response_carries_the_latest_job_without_changing_video_status() -> None:
    job = JobSummaryResponse(**_job_values())
    video = VideoResponse(
        id="video-1",
        original_name="match.mp4",
        display_name=None,
        size_bytes=1024,
        status="ready",
        progress=1.0,
        stage="ready",
        duration=42.0,
        width=1920,
        height=1080,
        fps=25.0,
        error=None,
        created_at="2026-08-21T08:00:00+00:00",
        updated_at="2026-08-21T09:00:02+00:00",
        media_url="/api/videos/video-1/media",
        thumbnail_url=None,
        latest_job=job,
    )

    assert video.status == "ready"
    assert video.latest_job == job
