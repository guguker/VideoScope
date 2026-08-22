from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

from videoscope.api import create_app
from videoscope.artifacts import (
    IndexingSpecifications,
    StageKind,
    StageSpecification,
    StageState,
)
from videoscope.config import AppSettings
from videoscope.jobs import VideoIndexPlanSnapshot
from videoscope.providers.whisper import WhisperPromptSnapshot
from videoscope.repository import Repository


def _stage(kind: StageKind) -> StageSpecification:
    parameters: dict[str, object] = {}
    if kind is StageKind.SPEECH:
        parameters = {
            "effective_prompt_sha256": hashlib.sha256(b"").hexdigest(),
            "glossary_state": "not_configured",
        }
    return StageSpecification(
        kind=kind,
        schema_version=1,
        implementation_revision=f"test.{kind.value}.v1",
        parameters=parameters,
        dependencies={"videoscope": "test"},
    )


def _plan() -> VideoIndexPlanSnapshot:
    prompt = WhisperPromptSnapshot(
        effective_prompt=None,
        effective_prompt_sha256=hashlib.sha256(b"").hexdigest(),
        glossary_state="not_configured",
    )
    return VideoIndexPlanSnapshot(
        schema_version=2,
        specifications=IndexingSpecifications(
            scenes=_stage(StageKind.SCENES),
            speech=_stage(StageKind.SPEECH),
            ocr=_stage(StageKind.OCR),
            objects=_stage(StageKind.OBJECTS),
            text_vectors=_stage(StageKind.TEXT_VECTORS),
        ),
        visual_dense_specification=_stage(StageKind.VISUAL_DENSE),
        lighthouse_specification=_stage(StageKind.LIGHTHOUSE),
        whisper_prompt_snapshot=prompt,
        executor_identity="sha256:" + "a" * 64,
    )


class RecordingQueue:
    def __init__(self, *, fail: bool = False) -> None:
        self.video_ids: list[str] = []
        self.fail = fail

    def submit(self, video_id: str) -> None:
        self.video_ids.append(video_id)
        if self.fail:
            raise RuntimeError("wake unavailable")

    def close(self) -> None:
        return None


def _app(tmp_path: Path, *, queue: RecordingQueue | None = None):  # type: ignore[no-untyped-def]
    settings = AppSettings(
        _env_file=None,
        data_dir=tmp_path / "data",
        max_upload_bytes=1024,
    )
    repository = Repository(settings.database_path)
    resolved_queue = queue or RecordingQueue()
    app = create_app(
        settings=settings,
        repository=repository,
        processing_queue=resolved_queue,
        video_index_plan_factory=_plan,
    )
    return settings, repository, resolved_queue, app


def _upload(client: TestClient):  # type: ignore[no-untyped-def]
    return client.post(
        "/api/videos",
        files={"file": ("match.mp4", b"synthetic video", "video/mp4")},
    )


def _record_terminal_receipts(
    repository: Repository,
    *,
    video_id: str,
    job_id: str,
    execution_token: str,
) -> None:
    plan = _plan()
    specifications = (
        *plan.specifications.segment_specifications,
        plan.specifications.text_vectors,
        plan.visual_dense_specification,
        plan.lighthouse_specification,
    )
    for specification in specifications:
        run = repository.create_stage_run(
            video_id=video_id,
            specification=specification,
            run_id=f"{job_id}-{specification.kind.value}",
            job_id=job_id,
            execution_token=execution_token,
        )
        repository.transition_stage_run(
            run.run_id,
            StageState.NOT_CONFIGURED,
            execution_token=execution_token,
        )


def test_upload_atomically_returns_video_with_durable_job_and_location(tmp_path) -> None:
    _settings, repository, queue, app = _app(tmp_path)
    with TestClient(app) as client:
        response = _upload(client)
        listing = client.get("/api/videos")

    assert response.status_code == 202
    video = response.json()
    job = video["latest_job"]
    assert response.headers["location"] == f"/api/jobs/{job['job_id']}"
    assert job["state"] == "queued"
    assert job["intent"] == "ingest"
    assert queue.video_ids == [video["id"]]
    assert repository.get_video_index_job(job["job_id"]) is not None
    assert repository.get_video_index_job_plan(job["job_id"]) == _plan()
    assert listing.json()[0]["latest_job"] == job


def test_video_collection_loads_latest_jobs_in_bounded_bulk_batches(
    tmp_path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    _settings, repository, _queue, app = _app(tmp_path)
    with TestClient(app) as client:
        uploaded = _upload(client).json()
        template = repository.get_video(uploaded["id"])
        assert template is not None
        records = [
            replace(template, id=f"video-{index:04d}")
            for index in range(1001)
        ]
        calls: list[tuple[str, ...]] = []

        monkeypatch.setattr(repository, "list_videos", lambda: records)

        def latest(video_ids: tuple[str, ...]):
            calls.append(video_ids)
            return {}

        monkeypatch.setattr(repository, "get_latest_video_index_jobs", latest)
        response = client.get("/api/videos")

    assert response.status_code == 200
    assert len(response.json()) == 1001
    assert [len(batch) for batch in calls] == [1000, 1]


def test_get_job_never_exposes_plan_source_or_execution_token(tmp_path) -> None:
    _settings, _repository, _queue, app = _app(tmp_path)
    with TestClient(app) as client:
        uploaded = _upload(client).json()
        response = client.get(f"/api/jobs/{uploaded['latest_job']['job_id']}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["video_id"] == uploaded["id"]
    for forbidden in ("execution_token", "plan_hash", "source_sha256", "idempotency_hash"):
        assert forbidden not in payload


def test_cancel_queued_job_is_immediate_and_repeated_request_is_idempotent(tmp_path) -> None:
    _settings, _repository, _queue, app = _app(tmp_path)
    with TestClient(app) as client:
        uploaded = _upload(client).json()
        job_id = uploaded["latest_job"]["job_id"]
        cancelled = client.post(f"/api/jobs/{job_id}/cancel")
        repeated = client.post(f"/api/jobs/{job_id}/cancel")

    assert cancelled.status_code == 200
    assert cancelled.json()["state"] == "cancelled"
    assert repeated.status_code == 200
    assert repeated.json() == cancelled.json()


def test_cancel_running_job_is_cooperative_and_idempotent(tmp_path) -> None:
    _settings, repository, _queue, app = _app(tmp_path)
    with TestClient(app) as client:
        uploaded = _upload(client).json()
        job_id = uploaded["latest_job"]["job_id"]
        claimed = repository.claim_next_video_index_job(
            execution_token="worker-token-aaaaaaaaaaaaaaaa"
        )
        assert claimed is not None
        requested = client.post(f"/api/jobs/{job_id}/cancel")
        repeated = client.post(f"/api/jobs/{job_id}/cancel")

    assert requested.status_code == 202
    assert requested.json()["state"] == "running"
    assert requested.json()["cancel_requested_at"] is not None
    assert repeated.status_code == 202
    assert repeated.json() == requested.json()


def test_retry_returns_exact_plan_child_and_updates_video_latest_job(tmp_path) -> None:
    _settings, repository, queue, app = _app(tmp_path)
    with TestClient(app) as client:
        uploaded = _upload(client).json()
        parent_id = uploaded["latest_job"]["job_id"]
        assert client.post(f"/api/jobs/{parent_id}/cancel").status_code == 200
        retried = client.post(f"/api/jobs/{parent_id}/retry")
        listing = client.get("/api/videos")

    assert retried.status_code == 202
    child = retried.json()
    assert retried.headers["location"] == f"/api/jobs/{child['job_id']}"
    assert child["retry_of_job_id"] == parent_id
    assert child["state"] == "queued"
    assert repository.get_video_index_job_plan(child["job_id"]) == _plan()
    assert listing.json()[0]["latest_job"]["job_id"] == child["job_id"]
    assert queue.video_ids == [uploaded["id"], uploaded["id"]]


def test_reindex_is_atomic_and_returns_updated_video_contract(tmp_path) -> None:
    _settings, repository, queue, app = _app(tmp_path)
    with TestClient(app) as client:
        uploaded = _upload(client).json()
        parent_id = uploaded["latest_job"]["job_id"]
        claimed = repository.claim_next_video_index_job(
            execution_token="worker-token-aaaaaaaaaaaaaaaa"
        )
        assert claimed is not None
        _record_terminal_receipts(
            repository,
            video_id=uploaded["id"],
            job_id=parent_id,
            execution_token=claimed.execution_token or "",
        )
        repository.complete_video_index_job(
            parent_id,
            execution_token=claimed.execution_token or "",
        )
        response = client.post(f"/api/videos/{uploaded['id']}/reindex")

    assert response.status_code == 202
    video = response.json()
    assert video["latest_job"]["intent"] == "reindex"
    assert video["latest_job"]["state"] == "queued"
    assert response.headers["location"].endswith(video["latest_job"]["job_id"])
    assert queue.video_ids == [uploaded["id"], uploaded["id"]]


def test_advisory_wake_failure_cannot_undo_or_mask_upload_commit(tmp_path) -> None:
    queue = RecordingQueue(fail=True)
    _settings, repository, _queue, app = _app(tmp_path, queue=queue)
    with TestClient(app) as client:
        response = _upload(client)

    assert response.status_code == 202
    payload = response.json()
    assert repository.get_video(payload["id"]) is not None
    assert repository.get_video_index_job(payload["latest_job"]["job_id"]) is not None


def test_unattested_plan_fails_before_upload_bytes_are_persisted(tmp_path) -> None:
    settings = AppSettings(
        _env_file=None,
        data_dir=tmp_path / "data",
        max_upload_bytes=1024,
    )
    repository = Repository(settings.database_path)

    def unavailable_plan() -> VideoIndexPlanSnapshot:
        raise ValueError("secret local attestation detail")

    app = create_app(
        settings=settings,
        repository=repository,
        processing_queue=RecordingQueue(),
        video_index_plan_factory=unavailable_plan,
    )
    with TestClient(app) as client:
        response = _upload(client)

    assert response.status_code == 503
    assert "secret" not in response.text
    assert list(settings.media_dir.iterdir()) == []
    assert repository.list_videos() == []


def test_unknown_or_invalid_job_ids_are_not_distinguishable(tmp_path) -> None:
    _settings, _repository, _queue, app = _app(tmp_path)
    with TestClient(app) as client:
        unknown = client.get("/api/jobs/missing-job")
        invalid = client.get("/api/jobs/bad!job")
    assert unknown.status_code == invalid.status_code == 404
