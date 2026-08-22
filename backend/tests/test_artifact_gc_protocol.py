from __future__ import annotations

from datetime import UTC, datetime, timedelta
import sqlite3
from threading import Event, Thread

import pytest

import videoscope.repository as repository_module
from videoscope.artifacts import (
    ArtifactGCOutcome,
    StageKind,
    StageSpecification,
    StageState,
    TextVectorBuildReceipt,
    TextVectorIndexSpecification,
)
from videoscope.repository import LATEST_SCHEMA_VERSION, Repository


def _stage_specification(kind: StageKind) -> StageSpecification:
    return StageSpecification(
        kind=kind,
        schema_version=1,
        implementation_revision=f"tests.{kind.value}.v1",
        parameters={"contract": f"{kind.value}-v1"},
        model_identity=f"tests/{kind.value}@v1",
    )


def _semantic_specifications() -> tuple[StageSpecification, ...]:
    return tuple(
        _stage_specification(kind)
        for kind in (StageKind.SPEECH, StageKind.OCR, StageKind.OBJECTS)
    )


def _text_specification(
    semantic: tuple[StageSpecification, ...],
) -> StageSpecification:
    by_kind = {item.kind: item for item in semantic}
    return StageSpecification(
        kind=StageKind.TEXT_VECTORS,
        schema_version=1,
        implementation_revision="tests.text-vectors.v1",
        parameters={"semantic_modalities": ["speech", "ocr", "objects"]},
        model_identity="tests/embedding@v1",
        dependencies={
            "speech_specification": by_kind[StageKind.SPEECH].specification_hash,
            "ocr_specification": by_kind[StageKind.OCR].specification_hash,
            "objects_specification": by_kind[StageKind.OBJECTS].specification_hash,
        },
    )


def _index_specification() -> TextVectorIndexSpecification:
    return TextVectorIndexSpecification(
        embedding_identity="tests/hash-embedding@v1",
        dimensions=8,
    )


def _repository(tmp_path, *, video_ids: tuple[str, ...] = ("video-1",)) -> Repository:  # type: ignore[no-untyped-def]
    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()
    for video_id in video_ids:
        repository.create_video_with_asset(
            video_id=video_id,
            original_name=f"{video_id}.mp4",
            stored_name=f"{video_id}.mp4",
            media_path=str(tmp_path / f"{video_id}.mp4"),
            size_bytes=1024,
            source_sha256=("a" if video_id == "video-1" else "b") * 64,
        )
    return repository


def _start_text_run(
    repository: Repository,
    specification: StageSpecification,
    *,
    video_id: str,
    run_id: str,
) -> None:
    repository.create_stage_run(
        video_id=video_id,
        specification=specification,
        run_id=run_id,
    )
    repository.transition_stage_run(run_id, StageState.RUNNING)


def _reserve(
    repository: Repository,
    *,
    video_id: str = "video-1",
    run_id: str,
    generation_id: str,
    lease_seconds: int = 900,
):  # type: ignore[no-untyped-def]
    semantic = _semantic_specifications()
    text = _text_specification(semantic)
    _start_text_run(repository, text, video_id=video_id, run_id=run_id)
    return repository.reserve_text_vector_generation(
        run_id,
        index_specification=_index_specification(),
        semantic_specifications=semantic,
        generation_id=generation_id,
        lease_seconds=lease_seconds,
    )


def _receipt(plan) -> TextVectorBuildReceipt:  # type: ignore[no-untyped-def]
    return TextVectorBuildReceipt(
        generation_id=plan.generation_id,
        index_specification_hash=plan.index_specification.specification_hash,
        point_count=len(plan.points),
        point_manifest_sha256=plan.point_manifest_sha256,
        vector_manifest_sha256="f" * 64,
    )


def _pending_gc_job(
    repository: Repository,
    *,
    suffix: str = "1",
    video_id: str = "video-1",
):  # type: ignore[no-untyped-def]
    plan = _reserve(
        repository,
        video_id=video_id,
        run_id=f"text-run-failed-{suffix}",
        generation_id=f"text-generation-failed-{suffix}",
    )
    repository.fail_text_vector_build(
        f"text-run-failed-{suffix}",
        error_code="text_vector_index_failed",
    )
    jobs = repository.list_pending_artifact_gc_jobs(limit=10)
    return plan, next(job for job in jobs if job.generation_id == plan.generation_id)


def _make_job_ready(repository: Repository, job_id: str) -> None:
    with repository._connect() as connection:
        connection.execute(
            """
            UPDATE artifact_gc_jobs
            SET available_at = created_at
            WHERE job_id = ?
            """,
            (job_id,),
        )


def _expire_job_lease(repository: Repository, job_id: str) -> None:
    with repository._connect() as connection:
        row = connection.execute(
            """
            SELECT attempt, updated_at FROM artifact_gc_jobs
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()
        assert row is not None
        expired_at = (
            datetime.fromisoformat(row["updated_at"]) + timedelta(microseconds=1)
        ).isoformat()
        connection.execute(
            "UPDATE artifact_gc_jobs SET lease_expires_at = ? WHERE job_id = ?",
            (expired_at, job_id),
        )
        connection.execute(
            """
            UPDATE artifact_gc_attempts SET lease_expires_at = ?
            WHERE job_id = ? AND attempt = ?
            """,
            (expired_at, job_id, row["attempt"]),
        )


def _advance_clock_when_write_lock_is_acquired(
    repository: Repository,
    monkeypatch,
    *,
    before: datetime,
    after: datetime,
) -> None:  # type: ignore[no-untyped-def]
    """Deterministically model time passing while BEGIN IMMEDIATE is blocked."""

    clock = {"now": before}
    original_connect = repository._connect

    class AdvancingConnection:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self.connection = connection

        def __enter__(self):  # type: ignore[no-untyped-def]
            self.connection.__enter__()
            return self

        def __exit__(self, exc_type, exc_value, traceback):  # type: ignore[no-untyped-def]
            return self.connection.__exit__(exc_type, exc_value, traceback)

        def execute(self, sql: str, parameters=()):  # type: ignore[no-untyped-def]
            if sql.strip().upper() == "BEGIN IMMEDIATE":
                clock["now"] = after
            return self.connection.execute(sql, parameters)

        def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
            return getattr(self.connection, name)

    monkeypatch.setattr(
        repository_module,
        "_now",
        lambda: clock["now"].isoformat(),
    )
    monkeypatch.setattr(
        repository,
        "_connect",
        lambda: AdvancingConnection(original_connect()),
    )


def test_schema_v8_preserves_v6_gc_rows_and_adds_recoverable_audit(tmp_path) -> None:
    database_path = tmp_path / "videoscope.sqlite3"
    repository = Repository(database_path)
    index = _index_specification()
    created_at = datetime.now(UTC).isoformat()
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("BEGIN IMMEDIATE")
        for version, migration in repository_module._SCHEMA_MIGRATIONS[:6]:
            migration(connection)
            connection.execute(f"PRAGMA user_version = {version}")
        connection.execute(
            """
            INSERT INTO text_vector_index_specifications (
                specification_hash, canonical_json, collection_name, created_at
            ) VALUES (?, ?, ?, ?)
            """,
            (index.specification_hash, index.canonical_json, index.collection_name, created_at),
        )
        connection.execute(
            """
            INSERT INTO artifact_gc_jobs (
                job_id, artifact_kind, generation_id, index_specification_hash,
                collection_name, reason, state, attempt, created_at, updated_at
            ) VALUES (?, 'text_vectors', ?, ?, ?, ?, 'running', 2, ?, ?)
            """,
            (
                "legacy-gc-job",
                "legacy-generation",
                index.specification_hash,
                index.collection_name,
                "text_vector_build_interrupted",
                created_at,
                created_at,
            ),
        )
        connection.commit()

    repository.initialize()
    repository.initialize()

    assert repository.schema_version() == LATEST_SCHEMA_VERSION == 9
    jobs = repository.list_pending_artifact_gc_jobs(limit=10)
    assert [(job.job_id, job.state, job.attempt) for job in jobs] == [
        ("legacy-gc-job", "pending", 2)
    ]
    assert jobs[0].backoff_level == 2
    with repository._connect() as connection:
        tombstone = connection.execute(
            """
            SELECT lifecycle_state
            FROM text_vector_generation_tombstones
            WHERE generation_id = 'legacy-generation'
            """
        ).fetchone()
        assert tombstone is not None and tombstone[0] == "gc_pending"
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.parametrize("persisted_version", [7, 8])
def test_schema_v8_marker_validation_rejects_a_missing_inherited_safety_trigger(
    tmp_path,
    persisted_version: int,
) -> None:
    repository = _repository(tmp_path)
    with repository._connect() as connection:
        connection.execute("DROP TRIGGER artifact_gc_jobs_reject_committed_insert")
        connection.execute(f"PRAGMA user_version = {persisted_version}")

    with pytest.raises(ValueError, match="migration is incomplete or corrupt"):
        repository.initialize()

    assert repository.schema_version() == persisted_version


def test_schema_v8_upgrades_legacy_immutable_audit_to_bounded_recent_window(
    tmp_path,
) -> None:
    repository = _repository(tmp_path)
    _plan, pending = _pending_gc_job(repository)
    timestamp = datetime.now(UTC)
    claimed_at = timestamp.isoformat()
    lease_expires_at = (timestamp + timedelta(seconds=1)).isoformat()
    finished_at = (timestamp + timedelta(seconds=2)).isoformat()
    with repository._connect() as connection:
        attempt_fence_sql = connection.execute(
            """
            SELECT sql FROM sqlite_master
            WHERE type = 'trigger' AND name = 'artifact_gc_jobs_attempt_fence'
            """
        ).fetchone()[0]
        connection.execute("DROP TRIGGER artifact_gc_jobs_attempt_fence")
        connection.execute(
            "UPDATE artifact_gc_jobs SET attempt = 300 WHERE job_id = ?",
            (pending.job_id,),
        )
        connection.executemany(
            """
            INSERT INTO artifact_gc_attempts (
                job_id, attempt, worker_id, lease_token,
                claimed_at, lease_expires_at, finished_at, outcome, error_code
            ) VALUES (?, ?, 'legacy-worker', ?, ?, ?, ?,
                'transient_failure', 'artifact_gc_storage_unavailable')
            """,
            (
                (
                    pending.job_id,
                    attempt,
                    f"legacy-token-{attempt}",
                    claimed_at,
                    lease_expires_at,
                    finished_at,
                )
                for attempt in range(1, 301)
            ),
        )
        connection.execute(attempt_fence_sql)
        connection.execute("DROP TRIGGER artifact_gc_attempts_bounded_delete")
        connection.execute(
            """
            CREATE TRIGGER artifact_gc_attempts_immutable_delete
            BEFORE DELETE ON artifact_gc_attempts
            BEGIN
                SELECT RAISE(ABORT, 'artifact GC attempt audit is immutable');
            END
            """
        )

    repository.initialize()
    repository.initialize()

    job = repository.list_pending_artifact_gc_jobs(limit=1)[0]
    assert job.job_id == pending.job_id
    assert job.attempt == 300
    audit = repository.list_artifact_gc_attempts(pending.job_id, limit=1000)
    assert len(audit) == 256
    assert [item.attempt for item in audit] == list(range(45, 301))
    with repository._connect() as connection:
        triggers = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            ).fetchall()
        }
        assert "artifact_gc_attempts_bounded_delete" in triggers
        assert "artifact_gc_attempts_immutable_delete" not in triggers
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_schema_v8_revives_only_exact_legacy_transient_failures(tmp_path) -> None:
    database_path = tmp_path / "videoscope.sqlite3"
    index = _index_specification()
    timestamp = datetime.now(UTC).isoformat()
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("BEGIN IMMEDIATE")
        for version, migration in repository_module._SCHEMA_MIGRATIONS[:7]:
            migration(connection)
            connection.execute(f"PRAGMA user_version = {version}")
        connection.execute(
            """
            INSERT INTO text_vector_index_specifications (
                specification_hash, canonical_json, collection_name, created_at
            ) VALUES (?, ?, ?, ?)
            """,
            (index.specification_hash, index.canonical_json, index.collection_name, timestamp),
        )
        for generation_id in ("legacy-transient", "legacy-permanent"):
            connection.execute(
                """
                INSERT INTO text_vector_generation_tombstones (
                    generation_id, index_specification_hash, collection_name,
                    lifecycle_state, created_at, updated_at
                ) VALUES (?, ?, ?, 'gc_pending', ?, ?)
                """,
                (
                    generation_id,
                    index.specification_hash,
                    index.collection_name,
                    timestamp,
                    timestamp,
                ),
            )
        for job_id, generation_id, error_code in (
            (
                "legacy-transient-job",
                "legacy-transient",
                "artifact_gc_storage_unavailable",
            ),
            (
                "legacy-permanent-job",
                "legacy-permanent",
                "artifact_gc_contract_invalid",
            ),
        ):
            connection.execute(
                """
                INSERT INTO artifact_gc_jobs (
                    job_id, artifact_kind, generation_id, index_specification_hash,
                    collection_name, reason, state, attempt, max_attempts,
                    available_at, worker_id, lease_token, lease_expires_at,
                    error_code, created_at, updated_at
                ) VALUES (?, 'text_vectors', ?, ?, ?, 'legacy_failure',
                    'failed', 5, 5, ?, NULL, NULL, NULL, ?, ?, ?)
                """,
                (
                    job_id,
                    generation_id,
                    index.specification_hash,
                    index.collection_name,
                    timestamp,
                    error_code,
                    timestamp,
                    timestamp,
                ),
            )
            connection.execute(
                """
                UPDATE text_vector_generation_tombstones
                SET lifecycle_state = 'gc_failed'
                WHERE generation_id = ?
                """,
                (generation_id,),
            )
        connection.commit()

    repository = Repository(database_path)
    repository.initialize()
    repository.initialize()

    pending = repository.list_pending_artifact_gc_jobs(limit=10)
    assert [(job.job_id, job.attempt, job.backoff_level) for job in pending] == [
        ("legacy-transient-job", 5, 5)
    ]
    with repository._connect() as connection:
        permanent = connection.execute(
            """
            SELECT state, error_code FROM artifact_gc_jobs
            WHERE job_id = 'legacy-permanent-job'
            """
        ).fetchone()
        lifecycles = dict(
            connection.execute(
                """
                SELECT generation_id, lifecycle_state
                FROM text_vector_generation_tombstones
                WHERE generation_id LIKE 'legacy-%'
                """
            ).fetchall()
        )
    assert tuple(permanent) == ("failed", "artifact_gc_contract_invalid")
    assert lifecycles == {
        "legacy-permanent": "gc_failed",
        "legacy-transient": "gc_pending",
    }


def test_generation_id_is_never_reusable_after_build_failure_or_gc(tmp_path) -> None:
    repository = _repository(tmp_path)
    _pending_gc_job(repository)
    semantic = _semantic_specifications()
    text = _text_specification(semantic)
    _start_text_run(repository, text, video_id="video-1", run_id="text-run-reuse")

    with pytest.raises(ValueError, match="identity already exists"):
        repository.reserve_text_vector_generation(
            "text-run-reuse",
            index_specification=_index_specification(),
            semantic_specifications=semantic,
            generation_id="text-generation-failed-1",
        )


def test_live_build_and_generation_tombstone_cannot_be_silently_deleted(tmp_path) -> None:
    repository = _repository(tmp_path)
    plan = _reserve(
        repository,
        run_id="text-run-live",
        generation_id="text-generation-live",
    )

    with repository._connect() as connection, pytest.raises(
        sqlite3.IntegrityError, match="must be terminal"
    ):
        connection.execute(
            "DELETE FROM text_vector_builds WHERE generation_id = ?",
            (plan.generation_id,),
        )

    assert repository.abandon_text_vector_builds(limit=1) == (plan.generation_id,)
    with repository._connect() as connection, pytest.raises(
        sqlite3.IntegrityError, match="tombstone is immutable"
    ):
        connection.execute(
            "DELETE FROM text_vector_generation_tombstones WHERE generation_id = ?",
            (plan.generation_id,),
        )


@pytest.mark.parametrize("active", [True, False])
def test_gc_enqueue_rejects_every_committed_generation(tmp_path, active: bool) -> None:
    repository = _repository(tmp_path)
    first_plan = _reserve(
        repository,
        run_id="text-run-1",
        generation_id="text-generation-1",
    )
    repository.commit_text_vector_generation("text-run-1", receipt=_receipt(first_plan))
    target = first_plan
    if not active:
        second_plan = _reserve(
            repository,
            run_id="text-run-2",
            generation_id="text-generation-2",
        )
        repository.commit_text_vector_generation("text-run-2", receipt=_receipt(second_plan))

    with repository._connect() as connection, pytest.raises(
        ValueError, match="committed generation"
    ):
        connection.execute("BEGIN IMMEDIATE")
        repository._enqueue_artifact_gc(
            connection,
            generation_id=target.generation_id,
            index_specification_hash=target.index_specification.specification_hash,
            collection_name=target.index_specification.collection_name,
            reason="manual_cleanup",
            timestamp=datetime.now(UTC).isoformat(),
        )


def test_claim_heartbeat_retry_and_attempt_fencing(tmp_path) -> None:
    repository = _repository(tmp_path)
    _plan, pending = _pending_gc_job(repository)

    claimed = repository.claim_next_artifact_gc_job(
        worker_id="worker-a",
        lease_seconds=30,
    )
    assert claimed is not None
    assert claimed.job_id == pending.job_id
    assert claimed.state == "running"
    assert claimed.attempt == 1
    assert claimed.worker_id == "worker-a"
    assert claimed.lease_token is not None

    with pytest.raises(ValueError, match="claim changed or expired"):
        repository.heartbeat_artifact_gc_job(
            claimed.job_id,
            worker_id="worker-a",
            lease_token="wrong-token",
            attempt=claimed.attempt,
        )
    heartbeat = repository.heartbeat_artifact_gc_job(
        claimed.job_id,
        worker_id="worker-a",
        lease_token=claimed.lease_token,
        attempt=claimed.attempt,
        lease_seconds=30,
    )
    assert heartbeat.lease_expires_at is not None
    assert heartbeat.lease_expires_at >= claimed.lease_expires_at  # type: ignore[operator]

    retry = repository.finish_artifact_gc_job(
        claimed.job_id,
        worker_id="worker-a",
        lease_token=claimed.lease_token,
        attempt=claimed.attempt,
        outcome=ArtifactGCOutcome.TRANSIENT_FAILURE,
        error_code="qdrant_unavailable",
    )
    assert retry.state == "pending"
    assert retry.error_code == "qdrant_unavailable"
    assert retry.available_at > retry.updated_at
    assert repository.claim_next_artifact_gc_job(worker_id="worker-b") is None

    _make_job_ready(repository, claimed.job_id)
    reclaimed = repository.claim_next_artifact_gc_job(worker_id="worker-b")
    assert reclaimed is not None
    assert reclaimed.attempt == 2
    assert reclaimed.lease_token != claimed.lease_token

    with pytest.raises(ValueError, match="claim changed or expired"):
        repository.finish_artifact_gc_job(
            claimed.job_id,
            worker_id="worker-a",
            lease_token=claimed.lease_token,
            attempt=claimed.attempt,
            outcome=ArtifactGCOutcome.SUCCESS,
        )

    failed = repository.finish_artifact_gc_job(
        reclaimed.job_id,
        worker_id="worker-b",
        lease_token=reclaimed.lease_token,  # type: ignore[arg-type]
        attempt=reclaimed.attempt,
        outcome=ArtifactGCOutcome.PERMANENT_FAILURE,
        error_code="qdrant_rejected_delete",
    )
    assert failed.state == "failed"
    assert failed.error_code == "qdrant_rejected_delete"


def test_gc_claim_lease_starts_after_the_write_lock_is_acquired(
    tmp_path,
    monkeypatch,
) -> None:
    repository = _repository(tmp_path)
    _pending_gc_job(repository)
    before = datetime.now(UTC) + timedelta(minutes=1)
    after = before + timedelta(seconds=2)
    _advance_clock_when_write_lock_is_acquired(
        repository,
        monkeypatch,
        before=before,
        after=after,
    )

    claimed = repository.claim_next_artifact_gc_job(
        worker_id="worker-a",
        lease_seconds=1,
    )

    assert claimed is not None and claimed.lease_expires_at is not None
    assert datetime.fromisoformat(claimed.updated_at) == after
    assert datetime.fromisoformat(claimed.lease_expires_at) == after + timedelta(seconds=1)


@pytest.mark.parametrize("operation", ["heartbeat", "finish"])
def test_gc_lease_cannot_be_extended_or_finished_after_expiring_while_waiting_for_lock(
    tmp_path,
    monkeypatch,
    operation: str,
) -> None:
    repository = _repository(tmp_path)
    _pending_gc_job(repository)
    claimed = repository.claim_next_artifact_gc_job(
        worker_id="worker-a",
        lease_seconds=30,
    )
    assert claimed is not None
    assert claimed.lease_token is not None and claimed.lease_expires_at is not None
    expiry = datetime.fromisoformat(claimed.lease_expires_at)
    _advance_clock_when_write_lock_is_acquired(
        repository,
        monkeypatch,
        before=expiry - timedelta(seconds=1),
        after=expiry + timedelta(seconds=1),
    )

    with pytest.raises(ValueError, match="claim changed or expired"):
        if operation == "heartbeat":
            repository.heartbeat_artifact_gc_job(
                claimed.job_id,
                worker_id="worker-a",
                lease_token=claimed.lease_token,
                attempt=claimed.attempt,
                lease_seconds=30,
            )
        else:
            repository.finish_artifact_gc_job(
                claimed.job_id,
                worker_id="worker-a",
                lease_token=claimed.lease_token,
                attempt=claimed.attempt,
                outcome=ArtifactGCOutcome.SUCCESS,
            )


def test_gc_finish_checks_lease_after_real_sqlite_writer_contention(
    tmp_path,
    monkeypatch,
) -> None:
    repository = _repository(tmp_path)
    _pending_gc_job(repository)
    claimed = repository.claim_next_artifact_gc_job(
        worker_id="worker-a",
        lease_seconds=30,
    )
    assert claimed is not None
    assert claimed.lease_token is not None and claimed.lease_expires_at is not None
    expiry = datetime.fromisoformat(claimed.lease_expires_at)
    clock = {"now": expiry - timedelta(seconds=1)}
    original_connect = repository._connect
    attempting_write = Event()

    class SignalingConnection:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self.connection = connection

        def __enter__(self):  # type: ignore[no-untyped-def]
            self.connection.__enter__()
            return self

        def __exit__(self, exc_type, exc_value, traceback):  # type: ignore[no-untyped-def]
            return self.connection.__exit__(exc_type, exc_value, traceback)

        def execute(self, sql: str, parameters=()):  # type: ignore[no-untyped-def]
            if sql.strip().upper() == "BEGIN IMMEDIATE":
                attempting_write.set()
            return self.connection.execute(sql, parameters)

        def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
            return getattr(self.connection, name)

    blocker = original_connect()
    blocker.execute("BEGIN IMMEDIATE")
    monkeypatch.setattr(repository_module, "_now", lambda: clock["now"].isoformat())
    monkeypatch.setattr(
        repository,
        "_connect",
        lambda: SignalingConnection(original_connect()),
    )
    outcome: dict[str, object] = {}

    def finish() -> None:
        try:
            outcome["job"] = repository.finish_artifact_gc_job(
                claimed.job_id,
                worker_id="worker-a",
                lease_token=claimed.lease_token,  # type: ignore[arg-type]
                attempt=claimed.attempt,
                outcome=ArtifactGCOutcome.SUCCESS,
            )
        except Exception as error:  # noqa: BLE001 - asserted across thread boundary
            outcome["error"] = error

    worker = Thread(target=finish)
    try:
        worker.start()
        assert attempting_write.wait(timeout=2)
        clock["now"] = expiry + timedelta(seconds=1)
        blocker.commit()
        worker.join(timeout=2)
    finally:
        if blocker.in_transaction:
            blocker.rollback()
        blocker.close()
        worker.join(timeout=2)

    assert not worker.is_alive()
    assert isinstance(outcome.get("error"), ValueError)
    assert "claim changed or expired" in str(outcome["error"])
    assert "job" not in outcome


def test_successful_gc_is_terminal_and_preserves_tombstone(tmp_path) -> None:
    repository = _repository(tmp_path)
    _plan, pending = _pending_gc_job(repository)
    claimed = repository.claim_next_artifact_gc_job(worker_id="worker-a")
    assert claimed is not None and claimed.lease_token is not None

    complete = repository.finish_artifact_gc_job(
        claimed.job_id,
        worker_id="worker-a",
        lease_token=claimed.lease_token,
        attempt=claimed.attempt,
        outcome=ArtifactGCOutcome.SUCCESS,
    )

    assert complete.state == "complete"
    assert complete.error_code is None
    assert repository.claim_next_artifact_gc_job(worker_id="worker-a") is None
    with repository._connect() as connection:
        tombstone = connection.execute(
            """
            SELECT lifecycle_state FROM text_vector_generation_tombstones
            WHERE generation_id = ?
            """,
            (pending.generation_id,),
        ).fetchone()
    assert tombstone[0] == "gc_complete"


@pytest.mark.parametrize(
    ("outcome", "error_code"),
    [
        (ArtifactGCOutcome.SUCCESS, "must_not_be_stored"),
        (ArtifactGCOutcome.TRANSIENT_FAILURE, None),
        (ArtifactGCOutcome.PERMANENT_FAILURE, "unsafe error / includes detail"),
    ],
)
def test_finish_accepts_only_sanitized_outcomes(
    tmp_path,
    outcome: ArtifactGCOutcome,
    error_code: str | None,
) -> None:
    repository = _repository(tmp_path)
    _plan, _pending = _pending_gc_job(repository)
    claimed = repository.claim_next_artifact_gc_job(worker_id="worker-a")
    assert claimed is not None and claimed.lease_token is not None

    with pytest.raises(ValueError):
        repository.finish_artifact_gc_job(
            claimed.job_id,
            worker_id="worker-a",
            lease_token=claimed.lease_token,
            attempt=claimed.attempt,
            outcome=outcome,
            error_code=error_code,
        )


def test_expiry_and_listing_are_strictly_bounded(tmp_path) -> None:
    repository = _repository(tmp_path, video_ids=("video-1", "video-2"))
    first = _reserve(
        repository,
        video_id="video-1",
        run_id="text-run-expire-1",
        generation_id="text-generation-expire-1",
        lease_seconds=1,
    )
    second = _reserve(
        repository,
        video_id="video-2",
        run_id="text-run-expire-2",
        generation_id="text-generation-expire-2",
        lease_seconds=1,
    )
    cutoff = (
        max(
            datetime.fromisoformat(first.lease_expires_at),
            datetime.fromisoformat(second.lease_expires_at),
        )
        + timedelta(seconds=1)
    ).isoformat()

    assert len(repository.expire_text_vector_builds(before=cutoff, limit=1)) == 1
    assert len(repository.list_pending_artifact_gc_jobs(limit=1)) == 1
    assert len(repository.expire_text_vector_builds(before=cutoff, limit=1)) == 1
    assert len(repository.list_pending_artifact_gc_jobs(limit=2)) == 2

    for invalid in (0, 1001, True):
        with pytest.raises(ValueError):
            repository.expire_text_vector_builds(before=cutoff, limit=invalid)
        with pytest.raises(ValueError):
            repository.list_pending_artifact_gc_jobs(limit=invalid)


def test_exclusive_runtime_recovery_can_abandon_unexpired_builds_in_batches(
    tmp_path,
) -> None:
    repository = _repository(tmp_path, video_ids=("video-1", "video-2"))
    _reserve(
        repository,
        video_id="video-1",
        run_id="text-run-abandon-1",
        generation_id="text-generation-abandon-1",
        lease_seconds=900,
    )
    _reserve(
        repository,
        video_id="video-2",
        run_id="text-run-abandon-2",
        generation_id="text-generation-abandon-2",
        lease_seconds=900,
    )

    assert len(repository.abandon_text_vector_builds(limit=1)) == 1
    assert len(repository.abandon_text_vector_builds(limit=1)) == 1
    assert repository.abandon_text_vector_builds(limit=1) == ()
    assert {
        repository.get_stage_run("text-run-abandon-1").error_code,  # type: ignore[union-attr]
        repository.get_stage_run("text-run-abandon-2").error_code,  # type: ignore[union-attr]
    } == {"text_vector_runtime_abandoned"}
    assert len(repository.list_pending_artifact_gc_jobs(limit=10)) == 2

    for invalid in (0, 1001, True):
        with pytest.raises(ValueError):
            repository.abandon_text_vector_builds(limit=invalid)


def test_gc_lease_is_deliberately_short_and_worker_identity_is_safe(tmp_path) -> None:
    repository = _repository(tmp_path)
    _pending_gc_job(repository)

    for invalid in (0, 301, True):
        with pytest.raises(ValueError):
            repository.claim_next_artifact_gc_job(
                worker_id="worker-a",
                lease_seconds=invalid,
            )
    with pytest.raises(ValueError):
        repository.claim_next_artifact_gc_job(worker_id="../../unsafe")


def test_transient_gc_retries_indefinitely_with_capped_backoff_and_full_audit(
    tmp_path,
) -> None:
    repository = _repository(tmp_path)
    _plan, pending = _pending_gc_job(repository)
    lease_tokens: set[str] = set()

    for expected_attempt in range(1, 10):
        _make_job_ready(repository, pending.job_id)
        claimed = repository.claim_next_artifact_gc_job(worker_id="retry-worker")
        assert claimed is not None and claimed.lease_token is not None
        assert claimed.attempt == expected_attempt
        assert claimed.lease_token not in lease_tokens
        lease_tokens.add(claimed.lease_token)
        pending = repository.finish_artifact_gc_job(
            claimed.job_id,
            worker_id="retry-worker",
            lease_token=claimed.lease_token,
            attempt=claimed.attempt,
            outcome=ArtifactGCOutcome.TRANSIENT_FAILURE,
            error_code="artifact_gc_storage_unavailable",
        )
        assert pending.state == "pending"
        assert pending.backoff_level == min(expected_attempt, 7)

    assert pending.attempt == 9
    assert pending.backoff_level == 7
    assert (
        datetime.fromisoformat(pending.available_at)
        - datetime.fromisoformat(pending.updated_at)
    ) == timedelta(seconds=300)
    with repository._connect() as connection:
        lifecycle = connection.execute(
            """
            SELECT lifecycle_state FROM text_vector_generation_tombstones
            WHERE generation_id = ?
            """,
            (pending.generation_id,),
        ).fetchone()[0]
    assert lifecycle == "gc_pending"

    audit = repository.list_artifact_gc_attempts(pending.job_id, limit=20)
    assert [item.attempt for item in audit] == list(range(1, 10))
    assert {item.outcome for item in audit} == {
        ArtifactGCOutcome.TRANSIENT_FAILURE.value
    }
    assert len({item.lease_token for item in audit}) == 9
    assert all(item.error_code == "artifact_gc_storage_unavailable" for item in audit)

    _make_job_ready(repository, pending.job_id)
    tenth = repository.claim_next_artifact_gc_job(worker_id="retry-worker")
    assert tenth is not None and tenth.attempt == 10


def test_gc_audit_retains_a_bounded_recent_window_without_weakening_fencing(
    tmp_path,
) -> None:
    repository = _repository(tmp_path)
    _plan, pending = _pending_gc_job(repository)
    first_lease_token: str | None = None

    for expected_attempt in range(1, 258):
        _make_job_ready(repository, pending.job_id)
        claimed = repository.claim_next_artifact_gc_job(worker_id="retry-worker")
        assert claimed is not None and claimed.lease_token is not None
        assert claimed.attempt == expected_attempt
        if first_lease_token is None:
            first_lease_token = claimed.lease_token
        pending = repository.finish_artifact_gc_job(
            claimed.job_id,
            worker_id="retry-worker",
            lease_token=claimed.lease_token,
            attempt=claimed.attempt,
            outcome=ArtifactGCOutcome.TRANSIENT_FAILURE,
            error_code="artifact_gc_storage_unavailable",
        )

    _make_job_ready(repository, pending.job_id)
    current = repository.claim_next_artifact_gc_job(worker_id="retry-worker")
    assert current is not None and current.lease_token is not None
    assert current.attempt == 258
    assert first_lease_token is not None

    audit = repository.list_artifact_gc_attempts(current.job_id, limit=1000)
    assert len(audit) == 256
    assert [item.attempt for item in audit] == list(range(3, 259))
    assert audit[-1].outcome is None

    with pytest.raises(ValueError, match="claim changed or expired"):
        repository.finish_artifact_gc_job(
            current.job_id,
            worker_id="retry-worker",
            lease_token=first_lease_token,
            attempt=1,
            outcome=ArtifactGCOutcome.SUCCESS,
        )
    with repository._connect() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="recent or open"):
            connection.execute(
                "DELETE FROM artifact_gc_attempts WHERE job_id = ? AND attempt = 257",
                (current.job_id,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="recent or open"):
            connection.execute(
                "DELETE FROM artifact_gc_attempts WHERE job_id = ? AND attempt = 258",
                (current.job_id,),
            )


def test_permanent_gc_failure_remains_terminal_and_audited(tmp_path) -> None:
    repository = _repository(tmp_path)
    _plan, _pending = _pending_gc_job(repository)
    claimed = repository.claim_next_artifact_gc_job(worker_id="worker-a")
    assert claimed is not None and claimed.lease_token is not None

    failed = repository.finish_artifact_gc_job(
        claimed.job_id,
        worker_id="worker-a",
        lease_token=claimed.lease_token,
        attempt=claimed.attempt,
        outcome=ArtifactGCOutcome.PERMANENT_FAILURE,
        error_code="artifact_gc_contract_invalid",
    )

    assert failed.state == "failed"
    assert repository.claim_next_artifact_gc_job(worker_id="worker-b") is None
    audit = repository.list_artifact_gc_attempts(failed.job_id, limit=10)
    assert [(item.attempt, item.outcome, item.error_code) for item in audit] == [
        (1, ArtifactGCOutcome.PERMANENT_FAILURE.value, "artifact_gc_contract_invalid")
    ]
    with repository._connect() as connection, pytest.raises(
        sqlite3.IntegrityError, match="artifact GC"
    ):
        connection.execute(
            "UPDATE artifact_gc_jobs SET state = 'pending' WHERE job_id = ?",
            (failed.job_id,),
        )


def test_expired_gc_lease_is_backed_off_and_remains_recoverable(tmp_path) -> None:
    repository = _repository(tmp_path)
    _plan, pending = _pending_gc_job(repository)
    first = repository.claim_next_artifact_gc_job(worker_id="worker-a")
    assert first is not None
    _expire_job_lease(repository, pending.job_id)

    assert repository.claim_next_artifact_gc_job(worker_id="worker-b") is None
    deferred = repository.list_pending_artifact_gc_jobs(limit=10)[0]
    assert deferred.attempt == 1
    assert deferred.backoff_level == 1
    assert deferred.error_code == "artifact_gc_lease_expired"
    audit = repository.list_artifact_gc_attempts(deferred.job_id, limit=10)
    assert [(item.attempt, item.outcome) for item in audit] == [
        (1, "lease_expired")
    ]

    _make_job_ready(repository, deferred.job_id)
    second = repository.claim_next_artifact_gc_job(worker_id="worker-b")
    assert second is not None and second.attempt == 2


def test_corrupt_first_gc_row_is_quarantined_and_next_valid_row_is_claimed(
    tmp_path,
) -> None:
    repository = _repository(tmp_path, video_ids=("video-1", "video-2"))
    _plan, corrupt = _pending_gc_job(repository, suffix="1", video_id="video-1")
    _plan, valid = _pending_gc_job(repository, suffix="2", video_id="video-2")
    with repository._connect() as connection:
        connection.execute("DROP TRIGGER artifact_gc_jobs_target_immutable_update")
        connection.execute(
            "UPDATE artifact_gc_jobs SET collection_name = ? WHERE job_id = ?",
            ("wrong-collection", corrupt.job_id),
        )

    claimed = repository.claim_next_artifact_gc_job(
        worker_id="worker-a",
        scan_limit=2,
    )

    assert claimed is not None and claimed.job_id == valid.job_id
    quarantines = repository.list_artifact_gc_quarantines(limit=10)
    assert len(quarantines) == 1
    assert quarantines[0].error_code == "artifact_gc_metadata_corrupt"
    assert not hasattr(quarantines[0], "generation_id")
    with repository._connect() as connection:
        corrupt_row = connection.execute(
            "SELECT state, attempt, worker_id FROM artifact_gc_jobs WHERE job_id = ?",
            (corrupt.job_id,),
        ).fetchone()
    assert tuple(corrupt_row) == ("pending", 0, None)


def test_gc_poison_scan_is_bounded(tmp_path) -> None:
    repository = _repository(tmp_path, video_ids=("video-1", "video-2"))
    _plan, corrupt = _pending_gc_job(repository, suffix="1", video_id="video-1")
    _plan, valid = _pending_gc_job(repository, suffix="2", video_id="video-2")
    with repository._connect() as connection:
        connection.execute("DROP TRIGGER artifact_gc_jobs_target_immutable_update")
        connection.execute(
            "UPDATE artifact_gc_jobs SET collection_name = ? WHERE job_id = ?",
            ("wrong-collection", corrupt.job_id),
        )

    assert repository.claim_next_artifact_gc_job(
        worker_id="worker-a",
        scan_limit=1,
    ) is None
    claimed = repository.claim_next_artifact_gc_job(
        worker_id="worker-a",
        scan_limit=1,
    )
    assert claimed is not None and claimed.job_id == valid.job_id
    for invalid in (0, 65, True):
        with pytest.raises(ValueError):
            repository.claim_next_artifact_gc_job(
                worker_id="worker-a",
                scan_limit=invalid,
            )


def test_corrupt_running_attempt_audit_is_quarantined_without_reclaim(
    tmp_path,
) -> None:
    repository = _repository(tmp_path, video_ids=("video-1", "video-2"))
    _plan, corrupt = _pending_gc_job(repository, suffix="1", video_id="video-1")
    _plan, valid = _pending_gc_job(repository, suffix="2", video_id="video-2")
    first = repository.claim_next_artifact_gc_job(worker_id="worker-a")
    assert first is not None and first.job_id == corrupt.job_id
    _expire_job_lease(repository, corrupt.job_id)
    with repository._connect() as connection:
        connection.execute("DROP TRIGGER artifact_gc_attempts_identity_immutable_update")
        created_at = connection.execute(
            "SELECT created_at FROM artifact_gc_jobs WHERE job_id = ?",
            (corrupt.job_id,),
        ).fetchone()[0]
        early_expiry = (
            datetime.fromisoformat(created_at) + timedelta(microseconds=1)
        ).isoformat()
        connection.execute(
            """
            UPDATE artifact_gc_jobs
            SET updated_at = ?, lease_expires_at = ?
            WHERE job_id = ?
            """,
            (created_at, early_expiry, corrupt.job_id),
        )
        connection.execute(
            """
            UPDATE artifact_gc_attempts
            SET worker_id = 'wrong-worker', claimed_at = ?, lease_expires_at = ?
            WHERE job_id = ? AND attempt = ?
            """,
            (created_at, early_expiry, corrupt.job_id, first.attempt),
        )

    claimed = repository.claim_next_artifact_gc_job(
        worker_id="worker-b",
        scan_limit=2,
    )

    assert claimed is not None and claimed.job_id == valid.job_id
    assert len(repository.list_artifact_gc_quarantines(limit=10)) == 1
    with repository._connect() as connection:
        corrupt_state = connection.execute(
            "SELECT state, worker_id FROM artifact_gc_jobs WHERE job_id = ?",
            (corrupt.job_id,),
        ).fetchone()
    assert tuple(corrupt_state) == ("running", "worker-a")


def test_periodic_expiry_quarantines_poison_and_processes_valid_build(
    tmp_path,
) -> None:
    repository = _repository(tmp_path, video_ids=("video-1", "video-2"))
    corrupt = _reserve(
        repository,
        video_id="video-1",
        run_id="text-run-corrupt",
        generation_id="text-generation-corrupt",
        lease_seconds=1,
    )
    valid = _reserve(
        repository,
        video_id="video-2",
        run_id="text-run-valid",
        generation_id="text-generation-valid",
        lease_seconds=1,
    )
    with repository._connect() as connection:
        connection.execute("DROP TRIGGER text_vector_builds_identity_immutable_update")
        connection.execute(
            """
            UPDATE text_vector_builds
            SET collection_name = 'wrong-collection'
            WHERE generation_id = ?
            """,
            (corrupt.generation_id,),
        )
    cutoff = (
        max(
            datetime.fromisoformat(corrupt.lease_expires_at),
            datetime.fromisoformat(valid.lease_expires_at),
        )
        + timedelta(seconds=1)
    ).isoformat()

    report = repository.recover_expired_text_vector_builds(
        before=cutoff,
        limit=2,
    )

    assert report.examined_count == 2
    assert report.terminalized_generation_ids == (valid.generation_id,)
    assert report.quarantined_count == 1
    assert report.has_more is False
    quarantines = repository.list_quarantined_text_vector_builds(limit=10)
    assert len(quarantines) == 1
    assert quarantines[0].error_code == "text_vector_build_metadata_corrupt"
    assert not hasattr(quarantines[0], "generation_id")
    with repository._connect() as connection:
        corrupt_row = connection.execute(
            """
            SELECT recovery_state FROM text_vector_builds
            WHERE generation_id = ?
            """,
            (corrupt.generation_id,),
        ).fetchone()
        corrupt_gc = connection.execute(
            "SELECT 1 FROM artifact_gc_jobs WHERE generation_id = ?",
            (corrupt.generation_id,),
        ).fetchone()
    assert corrupt_row[0] == "quarantined"
    assert corrupt_gc is None
    assert [job.generation_id for job in repository.list_pending_artifact_gc_jobs(limit=10)] == [
        valid.generation_id
    ]


def test_exclusive_startup_recovery_skips_existing_quarantine_and_isolates_new_poison(
    tmp_path,
) -> None:
    repository = _repository(tmp_path, video_ids=("video-1", "video-2"))
    corrupt = _reserve(
        repository,
        video_id="video-1",
        run_id="text-run-startup-corrupt",
        generation_id="text-generation-startup-corrupt",
        lease_seconds=1,
    )
    valid = _reserve(
        repository,
        video_id="video-2",
        run_id="text-run-startup-valid",
        generation_id="text-generation-startup-valid",
        lease_seconds=900,
    )
    with repository._connect() as connection:
        connection.execute("DROP TRIGGER text_vector_builds_identity_immutable_update")
        connection.execute(
            """
            UPDATE text_vector_builds
            SET collection_name = 'wrong-collection'
            WHERE generation_id = ?
            """,
            (corrupt.generation_id,),
        )

    report = repository.recover_abandoned_text_vector_builds(limit=2)

    assert report.examined_count == 2
    assert report.terminalized_generation_ids == (valid.generation_id,)
    assert report.quarantined_count == 1
    assert report.has_more is False
    assert len(repository.list_quarantined_text_vector_builds(limit=10)) == 1
    assert repository.recover_abandoned_text_vector_builds(limit=2).examined_count == 0
    assert repository.abandon_text_vector_builds(limit=2) == ()


def test_periodic_expiry_report_exposes_bounded_remaining_work(tmp_path) -> None:
    repository = _repository(tmp_path, video_ids=("video-1", "video-2"))
    first = _reserve(
        repository,
        video_id="video-1",
        run_id="text-run-periodic-1",
        generation_id="text-generation-periodic-1",
        lease_seconds=1,
    )
    second = _reserve(
        repository,
        video_id="video-2",
        run_id="text-run-periodic-2",
        generation_id="text-generation-periodic-2",
        lease_seconds=1,
    )
    cutoff = (
        max(
            datetime.fromisoformat(first.lease_expires_at),
            datetime.fromisoformat(second.lease_expires_at),
        )
        + timedelta(seconds=1)
    ).isoformat()

    first_report = repository.recover_expired_text_vector_builds(
        before=cutoff,
        limit=1,
    )
    second_report = repository.recover_expired_text_vector_builds(
        before=cutoff,
        limit=1,
    )

    assert first_report.examined_count == 1
    assert first_report.has_more is True
    assert second_report.examined_count == 1
    assert second_report.has_more is False
    assert set(
        first_report.terminalized_generation_ids
        + second_report.terminalized_generation_ids
    ) == {first.generation_id, second.generation_id}


def test_gc_audit_and_quarantine_list_limits_are_bounded(tmp_path) -> None:
    repository = _repository(tmp_path)
    _plan, pending = _pending_gc_job(repository)
    for invalid in (0, 1001, True):
        with pytest.raises(ValueError):
            repository.list_artifact_gc_attempts(pending.job_id, limit=invalid)
        with pytest.raises(ValueError):
            repository.list_artifact_gc_quarantines(limit=invalid)
        with pytest.raises(ValueError):
            repository.list_quarantined_text_vector_builds(limit=invalid)
