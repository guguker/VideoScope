from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
import stat
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from videoscope.artifacts import (
    ArtifactGCJob,
    ArtifactGCOutcome,
    TextVectorBuildQuarantine,
    TextVectorBuildRecoveryReport,
)
from videoscope.runtime import Runtime
from videoscope.runtime_lifecycle import (
    ArtifactGarbageCollector,
    ExclusiveRuntimeLock,
    TextVectorStorageGate,
)


def _running_gc_job(*, generation_id: str = "generation-1") -> ArtifactGCJob:
    now = datetime.now(UTC)
    return ArtifactGCJob(
        job_id=f"job-{generation_id}",
        artifact_kind="text_vectors",
        generation_id=generation_id,
        index_specification_hash="a" * 64,
        collection_name=f"videoscope_text_v1_{'a' * 32}",
        reason="text_vector_build_interrupted",
        state="running",
        attempt=1,
        backoff_level=0,
        available_at=now.isoformat(),
        worker_id="worker-1",
        lease_token="lease-1",
        lease_expires_at=(now + timedelta(seconds=60)).isoformat(),
        error_code=None,
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
    )


class FakeGCRepository:
    def __init__(self, jobs: list[ArtifactGCJob] | None = None) -> None:
        self.jobs = list(jobs or [])
        self.abandoned_recovery_limits: list[int] = []
        self.quarantine_limits: list[int] = []
        self.recovery_limits: list[int] = []
        self.heartbeats: list[tuple[str, str, str, int, int]] = []
        self.finished: list[tuple[str, str, str, int, ArtifactGCOutcome, str | None]] = []

    def recover_abandoned_text_vector_builds(
        self,
        *,
        limit: int = 100,
    ) -> TextVectorBuildRecoveryReport:
        self.abandoned_recovery_limits.append(limit)
        return TextVectorBuildRecoveryReport(
            examined_count=1,
            terminalized_generation_ids=("abandoned-generation",),
            quarantined_count=0,
            has_more=False,
        )

    def list_quarantined_text_vector_builds(
        self,
        *,
        limit: int = 100,
    ) -> tuple[TextVectorBuildQuarantine, ...]:
        self.quarantine_limits.append(limit)
        return ()

    def recover_expired_text_vector_builds(
        self,
        *,
        limit: int = 100,
    ) -> TextVectorBuildRecoveryReport:
        self.recovery_limits.append(limit)
        return TextVectorBuildRecoveryReport(0, (), 0, False)

    def claim_next_artifact_gc_job(
        self,
        *,
        worker_id: str,
        lease_seconds: int = 30,
    ) -> ArtifactGCJob | None:
        del worker_id, lease_seconds
        return self.jobs.pop(0) if self.jobs else None

    def heartbeat_artifact_gc_job(
        self,
        job_id: str,
        *,
        worker_id: str,
        lease_token: str,
        attempt: int,
        lease_seconds: int = 30,
    ) -> ArtifactGCJob:
        self.heartbeats.append(
            (job_id, worker_id, lease_token, attempt, lease_seconds)
        )
        return _running_gc_job(generation_id=job_id.removeprefix("job-"))

    def finish_artifact_gc_job(
        self,
        job_id: str,
        *,
        worker_id: str,
        lease_token: str,
        attempt: int,
        outcome: ArtifactGCOutcome,
        error_code: str | None = None,
    ) -> ArtifactGCJob:
        self.finished.append(
            (job_id, worker_id, lease_token, attempt, outcome, error_code)
        )
        return _running_gc_job(generation_id=job_id.removeprefix("job-"))


class RecordingGenerationStore:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.deleted: list[tuple[str, str, str]] = []

    def delete_generation_from_storage(
        self,
        generation_id: str,
        *,
        index_specification_hash: str,
        collection_name: str,
    ) -> None:
        if self.error is not None:
            raise self.error
        self.deleted.append(
            (generation_id, index_specification_hash, collection_name)
        )


def test_runtime_lock_is_exclusive_reusable_and_private(tmp_path) -> None:
    first = ExclusiveRuntimeLock(tmp_path)
    first.acquire()

    lock_stat = first.path.stat(follow_symlinks=False)
    assert stat.S_IMODE(lock_stat.st_mode) == 0o600
    with pytest.raises(RuntimeError, match="already owns"):
        ExclusiveRuntimeLock(tmp_path).acquire()

    first.close()
    first.close()
    second = ExclusiveRuntimeLock(tmp_path)
    second.acquire()
    second.close()


def test_runtime_lock_rejects_a_symbolic_link(tmp_path) -> None:
    target = tmp_path / "outside.lock"
    target.write_text("do not touch", encoding="utf-8")
    lock_path = tmp_path / ".videoscope-runtime.lock"
    lock_path.symlink_to(target)

    with pytest.raises(RuntimeError, match="regular file"):
        ExclusiveRuntimeLock(tmp_path).acquire()

    assert target.read_text(encoding="utf-8") == "do not touch"


def test_runtime_lock_rejects_a_symbolic_link_ancestor_before_creating_data(
    tmp_path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(outside, target_is_directory=True)
    data_dir = linked_parent / "data"

    with pytest.raises(RuntimeError, match="symbolic link"):
        ExclusiveRuntimeLock(data_dir).acquire()

    assert data_dir.exists() is False


def test_runtime_lock_cannot_be_acquired_twice_by_one_instance(tmp_path) -> None:
    lock = ExclusiveRuntimeLock(tmp_path)
    lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="already acquired"):
            lock.acquire()
    finally:
        lock.close()


def test_existing_runtime_lock_reuses_the_private_inode_without_mutating_it(
    tmp_path,
) -> None:
    creator = ExclusiveRuntimeLock(tmp_path)
    creator.acquire()
    creator.close()
    before = creator.path.stat(follow_symlinks=False)

    first = ExclusiveRuntimeLock(tmp_path)
    first.acquire_existing()
    try:
        with pytest.raises(RuntimeError, match="already owns"):
            ExclusiveRuntimeLock(tmp_path).acquire_existing()
    finally:
        first.close()
        first.close()

    after = creator.path.stat(follow_symlinks=False)
    assert (after.st_dev, after.st_ino, after.st_mode, after.st_nlink) == (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
    )
    assert (after.st_size, after.st_mtime_ns, after.st_ctime_ns) == (
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )


def test_existing_runtime_lock_never_creates_missing_state(tmp_path) -> None:
    missing_data = tmp_path / "missing"
    with pytest.raises(RuntimeError, match="unavailable"):
        ExclusiveRuntimeLock(missing_data).acquire_existing()
    assert missing_data.exists() is False

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with pytest.raises(RuntimeError, match="lock.*unavailable"):
        ExclusiveRuntimeLock(data_dir).acquire_existing()
    assert list(data_dir.iterdir()) == []


@pytest.mark.parametrize("unsafe_kind", ["symlink", "directory", "hardlink", "mode"])
def test_existing_runtime_lock_rejects_unsafe_persistent_files(
    tmp_path,
    unsafe_kind: str,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    lock_path = data_dir / ExclusiveRuntimeLock.filename
    outside = tmp_path / "outside.lock"
    outside.write_text("persistent", encoding="utf-8")
    outside.chmod(0o600)
    if unsafe_kind == "symlink":
        lock_path.symlink_to(outside)
    elif unsafe_kind == "directory":
        lock_path.mkdir()
    elif unsafe_kind == "hardlink":
        lock_path.hardlink_to(outside)
    else:
        lock_path.write_text("persistent", encoding="utf-8")
        lock_path.chmod(0o644)

    with pytest.raises(RuntimeError, match="managed regular file"):
        ExclusiveRuntimeLock(data_dir).acquire_existing()

    assert outside.read_text(encoding="utf-8") == "persistent"


def test_existing_runtime_lock_rejects_a_symbolic_link_ancestor(tmp_path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    lock_path = outside / ExclusiveRuntimeLock.filename
    lock_path.write_text("persistent", encoding="utf-8")
    lock_path.chmod(0o600)
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symbolic link"):
        ExclusiveRuntimeLock(linked).acquire_existing()

    assert lock_path.read_text(encoding="utf-8") == "persistent"


def test_artifact_gc_startup_recovery_is_bounded_and_explicit() -> None:
    repository = FakeGCRepository()
    collector = ArtifactGarbageCollector(
        repository,  # type: ignore[arg-type]
        RecordingGenerationStore(),  # type: ignore[arg-type]
        startup_recovery_limit=17,
    )

    assert collector.recover_startup() == ("abandoned-generation",)
    assert repository.abandoned_recovery_limits == [18]
    assert repository.quarantine_limits == [1]


def test_artifact_gc_startup_recovery_accepts_exact_configured_limit() -> None:
    class ExactLimitRepository(FakeGCRepository):
        def recover_abandoned_text_vector_builds(
            self,
            *,
            limit: int = 100,
        ) -> TextVectorBuildRecoveryReport:
            self.abandoned_recovery_limits.append(limit)
            generation_ids = tuple(
                f"generation-{index}" for index in range(limit - 1)
            )
            return TextVectorBuildRecoveryReport(
                examined_count=len(generation_ids),
                terminalized_generation_ids=generation_ids,
                quarantined_count=0,
                has_more=False,
            )

    repository = ExactLimitRepository()
    collector = ArtifactGarbageCollector(
        repository,  # type: ignore[arg-type]
        RecordingGenerationStore(),  # type: ignore[arg-type]
        startup_recovery_limit=3,
    )

    assert collector.recover_startup() == (
        "generation-0",
        "generation-1",
        "generation-2",
    )
    assert repository.abandoned_recovery_limits == [4]


def test_artifact_gc_startup_recovery_rejects_only_a_real_overflow() -> None:
    class OverflowRepository(FakeGCRepository):
        def recover_abandoned_text_vector_builds(
            self,
            *,
            limit: int = 100,
        ) -> TextVectorBuildRecoveryReport:
            self.abandoned_recovery_limits.append(limit)
            generation_ids = tuple(f"generation-{index}" for index in range(limit))
            return TextVectorBuildRecoveryReport(
                examined_count=len(generation_ids),
                terminalized_generation_ids=generation_ids,
                quarantined_count=0,
                has_more=False,
            )

    repository = OverflowRepository()
    collector = ArtifactGarbageCollector(
        repository,  # type: ignore[arg-type]
        RecordingGenerationStore(),  # type: ignore[arg-type]
        startup_recovery_limit=3,
    )

    with pytest.raises(RuntimeError, match="limit was exceeded"):
        collector.recover_startup()
    assert repository.abandoned_recovery_limits == [4]


def test_artifact_gc_startup_recovery_uses_has_more_at_maximum_limit() -> None:
    class HasMoreRepository(FakeGCRepository):
        def recover_abandoned_text_vector_builds(
            self,
            *,
            limit: int = 100,
        ) -> TextVectorBuildRecoveryReport:
            self.abandoned_recovery_limits.append(limit)
            generation_ids = tuple(f"generation-{index}" for index in range(limit))
            return TextVectorBuildRecoveryReport(
                examined_count=len(generation_ids),
                terminalized_generation_ids=generation_ids,
                quarantined_count=0,
                has_more=True,
            )

    repository = HasMoreRepository()
    collector = ArtifactGarbageCollector(
        repository,  # type: ignore[arg-type]
        RecordingGenerationStore(),  # type: ignore[arg-type]
        startup_recovery_limit=1000,
    )

    with pytest.raises(RuntimeError, match="limit was exceeded"):
        collector.recover_startup()
    assert repository.abandoned_recovery_limits == [1000]


def test_artifact_gc_startup_recovery_quarantines_without_deleting_target() -> None:
    report = TextVectorBuildRecoveryReport(
        examined_count=2,
        terminalized_generation_ids=("safe-generation",),
        quarantined_count=1,
        has_more=False,
    )

    class QuarantineRepository(FakeGCRepository):
        def recover_abandoned_text_vector_builds(
            self,
            *,
            limit: int = 100,
        ) -> TextVectorBuildRecoveryReport:
            self.abandoned_recovery_limits.append(limit)
            return report

    repository = QuarantineRepository()
    store = RecordingGenerationStore()
    collector = ArtifactGarbageCollector(
        repository,  # type: ignore[arg-type]
        store,  # type: ignore[arg-type]
        startup_recovery_limit=3,
    )

    assert collector.recover_startup() == ("safe-generation",)
    assert collector.recovery_degraded is True
    assert collector.last_recovery_report is report
    assert store.deleted == []
    assert repository.abandoned_recovery_limits == [4]


def test_artifact_gc_startup_recovery_preserves_existing_degraded_health() -> None:
    quarantine = TextVectorBuildQuarantine(
        row_id=1,
        error_code="text_vector_build_metadata_corrupt",
        quarantined_at=datetime.now(UTC).isoformat(),
    )

    class ExistingQuarantineRepository(FakeGCRepository):
        def recover_abandoned_text_vector_builds(
            self,
            *,
            limit: int = 100,
        ) -> TextVectorBuildRecoveryReport:
            self.abandoned_recovery_limits.append(limit)
            return TextVectorBuildRecoveryReport(0, (), 0, False)

        def list_quarantined_text_vector_builds(
            self,
            *,
            limit: int = 100,
        ) -> tuple[TextVectorBuildQuarantine, ...]:
            self.quarantine_limits.append(limit)
            return (quarantine,)

    repository = ExistingQuarantineRepository()
    collector = ArtifactGarbageCollector(
        repository,  # type: ignore[arg-type]
        RecordingGenerationStore(),  # type: ignore[arg-type]
        startup_recovery_limit=3,
    )

    assert collector.recover_startup() == ()
    assert collector.recovery_degraded is True
    assert repository.abandoned_recovery_limits == [4]
    assert repository.quarantine_limits == [1]


def test_artifact_gc_startup_recovery_rejects_an_invalid_report() -> None:
    class InvalidReportRepository(FakeGCRepository):
        def recover_abandoned_text_vector_builds(  # type: ignore[override]
            self,
            *,
            limit: int = 100,
        ) -> object:
            self.abandoned_recovery_limits.append(limit)
            return ("unvalidated-generation",)

    collector = ArtifactGarbageCollector(
        InvalidReportRepository(),  # type: ignore[arg-type]
        RecordingGenerationStore(),  # type: ignore[arg-type]
        startup_recovery_limit=3,
    )

    with pytest.raises(RuntimeError, match="invalid report"):
        collector.recover_startup()


def test_artifact_gc_success_uses_exact_scope_and_fenced_completion() -> None:
    job = _running_gc_job()
    repository = FakeGCRepository([job])
    store = RecordingGenerationStore()
    collector = ArtifactGarbageCollector(
        repository,  # type: ignore[arg-type]
        store,  # type: ignore[arg-type]
        worker_id=job.worker_id,
        batch_size=4,
    )

    assert collector.run_batch() == 1
    assert store.deleted == [
        (job.generation_id, job.index_specification_hash, job.collection_name)
    ]
    assert repository.finished == [
        (
            job.job_id,
            job.worker_id,
            job.lease_token,
            job.attempt,
            ArtifactGCOutcome.SUCCESS,
            None,
        )
    ]
    assert repository.recovery_limits == [4]


@pytest.mark.parametrize(
    ("error", "outcome", "error_code"),
    [
        (
            OSError("private qdrant path"),
            ArtifactGCOutcome.TRANSIENT_FAILURE,
            "artifact_gc_storage_unavailable",
        ),
        (
            ValueError("private corrupt scope"),
            ArtifactGCOutcome.PERMANENT_FAILURE,
            "artifact_gc_contract_invalid",
        ),
    ],
)
def test_artifact_gc_classifies_failures_without_persisting_provider_details(
    error: Exception,
    outcome: ArtifactGCOutcome,
    error_code: str,
) -> None:
    job = _running_gc_job()
    repository = FakeGCRepository([job])
    collector = ArtifactGarbageCollector(
        repository,  # type: ignore[arg-type]
        RecordingGenerationStore(error),  # type: ignore[arg-type]
        worker_id=job.worker_id,
    )

    assert collector.run_batch() == 1
    assert repository.finished[-1][-2:] == (outcome, error_code)
    assert "private" not in repr(repository.finished)


def test_artifact_gc_batch_never_claims_more_than_its_bound() -> None:
    jobs = [_running_gc_job(generation_id=f"generation-{index}") for index in range(4)]
    repository = FakeGCRepository(jobs)
    collector = ArtifactGarbageCollector(
        repository,  # type: ignore[arg-type]
        RecordingGenerationStore(),  # type: ignore[arg-type]
        worker_id="worker-1",
        batch_size=2,
    )

    assert collector.run_batch() == 2
    assert len(repository.jobs) == 2


def test_artifact_gc_skips_recovery_and_claim_while_vector_build_is_active() -> None:
    job = _running_gc_job()
    repository = FakeGCRepository([job])
    gate = TextVectorStorageGate()
    collector = ArtifactGarbageCollector(
        repository,  # type: ignore[arg-type]
        RecordingGenerationStore(),  # type: ignore[arg-type]
        worker_id=job.worker_id,
        storage_gate=gate,
    )

    with gate.build_activity():
        assert collector.run_batch() == 0

    assert repository.recovery_limits == []
    assert repository.jobs == [job]


def test_artifact_gc_maintenance_blocks_a_new_build_through_completion() -> None:
    job = _running_gc_job()
    order: list[str] = []
    delete_started = Event()
    allow_delete = Event()
    writer_started = Event()
    writer_entered = Event()

    class OrderedRepository(FakeGCRepository):
        def recover_expired_text_vector_builds(
            self,
            *,
            limit: int = 100,
        ) -> TextVectorBuildRecoveryReport:
            order.append("recover")
            return super().recover_expired_text_vector_builds(limit=limit)

        def claim_next_artifact_gc_job(
            self,
            *,
            worker_id: str,
            lease_seconds: int = 30,
        ) -> ArtifactGCJob | None:
            order.append("claim")
            return super().claim_next_artifact_gc_job(
                worker_id=worker_id,
                lease_seconds=lease_seconds,
            )

        def finish_artifact_gc_job(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            order.append("finish")
            return super().finish_artifact_gc_job(*args, **kwargs)

    class BlockingStore(RecordingGenerationStore):
        def delete_generation_from_storage(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            order.append("delete-start")
            delete_started.set()
            assert allow_delete.wait(timeout=2)
            super().delete_generation_from_storage(*args, **kwargs)
            order.append("delete-end")

    repository = OrderedRepository([job])
    gate = TextVectorStorageGate()
    collector = ArtifactGarbageCollector(
        repository,  # type: ignore[arg-type]
        BlockingStore(),  # type: ignore[arg-type]
        worker_id=job.worker_id,
        storage_gate=gate,
    )

    gc_thread = Thread(target=collector.run_batch)
    gc_thread.start()
    assert delete_started.wait(timeout=2)

    def enter_build() -> None:
        writer_started.set()
        with gate.build_activity():
            order.append("build")
            writer_entered.set()

    writer_thread = Thread(target=enter_build)
    writer_thread.start()
    assert writer_started.wait(timeout=2)
    assert writer_entered.wait(timeout=0.05) is False

    allow_delete.set()
    gc_thread.join(timeout=2)
    writer_thread.join(timeout=2)
    assert gc_thread.is_alive() is False
    assert writer_entered.is_set()
    assert order == [
        "recover",
        "claim",
        "delete-start",
        "delete-end",
        "finish",
        "claim",
        "build",
    ]


def test_artifact_gc_quarantined_recovery_is_degraded_without_a_delete() -> None:
    class QuarantiningRepository(FakeGCRepository):
        def recover_expired_text_vector_builds(
            self,
            *,
            limit: int = 100,
        ) -> TextVectorBuildRecoveryReport:
            self.recovery_limits.append(limit)
            return TextVectorBuildRecoveryReport(1, (), 1, False)

    store = RecordingGenerationStore()
    collector = ArtifactGarbageCollector(
        QuarantiningRepository(),  # type: ignore[arg-type]
        store,  # type: ignore[arg-type]
    )

    assert collector.run_batch() == 0
    assert collector.recovery_degraded is True
    assert store.deleted == []


def test_artifact_gc_shutdown_does_not_wait_for_an_active_build() -> None:
    repository = FakeGCRepository()
    gate = TextVectorStorageGate()
    collector = ArtifactGarbageCollector(
        repository,  # type: ignore[arg-type]
        RecordingGenerationStore(),  # type: ignore[arg-type]
        storage_gate=gate,
        poll_interval=60,
    )

    with gate.build_activity():
        collector.start()
        assert collector.close(timeout=1) is True

    assert repository.recovery_limits == []


def test_artifact_gc_thread_start_failure_is_safe_to_close(monkeypatch) -> None:
    class FailingThread:
        def __init__(self, **_kwargs) -> None:  # type: ignore[no-untyped-def]
            return None

        def start(self) -> None:
            raise RuntimeError("thread unavailable")

    collector = ArtifactGarbageCollector(
        FakeGCRepository(),  # type: ignore[arg-type]
        RecordingGenerationStore(),  # type: ignore[arg-type]
    )
    monkeypatch.setattr("videoscope.runtime_lifecycle.Thread", FailingThread)

    with pytest.raises(RuntimeError, match="thread unavailable"):
        collector.start()
    assert collector.close() is True


def test_runtime_orders_recovery_before_indexing_and_releases_resources() -> None:
    events: list[str] = []

    class RecordingLock:
        def acquire(self) -> None:
            events.append("lock-acquire")

        def close(self) -> None:
            events.append("lock-close")

    class RecordingCollector:
        def recover_startup(self) -> tuple[str, ...]:
            events.append("gc-recover")
            return ()

        def start(self) -> None:
            events.append("gc-start")

        def close(self, *, timeout: float | None = 5) -> bool:
            del timeout
            events.append("gc-close")
            return True

    class RecordingQueue:
        def recover_startup(self) -> tuple[str, ...]:
            events.append("queue-recover")
            return ()

        def start(self) -> None:
            events.append("queue-start")

        def close(self, *, timeout: float | None = 5) -> bool:
            del timeout
            events.append("queue-close")
            return True

    class RecordingStore:
        def close(self) -> None:
            events.append("store-close")

    class RecordingEmbeddingRuntime:
        def close(self) -> bool:
            events.append("embedding-close")
            return True

    class RecordingOCR:
        def close(self) -> None:
            events.append("ocr-close")

    runtime = Runtime(
        queue=RecordingQueue(),  # type: ignore[arg-type]
        search=object(),  # type: ignore[arg-type]
        clips=object(),  # type: ignore[arg-type]
        providers=object(),  # type: ignore[arg-type]
        runtime_lock=RecordingLock(),  # type: ignore[arg-type]
        artifact_gc=RecordingCollector(),  # type: ignore[arg-type]
        generation_store=RecordingStore(),  # type: ignore[arg-type]
        legacy_job_adopter=lambda: events.append("legacy-adopt"),
        text_embedding_runtime=RecordingEmbeddingRuntime(),  # type: ignore[arg-type]
        ocr_reader=RecordingOCR(),  # type: ignore[arg-type]
    )

    runtime.start()
    assert events == [
        "lock-acquire",
        "gc-recover",
        "legacy-adopt",
        "queue-recover",
        "gc-start",
        "queue-start",
    ]
    assert runtime.close() is True
    assert events[-6:] == [
        "queue-close",
        "gc-close",
        "ocr-close",
        "store-close",
        "embedding-close",
        "lock-close",
    ]
    completed_events = list(events)
    assert runtime.close() is True
    assert events == completed_events


def test_runtime_start_failure_releases_preacquired_ownership() -> None:
    events: list[str] = []

    class Lock:
        is_acquired = True

        def acquire(self) -> None:
            raise AssertionError("lock is already held")

        def close(self) -> None:
            events.append("lock-close")

    class Collector:
        def recover_startup(self) -> tuple[str, ...]:
            events.append("gc-recover")
            raise RuntimeError("recovery failed")

        def start(self) -> None:
            raise AssertionError("must not start")

        def close(self, *, timeout: float | None = 5) -> bool:
            events.append(f"gc-close:{timeout}")
            return True

    class Queue:
        def recover_startup(self) -> tuple[str, ...]:
            raise AssertionError("must not recover jobs")

        def start(self) -> None:
            raise AssertionError("must not start")

        def close(self, *, timeout: float | None = 5) -> bool:
            events.append(f"queue-close:{timeout}")
            return True

    class Store:
        def close(self) -> None:
            events.append("store-close")

    runtime = Runtime(
        queue=Queue(),  # type: ignore[arg-type]
        search=object(),  # type: ignore[arg-type]
        clips=object(),  # type: ignore[arg-type]
        providers=object(),  # type: ignore[arg-type]
        runtime_lock=Lock(),  # type: ignore[arg-type]
        artifact_gc=Collector(),  # type: ignore[arg-type]
        generation_store=Store(),  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="recovery failed"):
        runtime.start()
    assert events == [
        "gc-recover",
        "queue-close:5",
        "gc-close:5",
        "store-close",
        "lock-close",
    ]
    assert runtime.close() is True


def test_runtime_does_not_release_storage_lock_while_queue_is_still_running() -> None:
    events: list[str] = []
    ownership_released = Event()

    class Lock:
        def acquire(self) -> None:
            events.append("lock-acquire")

        def close(self) -> None:
            events.append("lock-close")
            ownership_released.set()

    class Collector:
        def recover_startup(self) -> tuple[str, ...]:
            return ()

        def start(self) -> None:
            return None

        def close(self, *, timeout: float | None = 5) -> bool:
            del timeout
            return True

    class Queue:
        close_results = [False, True]

        def recover_startup(self) -> tuple[str, ...]:
            return ()

        def start(self) -> None:
            return None

        def close(self, *, timeout: float | None = 5) -> bool:
            del timeout
            return self.close_results.pop(0)

    class Store:
        def close(self) -> None:
            events.append("store-close")

    runtime = Runtime(
        queue=Queue(),  # type: ignore[arg-type]
        search=object(),  # type: ignore[arg-type]
        clips=object(),  # type: ignore[arg-type]
        providers=object(),  # type: ignore[arg-type]
        runtime_lock=Lock(),  # type: ignore[arg-type]
        artifact_gc=Collector(),  # type: ignore[arg-type]
        generation_store=Store(),  # type: ignore[arg-type]
    )
    runtime.start()

    assert runtime.close() is False
    assert ownership_released.wait(timeout=2)
    assert runtime.close() is True
    assert events == ["lock-acquire", "store-close", "lock-close"]


def test_runtime_reaper_eventually_releases_ownership_after_slow_work() -> None:
    release = Event()
    ownership_released = Event()
    events: list[str] = []

    class Lock:
        is_acquired = False

        def acquire(self) -> None:
            self.is_acquired = True
            events.append("lock-acquire")

        def close(self) -> None:
            self.is_acquired = False
            events.append("lock-close")
            ownership_released.set()

    class Collector:
        def recover_startup(self) -> tuple[str, ...]:
            return ()

        def start(self) -> None:
            return None

        def close(self, *, timeout: float | None = 5) -> bool:
            del timeout
            return True

    class SlowQueue:
        close_calls: list[float | None] = []

        def recover_startup(self) -> tuple[str, ...]:
            return ()

        def start(self) -> None:
            return None

        def close(self, *, timeout: float | None = 5) -> bool:
            self.close_calls.append(timeout)
            if len(self.close_calls) == 1:
                return False
            release.wait(timeout=2)
            return release.is_set()

    class Store:
        def close(self) -> None:
            events.append("store-close")

    queue = SlowQueue()
    runtime = Runtime(
        queue=queue,  # type: ignore[arg-type]
        search=object(),  # type: ignore[arg-type]
        clips=object(),  # type: ignore[arg-type]
        providers=object(),  # type: ignore[arg-type]
        runtime_lock=Lock(),  # type: ignore[arg-type]
        artifact_gc=Collector(),  # type: ignore[arg-type]
        generation_store=Store(),  # type: ignore[arg-type]
    )
    runtime.start()

    assert runtime.close() is False
    assert events == ["lock-acquire"]
    release.set()
    assert ownership_released.wait(timeout=2)
    assert events == ["lock-acquire", "store-close", "lock-close"]
    assert queue.close_calls == [5, None]


def test_runtime_store_close_failure_retains_lock_until_reaper_retry_succeeds() -> None:
    retry_started = Event()
    allow_store_close = Event()
    ownership_released = Event()
    events: list[str] = []

    class Lock:
        is_acquired = False

        def acquire(self) -> None:
            self.is_acquired = True
            events.append("lock-acquire")

        def close(self) -> None:
            self.is_acquired = False
            events.append("lock-close")
            ownership_released.set()

    class Collector:
        def recover_startup(self) -> tuple[str, ...]:
            return ()

        def start(self) -> None:
            return None

        def close(self, *, timeout: float | None = 5) -> bool:
            del timeout
            return True

    class Queue:
        def recover_startup(self) -> tuple[str, ...]:
            return ()

        def start(self) -> None:
            return None

        def close(self, *, timeout: float | None = 5) -> bool:
            del timeout
            return True

    class FailOnceStore:
        close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                events.append("store-close-failed")
                raise RuntimeError("store unavailable")
            retry_started.set()
            assert allow_store_close.wait(timeout=2)
            events.append("store-close-succeeded")

    store = FailOnceStore()
    lock = Lock()
    runtime = Runtime(
        queue=Queue(),  # type: ignore[arg-type]
        search=object(),  # type: ignore[arg-type]
        clips=object(),  # type: ignore[arg-type]
        providers=object(),  # type: ignore[arg-type]
        runtime_lock=lock,  # type: ignore[arg-type]
        artifact_gc=Collector(),  # type: ignore[arg-type]
        generation_store=store,  # type: ignore[arg-type]
    )
    runtime.start()

    assert runtime.close() is False
    assert retry_started.wait(timeout=2)
    assert lock.is_acquired is True
    assert ownership_released.is_set() is False
    allow_store_close.set()
    assert ownership_released.wait(timeout=2)
    assert runtime.close() is True
    assert store.close_calls == 2
    assert events == [
        "lock-acquire",
        "store-close-failed",
        "store-close-succeeded",
        "lock-close",
    ]


def test_runtime_lock_close_error_retries_only_release_phase() -> None:
    release_retry_started = Event()
    allow_release = Event()
    events: list[str] = []

    class Lock:
        close_calls = 0

        def acquire(self) -> None:
            return None

        def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                events.append("lock-close-failed")
                raise OSError("release result is uncertain")
            release_retry_started.set()
            assert allow_release.wait(timeout=2)
            events.append("lock-close-succeeded")

    class Collector:
        def recover_startup(self) -> tuple[str, ...]:
            return ()

        def start(self) -> None:
            return None

        def close(self, *, timeout: float | None = 5) -> bool:
            del timeout
            return True

    class Queue:
        def recover_startup(self) -> tuple[str, ...]:
            return ()

        def start(self) -> None:
            return None

        def close(self, *, timeout: float | None = 5) -> bool:
            del timeout
            return True

    class Store:
        def close(self) -> None:
            events.append("store-close")

    lock = Lock()
    runtime = Runtime(
        queue=Queue(),  # type: ignore[arg-type]
        search=object(),  # type: ignore[arg-type]
        clips=object(),  # type: ignore[arg-type]
        providers=object(),  # type: ignore[arg-type]
        runtime_lock=lock,  # type: ignore[arg-type]
        artifact_gc=Collector(),  # type: ignore[arg-type]
        generation_store=Store(),  # type: ignore[arg-type]
    )
    runtime.start()

    assert runtime.close() is False
    assert release_retry_started.wait(timeout=2)
    assert events == ["store-close", "lock-close-failed"]
    allow_release.set()
    assert runtime._shutdown_complete.wait(timeout=2)
    assert runtime.close() is True
    assert lock.close_calls == 2
    assert events == [
        "store-close",
        "lock-close-failed",
        "lock-close-succeeded",
    ]


def test_runtime_reaper_start_failure_finishes_retries_synchronously(
    monkeypatch,
) -> None:
    events: list[str] = []

    class FailingThread:
        def __init__(self, **_kwargs) -> None:  # type: ignore[no-untyped-def]
            return None

        def start(self) -> None:
            raise RuntimeError("thread unavailable")

    class Lock:
        def acquire(self) -> None:
            return None

        def close(self) -> None:
            events.append("lock-close")

    class Collector:
        def recover_startup(self) -> tuple[str, ...]:
            return ()

        def start(self) -> None:
            return None

        def close(self, *, timeout: float | None = 5) -> bool:
            del timeout
            return True

    class Queue:
        def recover_startup(self) -> tuple[str, ...]:
            return ()

        def start(self) -> None:
            return None

        def close(self, *, timeout: float | None = 5) -> bool:
            del timeout
            return True

    class FailOnceStore:
        close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            events.append(f"store-close:{self.close_calls}")
            if self.close_calls == 1:
                raise RuntimeError("store unavailable")

    store = FailOnceStore()
    runtime = Runtime(
        queue=Queue(),  # type: ignore[arg-type]
        search=object(),  # type: ignore[arg-type]
        clips=object(),  # type: ignore[arg-type]
        providers=object(),  # type: ignore[arg-type]
        runtime_lock=Lock(),  # type: ignore[arg-type]
        artifact_gc=Collector(),  # type: ignore[arg-type]
        generation_store=store,  # type: ignore[arg-type]
    )
    runtime.start()
    monkeypatch.setattr("videoscope.runtime.Thread", FailingThread)

    assert runtime.close() is True
    assert events == ["store-close:1", "store-close:2", "lock-close"]


@pytest.mark.parametrize("pending_role", ["queue", "artifact_gc"])
def test_runtime_waits_for_workers_before_closing_owned_ocr(pending_role: str) -> None:
    allow_stop = Event()
    reaper_waiting = Event()
    ownership_released = Event()
    events: list[str] = []

    class PendingWorker:
        def close(self, *, timeout: float | None = 5) -> bool:
            if timeout is not None:
                return False
            reaper_waiting.set()
            assert allow_stop.wait(timeout=2)
            events.append("worker-stopped")
            return True

    def close_lock() -> None:
        events.append("lock-close")
        ownership_released.set()

    workers = {
        "queue": SimpleNamespace(close=lambda **_kwargs: True),
        "artifact_gc": SimpleNamespace(close=lambda **_kwargs: True),
    }
    workers[pending_role] = PendingWorker()
    runtime = Runtime(
        **workers,  # type: ignore[arg-type]
        search=object(),  # type: ignore[arg-type]
        clips=object(),  # type: ignore[arg-type]
        providers=object(),  # type: ignore[arg-type]
        runtime_lock=SimpleNamespace(close=close_lock),  # type: ignore[arg-type]
        generation_store=SimpleNamespace(close=lambda: events.append("store-close")),  # type: ignore[arg-type]
        ocr_reader=SimpleNamespace(close=lambda: events.append("ocr-close")),  # type: ignore[arg-type]
    )

    try:
        assert runtime.close() is False
        assert reaper_waiting.wait(timeout=2)
        assert events == []
        assert ownership_released.is_set() is False
    finally:
        allow_stop.set()
    assert ownership_released.wait(timeout=2)
    assert runtime.close() is True
    assert events == ["worker-stopped", "ocr-close", "store-close", "lock-close"]


@pytest.mark.parametrize("failed_resource", ["ocr", "store"])
def test_runtime_retries_owned_ocr_cleanup_without_repeating_success(
    failed_resource: str,
) -> None:
    retry_started = Event()
    allow_retry = Event()
    ownership_released = Event()
    events: list[str] = []
    calls = {"ocr": 0, "store": 0}

    def close_resource(role: str) -> None:
        calls[role] += 1
        if role == failed_resource:
            if calls[role] == 1:
                events.append(f"{role}-failed")
                raise RuntimeError("owned resource cleanup unavailable")
            retry_started.set()
            assert allow_retry.wait(timeout=2)
        events.append(f"{role}-closed")

    def close_lock() -> None:
        events.append("lock-close")
        ownership_released.set()

    runtime = Runtime(
        queue=SimpleNamespace(close=lambda **_kwargs: True),  # type: ignore[arg-type]
        artifact_gc=SimpleNamespace(close=lambda **_kwargs: True),  # type: ignore[arg-type]
        search=object(),  # type: ignore[arg-type]
        clips=object(),  # type: ignore[arg-type]
        providers=object(),  # type: ignore[arg-type]
        runtime_lock=SimpleNamespace(close=close_lock),  # type: ignore[arg-type]
        generation_store=SimpleNamespace(close=lambda: close_resource("store")),  # type: ignore[arg-type]
        ocr_reader=SimpleNamespace(close=lambda: close_resource("ocr")),  # type: ignore[arg-type]
    )

    try:
        assert runtime.close() is False
        assert retry_started.wait(timeout=2)
        assert ownership_released.is_set() is False
        assert events == (
            ["ocr-failed"]
            if failed_resource == "ocr"
            else ["ocr-closed", "store-failed"]
        )
    finally:
        allow_retry.set()
    assert ownership_released.wait(timeout=2)
    assert runtime.close() is True
    assert calls == {
        "ocr": 2 if failed_resource == "ocr" else 1,
        "store": 2 if failed_resource == "store" else 1,
    }
    assert events[-1] == "lock-close"


@pytest.mark.parametrize("ocr_ready", [False, True])
def test_build_runtime_owns_its_exact_ocr_reader_even_when_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ocr_ready: bool,
) -> None:
    import videoscope.runtime as runtime_module
    from videoscope.config import AppSettings
    from videoscope.providers.base import ProviderState, ProviderStatus
    from videoscope.repository import Repository

    settings = AppSettings(
        _env_file=None,
        data_dir=tmp_path / "data",
        ocr_worker_python=tmp_path / "missing-ocr-python",
        ocr_worker_script=Path(__file__).parents[2] / "scripts/paddle-ocr-worker.py",
    )
    settings.ensure_directories()
    repository = Repository(settings.database_path)
    repository.initialize()
    closed: list[object] = []
    state = ProviderState.READY if ocr_ready else ProviderState.NEEDS_CONFIGURATION
    ocr = SimpleNamespace(
        id="paddleocr",
        status=lambda: ProviderStatus("paddleocr", "PaddleOCR", state, "fixture"),
        close=lambda: closed.append(ocr),
    )
    monkeypatch.setattr(runtime_module, "PaddleOCRReader", lambda **_kwargs: ocr)
    toolchain = SimpleNamespace(
        verify_current=lambda: "sha256:" + "a" * 64,
        create_ffmpeg=runtime_module.FFmpeg,
    )
    runtime = runtime_module.build_runtime(
        settings,
        repository,
        indexing_toolchain=toolchain,  # type: ignore[arg-type]
    )
    try:
        assert runtime.ocr_reader is ocr
        assert runtime.queue.indexer.ocr is (ocr if ocr_ready else None)
    finally:
        assert runtime.close() is True
    assert closed == [ocr]
    assert runtime.close() is True
    assert closed == [ocr]
