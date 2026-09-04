from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import ctypes
import errno
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading

import pytest

from videoscope.benchmark import measurements as measurements_module
from videoscope.benchmark import (
    AssetProvenance,
    BenchmarkAsset,
    BenchmarkDataError,
    BenchmarkDataset,
    BenchmarkMeasurementEvidence,
    BenchmarkRunRegistry,
    BenchmarkStorageSnapshot,
    ComponentIdentity,
    HardwareProfile,
    LocalAssetResolver,
    MAX_MEASUREMENT_RSS_SAMPLES,
    MetricValue,
    QueryCase,
)
from videoscope.benchmark.measurements import (
    PROCESS_RSS_SAMPLE_INTERVAL_SECONDS,
    DarwinLibprocProcessSnapshotProvider,
    DeclaredStorageRoot,
    MeasurementError,
    MeasurementUnavailableError,
    ManagedProcessBinding,
    ProcessRecord,
    ProcessTreeRssSampler,
    StorageSnapshotter,
    StorageTraversalLimits,
    SystemMeasurementFactory,
    create_native_process_snapshot_provider,
    measurement_metrics_from_evidence,
)
from videoscope.benchmark.runner import (
    BenchmarkExecutionError,
    BenchmarkRunner,
    BenchmarkSearchHit,
    ExecutionIdentities,
    audit_run_manifest,
)
from videoscope.benchmark.profiles import get_profile
from videoscope.benchmark.serialization import run_from_dict, run_to_dict


_TEXT_MODEL_IDENTITY = (
    "fastembed@0.8.0:mean-pooling-v1:"
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2:"
    "xenova/paraphrase-multilingual-mpnet-base-v2@"
    "e5d116277351513fd260955ece953ecddde7046e:768"
)
_VISUAL_MODEL_IDENTITY = (
    "google/siglip2-base-patch16-224@"
    "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
)


class ScriptedProcessProvider:
    identity = "test-process-table@1"

    def __init__(
        self,
        snapshots: tuple[tuple[ProcessRecord, ...], ...],
        *,
        ready_after: int | None = None,
    ) -> None:
        self._snapshots = snapshots
        self._index = 0
        self.calls = 0
        self.ready = threading.Event()
        self._ready_after = ready_after
        self._lock = threading.Lock()

    def snapshot(self) -> tuple[ProcessRecord, ...]:
        with self._lock:
            index = min(self._index, len(self._snapshots) - 1)
            self._index += 1
            self.calls += 1
            if self._ready_after is not None and self.calls >= self._ready_after:
                self.ready.set()
            return self._snapshots[index]


class _ScriptedDarwinProcessApi:
    def __init__(
        self,
        *,
        records: dict[int, ProcessRecord | None],
        children: dict[int, tuple[int, ...]],
    ) -> None:
        self.records = records
        self.children = children
        self.record_calls: list[int] = []
        self.children_calls: list[int] = []

    def process_record(self, pid: int) -> ProcessRecord | None:
        self.record_calls.append(pid)
        return self.records.get(pid)

    def child_pids(self, pid: int) -> tuple[int, ...]:
        self.children_calls.append(pid)
        return self.children.get(pid, ())


def _constant_process_provider(*, rss_bytes: int = 100) -> ScriptedProcessProvider:
    return ScriptedProcessProvider(
        (
            (
                ProcessRecord(
                    pid=101,
                    parent_pid=1,
                    rss_bytes=rss_bytes,
                    start_token="root-start",
                    executable_identity="benchmark-python@1",
                ),
            ),
        )
    )


def test_darwin_provider_reads_only_the_declared_root_tree_without_ps() -> None:
    root = ProcessRecord(101, 1, 100, "root-start", "path-sha256:" + "a" * 64)
    worker = ProcessRecord(
        102,
        101,
        200,
        "worker-start",
        "path-sha256:" + "b" * 64,
    )
    helper = ProcessRecord(
        103,
        102,
        50,
        "helper-start",
        "path-sha256:" + "c" * 64,
    )
    api = _ScriptedDarwinProcessApi(
        records={101: root, 102: worker, 103: helper},
        children={101: (102,), 102: (103,)},
    )
    provider = DarwinLibprocProcessSnapshotProvider(
        root_pid=101,
        _api=api,
    )

    assert provider.snapshot() == (root, worker, helper)
    assert api.record_calls == [101, 102, 103]
    assert api.children_calls == [101, 102, 103]
    assert provider.identity == "darwin-libproc-rooted-process-tree@1"


def test_darwin_provider_fails_closed_on_cycles_or_parent_mismatch() -> None:
    root = ProcessRecord(101, 1, 100, "root", "path-sha256:" + "a" * 64)
    wrong_parent = ProcessRecord(
        102,
        999,
        200,
        "worker",
        "path-sha256:" + "b" * 64,
    )
    mismatch = DarwinLibprocProcessSnapshotProvider(
        root_pid=101,
        _api=_ScriptedDarwinProcessApi(
            records={101: root, 102: wrong_parent},
            children={101: (102,)},
        ),
    )

    with pytest.raises(MeasurementError) as mismatch_error:
        mismatch.snapshot()
    assert mismatch_error.value.code == "native_process_parent_changed"

    cycle = DarwinLibprocProcessSnapshotProvider(
        root_pid=101,
        _api=_ScriptedDarwinProcessApi(
            records={101: root},
            children={101: (101,)},
        ),
    )
    with pytest.raises(MeasurementError) as cycle_error:
        cycle.snapshot()
    assert cycle_error.value.code == "invalid_native_process_tree"


def test_native_provider_is_explicitly_unavailable_off_macos() -> None:
    with pytest.raises(MeasurementUnavailableError) as captured:
        create_native_process_snapshot_provider(
            root_pid=101,
            operating_system="Linux",
        )

    assert captured.value.code == "native_process_provider_unavailable"


def test_darwin_api_distinguishes_a_vanished_process_from_denied_access() -> None:
    class MissingLibrary:
        @staticmethod
        def proc_pidinfo(*_args) -> int:  # type: ignore[no-untyped-def]
            ctypes.set_errno(errno.ESRCH)
            return 0

    missing_api = object.__new__(measurements_module._CtypesDarwinProcessApi)
    missing_api._library = MissingLibrary()
    assert missing_api.process_record(101) is None

    class DeniedLibrary:
        @staticmethod
        def proc_pidinfo(*_args) -> int:  # type: ignore[no-untyped-def]
            ctypes.set_errno(errno.EPERM)
            return 0

    denied_api = object.__new__(measurements_module._CtypesDarwinProcessApi)
    denied_api._library = DeniedLibrary()
    with pytest.raises(MeasurementUnavailableError) as captured:
        denied_api.process_record(101)
    assert captured.value.code == "native_process_record_unavailable"


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS libproc contract")
def test_real_darwin_provider_attests_the_current_process() -> None:
    provider = create_native_process_snapshot_provider(root_pid=os.getpid())

    records = provider.snapshot()
    root = next(record for record in records if record.pid == os.getpid())

    assert root.parent_pid == os.getppid()
    assert root.rss_bytes > 0
    assert root.start_token.startswith("darwin-start:")
    assert root.executable_identity.startswith("path-sha256:")
    assert len(root.executable_identity.removeprefix("path-sha256:")) == 64


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS libproc contract")
def test_real_darwin_provider_includes_a_declared_root_descendant() -> None:
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(10)"],
    )
    try:
        records = create_native_process_snapshot_provider(
            root_pid=os.getpid()
        ).snapshot()
    finally:
        child.terminate()
        child.wait(timeout=5)

    by_pid = {record.pid: record for record in records}
    assert child.pid in by_pid
    assert by_pid[child.pid].parent_pid == os.getpid()
    assert by_pid[child.pid].rss_bytes > 0


def _worker_binding(pid: int = 102) -> ManagedProcessBinding:
    return ManagedProcessBinding(
        pid=pid,
        start_token="worker-start",
        executable_identity="vision-worker@1",
        role="vision",
    )


def _declared_roots(base: Path) -> tuple[DeclaredStorageRoot, ...]:
    active = base / "active"
    scratch = base / "scratch"
    active.mkdir(parents=True)
    scratch.mkdir()
    return (
        DeclaredStorageRoot(
            root_id="active-index",
            path=active,
            purpose="active_immutable_artifacts",
        ),
        DeclaredStorageRoot(
            root_id="benchmark-work",
            path=scratch,
            purpose="benchmark_scratch",
        ),
    )


def _portable_measurement_evidence() -> BenchmarkMeasurementEvidence:
    active = BenchmarkStorageSnapshot(
        root_id="active-index",
        purpose="active_immutable_artifacts",
        file_count=1,
        directory_count=1,
        logical_bytes=5,
        allocated_bytes=512,
        tree_digest="a" * 64,
    )
    scratch_before = BenchmarkStorageSnapshot(
        root_id="benchmark-work",
        purpose="benchmark_scratch",
        file_count=0,
        directory_count=1,
        logical_bytes=0,
        allocated_bytes=0,
        tree_digest="b" * 64,
    )
    scratch_after = BenchmarkStorageSnapshot(
        root_id="benchmark-work",
        purpose="benchmark_scratch",
        file_count=1,
        directory_count=1,
        logical_bytes=3,
        allocated_bytes=512,
        tree_digest="c" * 64,
    )
    return BenchmarkMeasurementEvidence(
        schema_version=1,
        rss_samples_bytes=(500, 550, 525),
        storage_before=(active, scratch_before),
        storage_after=(active, scratch_after),
        metal_telemetry_status="unavailable",
    )


def test_protocol_identity_is_path_private_and_freezes_the_complete_contract(
    tmp_path: Path,
) -> None:
    first_roots = _declared_roots(tmp_path / "first")
    second_roots = _declared_roots(tmp_path / "second")
    provider = _constant_process_provider()

    first = SystemMeasurementFactory(
        storage_roots=first_roots,
        execution_mode="warm",
        cache_policy_identity="preserve-pinned-caches@1",
        process_provider=provider,
        process_root_pid=101,
        managed_workers=(_worker_binding(102),),
    )
    equivalent = SystemMeasurementFactory(
        storage_roots=second_roots,
        execution_mode="warm",
        cache_policy_identity="preserve-pinned-caches@1",
        process_provider=provider,
        process_root_pid=101,
        managed_workers=(_worker_binding(202),),
    )
    changed_limits = SystemMeasurementFactory(
        storage_roots=second_roots,
        execution_mode="warm",
        cache_policy_identity="preserve-pinned-caches@1",
        process_provider=provider,
        process_root_pid=101,
        managed_workers=(_worker_binding(202),),
        storage_limits=StorageTraversalLimits(max_files=99),
    )

    assert first.protocol_identity() == equivalent.protocol_identity()
    assert first.protocol_identity() != changed_limits.protocol_identity()
    assert first.protocol_identity().component_id == "benchmark_measurement_protocol"
    assert first.protocol_identity().identity.startswith(
        "process-tree-rss-50ms-contained-storage@2:"
    )
    assert str(tmp_path) not in first.protocol_identity().identity
    assert PROCESS_RSS_SAMPLE_INTERVAL_SECONDS == 0.05


def test_process_sampler_includes_recursive_descendants_and_excludes_unrelated_pids(
) -> None:
    provider = ScriptedProcessProvider(
        (
            (
                ProcessRecord(101, 1, 100, "root-start", "benchmark-python@1"),
                ProcessRecord(102, 101, 25, "worker-start", "vision-worker@1"),
                ProcessRecord(999, 1, 10_000, "other-start", "unrelated@1"),
            ),
            (
                ProcessRecord(101, 1, 110, "root-start", "benchmark-python@1"),
                ProcessRecord(102, 101, 30, "worker-start", "vision-worker@1"),
                ProcessRecord(103, 102, 5, "child-start", "helper@1"),
                ProcessRecord(999, 1, 10_000, "other-start", "unrelated@1"),
            ),
            (
                ProcessRecord(101, 1, 105, "root-start", "benchmark-python@1"),
                ProcessRecord(102, 101, 20, "worker-start", "vision-worker@1"),
                ProcessRecord(999, 1, 10_000, "other-start", "unrelated@1"),
            ),
        ),
        ready_after=3,
    )
    sampler = ProcessTreeRssSampler(
        root_pid=101,
        provider=provider,
        managed_workers=(_worker_binding(),),
    )

    sampler.start()
    assert provider.ready.wait(timeout=1)
    result = sampler.finish()

    assert result.baseline_bytes == 125
    assert result.peak_bytes == 145
    assert result.increment_bytes == 20
    assert result.sample_count >= 3


def test_process_sampler_fails_when_root_or_declared_managed_worker_is_unavailable(
) -> None:
    missing_root = ProcessTreeRssSampler(
        root_pid=101,
        provider=ScriptedProcessProvider(
            ((ProcessRecord(999, 1, 100, "other", "unrelated@1"),),)
        ),
    )
    with pytest.raises(MeasurementUnavailableError) as root_error:
        missing_root.start()
    assert root_error.value.code == "process_root_unavailable"

    missing_worker = ProcessTreeRssSampler(
        root_pid=101,
        provider=_constant_process_provider(),
        managed_workers=(_worker_binding(),),
    )
    with pytest.raises(MeasurementUnavailableError) as worker_error:
        missing_worker.start()
    assert worker_error.value.code == "managed_worker_unavailable"


def test_process_sampler_fails_closed_on_an_invalid_or_failed_later_sample() -> None:
    class FailingProvider:
        identity = "failing-provider@1"

        def __init__(self) -> None:
            self.calls = 0
            self.failed = threading.Event()

        def snapshot(self) -> tuple[ProcessRecord, ...]:
            self.calls += 1
            if self.calls > 1:
                self.failed.set()
                raise RuntimeError("private process table failure")
            return (
                ProcessRecord(
                    101,
                    1,
                    100,
                    "root-start",
                    "benchmark-python@1",
                ),
            )

    provider = FailingProvider()
    sampler = ProcessTreeRssSampler(root_pid=101, provider=provider)
    sampler.start()
    assert provider.failed.wait(timeout=1)

    with pytest.raises(MeasurementError) as error:
        sampler.finish()
    assert error.value.code == "process_sample_failed"


def test_process_sampler_preserves_a_provider_measurement_failure_code() -> None:
    class UnavailableProvider:
        identity = "unavailable-native-provider@1"

        @staticmethod
        def snapshot() -> tuple[ProcessRecord, ...]:
            raise MeasurementUnavailableError(
                "native_process_record_unavailable"
            )

    sampler = ProcessTreeRssSampler(
        root_pid=101,
        provider=UnavailableProvider(),
    )

    with pytest.raises(MeasurementUnavailableError) as captured:
        sampler.start()

    assert captured.value.code == "native_process_record_unavailable"


def test_process_sampler_fails_closed_at_the_portable_sample_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(measurements_module, "MAX_MEASUREMENT_RSS_SAMPLES", 2)
    provider = ScriptedProcessProvider(
        (
            (ProcessRecord(101, 1, 100, "root", "benchmark-python@1"),),
            (ProcessRecord(101, 1, 110, "root", "benchmark-python@1"),),
            (ProcessRecord(101, 1, 120, "root", "benchmark-python@1"),),
        ),
        ready_after=3,
    )
    sampler = ProcessTreeRssSampler(root_pid=101, provider=provider)

    sampler.start()
    assert provider.ready.wait(timeout=1)
    with pytest.raises(MeasurementError) as error:
        sampler.finish()

    assert error.value.code == "process_sample_limit_exceeded"


def test_process_sampler_rejects_pid_reuse_and_worker_identity_drift() -> None:
    provider = ScriptedProcessProvider(
        (
            (
                ProcessRecord(101, 1, 100, "root-start", "benchmark-python@1"),
                ProcessRecord(102, 101, 25, "worker-start", "vision-worker@1"),
            ),
            (
                ProcessRecord(101, 1, 100, "root-start", "benchmark-python@1"),
                ProcessRecord(102, 101, 25, "reused-pid", "vision-worker@1"),
            ),
        ),
        ready_after=2,
    )
    sampler = ProcessTreeRssSampler(
        root_pid=101,
        provider=provider,
        managed_workers=(_worker_binding(),),
    )
    sampler.start()
    assert provider.ready.wait(timeout=1)

    with pytest.raises(MeasurementError) as error:
        sampler.finish()
    assert error.value.code == "managed_worker_identity_changed"


def test_storage_snapshot_counts_only_regular_files_beneath_declared_roots(
    tmp_path: Path,
) -> None:
    root = tmp_path / "contained"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (root / "one.bin").write_bytes(b"abc")
    (nested / "two.bin").write_bytes(b"12345")
    snapshotter = StorageSnapshotter()

    snapshot = snapshotter.snapshot(
        (
            DeclaredStorageRoot(
                root_id="scratch",
                path=root,
                purpose="benchmark_scratch",
            ),
        )
    )[0]

    assert snapshot.root_id == "scratch"
    assert snapshot.purpose == "benchmark_scratch"
    assert snapshot.file_count == 2
    assert snapshot.directory_count == 2
    assert snapshot.logical_bytes == 8
    assert snapshot.allocated_bytes >= 0
    assert len(snapshot.tree_digest) == 64
    assert str(tmp_path) not in snapshot.tree_digest


def test_storage_snapshot_rejects_symlink_roots_children_and_special_files(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "private.bin").write_bytes(b"private")
    root = tmp_path / "root"
    root.mkdir()
    snapshotter = StorageSnapshotter()

    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(root, target_is_directory=True)
    with pytest.raises(MeasurementError) as root_error:
        snapshotter.snapshot(
            (DeclaredStorageRoot("root", linked_root, "benchmark_scratch"),)
        )
    assert root_error.value.code == "unsafe_storage_root"

    (root / "linked-file").symlink_to(outside / "private.bin")
    with pytest.raises(MeasurementError) as child_error:
        snapshotter.snapshot(
            (DeclaredStorageRoot("root", root, "benchmark_scratch"),)
        )
    assert child_error.value.code == "unsafe_storage_entry"

    (root / "linked-file").unlink()
    os.mkfifo(root / "pipe")
    with pytest.raises(MeasurementError) as special_error:
        snapshotter.snapshot(
            (DeclaredStorageRoot("root", root, "benchmark_scratch"),)
        )
    assert special_error.value.code == "unsafe_storage_entry"


@pytest.mark.parametrize(
    ("limits", "expected_code"),
    (
        (StorageTraversalLimits(max_files=1), "storage_file_limit_exceeded"),
        (
            StorageTraversalLimits(max_logical_bytes=3),
            "storage_logical_bytes_limit_exceeded",
        ),
        (StorageTraversalLimits(max_depth=1), "storage_depth_limit_exceeded"),
    ),
)
def test_storage_snapshot_fails_closed_at_declared_bounds(
    tmp_path: Path,
    limits: StorageTraversalLimits,
    expected_code: str,
) -> None:
    root = tmp_path / "root"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (root / "one.bin").write_bytes(b"12")
    (nested / "two.bin").write_bytes(b"34")

    with pytest.raises(MeasurementError) as error:
        StorageSnapshotter(limits).snapshot(
            (DeclaredStorageRoot("root", root, "benchmark_scratch"),)
        )
    assert error.value.code == expected_code


def test_storage_directory_scan_stops_after_remaining_entry_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumed = 0

    class _Entry:
        def __init__(self, name: str) -> None:
            self.name = name

    class _Scan:
        def __enter__(self) -> _Scan:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def __iter__(self) -> _Scan:
            return self

        def __next__(self) -> _Entry:
            nonlocal consumed
            consumed += 1
            return _Entry(f"entry-{consumed}")

    monkeypatch.setattr(measurements_module.os, "scandir", lambda _fd: _Scan())

    with pytest.raises(MeasurementError) as captured:
        measurements_module._bounded_sorted_directory_names(
            123,
            limit=1,
            overflow_code="storage_entry_limit_exceeded",
            failure_code="storage_traversal_failed",
        )

    assert captured.value.code == "storage_entry_limit_exceeded"
    assert consumed == 2


def test_storage_snapshot_detects_concurrent_file_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from videoscope.benchmark import measurements

    root = tmp_path / "root"
    root.mkdir()
    target = root / "item.bin"
    target.write_bytes(b"before")
    real_stat = measurements.os.stat
    mutated = False

    def mutating_stat(path, *args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal mutated
        result = real_stat(path, *args, **kwargs)
        if path == "item.bin" and kwargs.get("dir_fd") is not None and not mutated:
            mutated = True
            target.write_bytes(b"after-with-a-different-size")
        return result

    monkeypatch.setattr(measurements.os, "stat", mutating_stat)

    with pytest.raises(MeasurementError) as error:
        StorageSnapshotter().snapshot(
            (DeclaredStorageRoot("root", root, "benchmark_scratch"),)
        )
    assert error.value.code == "storage_changed_during_snapshot"


def test_system_measurement_publishes_complete_memory_and_storage_metrics(
    tmp_path: Path,
) -> None:
    roots = _declared_roots(tmp_path)
    roots[0].path.joinpath("index.bin").write_bytes(b"index")
    session = SystemMeasurementFactory(
        storage_roots=roots,
        execution_mode="warm",
        cache_policy_identity="preserve-pinned-caches@1",
        process_provider=_constant_process_provider(rss_bytes=120),
        process_root_pid=101,
    ).open_session()

    session.start()
    roots[1].path.joinpath("temporary.bin").write_bytes(b"abc")
    evidence = session.finish()
    metrics = {
        metric.name: metric.value
        for metric in measurement_metrics_from_evidence(evidence)
    }
    session.close()

    assert evidence.rss_samples_bytes
    assert set(evidence.rss_samples_bytes) == {120}
    assert evidence.metal_telemetry_status == "unavailable"
    assert tuple(item.root_id for item in evidence.storage_before) == (
        "active-index",
        "benchmark-work",
    )
    assert tuple(item.root_id for item in evidence.storage_after) == (
        "active-index",
        "benchmark-work",
    )
    assert all(
        not hasattr(snapshot, field)
        for snapshot in (*evidence.storage_before, *evidence.storage_after)
        for field in ("path", "device", "inode")
    )
    assert metrics["sampled_process_tree_rss_baseline_bytes"] == 120
    assert metrics["sampled_peak_process_tree_rss_bytes"] == 120
    assert metrics["sampled_process_tree_rss_increment_bytes"] == 0
    assert metrics["process_tree_rss_sample_count"] >= 1
    assert metrics["active_immutable_artifact_logical_bytes"] == 5
    assert metrics["benchmark_scratch_logical_bytes_baseline"] == 0
    assert metrics["benchmark_scratch_logical_bytes_final"] == 3
    assert metrics["benchmark_scratch_logical_bytes_growth"] == 3
    assert not any("metal" in name for name in metrics)


def test_portable_measurement_evidence_is_bounded_and_requires_matching_roots() -> None:
    active = BenchmarkStorageSnapshot(
        root_id="active-index",
        purpose="active_immutable_artifacts",
        file_count=1,
        directory_count=1,
        logical_bytes=5,
        allocated_bytes=512,
        tree_digest="a" * 64,
    )
    scratch = BenchmarkStorageSnapshot(
        root_id="benchmark-work",
        purpose="benchmark_scratch",
        file_count=0,
        directory_count=1,
        logical_bytes=0,
        allocated_bytes=0,
        tree_digest="b" * 64,
    )

    with pytest.raises(BenchmarkDataError, match="RSS samples"):
        BenchmarkMeasurementEvidence(
            schema_version=1,
            rss_samples_bytes=(1,) * (MAX_MEASUREMENT_RSS_SAMPLES + 1),
            storage_before=(active, scratch),
            storage_after=(active, scratch),
            metal_telemetry_status="unavailable",
        )

    with pytest.raises(BenchmarkDataError, match="same roots"):
        BenchmarkMeasurementEvidence(
            schema_version=1,
            rss_samples_bytes=(100,),
            storage_before=(active, scratch),
            storage_after=(active,),
            metal_telemetry_status="unavailable",
        )

    with pytest.raises(BenchmarkDataError, match="Metal telemetry"):
        BenchmarkMeasurementEvidence(
            schema_version=1,
            rss_samples_bytes=(100,),
            storage_before=(active, scratch),
            storage_after=(active, scratch),
            metal_telemetry_status="complete",  # type: ignore[arg-type]
        )


def test_system_measurement_rejects_external_workers_and_active_artifact_drift(
    tmp_path: Path,
) -> None:
    roots = _declared_roots(tmp_path / "external")
    unavailable = SystemMeasurementFactory(
        storage_roots=roots,
        execution_mode="warm",
        cache_policy_identity="preserve-pinned-caches@1",
        process_provider=_constant_process_provider(),
        process_root_pid=101,
        external_worker_pids=(404,),
    ).open_session()
    with pytest.raises(MeasurementUnavailableError) as external_error:
        unavailable.start()
    assert external_error.value.code == "external_workers_unavailable"
    unavailable.close()

    roots = _declared_roots(tmp_path / "drift")
    active_file = roots[0].path / "index.bin"
    active_file.write_bytes(b"first")
    drift = SystemMeasurementFactory(
        storage_roots=roots,
        execution_mode="warm",
        cache_policy_identity="preserve-pinned-caches@1",
        process_provider=_constant_process_provider(),
        process_root_pid=101,
    ).open_session()
    drift.start()
    active_file.write_bytes(b"changed")
    with pytest.raises(MeasurementError) as drift_error:
        drift.finish()
    assert drift_error.value.code == "active_artifacts_changed"
    drift.close()


def test_system_measurement_has_no_implicit_process_provider(
    tmp_path: Path,
) -> None:
    session = SystemMeasurementFactory(
        storage_roots=_declared_roots(tmp_path),
        execution_mode="warm",
        cache_policy_identity="preserve-pinned-caches@1",
        process_root_pid=101,
    ).open_session()

    with pytest.raises(MeasurementUnavailableError) as error:
        session.start()
    assert error.value.code == "process_provider_unavailable"
    session.close()


@dataclass(frozen=True)
class _RepositoryAsset:
    id: str
    sha256: str
    byte_size: int
    duration_seconds: float
    video_id: str


class _Repository:
    def __init__(self, asset: BenchmarkAsset) -> None:
        self.asset = _RepositoryAsset(
            id=f"sha256:{asset.sha256}",
            sha256=asset.sha256,
            byte_size=asset.byte_size,
            duration_seconds=asset.duration_seconds,
            video_id="local-video",
        )

    def find_assets_by_sha256(self, digest: str) -> tuple[_RepositoryAsset, ...]:
        return (self.asset,) if digest == self.asset.sha256 else ()


class _SearchSession:
    def identities(self) -> ExecutionIdentities:
        return ExecutionIdentities(
            model_identities=(
                ComponentIdentity("text_embedding", _TEXT_MODEL_IDENTITY),
                ComponentIdentity("visual_embedding", _VISUAL_MODEL_IDENTITY),
            ),
            index_identities=(
                ComponentIdentity("text_vector_index", "2" * 64),
                ComponentIdentity("text_vector_generations", "sha256:" + "3" * 64),
                ComponentIdentity("visual_generations", "sha256:" + "4" * 64),
            ),
            config_identities=(
                ComponentIdentity(
                    "benchmark_product_environment",
                    "benchmark-product-environment@2:" + "5" * 64,
                ),
                ComponentIdentity(
                    "evaluation_search_configuration",
                    get_profile("dense_siglip").search_plan.identity.replace(
                        "evaluation-search-plan",
                        "evaluation-search-configuration",
                        1,
                    ),
                ),
                ComponentIdentity(
                    "product_search_lifecycle",
                    "warm:process-cache-preserved@1",
                ),
                ComponentIdentity("product_search_runtime", "sha256:" + "6" * 64),
            ),
        )

    def lifecycle_identity(self) -> ComponentIdentity:
        return ComponentIdentity(
            "benchmark_execution_lifecycle",
            "warm:process-cache-preserved@1",
        )

    def capability_state(self, asset, capability):  # type: ignore[no-untyped-def]
        del asset, capability
        return "complete"

    def search(self, query, assets, *, limit):  # type: ignore[no-untyped-def]
        del query, assets, limit
        return ()

    def close(self) -> None:
        return None


class _SearchAdapter:
    def open_session(  # type: ignore[no-untyped-def]
        self,
        profile,
        assets,
        *,
        execution_mode,
    ):
        del profile, assets
        assert execution_mode == "warm"
        return _SearchSession()

    def close(self) -> None:
        return None


class _MeasurementSession:
    def __init__(
        self,
        *,
        fail: bool = False,
        invalid_metrics: bool = False,
    ) -> None:
        self.fail = fail
        self.invalid_metrics = invalid_metrics
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def finish(self) -> BenchmarkMeasurementEvidence:
        if self.fail:
            raise MeasurementError("measurement_failed")
        if self.invalid_metrics:
            return ("partial-invalid-value",)  # type: ignore[return-value]
        return _portable_measurement_evidence()

    def close(self) -> None:
        self.closed = True


class _MeasurementFactory:
    execution_mode = "warm"
    cache_policy_identity = "process-cache-preserved@1"

    def __init__(self, session: _MeasurementSession) -> None:
        self.session = session

    def protocol_identity(self) -> ComponentIdentity:
        return ComponentIdentity(
            "benchmark_measurement_protocol",
            "test-process-tree-rss-50ms-contained-storage@1",
        )

    def open_session(self) -> _MeasurementSession:
        return self.session


def _dataset_and_repository() -> tuple[BenchmarkDataset, _Repository]:
    asset = BenchmarkAsset(
        asset_id="asset-a",
        sha256="a" * 64,
        byte_size=10,
        duration_seconds=10,
        provenance=AssetProvenance(
            source="Measurement fixture",
            license_id="CC0-1.0",
        ),
    )
    dataset = BenchmarkDataset(
        schema_version=1,
        dataset_id="measurement-runner",
        dataset_version="1.0.0",
        description="Measurement runner fixture",
        assets=(asset,),
        cases=(
            QueryCase(
                case_id="negative",
                query="nothing happens",
                asset_ids=(asset.asset_id,),
                domain="generic",
                modalities=("visual",),
                label_quality="gold",
                split_group="group-a",
            ),
        ),
    )
    return dataset, _Repository(asset)


def _measured_runner(
    tmp_path: Path,
    measurement: _MeasurementFactory,
) -> BenchmarkRunner:
    dataset, repository = _dataset_and_repository()
    del dataset
    timer_value = -0.001

    def timer() -> float:
        nonlocal timer_value
        timer_value += 0.001
        return timer_value

    return BenchmarkRunner(
        registry=BenchmarkRunRegistry(tmp_path / "runs"),
        asset_resolver=LocalAssetResolver(repository),
        search=_SearchAdapter(),
        hardware=HardwareProfile("macOS 15", "arm64", "Apple", 1024),
        code_sha="c" * 40,
        measurement=measurement,
        clock=lambda: datetime(2026, 8, 19, 12, 0, tzinfo=UTC),
        timer=timer,
    )


def test_runner_publishes_an_injected_complete_measurement_atomically(
    tmp_path: Path,
) -> None:
    dataset, _repository = _dataset_and_repository()
    session = _MeasurementSession()
    runner = _measured_runner(tmp_path, _MeasurementFactory(session))

    run = runner.run(
        dataset,
        profile_id="dense_siglip",
        run_id="measured",
        execution_mode="warm",
    )

    assert session.started is True
    assert session.closed is True
    assert run.measurement_status == "complete"
    assert run.measurement_evidence_status == "complete"
    assert run.measurement_evidence == _portable_measurement_evidence()
    assert run.measurement_started_at == "2026-08-19T12:00:00Z"
    assert run.measurement_finished_at == "2026-08-19T12:00:00Z"
    assert run.system_metrics == measurement_metrics_from_evidence(
        _portable_measurement_evidence()
    )
    assert runner.registry.read("measured") == run

    payload = run_to_dict(run)
    encoded = json.dumps(payload, sort_keys=True)
    assert '"path"' not in encoded
    assert '"device"' not in encoded
    assert '"inode"' not in encoded
    assert '"metal_telemetry_status": "unavailable"' in encoded
    loaded = run_from_dict(payload)
    assert loaded == run
    audit_run_manifest(dataset, loaded)


@pytest.mark.parametrize("tamper", ["rss", "storage"])
def test_audit_recomputes_system_metrics_and_rejects_raw_evidence_tampering(
    tmp_path: Path,
    tamper: str,
) -> None:
    dataset, _repository = _dataset_and_repository()
    runner = _measured_runner(
        tmp_path,
        _MeasurementFactory(_MeasurementSession()),
    )
    run = runner.run(
        dataset,
        profile_id="dense_siglip",
        run_id=f"tamper-{tamper}",
        execution_mode="warm",
    )
    payload = run_to_dict(run)
    raw = payload["measurement_evidence"]
    assert isinstance(raw, dict)
    if tamper == "rss":
        samples = raw["rss_samples_bytes"]
        assert isinstance(samples, list)
        samples[1] = 600
    else:
        after = raw["storage_after"]
        assert isinstance(after, list)
        scratch = after[1]
        assert isinstance(scratch, dict)
        scratch["logical_bytes"] = 4

    forged = run_from_dict(payload)
    with pytest.raises(BenchmarkExecutionError, match="system metrics mismatch"):
        audit_run_manifest(dataset, forged)


@pytest.mark.parametrize(
    ("field", "value"),
    (("path", "/private/index"), ("device", 42), ("inode", 84)),
)
def test_measurement_evidence_parser_rejects_local_storage_identity_fields(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    dataset, _repository = _dataset_and_repository()
    run = _measured_runner(
        tmp_path,
        _MeasurementFactory(_MeasurementSession()),
    ).run(
        dataset,
        profile_id="dense_siglip",
        run_id=f"private-{field}",
        execution_mode="warm",
    )
    payload = run_to_dict(run)
    raw = payload["measurement_evidence"]
    assert isinstance(raw, dict)
    before = raw["storage_before"]
    assert isinstance(before, list)
    snapshot = before[0]
    assert isinstance(snapshot, dict)
    snapshot[field] = value

    with pytest.raises(BenchmarkDataError, match="unexpected fields"):
        run_from_dict(payload)


def test_audit_rejects_legacy_aggregate_only_measurement_without_raw_evidence(
    tmp_path: Path,
) -> None:
    dataset, _repository = _dataset_and_repository()
    run = _measured_runner(
        tmp_path,
        _MeasurementFactory(_MeasurementSession()),
    ).run(
        dataset,
        profile_id="dense_siglip",
        run_id="legacy-measurement",
        execution_mode="warm",
    )
    legacy = run_to_dict(run)
    legacy["schema_version"] = 2
    legacy.pop("measurement_evidence_status")
    legacy.pop("measurement_evidence")
    migrated = run_from_dict(legacy)

    assert migrated.measurement_evidence_status == "legacy_unavailable"
    with pytest.raises(BenchmarkExecutionError, match="raw measurement evidence"):
        audit_run_manifest(dataset, migrated)


def test_runner_finishes_measurement_before_closing_the_pinned_search_session(
    tmp_path: Path,
) -> None:
    dataset, repository = _dataset_and_repository()
    events: list[str] = []

    class OrderedSearchSession(_SearchSession):
        def close(self) -> None:
            events.append("search_close")

    class OrderedSearchAdapter:
        def open_session(  # type: ignore[no-untyped-def]
            self,
            profile,
            assets,
            *,
            execution_mode,
        ):
            del profile, assets
            assert execution_mode == "warm"
            return OrderedSearchSession()

        def close(self) -> None:
            return None

    class OrderedMeasurementSession(_MeasurementSession):
        def finish(self) -> BenchmarkMeasurementEvidence:
            events.append("measurement_finish")
            return super().finish()

        def close(self) -> None:
            events.append("measurement_close")
            super().close()

    measurement_session = OrderedMeasurementSession()
    runner = BenchmarkRunner(
        registry=BenchmarkRunRegistry(tmp_path / "runs"),
        asset_resolver=LocalAssetResolver(repository),
        search=OrderedSearchAdapter(),  # type: ignore[arg-type]
        hardware=HardwareProfile("macOS 15", "arm64", "Apple", 1024),
        code_sha="c" * 40,
        measurement=_MeasurementFactory(measurement_session),
        clock=lambda: datetime(2026, 8, 19, 12, 0, tzinfo=UTC),
    )

    runner.run(
        dataset,
        profile_id="dense_siglip",
        run_id="measurement-before-search-close",
        execution_mode="warm",
    )

    assert events == [
        "measurement_finish",
        "search_close",
        "measurement_close",
    ]


def test_runner_persists_failed_measurement_without_partial_metrics(
    tmp_path: Path,
) -> None:
    dataset, _repository = _dataset_and_repository()
    session = _MeasurementSession(fail=True)
    runner = _measured_runner(tmp_path, _MeasurementFactory(session))

    run = runner.run(
        dataset,
        profile_id="dense_siglip",
        run_id="measurement-failed",
        execution_mode="warm",
    )

    assert session.closed is True
    assert run.run_status == "complete"
    assert run.measurement_status == "failed"
    assert run.measurement_evidence_status == "not_applicable"
    assert run.measurement_evidence is None
    assert run.system_metrics == ()
    assert run.measurement_protocol.identity == (
        "test-process-tree-rss-50ms-contained-storage@1"
    )


def test_runner_discards_all_metrics_when_the_measurement_contract_is_invalid(
    tmp_path: Path,
) -> None:
    dataset, _repository = _dataset_and_repository()
    session = _MeasurementSession(invalid_metrics=True)
    runner = _measured_runner(tmp_path, _MeasurementFactory(session))

    run = runner.run(
        dataset,
        profile_id="dense_siglip",
        run_id="measurement-invalid",
        execution_mode="warm",
    )

    assert session.closed is True
    assert run.measurement_status == "failed"
    assert run.measurement_evidence is None
    assert run.system_metrics == ()


def test_runner_discards_metrics_when_measurement_cleanup_fails(
    tmp_path: Path,
) -> None:
    dataset, _repository = _dataset_and_repository()

    class CloseFailureSession(_MeasurementSession):
        def close(self) -> None:
            self.closed = True
            raise RuntimeError("private cleanup failure")

    session = CloseFailureSession()
    runner = _measured_runner(tmp_path, _MeasurementFactory(session))

    run = runner.run(
        dataset,
        profile_id="dense_siglip",
        run_id="measurement-close-failed",
        execution_mode="warm",
    )

    assert session.closed is True
    assert run.measurement_status == "failed"
    assert run.measurement_evidence is None
    assert run.system_metrics == ()


def test_runner_rejects_a_lifecycle_identity_that_does_not_attest_the_mode(
    tmp_path: Path,
) -> None:
    dataset, repository = _dataset_and_repository()

    class BadLifecycleSession(_SearchSession):
        def lifecycle_identity(self) -> ComponentIdentity:
            return ComponentIdentity(
                "benchmark_execution_lifecycle",
                "cold:wrong-mode@1",
            )

    class BadLifecycleAdapter:
        def open_session(  # type: ignore[no-untyped-def]
            self,
            profile,
            assets,
            *,
            execution_mode,
        ):
            del profile, assets
            assert execution_mode == "warm"
            return BadLifecycleSession()

        def close(self) -> None:
            return None

    runner = BenchmarkRunner(
        registry=BenchmarkRunRegistry(tmp_path / "runs"),
        asset_resolver=LocalAssetResolver(repository),
        search=BadLifecycleAdapter(),  # type: ignore[arg-type]
        hardware=HardwareProfile("macOS 15", "arm64", "Apple", 1024),
        code_sha="c" * 40,
        clock=lambda: datetime(2026, 8, 19, 12, 0, tzinfo=UTC),
    )

    with pytest.raises(BenchmarkExecutionError, match="does not attest"):
        runner.run(
            dataset,
            profile_id="dense_siglip",
            run_id="bad-lifecycle",
            execution_mode="warm",
        )
    assert runner.registry.list() == ()


def test_runner_rejects_measurement_and_search_cache_policy_disagreement(
    tmp_path: Path,
) -> None:
    dataset, _repository = _dataset_and_repository()

    class MismatchedMeasurementFactory(_MeasurementFactory):
        cache_policy_identity = "different-cache-policy@1"

    session = _MeasurementSession()
    runner = _measured_runner(
        tmp_path,
        MismatchedMeasurementFactory(session),
    )

    with pytest.raises(BenchmarkExecutionError, match="policies disagree"):
        runner.run(
            dataset,
            profile_id="dense_siglip",
            run_id="policy-disagreement",
            execution_mode="warm",
        )
    assert session.started is True
    assert session.closed is True
    assert runner.registry.list() == ()
