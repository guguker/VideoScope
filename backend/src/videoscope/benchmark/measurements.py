from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
import threading
from typing import Iterable, Literal, Protocol

from .schema import (
    MEASUREMENT_PROTOCOL_COMPONENT_ID,
    BenchmarkDataError,
    ComponentIdentity,
    MetricValue,
    _require_id,
    _require_string,
)


PROCESS_RSS_SAMPLE_INTERVAL_SECONDS = 0.05
MEASUREMENT_PROTOCOL_IDENTITY_PREFIX = (
    "process-tree-rss-50ms-contained-storage@1"
)
MAX_PROCESS_RECORDS = 100_000
_MAX_RSS_BYTES = (1 << 63) - 1
_STORAGE_PURPOSES = frozenset(
    {"active_immutable_artifacts", "benchmark_scratch"}
)


class MeasurementError(RuntimeError):
    """A system measurement could not be completed without partial evidence."""

    def __init__(self, code: str) -> None:
        _require_id(code, "measurement error code")
        self.code = code
        super().__init__(code)


class MeasurementUnavailableError(MeasurementError):
    """The declared measurement boundary is unavailable in this runtime."""


def _bounded_sorted_directory_names(
    descriptor: int,
    *,
    limit: int,
    overflow_code: str,
    failure_code: str,
) -> tuple[str, ...]:
    if type(limit) is not int or limit < 0:
        raise MeasurementError(overflow_code)
    names: list[str] = []
    try:
        with os.scandir(descriptor) as entries:
            for entry in entries:
                names.append(entry.name)
                if len(names) > limit:
                    raise MeasurementError(overflow_code)
    except MeasurementError:
        raise
    except Exception:
        raise MeasurementError(failure_code) from None
    return tuple(sorted(names))


@dataclass(frozen=True, slots=True)
class ProcessRecord:
    pid: int
    parent_pid: int
    rss_bytes: int
    start_token: str
    executable_identity: str

    def __post_init__(self) -> None:
        if type(self.pid) is not int or self.pid <= 0:
            raise BenchmarkDataError("process pid must be a positive integer")
        if type(self.parent_pid) is not int or self.parent_pid < 0:
            raise BenchmarkDataError(
                "process parent_pid must be a non-negative integer"
            )
        if self.parent_pid == self.pid:
            raise BenchmarkDataError("a process must not be its own parent")
        if (
            type(self.rss_bytes) is not int
            or self.rss_bytes < 0
            or self.rss_bytes > _MAX_RSS_BYTES
        ):
            raise BenchmarkDataError(
                "process rss_bytes must be a bounded non-negative integer"
            )
        _require_string(self.start_token, "process start_token", max_length=512)
        _require_string(
            self.executable_identity,
            "process executable_identity",
            max_length=512,
        )


@dataclass(frozen=True, slots=True)
class ManagedProcessBinding:
    pid: int
    start_token: str
    executable_identity: str
    role: str

    def __post_init__(self) -> None:
        _require_pid(self.pid, "managed process pid")
        _require_string(
            self.start_token,
            "managed process start_token",
            max_length=512,
        )
        _require_string(
            self.executable_identity,
            "managed process executable_identity",
            max_length=512,
        )
        _require_id(self.role, "managed process role")


class ProcessSnapshotProvider(Protocol):
    """Pinned process-table source supplied by the concrete runner runtime.

    The base backend intentionally has no default implementation.  Sampling a
    subprocess-based ``ps`` command every 50 ms would materially perturb the
    benchmark, while the Python standard library exposes no stable macOS
    process-tree RSS primitive.  A future runtime may supply a pinned native
    provider with an auditable identity.
    """

    identity: str

    def snapshot(self) -> Iterable[ProcessRecord]: ...


@dataclass(frozen=True, slots=True)
class ProcessTreeRssMeasurement:
    baseline_bytes: int
    peak_bytes: int
    increment_bytes: int
    sample_count: int

    def __post_init__(self) -> None:
        for field_name in (
            "baseline_bytes",
            "peak_bytes",
            "increment_bytes",
            "sample_count",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise BenchmarkDataError(
                    f"process RSS {field_name} must be a non-negative integer"
                )
        if self.sample_count == 0:
            raise BenchmarkDataError("process RSS sample_count must be positive")
        if self.peak_bytes < self.baseline_bytes:
            raise BenchmarkDataError(
                "process RSS peak_bytes must be at least baseline_bytes"
            )
        if self.increment_bytes != self.peak_bytes - self.baseline_bytes:
            raise BenchmarkDataError(
                "process RSS increment_bytes must equal peak minus baseline"
            )


class ProcessTreeRssSampler:
    """Sample one managed process tree at the frozen 50 ms cadence."""

    def __init__(
        self,
        *,
        root_pid: int,
        provider: ProcessSnapshotProvider,
        managed_workers: tuple[ManagedProcessBinding, ...] = (),
        expected_provider_identity: str | None = None,
    ) -> None:
        _require_pid(root_pid, "process root pid")
        _validate_managed_workers(managed_workers)
        if root_pid in {worker.pid for worker in managed_workers}:
            raise BenchmarkDataError(
                "managed_workers must not contain the process root"
            )
        _validate_process_provider(provider)
        if (
            expected_provider_identity is not None
            and provider.identity != expected_provider_identity
        ):
            raise MeasurementError("process_provider_identity_changed")
        self._root_pid = root_pid
        self._provider = provider
        self._provider_identity = expected_provider_identity or provider.identity
        self._managed_workers = managed_workers
        self._root_identity: tuple[str, str] | None = None
        self._observed_identities: dict[int, tuple[str, str]] = {}
        self._samples: list[int] = []
        self._error: MeasurementError | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._started = False
        self._finished = False

    def start(self) -> None:
        if self._started or self._finished:
            raise MeasurementError("process_sampler_invalid_state")
        baseline = self._sample_once()
        with self._lock:
            self._samples.append(baseline)
        self._started = True
        thread = threading.Thread(
            target=self._sample_loop,
            name="videoscope-benchmark-rss",
            daemon=True,
        )
        self._thread = thread
        try:
            thread.start()
        except Exception:
            self._error = MeasurementError("process_sampler_start_failed")
            raise self._error from None

    def finish(self) -> ProcessTreeRssMeasurement:
        if not self._started or self._finished:
            raise MeasurementError("process_sampler_invalid_state")
        self._finished = True
        self._stop_and_join()
        with self._lock:
            error = self._error
            samples = tuple(self._samples)
        if error is not None:
            raise error
        if not samples:
            raise MeasurementError("process_samples_unavailable")
        baseline = samples[0]
        peak = max(samples)
        return ProcessTreeRssMeasurement(
            baseline_bytes=baseline,
            peak_bytes=peak,
            increment_bytes=peak - baseline,
            sample_count=len(samples),
        )

    def close(self) -> None:
        self._stop_and_join()

    def _sample_loop(self) -> None:
        while not self._stop.wait(PROCESS_RSS_SAMPLE_INTERVAL_SECONDS):
            try:
                sample = self._sample_once()
            except MeasurementError as exc:
                with self._lock:
                    self._error = exc
                self._stop.set()
                return
            except Exception:
                with self._lock:
                    self._error = MeasurementError("process_sample_failed")
                self._stop.set()
                return
            with self._lock:
                self._samples.append(sample)

    def _sample_once(self) -> int:
        if self._provider.identity != self._provider_identity:
            raise MeasurementError("process_provider_identity_changed")
        try:
            values = iter(self._provider.snapshot())
        except Exception:
            raise MeasurementError("process_sample_failed") from None
        records: list[ProcessRecord] = []
        try:
            for value in values:
                if len(records) >= MAX_PROCESS_RECORDS:
                    raise MeasurementError("process_record_limit_exceeded")
                if not isinstance(value, ProcessRecord):
                    raise MeasurementError("invalid_process_snapshot")
                records.append(value)
        except MeasurementError:
            raise
        except Exception:
            raise MeasurementError("process_sample_failed") from None

        by_pid: dict[int, ProcessRecord] = {}
        children: dict[int, list[int]] = {}
        for record in records:
            if record.pid in by_pid:
                raise MeasurementError("invalid_process_snapshot")
            by_pid[record.pid] = record
            children.setdefault(record.parent_pid, []).append(record.pid)
        if self._root_pid not in by_pid:
            raise MeasurementUnavailableError("process_root_unavailable")

        included: set[int] = set()
        pending = [self._root_pid]
        while pending:
            pid = pending.pop()
            if pid in included:
                continue
            included.add(pid)
            pending.extend(children.get(pid, ()))
        root_record = by_pid[self._root_pid]
        root_identity = (
            root_record.start_token,
            root_record.executable_identity,
        )
        if self._root_identity is None:
            self._root_identity = root_identity
        elif root_identity != self._root_identity:
            raise MeasurementError("process_root_identity_changed")

        for worker in self._managed_workers:
            record = by_pid.get(worker.pid)
            if record is None or worker.pid not in included:
                raise MeasurementUnavailableError("managed_worker_unavailable")
            if (
                record.start_token != worker.start_token
                or record.executable_identity != worker.executable_identity
            ):
                raise MeasurementError("managed_worker_identity_changed")

        for pid in included:
            record = by_pid[pid]
            identity = (record.start_token, record.executable_identity)
            previous = self._observed_identities.setdefault(pid, identity)
            if previous != identity:
                raise MeasurementError("process_identity_changed")

        total = sum(by_pid[pid].rss_bytes for pid in included)
        if total > _MAX_RSS_BYTES:
            raise MeasurementError("process_rss_overflow")
        return total

    def _stop_and_join(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is None or thread is threading.current_thread():
            return
        thread.join(timeout=1.0)
        if thread.is_alive():
            with self._lock:
                self._error = MeasurementError(
                    "process_sampler_shutdown_failed"
                )


@dataclass(frozen=True, slots=True)
class StorageTraversalLimits:
    max_roots: int = 16
    max_depth: int = 64
    max_entries: int = 100_000
    max_files: int = 100_000
    max_logical_bytes: int = 1 << 50
    max_allocated_bytes: int = 1 << 50

    def __post_init__(self) -> None:
        for field_name in (
            "max_roots",
            "max_depth",
            "max_entries",
            "max_files",
            "max_logical_bytes",
            "max_allocated_bytes",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value <= 0:
                raise BenchmarkDataError(
                    f"storage limit {field_name} must be a positive integer"
                )


@dataclass(frozen=True, slots=True)
class DeclaredStorageRoot:
    root_id: str
    path: Path
    purpose: Literal["active_immutable_artifacts", "benchmark_scratch"]

    def __post_init__(self) -> None:
        _require_id(self.root_id, "storage root id")
        try:
            path = Path(self.path)
        except TypeError as exc:
            raise BenchmarkDataError("storage root path must be path-like") from exc
        if not path.is_absolute() or path == Path(path.anchor):
            raise BenchmarkDataError(
                "storage root path must be an absolute non-filesystem-root path"
            )
        if ".." in path.parts or "\x00" in os.fspath(path):
            raise BenchmarkDataError(
                "storage root path must be lexically contained"
            )
        if self.purpose not in _STORAGE_PURPOSES:
            raise BenchmarkDataError("storage root purpose is invalid")
        object.__setattr__(self, "path", path)


@dataclass(frozen=True, slots=True)
class StorageRootSnapshot:
    root_id: str
    purpose: Literal["active_immutable_artifacts", "benchmark_scratch"]
    file_count: int
    directory_count: int
    logical_bytes: int
    allocated_bytes: int
    tree_digest: str
    device: int = field(repr=False)
    inode: int = field(repr=False)


@dataclass(slots=True)
class _StorageBudget:
    entries: int = 0
    files: int = 0
    logical_bytes: int = 0
    allocated_bytes: int = 0


@dataclass(frozen=True, slots=True)
class _TreeTotals:
    file_count: int
    directory_count: int
    logical_bytes: int
    allocated_bytes: int


class StorageSnapshotter:
    """Descriptor-relative, no-follow storage accounting for declared roots."""

    def __init__(
        self,
        limits: StorageTraversalLimits | None = None,
    ) -> None:
        self.limits = limits or StorageTraversalLimits()
        if not isinstance(self.limits, StorageTraversalLimits):
            raise BenchmarkDataError(
                "storage limits must be a StorageTraversalLimits value"
            )

    def snapshot(
        self,
        roots: tuple[DeclaredStorageRoot, ...],
    ) -> tuple[StorageRootSnapshot, ...]:
        _validate_storage_roots(roots, self.limits)
        budget = _StorageBudget()
        snapshots = tuple(
            self._snapshot_root(root, budget)
            for root in sorted(roots, key=lambda item: item.root_id)
        )
        identities = tuple((item.device, item.inode) for item in snapshots)
        if len(identities) != len(set(identities)):
            raise MeasurementError("overlapping_storage_roots")
        return snapshots

    def _snapshot_root(
        self,
        root: DeclaredStorageRoot,
        budget: _StorageBudget,
    ) -> StorageRootSnapshot:
        descriptor, opened = _open_directory_chain(root.path)
        hasher = sha256()
        try:
            totals = self._walk_directory(
                descriptor,
                relative_parts=(),
                depth=0,
                budget=budget,
                hasher=hasher,
            )
            final_opened = os.fstat(descriptor)
            if _stable_stat(opened) != _stable_stat(final_opened):
                raise MeasurementError("storage_changed_during_snapshot")
            verification_descriptor, verification = _open_directory_chain(root.path)
            try:
                if _stable_stat(opened) != _stable_stat(verification):
                    raise MeasurementError("storage_changed_during_snapshot")
            finally:
                os.close(verification_descriptor)
        finally:
            os.close(descriptor)
        return StorageRootSnapshot(
            root_id=root.root_id,
            purpose=root.purpose,
            file_count=totals.file_count,
            directory_count=totals.directory_count,
            logical_bytes=totals.logical_bytes,
            allocated_bytes=totals.allocated_bytes,
            tree_digest=hasher.hexdigest(),
            device=opened.st_dev,
            inode=opened.st_ino,
        )

    def _walk_directory(
        self,
        descriptor: int,
        *,
        relative_parts: tuple[str, ...],
        depth: int,
        budget: _StorageBudget,
        hasher,
    ) -> _TreeTotals:
        if depth > self.limits.max_depth:
            raise MeasurementError("storage_depth_limit_exceeded")
        before = os.fstat(descriptor)
        if not stat.S_ISDIR(before.st_mode):
            raise MeasurementError("unsafe_storage_entry")
        directory_allocated = _allocated_bytes(before)
        self._consume_budget(
            budget,
            files=0,
            logical_bytes=0,
            allocated_bytes=directory_allocated,
        )
        _update_tree_digest(hasher, "directory", relative_parts, before)
        names_before = _bounded_sorted_directory_names(
            descriptor,
            limit=self.limits.max_entries - budget.entries,
            overflow_code="storage_entry_limit_exceeded",
            failure_code="storage_traversal_failed",
        )

        file_count = 0
        directory_count = 1
        logical_bytes = 0
        allocated_bytes = directory_allocated
        for name in names_before:
            _validate_entry_name(name)
            child_parts = (*relative_parts, name)
            try:
                observed = os.stat(
                    name,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except Exception:
                raise MeasurementError("storage_changed_during_snapshot") from None
            if stat.S_ISLNK(observed.st_mode) or not (
                stat.S_ISDIR(observed.st_mode) or stat.S_ISREG(observed.st_mode)
            ):
                raise MeasurementError("unsafe_storage_entry")
            if stat.S_ISREG(observed.st_mode):
                child = self._snapshot_file(
                    descriptor,
                    name,
                    child_parts,
                    depth=depth + 1,
                    observed=observed,
                    budget=budget,
                    hasher=hasher,
                )
            else:
                child_descriptor = _open_child_directory(
                    descriptor,
                    name,
                    observed,
                )
                try:
                    child = self._walk_directory(
                        child_descriptor,
                        relative_parts=child_parts,
                        depth=depth + 1,
                        budget=budget,
                        hasher=hasher,
                    )
                    after_child = os.fstat(child_descriptor)
                    after_name = os.stat(
                        name,
                        dir_fd=descriptor,
                        follow_symlinks=False,
                    )
                    if not (
                        _stable_stat(observed)
                        == _stable_stat(after_child)
                        == _stable_stat(after_name)
                    ):
                        raise MeasurementError("storage_changed_during_snapshot")
                except MeasurementError:
                    raise
                except Exception:
                    raise MeasurementError(
                        "storage_changed_during_snapshot"
                    ) from None
                finally:
                    os.close(child_descriptor)
                directory_count += child.directory_count
            file_count += child.file_count
            logical_bytes += child.logical_bytes
            allocated_bytes += child.allocated_bytes

        names_after = _bounded_sorted_directory_names(
            descriptor,
            limit=len(names_before),
            overflow_code="storage_changed_during_snapshot",
            failure_code="storage_changed_during_snapshot",
        )
        try:
            after = os.fstat(descriptor)
        except Exception:
            raise MeasurementError("storage_changed_during_snapshot") from None
        if names_before != names_after or _stable_stat(before) != _stable_stat(after):
            raise MeasurementError("storage_changed_during_snapshot")
        return _TreeTotals(
            file_count=file_count,
            directory_count=directory_count,
            logical_bytes=logical_bytes,
            allocated_bytes=allocated_bytes,
        )

    def _snapshot_file(
        self,
        parent_descriptor: int,
        name: str,
        relative_parts: tuple[str, ...],
        *,
        depth: int,
        observed: os.stat_result,
        budget: _StorageBudget,
        hasher,
    ) -> _TreeTotals:
        if depth > self.limits.max_depth:
            raise MeasurementError("storage_depth_limit_exceeded")
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            descriptor = os.open(name, flags, dir_fd=parent_descriptor)
        except Exception:
            raise MeasurementError("storage_changed_during_snapshot") from None
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
                raise MeasurementError("unsafe_storage_entry")
            current = os.stat(
                name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            final_opened = os.fstat(descriptor)
            if not (
                _stable_stat(observed)
                == _stable_stat(opened)
                == _stable_stat(current)
                == _stable_stat(final_opened)
            ):
                raise MeasurementError("storage_changed_during_snapshot")
            allocated = _allocated_bytes(opened)
            self._consume_budget(
                budget,
                files=1,
                logical_bytes=opened.st_size,
                allocated_bytes=allocated,
            )
            _update_tree_digest(hasher, "file", relative_parts, opened)
            return _TreeTotals(1, 0, opened.st_size, allocated)
        except MeasurementError:
            raise
        except Exception:
            raise MeasurementError("storage_changed_during_snapshot") from None
        finally:
            os.close(descriptor)

    def _consume_budget(
        self,
        budget: _StorageBudget,
        *,
        files: int,
        logical_bytes: int,
        allocated_bytes: int,
    ) -> None:
        budget.entries += 1
        budget.files += files
        budget.logical_bytes += logical_bytes
        budget.allocated_bytes += allocated_bytes
        if budget.entries > self.limits.max_entries:
            raise MeasurementError("storage_entry_limit_exceeded")
        if budget.files > self.limits.max_files:
            raise MeasurementError("storage_file_limit_exceeded")
        if budget.logical_bytes > self.limits.max_logical_bytes:
            raise MeasurementError("storage_logical_bytes_limit_exceeded")
        if budget.allocated_bytes > self.limits.max_allocated_bytes:
            raise MeasurementError("storage_allocated_bytes_limit_exceeded")


class SystemMeasurementSession:
    """One all-or-nothing process and contained-storage measurement."""

    def __init__(
        self,
        *,
        storage_roots: tuple[DeclaredStorageRoot, ...],
        storage_limits: StorageTraversalLimits,
        process_provider: ProcessSnapshotProvider | None,
        process_root_pid: int,
        managed_workers: tuple[ManagedProcessBinding, ...],
        external_worker_pids: tuple[int, ...],
        expected_process_provider_identity: str | None,
    ) -> None:
        self._storage_roots = storage_roots
        self._snapshotter = StorageSnapshotter(storage_limits)
        self._process_provider = process_provider
        self._process_root_pid = process_root_pid
        self._managed_workers = managed_workers
        self._external_worker_pids = external_worker_pids
        self._expected_process_provider_identity = (
            expected_process_provider_identity
        )
        self._sampler: ProcessTreeRssSampler | None = None
        self._storage_before: tuple[StorageRootSnapshot, ...] | None = None
        self._started = False
        self._finished = False

    def start(self) -> None:
        if self._started or self._finished:
            raise MeasurementError("measurement_session_invalid_state")
        if self._external_worker_pids:
            raise MeasurementUnavailableError("external_workers_unavailable")
        if self._process_provider is None:
            raise MeasurementUnavailableError("process_provider_unavailable")
        storage_before = self._snapshotter.snapshot(self._storage_roots)
        sampler = ProcessTreeRssSampler(
            root_pid=self._process_root_pid,
            provider=self._process_provider,
            managed_workers=self._managed_workers,
            expected_provider_identity=self._expected_process_provider_identity,
        )
        try:
            sampler.start()
        except Exception:
            sampler.close()
            raise
        self._storage_before = storage_before
        self._sampler = sampler
        self._started = True

    def finish(self) -> tuple[MetricValue, ...]:
        if not self._started or self._finished:
            raise MeasurementError("measurement_session_invalid_state")
        self._finished = True
        sampler = self._sampler
        before = self._storage_before
        if sampler is None or before is None:
            raise MeasurementError("measurement_session_invalid_state")
        process = sampler.finish()
        after = self._snapshotter.snapshot(self._storage_roots)
        _require_stable_root_identities(before, after)
        before_active = tuple(
            item for item in before if item.purpose == "active_immutable_artifacts"
        )
        after_active = tuple(
            item for item in after if item.purpose == "active_immutable_artifacts"
        )
        if before_active != after_active:
            raise MeasurementError("active_artifacts_changed")
        return _measurement_metrics(process, before, after)

    def close(self) -> None:
        sampler = self._sampler
        if sampler is not None:
            sampler.close()


@dataclass(frozen=True, slots=True)
class SystemMeasurementFactory:
    storage_roots: tuple[DeclaredStorageRoot, ...]
    execution_mode: Literal["cold", "warm"]
    cache_policy_identity: str
    process_provider: ProcessSnapshotProvider | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    process_root_pid: int = field(default_factory=os.getpid)
    managed_workers: tuple[ManagedProcessBinding, ...] = ()
    external_worker_pids: tuple[int, ...] = ()
    storage_limits: StorageTraversalLimits = field(
        default_factory=StorageTraversalLimits
    )
    _process_provider_identity: str | None = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.storage_limits, StorageTraversalLimits):
            raise BenchmarkDataError(
                "storage_limits must be a StorageTraversalLimits value"
            )
        _validate_storage_roots(self.storage_roots, self.storage_limits)
        if self.execution_mode not in {"cold", "warm"}:
            raise BenchmarkDataError("measurement execution_mode must be cold or warm")
        _require_string(
            self.cache_policy_identity,
            "measurement cache_policy_identity",
            max_length=512,
        )
        _require_pid(self.process_root_pid, "process_root_pid")
        _validate_managed_workers(self.managed_workers)
        _validate_pid_tuple(self.external_worker_pids, "external_worker_pids")
        if self.process_root_pid in {
            *(worker.pid for worker in self.managed_workers),
            *self.external_worker_pids,
        }:
            raise BenchmarkDataError(
                "worker pids must not contain the process root"
            )
        if {worker.pid for worker in self.managed_workers} & set(
            self.external_worker_pids
        ):
            raise BenchmarkDataError(
                "managed and external worker pids must be disjoint"
            )
        if self.process_provider is not None:
            _validate_process_provider(self.process_provider)
            object.__setattr__(
                self,
                "_process_provider_identity",
                self.process_provider.identity,
            )
        else:
            object.__setattr__(self, "_process_provider_identity", None)

    def protocol_identity(self) -> ComponentIdentity:
        provider_identity = self._process_provider_identity or "unavailable"
        contract = {
            "allocated_byte_unit": 512,
            "external_worker_policy": "measurement-fails",
            "external_worker_count": len(self.external_worker_pids),
            "execution_mode": self.execution_mode,
            "cache_policy_identity": self.cache_policy_identity,
            "managed_descendants": "recursive-process-tree",
            "managed_workers": [
                {
                    "executable_identity": worker.executable_identity,
                    "role": worker.role,
                }
                for worker in sorted(
                    self.managed_workers,
                    key=lambda item: (item.role, item.executable_identity),
                )
            ],
            "metric_contract": "all-or-nothing-v1",
            "process_provider_identity": provider_identity,
            "rss_sample_interval_milliseconds": 50,
            "storage_limits": asdict(self.storage_limits),
            "storage_roots": [
                {"purpose": root.purpose, "root_id": root.root_id}
                for root in sorted(
                    self.storage_roots,
                    key=lambda item: (item.purpose, item.root_id),
                )
            ],
            "storage_semantics": (
                "nofollow-regular-files-active-immutable-scratch-delta-v1"
            ),
        }
        digest = sha256(
            json.dumps(
                contract,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("ascii")
        ).hexdigest()
        return ComponentIdentity(
            MEASUREMENT_PROTOCOL_COMPONENT_ID,
            f"{MEASUREMENT_PROTOCOL_IDENTITY_PREFIX}:{digest}",
        )

    def open_session(self) -> SystemMeasurementSession:
        return SystemMeasurementSession(
            storage_roots=self.storage_roots,
            storage_limits=self.storage_limits,
            process_provider=self.process_provider,
            process_root_pid=self.process_root_pid,
            managed_workers=self.managed_workers,
            external_worker_pids=self.external_worker_pids,
            expected_process_provider_identity=self._process_provider_identity,
        )


class BenchmarkMeasurementSession(Protocol):
    def start(self) -> None: ...

    def finish(self) -> tuple[MetricValue, ...]: ...

    def close(self) -> None: ...


class BenchmarkMeasurementFactory(Protocol):
    execution_mode: Literal["cold", "warm"]
    cache_policy_identity: str

    def protocol_identity(self) -> ComponentIdentity: ...

    def open_session(self) -> BenchmarkMeasurementSession: ...


def _measurement_metrics(
    process: ProcessTreeRssMeasurement,
    before: tuple[StorageRootSnapshot, ...],
    after: tuple[StorageRootSnapshot, ...],
) -> tuple[MetricValue, ...]:
    active = _aggregate_storage(
        item for item in before if item.purpose == "active_immutable_artifacts"
    )
    scratch_before = _aggregate_storage(
        item for item in before if item.purpose == "benchmark_scratch"
    )
    scratch_after = _aggregate_storage(
        item for item in after if item.purpose == "benchmark_scratch"
    )
    values = (
        MetricValue(
            "active_immutable_artifact_allocated_bytes",
            active.allocated_bytes,
            "bytes",
        ),
        MetricValue(
            "active_immutable_artifact_file_count",
            active.file_count,
            "count",
        ),
        MetricValue(
            "active_immutable_artifact_logical_bytes",
            active.logical_bytes,
            "bytes",
        ),
        MetricValue(
            "benchmark_scratch_allocated_bytes_baseline",
            scratch_before.allocated_bytes,
            "bytes",
        ),
        MetricValue(
            "benchmark_scratch_allocated_bytes_final",
            scratch_after.allocated_bytes,
            "bytes",
        ),
        MetricValue(
            "benchmark_scratch_allocated_bytes_growth",
            scratch_after.allocated_bytes - scratch_before.allocated_bytes,
            "bytes",
        ),
        MetricValue(
            "benchmark_scratch_file_count_baseline",
            scratch_before.file_count,
            "count",
        ),
        MetricValue(
            "benchmark_scratch_file_count_final",
            scratch_after.file_count,
            "count",
        ),
        MetricValue(
            "benchmark_scratch_file_count_growth",
            scratch_after.file_count - scratch_before.file_count,
            "count",
        ),
        MetricValue(
            "benchmark_scratch_logical_bytes_baseline",
            scratch_before.logical_bytes,
            "bytes",
        ),
        MetricValue(
            "benchmark_scratch_logical_bytes_final",
            scratch_after.logical_bytes,
            "bytes",
        ),
        MetricValue(
            "benchmark_scratch_logical_bytes_growth",
            scratch_after.logical_bytes - scratch_before.logical_bytes,
            "bytes",
        ),
        MetricValue(
            "process_tree_rss_sample_count",
            process.sample_count,
            "count",
        ),
        MetricValue(
            "sampled_peak_process_tree_rss_bytes",
            process.peak_bytes,
            "bytes",
        ),
        MetricValue(
            "sampled_process_tree_rss_baseline_bytes",
            process.baseline_bytes,
            "bytes",
        ),
        MetricValue(
            "sampled_process_tree_rss_increment_bytes",
            process.increment_bytes,
            "bytes",
        ),
    )
    return tuple(sorted(values, key=lambda item: item.name))


def _aggregate_storage(values: Iterable[StorageRootSnapshot]) -> _TreeTotals:
    file_count = 0
    directory_count = 0
    logical_bytes = 0
    allocated_bytes = 0
    for value in values:
        file_count += value.file_count
        directory_count += value.directory_count
        logical_bytes += value.logical_bytes
        allocated_bytes += value.allocated_bytes
    return _TreeTotals(
        file_count,
        directory_count,
        logical_bytes,
        allocated_bytes,
    )


def _require_stable_root_identities(
    before: tuple[StorageRootSnapshot, ...],
    after: tuple[StorageRootSnapshot, ...],
) -> None:
    before_identities = {
        item.root_id: (item.purpose, item.device, item.inode) for item in before
    }
    after_identities = {
        item.root_id: (item.purpose, item.device, item.inode) for item in after
    }
    if before_identities != after_identities:
        raise MeasurementError("storage_root_replaced")


def _validate_process_provider(provider: object) -> None:
    identity = getattr(provider, "identity", None)
    _require_string(identity, "process provider identity", max_length=512)
    if not callable(getattr(provider, "snapshot", None)):
        raise BenchmarkDataError("process provider snapshot must be callable")


def _require_pid(value: object, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise BenchmarkDataError(f"{field_name} must be a positive integer")
    return value


def _validate_pid_tuple(values: object, field_name: str) -> tuple[int, ...]:
    if not isinstance(values, tuple):
        raise BenchmarkDataError(f"{field_name} must be an immutable tuple")
    for value in values:
        _require_pid(value, field_name)
    if len(values) != len(set(values)):
        raise BenchmarkDataError(f"{field_name} must contain unique pids")
    return values


def _validate_managed_workers(values: object) -> tuple[ManagedProcessBinding, ...]:
    if not isinstance(values, tuple):
        raise BenchmarkDataError("managed_workers must be an immutable tuple")
    if any(not isinstance(value, ManagedProcessBinding) for value in values):
        raise BenchmarkDataError(
            "managed_workers must contain only ManagedProcessBinding values"
        )
    pids = tuple(value.pid for value in values)
    roles = tuple(value.role for value in values)
    if len(pids) != len(set(pids)):
        raise BenchmarkDataError("managed_workers must contain unique pids")
    if len(roles) != len(set(roles)):
        raise BenchmarkDataError("managed_workers must contain unique roles")
    return values


def _validate_storage_roots(
    roots: object,
    limits: StorageTraversalLimits,
) -> tuple[DeclaredStorageRoot, ...]:
    if not isinstance(roots, tuple):
        raise BenchmarkDataError("storage_roots must be an immutable tuple")
    if not roots:
        raise BenchmarkDataError("storage_roots must not be empty")
    if len(roots) > limits.max_roots:
        raise BenchmarkDataError("storage_roots exceeds max_roots")
    if any(not isinstance(root, DeclaredStorageRoot) for root in roots):
        raise BenchmarkDataError(
            "storage_roots must contain only DeclaredStorageRoot values"
        )
    root_ids = tuple(root.root_id for root in roots)
    if len(root_ids) != len(set(root_ids)):
        raise BenchmarkDataError("storage_roots must use unique root ids")
    ordered = sorted((root.path, root.root_id) for root in roots)
    for index, (path, _root_id) in enumerate(ordered):
        for other, _other_id in ordered[index + 1 :]:
            if _is_lexically_within(other, path):
                raise BenchmarkDataError("storage_roots must not overlap")
    return roots


def _is_lexically_within(candidate: Path, parent: Path) -> bool:
    try:
        candidate.relative_to(parent)
    except ValueError:
        return False
    return True


def _directory_flags() -> int:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if not nofollow or not directory:
        raise MeasurementUnavailableError("nofollow_storage_unavailable")
    return os.O_RDONLY | nofollow | directory | getattr(os, "O_CLOEXEC", 0)


def _open_directory_chain(path: Path) -> tuple[int, os.stat_result]:
    flags = _directory_flags()
    try:
        descriptor = os.open(path.anchor, flags)
    except Exception:
        raise MeasurementUnavailableError("storage_root_unavailable") from None
    try:
        for part in path.parts[1:]:
            try:
                observed = os.stat(
                    part,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except Exception:
                raise MeasurementUnavailableError(
                    "storage_root_unavailable"
                ) from None
            if not stat.S_ISDIR(observed.st_mode):
                raise MeasurementError("unsafe_storage_root")
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except Exception:
                raise MeasurementError("unsafe_storage_root") from None
            opened = os.fstat(child)
            current = os.stat(
                part,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
            if not (
                _stable_stat(observed)
                == _stable_stat(opened)
                == _stable_stat(current)
            ):
                os.close(child)
                raise MeasurementError("storage_changed_during_snapshot")
            os.close(descriptor)
            descriptor = child
        opened = os.fstat(descriptor)
        return descriptor, opened
    except Exception:
        os.close(descriptor)
        raise


def _open_child_directory(
    parent_descriptor: int,
    name: str,
    observed: os.stat_result,
) -> int:
    try:
        descriptor = os.open(
            name,
            _directory_flags(),
            dir_fd=parent_descriptor,
        )
        opened = os.fstat(descriptor)
        current = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not (
            _stable_stat(observed)
            == _stable_stat(opened)
            == _stable_stat(current)
        ):
            os.close(descriptor)
            raise MeasurementError("storage_changed_during_snapshot")
        return descriptor
    except MeasurementError:
        raise
    except Exception:
        raise MeasurementError("storage_changed_during_snapshot") from None


def _stable_stat(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        getattr(value, "st_blocks", -1),
    )


def _allocated_bytes(value: os.stat_result) -> int:
    blocks = getattr(value, "st_blocks", None)
    if type(blocks) is not int or blocks < 0:
        raise MeasurementUnavailableError("allocated_bytes_unavailable")
    allocated = blocks * 512
    if allocated > _MAX_RSS_BYTES:
        raise MeasurementError("storage_allocated_bytes_overflow")
    return allocated


def _validate_entry_name(name: object) -> str:
    if not isinstance(name, str) or not name or name in {".", ".."}:
        raise MeasurementError("unsafe_storage_entry")
    try:
        name.encode("utf-8", errors="strict")
    except UnicodeError:
        raise MeasurementError("unsafe_storage_entry") from None
    return name


def _update_tree_digest(
    hasher,
    entry_type: str,
    relative_parts: tuple[str, ...],
    value: os.stat_result,
) -> None:
    record = json.dumps(
        {
            "path": list(relative_parts),
            "stat": list(_stable_stat(value)),
            "type": entry_type,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8", errors="strict")
    hasher.update(len(record).to_bytes(8, "big"))
    hasher.update(record)


__all__ = [
    "MEASUREMENT_PROTOCOL_IDENTITY_PREFIX",
    "PROCESS_RSS_SAMPLE_INTERVAL_SECONDS",
    "BenchmarkMeasurementFactory",
    "BenchmarkMeasurementSession",
    "DeclaredStorageRoot",
    "ManagedProcessBinding",
    "MeasurementError",
    "MeasurementUnavailableError",
    "ProcessRecord",
    "ProcessSnapshotProvider",
    "ProcessTreeRssMeasurement",
    "ProcessTreeRssSampler",
    "StorageRootSnapshot",
    "StorageSnapshotter",
    "StorageTraversalLimits",
    "SystemMeasurementFactory",
    "SystemMeasurementSession",
]
