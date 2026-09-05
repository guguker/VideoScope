"""Opt-in, bounded smoke diagnostics; no replacement for formal measurements."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from hashlib import sha256
import json
import re
import threading
from time import monotonic_ns

from .measurements import (
    MAX_PROCESS_RECORDS, ManagedProcessBinding, MeasurementError, ProcessRecord,
)


_MAX_INTEGER = (1 << 63) - 1
_MAX_EVENTS = 4096
_MAX_SAMPLES = 100_000
_EVENT_ID = re.compile(r"[a-z][a-z0-9_.-]{0,95}")
_ROLES = frozenset({"vision", "vision_index", "whisper", "ocr", "lighthouse", "qwen"})
_CODE_SHA = re.compile(r"[a-f0-9]{40}")


def _integer(value: object) -> bool:
    return type(value) is int and 0 <= value <= _MAX_INTEGER


def _identity(record: ProcessRecord) -> str:
    # No PID, absolute path or raw start token escapes the in-memory collector.
    value = [record.start_token, record.executable_identity]
    return sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


class SmokeTimeline:
    """Timestamp fixed events and partition one already-validated process tree.

    Roles are exclusive: a managed subtree stops at another managed root. The
    owner bucket contains the root and all remaining unmanaged descendants.
    Separate event and snapshot clocks permit concurrent sampling; neither is
    joined by nominal sample index or interpreted as per-process swap evidence.
    """

    def __init__(
        self,
        event_ids: frozenset[str],
        *,
        max_events: int = 256,
        max_samples: int = 10_000,
        clock_ns: Callable[[], int] = monotonic_ns,
    ) -> None:
        if (
            type(event_ids) is not frozenset or not event_ids
            or len(event_ids) > _MAX_EVENTS
            or any(type(value) is not str or _EVENT_ID.fullmatch(value) is None for value in event_ids)
            or type(max_events) is not int or not 1 <= max_events <= _MAX_EVENTS
            or type(max_samples) is not int or not 1 <= max_samples <= _MAX_SAMPLES
            or not callable(clock_ns)
        ):
            raise MeasurementError("smoke_timeline_configuration_invalid")
        try:
            epoch = clock_ns()
        except Exception:
            raise MeasurementError("smoke_timeline_clock_invalid") from None
        if not _integer(epoch):
            raise MeasurementError("smoke_timeline_clock_invalid")
        self._epoch_ns = epoch
        self._clock_ns = clock_ns
        self._event_ids = event_ids
        self._max_events = max_events
        self._max_samples = max_samples
        self._events: list[dict[str, object]] = []
        self._samples: list[dict[str, object]] = []
        self._host_epoch: int | None = None
        self._bindings: tuple[object, ...] | None = None
        self._failure_code: str | None = None
        self._lock = threading.RLock()

    def _fail(self, code: str) -> None:
        self._failure_code = self._failure_code or code
        raise MeasurementError(code)

    def _offset(self, value: object) -> int:
        if not _integer(value) or value < self._epoch_ns:
            self._fail("smoke_timeline_clock_invalid")
        return value - self._epoch_ns

    def record_event(self, event_id: str) -> None:
        with self._lock:
            if type(event_id) is not str or event_id not in self._event_ids:
                self._fail("smoke_timeline_event_invalid")
            if len(self._events) >= self._max_events:
                self._fail("smoke_timeline_event_limit_exceeded")
            try:
                value = self._clock_ns()
            except Exception:
                self._fail("smoke_timeline_clock_invalid")
            elapsed = self._offset(value)
            if self._events and elapsed < self._events[-1]["elapsed_nanoseconds"]:
                self._fail("smoke_timeline_clock_invalid")
            self._events.append({"id": event_id, "elapsed_nanoseconds": elapsed})

    def record_host_sampler_epoch(self, started_monotonic_ns: int) -> None:
        with self._lock:
            elapsed = self._offset(started_monotonic_ns)
            if self._host_epoch is not None and self._host_epoch != elapsed:
                self._fail("smoke_timeline_host_epoch_changed")
            self._host_epoch = elapsed

    def observe_process_snapshot(
        self,
        root_pid: int,
        managed_workers: tuple[ManagedProcessBinding, ...],
        records: tuple[ProcessRecord, ...],
        snapshot_started_ns: int,
        snapshot_finished_ns: int,
    ) -> None:
        with self._lock:
            if len(self._samples) >= self._max_samples:
                self._fail("smoke_timeline_sample_limit_exceeded")
            started = self._offset(snapshot_started_ns)
            finished = self._offset(snapshot_finished_ns)
            if finished < started or (
                self._samples and started < self._samples[-1]["snapshot_finished_elapsed_ns"]
            ):
                self._fail("smoke_timeline_clock_invalid")
            if (
                type(root_pid) is not int or root_pid <= 0
                or type(records) is not tuple or not 1 <= len(records) <= MAX_PROCESS_RECORDS
                or any(not isinstance(record, ProcessRecord) for record in records)
                or type(managed_workers) is not tuple or len(managed_workers) > len(_ROLES)
                or any(not isinstance(worker, ManagedProcessBinding) or worker.role not in _ROLES for worker in managed_workers)
            ):
                self._fail("smoke_timeline_process_contract_invalid")
            by_pid = {record.pid: record for record in records}
            workers = {worker.pid: worker for worker in managed_workers}
            if (
                len(by_pid) != len(records) or root_pid not in by_pid
                or len(workers) != len(managed_workers) or root_pid in workers
                or len({worker.role for worker in managed_workers}) != len(workers)
            ):
                self._fail("smoke_timeline_process_contract_invalid")
            root = by_pid[root_pid]
            bindings = (
                root_pid, root.start_token, root.executable_identity,
                tuple(sorted((w.role, w.pid, w.start_token, w.executable_identity) for w in managed_workers)),
            )
            if self._bindings is not None and self._bindings != bindings:
                self._fail("smoke_timeline_process_identity_changed")
            for worker in managed_workers:
                record = by_pid.get(worker.pid)
                if record is None or (
                    record.start_token != worker.start_token
                    or record.executable_identity != worker.executable_identity
                ):
                    self._fail("smoke_timeline_process_identity_changed")
            children: dict[int, list[int]] = {}
            for record in records:
                children.setdefault(record.parent_pid, []).append(record.pid)
            role_records = {"owner": root, **{w.role: by_pid[w.pid] for w in managed_workers}}
            totals = {role: 0 for role in role_records}
            counts = {role: 0 for role in role_records}
            pending = [(root_pid, "owner")]
            included: set[int] = set()
            while pending:
                pid, role = pending.pop()
                if pid in included:
                    self._fail("smoke_timeline_process_contract_invalid")
                included.add(pid)
                if pid in workers:
                    role = workers[pid].role
                totals[role] += by_pid[pid].rss_bytes
                counts[role] += 1
                pending.extend((child, role) for child in children.get(pid, ()))
            if not set(workers).issubset(included) or not _integer(sum(totals.values())):
                self._fail("smoke_timeline_process_contract_invalid")
            self._bindings = bindings
            self._samples.append({
                "snapshot_started_elapsed_ns": started,
                "snapshot_finished_elapsed_ns": finished,
                "total_rss_bytes": sum(totals.values()),
                "roles": [
                    {"role": role, "rss_bytes": totals[role], "process_count": counts[role],
                     "identity_sha256": _identity(role_records[role])}
                    for role in sorted(totals)
                ],
            })

    def to_portable_dict(self, *, code_sha: str, status: str = "complete") -> dict[str, object]:
        with self._lock:
            if (
                type(code_sha) is not str or _CODE_SHA.fullmatch(code_sha) is None
                or type(status) is not str or status not in {"complete", "failed"}
            ):
                raise MeasurementError("smoke_timeline_publication_invalid")
            if status == "complete" and (self._failure_code or self._host_epoch is None or not self._samples):
                raise MeasurementError("smoke_timeline_incomplete")
            return {
                "schema_version": 1,
                "provider_identity": "videoscope-smoke-timeline@1",
                "code_sha": code_sha,
                "status": status,
                "failure_code": self._failure_code,
                "attribution": "temporal_association_not_causality",
                "measurement_use": "diagnostic_only_not_gate_evidence",
                "clock": "monotonic_ns_relative_to_timeline_start",
                "host_sampler_started_elapsed_ns": self._host_epoch,
                "events": deepcopy(self._events),
                "process_samples": deepcopy(self._samples),
            }
