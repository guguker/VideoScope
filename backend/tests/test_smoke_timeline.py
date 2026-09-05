from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from videoscope.benchmark import measurements as measurement_module
from videoscope.benchmark.host_resources import HostResourceSampler, HostResourceSnapshot
from videoscope.benchmark.measurements import (
    ManagedProcessBinding, MeasurementError, MeasurementUnavailableError,
    ProcessRecord, ProcessTreeRssSampler,
)


def _records() -> tuple[ProcessRecord, ...]:
    return (
        ProcessRecord(101, 1, 100, "private-owner-token", "/private/owner/python"),
        ProcessRecord(102, 101, 200, "private-vision-token", "/private/vision/python"),
        ProcessRecord(103, 102, 30, "private-child-token", "/private/ffmpeg"),
        ProcessRecord(104, 101, 40, "private-other-token", "/private/ocr"),
        ProcessRecord(105, 101, 300, "private-qwen-token", "/private/qwen/python"),
        ProcessRecord(999, 1, 9999, "unrelated-token", "/private/unrelated"),
    )


def _workers() -> tuple[ManagedProcessBinding, ...]:
    return tuple(
        ManagedProcessBinding(item.pid, item.start_token, item.executable_identity, role)
        for item, role in ((_records()[1], "vision"), (_records()[4], "qwen"))
    )


class _Provider:
    identity = "synthetic-process-provider@1"

    def __init__(self, records=None):  # type: ignore[no-untyped-def]
        self.records = _records() if records is None else records
        self.calls = 0

    def snapshot(self):  # type: ignore[no-untyped-def]
        self.calls += 1
        return self.records


def _timeline(**kwargs):  # type: ignore[no-untyped-def]
    from videoscope.benchmark.smoke_timeline import SmokeTimeline

    return SmokeTimeline(
        frozenset({"workers.ready.begin", "workers.ready.end", "qwen.begin", "qwen.end"}),
        clock_ns=kwargs.pop("clock_ns", lambda: 1000), **kwargs,
    )


def test_process_observer_receives_one_validated_snapshot_and_clock_range(monkeypatch) -> None:
    clock = iter((1100, 1200))
    monkeypatch.setattr(measurement_module, "monotonic_ns", lambda: next(clock), raising=False)
    observed = []
    provider = _Provider()
    sampler = ProcessTreeRssSampler(
        root_pid=101, provider=provider, managed_workers=_workers(),
        diagnostic_observer=lambda *args: observed.append(args),
    )
    assert sampler._sample_once() == 670
    assert provider.calls == 1
    assert observed == [(101, _workers(), _records(), 1100, 1200)]


def test_process_sampler_without_observer_keeps_existing_work(monkeypatch) -> None:
    def forbidden_clock():
        raise AssertionError("diagnostic clock used without opt-in")

    monkeypatch.setattr(measurement_module, "monotonic_ns", forbidden_clock, raising=False)
    provider = _Provider()
    sampler = ProcessTreeRssSampler(root_pid=101, provider=provider, managed_workers=_workers())
    assert sampler._sample_once() == 670
    assert provider.calls == 1


@pytest.mark.parametrize("fault", ["missing", "pid_reuse", "executable_swap", "root_reuse"])
def test_invalid_tree_never_reaches_diagnostic_observer(fault: str) -> None:
    provider = _Provider()
    observed = []
    sampler = ProcessTreeRssSampler(
        root_pid=101, provider=provider, managed_workers=_workers(),
        diagnostic_observer=lambda *args: observed.append(args),
    )
    assert sampler._sample_once() == 670
    values = list(_records())
    if fault == "missing":
        values.pop(1)
    elif fault == "pid_reuse":
        values[1] = replace(values[1], start_token="reused")
    elif fault == "executable_swap":
        values[1] = replace(values[1], executable_identity="changed")
    else:
        values[0] = replace(values[0], start_token="reused")
    provider.records = tuple(values)
    with pytest.raises(MeasurementError):
        sampler._sample_once()
    assert len(observed) == 1


def test_observer_failure_is_typed_and_sanitized() -> None:
    def broken(*_args):
        raise RuntimeError("/private/user-data secret token")

    sampler = ProcessTreeRssSampler(root_pid=101, provider=_Provider(), diagnostic_observer=broken)
    with pytest.raises(MeasurementError, match="^process_diagnostic_observer_failed$"):
        sampler._sample_once()


def test_observer_must_be_callable_and_attached_before_start() -> None:
    sampler = ProcessTreeRssSampler(root_pid=101, provider=_Provider())
    with pytest.raises(MeasurementUnavailableError, match="observer_unavailable"):
        sampler.attach_diagnostic_observer("not a callback")
    sampler.attach_diagnostic_observer(lambda *_args: None)
    sampler.start()
    try:
        with pytest.raises(MeasurementError, match="observer_invalid_state"):
            sampler.attach_diagnostic_observer(lambda *_args: None)
    finally:
        sampler.close()


@pytest.mark.parametrize("clock_values", [(1200, 1100), (True, 1200), (1100, float("nan"))])
def test_observer_clock_ranges_are_finite_monotonic_integers(monkeypatch, clock_values) -> None:
    values = iter(clock_values)
    monkeypatch.setattr(measurement_module, "monotonic_ns", lambda: next(values), raising=False)
    sampler = ProcessTreeRssSampler(root_pid=101, provider=_Provider(), diagnostic_observer=lambda *_args: None)
    with pytest.raises(MeasurementError, match="diagnostic_clock_invalid"):
        sampler._sample_once()


def test_host_sampler_exposes_existing_epoch_without_extra_snapshot() -> None:
    calls = []
    provider = SimpleNamespace(
        identity="synthetic-host@1", scope="system_wide",
        snapshot=lambda: (calls.append(True) or HostResourceSnapshot(1, 2, 0, 0, 0, 16384)),
    )
    sampler = HostResourceSampler(provider, sample_interval_seconds=60.0)
    assert sampler.started_monotonic_ns is None
    sampler.start()
    started = sampler.started_monotonic_ns
    assert type(started) is int
    assert calls == [True]
    with pytest.raises(AttributeError):
        sampler.started_monotonic_ns = 0
    receipt = sampler.finish()
    assert sampler.started_monotonic_ns == started
    assert len(receipt.raw_samples) == 2
    assert calls == [True, True]


def test_timeline_aligns_role_rss_and_keeps_private_identity_out_of_payload() -> None:
    values = iter((1000, 1010, 1090))
    timeline = _timeline(clock_ns=lambda: next(values))
    timeline.record_event("workers.ready.begin")
    timeline.record_host_sampler_epoch(1020)
    timeline.observe_process_snapshot(101, _workers(), _records(), 1030, 1050)
    timeline.record_event("workers.ready.end")
    result = timeline.to_portable_dict(code_sha="a" * 40, status="complete")
    assert result["host_sampler_started_elapsed_ns"] == 20
    assert result["events"] == [
        {"id": "workers.ready.begin", "elapsed_nanoseconds": 10},
        {"id": "workers.ready.end", "elapsed_nanoseconds": 90},
    ]
    sample = result["process_samples"][0]
    assert sample["snapshot_started_elapsed_ns"] == 30
    assert sample["snapshot_finished_elapsed_ns"] == 50
    assert sample["total_rss_bytes"] == 670
    roles = {item["role"]: item for item in sample["roles"]}
    assert {role: item["rss_bytes"] for role, item in roles.items()} == {"owner": 140, "vision": 230, "qwen": 300}
    assert {role: item["process_count"] for role, item in roles.items()} == {"owner": 2, "vision": 2, "qwen": 1}
    assert all(len(item["identity_sha256"]) == 64 for item in roles.values())
    encoded = json.dumps(result)
    for forbidden in ("/private", "private-owner-token", "private-vision-token", "parent_pid", '"pid"', "unrelated-token"):
        assert forbidden not in encoded
    assert result["attribution"] == "temporal_association_not_causality"
    result["process_samples"].clear()
    assert len(timeline.to_portable_dict(code_sha="a" * 40, status="complete")["process_samples"]) == 1


@pytest.mark.parametrize("start,end", [(900, 1100), (1200, 1100), (True, 1200), (1100, 2**63)])
def test_timeline_rejects_invalid_ranges_without_unsanitized_values(start, end) -> None:
    timeline = _timeline()
    with pytest.raises(MeasurementError):
        timeline.observe_process_snapshot(101, _workers(), _records(), start, end)
    result = timeline.to_portable_dict(code_sha="a" * 40, status="failed")
    assert result["process_samples"] == []
    assert result["failure_code"] == "smoke_timeline_clock_invalid"


def test_timeline_rejects_backward_events_and_overlapping_snapshot_ranges() -> None:
    values = iter((1000, 1100, 1050))
    timeline = _timeline(clock_ns=lambda: next(values))
    timeline.record_event("qwen.begin")
    with pytest.raises(MeasurementError, match="clock_invalid"):
        timeline.record_event("qwen.end")
    fresh = _timeline()
    fresh.observe_process_snapshot(101, _workers(), _records(), 1100, 1200)
    with pytest.raises(MeasurementError, match="clock_invalid"):
        fresh.observe_process_snapshot(101, _workers(), _records(), 1199, 1300)


@pytest.mark.parametrize("bound", ["events", "samples"])
def test_timeline_bound_failure_preserves_partial_data_and_cannot_be_complete(bound: str) -> None:
    timeline = _timeline(max_events=1, max_samples=1)
    timeline.record_host_sampler_epoch(1000)
    timeline.record_event("qwen.begin")
    timeline.observe_process_snapshot(101, _workers(), _records(), 1000, 1001)
    with pytest.raises(MeasurementError, match="limit_exceeded"):
        if bound == "events":
            timeline.record_event("qwen.end")
        else:
            timeline.observe_process_snapshot(101, _workers(), _records(), 1001, 1002)
    with pytest.raises(MeasurementError):
        timeline.to_portable_dict(code_sha="a" * 40, status="complete")
    partial = timeline.to_portable_dict(code_sha="a" * 40, status="failed")
    assert len(partial["events"]) == len(partial["process_samples"]) == 1


def test_timeline_rejects_unknown_event_and_private_allowlist() -> None:
    from videoscope.benchmark.smoke_timeline import SmokeTimeline

    with pytest.raises(MeasurementError):
        SmokeTimeline(frozenset({"/private/user/query"}))
    timeline = _timeline()
    with pytest.raises(MeasurementError, match="event_invalid"):
        timeline.record_event("unreviewed-user-value")
    assert "unreviewed-user-value" not in json.dumps(timeline.to_portable_dict(code_sha="a" * 40, status="failed"))


@pytest.mark.parametrize("fault", ["pid_reuse", "executable_swap", "root_reuse", "unknown_role"])
def test_timeline_independently_rejects_identity_or_role_drift(fault: str) -> None:
    timeline = _timeline()
    timeline.observe_process_snapshot(101, _workers(), _records(), 1100, 1200)
    records = list(_records())
    workers = _workers()
    if fault == "pid_reuse":
        records[1] = replace(records[1], start_token="reused")
    elif fault == "executable_swap":
        records[1] = replace(records[1], executable_identity="new executable")
    elif fault == "root_reuse":
        records[0] = replace(records[0], start_token="reused")
    else:
        workers = (replace(workers[0], role="private-user-name"), workers[1])
    with pytest.raises(MeasurementError):
        timeline.observe_process_snapshot(101, workers, tuple(records), 1300, 1400)
    assert len(timeline.to_portable_dict(code_sha="a" * 40, status="failed")["process_samples"]) == 1


def test_timeline_handles_concurrent_events_and_samples_without_serializing_raw_data() -> None:
    timeline = _timeline(max_events=100)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(timeline.record_event, "qwen.begin") for _ in range(40)]
        futures += [executor.submit(timeline.observe_process_snapshot, 101, _workers(), _records(), 1100, 1200)]
        for future in futures:
            future.result()
    timeline.record_host_sampler_epoch(1000)
    result = timeline.to_portable_dict(code_sha="a" * 40, status="complete")
    assert len(result["events"]) == 40
    assert len(result["process_samples"]) == 1


@pytest.mark.parametrize("code_sha,status", [("/private/source", "failed"), ("a" * 40, []), ("a" * 40, "/private/error")])
def test_timeline_publication_rejects_nonportable_identity_or_status(code_sha, status) -> None:
    with pytest.raises(MeasurementError, match="publication_invalid"):
        _timeline().to_portable_dict(code_sha=code_sha, status=status)


def test_timeline_partition_preserves_unmanaged_pid_reuse_and_nested_worker_counts() -> None:
    timeline = _timeline()
    records = list(_records())
    records[4] = replace(records[4], parent_pid=102)
    timeline.observe_process_snapshot(101, _workers(), tuple(records), 1100, 1200)
    records[2] = replace(records[2], start_token="reused-child", executable_identity="different-child")
    timeline.observe_process_snapshot(101, _workers(), tuple(records), 1200, 1300)
    timeline.record_host_sampler_epoch(1000)
    result = timeline.to_portable_dict(code_sha="a" * 40, status="complete")
    assert [sample["total_rss_bytes"] for sample in result["process_samples"]] == [670, 670]
    for sample in result["process_samples"]:
        assert sum(role["rss_bytes"] for role in sample["roles"]) == sample["total_rss_bytes"]
        assert {role["role"]: role["rss_bytes"] for role in sample["roles"]} == {"owner": 140, "vision": 230, "qwen": 300}
