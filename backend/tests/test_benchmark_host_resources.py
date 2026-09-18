from __future__ import annotations

import os
import platform
import threading
import ctypes

import pytest

from videoscope.benchmark.host_resources import (
    HOST_RESOURCE_IDENTITY,
    HOST_RESOURCE_MEMORY_ACCOUNTING,
    HOST_RESOURCE_SCOPE,
    DarwinHostResourceSnapshotProvider,
    HostResourceError,
    HostResourceRawSample,
    HostResourceSampler,
    HostResourceSnapshot,
    HostResourceUnavailableError,
    create_host_resource_snapshot_provider,
)
from videoscope.benchmark import host_resources as host_resources_module


class _ScriptedDarwinApi:
    def __init__(
        self,
        *,
        accelerators: tuple[dict[str, object], ...],
        vm: dict[str, object],
    ) -> None:
        self.accelerators = accelerators
        self.vm = vm

    def accelerator_performance_statistics(
        self,
    ) -> tuple[dict[str, object], ...]:
        return self.accelerators

    def virtual_memory_statistics(self) -> dict[str, object]:
        return self.vm


def _snapshot(
    value: int,
    *,
    recovery_count: int = 10,
    swapins_pages: int = 20,
    swapouts_pages: int = 30,
) -> HostResourceSnapshot:
    return HostResourceSnapshot(
        metal_in_use_system_memory_bytes=value,
        metal_alloc_system_memory_bytes=value + 100,
        metal_recovery_count=recovery_count,
        swapins_pages=swapins_pages,
        swapouts_pages=swapouts_pages,
        page_size_bytes=16_384,
    )


def test_darwin_provider_aggregates_system_wide_accelerators_and_vm() -> None:
    provider = DarwinHostResourceSnapshotProvider(
        _api=_ScriptedDarwinApi(
            accelerators=(
                {
                    "In use system memory": 100,
                    "Alloc system memory": 200,
                    "recoveryCount": 3,
                },
                {
                    "In use system memory": 400,
                    "Alloc system memory": 800,
                    "recoveryCount": 5,
                },
            ),
            vm={
                "swapins": 7,
                "swapouts": 11,
                "page_size": 16_384,
            },
        )
    )

    assert provider.identity == HOST_RESOURCE_IDENTITY
    assert provider.scope == HOST_RESOURCE_SCOPE == "system_wide"
    assert provider.snapshot() == HostResourceSnapshot(
        metal_in_use_system_memory_bytes=500,
        metal_alloc_system_memory_bytes=1_000,
        metal_recovery_count=8,
        swapins_pages=7,
        swapouts_pages=11,
        page_size_bytes=16_384,
    )


def test_darwin_provider_and_receipt_preserve_independent_metal_gauges() -> None:
    # Observed on the macOS CI runner: in-use exceeds allocated memory.
    provider = DarwinHostResourceSnapshotProvider(
        _api=_ScriptedDarwinApi(
            accelerators=(
                {
                    "In use system memory": 35_163_136,
                    "Alloc system memory": 27_492_352,
                    "recoveryCount": 0,
                },
            ),
            vm={"swapins": 0, "swapouts": 0, "page_size": 16_384},
        )
    )

    snapshot = provider.snapshot()
    assert snapshot.metal_in_use_system_memory_bytes == 35_163_136
    assert snapshot.metal_alloc_system_memory_bytes == 27_492_352

    receipt = HostResourceSampler.receipt_from_samples(
        provider_identity=provider.identity,
        sample_interval_milliseconds=250,
        samples=(HostResourceRawSample(0, snapshot),),
    )
    payload = receipt.to_portable_dict()
    assert payload["raw_samples"][0]["metal"] == {
        "in_use_system_memory_bytes": 35_163_136,
        "alloc_system_memory_bytes": 27_492_352,
        "recovery_count": 0,
    }
    assert payload["metal"]["in_use_system_memory"] == {
        "baseline_bytes": 35_163_136,
        "peak_bytes": 35_163_136,
        "increment_bytes": 0,
    }
    assert payload["metal"]["alloc_system_memory"] == {
        "baseline_bytes": 27_492_352,
        "peak_bytes": 27_492_352,
        "increment_bytes": 0,
    }


@pytest.mark.parametrize(
    ("accelerators", "vm", "code"),
    (
        (
            (),
            {"swapins": 1, "swapouts": 2, "page_size": 16_384},
            "metal_accelerator_missing",
        ),
        (
            ({"Alloc system memory": 2, "recoveryCount": 3},),
            {"swapins": 1, "swapouts": 2, "page_size": 16_384},
            "metal_performance_statistic_missing",
        ),
        (
            (
                {
                    "In use system memory": True,
                    "Alloc system memory": 2,
                    "recoveryCount": 3,
                },
            ),
            {"swapins": 1, "swapouts": 2, "page_size": 16_384},
            "metal_performance_statistic_invalid",
        ),
        (
            (
                {
                    "In use system memory": 1,
                    "Alloc system memory": 2,
                    "recoveryCount": 3,
                },
            ),
            {"swapins": -1, "swapouts": 2, "page_size": 16_384},
            "vm_statistic_invalid",
        ),
    ),
)
def test_darwin_provider_fails_closed_on_missing_type_and_range(
    accelerators: tuple[dict[str, object], ...],
    vm: dict[str, object],
    code: str,
) -> None:
    provider = DarwinHostResourceSnapshotProvider(
        _api=_ScriptedDarwinApi(accelerators=accelerators, vm=vm)
    )

    with pytest.raises(HostResourceError) as captured:
        provider.snapshot()
    assert captured.value.code == code


def test_darwin_provider_fails_closed_on_aggregate_overflow() -> None:
    provider = DarwinHostResourceSnapshotProvider(
        _api=_ScriptedDarwinApi(
            accelerators=(
                {
                    "In use system memory": (1 << 63) - 1,
                    "Alloc system memory": 1,
                    "recoveryCount": 1,
                },
                {
                    "In use system memory": 1,
                    "Alloc system memory": 1,
                    "recoveryCount": 1,
                },
            ),
            vm={"swapins": 1, "swapouts": 2, "page_size": 16_384},
        )
    )

    with pytest.raises(HostResourceError) as captured:
        provider.snapshot()
    assert captured.value.code == "metal_performance_statistic_overflow"


@pytest.mark.parametrize(
    "changes",
    (
        {"metal_in_use_system_memory_bytes": -1},
        {"metal_alloc_system_memory_bytes": True},
        {"metal_recovery_count": 1 << 64},
        {"swapins_pages": -1},
        {"page_size_bytes": 12_345},
    ),
)
def test_snapshot_has_strict_bounded_integer_contract(
    changes: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "metal_in_use_system_memory_bytes": 1,
        "metal_alloc_system_memory_bytes": 2,
        "metal_recovery_count": 3,
        "swapins_pages": 4,
        "swapouts_pages": 5,
        "page_size_bytes": 16_384,
    }
    values.update(changes)

    with pytest.raises(ValueError):
        HostResourceSnapshot(**values)  # type: ignore[arg-type]


def test_receipt_recomputes_peaks_deltas_and_is_portable() -> None:
    samples = (
        HostResourceRawSample(elapsed_nanoseconds=0, snapshot=_snapshot(100)),
        HostResourceRawSample(
            elapsed_nanoseconds=250_000_000,
            snapshot=_snapshot(
                350,
                recovery_count=12,
                swapins_pages=23,
                swapouts_pages=34,
            ),
        ),
        HostResourceRawSample(
            elapsed_nanoseconds=500_000_000,
            snapshot=_snapshot(
                250,
                recovery_count=13,
                swapins_pages=25,
                swapouts_pages=37,
            ),
        ),
    )

    receipt = HostResourceSampler.receipt_from_samples(
        provider_identity=HOST_RESOURCE_IDENTITY,
        sample_interval_milliseconds=250,
        samples=samples,
    )

    assert receipt.scope == "system_wide"
    assert receipt.memory_accounting == HOST_RESOURCE_MEMORY_ACCOUNTING
    assert receipt.metal_in_use_baseline_bytes == 100
    assert receipt.metal_in_use_peak_bytes == 350
    assert receipt.metal_in_use_increment_bytes == 250
    assert receipt.metal_alloc_baseline_bytes == 200
    assert receipt.metal_alloc_peak_bytes == 450
    assert receipt.metal_alloc_increment_bytes == 250
    assert receipt.metal_recovery_delta == 3
    assert receipt.swapins_delta_pages == 5
    assert receipt.swapins_delta_bytes == 5 * 16_384
    assert receipt.swapouts_delta_pages == 7
    assert receipt.swapouts_delta_bytes == 7 * 16_384
    assert receipt.raw_samples == samples

    payload = receipt.to_portable_dict()
    assert payload["scope"] == "system_wide"
    assert payload["memory_accounting"] == (
        "metal_standalone_not_additive_with_process_rss"
    )
    serialized = repr(payload).lower()
    assert "pid" not in serialized
    assert "path" not in serialized
    assert "rss_bytes" not in serialized


@pytest.mark.parametrize(
    "samples",
    (
        (
            HostResourceRawSample(0, _snapshot(100, recovery_count=10)),
            HostResourceRawSample(1, _snapshot(100, recovery_count=9)),
        ),
        (
            HostResourceRawSample(0, _snapshot(100, swapins_pages=20)),
            HostResourceRawSample(1, _snapshot(100, swapins_pages=19)),
        ),
        (
            HostResourceRawSample(0, _snapshot(100, swapouts_pages=30)),
            HostResourceRawSample(1, _snapshot(100, swapouts_pages=29)),
        ),
    ),
)
def test_receipt_rejects_counter_reset_or_wrap(
    samples: tuple[HostResourceRawSample, ...],
) -> None:
    with pytest.raises(HostResourceError) as captured:
        HostResourceSampler.receipt_from_samples(
            provider_identity=HOST_RESOURCE_IDENTITY,
            sample_interval_milliseconds=250,
            samples=samples,
        )
    assert captured.value.code == "host_resource_counter_reset_or_wrap"


def test_receipt_rejects_page_size_drift_or_non_monotonic_time() -> None:
    changed_page_size = HostResourceSnapshot(
        metal_in_use_system_memory_bytes=100,
        metal_alloc_system_memory_bytes=200,
        metal_recovery_count=10,
        swapins_pages=20,
        swapouts_pages=30,
        page_size_bytes=4_096,
    )
    with pytest.raises(HostResourceError) as page_error:
        HostResourceSampler.receipt_from_samples(
            provider_identity=HOST_RESOURCE_IDENTITY,
            sample_interval_milliseconds=250,
            samples=(
                HostResourceRawSample(0, _snapshot(100)),
                HostResourceRawSample(1, changed_page_size),
            ),
        )
    assert page_error.value.code == "host_resource_page_size_changed"

    with pytest.raises(ValueError, match="strictly increasing"):
        HostResourceSampler.receipt_from_samples(
            provider_identity=HOST_RESOURCE_IDENTITY,
            sample_interval_milliseconds=250,
            samples=(
                HostResourceRawSample(0, _snapshot(100)),
                HostResourceRawSample(0, _snapshot(101)),
            ),
        )


class _ThreadedScriptedProvider:
    identity = "scripted-host-resources@1"
    scope = "system_wide"

    def __init__(self, *, ready_after: int, fail_after: int | None = None) -> None:
        self.calls = 0
        self.ready_after = ready_after
        self.fail_after = fail_after
        self.ready = threading.Event()
        self._lock = threading.Lock()

    def snapshot(self) -> HostResourceSnapshot:
        with self._lock:
            self.calls += 1
            if self.calls >= self.ready_after:
                self.ready.set()
            if self.fail_after is not None and self.calls >= self.fail_after:
                raise HostResourceError("scripted_native_failure")
            return _snapshot(
                self.calls * 100,
                recovery_count=10 + self.calls,
                swapins_pages=20 + self.calls,
                swapouts_pages=30 + self.calls,
            )


def test_threaded_sampler_captures_baseline_periodic_and_final_samples() -> None:
    provider = _ThreadedScriptedProvider(ready_after=3)
    sampler = HostResourceSampler(
        provider,
        sample_interval_seconds=0.005,
        max_samples=16,
    )

    sampler.start()
    assert provider.ready.wait(timeout=1)
    receipt = sampler.finish()

    assert receipt.provider_identity == "scripted-host-resources@1"
    assert 4 <= len(receipt.raw_samples) <= 16
    assert receipt.raw_samples[0].elapsed_nanoseconds == 0
    assert receipt.metal_in_use_baseline_bytes == 100
    assert receipt.metal_in_use_peak_bytes >= 400


def test_threaded_sampler_fails_closed_without_partial_receipt() -> None:
    provider = _ThreadedScriptedProvider(ready_after=2, fail_after=2)
    sampler = HostResourceSampler(
        provider,
        sample_interval_seconds=0.005,
        max_samples=16,
    )

    sampler.start()
    assert provider.ready.wait(timeout=1)
    with pytest.raises(HostResourceError) as captured:
        sampler.finish()
    assert captured.value.code == "scripted_native_failure"


def test_threaded_sampler_fails_closed_at_the_raw_sample_bound() -> None:
    provider = _ThreadedScriptedProvider(ready_after=3)
    sampler = HostResourceSampler(
        provider,
        sample_interval_seconds=0.005,
        max_samples=2,
    )

    sampler.start()
    assert provider.ready.wait(timeout=1)
    with pytest.raises(HostResourceError) as captured:
        sampler.finish()
    assert captured.value.code == "host_resource_sample_limit_exceeded"


def test_native_provider_is_unavailable_off_darwin() -> None:
    with pytest.raises(HostResourceUnavailableError) as captured:
        create_host_resource_snapshot_provider(operating_system="Linux")
    assert captured.value.code == "host_resource_provider_unavailable"


class _FakeMachSystem:
    def __init__(self, *, statistics_result: int = 0) -> None:
        self.statistics_result = statistics_result
        self.released: list[tuple[int, int]] = []

    def mach_host_self(self) -> int:
        return 41

    def mach_task_self(self) -> int:
        return 17

    def host_statistics64(
        self,
        _host: int,
        _selector: int,
        statistics: object,
        _count: object,
    ) -> int:
        typed = ctypes.cast(
            statistics,
            ctypes.POINTER(host_resources_module._VmStatistics64),
        )
        typed.contents.swapins = 7
        typed.contents.swapouts = 11
        return self.statistics_result

    def host_page_size(self, _host: int, page_size: object) -> int:
        ctypes.cast(page_size, ctypes.POINTER(ctypes.c_uint32)).contents.value = 16_384
        return 0

    def mach_port_deallocate(self, task: int, host: int) -> int:
        self.released.append((task, host))
        return 0


def test_native_vm_snapshot_releases_mach_host_send_right() -> None:
    system = _FakeMachSystem()
    api = host_resources_module._CtypesDarwinHostResourceApi.__new__(
        host_resources_module._CtypesDarwinHostResourceApi
    )
    api._system = system

    assert api.virtual_memory_statistics() == {
        "swapins": 7,
        "swapouts": 11,
        "page_size": 16_384,
    }
    assert system.released == [(17, 41)]


def test_native_vm_snapshot_releases_mach_host_on_statistics_failure() -> None:
    system = _FakeMachSystem(statistics_result=5)
    api = host_resources_module._CtypesDarwinHostResourceApi.__new__(
        host_resources_module._CtypesDarwinHostResourceApi
    )
    api._system = system

    with pytest.raises(HostResourceUnavailableError) as captured:
        api.virtual_memory_statistics()

    assert captured.value.code == "mach_vm_statistics_unavailable"
    assert system.released == [(17, 41)]


class _IteratorIokit:
    def __init__(self, *, result: int = 0, iterator: int = 77) -> None:
        self.result = result
        self.iterator = iterator
        self.releases: list[int] = []

    def IOServiceMatching(self, _name: bytes) -> int:
        return 1

    def IOServiceGetMatchingServices(
        self,
        _master_port: int,
        _matching: int,
        output: object,
    ) -> int:
        output._obj.value = self.iterator  # type: ignore[attr-defined]
        return self.result

    def IOObjectRelease(self, value: int) -> int:
        self.releases.append(value)
        return 0


def test_native_metal_snapshot_releases_iterator_when_cf_key_allocation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    iokit = _IteratorIokit()
    api = host_resources_module._CtypesDarwinHostResourceApi.__new__(
        host_resources_module._CtypesDarwinHostResourceApi
    )
    api._iokit = iokit
    monkeypatch.setattr(
        api,
        "_new_cf_string",
        lambda _value: (_ for _ in ()).throw(
            HostResourceUnavailableError("core_foundation_string_unavailable")
        ),
    )

    with pytest.raises(HostResourceUnavailableError):
        api.accelerator_performance_statistics()

    assert iokit.releases == [77]


def test_native_metal_snapshot_releases_nonzero_iterator_on_matching_failure() -> None:
    iokit = _IteratorIokit(result=1)
    api = host_resources_module._CtypesDarwinHostResourceApi.__new__(
        host_resources_module._CtypesDarwinHostResourceApi
    )
    api._iokit = iokit

    with pytest.raises(HostResourceUnavailableError):
        api.accelerator_performance_statistics()

    assert iokit.releases == [77]


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS IOKit contract")
def test_real_darwin_provider_reads_system_wide_metal_and_swap() -> None:
    provider = create_host_resource_snapshot_provider()

    snapshot = provider.snapshot()

    assert provider.scope == "system_wide"
    # The snapshot contract bounds each raw gauge independently.
    assert snapshot.metal_in_use_system_memory_bytes >= 0
    assert snapshot.metal_alloc_system_memory_bytes >= 0
    assert snapshot.metal_recovery_count >= 0
    assert snapshot.page_size_bytes == os.sysconf("SC_PAGE_SIZE")
    assert snapshot.swapins_pages >= 0
    assert snapshot.swapouts_pages >= 0
