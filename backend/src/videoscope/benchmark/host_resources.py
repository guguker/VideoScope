"""Low-overhead, system-wide host resource evidence for macOS benchmarks.

This module deliberately keeps Metal accounting separate from process RSS.
Apple silicon uses unified memory, so adding the two gauges would double count
some allocations.  The portable receipt therefore carries an explicit
non-additive accounting contract and contains no process or filesystem identity.
"""

from __future__ import annotations

from collections.abc import Mapping
import ctypes
import ctypes.util
from dataclasses import dataclass
import platform
import re
import threading
import time
from typing import Literal, Protocol


HOST_RESOURCE_SCOPE: Literal["system_wide"] = "system_wide"
HOST_RESOURCE_IDENTITY = (
    "darwin-iokit-ioaccelerator-performance-statistics-mach-vm-statistics64@1"
)
HOST_RESOURCE_MEMORY_ACCOUNTING = (
    "metal_standalone_not_additive_with_process_rss"
)
HOST_RESOURCE_SAMPLE_INTERVAL_SECONDS = 0.250
HOST_RESOURCE_RECEIPT_SCHEMA_VERSION = 1
MAX_HOST_RESOURCE_RAW_SAMPLES = 100_000

_MAX_UINT64 = (1 << 64) - 1
_MAX_SINT64 = (1 << 63) - 1
_MAX_ELAPSED_NANOSECONDS = (1 << 63) - 1
_MAX_ACCELERATORS = 32
_MIN_PAGE_SIZE_BYTES = 4_096
_MAX_PAGE_SIZE_BYTES = 1 << 30
_MAX_SAMPLE_INTERVAL_MILLISECONDS = 60_000
_IDENTITY_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.:+@-]{0,255}")

_KERN_SUCCESS = 0
_HOST_VM_INFO64 = 4
_K_CF_STRING_ENCODING_UTF8 = 0x08000100
_K_CF_NUMBER_SINT64_TYPE = 4

_METAL_IN_USE_KEY = "In use system memory"
_METAL_ALLOC_KEY = "Alloc system memory"
_METAL_RECOVERY_KEY = "recoveryCount"


class HostResourceError(RuntimeError):
    """Host resource evidence could not be completed without ambiguity."""

    def __init__(self, code: str) -> None:
        if type(code) is not str or _IDENTITY_PATTERN.fullmatch(code) is None:
            raise ValueError("host resource error code must be a portable id")
        self.code = code
        super().__init__(code)


class HostResourceUnavailableError(HostResourceError):
    """The native host telemetry boundary is unavailable on this machine."""


def _require_bounded_integer(
    value: object,
    field: str,
    *,
    minimum: int = 0,
    maximum: int = _MAX_UINT64,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(
            f"{field} must be an integer in [{minimum}, {maximum}]"
        )
    return value


def _require_identity(value: object, field: str) -> str:
    if type(value) is not str or _IDENTITY_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a portable identity")
    return value


def _require_page_size(value: object) -> int:
    page_size = _require_bounded_integer(
        value,
        "page_size_bytes",
        minimum=_MIN_PAGE_SIZE_BYTES,
        maximum=_MAX_PAGE_SIZE_BYTES,
    )
    if page_size & (page_size - 1):
        raise ValueError("page_size_bytes must be a power of two")
    return page_size


@dataclass(frozen=True, slots=True)
class HostResourceSnapshot:
    """One simultaneous system-wide Metal and VM counter observation."""

    metal_in_use_system_memory_bytes: int
    metal_alloc_system_memory_bytes: int
    metal_recovery_count: int
    swapins_pages: int
    swapouts_pages: int
    page_size_bytes: int

    def __post_init__(self) -> None:
        _require_bounded_integer(
            self.metal_in_use_system_memory_bytes,
            "metal_in_use_system_memory_bytes",
            maximum=_MAX_SINT64,
        )
        _require_bounded_integer(
            self.metal_alloc_system_memory_bytes,
            "metal_alloc_system_memory_bytes",
            maximum=_MAX_SINT64,
        )
        _require_bounded_integer(
            self.metal_recovery_count,
            "metal_recovery_count",
        )
        _require_bounded_integer(self.swapins_pages, "swapins_pages")
        _require_bounded_integer(self.swapouts_pages, "swapouts_pages")
        _require_page_size(self.page_size_bytes)


@dataclass(frozen=True, slots=True)
class HostResourceRawSample:
    """Portable raw observation at a monotonic offset from sampler start."""

    elapsed_nanoseconds: int
    snapshot: HostResourceSnapshot

    def __post_init__(self) -> None:
        _require_bounded_integer(
            self.elapsed_nanoseconds,
            "elapsed_nanoseconds",
            maximum=_MAX_ELAPSED_NANOSECONDS,
        )
        if not isinstance(self.snapshot, HostResourceSnapshot):
            raise ValueError("snapshot must be a HostResourceSnapshot")

    def to_portable_dict(self) -> dict[str, object]:
        snapshot = self.snapshot
        return {
            "elapsed_nanoseconds": self.elapsed_nanoseconds,
            "metal": {
                "in_use_system_memory_bytes": (
                    snapshot.metal_in_use_system_memory_bytes
                ),
                "alloc_system_memory_bytes": (
                    snapshot.metal_alloc_system_memory_bytes
                ),
                "recovery_count": snapshot.metal_recovery_count,
            },
            "virtual_memory": {
                "swapins_pages": snapshot.swapins_pages,
                "swapouts_pages": snapshot.swapouts_pages,
                "page_size_bytes": snapshot.page_size_bytes,
            },
        }


@dataclass(frozen=True, slots=True)
class HostResourceMeasurementReceipt:
    """Self-validating portable aggregate plus the bounded raw observations."""

    schema_version: int
    provider_identity: str
    scope: Literal["system_wide"]
    memory_accounting: Literal[
        "metal_standalone_not_additive_with_process_rss"
    ]
    sample_interval_milliseconds: int
    raw_samples: tuple[HostResourceRawSample, ...]
    metal_in_use_baseline_bytes: int
    metal_in_use_peak_bytes: int
    metal_in_use_increment_bytes: int
    metal_alloc_baseline_bytes: int
    metal_alloc_peak_bytes: int
    metal_alloc_increment_bytes: int
    metal_recovery_delta: int
    swapins_delta_pages: int
    swapins_delta_bytes: int
    swapouts_delta_pages: int
    swapouts_delta_bytes: int

    def __post_init__(self) -> None:
        if self.schema_version != HOST_RESOURCE_RECEIPT_SCHEMA_VERSION:
            raise ValueError("unsupported host resource receipt schema")
        _require_identity(self.provider_identity, "provider_identity")
        if self.scope != HOST_RESOURCE_SCOPE:
            raise ValueError("host resource scope must be system_wide")
        if self.memory_accounting != HOST_RESOURCE_MEMORY_ACCOUNTING:
            raise ValueError("host resource memory accounting contract is invalid")
        _require_bounded_integer(
            self.sample_interval_milliseconds,
            "sample_interval_milliseconds",
            minimum=1,
            maximum=_MAX_SAMPLE_INTERVAL_MILLISECONDS,
        )
        if type(self.raw_samples) is not tuple or not self.raw_samples:
            raise ValueError("host resource raw_samples must be a non-empty tuple")
        if len(self.raw_samples) > MAX_HOST_RESOURCE_RAW_SAMPLES:
            raise ValueError("host resource raw_samples exceed the portable bound")
        for sample in self.raw_samples:
            if not isinstance(sample, HostResourceRawSample):
                raise ValueError("host resource raw_samples contain an invalid value")
        derived = _derive_measurement(self.raw_samples)
        for field_name, expected in derived.items():
            actual = getattr(self, field_name)
            _require_bounded_integer(actual, field_name)
            if actual != expected:
                raise ValueError(
                    f"{field_name} does not match the raw host resource samples"
                )

    def to_portable_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "provider_identity": self.provider_identity,
            "scope": self.scope,
            "memory_accounting": self.memory_accounting,
            "sample_interval_milliseconds": self.sample_interval_milliseconds,
            "raw_samples": [
                sample.to_portable_dict() for sample in self.raw_samples
            ],
            "metal": {
                "in_use_system_memory": {
                    "baseline_bytes": self.metal_in_use_baseline_bytes,
                    "peak_bytes": self.metal_in_use_peak_bytes,
                    "increment_bytes": self.metal_in_use_increment_bytes,
                },
                "alloc_system_memory": {
                    "baseline_bytes": self.metal_alloc_baseline_bytes,
                    "peak_bytes": self.metal_alloc_peak_bytes,
                    "increment_bytes": self.metal_alloc_increment_bytes,
                },
                "recovery_delta": self.metal_recovery_delta,
            },
            "virtual_memory": {
                "swapins_delta_pages": self.swapins_delta_pages,
                "swapins_delta_bytes": self.swapins_delta_bytes,
                "swapouts_delta_pages": self.swapouts_delta_pages,
                "swapouts_delta_bytes": self.swapouts_delta_bytes,
                "page_size_bytes": self.raw_samples[0].snapshot.page_size_bytes,
            },
        }


class HostResourceSnapshotProvider(Protocol):
    """Native, non-subprocess system-wide snapshot source."""

    identity: str
    scope: Literal["system_wide"]

    def snapshot(self) -> HostResourceSnapshot: ...


class _DarwinHostResourceApi(Protocol):
    def accelerator_performance_statistics(
        self,
    ) -> tuple[Mapping[str, object], ...]: ...

    def virtual_memory_statistics(self) -> Mapping[str, object]: ...


class _VmStatistics64(ctypes.Structure):
    """ABI layout of Darwin's ``vm_statistics64_data_t``."""

    _fields_ = (
        ("free_count", ctypes.c_uint32),
        ("active_count", ctypes.c_uint32),
        ("inactive_count", ctypes.c_uint32),
        ("wire_count", ctypes.c_uint32),
        ("zero_fill_count", ctypes.c_uint64),
        ("reactivations", ctypes.c_uint64),
        ("pageins", ctypes.c_uint64),
        ("pageouts", ctypes.c_uint64),
        ("faults", ctypes.c_uint64),
        ("cow_faults", ctypes.c_uint64),
        ("lookups", ctypes.c_uint64),
        ("hits", ctypes.c_uint64),
        ("purges", ctypes.c_uint64),
        ("purgeable_count", ctypes.c_uint32),
        ("speculative_count", ctypes.c_uint32),
        ("decompressions", ctypes.c_uint64),
        ("compressions", ctypes.c_uint64),
        ("swapins", ctypes.c_uint64),
        ("swapouts", ctypes.c_uint64),
        ("compressor_page_count", ctypes.c_uint32),
        ("throttled_count", ctypes.c_uint32),
        ("external_page_count", ctypes.c_uint32),
        ("internal_page_count", ctypes.c_uint32),
        ("total_uncompressed_pages_in_compressor", ctypes.c_uint64),
    )


class _CtypesDarwinHostResourceApi:
    """Minimal IOKit/CoreFoundation/Mach binding with no child process."""

    def __init__(self) -> None:
        try:
            self._iokit = ctypes.CDLL(
                "/System/Library/Frameworks/IOKit.framework/IOKit"
            )
            self._core_foundation = ctypes.CDLL(
                "/System/Library/Frameworks/CoreFoundation.framework/"
                "CoreFoundation"
            )
            system_library = (
                ctypes.util.find_library("System")
                or "/usr/lib/libSystem.B.dylib"
            )
            self._system = ctypes.CDLL(system_library)
            self._configure_signatures()
        except (AttributeError, OSError):
            raise HostResourceUnavailableError(
                "host_resource_provider_unavailable"
            ) from None

    def _configure_signatures(self) -> None:
        self._iokit.IOServiceMatching.argtypes = (ctypes.c_char_p,)
        self._iokit.IOServiceMatching.restype = ctypes.c_void_p
        self._iokit.IOServiceGetMatchingServices.argtypes = (
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
        )
        self._iokit.IOServiceGetMatchingServices.restype = ctypes.c_int
        self._iokit.IOIteratorNext.argtypes = (ctypes.c_uint32,)
        self._iokit.IOIteratorNext.restype = ctypes.c_uint32
        self._iokit.IORegistryEntryCreateCFProperty.argtypes = (
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
        )
        self._iokit.IORegistryEntryCreateCFProperty.restype = ctypes.c_void_p
        self._iokit.IOObjectRelease.argtypes = (ctypes.c_uint32,)
        self._iokit.IOObjectRelease.restype = ctypes.c_int

        self._core_foundation.CFStringCreateWithCString.argtypes = (
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_uint32,
        )
        self._core_foundation.CFStringCreateWithCString.restype = ctypes.c_void_p
        self._core_foundation.CFGetTypeID.argtypes = (ctypes.c_void_p,)
        self._core_foundation.CFGetTypeID.restype = ctypes.c_ulong
        self._core_foundation.CFDictionaryGetTypeID.argtypes = ()
        self._core_foundation.CFDictionaryGetTypeID.restype = ctypes.c_ulong
        self._core_foundation.CFNumberGetTypeID.argtypes = ()
        self._core_foundation.CFNumberGetTypeID.restype = ctypes.c_ulong
        self._core_foundation.CFDictionaryGetValue.argtypes = (
            ctypes.c_void_p,
            ctypes.c_void_p,
        )
        self._core_foundation.CFDictionaryGetValue.restype = ctypes.c_void_p
        self._core_foundation.CFNumberIsFloatType.argtypes = (ctypes.c_void_p,)
        self._core_foundation.CFNumberIsFloatType.restype = ctypes.c_bool
        self._core_foundation.CFNumberGetValue.argtypes = (
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        )
        self._core_foundation.CFNumberGetValue.restype = ctypes.c_bool
        self._core_foundation.CFRelease.argtypes = (ctypes.c_void_p,)
        self._core_foundation.CFRelease.restype = None

        self._system.mach_host_self.argtypes = ()
        self._system.mach_host_self.restype = ctypes.c_uint32
        self._system.mach_task_self.argtypes = ()
        self._system.mach_task_self.restype = ctypes.c_uint32
        self._system.mach_port_deallocate.argtypes = (
            ctypes.c_uint32,
            ctypes.c_uint32,
        )
        self._system.mach_port_deallocate.restype = ctypes.c_int
        self._system.host_statistics64.argtypes = (
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_uint32),
        )
        self._system.host_statistics64.restype = ctypes.c_int
        self._system.host_page_size.argtypes = (
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
        )
        self._system.host_page_size.restype = ctypes.c_int

    def _new_cf_string(self, value: str) -> int:
        reference = self._core_foundation.CFStringCreateWithCString(
            None,
            value.encode("utf-8"),
            _K_CF_STRING_ENCODING_UTF8,
        )
        if not reference:
            raise HostResourceUnavailableError(
                "core_foundation_string_unavailable"
            )
        return int(reference)

    def _read_dictionary_integer(
        self,
        dictionary: int,
        key: str,
    ) -> int:
        key_reference = self._new_cf_string(key)
        try:
            value_reference = self._core_foundation.CFDictionaryGetValue(
                dictionary,
                key_reference,
            )
        finally:
            self._core_foundation.CFRelease(key_reference)
        if not value_reference:
            raise HostResourceError("metal_performance_statistic_missing")
        if (
            self._core_foundation.CFGetTypeID(value_reference)
            != self._core_foundation.CFNumberGetTypeID()
            or self._core_foundation.CFNumberIsFloatType(value_reference)
        ):
            raise HostResourceError("metal_performance_statistic_invalid")
        output = ctypes.c_int64()
        converted = self._core_foundation.CFNumberGetValue(
            value_reference,
            _K_CF_NUMBER_SINT64_TYPE,
            ctypes.byref(output),
        )
        if not converted or output.value < 0:
            raise HostResourceError("metal_performance_statistic_invalid")
        return output.value

    def accelerator_performance_statistics(
        self,
    ) -> tuple[Mapping[str, object], ...]:
        matching = self._iokit.IOServiceMatching(b"IOAccelerator")
        if not matching:
            raise HostResourceUnavailableError("metal_accelerator_missing")
        iterator = ctypes.c_uint32()
        result = self._iokit.IOServiceGetMatchingServices(
            0,
            matching,
            ctypes.byref(iterator),
        )
        if result != _KERN_SUCCESS or iterator.value == 0:
            if iterator.value != 0:
                self._iokit.IOObjectRelease(iterator.value)
            raise HostResourceUnavailableError("metal_accelerator_missing")
        statistics_key: int | None = None
        rows: list[Mapping[str, object]] = []
        try:
            statistics_key = self._new_cf_string("PerformanceStatistics")
            while True:
                service = self._iokit.IOIteratorNext(iterator.value)
                if service == 0:
                    break
                try:
                    properties = self._iokit.IORegistryEntryCreateCFProperty(
                        service,
                        statistics_key,
                        None,
                        0,
                    )
                    if not properties:
                        raise HostResourceError(
                            "metal_performance_statistics_missing"
                        )
                    try:
                        if (
                            self._core_foundation.CFGetTypeID(properties)
                            != self._core_foundation.CFDictionaryGetTypeID()
                        ):
                            raise HostResourceError(
                                "metal_performance_statistics_invalid"
                            )
                        rows.append(
                            {
                                _METAL_IN_USE_KEY: self._read_dictionary_integer(
                                    properties, _METAL_IN_USE_KEY
                                ),
                                _METAL_ALLOC_KEY: self._read_dictionary_integer(
                                    properties, _METAL_ALLOC_KEY
                                ),
                                _METAL_RECOVERY_KEY: (
                                    self._read_dictionary_integer(
                                        properties, _METAL_RECOVERY_KEY
                                    )
                                ),
                            }
                        )
                    finally:
                        self._core_foundation.CFRelease(properties)
                finally:
                    self._iokit.IOObjectRelease(service)
                if len(rows) > _MAX_ACCELERATORS:
                    raise HostResourceError("metal_accelerator_limit_exceeded")
        finally:
            if statistics_key is not None:
                self._core_foundation.CFRelease(statistics_key)
            self._iokit.IOObjectRelease(iterator.value)
        if not rows:
            raise HostResourceUnavailableError("metal_accelerator_missing")
        return tuple(rows)

    def virtual_memory_statistics(self) -> Mapping[str, object]:
        host = self._system.mach_host_self()
        if host == 0:
            raise HostResourceUnavailableError("mach_host_unavailable")
        try:
            statistics = _VmStatistics64()
            expected_count = ctypes.sizeof(statistics) // ctypes.sizeof(
                ctypes.c_int32
            )
            count = ctypes.c_uint32(expected_count)
            result = self._system.host_statistics64(
                host,
                _HOST_VM_INFO64,
                ctypes.cast(
                    ctypes.byref(statistics),
                    ctypes.POINTER(ctypes.c_int32),
                ),
                ctypes.byref(count),
            )
            if result != _KERN_SUCCESS or count.value != expected_count:
                raise HostResourceUnavailableError(
                    "mach_vm_statistics_unavailable"
                )
            page_size = ctypes.c_uint32()
            result = self._system.host_page_size(host, ctypes.byref(page_size))
            if result != _KERN_SUCCESS:
                raise HostResourceUnavailableError("mach_page_size_unavailable")
            return {
                "swapins": int(statistics.swapins),
                "swapouts": int(statistics.swapouts),
                "page_size": int(page_size.value),
            }
        finally:
            task = self._system.mach_task_self()
            if (
                task == 0
                or self._system.mach_port_deallocate(task, host) != _KERN_SUCCESS
            ):
                raise HostResourceError("mach_host_port_release_failed")


class DarwinHostResourceSnapshotProvider:
    """System-wide Metal and VM snapshots backed only by native macOS APIs."""

    identity = HOST_RESOURCE_IDENTITY
    scope: Literal["system_wide"] = HOST_RESOURCE_SCOPE

    def __init__(self, *, _api: _DarwinHostResourceApi | None = None) -> None:
        self._api = _api or _CtypesDarwinHostResourceApi()

    def snapshot(self) -> HostResourceSnapshot:
        accelerators = self._api.accelerator_performance_statistics()
        if type(accelerators) is not tuple or not accelerators:
            raise HostResourceUnavailableError("metal_accelerator_missing")
        if len(accelerators) > _MAX_ACCELERATORS:
            raise HostResourceError("metal_accelerator_limit_exceeded")
        totals = {
            _METAL_IN_USE_KEY: 0,
            _METAL_ALLOC_KEY: 0,
            _METAL_RECOVERY_KEY: 0,
        }
        for row in accelerators:
            if not isinstance(row, Mapping):
                raise HostResourceError("metal_performance_statistics_invalid")
            for key in totals:
                if key not in row:
                    raise HostResourceError(
                        "metal_performance_statistic_missing"
                    )
                value = row[key]
                if type(value) is not int or not 0 <= value <= _MAX_SINT64:
                    raise HostResourceError(
                        "metal_performance_statistic_invalid"
                    )
                new_total = totals[key] + value
                if new_total > _MAX_SINT64:
                    raise HostResourceError(
                        "metal_performance_statistic_overflow"
                    )
                totals[key] = new_total

        vm = self._api.virtual_memory_statistics()
        if not isinstance(vm, Mapping):
            raise HostResourceError("vm_statistics_invalid")
        vm_values: dict[str, int] = {}
        for key in ("swapins", "swapouts", "page_size"):
            if key not in vm:
                raise HostResourceError("vm_statistic_missing")
            value = vm[key]
            if type(value) is not int or not 0 <= value <= _MAX_UINT64:
                raise HostResourceError("vm_statistic_invalid")
            vm_values[key] = value
        try:
            _require_page_size(vm_values["page_size"])
        except ValueError:
            raise HostResourceError("vm_statistic_invalid") from None
        return HostResourceSnapshot(
            metal_in_use_system_memory_bytes=totals[_METAL_IN_USE_KEY],
            metal_alloc_system_memory_bytes=totals[_METAL_ALLOC_KEY],
            metal_recovery_count=totals[_METAL_RECOVERY_KEY],
            swapins_pages=vm_values["swapins"],
            swapouts_pages=vm_values["swapouts"],
            page_size_bytes=vm_values["page_size"],
        )


def create_host_resource_snapshot_provider(
    *,
    operating_system: str | None = None,
) -> HostResourceSnapshotProvider:
    """Create the native provider, explicitly refusing non-Darwin platforms."""

    system_name = platform.system() if operating_system is None else operating_system
    if system_name != "Darwin":
        raise HostResourceUnavailableError("host_resource_provider_unavailable")
    return DarwinHostResourceSnapshotProvider()


def _derive_measurement(
    samples: tuple[HostResourceRawSample, ...],
) -> dict[str, int]:
    if not samples:
        raise ValueError("host resource samples must not be empty")
    if samples[0].elapsed_nanoseconds != 0:
        raise ValueError("the baseline sample must have zero elapsed time")
    previous = samples[0]
    page_size = previous.snapshot.page_size_bytes
    for sample in samples[1:]:
        if sample.elapsed_nanoseconds <= previous.elapsed_nanoseconds:
            raise ValueError(
                "host resource sample times must be strictly increasing"
            )
        current_snapshot = sample.snapshot
        previous_snapshot = previous.snapshot
        if current_snapshot.page_size_bytes != page_size:
            raise HostResourceError("host_resource_page_size_changed")
        if (
            current_snapshot.metal_recovery_count
            < previous_snapshot.metal_recovery_count
            or current_snapshot.swapins_pages < previous_snapshot.swapins_pages
            or current_snapshot.swapouts_pages < previous_snapshot.swapouts_pages
        ):
            raise HostResourceError("host_resource_counter_reset_or_wrap")
        previous = sample

    baseline = samples[0].snapshot
    final = samples[-1].snapshot
    in_use_peak = max(
        sample.snapshot.metal_in_use_system_memory_bytes for sample in samples
    )
    alloc_peak = max(
        sample.snapshot.metal_alloc_system_memory_bytes for sample in samples
    )
    swapins_delta_pages = final.swapins_pages - baseline.swapins_pages
    swapouts_delta_pages = final.swapouts_pages - baseline.swapouts_pages
    swapins_delta_bytes = swapins_delta_pages * page_size
    swapouts_delta_bytes = swapouts_delta_pages * page_size
    if (
        swapins_delta_bytes > _MAX_UINT64
        or swapouts_delta_bytes > _MAX_UINT64
    ):
        raise HostResourceError("host_resource_delta_overflow")
    return {
        "metal_in_use_baseline_bytes": (
            baseline.metal_in_use_system_memory_bytes
        ),
        "metal_in_use_peak_bytes": in_use_peak,
        "metal_in_use_increment_bytes": (
            in_use_peak - baseline.metal_in_use_system_memory_bytes
        ),
        "metal_alloc_baseline_bytes": baseline.metal_alloc_system_memory_bytes,
        "metal_alloc_peak_bytes": alloc_peak,
        "metal_alloc_increment_bytes": (
            alloc_peak - baseline.metal_alloc_system_memory_bytes
        ),
        "metal_recovery_delta": (
            final.metal_recovery_count - baseline.metal_recovery_count
        ),
        "swapins_delta_pages": swapins_delta_pages,
        "swapins_delta_bytes": swapins_delta_bytes,
        "swapouts_delta_pages": swapouts_delta_pages,
        "swapouts_delta_bytes": swapouts_delta_bytes,
    }


class HostResourceSampler:
    """Bounded periodic sampler with synchronous baseline and final snapshots."""

    def __init__(
        self,
        provider: HostResourceSnapshotProvider,
        *,
        sample_interval_seconds: float = HOST_RESOURCE_SAMPLE_INTERVAL_SECONDS,
        max_samples: int = MAX_HOST_RESOURCE_RAW_SAMPLES,
    ) -> None:
        provider_identity = getattr(provider, "identity", None)
        _require_identity(provider_identity, "provider identity")
        if getattr(provider, "scope", None) != HOST_RESOURCE_SCOPE:
            raise ValueError("host resource provider scope must be system_wide")
        if (
            type(sample_interval_seconds) is not float
            or not 0 < sample_interval_seconds
            <= _MAX_SAMPLE_INTERVAL_MILLISECONDS / 1_000
        ):
            raise ValueError("sample interval must be a bounded positive float")
        interval_milliseconds = round(sample_interval_seconds * 1_000)
        if (
            interval_milliseconds < 1
            or abs(
                sample_interval_seconds
                - interval_milliseconds / 1_000
            )
            > 1e-12
        ):
            raise ValueError("sample interval must use whole milliseconds")
        _require_bounded_integer(
            max_samples,
            "max_samples",
            minimum=2,
            maximum=MAX_HOST_RESOURCE_RAW_SAMPLES,
        )
        self._provider = provider
        self._provider_identity = provider_identity
        self._sample_interval_seconds = sample_interval_seconds
        self._sample_interval_milliseconds = interval_milliseconds
        self._max_samples = max_samples
        self._samples: list[HostResourceRawSample] = []
        self._start_ns: int | None = None
        self._state = "new"
        self._failure: HostResourceError | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    @property
    def started_monotonic_ns(self) -> int | None:
        """Existing sampler epoch for optional diagnostic alignment only."""
        with self._lock:
            return self._start_ns

    def start(self) -> None:
        with self._lock:
            if self._state != "new":
                raise HostResourceError("host_resource_sampler_invalid_state")
            start_ns = time.monotonic_ns()
            try:
                snapshot = self._take_snapshot()
            except HostResourceError:
                self._state = "failed"
                raise
            self._start_ns = start_ns
            self._samples = [HostResourceRawSample(0, snapshot)]
            self._state = "running"
            self._thread = threading.Thread(
                target=self._sampling_loop,
                name="videoscope-host-resource-sampler",
                daemon=True,
            )
            self._thread.start()

    def finish(self) -> HostResourceMeasurementReceipt:
        with self._lock:
            if self._state != "running":
                raise HostResourceError("host_resource_sampler_invalid_state")
            self._stop_event.set()
            thread = self._thread
        if thread is None:
            raise HostResourceError("host_resource_sampler_invalid_state")
        thread.join(timeout=5.0)
        if thread.is_alive():
            with self._lock:
                self._state = "failed"
            raise HostResourceError("host_resource_sampler_shutdown_timeout")
        with self._lock:
            if self._failure is not None:
                self._state = "failed"
                raise self._failure
            try:
                self._append_snapshot_locked(self._take_snapshot())
            except HostResourceError:
                self._state = "failed"
                raise
            samples = tuple(self._samples)
            self._state = "finished"
        return self.receipt_from_samples(
            provider_identity=self._provider_identity,
            sample_interval_milliseconds=self._sample_interval_milliseconds,
            samples=samples,
        )

    def _sampling_loop(self) -> None:
        while not self._stop_event.wait(self._sample_interval_seconds):
            try:
                snapshot = self._take_snapshot()
                with self._lock:
                    self._append_snapshot_locked(snapshot)
            except HostResourceError as error:
                with self._lock:
                    self._failure = error
                self._stop_event.set()
                return
            except Exception:
                with self._lock:
                    self._failure = HostResourceError(
                        "host_resource_sampler_failed"
                    )
                self._stop_event.set()
                return

    def _take_snapshot(self) -> HostResourceSnapshot:
        try:
            snapshot = self._provider.snapshot()
        except HostResourceError:
            raise
        except Exception:
            raise HostResourceError("host_resource_snapshot_failed") from None
        if not isinstance(snapshot, HostResourceSnapshot):
            raise HostResourceError("host_resource_snapshot_invalid")
        return snapshot

    def _append_snapshot_locked(self, snapshot: HostResourceSnapshot) -> None:
        if len(self._samples) >= self._max_samples:
            raise HostResourceError("host_resource_sample_limit_exceeded")
        if self._start_ns is None:
            raise HostResourceError("host_resource_sampler_invalid_state")
        elapsed = time.monotonic_ns() - self._start_ns
        if elapsed < 0 or elapsed > _MAX_ELAPSED_NANOSECONDS:
            raise HostResourceError("host_resource_clock_invalid")
        if self._samples and elapsed <= self._samples[-1].elapsed_nanoseconds:
            raise HostResourceError("host_resource_clock_not_monotonic")
        self._samples.append(HostResourceRawSample(elapsed, snapshot))

    @staticmethod
    def receipt_from_samples(
        *,
        provider_identity: str,
        sample_interval_milliseconds: int,
        samples: tuple[HostResourceRawSample, ...],
    ) -> HostResourceMeasurementReceipt:
        _require_identity(provider_identity, "provider_identity")
        _require_bounded_integer(
            sample_interval_milliseconds,
            "sample_interval_milliseconds",
            minimum=1,
            maximum=_MAX_SAMPLE_INTERVAL_MILLISECONDS,
        )
        if type(samples) is not tuple or not samples:
            raise ValueError("host resource samples must be a non-empty tuple")
        if len(samples) > MAX_HOST_RESOURCE_RAW_SAMPLES:
            raise ValueError("host resource samples exceed the portable bound")
        for sample in samples:
            if not isinstance(sample, HostResourceRawSample):
                raise ValueError("host resource samples contain an invalid value")
        derived = _derive_measurement(samples)
        return HostResourceMeasurementReceipt(
            schema_version=HOST_RESOURCE_RECEIPT_SCHEMA_VERSION,
            provider_identity=provider_identity,
            scope=HOST_RESOURCE_SCOPE,
            memory_accounting=HOST_RESOURCE_MEMORY_ACCOUNTING,
            sample_interval_milliseconds=sample_interval_milliseconds,
            raw_samples=samples,
            **derived,
        )


__all__ = [
    "HOST_RESOURCE_IDENTITY",
    "HOST_RESOURCE_MEMORY_ACCOUNTING",
    "HOST_RESOURCE_RECEIPT_SCHEMA_VERSION",
    "HOST_RESOURCE_SAMPLE_INTERVAL_SECONDS",
    "HOST_RESOURCE_SCOPE",
    "MAX_HOST_RESOURCE_RAW_SAMPLES",
    "DarwinHostResourceSnapshotProvider",
    "HostResourceError",
    "HostResourceMeasurementReceipt",
    "HostResourceRawSample",
    "HostResourceSampler",
    "HostResourceSnapshot",
    "HostResourceSnapshotProvider",
    "HostResourceUnavailableError",
    "create_host_resource_snapshot_provider",
]
