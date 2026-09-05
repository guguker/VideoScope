#!/usr/bin/env python3
"""Run a fail-closed offline smoke of every local VideoScope ML provider."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from hashlib import sha256
from ipaddress import ip_address
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import socket
import stat
import subprocess
import sys
from time import monotonic, sleep
from typing import NamedTuple
from urllib.parse import urlsplit


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_BACKEND_SOURCE_ROOT = os.fspath(_PROJECT_ROOT / "backend" / "src")
sys.path[:] = [
    _BACKEND_SOURCE_ROOT,
    *(entry for entry in sys.path if entry != _BACKEND_SOURCE_ROOT),
]

import numpy as np
from pydantic import Field

from videoscope.benchmark.host_resources import (
    HostResourceError,
    HostResourceMeasurementReceipt,
    HostResourceSampler,
    HostResourceUnavailableError,
    create_host_resource_snapshot_provider,
)
from videoscope.benchmark.measurements import (
    PROCESS_RSS_SAMPLE_INTERVAL_SECONDS,
    MeasurementError,
    MeasurementUnavailableError,
    ManagedProcessBinding,
    ProcessRecord,
    ProcessTreeRssSampler,
    create_native_process_snapshot_provider,
)
from videoscope.benchmark.schema import MAX_MEASUREMENT_RSS_SAMPLES
from videoscope.config import AppSettings
from videoscope.indexing_attestation import attest_indexing_toolchain
from videoscope.ml_environment_attestation import (
    DEFAULT_ML_ENVIRONMENT_MANIFEST_SHA256,
    attest_ml_environment,
    load_ml_environment_manifest,
)
from videoscope.model_manifest import (
    QWEN_VIDEO_MODEL,
    SIGLIP_224_MODEL,
    WHISPER_MODEL,
    model_identity,
    model_revision,
)
from videoscope.providers.base import ProviderState
from videoscope.providers.lighthouse_worker import LighthouseWorkerClient
from videoscope.providers.paddle_ocr import (
    OCR_MODEL_ARTIFACT_IDENTITY,
    OCR_WORKER_DEPENDENCY_IDENTITY,
    OCR_WORKER_RUNTIME_IDENTITY,
    PaddleOCRReader,
)
from videoscope.providers.qwen_video import QwenVideoJudgement
from videoscope.providers.qwen_worker import QwenWorkerClient
from videoscope.providers.vision_worker_client import VisionWorkerClient
from videoscope.providers.whisper import snapshot_whisper_prompt_from_content
from videoscope.providers.whisper_worker import WhisperWorkerClient
from videoscope.runtime import (
    _reviewed_ocr_script_sha256,
    create_vision_worker_specification,
)
from videoscope.providers.vision_worker_contract import (
    RFDETR_SMALL_CHECKPOINT_SHA256,
)


SCHEMA_VERSION = 2
EXIT_SUCCESS = 0
EXIT_CONFIGURATION = 2
EXIT_INFRASTRUCTURE = 3
EXIT_CONTRACT = 4
EXIT_OUT_OF_MEMORY = 6
EXIT_INTERNAL = 70
EXIT_INTERRUPTED = 130

_QUERY = "synthetic local video"
_QWEN_SMOKE_MAX_TOKENS = 320
_OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
    "TRANSFORMERS_OFFLINE": "1",
}
_MODELS_ROOT_ENVIRONMENT_VARIABLE = "VIDEOSCOPE_FULL_ML_SMOKE_MODELS_ROOT"
_FFMPEG_ENVIRONMENT_VARIABLE = "VIDEOSCOPE_FULL_ML_SMOKE_FFMPEG_BINARY"
_FFPROBE_ENVIRONMENT_VARIABLE = "VIDEOSCOPE_FULL_ML_SMOKE_FFPROBE_BINARY"
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_TOKEN_CHARACTERS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._~-"
)
_MAX_RSS_BYTES = (1 << 63) - 1
_MAX_BASELINE_ENTRIES = 4096
_MAX_BASELINE_FILE_BYTES = 1024 * 1024
_MAX_BASELINE_TOTAL_BYTES = 8 * 1024 * 1024
_REMOTE_WORKER_COMPONENTS = frozenset(
    {"vision", "whisper", "lighthouse", "qwen"}
)
_OOM_SIGNATURES = (
    "out of memory",
    "out-of-memory",
    "metal oom",
    "kiogpucommandbuffercallbackerroroutofmemory",
)
_EXECUTED_FROZEN_PROFILE_IDS = (
    "lexical_qdrant",
    "dense_siglip",
    "temporal_refinement",
    "lighthouse",
    "qwen_verification",
)
_ALL_FROZEN_PROFILE_IDS = (*_EXECUTED_FROZEN_PROFILE_IDS, "internvideo")
_PRODUCT_SEARCH_COMPONENT_IDS = (
    "text_vectors",
    "lexical_text",
    "visual_dense",
    "temporal_refinement",
    "lighthouse",
    "qwen_verification",
)
_MANAGED_WORKER_ROLES = ("vision", "whisper", "lighthouse", "qwen")
_REQUIRED_WORKER_PYTHON_ROLES = (*_MANAGED_WORKER_ROLES, "ocr")
_MANAGED_OCR_ENVIRONMENT_FIELDS = frozenset(
    {"HF_HOME", "HOME", "PATH", "TMPDIR", "XDG_CACHE_HOME"}
)
_WORKER_PYTHON_ENVIRONMENT = {
    role: f"VIDEOSCOPE_FULL_ML_SMOKE_{role.upper()}_PYTHON"
    for role in _REQUIRED_WORKER_PYTHON_ROLES
}
_ML_ENVIRONMENT_ATTESTATION_ID = "videoscope-phase0-ml-environment-v1"
_ML_ENVIRONMENT_ROLE_BINDINGS = {
    "owner": "base",
    "vision": "vision",
    "whisper": "whisper",
    "ocr": "ocr",
    "lighthouse": "lighthouse",
    "qwen": "qwen",
}
_WORKER_START_TIMEOUT_SECONDS = 300.0
_WORKER_BINDING_TIMEOUT_SECONDS = 5.0
_PRODUCT_SNAPSHOT_CLOSE_ATTEMPTS = 16


class SmokeError(RuntimeError):
    kind = "internal"

    def __init__(self, code: str, *, component: str | None = None) -> None:
        self.code = code
        self.component = component
        super().__init__(f"full ML smoke failed ({code})")


class SmokeConfigurationError(SmokeError):
    kind = "configuration"


class SmokeInfrastructureError(SmokeError):
    kind = "infrastructure"


class SmokeOutOfMemoryError(SmokeError):
    kind = "out_of_memory"


class SmokeContractError(SmokeError):
    kind = "contract"


class SmokeClients(NamedTuple):
    vision: object
    whisper: object
    ocr: object
    lighthouse: object
    qwen: object


class SmokeWorkspace(NamedTuple):
    image: Path
    lighthouse_cache: Path
    media_video: Path
    qwen_video: Path
    video_id: str


class SmokeFixture(NamedTuple):
    duration_seconds: float
    image: Path
    media_video: Path
    qwen_video: Path
    video_id: str


class SmokeDependencies(NamedTuple):
    attest_ml_environment: Callable[[Path, Mapping[str, str]], Mapping[str, object]]
    attest_toolchain: Callable[[Mapping[str, str]], object]
    build_clients: Callable[[object, Path], SmokeClients]
    build_fixture: Callable[[object, SmokeWorkspace], SmokeFixture]
    build_resource_monitor: Callable[[tuple[ManagedProcessBinding, ...]], object]
    code_identity_resolver: Callable[[], str]
    load_settings: Callable[[Path, Path, Mapping[str, object]], object]
    product_integration: Callable[..., Mapping[str, object]] | None
    start_workers: Callable[[Path, Path, Mapping[str, str]], object]


class _RootSeal(NamedTuple):
    device: int
    inode: int
    mode: int
    owner: int


class _BaselineEntry(NamedTuple):
    device: int
    digest: str | None
    inode: int
    kind: str
    mode: int
    owner: int
    size: int


class _WorkspaceBaseline(NamedTuple):
    entries: dict[str, _BaselineEntry]


class _WorkerProcess(NamedTuple):
    role: str
    port: int
    process: object


class _DisposableProductSettings(AppSettings):
    """Production settings with read-only model roots split from mutable state."""

    smoke_immutable_models_dir: Path
    ffmpeg_binary: Path = Field(exclude=True)
    ffprobe_binary: Path = Field(exclude=True)
    smoke_ocr_client_environment: dict[str, str] = Field(
        default_factory=dict,
        exclude=True,
    )
    ocr_worker_environment: dict[str, str] = Field(
        default_factory=dict,
        exclude=True,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):  # type: ignore[no-untyped-def]
        del cls, settings_cls, env_settings, dotenv_settings, file_secret_settings
        return (init_settings,)

    @property
    def models_dir(self) -> Path:
        return self.smoke_immutable_models_dir


_DisposableProductSettings.model_rebuild(_types_namespace={"Path": Path})


class _NativeProcessTreeRSSMonitor:
    """Fail-closed process-tree plus system-wide Metal/VM sampler."""

    def __init__(
        self,
        managed_workers: tuple[ManagedProcessBinding, ...],
    ) -> None:
        root_pid = os.getpid()
        if (
            not isinstance(managed_workers, tuple)
            or tuple(worker.role for worker in managed_workers)
            != _MANAGED_WORKER_ROLES
        ):
            raise SmokeConfigurationError(
                "external_worker_process_binding_unavailable",
                component="resources",
            )
        try:
            provider = create_native_process_snapshot_provider(root_pid=root_pid)
            self._process_sampler = ProcessTreeRssSampler(
                root_pid=root_pid,
                provider=provider,
                managed_workers=managed_workers,
            )
            self._host_sampler = HostResourceSampler(
                create_host_resource_snapshot_provider()
            )
        except MeasurementUnavailableError as error:
            raise SmokeInfrastructureError(
                "native_rss_measurement_unavailable",
                component="resources",
            ) from error
        except MeasurementError as error:
            raise SmokeInfrastructureError(
                "native_rss_measurement_failed",
                component="resources",
            ) from error
        except HostResourceUnavailableError as error:
            raise SmokeInfrastructureError(
                "native_host_resource_measurement_unavailable",
                component="resources",
            ) from error
        except HostResourceError as error:
            raise SmokeInfrastructureError(
                "native_host_resource_measurement_failed",
                component="resources",
            ) from error
        self._process_started = False
        self._process_finished = False
        self._host_started = False
        self._host_finished = False

    def start(self) -> None:
        try:
            self._host_sampler.start()
            self._host_started = True
            self._process_sampler.start()
            self._process_started = True
        except MeasurementUnavailableError as error:
            self._cleanup_after_start_failure()
            raise SmokeInfrastructureError(
                "native_rss_measurement_unavailable",
                component="resources",
            ) from error
        except MeasurementError as error:
            self._cleanup_after_start_failure()
            raise SmokeInfrastructureError(
                "native_rss_measurement_failed",
                component="resources",
            ) from error
        except HostResourceUnavailableError as error:
            self._cleanup_after_start_failure()
            raise SmokeInfrastructureError(
                "native_host_resource_measurement_unavailable",
                component="resources",
            ) from error
        except HostResourceError as error:
            self._cleanup_after_start_failure()
            raise SmokeInfrastructureError(
                "native_host_resource_measurement_failed",
                component="resources",
            ) from error

    def finish(self) -> object:
        process_receipt: object | None = None
        host_receipt: object | None = None
        failure: SmokeError | None = None
        try:
            process_receipt = self._process_sampler.finish()
        except MeasurementUnavailableError as error:
            failure = SmokeInfrastructureError(
                "native_rss_measurement_unavailable",
                component="resources",
            )
            failure.__cause__ = error
        except MeasurementError as error:
            failure = SmokeInfrastructureError(
                "native_rss_measurement_failed",
                component="resources",
            )
            failure.__cause__ = error
        finally:
            self._process_finished = True
        try:
            host_receipt = self._host_sampler.finish()
        except HostResourceUnavailableError as error:
            failure = failure or SmokeInfrastructureError(
                "native_host_resource_measurement_unavailable",
                component="resources",
            )
            if failure.__cause__ is None:
                failure.__cause__ = error
        except HostResourceError as error:
            failure = failure or SmokeInfrastructureError(
                "native_host_resource_measurement_failed",
                component="resources",
            )
            if failure.__cause__ is None:
                failure.__cause__ = error
        finally:
            self._host_finished = True
        if failure is not None:
            raise failure
        if process_receipt is None or host_receipt is None:
            raise SmokeInfrastructureError(
                "native_resource_measurement_unavailable",
                component="resources",
            )
        return {
            "process_tree": process_receipt,
            "host_resources": host_receipt,
        }

    def close(self) -> None:
        failure: SmokeError | None = None
        if self._host_started and not self._host_finished:
            try:
                self._host_sampler.finish()
            except HostResourceError as error:
                failure = SmokeInfrastructureError(
                    "native_host_resource_measurement_failed",
                    component="resources",
                )
                failure.__cause__ = error
            finally:
                self._host_finished = True
        try:
            self._process_sampler.close()
        except MeasurementError as error:
            failure = failure or SmokeInfrastructureError(
                "native_rss_measurement_failed",
                component="resources",
            )
            if failure.__cause__ is None:
                failure.__cause__ = error
        if failure is not None:
            raise failure

    def _cleanup_after_start_failure(self) -> None:
        if self._host_started and not self._host_finished:
            try:
                self._host_sampler.finish()
            except HostResourceError:
                pass
            self._host_finished = True
        try:
            self._process_sampler.close()
        except MeasurementError:
            pass


class _ManagedWorkerCluster:
    """Own four loopback worker subprocesses for one disposable smoke run."""

    all_workers_managed = True

    def __init__(
        self,
        *,
        processes: tuple[_WorkerProcess, ...],
        managed_workers: tuple[ManagedProcessBinding, ...],
        settings_overrides: Mapping[str, object],
    ) -> None:
        self._processes = processes
        self.managed_workers = managed_workers
        self.settings_overrides = dict(settings_overrides)
        self._closed = False

    def wait_ready(self) -> None:
        pending = {worker.role: worker for worker in self._processes}
        deadline = monotonic() + _WORKER_START_TIMEOUT_SECONDS
        while pending and monotonic() < deadline:
            for role, worker in tuple(pending.items()):
                poll = getattr(worker.process, "poll", None)
                if not callable(poll) or poll() is not None:
                    raise SmokeInfrastructureError(
                        "managed_worker_start_failed",
                        component=role,
                    )
                try:
                    with socket.create_connection(
                        ("127.0.0.1", worker.port),
                        timeout=0.1,
                    ):
                        pass
                except OSError:
                    continue
                pending.pop(role)
            if pending:
                sleep(0.05)
        if pending:
            raise SmokeInfrastructureError(
                "managed_worker_readiness_timeout",
                component=next(iter(pending)),
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        failure = False
        for worker in reversed(self._processes):
            process = worker.process
            try:
                if process.poll() is None:
                    process.terminate()
            except Exception:
                failure = True
        deadline = monotonic() + 15.0
        while monotonic() < deadline:
            try:
                if all(worker.process.poll() is not None for worker in self._processes):
                    break
            except Exception:
                failure = True
                break
            sleep(0.05)
        for worker in reversed(self._processes):
            process = worker.process
            try:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5.0)
            except Exception:
                failure = True
        if failure or any(
            worker.process.poll() is None for worker in self._processes
        ):
            raise SmokeInfrastructureError(
                "managed_worker_cleanup_failed",
                component="workers",
            )


def _is_sha256(value: object, *, prefix: bool = False) -> bool:
    if type(value) is not str:
        return False
    candidate = value.removeprefix("sha256:") if prefix else value
    return (
        (not prefix or value.startswith("sha256:"))
        and len(candidate) == 64
        and all(character in _SHA256_CHARACTERS for character in candidate)
    )


def _root_seal(metadata: os.stat_result) -> _RootSeal:
    return _RootSeal(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        mode=metadata.st_mode,
        owner=metadata.st_uid,
    )


def _validate_disposable_root(root: Path) -> tuple[Path, _RootSeal]:
    try:
        lexical = Path(os.path.abspath(os.fspath(root)))
        metadata = os.lstat(lexical)
        resolved = lexical.resolve(strict=True)
    except (OSError, TypeError, ValueError) as error:
        raise SmokeConfigurationError("disposable_root_invalid") from error
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise SmokeConfigurationError("disposable_root_invalid")
    forbidden = (
        resolved == Path(resolved.anchor)
        or resolved.is_relative_to(Path.home().resolve())
        or resolved == _PROJECT_ROOT
        or resolved.is_relative_to(_PROJECT_ROOT)
    )
    if forbidden:
        raise SmokeConfigurationError("disposable_root_forbidden")
    return resolved, _root_seal(metadata)


def _baseline_file_digest(path: Path, expected: os.stat_result) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    digest = sha256()
    try:
        descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            while chunk := os.read(descriptor, 1024 * 1024):
                digest.update(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        current = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise SmokeConfigurationError("disposable_root_baseline_invalid") from error
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if not (identity(expected) == identity(before) == identity(after) == identity(current)):
        raise SmokeConfigurationError("disposable_root_baseline_changed")
    return digest.hexdigest()


def _snapshot_workspace_baseline(root: Path) -> _WorkspaceBaseline:
    entries: dict[str, _BaselineEntry] = {}
    total_bytes = 0

    def visit(directory: Path, relative_parts: tuple[str, ...]) -> None:
        nonlocal total_bytes
        if len(relative_parts) > 32:
            raise SmokeConfigurationError("disposable_root_baseline_too_large")
        try:
            with os.scandir(directory) as iterator:
                children = tuple(iterator)
        except OSError as error:
            raise SmokeConfigurationError(
                "disposable_root_baseline_invalid"
            ) from error
        for child in children:
            if len(entries) >= _MAX_BASELINE_ENTRIES:
                raise SmokeConfigurationError(
                    "disposable_root_baseline_too_large"
                )
            name = child.name
            if not name or name in {".", ".."} or "/" in name or "\x00" in name:
                raise SmokeConfigurationError(
                    "disposable_root_baseline_invalid"
                )
            relative = "/".join((*relative_parts, name))
            path = Path(child.path)
            try:
                metadata = child.stat(follow_symlinks=False)
            except OSError as error:
                raise SmokeConfigurationError(
                    "disposable_root_baseline_invalid"
                ) from error
            if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
                entries[relative] = _BaselineEntry(
                    metadata.st_dev,
                    None,
                    metadata.st_ino,
                    "directory",
                    stat.S_IMODE(metadata.st_mode),
                    metadata.st_uid,
                    0,
                )
                visit(path, (*relative_parts, name))
                continue
            if (
                not stat.S_ISREG(metadata.st_mode)
                or stat.S_ISLNK(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_uid != os.geteuid()
                or metadata.st_size < 0
                or metadata.st_size > _MAX_BASELINE_FILE_BYTES
            ):
                raise SmokeConfigurationError(
                    "disposable_root_baseline_invalid"
                )
            total_bytes += metadata.st_size
            if total_bytes > _MAX_BASELINE_TOTAL_BYTES:
                raise SmokeConfigurationError(
                    "disposable_root_baseline_too_large"
                )
            entries[relative] = _BaselineEntry(
                metadata.st_dev,
                _baseline_file_digest(path, metadata),
                metadata.st_ino,
                "file",
                stat.S_IMODE(metadata.st_mode),
                metadata.st_uid,
                metadata.st_size,
            )

    visit(root, ())
    return _WorkspaceBaseline(entries)


def _verify_disposable_root(root: Path, expected: _RootSeal) -> None:
    try:
        current = os.lstat(root)
    except OSError as error:
        raise SmokeInfrastructureError(
            "disposable_root_changed",
            component="workspace",
        ) from error
    if _root_seal(current) != expected or stat.S_IMODE(current.st_mode) != 0o700:
        raise SmokeInfrastructureError(
            "disposable_root_changed",
            component="workspace",
        )


def _configured_path(value: object) -> Path | None:
    if type(value) is not str or not value or "\x00" in value:
        return None
    try:
        return Path(os.path.abspath(value))
    except (OSError, ValueError):
        return None


def _validated_media_executables(
    environ: Mapping[str, str],
) -> tuple[Path, Path]:
    values: list[Path] = []
    for variable, expected_name in (
        (_FFMPEG_ENVIRONMENT_VARIABLE, "ffmpeg"),
        (_FFPROBE_ENVIRONMENT_VARIABLE, "ffprobe"),
    ):
        raw = environ.get(variable)
        try:
            candidate = Path(raw) if type(raw) is str else None
            if (
                candidate is None
                or not candidate.is_absolute()
                or "\x00" in raw
                or Path(os.path.abspath(raw)) != candidate
            ):
                raise OSError
            lexical = os.lstat(candidate)
            resolved = candidate.resolve(strict=True)
            metadata = os.stat(resolved)
            if (
                candidate.name != expected_name
                or resolved.name != expected_name
                or not (
                    stat.S_ISREG(lexical.st_mode)
                    or stat.S_ISLNK(lexical.st_mode)
                )
                or not stat.S_ISREG(metadata.st_mode)
                or not os.access(candidate, os.X_OK)
            ):
                raise OSError
        except (OSError, TypeError, ValueError) as error:
            raise SmokeConfigurationError(
                "media_toolchain_binding_required",
                component="ffmpeg",
            ) from error
        values.append(resolved)
    if values[0].parent != values[1].parent:
        raise SmokeConfigurationError(
            "media_toolchain_binding_required",
            component="ffmpeg",
        )
    return values[0], values[1]


def _bind_attested_media_environment(
    environ: Mapping[str, str],
    toolchain: object,
) -> dict[str, str]:
    try:
        adapter = toolchain.create_ffmpeg()  # type: ignore[attr-defined]
        canonical = _validated_media_executables(
            {
                _FFMPEG_ENVIRONMENT_VARIABLE: adapter.ffmpeg_binary,
                _FFPROBE_ENVIRONMENT_VARIABLE: adapter.ffprobe_binary,
            }
        )
    except SmokeError:
        raise
    except Exception as error:
        raise SmokeContractError(
            "attestation_contract_invalid",
            component="ffmpeg",
        ) from error
    bound = dict(environ)
    bound[_FFMPEG_ENVIRONMENT_VARIABLE] = os.fspath(canonical[0])
    bound[_FFPROBE_ENVIRONMENT_VARIABLE] = os.fspath(canonical[1])
    return bound


def _validated_hf_home(root: Path, environ: Mapping[str, str]) -> Path:
    raw = environ.get("HF_HOME")
    try:
        candidate = Path(raw) if type(raw) is str else None
        if (
            candidate is None
            or not candidate.is_absolute()
            or "\x00" in raw
            or Path(os.path.abspath(raw)) != candidate
        ):
            raise OSError
        metadata = os.lstat(candidate)
        resolved = candidate.resolve(strict=True)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or resolved != candidate
            or resolved == root
            or resolved.is_relative_to(root)
            or root.is_relative_to(resolved)
            or resolved == _PROJECT_ROOT
            or resolved.is_relative_to(_PROJECT_ROOT)
        ):
            raise OSError
    except (OSError, TypeError, ValueError) as error:
        raise SmokeConfigurationError(
            "hf_home_binding_required",
            component="environment",
        ) from error
    return candidate


def _attest_current_toolchain(environ: Mapping[str, str]) -> object:
    ffmpeg_binary, ffprobe_binary = _validated_media_executables(environ)
    try:
        return attest_indexing_toolchain(
            ffmpeg_binary=ffmpeg_binary,
            ffprobe_binary=ffprobe_binary,
        )
    except SmokeError:
        raise
    except Exception as error:
        raise SmokeInfrastructureError(
            "attestation_failed",
            component="ffmpeg",
        ) from error


def _validate_offline_environment(
    root: Path,
    environ: Mapping[str, str],
) -> None:
    if any(environ.get(name) != value for name, value in _OFFLINE_ENVIRONMENT.items()):
        raise SmokeConfigurationError("offline_environment_required")
    expected_roots = {
        "VIDEOSCOPE_DATA_DIR": root / "product",
        "VIDEOSCOPE_VISION_WORKER_INPUT_ROOT": root,
        "VIDEOSCOPE_WHISPER_WORKER_INPUT_ROOT": root / "product" / "media",
        "VIDEOSCOPE_WHISPER_WORKER_WORK_ROOT": root / "tmp" / "whisper-worker",
    }
    if any(
        _configured_path(environ.get(name)) != expected
        for name, expected in expected_roots.items()
    ):
        raise SmokeConfigurationError("worker_root_binding_required")


def _validate_models_root(
    root: Path,
    models_root: Path,
    environ: Mapping[str, str],
) -> Path:
    try:
        candidate = Path(models_root)
        if not candidate.is_absolute() or "\x00" in os.fspath(candidate):
            raise ValueError
        lexical = Path(os.path.abspath(os.fspath(candidate)))
    except (OSError, TypeError, ValueError) as error:
        raise SmokeConfigurationError("immutable_model_root_invalid") from error

    configured = environ.get(_MODELS_ROOT_ENVIRONMENT_VARIABLE)
    try:
        configured_path = Path(configured) if type(configured) is str else None
        if (
            configured_path is None
            or not configured_path.is_absolute()
            or "\x00" in configured
            or Path(os.path.abspath(configured)) != lexical
        ):
            raise ValueError
    except (OSError, TypeError, ValueError) as error:
        raise SmokeConfigurationError("models_root_binding_required") from error

    descriptor: int | None = None
    try:
        metadata = os.lstat(lexical)
        resolved = lexical.resolve(strict=True)
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or lexical != resolved
            or resolved == Path(resolved.anchor)
            or resolved == root
            or resolved.is_relative_to(root)
            or root.is_relative_to(resolved)
            or resolved == _PROJECT_ROOT
            or resolved.is_relative_to(_PROJECT_ROOT)
        ):
            raise OSError
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(resolved, flags)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or (opened.st_dev, opened.st_ino)
            != (metadata.st_dev, metadata.st_ino)
        ):
            raise OSError
    except OSError as error:
        raise SmokeConfigurationError("immutable_model_root_invalid") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return resolved


def _validated_worker_pythons(
    root: Path,
    environ: Mapping[str, str],
) -> dict[str, Path]:
    executables: dict[str, Path] = {}
    for role in _REQUIRED_WORKER_PYTHON_ROLES:
        raw = environ.get(_WORKER_PYTHON_ENVIRONMENT[role])
        try:
            candidate = Path(raw) if type(raw) is str else None
            if (
                candidate is None
                or not candidate.is_absolute()
                or "\x00" in raw
            ):
                raise OSError
            lexical = Path(os.path.abspath(raw))
            metadata = os.lstat(lexical)
            resolved = lexical.resolve(strict=True)
            resolved_metadata = os.stat(resolved)
            if (
                not (
                    stat.S_ISREG(metadata.st_mode)
                    or stat.S_ISLNK(metadata.st_mode)
                )
                or not stat.S_ISREG(resolved_metadata.st_mode)
                or not os.access(lexical, os.X_OK)
                or lexical.is_relative_to(root)
                or resolved.is_relative_to(root)
            ):
                raise OSError
        except (OSError, TypeError, ValueError) as error:
            raise SmokeConfigurationError(
                "worker_python_binding_required",
                component=role,
            ) from error
        executables[role] = lexical
    if len(set(executables.values())) != len(_REQUIRED_WORKER_PYTHON_ROLES):
        raise SmokeConfigurationError("worker_python_bindings_not_isolated")
    return executables


def _validate_environment_executable_bindings(
    root: Path,
    environ: Mapping[str, str],
    manifest: object,
    *,
    owner_executable: Path,
) -> dict[str, str]:
    """Bind the owner and every launched role to the manifest's exact venv path."""

    raw_environments = getattr(manifest, "environments", None)
    if not isinstance(raw_environments, tuple):
        raise SmokeContractError(
            "ml_environment_attestation_invalid",
            component="environment",
        )
    directories: dict[str, str] = {}
    for item in raw_environments:
        environment_id = getattr(item, "environment_id", None)
        directory = getattr(item, "directory", None)
        if (
            type(environment_id) is not str
            or type(directory) is not str
            or environment_id in directories
        ):
            raise SmokeContractError(
                "ml_environment_attestation_invalid",
                component="environment",
            )
        directories[environment_id] = directory
    if set(_ML_ENVIRONMENT_ROLE_BINDINGS.values()) != set(directories):
        raise SmokeContractError(
            "ml_environment_attestation_invalid",
            component="environment",
        )

    worker_executables = _validated_worker_pythons(root, environ)
    try:
        lexical_owner = Path(os.path.abspath(os.fspath(owner_executable)))
    except (OSError, TypeError, ValueError) as error:
        raise SmokeConfigurationError(
            "ml_environment_binding_mismatch",
            component="owner",
        ) from error
    actual_by_role = {"owner": lexical_owner, **worker_executables}
    for role, environment_id in _ML_ENVIRONMENT_ROLE_BINDINGS.items():
        expected = _PROJECT_ROOT / directories[environment_id] / "bin" / "python"
        if actual_by_role.get(role) != expected:
            raise SmokeConfigurationError(
                "ml_environment_binding_mismatch",
                component=role,
            )
    return dict(_ML_ENVIRONMENT_ROLE_BINDINGS)


def _attest_current_ml_environment(
    root: Path,
    environ: Mapping[str, str],
) -> Mapping[str, object]:
    manifest_path = _PROJECT_ROOT / "workers" / "ml-environment.lock.json"
    try:
        manifest = load_ml_environment_manifest(
            manifest_path,
            expected_sha256=DEFAULT_ML_ENVIRONMENT_MANIFEST_SHA256,
        )
        bindings = _validate_environment_executable_bindings(
            root,
            environ,
            manifest,
            owner_executable=Path(sys.executable),
        )
        report = attest_ml_environment(
            _PROJECT_ROOT,
            manifest_path=manifest_path,
            expected_manifest_sha256=DEFAULT_ML_ENVIRONMENT_MANIFEST_SHA256,
            environ=environ,
        )
    except SmokeError:
        raise
    except Exception as error:
        raise SmokeConfigurationError(
            "ml_environment_attestation_failed",
            component="environment",
        ) from error
    expected_identity = "sha256:" + DEFAULT_ML_ENVIRONMENT_MANIFEST_SHA256
    if (
        report.get("status") != "complete"
        or report.get("failures") != []
        or report.get("attestation_id") != _ML_ENVIRONMENT_ATTESTATION_ID
        or report.get("manifest_identity") != expected_identity
        or getattr(manifest, "attestation_id", None)
        != _ML_ENVIRONMENT_ATTESTATION_ID
        or getattr(manifest, "raw_sha256", None)
        != DEFAULT_ML_ENVIRONMENT_MANIFEST_SHA256
    ):
        raise SmokeConfigurationError(
            "ml_environment_attestation_failed",
            component="environment",
        )
    return {
        "environment_bindings": bindings,
        "ml_environment_attestation_id": _ML_ENVIRONMENT_ATTESTATION_ID,
        "ml_environment_manifest_identity": expected_identity,
    }


def _validated_ml_environment_receipt(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "environment_bindings",
        "ml_environment_attestation_id",
        "ml_environment_manifest_identity",
    }:
        raise SmokeContractError(
            "ml_environment_attestation_invalid",
            component="environment",
        )
    bindings = value.get("environment_bindings")
    attestation_id = value.get("ml_environment_attestation_id")
    manifest_identity = value.get("ml_environment_manifest_identity")
    if (
        not isinstance(bindings, Mapping)
        or dict(bindings) != _ML_ENVIRONMENT_ROLE_BINDINGS
        or attestation_id != _ML_ENVIRONMENT_ATTESTATION_ID
        or not _is_sha256(manifest_identity, prefix=True)
    ):
        raise SmokeContractError(
            "ml_environment_attestation_invalid",
            component="environment",
        )
    return {
        "environment_bindings": dict(_ML_ENVIRONMENT_ROLE_BINDINGS),
        "ml_environment_attestation_id": _ML_ENVIRONMENT_ATTESTATION_ID,
        "ml_environment_manifest_identity": manifest_identity,
    }


def _allocate_loopback_ports() -> dict[str, int]:
    ports: dict[str, int] = {}
    used: set[int] = set()
    for role in _MANAGED_WORKER_ROLES:
        for _attempt in range(64):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                    listener.bind(("127.0.0.1", 0))
                    port = int(listener.getsockname()[1])
            except OSError as error:
                raise SmokeInfrastructureError(
                    "loopback_port_allocation_failed",
                    component=role,
                ) from error
            if port not in used:
                ports[role] = port
                used.add(port)
                break
        else:
            raise SmokeInfrastructureError(
                "loopback_port_allocation_failed",
                component=role,
            )
    return ports


def _worker_base_environment(
    environ: Mapping[str, str],
    root: Path,
    role: str,
) -> dict[str, str]:
    ffmpeg_binary, _ffprobe_binary = _validated_media_executables(environ)
    hf_home = _validated_hf_home(root, environ)
    temporary = root / "tmp" / f"worker-{role}"
    try:
        temporary.mkdir(mode=0o700, parents=True, exist_ok=False)
    except OSError as error:
        raise SmokeInfrastructureError(
            "worker_temporary_root_unavailable",
            component=role,
        ) from error
    child = {
        **_OFFLINE_ENVIRONMENT,
        "HF_DATASETS_OFFLINE": "1",
        "HF_HOME": os.fspath(hf_home),
        "HF_HUB_CACHE": os.fspath(hf_home / "hub"),
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HOME": os.fspath(temporary),
        "NO_PROXY": "127.0.0.1",
        "PATH": os.fspath(ffmpeg_binary.parent),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(_PROJECT_ROOT / "backend" / "src"),
        "TMPDIR": os.fspath(temporary),
        "TRANSFORMERS_CACHE": os.fspath(hf_home / "hub"),
        "TRANSFORMERS_OFFLINE": "1",
        "UV_OFFLINE": "1",
        "XDG_CACHE_HOME": os.fspath(hf_home),
        "no_proxy": "127.0.0.1",
    }
    return child


def _isolated_ocr_worker_environment(
    root: Path,
    environ: Mapping[str, str],
    *,
    role: str,
) -> dict[str, str]:
    if role not in {"ocr-client", "ocr-product"}:
        raise SmokeConfigurationError(
            "worker_configuration_invalid",
            component="ocr",
        )
    ffmpeg_binary, _ffprobe_binary = _validated_media_executables(environ)
    hf_home = _validated_hf_home(root, environ)
    temporary = root / "tmp" / f"worker-{role}"
    cache = temporary / "cache"
    try:
        temporary.mkdir(mode=0o700, parents=True, exist_ok=False)
        cache.mkdir(mode=0o700, exist_ok=False)
    except OSError as error:
        raise SmokeInfrastructureError(
            "worker_temporary_root_unavailable",
            component="ocr",
        ) from error
    return {
        "HF_HOME": os.fspath(hf_home),
        "HOME": os.fspath(temporary),
        "PATH": os.fspath(ffmpeg_binary.parent),
        "TMPDIR": os.fspath(temporary),
        "XDG_CACHE_HOME": os.fspath(cache),
    }


def _worker_environments(
    root: Path,
    models_root: Path,
    environ: Mapping[str, str],
    ports: Mapping[str, int],
    tokens: Mapping[str, str],
) -> dict[str, dict[str, str]]:
    def base(role: str) -> dict[str, str]:
        return _worker_base_environment(environ, root, role)

    whisper_revision = model_revision(WHISPER_MODEL)
    qwen_revision = model_revision(QWEN_VIDEO_MODEL)
    if whisper_revision is None or qwen_revision is None:
        raise SmokeContractError("managed_worker_model_revision_missing")
    return {
        "vision": {
            **base("vision"),
            "VIDEOSCOPE_SIGLIP_MODEL": SIGLIP_224_MODEL,
            "VIDEOSCOPE_VISION_DETECTOR_CHECKPOINT_SHA256": (
                RFDETR_SMALL_CHECKPOINT_SHA256
            ),
            "VIDEOSCOPE_VISION_DETECTOR_MODEL_ID": "rfdetr-small",
            "VIDEOSCOPE_VISION_WORKER_API_KEY": tokens["vision"],
            "VIDEOSCOPE_VISION_WORKER_HOST": "127.0.0.1",
            "VIDEOSCOPE_VISION_WORKER_INPUT_ROOT": str(root),
            "VIDEOSCOPE_VISION_WORKER_PRODUCT_DATA_SUBDIRECTORY": "product",
            "VIDEOSCOPE_VISION_WORKER_PORT": str(ports["vision"]),
            "VIDEOSCOPE_VISION_WORKER_RFDETR_CHECKPOINT": str(
                models_root / "rfdetr" / "rf-detr-small.pth"
            ),
        },
        "whisper": {
            **base("whisper"),
            "VIDEOSCOPE_WHISPER_WORKER_API_KEY": tokens["whisper"],
            "VIDEOSCOPE_WHISPER_WORKER_HOST": "127.0.0.1",
            "VIDEOSCOPE_WHISPER_WORKER_INPUT_ROOT": str(
                root / "product" / "media"
            ),
            "VIDEOSCOPE_WHISPER_WORKER_LOG_LEVEL": "warning",
            "VIDEOSCOPE_WHISPER_WORKER_MODEL_NAME": WHISPER_MODEL,
            "VIDEOSCOPE_WHISPER_WORKER_MODEL_REVISION": whisper_revision,
            "VIDEOSCOPE_WHISPER_WORKER_PORT": str(ports["whisper"]),
            "VIDEOSCOPE_WHISPER_WORKER_WORK_ROOT": str(
                root / "tmp" / "whisper-worker"
            ),
        },
        "lighthouse": {
            **base("lighthouse"),
            "VIDEOSCOPE_DATA_DIR": str(root / "product"),
            "VIDEOSCOPE_LIGHTHOUSE_API_KEY": tokens["lighthouse"],
            "VIDEOSCOPE_LIGHTHOUSE_CHECKPOINT": str(
                models_root
                / "lighthouse"
                / "clip_qd_detr_qvhighlight.ckpt"
            ),
            "VIDEOSCOPE_LIGHTHOUSE_CLIP_CHECKPOINT": str(
                models_root / "lighthouse" / "ViT-B-32.pt"
            ),
            "VIDEOSCOPE_LIGHTHOUSE_WORKER_PORT": str(ports["lighthouse"]),
        },
        "qwen": {
            **base("qwen"),
            "QWEN_VIDEO_API_KEY": tokens["qwen"],
            "QWEN_VIDEO_MODEL": QWEN_VIDEO_MODEL,
            "QWEN_WORKER_MAX_CONCURRENCY": "1",
            "QWEN_WORKER_PORT": str(ports["qwen"]),
            "VIDEOSCOPE_DATA_DIR": str(root / "product"),
            "VIDEOSCOPE_QWEN_WORKER_INPUT_ROOT": str(root),
            "VIDEOSCOPE_QWEN_VIDEO_MODEL_REVISION": qwen_revision,
        },
    }


def _managed_settings_overrides(
    ports: Mapping[str, int],
    tokens: Mapping[str, str],
    executables: Mapping[str, Path],
    *,
    ocr_client_environment: Mapping[str, str],
    ocr_product_environment: Mapping[str, str],
) -> dict[str, object]:
    return {
        "ocr_worker_python": executables["ocr"],
        "ocr_worker_script": _PROJECT_ROOT / "scripts" / "paddle-ocr-worker.py",
        "smoke_ocr_client_environment": dict(ocr_client_environment),
        "ocr_worker_environment": dict(ocr_product_environment),
        "vision_worker_endpoint": f"http://127.0.0.1:{ports['vision']}",
        "vision_worker_api_key": tokens["vision"],
        "whisper_worker_endpoint": f"http://127.0.0.1:{ports['whisper']}",
        "whisper_worker_api_key": tokens["whisper"],
        "lighthouse_endpoint": f"http://127.0.0.1:{ports['lighthouse']}",
        "lighthouse_api_key": tokens["lighthouse"],
        "qwen_video_endpoint": f"http://127.0.0.1:{ports['qwen']}",
        "qwen_video_api_key": tokens["qwen"],
    }


def start_managed_workers(
    root: Path,
    models_root: Path,
    environ: Mapping[str, str],
) -> _ManagedWorkerCluster:
    """Launch the four HTTP ML workers as measured child processes."""

    executables = _validated_worker_pythons(root, environ)
    ports = _allocate_loopback_ports()
    tokens = {role: secrets.token_hex(32) for role in _MANAGED_WORKER_ROLES}
    environments = _worker_environments(
        root,
        models_root,
        environ,
        ports,
        tokens,
    )
    ocr_client_environment = _isolated_ocr_worker_environment(
        root,
        environ,
        role="ocr-client",
    )
    ocr_product_environment = _isolated_ocr_worker_environment(
        root,
        environ,
        role="ocr-product",
    )
    modules = {
        role: f"videoscope.providers.{role}_worker"
        for role in _MANAGED_WORKER_ROLES
    }
    processes: list[_WorkerProcess] = []
    try:
        for role in _MANAGED_WORKER_ROLES:
            process = subprocess.Popen(
                [os.fspath(executables[role]), "-m", modules[role]],
                cwd=root,
                env=environments[role],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
            processes.append(_WorkerProcess(role, ports[role], process))
        provider = create_native_process_snapshot_provider(root_pid=os.getpid())
        deadline = monotonic() + _WORKER_BINDING_TIMEOUT_SECONDS
        records: dict[int, ProcessRecord] = {}
        while monotonic() < deadline:
            exited = next(
                (
                    worker
                    for worker in processes
                    if worker.process.poll() is not None
                ),
                None,
            )
            if exited is not None:
                raise SmokeInfrastructureError(
                    "managed_worker_process_binding_unavailable",
                    component=exited.role,
                )
            records = {
                record.pid: record
                for record in provider.snapshot()
                if isinstance(record, ProcessRecord)
            }
            if all(worker.process.pid in records for worker in processes):
                break
            sleep(0.02)
        bindings: list[ManagedProcessBinding] = []
        for worker in processes:
            pid = getattr(worker.process, "pid", None)
            record = records.get(pid) if type(pid) is int else None
            if record is None:
                raise SmokeInfrastructureError(
                    "managed_worker_process_binding_unavailable",
                    component=worker.role,
                )
            bindings.append(
                ManagedProcessBinding(
                    pid=record.pid,
                    start_token=record.start_token,
                    executable_identity=record.executable_identity,
                    role=worker.role,
                )
            )
        return _ManagedWorkerCluster(
            processes=tuple(processes),
            managed_workers=tuple(bindings),
            settings_overrides=_managed_settings_overrides(
                ports,
                tokens,
                executables,
                ocr_client_environment=ocr_client_environment,
                ocr_product_environment=ocr_product_environment,
            ),
        )
    except SmokeError:
        cluster = _ManagedWorkerCluster(
            processes=tuple(processes),
            managed_workers=(),
            settings_overrides={},
        )
        cluster.close()
        raise
    except Exception as error:
        cluster = _ManagedWorkerCluster(
            processes=tuple(processes),
            managed_workers=(),
            settings_overrides={},
        )
        try:
            cluster.close()
        except SmokeError:
            pass
        if _is_out_of_memory(error):
            raise SmokeOutOfMemoryError(
                "out_of_memory",
                component="workers",
            ) from error
        raise SmokeInfrastructureError(
            "managed_worker_start_failed",
            component="workers",
        ) from error


def _literal_loopback_origin(value: object) -> bool:
    if type(value) is not str:
        return False
    try:
        parsed = urlsplit(value)
        address = ip_address(parsed.hostname or "")
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme == "http"
        and str(address) == "127.0.0.1"
        and port is not None
        and parsed.username is None
        and parsed.password is None
        and parsed.path in {"", "/"}
        and not parsed.query
        and not parsed.fragment
    )


def _valid_token(value: object) -> bool:
    return (
        type(value) is str
        and 32 <= len(value) <= 256
        and all(character in _TOKEN_CHARACTERS for character in value)
    )


def _validate_required_settings(settings: object) -> None:
    providers = (
        ("vision", "vision_worker_endpoint", "vision_worker_api_key"),
        ("whisper", "whisper_worker_endpoint", "whisper_worker_api_key"),
        ("lighthouse", "lighthouse_endpoint", "lighthouse_api_key"),
        ("qwen", "qwen_video_endpoint", "qwen_video_api_key"),
    )
    for component, endpoint_name, key_name in providers:
        if not _literal_loopback_origin(getattr(settings, endpoint_name, None)):
            raise SmokeConfigurationError(
                "loopback_worker_configuration_required",
                component=component,
            )
        if not _valid_token(getattr(settings, key_name, None)):
            raise SmokeConfigurationError(
                "worker_token_required",
                component=component,
            )
    qwen_model = getattr(settings, "qwen_video_model", None)
    whisper_model = getattr(settings, "whisper_model", None)
    if model_revision(qwen_model) is None:
        raise SmokeConfigurationError(
            "pinned_model_required",
            component="qwen",
        )
    if model_revision(whisper_model) is None:
        raise SmokeConfigurationError(
            "pinned_model_required",
            component="whisper",
        )
    if getattr(settings, "qwen_video_allow_in_process", False) is not False:
        raise SmokeConfigurationError(
            "in_process_provider_forbidden",
            component="qwen",
        )
    if getattr(settings, "lighthouse_allow_in_process", False) is not False:
        raise SmokeConfigurationError(
            "in_process_provider_forbidden",
            component="lighthouse",
        )


def _load_settings(
    root: Path,
    models_root: Path,
    worker_overrides: Mapping[str, object],
) -> AppSettings:
    try:
        return _DisposableProductSettings(
            data_dir=root / "product",
            internvideo_api_key=None,
            internvideo_endpoint=None,
            lighthouse_checkpoint=(
                models_root
                / "lighthouse"
                / "clip_qd_detr_qvhighlight.ckpt"
            ),
            lighthouse_clip_checkpoint=(
                models_root / "lighthouse" / "ViT-B-32.pt"
            ),
            qwen_video_model=QWEN_VIDEO_MODEL,
            semantic_text_min_score=0.0,
            siglip_model=SIGLIP_224_MODEL,
            smoke_immutable_models_dir=models_root,
            visual_min_score=0.0,
            whisper_model=WHISPER_MODEL,
            **dict(worker_overrides),
        )
    except Exception as error:
        raise SmokeConfigurationError("worker_configuration_invalid") from error


def _ensure_workspace_directory(root: Path, name: str) -> Path:
    target = root / name
    try:
        if not target.exists():
            target.mkdir(mode=0o700)
        metadata = os.lstat(target)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or target.resolve(strict=True).parent != root
        ):
            raise OSError
    except OSError as error:
        raise SmokeInfrastructureError(
            "workspace_directory_invalid",
            component="workspace",
        ) from error
    return target


def _allocate_workspace(root: Path) -> SmokeWorkspace:
    product = _ensure_workspace_directory(root, "product")
    media = _ensure_workspace_directory(product, "media")
    temporary = _ensure_workspace_directory(product, "tmp")
    cache = _ensure_workspace_directory(product, "cache")
    lighthouse_cache = cache / "lighthouse"
    for _attempt in range(64):
        suffix = secrets.token_hex(12)
        video_id = f"smoke_{suffix}"
        media_video = media / f"full-ml-smoke-{suffix}.mp4"
        qwen_video = temporary / f"full-ml-smoke-{suffix}.mp4"
        image = temporary / f"full-ml-smoke-{suffix}.jpg"
        if not any(
            path.exists()
            for path in (
                media_video,
                qwen_video,
                image,
                lighthouse_cache / video_id,
            )
        ):
            return SmokeWorkspace(
                image=image,
                lighthouse_cache=lighthouse_cache / video_id,
                media_video=media_video,
                qwen_video=qwen_video,
                video_id=video_id,
            )
    raise SmokeInfrastructureError(
        "workspace_name_unavailable",
        component="workspace",
    )


def _safe_copy(source: Path, destination: Path) -> None:
    descriptor = os.open(
        destination,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with source.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                view = memoryview(block)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("short disposable fixture write")
                    view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_synthetic_fixture(
    toolchain: object,
    workspace: SmokeWorkspace,
) -> SmokeFixture:
    try:
        from PIL import Image, ImageDraw, ImageFont

        ffmpeg_binary = getattr(toolchain, "ffmpeg_binary")
        create_ffmpeg = getattr(toolchain, "create_ffmpeg")
        if type(ffmpeg_binary) is not str or not callable(create_ffmpeg):
            raise ValueError
        canvas = Image.new("RGB", (960, 540), color=(8, 12, 20))
        drawing = ImageDraw.Draw(canvas)
        font = ImageFont.load_default(size=64)
        label = "SYNTHETIC LOCAL VIDEO"
        bounds = drawing.textbbox((0, 0), label, font=font, stroke_width=2)
        text_width = bounds[2] - bounds[0]
        text_height = bounds[3] - bounds[1]
        drawing.rectangle(
            (
                24,
                24,
                canvas.width - 24,
                canvas.height - 24,
            ),
            outline=(0, 190, 255),
            width=12,
        )
        drawing.text(
            (
                (canvas.width - text_width) / 2,
                (canvas.height - text_height) / 2,
            ),
            label,
            fill=(255, 255, 255),
            font=font,
            stroke_fill=(0, 0, 0),
            stroke_width=2,
        )
        canvas.save(workspace.image, format="JPEG", quality=95)
        completed = subprocess.run(
            [
                ffmpeg_binary,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-loop",
                "1",
                "-framerate",
                "2",
                "-i",
                str(workspace.image),
                "-t",
                "2",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=16000:duration=2",
                "-shortest",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                str(workspace.media_video),
            ],
            check=False,
            capture_output=True,
            env={},
            timeout=120,
        )
        if completed.returncode != 0 or not workspace.media_video.is_file():
            raise RuntimeError
        os.chmod(workspace.media_video, 0o600)
        ffmpeg = create_ffmpeg()
        probe = ffmpeg.probe(workspace.media_video)
        ffmpeg.extract_frame(
            workspace.media_video,
            workspace.image,
            min(0.5, float(probe.duration) / 2),
            max_width=960,
        )
        os.chmod(workspace.image, 0o600)
        _safe_copy(workspace.media_video, workspace.qwen_video)
        if (
            not math.isfinite(float(probe.duration))
            or not 0.5 <= float(probe.duration) <= 10
            or probe.width <= 0
            or probe.height <= 0
        ):
            raise ValueError
        return SmokeFixture(
            duration_seconds=float(probe.duration),
            image=workspace.image,
            media_video=workspace.media_video,
            qwen_video=workspace.qwen_video,
            video_id=workspace.video_id,
        )
    except Exception as error:
        raise SmokeInfrastructureError(
            "synthetic_fixture_failed",
            component="ffmpeg",
        ) from error


def build_real_clients(settings: object, root: Path) -> SmokeClients:
    if not isinstance(settings, AppSettings):
        raise SmokeConfigurationError("worker_configuration_invalid")
    try:
        vision = VisionWorkerClient(
            endpoint=settings.vision_worker_endpoint or "",
            api_key=settings.vision_worker_api_key or "",
            input_root=root,
            specification=create_vision_worker_specification(settings),
            timeout=settings.vision_worker_timeout,
            health_timeout=_WORKER_START_TIMEOUT_SECONDS,
        )
    except Exception as error:
        raise SmokeInfrastructureError(
            "client_construction_failed",
            component="vision",
        ) from error
    try:
        whisper_revision = model_revision(settings.whisper_model)
        assert whisper_revision is not None
        whisper = WhisperWorkerClient(
            endpoint=settings.whisper_worker_endpoint or "",
            api_key=settings.whisper_worker_api_key or "",
            input_root=settings.media_dir,
            expected_model_identity=model_identity(
                settings.whisper_model,
                whisper_revision,
            ),
            timeout=settings.whisper_worker_timeout,
        )
    except Exception as error:
        raise SmokeInfrastructureError(
            "client_construction_failed",
            component="whisper",
        ) from error
    try:
        ocr = PaddleOCRReader(
            worker_python=settings.ocr_worker_python,
            worker_script=settings.ocr_worker_script,
            expected_dependency_identity=OCR_WORKER_DEPENDENCY_IDENTITY,
            expected_runtime_identity=OCR_WORKER_RUNTIME_IDENTITY,
            expected_model_identity=OCR_MODEL_ARTIFACT_IDENTITY,
            expected_script_sha256=_reviewed_ocr_script_sha256(
                settings.ocr_worker_script
            ),
            worker_environment=settings.smoke_ocr_client_environment,
        )
    except Exception as error:
        raise SmokeInfrastructureError(
            "client_construction_failed",
            component="ocr",
        ) from error
    try:
        lighthouse = LighthouseWorkerClient(
            endpoint=settings.lighthouse_endpoint or "",
            api_key=settings.lighthouse_api_key or "",
            input_root=settings.media_dir,
            cache_dir=settings.cache_dir,
            timeout=settings.lighthouse_timeout,
        )
    except Exception as error:
        ocr.close()
        raise SmokeInfrastructureError(
            "client_construction_failed",
            component="lighthouse",
        ) from error
    try:
        qwen_model = settings.qwen_video_model
        qwen_revision = model_revision(qwen_model)
        assert qwen_model is not None and qwen_revision is not None
        qwen = QwenWorkerClient(
            endpoint=settings.qwen_video_endpoint or "",
            api_key=settings.qwen_video_api_key or "",
            input_root=root,
            expected_model_identity=model_identity(qwen_model, qwen_revision),
            timeout=settings.qwen_video_timeout,
        )
    except Exception as error:
        ocr.close()
        raise SmokeInfrastructureError(
            "client_construction_failed",
            component="qwen",
        ) from error
    return SmokeClients(vision, whisper, ocr, lighthouse, qwen)


class _QueueWakeup:
    def __init__(self, queue: object) -> None:
        self._queue = queue

    def wake(self, _video_id: str) -> None:
        wake = getattr(self._queue, "wake", None)
        if not callable(wake):
            raise RuntimeError("product queue cannot be woken")
        wake()


def _ensure_product_directories(settings: object, root: Path) -> None:
    paths = (
        getattr(settings, "data_dir", None),
        getattr(settings, "media_dir", None),
        getattr(settings, "thumbnails_dir", None),
        getattr(settings, "clips_dir", None),
        getattr(settings, "cache_dir", None),
        getattr(settings, "qdrant_dir", None),
        getattr(settings, "visual_index_dir", None),
        getattr(settings, "temp_dir", None),
    )
    if any(not isinstance(path, Path) for path in paths):
        raise SmokeContractError(
            "product_storage_contract_invalid",
            component="product",
        )
    for raw_path in paths:
        assert isinstance(raw_path, Path)
        lexical = Path(os.path.abspath(raw_path))
        if lexical != root and not lexical.is_relative_to(root):
            raise SmokeConfigurationError(
                "product_storage_outside_disposable_root",
                component="product",
            )
        try:
            lexical.mkdir(mode=0o700, parents=True, exist_ok=True)
            metadata = os.lstat(lexical)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or stat.S_ISLNK(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or lexical.resolve(strict=True) != lexical
            ):
                raise OSError
            os.chmod(lexical, 0o700)
        except OSError as error:
            raise SmokeInfrastructureError(
                "product_storage_initialization_failed",
                component="product",
            ) from error


def _file_digest(path: Path) -> tuple[int, str]:
    digest = sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                size += len(chunk)
                digest.update(chunk)
    except OSError as error:
        raise SmokeInfrastructureError(
            "product_upload_read_failed",
            component="product",
        ) from error
    if size <= 0:
        raise SmokeContractError(
            "product_upload_contract_invalid",
            component="product",
        )
    return size, digest.hexdigest()


def _wait_for_product_job(
    repository: object,
    job_id: str,
    *,
    timeout_seconds: float,
) -> object:
    from videoscope.jobs import JobState

    deadline = monotonic() + timeout_seconds
    while monotonic() < deadline:
        job = repository.get_video_index_job(job_id)  # type: ignore[attr-defined]
        if job is None:
            raise SmokeContractError(
                "product_index_job_missing",
                component="product",
            )
        state = getattr(job, "state", None)
        if state is JobState.COMPLETE:
            return job
        if state in {JobState.FAILED, JobState.CANCELLED}:
            raise SmokeInfrastructureError(
                "product_index_failed",
                component="product",
            )
        sleep(0.1)
    raise SmokeInfrastructureError(
        "product_index_timeout",
        component="product",
    )


def _close_product_runtime(runtime: object) -> None:
    close = getattr(runtime, "close", None)
    if not callable(close):
        raise SmokeContractError(
            "product_runtime_contract_invalid",
            component="product",
        )
    deadline = monotonic() + 60
    while monotonic() < deadline:
        if close() is True:
            return
        sleep(0.1)
    raise SmokeInfrastructureError(
        "product_runtime_cleanup_failed",
        component="product",
    )


def _close_product_snapshot_environment(environment: object) -> None:
    close = getattr(environment, "close", None)
    if not callable(close):
        raise SmokeContractError(
            "product_snapshot_contract_invalid",
            component="product",
        )
    last_error: BaseException | None = None
    cancellation: BaseException | None = None
    for _attempt in range(_PRODUCT_SNAPSHOT_CLOSE_ATTEMPTS):
        try:
            close()
        except (KeyboardInterrupt, SystemExit) as interrupted:
            cancellation = cancellation or interrupted
            last_error = interrupted
            if getattr(environment, "is_closed", None) is True:
                raise cancellation
            continue
        except BaseException as error:
            last_error = error
            if getattr(environment, "is_closed", None) is True:
                raise SmokeInfrastructureError(
                    "product_snapshot_final_attestation_failed",
                    component="product",
                ) from error
            continue
        if getattr(environment, "is_closed", None) is True:
            if cancellation is not None:
                raise cancellation
            return
        last_error = SmokeContractError(
            "product_snapshot_close_contract_incomplete",
            component="product",
        )
    raise SmokeInfrastructureError(
        "product_snapshot_cleanup_failed",
        component="product",
    ) from last_error


def _retry_product_snapshot_cleanup(error: object) -> None:
    retry_cleanup = getattr(error, "retry_cleanup", None)
    if not callable(retry_cleanup):
        raise SmokeContractError(
            "product_snapshot_cleanup_contract_invalid",
            component="product",
        )
    last_error: BaseException | None = None
    cancellation: BaseException | None = None
    for _attempt in range(_PRODUCT_SNAPSHOT_CLOSE_ATTEMPTS):
        try:
            retry_cleanup()
        except (KeyboardInterrupt, SystemExit) as interrupted:
            cancellation = cancellation or interrupted
            last_error = interrupted
        except BaseException as cleanup_error:
            last_error = cleanup_error
        cleanup_pending = getattr(error, "cleanup_pending", None)
        if cleanup_pending is False:
            if cancellation is not None:
                raise cancellation
            return
        if cleanup_pending is not True:
            raise SmokeContractError(
                "product_snapshot_cleanup_contract_invalid",
                component="product",
            )
    raise SmokeInfrastructureError(
        "product_snapshot_cleanup_failed",
        component="product",
    ) from last_error


def _validate_product_search_execution_receipt(
    profile: object,
    value: object,
) -> object:
    from videoscope.benchmark.adapter import (
        ProductBenchmarkSearchAdapter,
        ProductSearchExecutionReceipt,
    )
    from videoscope.benchmark.profiles import BenchmarkProfile

    if isinstance(profile, BenchmarkProfile):
        plan = profile.search_plan
        selected_components: list[str] = []
        if plan.text_search != "disabled":
            selected_components.extend(("text_vectors", "lexical_text"))
        if plan.visual_search != "disabled":
            selected_components.append("visual_dense")
        if plan.temporal_refinement:
            selected_components.append("temporal_refinement")
        if plan.lighthouse:
            selected_components.append("lighthouse")
        if plan.reranker == "qwen":
            selected_components.append("qwen_verification")
        expected_configuration_identity = (
            ProductBenchmarkSearchAdapter._configuration(profile).identity
        )
    else:
        selected_components = []
        expected_configuration_identity = ""
    if (
        not isinstance(profile, BenchmarkProfile)
        or not isinstance(value, ProductSearchExecutionReceipt)
        or value.profile_id != profile.profile_id
        or value.profile_identity != profile.identity
        or value.search_configuration_identity
        != expected_configuration_identity
        or value.total_evidence_count <= 0
    ):
        raise SmokeContractError(
            "product_search_execution_receipt_invalid",
            component="product",
        )
    if value.invoked_component_ids != tuple(selected_components):
        raise SmokeContractError(
            "product_search_component_execution_unproven",
            component="product",
        )
    if any(
        value.component_input_count(component_id) <= 0
        or value.component_output_count(component_id) <= 0
        or value.component_evidence_count(component_id) <= 0
        for component_id in selected_components
    ):
        raise SmokeContractError(
            "product_search_component_execution_unproven",
            component="product",
        )
    return value


def _product_search_execution_payload(value: object) -> dict[str, object]:
    from videoscope.benchmark.adapter import ProductSearchExecutionReceipt

    if not isinstance(value, ProductSearchExecutionReceipt):
        raise SmokeContractError(
            "product_search_execution_receipt_invalid",
            component="product",
        )
    return {
        "schema_version": value.schema_version,
        "profile_identity": value.profile_identity,
        "search_configuration_identity": value.search_configuration_identity,
        "invoked_component_ids": list(value.invoked_component_ids),
        "component_input_counts": dict(value.component_input_counts),
        "component_output_counts": dict(value.component_output_counts),
        "component_evidence_counts": dict(value.component_evidence_counts),
    }


def run_disposable_product_integration(
    root: Path,
    fixture: SmokeFixture,
    _clients: SmokeClients,
    toolchain: object,
    settings: object,
) -> Mapping[str, object]:
    """Exercise production durable ingest and frozen generation-pinned search."""

    from videoscope.benchmark import (
        AssetProvenance,
        BenchmarkAsset,
        BenchmarkEnvironmentCleanupError,
        open_product_benchmark_environment,
    )
    from videoscope.benchmark.product_runtime import ProductSnapshotCleanupError
    from videoscope.benchmark.profiles import get_profile
    from videoscope.clips import ClipSelection
    from videoscope.processing.coordinator import VideoIndexCoordinator
    from videoscope.repository import Repository
    from videoscope.runtime import build_runtime
    from videoscope.media.uploads import validate_upload
    from videoscope.runtime_lifecycle import ExclusiveRuntimeLock

    if not isinstance(settings, _DisposableProductSettings):
        raise SmokeConfigurationError(
            "disposable_product_settings_required",
            component="product",
        )
    try:
        immutable_models = settings.models_dir.resolve(strict=True)
    except OSError as error:
        raise SmokeConfigurationError(
            "immutable_model_root_unavailable",
            component="product",
        ) from error
    if not immutable_models.is_dir() or immutable_models.is_relative_to(root):
        raise SmokeConfigurationError(
            "immutable_model_root_invalid",
            component="product",
        )

    _ensure_product_directories(settings, root)
    repository = Repository(settings.database_path)
    repository.initialize()
    runtime: object | None = None
    primary_error: BaseException | None = None
    evidence_count = 0
    profile_receipts: dict[str, dict[str, object]] = {}
    export_interval: tuple[float, float] | None = None
    clip_service: object | None = None
    try:
        runtime = build_runtime(
            settings,
            repository,
            indexing_toolchain=toolchain,
            ocr_worker_environment=settings.ocr_worker_environment,
            vision_worker_input_root=root,
        )
        runtime.start()
        plan_factory = getattr(runtime, "video_index_plan_factory", None)
        if not callable(plan_factory):
            raise SmokeContractError(
                "product_index_plan_unavailable",
                component="product",
            )

        video_id = f"smokeproduct{secrets.token_hex(12)}"
        stored_name = f"{video_id}.mp4"
        temporary = settings.temp_dir / f"{video_id}.upload"
        destination = settings.media_dir / stored_name
        _safe_copy(fixture.media_video, temporary)
        size_bytes, source_sha256 = _file_digest(temporary)
        validate_upload(
            "full-ml-smoke.mp4",
            size_bytes,
            max_bytes=settings.max_upload_bytes,
        )
        temporary.replace(destination)

        coordinator = VideoIndexCoordinator(
            repository,
            plan_factory=plan_factory,
            wakeup=_QueueWakeup(runtime.queue),
        )
        _video, job = coordinator.create_ingest(
            video_id=video_id,
            original_name="full-ml-smoke.mp4",
            stored_name=stored_name,
            media_path=str(destination),
            size_bytes=size_bytes,
            source_sha256=source_sha256,
        )
        _wait_for_product_job(
            repository,
            job.job_id,
            timeout_seconds=1800,
        )

        asset_records = repository.find_assets_by_sha256_bounded(
            source_sha256,
            limit=1,
            video_id=video_id,
        )
        if len(asset_records) != 1:
            raise SmokeContractError(
                "product_asset_binding_invalid",
                component="product",
            )
        asset = asset_records[0]
        portable_asset = BenchmarkAsset(
            asset_id="synthetic-smoke-asset",
            sha256=asset.sha256,
            byte_size=asset.byte_size,
            duration_seconds=asset.duration_seconds,
            provenance=AssetProvenance(
                source="Phase 0 generated synthetic fixture",
                license_id="project-generated-test-fixture",
            ),
        )
        clip_service = getattr(runtime, "clips", None)
        _close_product_runtime(runtime)
        runtime = None
        for profile_id in _EXECUTED_FROZEN_PROFILE_IDS:
            profile = get_profile(profile_id)
            environment: object | None = None
            session: object | None = None
            profile_error: BaseException | None = None
            profile_evidence_count = 0
            profile_execution_receipt: object | None = None
            try:
                try:
                    environment = open_product_benchmark_environment(
                        settings,
                        root,
                        profile_id=profile_id,
                        execution_mode="warm",
                    )
                except BenchmarkEnvironmentCleanupError as error:
                    try:
                        _close_product_snapshot_environment(error.environment)
                    except Exception as cleanup_error:
                        raise SmokeInfrastructureError(
                            "product_snapshot_cleanup_failed",
                            component="product",
                        ) from cleanup_error
                    raise SmokeInfrastructureError(
                        "product_snapshot_open_failed",
                        component="product",
                    ) from error
                except ProductSnapshotCleanupError as error:
                    try:
                        _retry_product_snapshot_cleanup(error)
                    except Exception as cleanup_error:
                        raise SmokeInfrastructureError(
                            "product_snapshot_cleanup_failed",
                            component="product",
                        ) from cleanup_error
                    raise SmokeInfrastructureError(
                        "product_snapshot_open_failed",
                        component="product",
                    ) from error
                probe_sources = getattr(environment, "probe_worker_sources", None)
                if not callable(probe_sources):
                    raise SmokeContractError(
                        "product_worker_source_probe_unavailable",
                        component="product",
                    )
                expected_probe_roles = {
                    "dense_siglip": ("vision",),
                    "temporal_refinement": ("vision",),
                    "lighthouse": ("vision",),
                    "qwen_verification": ("qwen", "vision"),
                }.get(profile_id, ())
                if probe_sources() != expected_probe_roles:
                    raise SmokeContractError(
                        "product_worker_source_probe_invalid",
                        component="product",
                    )
                asset_resolver = getattr(environment, "asset_resolver", None)
                resolve_asset = getattr(asset_resolver, "resolve", None)
                search_adapter = getattr(environment, "search_adapter", None)
                open_session = getattr(search_adapter, "open_session", None)
                if not callable(resolve_asset) or not callable(open_session):
                    raise SmokeContractError(
                        "product_snapshot_contract_invalid",
                        component="product",
                    )
                resolved_asset = resolve_asset(portable_asset)
                session = open_session(
                    profile,
                    (resolved_asset,),
                    execution_mode="warm",
                )
                if any(
                    session.capability_state(resolved_asset, capability) != "complete"
                    for capability in profile.required_capabilities
                ):
                    raise SmokeContractError(
                        "product_generation_binding_incomplete",
                        component="product",
                    )
                identities = session.identities()
                index_components = {
                    identity.component_id for identity in identities.index_identities
                }
                config_components = {
                    identity.component_id for identity in identities.config_identities
                }
                expected_index_components = {"text_vector_generations"}
                if "visual_dense" in profile.required_capabilities:
                    expected_index_components.add("visual_generations")
                if "lighthouse" in profile.required_capabilities:
                    expected_index_components.add("lighthouse_generations")
                if (
                    not expected_index_components <= index_components
                    or "benchmark_product_environment" not in config_components
                ):
                    raise SmokeContractError(
                        "product_generation_identity_incomplete",
                        component="product",
                    )
                results = session.search(
                    _QUERY,
                    (resolved_asset,),
                    limit=profile.search_plan.result_limit,
                )
                if not results:
                    raise SmokeContractError(
                        "product_search_returned_no_evidence",
                        component="product",
                    )
                for result in results:
                    if (
                        result.asset_id != portable_asset.asset_id
                        or not math.isfinite(float(result.start_seconds))
                        or not math.isfinite(float(result.end_seconds))
                        or result.start_seconds < 0
                        or result.end_seconds <= result.start_seconds
                        or result.end_seconds > asset.duration_seconds + 0.05
                    ):
                        raise SmokeContractError(
                            "product_search_evidence_invalid",
                            component="product",
                        )
                execution_receipt_reader = getattr(
                    session,
                    "last_search_execution_receipt",
                    None,
                )
                if not callable(execution_receipt_reader):
                    raise SmokeContractError(
                        "product_search_execution_receipt_invalid",
                        component="product",
                    )
                profile_execution_receipt = (
                    _validate_product_search_execution_receipt(
                        profile,
                        execution_receipt_reader(),
                    )
                )
                profile_evidence_count = int(
                    getattr(profile_execution_receipt, "total_evidence_count")
                )
                if profile_id == "lexical_qdrant":
                    first = results[0]
                    if first.end_seconds - first.start_seconds < 0.2:
                        raise SmokeContractError(
                            "product_export_interval_invalid",
                            component="product",
                        )
                    export_interval = (
                        float(first.start_seconds),
                        float(first.end_seconds),
                    )
            except BaseException as error:
                profile_error = error
                raise
            finally:
                cleanup_failure: BaseException | None = None
                if session is not None:
                    try:
                        session.close()
                    except BaseException as error:
                        if isinstance(error, (KeyboardInterrupt, SystemExit)):
                            cleanup_failure = error
                        else:
                            cleanup_failure = SmokeInfrastructureError(
                                "product_search_cleanup_failed",
                                component="product",
                            )
                            cleanup_failure.__cause__ = error
                if environment is not None:
                    try:
                        _close_product_snapshot_environment(environment)
                    except BaseException as error:
                        cleanup_failure = cleanup_failure or error
                if cleanup_failure is not None:
                    raise cleanup_failure from profile_error
            if profile_evidence_count <= 0:
                raise SmokeContractError(
                    "product_search_returned_no_evidence",
                    component="product",
                )
            if profile_execution_receipt is None:
                raise SmokeContractError(
                    "product_search_execution_receipt_invalid",
                    component="product",
                )
            evidence_count += profile_evidence_count
            profile_receipts[profile_id] = {
                "close": "complete",
                "status": "complete",
                "generation_bound": True,
                "evidence_count": profile_evidence_count,
                "component_execution": _product_search_execution_payload(
                    profile_execution_receipt
                ),
                "open": "complete",
                "search": "complete",
            }
        profile_receipts["internvideo"] = {
            "status": "not_configured",
            "reason_code": "provider_not_configured",
        }
        if evidence_count <= 0:
            raise SmokeContractError(
                "product_search_returned_no_evidence",
                component="product",
            )
        if export_interval is None:
            raise SmokeContractError(
                "product_export_interval_unavailable",
                component="product",
            )
        export = getattr(clip_service, "export", None)
        if not callable(export):
            raise SmokeContractError(
                "product_clip_export_unavailable",
                component="product",
            )
        export_lock = ExclusiveRuntimeLock(settings.data_dir)
        try:
            export_lock.acquire_existing()
            exported = export(
                "phase0-full-path",
                [
                    ClipSelection(
                        video_id=video_id,
                        start=export_interval[0],
                        end=export_interval[1],
                    )
                ],
            )
        except SmokeError:
            raise
        except Exception as error:
            raise SmokeInfrastructureError(
                "product_clip_export_failed",
                component="product",
            ) from error
        finally:
            try:
                export_lock.close()
            except Exception as error:
                raise SmokeInfrastructureError(
                    "product_export_lock_cleanup_failed",
                    component="product",
                ) from error
        exported_path = getattr(exported, "path", None)
        exported_duration = getattr(exported, "duration", None)
        if not isinstance(exported_path, Path):
            raise SmokeContractError(
                "product_clip_export_invalid",
                component="product",
            )
        try:
            output_metadata = exported_path.lstat()
            output_root = settings.clips_dir.resolve(strict=True)
            resolved_output = exported_path.resolve(strict=True)
        except OSError as error:
            raise SmokeInfrastructureError(
                "product_clip_export_unavailable",
                component="product",
            ) from error
        if (
            stat.S_ISLNK(output_metadata.st_mode)
            or not stat.S_ISREG(output_metadata.st_mode)
            or output_metadata.st_nlink != 1
            or output_metadata.st_size <= 0
            or not resolved_output.is_relative_to(output_root)
            or isinstance(exported_duration, bool)
            or not isinstance(exported_duration, (int, float))
            or not math.isfinite(float(exported_duration))
            or float(exported_duration) <= 0
        ):
            raise SmokeContractError(
                "product_clip_export_invalid",
                component="product",
            )
        create_ffmpeg = getattr(toolchain, "create_ffmpeg", None)
        if not callable(create_ffmpeg):
            raise SmokeContractError(
                "product_clip_probe_unavailable",
                component="product",
            )
        probe = create_ffmpeg().probe(resolved_output)
        probe_duration = getattr(probe, "duration", None)
        if (
            isinstance(probe_duration, bool)
            or not isinstance(probe_duration, (int, float))
            or not math.isfinite(float(probe_duration))
            or float(probe_duration) <= 0
            or abs(float(probe_duration) - float(exported_duration)) > 0.25
        ):
            raise SmokeContractError(
                "product_clip_probe_invalid",
                component="product",
            )
    except BaseException as error:
        primary_error = error
        raise
    finally:
        cleanup_error: SmokeError | None = None
        if runtime is not None:
            try:
                _close_product_runtime(runtime)
            except SmokeError as error:
                cleanup_error = cleanup_error or error
            except Exception:
                cleanup_error = cleanup_error or SmokeInfrastructureError(
                    "product_runtime_cleanup_failed",
                    component="product",
                )
        if cleanup_error is not None and primary_error is None:
            raise cleanup_error

    return {
        "status": "complete",
        "upload": "complete",
        "index": "complete",
        "search": "complete",
        "export": {
            "byte_size": output_metadata.st_size,
            "container": "mp4",
            "duration_seconds": float(probe_duration),
            "source_profile_id": "lexical_qdrant",
            "status": "complete",
        },
        "generation_bound": True,
        "evidence_count": evidence_count,
        "profiles": profile_receipts,
        "runtime_cleanup": "complete",
    }


def resolve_clean_code_sha() -> str:
    try:
        from videoscope.benchmark.cli import _current_code_sha

        return _current_code_sha()
    except SmokeError:
        raise
    except Exception as error:
        raise SmokeContractError(
            "code_identity_unavailable",
            component="code",
        ) from error


def _validated_code_sha(resolver: Callable[[], str]) -> str:
    try:
        code_sha = resolver()
    except SmokeError:
        raise
    except Exception as error:
        raise SmokeContractError(
            "code_identity_unavailable",
            component="code",
        ) from error
    if (
        type(code_sha) is not str
        or len(code_sha) not in {40, 64}
        or any(character not in _SHA256_CHARACTERS for character in code_sha)
    ):
        raise SmokeContractError(
            "code_identity_invalid",
            component="code",
        )
    return code_sha


def production_dependencies() -> SmokeDependencies:
    return SmokeDependencies(
        attest_ml_environment=_attest_current_ml_environment,
        attest_toolchain=_attest_current_toolchain,
        build_clients=build_real_clients,
        build_fixture=build_synthetic_fixture,
        build_resource_monitor=build_resource_monitor,
        code_identity_resolver=resolve_clean_code_sha,
        load_settings=_load_settings,
        product_integration=run_disposable_product_integration,
        start_workers=start_managed_workers,
    )


def build_resource_monitor(
    managed_workers: tuple[ManagedProcessBinding, ...],
) -> _NativeProcessTreeRSSMonitor:
    return _NativeProcessTreeRSSMonitor(managed_workers)


def _provider_ready(status: object) -> bool:
    if getattr(status, "ready", None) is True:
        return True
    state = getattr(status, "state", None)
    return getattr(state, "value", state) == ProviderState.READY.value


def _infrastructure_call(
    component: str,
    code: str,
    operation: Callable[[], object],
) -> object:
    try:
        return operation()
    except SmokeError:
        raise
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        if _is_out_of_memory(error):
            raise SmokeOutOfMemoryError(
                "out_of_memory",
                component=component,
            ) from error
        raise SmokeInfrastructureError(code, component=component) from error


def _is_out_of_memory(error: BaseException) -> bool:
    pending: list[BaseException] = [error]
    observed: set[int] = set()
    for _ in range(32):
        if not pending:
            return False
        candidate = pending.pop()
        identity = id(candidate)
        if identity in observed:
            continue
        observed.add(identity)
        if isinstance(candidate, MemoryError):
            return True
        message = str(candidate).casefold()
        if any(signature in message for signature in _OOM_SIGNATURES):
            return True
        nested = getattr(candidate, "exceptions", ())
        if isinstance(nested, tuple):
            pending.extend(
                item for item in nested if isinstance(item, BaseException)
            )
        cause = candidate.__cause__
        context = candidate.__context__
        if cause is not None:
            pending.append(cause)
        if context is not None:
            pending.append(context)
    return False


def _require_ready(component: str, status: object) -> None:
    if not _provider_ready(status):
        raise SmokeInfrastructureError(
            "worker_not_ready",
            component=component,
        )


def _validated_vectors(
    value: object,
    *,
    dimensions: int,
    component: str,
    code: str,
) -> np.ndarray:
    try:
        vectors = np.asarray(value, dtype=np.float32)
    except Exception as error:
        raise SmokeContractError(code, component=component) from error
    if (
        type(dimensions) is not int
        or dimensions <= 0
        or vectors.shape != (1, dimensions)
        or not bool(np.all(np.isfinite(vectors)))
        or float(np.linalg.norm(vectors[0])) <= 0
    ):
        raise SmokeContractError(code, component=component)
    return vectors


def _validated_sequence(
    value: object,
    *,
    component: str,
    code: str,
) -> Sequence[object]:
    if not isinstance(value, (list, tuple)):
        raise SmokeContractError(code, component=component)
    return value


def _validate_fixture(fixture: object, root: Path) -> SmokeFixture:
    if not isinstance(fixture, SmokeFixture):
        raise SmokeContractError("fixture_contract_invalid", component="ffmpeg")
    if (
        not math.isfinite(float(fixture.duration_seconds))
        or fixture.duration_seconds <= 0
        or not fixture.video_id.startswith("smoke_")
    ):
        raise SmokeContractError("fixture_contract_invalid", component="ffmpeg")
    for path in (fixture.image, fixture.media_video, fixture.qwen_video):
        try:
            resolved = path.resolve(strict=True)
            metadata = os.lstat(path)
        except OSError as error:
            raise SmokeContractError(
                "fixture_contract_invalid",
                component="ffmpeg",
            ) from error
        if (
            not resolved.is_relative_to(root)
            or not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_size <= 0
        ):
            raise SmokeContractError(
                "fixture_contract_invalid",
                component="ffmpeg",
            )
    return fixture


def _step(step_id: str, **observations: object) -> dict[str, object]:
    return {
        "id": step_id,
        "observations": observations,
        "status": "complete",
    }


def _run_components(
    settings: object,
    clients: SmokeClients,
    fixture: SmokeFixture,
) -> list[dict[str, object]]:
    steps: list[dict[str, object]] = []

    vision_status = _infrastructure_call(
        "vision",
        "health_failed",
        clients.vision.capability,  # type: ignore[attr-defined]
    )
    _require_ready("vision", vision_status)
    dimensions = getattr(
        getattr(clients.vision, "specification", None),
        "embedding_dimensions",
        None,
    )
    image_vectors = _infrastructure_call(
        "vision",
        "image_embedding_failed",
        lambda: clients.vision.image_vectors([fixture.image]),  # type: ignore[attr-defined]
    )
    image_vectors = _validated_vectors(
        image_vectors,
        dimensions=dimensions,
        component="vision",
        code="image_embedding_contract_invalid",
    )
    steps.append(
        _step(
            "vision.image_embedding",
            dimensions=int(image_vectors.shape[1]),
            vector_count=int(image_vectors.shape[0]),
        )
    )
    text_vectors = _infrastructure_call(
        "vision",
        "text_embedding_failed",
        lambda: clients.vision.text_vectors([_QUERY]),  # type: ignore[attr-defined]
    )
    text_vectors = _validated_vectors(
        text_vectors,
        dimensions=dimensions,
        component="vision",
        code="text_embedding_contract_invalid",
    )
    steps.append(
        _step(
            "vision.text_embedding",
            dimensions=int(text_vectors.shape[1]),
            vector_count=int(text_vectors.shape[0]),
        )
    )
    detections = _infrastructure_call(
        "vision",
        "rfdetr_failed",
        lambda: clients.vision.detect(fixture.image),  # type: ignore[attr-defined]
    )
    detections = _validated_sequence(
        detections,
        component="vision",
        code="rfdetr_contract_invalid",
    )
    steps.append(_step("vision.rfdetr", detection_count=len(detections)))

    _infrastructure_call(
        "vision",
        "ingestion_resource_release_failed",
        lambda: clients.vision.release_ingestion_resources(),  # type: ignore[attr-defined]
    )

    whisper_status = _infrastructure_call(
        "whisper",
        "health_failed",
        clients.whisper.status,  # type: ignore[attr-defined]
    )
    _require_ready("whisper", whisper_status)
    prompt = snapshot_whisper_prompt_from_content(
        getattr(settings, "whisper_initial_prompt", None),
        None,
        glossary_state="not_configured",
    )
    transcript = _infrastructure_call(
        "whisper",
        "transcription_failed",
        lambda: clients.whisper.transcribe(  # type: ignore[attr-defined]
            fixture.media_video,
            language=getattr(settings, "whisper_language", "auto"),
            prompt_snapshot=prompt,
        ),
    )
    transcript = _validated_sequence(
        transcript,
        component="whisper",
        code="transcription_contract_invalid",
    )
    steps.append(_step("whisper.transcribe", segment_count=len(transcript)))

    ocr_status = _infrastructure_call(
        "ocr",
        "health_failed",
        clients.ocr.status,  # type: ignore[attr-defined]
    )
    _require_ready("ocr", ocr_status)
    ocr_items = _infrastructure_call(
        "ocr",
        "read_failed",
        lambda: clients.ocr.read(fixture.image),  # type: ignore[attr-defined]
    )
    ocr_items = _validated_sequence(
        ocr_items,
        component="ocr",
        code="read_contract_invalid",
    )
    steps.append(_step("ocr.read", item_count=len(ocr_items)))

    lighthouse_status = _infrastructure_call(
        "lighthouse",
        "health_failed",
        clients.lighthouse.status,  # type: ignore[attr-defined]
    )
    _require_ready("lighthouse", lighthouse_status)
    descriptor = _infrastructure_call(
        "lighthouse",
        "generation_failed",
        lambda: clients.lighthouse.build_video_source(  # type: ignore[attr-defined]
            fixture.video_id,
            fixture.media_video,
            fixture.duration_seconds,
        ),
    )
    if not isinstance(descriptor, dict) or descriptor.get("video_id") != fixture.video_id:
        raise SmokeContractError(
            "generation_contract_invalid",
            component="lighthouse",
        )
    generation_id = descriptor.get("generation_id")
    if type(generation_id) is not str or len(generation_id) != 32:
        raise SmokeContractError(
            "generation_contract_invalid",
            component="lighthouse",
        )
    steps.append(_step("lighthouse.generation", generation_count=1))
    lighthouse_hits = _infrastructure_call(
        "lighthouse",
        "search_failed",
        lambda: clients.lighthouse.search_generations(  # type: ignore[attr-defined]
            _QUERY,
            {fixture.video_id: descriptor},
            limit=1,
        ),
    )
    lighthouse_hits = _validated_sequence(
        lighthouse_hits,
        component="lighthouse",
        code="search_contract_invalid",
    )
    steps.append(_step("lighthouse.search", hit_count=len(lighthouse_hits)))

    return steps


def _run_qwen_component(
    clients: SmokeClients,
    fixture: SmokeFixture,
) -> dict[str, object]:
    qwen_status = _infrastructure_call(
        "qwen",
        "health_failed",
        clients.qwen.status,  # type: ignore[attr-defined]
    )
    _require_ready("qwen", qwen_status)
    judgement = _infrastructure_call(
        "qwen",
        "judge_failed",
        lambda: clients.qwen.judge_video(  # type: ignore[attr-defined]
            fixture.qwen_video,
            fps=1.0,
            max_tokens=_QWEN_SMOKE_MAX_TOKENS,
        ),
    )
    if not isinstance(judgement, QwenVideoJudgement):
        raise SmokeContractError("judge_contract_invalid", component="qwen")
    return _step("qwen.judge", judgement_count=1)


def _validate_serialized_product_search_execution(
    profile_id: str,
    evidence_count: int,
    value: object,
) -> dict[str, object]:
    from videoscope.benchmark.adapter import ProductSearchExecutionReceipt
    from videoscope.benchmark.profiles import get_profile

    if not isinstance(value, Mapping):
        raise SmokeContractError(
            "product_integration_receipt_invalid",
            component="product",
        )
    payload = dict(value)
    invoked = payload.get("invoked_component_ids")
    input_counts = payload.get("component_input_counts")
    output_counts = payload.get("component_output_counts")
    component_evidence_counts = payload.get("component_evidence_counts")
    if (
        set(payload)
        != {
            "schema_version",
            "profile_identity",
            "search_configuration_identity",
            "invoked_component_ids",
            "component_input_counts",
            "component_output_counts",
            "component_evidence_counts",
        }
        or not isinstance(invoked, list)
        or not isinstance(input_counts, Mapping)
        or not isinstance(output_counts, Mapping)
        or not isinstance(component_evidence_counts, Mapping)
        or set(input_counts) != set(_PRODUCT_SEARCH_COMPONENT_IDS)
        or set(output_counts) != set(_PRODUCT_SEARCH_COMPONENT_IDS)
        or set(component_evidence_counts) != set(_PRODUCT_SEARCH_COMPONENT_IDS)
    ):
        raise SmokeContractError(
            "product_integration_receipt_invalid",
            component="product",
        )
    try:
        receipt = ProductSearchExecutionReceipt(
            schema_version=payload["schema_version"],
            profile_id=profile_id,
            profile_identity=payload["profile_identity"],
            search_configuration_identity=payload[
                "search_configuration_identity"
            ],
            total_evidence_count=evidence_count,
            invoked_component_ids=tuple(invoked),
            component_input_counts=tuple(
                (component_id, input_counts[component_id])
                for component_id in _PRODUCT_SEARCH_COMPONENT_IDS
            ),
            component_output_counts=tuple(
                (component_id, output_counts[component_id])
                for component_id in _PRODUCT_SEARCH_COMPONENT_IDS
            ),
            component_evidence_counts=tuple(
                (component_id, component_evidence_counts[component_id])
                for component_id in _PRODUCT_SEARCH_COMPONENT_IDS
            ),
        )
        _validate_product_search_execution_receipt(
            get_profile(profile_id),
            receipt,
        )
    except (KeyError, TypeError, ValueError, SmokeContractError) as error:
        raise SmokeContractError(
            "product_integration_receipt_invalid",
            component="product",
        ) from error
    return _product_search_execution_payload(receipt)


def _validate_product_receipt(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise SmokeContractError(
            "product_integration_receipt_invalid",
            component="product",
        )
    receipt = dict(value)
    profiles = receipt.get("profiles")
    export = receipt.get("export")
    expected_fields = {
        "status",
        "upload",
        "index",
        "search",
        "export",
        "generation_bound",
        "evidence_count",
        "profiles",
        "runtime_cleanup",
    }
    if (
        set(receipt) != expected_fields
        or receipt.get("status") != "complete"
        or receipt.get("upload") != "complete"
        or receipt.get("index") != "complete"
        or receipt.get("search") != "complete"
        or receipt.get("generation_bound") is not True
        or type(receipt.get("evidence_count")) is not int
        or int(receipt["evidence_count"]) <= 0
        or not isinstance(profiles, Mapping)
        or set(profiles) != set(_ALL_FROZEN_PROFILE_IDS)
        or receipt.get("runtime_cleanup") != "complete"
    ):
        raise SmokeContractError(
            "product_integration_receipt_invalid",
            component="product",
        )
    if (
        not isinstance(export, Mapping)
        or set(export)
        != {
            "byte_size",
            "container",
            "duration_seconds",
            "source_profile_id",
            "status",
        }
        or export.get("status") != "complete"
        or export.get("container") != "mp4"
        or export.get("source_profile_id") != "lexical_qdrant"
        or type(export.get("byte_size")) is not int
        or int(export["byte_size"]) <= 0
        or isinstance(export.get("duration_seconds"), bool)
        or not isinstance(export.get("duration_seconds"), (int, float))
        or not math.isfinite(float(export["duration_seconds"]))
        or float(export["duration_seconds"]) <= 0
    ):
        raise SmokeContractError(
            "product_integration_receipt_invalid",
            component="product",
        )
    normalized_profiles: dict[str, dict[str, object]] = {}
    observed_evidence_count = 0
    for profile_id in _EXECUTED_FROZEN_PROFILE_IDS:
        profile_receipt = profiles.get(profile_id)
        if (
            not isinstance(profile_receipt, Mapping)
            or set(profile_receipt)
            != {
                "close",
                "status",
                "generation_bound",
                "evidence_count",
                "component_execution",
                "open",
                "search",
            }
            or profile_receipt.get("status") != "complete"
            or profile_receipt.get("open") != "complete"
            or profile_receipt.get("search") != "complete"
            or profile_receipt.get("close") != "complete"
            or profile_receipt.get("generation_bound") is not True
            or type(profile_receipt.get("evidence_count")) is not int
            or int(profile_receipt["evidence_count"]) <= 0
        ):
            raise SmokeContractError(
                "product_integration_receipt_invalid",
                component="product",
            )
        count = int(profile_receipt["evidence_count"])
        component_execution = _validate_serialized_product_search_execution(
            profile_id,
            count,
            profile_receipt.get("component_execution"),
        )
        observed_evidence_count += count
        normalized_profiles[profile_id] = {
            "close": "complete",
            "status": "complete",
            "generation_bound": True,
            "evidence_count": count,
            "component_execution": component_execution,
            "open": "complete",
            "search": "complete",
        }
    internvideo = profiles.get("internvideo")
    if (
        not isinstance(internvideo, Mapping)
        or internvideo.get("status") != "not_configured"
        or internvideo.get("reason_code") != "provider_not_configured"
    ):
        raise SmokeContractError(
            "product_integration_receipt_invalid",
            component="product",
        )
    normalized_profiles["internvideo"] = {
        "status": "not_configured",
        "reason_code": "provider_not_configured",
    }
    if observed_evidence_count != int(receipt["evidence_count"]):
        raise SmokeContractError(
            "product_integration_receipt_invalid",
            component="product",
        )
    return {
        "status": "complete",
        "upload": "complete",
        "index": "complete",
        "search": "complete",
        "export": {
            "byte_size": int(export["byte_size"]),
            "container": "mp4",
            "duration_seconds": float(export["duration_seconds"]),
            "source_profile_id": "lexical_qdrant",
            "status": "complete",
        },
        "generation_bound": True,
        "evidence_count": observed_evidence_count,
        "profiles": normalized_profiles,
        "runtime_cleanup": "complete",
    }


def _restore_disposable_baseline(
    root: Path,
    baseline: _WorkspaceBaseline,
) -> None:
    observed: set[str] = set()

    def matches(metadata: os.stat_result, expected: _BaselineEntry) -> bool:
        return (
            metadata.st_dev == expected.device
            and metadata.st_ino == expected.inode
            and stat.S_IMODE(metadata.st_mode) == expected.mode
            and metadata.st_uid == expected.owner
        )

    def visit(directory: Path, relative_parts: tuple[str, ...]) -> None:
        try:
            with os.scandir(directory) as iterator:
                children = tuple(iterator)
        except OSError as error:
            raise SmokeInfrastructureError(
                "workspace_cleanup_failed",
                component="workspace",
            ) from error
        for child in children:
            relative = "/".join((*relative_parts, child.name))
            path = Path(child.path)
            expected = baseline.entries.get(relative)
            try:
                metadata = child.stat(follow_symlinks=False)
            except OSError as error:
                raise SmokeInfrastructureError(
                    "workspace_cleanup_failed",
                    component="workspace",
                ) from error
            if expected is None:
                try:
                    if stat.S_ISDIR(metadata.st_mode):
                        shutil.rmtree(path)
                    else:
                        path.unlink()
                except OSError as error:
                    raise SmokeInfrastructureError(
                        "workspace_cleanup_failed",
                        component="workspace",
                    ) from error
                continue
            observed.add(relative)
            if expected.kind == "directory":
                if not stat.S_ISDIR(metadata.st_mode) or not matches(metadata, expected):
                    raise SmokeInfrastructureError(
                        "workspace_baseline_changed",
                        component="workspace",
                    )
                visit(path, (*relative_parts, child.name))
                continue
            if (
                expected.kind != "file"
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_size != expected.size
                or not matches(metadata, expected)
                or _baseline_file_digest(path, metadata) != expected.digest
            ):
                raise SmokeInfrastructureError(
                    "workspace_baseline_changed",
                    component="workspace",
                )

    try:
        visit(root, ())
    except SmokeConfigurationError as error:
        raise SmokeInfrastructureError(
            "workspace_baseline_changed",
            component="workspace",
        ) from error
    if observed != set(baseline.entries):
        raise SmokeInfrastructureError(
            "workspace_baseline_changed",
            component="workspace",
        )


def _measurement_field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _managed_ocr_environment_paths(
    value: object,
) -> dict[str, Path] | None:
    if type(value) is not dict or set(value) != _MANAGED_OCR_ENVIRONMENT_FIELDS:
        return None
    paths: dict[str, Path] = {}
    for name in _MANAGED_OCR_ENVIRONMENT_FIELDS:
        raw = value.get(name)
        configured = _configured_path(raw)
        if (
            configured is None
            or type(raw) is not str
            or not Path(raw).is_absolute()
            or configured != Path(raw)
        ):
            return None
        paths[name] = configured
    if (
        paths["HOME"] != paths["TMPDIR"]
        or paths["XDG_CACHE_HOME"] != paths["HOME"] / "cache"
    ):
        return None
    return paths


def _validate_worker_cluster(value: object) -> object:
    settings_overrides = getattr(value, "settings_overrides", None)
    managed_workers = getattr(value, "managed_workers", None)
    all_workers_managed = getattr(value, "all_workers_managed", None)
    if (
        not isinstance(settings_overrides, Mapping)
        or not isinstance(managed_workers, tuple)
        or type(all_workers_managed) is not bool
        or not callable(getattr(value, "wait_ready", None))
        or not callable(getattr(value, "close", None))
    ):
        raise SmokeContractError(
            "managed_worker_cluster_contract_invalid",
            component="workers",
        )
    if all_workers_managed and (
        any(
            not isinstance(worker, ManagedProcessBinding)
            for worker in managed_workers
        )
        or tuple(worker.role for worker in managed_workers)
        != _MANAGED_WORKER_ROLES
        or set(settings_overrides)
        != {
            "vision_worker_endpoint",
            "vision_worker_api_key",
            "whisper_worker_endpoint",
            "whisper_worker_api_key",
            "lighthouse_endpoint",
            "lighthouse_api_key",
            "qwen_video_endpoint",
            "qwen_video_api_key",
            "ocr_worker_python",
            "ocr_worker_script",
            "smoke_ocr_client_environment",
            "ocr_worker_environment",
        }
    ):
        raise SmokeContractError(
            "managed_worker_cluster_contract_invalid",
            component="workers",
        )
    if all_workers_managed:
        client_environment = _managed_ocr_environment_paths(
            settings_overrides["smoke_ocr_client_environment"]
        )
        product_environment = _managed_ocr_environment_paths(
            settings_overrides["ocr_worker_environment"]
        )
        if (
            client_environment is None
            or product_environment is None
            or client_environment["HF_HOME"] != product_environment["HF_HOME"]
            or client_environment["PATH"] != product_environment["PATH"]
            or client_environment["HOME"] == product_environment["HOME"]
            or client_environment["HOME"].name != "worker-ocr-client"
            or product_environment["HOME"].name != "worker-ocr-product"
            or client_environment["HOME"].parent
            != product_environment["HOME"].parent
        ):
            raise SmokeContractError(
                "managed_worker_cluster_contract_invalid",
                component="workers",
            )
    if not all_workers_managed and (managed_workers or settings_overrides):
        raise SmokeContractError(
            "managed_worker_cluster_contract_invalid",
            component="workers",
        )
    return value


def _validate_process_tree_measurement(value: object) -> dict[str, object]:
    fields = {
        name: _measurement_field(value, name)
        for name in (
            "baseline_bytes",
            "increment_bytes",
            "peak_bytes",
            "sample_count",
            "samples_bytes",
        )
    }
    samples = fields["samples_bytes"]
    if (
        any(
            type(fields[name]) is not int
            for name in (
                "baseline_bytes",
                "increment_bytes",
                "peak_bytes",
                "sample_count",
            )
        )
        or int(fields["sample_count"]) <= 0
        or not isinstance(samples, tuple)
        or len(samples) != int(fields["sample_count"])
        or len(samples) > MAX_MEASUREMENT_RSS_SAMPLES
        or any(
            type(sample) is not int or not 0 <= sample <= _MAX_RSS_BYTES
            for sample in samples
        )
        or not 0 <= int(fields["baseline_bytes"]) <= _MAX_RSS_BYTES
        or not 0 <= int(fields["peak_bytes"]) <= _MAX_RSS_BYTES
        or not 0 <= int(fields["increment_bytes"]) <= _MAX_RSS_BYTES
        or int(fields["baseline_bytes"]) != samples[0]
        or int(fields["peak_bytes"]) != max(samples)
        or int(fields["increment_bytes"])
        != int(fields["peak_bytes"]) - int(fields["baseline_bytes"])
    ):
        raise SmokeContractError(
            "rss_measurement_contract_invalid",
            component="resources",
        )
    return {
        "baseline_bytes": int(fields["baseline_bytes"]),
        "increment_bytes": int(fields["increment_bytes"]),
        "peak_bytes": int(fields["peak_bytes"]),
        "sample_count": int(fields["sample_count"]),
        "samples_bytes": samples,
    }


def _validate_resource_measurement(value: object) -> dict[str, object]:
    process_value = _measurement_field(value, "process_tree")
    host_value = _measurement_field(value, "host_resources")
    process_tree = _validate_process_tree_measurement(process_value)
    if not isinstance(host_value, HostResourceMeasurementReceipt):
        raise SmokeContractError(
            "host_resource_measurement_contract_invalid",
            component="resources",
        )
    try:
        host_resources = host_value.to_portable_dict()
        json.dumps(
            host_resources,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise SmokeContractError(
            "host_resource_measurement_contract_invalid",
            component="resources",
        ) from error
    return {
        "process_tree": process_tree,
        "host_resources": host_resources,
    }


def _resource_report(
    measurement: Mapping[str, object],
    *,
    all_workers_managed: bool,
) -> dict[str, object]:
    process_tree = measurement["process_tree"]
    host_resources = measurement["host_resources"]
    if not isinstance(process_tree, Mapping) or not isinstance(
        host_resources,
        Mapping,
    ):
        raise SmokeContractError(
            "resource_measurement_contract_invalid",
            component="resources",
        )
    metal = host_resources["metal"]
    virtual_memory = host_resources["virtual_memory"]
    if not isinstance(metal, Mapping) or not isinstance(
        virtual_memory,
        Mapping,
    ):
        raise SmokeContractError(
            "resource_measurement_contract_invalid",
            component="resources",
        )
    in_use = metal["in_use_system_memory"]
    if not isinstance(in_use, Mapping):
        raise SmokeContractError(
            "resource_measurement_contract_invalid",
            component="resources",
        )
    return {
        "host_resources": host_resources,
        "measurement_caveat": {
            "memory_accounting": host_resources["memory_accounting"],
            "scope": host_resources["scope"],
        },
        "oom": {"status": "not_observed"},
        "peak_metal_bytes": {
            "baseline_bytes": in_use["baseline_bytes"],
            "increment_bytes": in_use["increment_bytes"],
            "scope": host_resources["scope"],
            "status": "measured",
            "value": in_use["peak_bytes"],
        },
        "sampled_peak_process_tree_rss_bytes": {
            "baseline_bytes": process_tree["baseline_bytes"],
            "external_loopback_workers_included": all_workers_managed,
            "increment_bytes": process_tree["increment_bytes"],
            "managed_worker_roles": (
                list(_MANAGED_WORKER_ROLES) if all_workers_managed else []
            ),
            "sample_count": process_tree["sample_count"],
            "samples_bytes": list(process_tree["samples_bytes"]),
            "sampling_interval_ms": int(
                PROCESS_RSS_SAMPLE_INTERVAL_SECONDS * 1000
            ),
            "scope": "smoke_process_and_descendants",
            "status": "measured",
            "value": process_tree["peak_bytes"],
        },
        "system_wide_pressure_deltas": {
            "metal_recovery_count": metal["recovery_delta"],
            "swapins_bytes": virtual_memory["swapins_delta_bytes"],
            "swapins_pages": virtual_memory["swapins_delta_pages"],
            "swapouts_bytes": virtual_memory["swapouts_delta_bytes"],
            "swapouts_pages": virtual_memory["swapouts_delta_pages"],
        },
    }


def execute(
    root: Path,
    *,
    models_root: Path,
    environ: Mapping[str, str],
    dependencies: SmokeDependencies | None = None,
) -> dict[str, object]:
    production_run = dependencies is None
    resolved_root, root_identity = _validate_disposable_root(root)
    _validate_offline_environment(resolved_root, environ)
    resolved_models_root = _validate_models_root(
        resolved_root,
        models_root,
        environ,
    )
    workspace_baseline = _snapshot_workspace_baseline(resolved_root)
    selected = dependencies or production_dependencies()
    code_sha = _validated_code_sha(selected.code_identity_resolver)
    try:
        ml_environment = _validated_ml_environment_receipt(
            selected.attest_ml_environment(resolved_root, environ)
        )
    except SmokeError:
        raise
    except Exception as error:
        raise SmokeInfrastructureError(
            "ml_environment_attestation_failed",
            component="environment",
        ) from error
    try:
        toolchain = selected.attest_toolchain(environ)
        toolchain_identity = toolchain.verify_current()
    except SmokeError:
        raise
    except Exception as error:
        if _is_out_of_memory(error):
            raise SmokeOutOfMemoryError(
                "out_of_memory",
                component="ffmpeg",
            ) from error
        raise SmokeInfrastructureError(
            "attestation_failed",
            component="ffmpeg",
        ) from error
    if not _is_sha256(toolchain_identity, prefix=True):
        raise SmokeContractError(
            "attestation_contract_invalid",
            component="ffmpeg",
        )
    execution_environment = (
        _bind_attested_media_environment(environ, toolchain)
        if production_run
        else dict(environ)
    )

    worker_cluster: object | None = None
    resource_monitor: object | None = None
    monitor_started = False
    finish_monitor: Callable[[], object] | None = None
    close_monitor: Callable[[], object] | None = None
    clients: SmokeClients | None = None
    primary_error: BaseException | None = None
    resource_measurement: dict[str, object] | None = None
    settings: object | None = None
    product: dict[str, object] | None = None
    status: str | None = None
    steps: list[dict[str, object]] | None = None
    code_sha_after: str | None = None
    try:
        workspace = _allocate_workspace(resolved_root)
        try:
            worker_cluster = _validate_worker_cluster(
                selected.start_workers(
                    resolved_root,
                    resolved_models_root,
                    execution_environment,
                )
            )
        except SmokeError:
            raise
        except Exception as error:
            if _is_out_of_memory(error):
                raise SmokeOutOfMemoryError(
                    "out_of_memory",
                    component="workers",
                ) from error
            raise SmokeInfrastructureError(
                "managed_worker_start_failed",
                component="workers",
            ) from error

        try:
            resource_monitor = selected.build_resource_monitor(
                worker_cluster.managed_workers  # type: ignore[attr-defined]
            )
            start_monitor = getattr(resource_monitor, "start", None)
            finish_monitor = getattr(resource_monitor, "finish", None)
            close_monitor = getattr(resource_monitor, "close", None)
            if not all(
                callable(item)
                for item in (start_monitor, finish_monitor, close_monitor)
            ):
                raise SmokeContractError(
                    "rss_monitor_contract_invalid",
                    component="resources",
                )
            start_monitor()
            monitor_started = True
        except SmokeError:
            raise
        except Exception as error:
            if _is_out_of_memory(error):
                raise SmokeOutOfMemoryError(
                    "out_of_memory",
                    component="resources",
                ) from error
            raise SmokeInfrastructureError(
                "rss_monitor_start_failed",
                component="resources",
            ) from error

        try:
            worker_cluster.wait_ready()  # type: ignore[attr-defined]
        except SmokeError:
            raise
        except Exception as error:
            raise SmokeInfrastructureError(
                "managed_worker_readiness_failed",
                component="workers",
            ) from error

        try:
            settings_overrides = dict(
                worker_cluster.settings_overrides  # type: ignore[attr-defined]
            )
            if production_run:
                ffmpeg_binary, ffprobe_binary = _validated_media_executables(
                    execution_environment
                )
                settings_overrides.update(
                    {
                        "ffmpeg_binary": ffmpeg_binary,
                        "ffprobe_binary": ffprobe_binary,
                    }
                )
            settings = selected.load_settings(
                resolved_root,
                resolved_models_root,
                settings_overrides,
            )
        except SmokeError:
            raise
        except Exception as error:
            raise SmokeConfigurationError(
                "worker_configuration_invalid"
            ) from error
        _validate_required_settings(settings)

        fixture = _validate_fixture(
            _infrastructure_call(
                "ffmpeg",
                "synthetic_fixture_failed",
                lambda: selected.build_fixture(toolchain, workspace),
            ),
            resolved_root,
        )
        try:
            clients = selected.build_clients(settings, resolved_root)
        except SmokeError:
            raise
        except Exception as error:
            if _is_out_of_memory(error):
                raise SmokeOutOfMemoryError(
                    "out_of_memory",
                    component="workers",
                ) from error
            raise SmokeInfrastructureError(
                "client_construction_failed",
                component="workers",
            ) from error
        if not isinstance(clients, SmokeClients):
            raise SmokeContractError(
                "client_bundle_invalid",
                component="workers",
            )
        steps = _run_components(settings, clients, fixture)
        if selected.product_integration is None:
            raise SmokeContractError(
                "product_integration_not_configured",
                component="product",
            )
        product = _validate_product_receipt(
            _infrastructure_call(
                "product",
                "product_integration_failed",
                lambda: selected.product_integration(
                    resolved_root,
                    fixture,
                    clients,
                    toolchain,
                    settings,
                ),
            )
        )
        steps.append(_run_qwen_component(clients, fixture))
        status = "ready"
    except BaseException as error:
        primary_error = error
        raise
    finally:
        cleanup_error: SmokeError | None = None
        if clients is not None:
            try:
                close = getattr(clients.ocr, "close", None)
                if callable(close):
                    close()
            except Exception:
                cleanup_error = SmokeInfrastructureError(
                    "ocr_cleanup_failed",
                    component="ocr",
                )
        try:
            current_identity = toolchain.verify_current()
            if current_identity != toolchain_identity:
                raise SmokeContractError(
                    "attestation_changed",
                    component="ffmpeg",
                )
        except SmokeError as error:
            cleanup_error = cleanup_error or error
        except Exception:
            cleanup_error = cleanup_error or SmokeInfrastructureError(
                "attestation_failed",
                component="ffmpeg",
            )
        if monitor_started and finish_monitor is not None:
            try:
                resource_measurement = _validate_resource_measurement(
                    finish_monitor()
                )
            except SmokeError as error:
                cleanup_error = cleanup_error or error
            except Exception:
                cleanup_error = cleanup_error or SmokeInfrastructureError(
                    "rss_measurement_failed",
                    component="resources",
                )
        if close_monitor is not None:
            try:
                close_monitor()
            except SmokeError as error:
                cleanup_error = cleanup_error or error
            except Exception:
                cleanup_error = cleanup_error or SmokeInfrastructureError(
                    "rss_monitor_cleanup_failed",
                    component="resources",
                )
        if worker_cluster is not None:
            try:
                worker_cluster.close()  # type: ignore[attr-defined]
            except SmokeError as error:
                cleanup_error = cleanup_error or error
            except Exception:
                cleanup_error = cleanup_error or SmokeInfrastructureError(
                    "managed_worker_cleanup_failed",
                    component="workers",
                )
        try:
            _verify_disposable_root(resolved_root, root_identity)
            _restore_disposable_baseline(
                resolved_root,
                workspace_baseline,
            )
            _verify_disposable_root(resolved_root, root_identity)
        except SmokeError as error:
            cleanup_error = cleanup_error or error
        try:
            code_sha_after = _validated_code_sha(selected.code_identity_resolver)
            if code_sha_after != code_sha:
                raise SmokeContractError(
                    "code_identity_changed",
                    component="code",
                )
        except SmokeError as error:
            cleanup_error = cleanup_error or error
        if cleanup_error is not None and primary_error is None:
            raise cleanup_error

    if (
        resource_measurement is None
        or worker_cluster is None
        or product is None
        or status is None
        or steps is None
        or code_sha_after is None
    ):
        raise SmokeInfrastructureError(
            "smoke_result_unavailable",
            component="smoke",
        )
    return {
        "code_sha_after": code_sha_after,
        "code_sha_before": code_sha,
        "environment_bindings": ml_environment["environment_bindings"],
        "ml_environment_attestation_id": ml_environment[
            "ml_environment_attestation_id"
        ],
        "ml_environment_manifest_identity": ml_environment[
            "ml_environment_manifest_identity"
        ],
        "offline": True,
        "product_integration": product,
        "resources": _resource_report(
            resource_measurement,
            all_workers_managed=worker_cluster.all_workers_managed,  # type: ignore[attr-defined]
        ),
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "steps": steps,
        "toolchain_identity": toolchain_identity,
        "workspace_cleanup": "complete",
    }


def _error_payload(error: SmokeError) -> dict[str, object]:
    details: dict[str, object] = {
        "code": error.code,
        "kind": error.kind,
    }
    if error.component is not None:
        details["component"] = error.component
    payload: dict[str, object] = {
        "error": details,
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
    }
    if isinstance(error, SmokeOutOfMemoryError):
        payload["oom"] = {"status": "observed"}
    elif (
        isinstance(error, SmokeInfrastructureError)
        and error.component in _REMOTE_WORKER_COMPONENTS
    ):
        payload["oom"] = {
            "reason_code": "worker_failure_contract_missing_oom_code",
            "status": "unknown",
        }
    return payload


def _write_json(stream: object, value: object) -> None:
    stream.write(  # type: ignore[attr-defined]
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--models-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        report = execute(
            arguments.root,
            models_root=arguments.models_root,
            environ=os.environ,
        )
    except SmokeConfigurationError as error:
        _write_json(sys.stderr, _error_payload(error))
        return EXIT_CONFIGURATION
    except SmokeInfrastructureError as error:
        _write_json(sys.stderr, _error_payload(error))
        return EXIT_INFRASTRUCTURE
    except SmokeOutOfMemoryError as error:
        _write_json(sys.stderr, _error_payload(error))
        return EXIT_OUT_OF_MEMORY
    except SmokeContractError as error:
        _write_json(sys.stderr, _error_payload(error))
        return EXIT_CONTRACT
    except KeyboardInterrupt:
        _write_json(
            sys.stderr,
            {
                "error": {"code": "interrupted", "kind": "interrupted"},
                "schema_version": SCHEMA_VERSION,
                "status": "failed",
            },
        )
        return EXIT_INTERRUPTED
    except SystemExit:
        raise
    except BaseException:
        _write_json(
            sys.stderr,
            {
                "error": {"code": "internal_error", "kind": "internal"},
                "schema_version": SCHEMA_VERSION,
                "status": "failed",
            },
        )
        return EXIT_INTERNAL
    _write_json(sys.stdout, report)
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
