from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
import secrets
import socket
import stat
import subprocess
from time import monotonic, sleep

from videoscope.model_manifest import (
    QWEN_VIDEO_MODEL,
    SIGLIP_224_MODEL,
    WHISPER_MODEL,
    model_revision,
)
from videoscope.providers.vision_worker_contract import (
    RFDETR_SMALL_CHECKPOINT_SHA256,
)

from .measurements import (
    ManagedProcessBinding,
    ProcessRecord,
    create_native_process_snapshot_provider,
)
from .serialization import expect_fields, expect_object, parse_json_object
from .storage import _read_bounded_file


WORKER_LAUNCH_SCHEMA_VERSION = 1
MAX_WORKER_LAUNCH_MANIFEST_BYTES = 64 * 1024
MANAGED_WORKER_ROLES = (
    "vision_index",
    "vision",
    "whisper",
    "lighthouse",
    "qwen",
)
_EXECUTABLE_ROLES = ("vision", "whisper", "lighthouse", "qwen", "ocr")
_START_TIMEOUT_SECONDS = 300.0
_BINDING_TIMEOUT_SECONDS = 5.0
_PROJECT_ROOT = Path(__file__).resolve().parents[4]


class ManagedWorkerError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"managed regression workers failed ({code})")


@dataclass(frozen=True, slots=True)
class WorkerLaunchConfiguration:
    schema_version: int
    executables: tuple[tuple[str, Path], ...]
    hf_home: Path
    ocr_model_root: Path
    ffmpeg_binary: Path
    ffprobe_binary: Path

    def executable(self, role: str) -> Path:
        try:
            return dict(self.executables)[role]
        except KeyError as exc:
            raise ManagedWorkerError("worker_executable_missing") from exc


@dataclass(frozen=True, slots=True)
class _WorkerProcess:
    role: str
    port: int
    process: subprocess.Popen[bytes]


class ManagedRegressionWorkerCluster:
    """Own every HTTP ML process used by one Phase-0 regression batch."""

    def __init__(
        self,
        *,
        processes: tuple[_WorkerProcess, ...],
        managed_workers: tuple[ManagedProcessBinding, ...],
        ingest_overrides: Mapping[str, object],
        benchmark_overrides: Mapping[str, object],
        ocr_model_root: Path,
    ) -> None:
        self._processes = processes
        self.managed_workers = managed_workers
        self.ingest_overrides = dict(ingest_overrides)
        self.benchmark_overrides = dict(benchmark_overrides)
        self.ocr_model_root = ocr_model_root
        self.retired_worker_roles: tuple[str, ...] = ()
        self._closed = False

    def wait_ready(self) -> None:
        pending = {item.role: item for item in self._processes}
        deadline = monotonic() + _START_TIMEOUT_SECONDS
        while pending and monotonic() < deadline:
            for role, worker in tuple(pending.items()):
                if worker.process.poll() is not None:
                    raise ManagedWorkerError("managed_worker_start_failed")
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
            raise ManagedWorkerError("managed_worker_readiness_timeout")

    def close(self) -> None:
        if self._closed:
            return
        self._stop_processes(self._processes)
        self._closed = True

    def retire_ingest_workers(self) -> None:
        """Release workers not used by retrieval before RSS measurement begins."""

        if self._closed or self.retired_worker_roles:
            raise ManagedWorkerError("managed_worker_lifecycle_invalid")
        retired_roles = ("vision_index", "whisper")
        retired = tuple(
            item for item in self._processes if item.role in retired_roles
        )
        if tuple(item.role for item in retired) != retired_roles:
            raise ManagedWorkerError("managed_worker_lifecycle_invalid")
        self._stop_processes(retired)
        self._processes = tuple(
            item for item in self._processes if item.role not in retired_roles
        )
        self.managed_workers = tuple(
            item for item in self.managed_workers if item.role not in retired_roles
        )
        self.retired_worker_roles = retired_roles

    @staticmethod
    def _stop_processes(processes: tuple[_WorkerProcess, ...]) -> None:
        failed = False
        for worker in reversed(processes):
            try:
                if worker.process.poll() is None:
                    worker.process.terminate()
            except Exception:
                failed = True
        deadline = monotonic() + 15.0
        while monotonic() < deadline:
            try:
                if all(item.process.poll() is not None for item in processes):
                    break
            except Exception:
                failed = True
                break
            sleep(0.05)
        for worker in reversed(processes):
            try:
                if worker.process.poll() is None:
                    worker.process.kill()
                worker.process.wait(timeout=5.0)
            except Exception:
                failed = True
        if failed or any(item.process.poll() is None for item in processes):
            raise ManagedWorkerError("managed_worker_cleanup_failed")


def load_worker_launch_configuration(path: Path) -> WorkerLaunchConfiguration:
    value = parse_json_object(
        _read_bounded_file(
            Path(path),
            MAX_WORKER_LAUNCH_MANIFEST_BYTES,
            "managed worker launch manifest",
        ),
        "managed worker launch manifest",
    )
    expect_fields(
        value,
        {
            "schema_version",
            "executables",
            "hf_home",
            "ocr_model_root",
            "ffmpeg_binary",
            "ffprobe_binary",
        },
        "managed worker launch manifest",
    )
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != WORKER_LAUNCH_SCHEMA_VERSION
    ):
        raise ManagedWorkerError("worker_launch_schema_unsupported")
    executable_value = expect_object(
        value["executables"],
        "managed worker launch manifest executables",
    )
    expect_fields(
        executable_value,
        set(_EXECUTABLE_ROLES),
        "managed worker launch manifest executables",
    )
    executables = tuple(
        (role, _validated_executable(executable_value[role]))
        for role in _EXECUTABLE_ROLES
    )
    lexical_paths = tuple(path for _role, path in executables)
    if len(lexical_paths) != len(set(lexical_paths)):
        raise ManagedWorkerError("worker_python_bindings_not_isolated")
    ffmpeg_input = _validated_executable(value["ffmpeg_binary"])
    ffprobe_input = _validated_executable(value["ffprobe_binary"])
    try:
        ffmpeg_binary = ffmpeg_input.resolve(strict=True)
        ffprobe_binary = ffprobe_input.resolve(strict=True)
        media_paths_are_canonical = all(
            stat.S_ISREG(os.lstat(path).st_mode)
            for path in (ffmpeg_binary, ffprobe_binary)
        )
    except OSError:
        media_paths_are_canonical = False
    if (
        not media_paths_are_canonical
        or ffmpeg_input.name != "ffmpeg"
        or ffprobe_input.name != "ffprobe"
        or ffmpeg_binary.name != "ffmpeg"
        or ffprobe_binary.name != "ffprobe"
        or ffmpeg_binary.parent != ffprobe_binary.parent
    ):
        raise ManagedWorkerError("ffmpeg_toolchain_binding_invalid")
    return WorkerLaunchConfiguration(
        schema_version=WORKER_LAUNCH_SCHEMA_VERSION,
        executables=executables,
        hf_home=_validated_external_root(
            value["hf_home"],
            code="hf_home_invalid",
        ),
        ocr_model_root=_validated_external_root(
            value["ocr_model_root"],
            code="ocr_model_root_invalid",
        ),
        ffmpeg_binary=ffmpeg_binary,
        ffprobe_binary=ffprobe_binary,
    )


def start_regression_worker_cluster(
    *,
    data_root: Path,
    scratch_parent: Path,
    models_root: Path,
    configuration: WorkerLaunchConfiguration,
) -> ManagedRegressionWorkerCluster:
    if not isinstance(configuration, WorkerLaunchConfiguration):
        raise ManagedWorkerError("worker_launch_configuration_invalid")
    data_root = _validated_runtime_root(data_root, "data_root_invalid")
    scratch_parent = _validated_runtime_root(
        scratch_parent,
        "scratch_parent_invalid",
    )
    models_root = _validated_external_root(models_root, code="models_root_invalid")
    roots = (data_root, scratch_parent, models_root, configuration.hf_home)
    for index, root in enumerate(roots):
        for other in roots[index + 1 :]:
            if root == other or root.is_relative_to(other) or other.is_relative_to(root):
                raise ManagedWorkerError("worker_roots_overlap")
    for required in (
        data_root / "media",
        data_root / "cache",
        data_root / "tmp",
    ):
        _validated_runtime_root(required, "worker_runtime_directory_invalid")

    ports = _allocate_ports()
    tokens = {role: secrets.token_hex(32) for role in MANAGED_WORKER_ROLES}
    environments = _worker_environments(
        data_root=data_root,
        scratch_parent=scratch_parent,
        models_root=models_root,
        configuration=configuration,
        ports=ports,
        tokens=tokens,
    )
    ocr_worker_environment = _ocr_worker_environment(
        data_root=data_root,
        configuration=configuration,
    )
    module_by_role = {
        "vision_index": "videoscope.providers.vision_worker",
        "vision": "videoscope.providers.vision_worker",
        "whisper": "videoscope.providers.whisper_worker",
        "lighthouse": "videoscope.providers.lighthouse_worker",
        "qwen": "videoscope.providers.qwen_worker",
    }
    executable_role = {
        "vision_index": "vision",
        "vision": "vision",
        "whisper": "whisper",
        "lighthouse": "lighthouse",
        "qwen": "qwen",
    }
    processes: list[_WorkerProcess] = []
    try:
        for role in MANAGED_WORKER_ROLES:
            process = subprocess.Popen(
                [
                    os.fspath(configuration.executable(executable_role[role])),
                    "-m",
                    module_by_role[role],
                ],
                cwd=data_root,
                env=environments[role],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
            processes.append(_WorkerProcess(role, ports[role], process))
        managed = _capture_bindings(tuple(processes))
        common_overrides: dict[str, object] = {
            "ocr_worker_python": configuration.executable("ocr"),
            "ocr_worker_script": _PROJECT_ROOT / "scripts" / "paddle-ocr-worker.py",
            "ocr_worker_environment": ocr_worker_environment,
            "whisper_worker_endpoint": f"http://127.0.0.1:{ports['whisper']}",
            "whisper_worker_api_key": tokens["whisper"],
            "lighthouse_endpoint": f"http://127.0.0.1:{ports['lighthouse']}",
            "lighthouse_api_key": tokens["lighthouse"],
            "qwen_video_endpoint": f"http://127.0.0.1:{ports['qwen']}",
            "qwen_video_api_key": tokens["qwen"],
        }
        cluster = ManagedRegressionWorkerCluster(
            processes=tuple(processes),
            managed_workers=managed,
            ingest_overrides={
                **common_overrides,
                "vision_worker_endpoint": (
                    f"http://127.0.0.1:{ports['vision_index']}"
                ),
                "vision_worker_api_key": tokens["vision_index"],
            },
            benchmark_overrides={
                **common_overrides,
                "vision_worker_endpoint": f"http://127.0.0.1:{ports['vision']}",
                "vision_worker_api_key": tokens["vision"],
            },
            ocr_model_root=configuration.ocr_model_root,
        )
        cluster.wait_ready()
        return cluster
    except BaseException as error:
        cluster = ManagedRegressionWorkerCluster(
            processes=tuple(processes),
            managed_workers=(),
            ingest_overrides={},
            benchmark_overrides={},
            ocr_model_root=configuration.ocr_model_root,
        )
        try:
            cluster.close()
        except Exception:
            pass
        if isinstance(error, (KeyboardInterrupt, SystemExit, ManagedWorkerError)):
            raise
        raise ManagedWorkerError("managed_worker_start_failed") from error


def _worker_environments(
    *,
    data_root: Path,
    scratch_parent: Path,
    models_root: Path,
    configuration: WorkerLaunchConfiguration,
    ports: Mapping[str, int],
    tokens: Mapping[str, str],
) -> dict[str, dict[str, str]]:
    whisper_revision = model_revision(WHISPER_MODEL)
    qwen_revision = model_revision(QWEN_VIDEO_MODEL)
    if whisper_revision is None or qwen_revision is None:
        raise ManagedWorkerError("managed_worker_model_revision_missing")

    def base(role: str) -> dict[str, str]:
        temporary = data_root / "tmp" / f"worker-{role}"
        temporary.mkdir(mode=0o700, parents=True, exist_ok=False)
        return {
            "HF_DATASETS_OFFLINE": "1",
            "HF_HOME": os.fspath(configuration.hf_home),
            "HF_HUB_CACHE": os.fspath(configuration.hf_home / "hub"),
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_OFFLINE": "1",
            "HOME": os.fspath(temporary),
            "NO_PROXY": "127.0.0.1",
            "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
            "PATH": os.fspath(configuration.ffmpeg_binary.parent),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": os.fspath(_PROJECT_ROOT / "backend" / "src"),
            "TMPDIR": os.fspath(temporary),
            "TRANSFORMERS_CACHE": os.fspath(configuration.hf_home / "hub"),
            "TRANSFORMERS_OFFLINE": "1",
            "UV_OFFLINE": "1",
            "XDG_CACHE_HOME": os.fspath(configuration.hf_home),
            "no_proxy": "127.0.0.1",
        }

    def vision(role: str, root: Path) -> dict[str, str]:
        return {
            **base(role),
            "VIDEOSCOPE_SIGLIP_MODEL": SIGLIP_224_MODEL,
            "VIDEOSCOPE_VISION_DETECTOR_CHECKPOINT_SHA256": (
                RFDETR_SMALL_CHECKPOINT_SHA256
            ),
            "VIDEOSCOPE_VISION_DETECTOR_MODEL_ID": "rfdetr-small",
            "VIDEOSCOPE_VISION_WORKER_API_KEY": tokens[role],
            "VIDEOSCOPE_VISION_WORKER_HOST": "127.0.0.1",
            "VIDEOSCOPE_VISION_WORKER_INPUT_ROOT": os.fspath(root),
            "VIDEOSCOPE_VISION_WORKER_PORT": str(ports[role]),
            "VIDEOSCOPE_VISION_WORKER_RFDETR_CHECKPOINT": os.fspath(
                models_root / "rfdetr" / "rf-detr-small.pth"
            ),
        }

    return {
        "vision_index": vision("vision_index", data_root),
        "vision": vision("vision", scratch_parent),
        "whisper": {
            **base("whisper"),
            "VIDEOSCOPE_WHISPER_WORKER_API_KEY": tokens["whisper"],
            "VIDEOSCOPE_WHISPER_WORKER_HOST": "127.0.0.1",
            "VIDEOSCOPE_WHISPER_WORKER_INPUT_ROOT": os.fspath(data_root / "media"),
            "VIDEOSCOPE_WHISPER_WORKER_LOG_LEVEL": "warning",
            "VIDEOSCOPE_WHISPER_WORKER_MODEL_NAME": WHISPER_MODEL,
            "VIDEOSCOPE_WHISPER_WORKER_MODEL_REVISION": whisper_revision,
            "VIDEOSCOPE_WHISPER_WORKER_PORT": str(ports["whisper"]),
            "VIDEOSCOPE_WHISPER_WORKER_WORK_ROOT": os.fspath(
                data_root / "tmp" / "whisper-worker"
            ),
        },
        "lighthouse": {
            **base("lighthouse"),
            "VIDEOSCOPE_DATA_DIR": os.fspath(data_root),
            "VIDEOSCOPE_LIGHTHOUSE_API_KEY": tokens["lighthouse"],
            "VIDEOSCOPE_LIGHTHOUSE_CHECKPOINT": os.fspath(
                models_root / "lighthouse" / "clip_qd_detr_qvhighlight.ckpt"
            ),
            "VIDEOSCOPE_LIGHTHOUSE_CLIP_CHECKPOINT": os.fspath(
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
            "VIDEOSCOPE_DATA_DIR": os.fspath(data_root),
            "VIDEOSCOPE_QWEN_WORKER_INPUT_ROOT": os.fspath(scratch_parent),
            "VIDEOSCOPE_QWEN_VIDEO_MODEL_REVISION": qwen_revision,
        },
    }


def _ocr_worker_environment(
    *,
    data_root: Path,
    configuration: WorkerLaunchConfiguration,
) -> dict[str, str]:
    temporary = data_root / "tmp" / "worker-ocr"
    cache = temporary / "cache"
    try:
        temporary.mkdir(mode=0o700, parents=True, exist_ok=False)
        cache.mkdir(mode=0o700, exist_ok=False)
    except OSError as exc:
        raise ManagedWorkerError("worker_runtime_directory_invalid") from exc
    return {
        "HF_HOME": os.fspath(configuration.hf_home),
        "HOME": os.fspath(temporary),
        "PATH": os.fspath(configuration.ffmpeg_binary.parent),
        "TMPDIR": os.fspath(temporary),
        "XDG_CACHE_HOME": os.fspath(cache),
    }


def _capture_bindings(
    processes: tuple[_WorkerProcess, ...],
) -> tuple[ManagedProcessBinding, ...]:
    provider = create_native_process_snapshot_provider(root_pid=os.getpid())
    deadline = monotonic() + _BINDING_TIMEOUT_SECONDS
    records: dict[int, ProcessRecord] = {}
    while monotonic() < deadline:
        records = {
            item.pid: item
            for item in provider.snapshot()
            if isinstance(item, ProcessRecord)
        }
        if all(item.process.pid in records for item in processes):
            break
        if any(item.process.poll() is not None for item in processes):
            raise ManagedWorkerError("managed_worker_process_binding_unavailable")
        sleep(0.02)
    bindings: list[ManagedProcessBinding] = []
    for worker in processes:
        record = records.get(worker.process.pid)
        if record is None:
            raise ManagedWorkerError("managed_worker_process_binding_unavailable")
        bindings.append(
            ManagedProcessBinding(
                pid=record.pid,
                start_token=record.start_token,
                executable_identity=record.executable_identity,
                role=worker.role,
            )
        )
    return tuple(bindings)


def _allocate_ports() -> dict[str, int]:
    ports: dict[str, int] = {}
    used: set[int] = set()
    for role in MANAGED_WORKER_ROLES:
        for _attempt in range(64):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                    listener.bind(("127.0.0.1", 0))
                    port = int(listener.getsockname()[1])
            except OSError as exc:
                raise ManagedWorkerError("loopback_port_allocation_failed") from exc
            if port not in used:
                ports[role] = port
                used.add(port)
                break
        else:
            raise ManagedWorkerError("loopback_port_allocation_failed")
    return ports


def _validated_executable(value: object) -> Path:
    if type(value) is not str or not value or "\x00" in value:
        raise ManagedWorkerError("worker_python_binding_invalid")
    candidate = Path(value)
    if not candidate.is_absolute() or Path(os.path.abspath(candidate)) != candidate:
        raise ManagedWorkerError("worker_python_binding_invalid")
    try:
        lexical = os.lstat(candidate)
        resolved = candidate.resolve(strict=True)
        metadata = os.stat(resolved)
    except OSError as exc:
        raise ManagedWorkerError("worker_python_binding_invalid") from exc
    if (
        not (stat.S_ISREG(lexical.st_mode) or stat.S_ISLNK(lexical.st_mode))
        or not stat.S_ISREG(metadata.st_mode)
        or not os.access(candidate, os.X_OK)
    ):
        raise ManagedWorkerError("worker_python_binding_invalid")
    return candidate


def _validated_external_root(value: object, *, code: str) -> Path:
    if isinstance(value, Path):
        candidate = value
    elif type(value) is str and value and "\x00" not in value:
        candidate = Path(value)
    else:
        raise ManagedWorkerError(code)
    if not candidate.is_absolute() or Path(os.path.abspath(candidate)) != candidate:
        raise ManagedWorkerError(code)
    try:
        metadata = os.lstat(candidate)
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ManagedWorkerError(code) from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or resolved == Path(resolved.anchor)
        or resolved == _PROJECT_ROOT
        or resolved.is_relative_to(_PROJECT_ROOT)
    ):
        raise ManagedWorkerError(code)
    return resolved


def _validated_runtime_root(value: Path, code: str) -> Path:
    if not isinstance(value, Path) or not value.is_absolute() or value.is_symlink():
        raise ManagedWorkerError(code)
    try:
        metadata = os.lstat(value)
        resolved = value.resolve(strict=True)
    except OSError as exc:
        raise ManagedWorkerError(code) from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or resolved != value
        or resolved == Path(resolved.anchor)
        or resolved.is_relative_to(Path.home().resolve())
        or resolved == _PROJECT_ROOT
        or resolved.is_relative_to(_PROJECT_ROOT)
    ):
        raise ManagedWorkerError(code)
    return resolved


__all__ = [
    "MANAGED_WORKER_ROLES",
    "MAX_WORKER_LAUNCH_MANIFEST_BYTES",
    "WORKER_LAUNCH_SCHEMA_VERSION",
    "ManagedRegressionWorkerCluster",
    "ManagedWorkerError",
    "WorkerLaunchConfiguration",
    "load_worker_launch_configuration",
    "start_regression_worker_cluster",
]
