from __future__ import annotations

import json
from pathlib import Path

import pytest

from videoscope.benchmark import managed_workers as subject
from videoscope.benchmark.measurements import ManagedProcessBinding


def _launch_manifest(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    executable_root = tmp_path / "executables"
    executable_root.mkdir()
    executables: dict[str, str] = {}
    for role in ("vision", "whisper", "lighthouse", "qwen", "ocr"):
        path = executable_root / role
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o700)
        executables[role] = str(path)
    for name in ("ffmpeg", "ffprobe"):
        path = executable_root / name
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o700)
    hf_home = tmp_path / "hf-home"
    ocr_models = tmp_path / "ocr-models"
    hf_home.mkdir()
    ocr_models.mkdir()
    value: dict[str, object] = {
        "schema_version": subject.WORKER_LAUNCH_SCHEMA_VERSION,
        "executables": executables,
        "hf_home": str(hf_home),
        "ocr_model_root": str(ocr_models),
        "ffmpeg_binary": str(executable_root / "ffmpeg"),
        "ffprobe_binary": str(executable_root / "ffprobe"),
    }
    manifest = tmp_path / "worker-launch.json"
    manifest.write_text(json.dumps(value), encoding="utf-8")
    return manifest, value


def test_worker_launch_manifest_is_exact_bounded_and_path_pinned(
    tmp_path: Path,
) -> None:
    manifest, value = _launch_manifest(tmp_path)

    configuration = subject.load_worker_launch_configuration(manifest)

    assert configuration.schema_version == 1
    assert tuple(role for role, _path in configuration.executables) == (
        "vision",
        "whisper",
        "lighthouse",
        "qwen",
        "ocr",
    )
    assert all(path.is_absolute() for _role, path in configuration.executables)
    assert configuration.hf_home == Path(value["hf_home"])
    assert configuration.ocr_model_root == Path(value["ocr_model_root"])
    assert configuration.ffmpeg_binary == Path(value["ffmpeg_binary"])
    assert configuration.ffprobe_binary == Path(value["ffprobe_binary"])

    unknown = dict(value)
    unknown["api_key"] = "must-not-be-accepted"
    manifest.write_text(json.dumps(unknown), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected"):
        subject.load_worker_launch_configuration(manifest)

    invalid_schema = dict(value)
    invalid_schema["schema_version"] = True
    manifest.write_text(json.dumps(invalid_schema), encoding="utf-8")
    with pytest.raises(subject.ManagedWorkerError, match="schema_unsupported"):
        subject.load_worker_launch_configuration(manifest)


def test_worker_launch_manifest_requires_one_exact_ffmpeg_tool_directory(
    tmp_path: Path,
) -> None:
    manifest, value = _launch_manifest(tmp_path)
    other = tmp_path / "other" / "ffprobe"
    other.parent.mkdir()
    other.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    other.chmod(0o700)
    value["ffprobe_binary"] = str(other)
    manifest.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(subject.ManagedWorkerError, match="toolchain_binding_invalid"):
        subject.load_worker_launch_configuration(manifest)

    symlink_case = tmp_path / "symlink-case"
    symlink_case.mkdir()
    manifest, value = _launch_manifest(symlink_case)
    canonical_bin = symlink_case / "cellar" / "bin"
    canonical_bin.mkdir(parents=True)
    for name in ("ffmpeg", "ffprobe"):
        target = canonical_bin / name
        target.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        target.chmod(0o700)
        linked = Path(value[f"{name}_binary"])
        linked.unlink()
        linked.symlink_to(target)

    configuration = subject.load_worker_launch_configuration(manifest)
    assert configuration.ffmpeg_binary == canonical_bin / "ffmpeg"
    assert configuration.ffprobe_binary == canonical_bin / "ffprobe"


def test_worker_launch_manifest_rejects_symlink_and_duplicate_interpreters(
    tmp_path: Path,
) -> None:
    manifest, value = _launch_manifest(tmp_path)
    linked = tmp_path / "linked.json"
    linked.symlink_to(manifest)
    with pytest.raises(ValueError, match="symbolic link"):
        subject.load_worker_launch_configuration(linked)

    executables = dict(value["executables"])  # type: ignore[arg-type]
    executables["ocr"] = executables["qwen"]
    value["executables"] = executables
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(
        subject.ManagedWorkerError,
        match="worker_python_bindings_not_isolated",
    ):
        subject.load_worker_launch_configuration(manifest)


def test_worker_environments_use_explicit_roots_and_separate_vision_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, _value = _launch_manifest(tmp_path)
    configuration = subject.load_worker_launch_configuration(manifest)
    data_root = tmp_path / "data"
    scratch = tmp_path / "scratch"
    models = tmp_path / "models"
    for path in (data_root, scratch, models):
        path.mkdir(mode=0o700)
        path.chmod(0o700)
    for path in (data_root / "media", data_root / "cache", data_root / "tmp"):
        path.mkdir(mode=0o700)
        path.chmod(0o700)
    monkeypatch.setenv("QWEN_VIDEO_API_KEY", "ambient-secret-must-not-leak")
    monkeypatch.setenv("PATH", "/ambient/untrusted/bin")
    roles = subject.MANAGED_WORKER_ROLES
    ports = {role: 32000 + index for index, role in enumerate(roles)}
    tokens = {role: role.replace("_", "") * 32 for role in roles}

    environments = subject._worker_environments(
        data_root=data_root,
        scratch_parent=scratch,
        models_root=models,
        configuration=configuration,
        ports=ports,
        tokens=tokens,
    )

    assert tuple(environments) == roles
    assert (
        environments["vision_index"]["VIDEOSCOPE_VISION_WORKER_INPUT_ROOT"]
        == str(data_root)
    )
    assert (
        environments["vision"]["VIDEOSCOPE_VISION_WORKER_INPUT_ROOT"]
        == str(scratch)
    )
    assert environments["qwen"]["VIDEOSCOPE_QWEN_WORKER_INPUT_ROOT"] == str(
        scratch
    )
    assert environments["whisper"]["VIDEOSCOPE_WHISPER_WORKER_INPUT_ROOT"] == str(
        data_root / "media"
    )
    assert all(
        "ambient-secret-must-not-leak" not in child.values()
        for child in environments.values()
    )
    assert {
        child["PATH"] for child in environments.values()
    } == {str(configuration.ffmpeg_binary.parent)}

    ocr_environment = subject._ocr_worker_environment(
        data_root=data_root,
        configuration=configuration,
    )
    ocr_private_root = data_root / "tmp" / "worker-ocr"
    assert ocr_environment == {
        "HF_HOME": str(configuration.hf_home),
        "HOME": str(ocr_private_root),
        "PATH": str(configuration.ffmpeg_binary.parent),
        "TMPDIR": str(ocr_private_root),
        "XDG_CACHE_HOME": str(ocr_private_root / "cache"),
    }
    assert not any("ambient" in value for value in ocr_environment.values())


def test_ingest_only_workers_are_retired_and_removed_from_measurement_bindings(
    tmp_path: Path,
) -> None:
    class Process:
        def __init__(self, pid: int) -> None:
            self.pid = pid
            self.returncode: int | None = None

        def poll(self):  # type: ignore[no-untyped-def]
            return self.returncode

        def terminate(self) -> None:
            self.returncode = 0

        def kill(self) -> None:
            self.returncode = -9

        def wait(self, timeout: float) -> int:
            assert timeout == 5.0
            assert self.returncode is not None
            return self.returncode

    processes = tuple(
        subject._WorkerProcess(role, 32000 + index, Process(4000 + index))  # type: ignore[arg-type]
        for index, role in enumerate(subject.MANAGED_WORKER_ROLES)
    )
    bindings = tuple(
        ManagedProcessBinding(
            pid=process.process.pid,
            start_token=f"start-{process.role}",
            executable_identity=f"sha256:{index:064x}",
            role=process.role,
        )
        for index, process in enumerate(processes, start=1)
    )
    cluster = subject.ManagedRegressionWorkerCluster(
        processes=processes,  # type: ignore[arg-type]
        managed_workers=bindings,
        ingest_overrides={},
        benchmark_overrides={},
        ocr_model_root=tmp_path,
    )

    cluster.retire_ingest_workers()

    assert cluster.retired_worker_roles == ("vision_index", "whisper")
    assert tuple(item.role for item in cluster.managed_workers) == (
        "vision",
        "lighthouse",
        "qwen",
    )
    state = {item.role: item.process.poll() for item in processes}
    assert state["vision_index"] == 0
    assert state["whisper"] == 0
    assert state["vision"] is None
    cluster.close()
    assert all(item.process.poll() == 0 for item in processes)


def test_cluster_cleanup_can_be_retried_after_a_transient_failure(
    tmp_path: Path,
) -> None:
    class Process:
        pid = 9001

        def __init__(self) -> None:
            self.returncode: int | None = None
            self.fail_once = True

        def poll(self):  # type: ignore[no-untyped-def]
            return self.returncode

        def terminate(self) -> None:
            self.returncode = 0
            if self.fail_once:
                self.fail_once = False
                raise OSError("synthetic transient cleanup failure")

        def kill(self) -> None:
            self.returncode = -9

        def wait(self, timeout: float) -> int:
            assert timeout == 5.0
            assert self.returncode is not None
            return self.returncode

    process = Process()
    cluster = subject.ManagedRegressionWorkerCluster(
        processes=(subject._WorkerProcess("vision", 32000, process),),  # type: ignore[arg-type]
        managed_workers=(),
        ingest_overrides={},
        benchmark_overrides={},
        ocr_model_root=tmp_path,
    )

    with pytest.raises(subject.ManagedWorkerError, match="cleanup_failed"):
        cluster.close()

    cluster.close()
