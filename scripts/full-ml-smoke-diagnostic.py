#!/usr/bin/env python3
"""Run the unchanged full ML smoke with a separate, create-once diagnostic timeline."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from contextlib import redirect_stdout
from hashlib import sha256
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import stat
import sys


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_BACKEND_SOURCE_ROOT = os.fspath(_PROJECT_ROOT / "backend" / "src")
sys.path[:] = [_BACKEND_SOURCE_ROOT, *(p for p in sys.path if p != _BACKEND_SOURCE_ROOT)]

from videoscope.benchmark.smoke_timeline import SmokeTimeline


MAX_STDOUT_BYTES = 64 * 1024 * 1024
_CODE_SHA = re.compile(r"[a-f0-9]{40}")
_PROTECTED_ENVIRONMENT_ROOTS = (
    "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
    "VIDEOSCOPE_OCR_MODEL_ROOT", "VIDEOSCOPE_FULL_ML_SMOKE_MODELS_ROOT",
)


class _DiagnosticError(Exception):
    def __init__(self, code: str) -> None:
        self.code = code


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _DiagnosticError("diagnostic_arguments_invalid")


class _BoundedBytes(io.RawIOBase):
    def __init__(self) -> None:
        self.data = bytearray()
        self.failure_code: str | None = None

    def write(self, value: bytes) -> int:
        if len(self.data) + len(value) > MAX_STDOUT_BYTES:
            self.failure_code = "stdout_limit_exceeded"
            raise _DiagnosticError(self.failure_code)
        self.data.extend(value)
        return len(value)


class _CapturedStdout(io.TextIOBase):
    def __init__(self) -> None:
        self.buffer = _BoundedBytes()

    @property
    def encoding(self) -> str:
        return "utf-8"

    def write(self, value: str) -> int:
        self.buffer.write(value.encode("utf-8"))
        return len(value)


def _load_smoke():
    specification = importlib.util.spec_from_file_location(
        "videoscope_full_ml_smoke_diagnostic_target",
        _PROJECT_ROOT / "scripts" / "full-ml-smoke.py",
    )
    if specification is None or specification.loader is None:
        raise _DiagnosticError("smoke_module_unavailable")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _reserve_sidecar(target: Path, *, root: Path, models_root: Path) -> int:
    if not target.is_absolute() or not target.name or ".." in target.parts:
        raise _DiagnosticError("diagnostic_path_invalid")
    protected = [_PROJECT_ROOT, Path.home(), root, models_root]
    protected.extend(
        Path(value) for key in _PROTECTED_ENVIRONMENT_ROOTS
        if (value := os.environ.get(key))
    )
    if any(target.is_relative_to(path.expanduser().resolve()) for path in protected):
        raise _DiagnosticError("diagnostic_path_invalid")
    # Walk directory descriptors: symlinks in any ancestor cannot redirect the
    # final O_EXCL operation after an earlier path check.
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    directory = os.open(target.anchor, directory_flags)
    try:
        for part in target.parent.parts[1:]:
            child = os.open(part, directory_flags, dir_fd=directory)
            os.close(directory)
            directory = child
        metadata = os.fstat(directory)
        if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
            raise _DiagnosticError("diagnostic_path_invalid")
        descriptor = os.open(
            target.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=directory,
        )
        try:
            os.fchmod(descriptor, 0o600)
        except BaseException:
            os.close(descriptor)
            raise
        return descriptor
    finally:
        os.close(directory)


def _write_sidecar(descriptor: int, payload: dict[str, object]) -> None:
    data = (json.dumps(
        payload, ensure_ascii=True, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ) + "\n").encode()
    remaining = memoryview(data)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise _DiagnosticError("diagnostic_publication_failed")
        remaining = remaining[written:]
    os.fsync(descriptor)


def _failure(code: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "failed",
        "error": {"kind": "diagnostic", "code": code},
    }


def _write_error(code: str) -> None:
    sys.stderr.write(
        json.dumps(_failure(code), sort_keys=True, separators=(",", ":")) + "\n"
    )


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError


def _read_payload(data: bytes) -> dict[str, object]:
    value = json.loads(
        data.decode("utf-8"), object_pairs_hook=_strict_object,
        parse_constant=_invalid_constant,
    )
    if (
        not isinstance(value, dict)
        or type(value.get("schema_version")) is not int
        or value["schema_version"] != 2
        or value.get("status") not in {"ready", "failed"}
    ):
        raise ValueError
    return value


def _code_identity(payload: dict[str, object]) -> str | None:
    source = payload if payload.get("status") == "ready" else payload.get("diagnostics")
    value = source.get("code_sha_before") if isinstance(source, dict) else None
    return value if type(value) is str and _CODE_SHA.fullmatch(value) is not None else None


def main(argv: Sequence[str] | None = None) -> int:
    parser = _Parser(description=__doc__)
    parser.add_argument("--timeline", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--models-root", required=True, type=Path)
    try:
        arguments = parser.parse_args(argv)
        descriptor = _reserve_sidecar(
            arguments.timeline, root=arguments.root,
            models_root=arguments.models_root,
        )
    except _DiagnosticError as error:
        _write_error(error.code)
        return 2
    except (OSError, ValueError, RuntimeError):
        _write_error("diagnostic_path_invalid")
        return 2

    captured = None
    timeline = None
    native_exit = 70
    error_code = None
    payload = None
    code_sha = None
    raw = b""
    try:
        try:
            captured = _CapturedStdout()
            smoke = _load_smoke()
            timeline = SmokeTimeline(smoke._DIAGNOSTIC_EVENT_IDS)
            token = smoke._DIAGNOSTIC_TIMELINE.set(timeline)
            failure_token = smoke._DIAGNOSTIC_FAILURE.set(False)
            try:
                with redirect_stdout(captured):
                    result = smoke.main([
                        "--root", os.fspath(arguments.root),
                        "--models-root", os.fspath(arguments.models_root),
                    ])
                if type(result) is not int or not 0 <= result <= 255:
                    raise _DiagnosticError("smoke_exit_invalid")
                native_exit = result
            finally:
                if smoke._DIAGNOSTIC_FAILURE.get():
                    error_code = "diagnostic_observer_failed"
                smoke._DIAGNOSTIC_FAILURE.reset(failure_token)
                smoke._DIAGNOSTIC_TIMELINE.reset(token)
        except KeyboardInterrupt:
            native_exit = 130
            error_code = "diagnostic_interrupted"
        except _DiagnosticError as error:
            error_code = error.code
        except BaseException:
            error_code = "diagnostic_execution_failed"
        if captured is not None:
            raw = bytes(captured.buffer.data)
            error_code = captured.buffer.failure_code or error_code
        if raw and captured is not None and captured.buffer.failure_code is None:
            try:
                payload = _read_payload(raw)
                code_sha = _code_identity(payload)
                if native_exit == 0 and payload["status"] != "ready":
                    error_code = error_code or "smoke_stdout_invalid"
            except (ValueError, UnicodeError, TypeError, RecursionError):
                error_code = error_code or "smoke_stdout_invalid"
        if code_sha is None:
            error_code = error_code or "code_identity_unavailable"
        sidecar: dict[str, object] = _failure(
            error_code or "diagnostic_timeline_unavailable"
        )
        if timeline is not None and code_sha is not None:
            try:
                sidecar = timeline.to_portable_dict(
                    code_sha=code_sha,
                    status="complete" if native_exit == 0 and error_code is None else "failed",
                )
            except BaseException:
                error_code = error_code or "diagnostic_timeline_unavailable"
                try:
                    sidecar = timeline.to_portable_dict(code_sha=code_sha, status="failed")
                except BaseException:
                    sidecar = _failure(error_code)
        if code_sha is not None:
            sidecar["code_sha"] = code_sha
        sidecar.update({
            "smoke_stdout_sha256": sha256(raw).hexdigest(),
            "smoke_stdout_byte_size": len(raw),
            "smoke_exit_code": native_exit,
            "promotion_eligible": False,
            "attribution": "temporal_association_not_causality",
            "measurement_use": "diagnostic_only_not_gate_evidence",
        })
        if error_code is not None:
            sidecar["status"] = "failed"
            sidecar["error"] = {"kind": "diagnostic", "code": error_code}
        try:
            _write_sidecar(descriptor, sidecar)
        except BaseException:
            error_code = error_code or "diagnostic_publication_failed"
    except BaseException:
        error_code = error_code or "diagnostic_execution_failed"
    finally:
        try:
            os.close(descriptor)
        except OSError:
            error_code = error_code or "diagnostic_publication_failed"
    if payload is not None:
        try:
            sys.stdout.write(raw.decode("utf-8"))
        except (OSError, UnicodeError):
            error_code = error_code or "diagnostic_stdout_write_failed"
    if error_code is not None:
        _write_error(error_code)
    return native_exit or (70 if error_code is not None else 0)


if __name__ == "__main__":
    raise SystemExit(main())
