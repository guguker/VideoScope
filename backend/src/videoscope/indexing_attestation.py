from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import stat
import subprocess
import sys
from threading import Event, Lock, Thread
import tomllib
from typing import TYPE_CHECKING

from videoscope.media.ffmpeg import ATTESTED_SUBPROCESS_ENVIRONMENT_POLICY

if TYPE_CHECKING:
    from videoscope.media.ffmpeg import FFmpeg


DEFAULT_REVIEWED_UV_LOCK_SHA256 = (
    "sha256:797d9e4824bd3d71c6ff7c1624abfaa3d6b6d2d126236f23fb7037dcf845d457"
)
MAX_ATTESTED_EXECUTABLE_BYTES = 256 * 1024 * 1024
MAX_ATTESTED_VERSION_OUTPUT_BYTES = 64 * 1024

_MAX_LOCK_BYTES = 1024 * 1024
_READ_CHUNK_BYTES = 1024 * 1024
_VERSION_READ_CHUNK_BYTES = 8192
_SHA256_IDENTITY_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.!+_-]{0,127}$")
_RUNTIME_TOKEN_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._+_-]{0,63}$")
_PYTHON_VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_LOCKED_DISTRIBUTIONS = (
    "numpy",
    "opencv-python",
    "qdrant-client",
    "scenedetect",
    "videoscope-backend",
)


class IndexingToolchainAttestationError(RuntimeError):
    """A bounded, path-free failure from local indexing attestation."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"indexing toolchain attestation failed ({code})")


@dataclass(frozen=True, slots=True)
class _FileSeal:
    device: int
    inode: int
    mode: int
    links: int
    size: int
    modified_ns: int
    changed_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> _FileSeal:
        return cls(
            device=int(value.st_dev),
            inode=int(value.st_ino),
            mode=int(value.st_mode),
            links=int(value.st_nlink),
            size=int(value.st_size),
            modified_ns=int(value.st_mtime_ns),
            changed_ns=int(value.st_ctime_ns),
        )


@dataclass(frozen=True, slots=True)
class _StableFile:
    path: Path = field(repr=False)
    content_sha256: str
    content: bytes | None = field(repr=False)
    seal: _FileSeal = field(repr=False)


@dataclass(frozen=True, slots=True)
class _ExecutableAttestation:
    content_sha256: str
    version_output_sha256: str
    version_output_size: int
    path: Path = field(repr=False, compare=False)
    seal: _FileSeal = field(repr=False, compare=False)

    @property
    def canonical_payload(self) -> dict[str, object]:
        return {
            "content_sha256": self.content_sha256,
            "version_output_sha256": self.version_output_sha256,
            "version_output_size": self.version_output_size,
        }


@dataclass(frozen=True, slots=True)
class _PythonRuntimeAttestation:
    implementation: str
    version: str
    system: str
    machine: str

    @property
    def canonical_payload(self) -> dict[str, str]:
        return {
            "implementation": self.implementation,
            "machine": self.machine,
            "system": self.system,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class _LockContract:
    identity: str
    distributions: tuple[tuple[str, str], ...]
    path: Path = field(repr=False, compare=False)
    seal: _FileSeal = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class AttestedIndexingToolchain:
    """Immutable executor identity plus a fail-closed current-state hook.

    File paths and mutable file-system health are deliberately private and are
    not part of ``canonical_json``. The digest contains only reviewed content,
    exact version-output digests, exact lock-derived distribution versions,
    the pathless Python runtime/platform, and the subprocess environment policy.
    """

    _ffmpeg: _ExecutableAttestation = field(repr=False)
    _ffprobe: _ExecutableAttestation = field(repr=False)
    _lock: _LockContract = field(repr=False)
    _python_runtime: _PythonRuntimeAttestation = field(repr=False)
    _distribution_version: Callable[[str], str] = field(repr=False, compare=False)
    _python_runtime_identity: Callable[[], Mapping[str, str]] = field(
        repr=False,
        compare=False,
    )
    _version_timeout: float = field(repr=False, compare=False)
    _canonical_json: str = field(init=False, repr=False, compare=False)
    _identity: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        payload = {
            "execution_policy": {
                "subprocess_environment": (
                    ATTESTED_SUBPROCESS_ENVIRONMENT_POLICY
                ),
            },
            "executables": {
                "ffmpeg": self._ffmpeg.canonical_payload,
                "ffprobe": self._ffprobe.canonical_payload,
            },
            "python_distributions": dict(self._lock.distributions),
            "python_runtime": self._python_runtime.canonical_payload,
            "reviewed_lock_sha256": self._lock.identity,
            "schema_version": 2,
        }
        canonical_json = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        identity = "sha256:" + hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
        object.__setattr__(self, "_canonical_json", canonical_json)
        object.__setattr__(self, "_identity", identity)

    @property
    def canonical_json(self) -> str:
        return self._canonical_json

    @property
    def identity(self) -> str:
        return self._identity

    @property
    def ffmpeg_binary(self) -> str:
        return str(self._ffmpeg.path)

    @property
    def ffprobe_binary(self) -> str:
        return str(self._ffprobe.path)

    @property
    def distribution_versions(self) -> Mapping[str, str]:
        return dict(self._lock.distributions)

    @property
    def python_runtime(self) -> Mapping[str, str]:
        return dict(self._python_runtime.canonical_payload)

    def create_ffmpeg(self) -> FFmpeg:
        """Create the media adapter using only the two pinned absolute paths."""
        from videoscope.media.ffmpeg import FFmpeg

        return FFmpeg.from_attested_paths(self._ffmpeg.path, self._ffprobe.path)

    def verify_current(self) -> str:
        """Re-attest lock, distributions, executables, and exact version output.

        The method is suitable as both a pre- and post-execution hook. It
        returns the original identity on success and raises a sanitized error
        on any drift.
        """
        current_python_runtime = _attest_python_runtime(
            self._python_runtime_identity,
        )
        if current_python_runtime != self._python_runtime:
            raise IndexingToolchainAttestationError("python_runtime_drift")
        current_lock = _load_lock_contract(
            self._lock.path,
            reviewed_lock_sha256=self._lock.identity,
        )
        if (
            current_lock.path != self._lock.path
            or current_lock.seal != self._lock.seal
            or current_lock.distributions != self._lock.distributions
        ):
            raise IndexingToolchainAttestationError("lock_drift")
        _attest_distribution_versions(
            current_lock.distributions,
            distribution_version=self._distribution_version,
        )

        current_ffmpeg = _attest_executable(
            self._ffmpeg.path,
            version_timeout=self._version_timeout,
            expected=self._ffmpeg,
        )
        current_ffprobe = _attest_executable(
            self._ffprobe.path,
            version_timeout=self._version_timeout,
            expected=self._ffprobe,
        )
        for expected, current in (
            (self._ffmpeg, current_ffmpeg),
            (self._ffprobe, current_ffprobe),
        ):
            if (
                current.path != expected.path
                or current.seal != expected.seal
                or current.canonical_payload != expected.canonical_payload
            ):
                raise IndexingToolchainAttestationError("executable_drift")
        return self.identity


def _default_lock_path() -> Path:
    return Path(__file__).resolve().parents[2] / "uv.lock"


def _configuration_error() -> IndexingToolchainAttestationError:
    return IndexingToolchainAttestationError("invalid_configuration")


def _current_python_runtime_identity() -> Mapping[str, str]:
    version = sys.version_info
    return {
        "implementation": sys.implementation.name,
        "version": f"{version.major}.{version.minor}.{version.micro}",
        "system": platform.system(),
        "machine": platform.machine(),
    }


def _normalized_runtime_token(value: object, *, machine: bool = False) -> str:
    if type(value) is not str:
        raise IndexingToolchainAttestationError("python_runtime_invalid")
    normalized = value.lower()
    if machine:
        normalized = normalized.replace("-", "_")
        normalized = {
            "amd64": "x86_64",
            "arm64": "aarch64",
            "x64": "x86_64",
        }.get(normalized, normalized)
    if not _RUNTIME_TOKEN_PATTERN.fullmatch(normalized):
        raise IndexingToolchainAttestationError("python_runtime_invalid")
    return normalized


def _attest_python_runtime(
    identity: Callable[[], Mapping[str, str]],
) -> _PythonRuntimeAttestation:
    try:
        payload = identity()
        if not isinstance(payload, Mapping) or set(payload) != {
            "implementation",
            "machine",
            "system",
            "version",
        }:
            raise IndexingToolchainAttestationError("python_runtime_invalid")
        version = payload["version"]
        if type(version) is not str or not _PYTHON_VERSION_PATTERN.fullmatch(version):
            raise IndexingToolchainAttestationError("python_runtime_invalid")
        return _PythonRuntimeAttestation(
            implementation=_normalized_runtime_token(payload["implementation"]),
            version=version,
            system=_normalized_runtime_token(payload["system"]),
            machine=_normalized_runtime_token(payload["machine"], machine=True),
        )
    except IndexingToolchainAttestationError:
        raise
    except Exception:
        raise IndexingToolchainAttestationError("python_runtime_unavailable") from None


def _resolve_executable_path(value: str | os.PathLike[str]) -> Path:
    try:
        raw = os.fspath(value)
    except TypeError:
        raise _configuration_error() from None
    if type(raw) is not str or not raw or "\x00" in raw or len(raw) > 4096:
        raise _configuration_error()

    candidate = Path(raw)
    if not candidate.is_absolute():
        if len(candidate.parts) != 1 or raw in {".", ".."}:
            raise _configuration_error()
        discovered = shutil.which(raw, path=os.environ.get("PATH", ""))
        if not discovered:
            raise IndexingToolchainAttestationError("executable_not_found")
        candidate = Path(discovered)
    try:
        return candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        raise IndexingToolchainAttestationError("executable_not_found") from None


def _stable_regular_file(
    path: Path,
    *,
    max_bytes: int,
    kind: str,
    require_executable: bool,
    reject_input_symlink: bool,
    retain_content: bool,
) -> _StableFile:
    if not path.is_absolute():
        raise _configuration_error()
    try:
        input_status = os.lstat(path)
    except OSError:
        code = "executable_not_found" if kind == "executable" else "lock_unavailable"
        raise IndexingToolchainAttestationError(code) from None
    if reject_input_symlink and stat.S_ISLNK(input_status.st_mode):
        raise IndexingToolchainAttestationError(f"{kind}_not_stable")

    try:
        resolved = path.resolve(strict=True)
        initial_status = os.lstat(resolved)
    except (OSError, RuntimeError):
        code = "executable_not_found" if kind == "executable" else "lock_unavailable"
        raise IndexingToolchainAttestationError(code) from None
    if stat.S_ISLNK(initial_status.st_mode):
        raise IndexingToolchainAttestationError(f"{kind}_not_stable")
    if not stat.S_ISREG(initial_status.st_mode):
        raise IndexingToolchainAttestationError(f"{kind}_not_regular")
    if initial_status.st_nlink != 1:
        raise IndexingToolchainAttestationError(f"{kind}_not_stable")
    if initial_status.st_size < 1:
        raise IndexingToolchainAttestationError(f"{kind}_invalid")
    if initial_status.st_size > max_bytes:
        raise IndexingToolchainAttestationError(f"{kind}_too_large")
    if require_executable and not initial_status.st_mode & 0o111:
        raise IndexingToolchainAttestationError("executable_not_executable")
    if require_executable and initial_status.st_mode & (stat.S_ISUID | stat.S_ISGID):
        raise IndexingToolchainAttestationError("executable_not_stable")

    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(resolved, flags)
    except OSError:
        raise IndexingToolchainAttestationError(f"{kind}_not_stable") from None
    try:
        before = _FileSeal.from_stat(os.fstat(descriptor))
        if not stat.S_ISREG(before.mode):
            raise IndexingToolchainAttestationError(f"{kind}_not_regular")
        if before.links != 1 or before != _FileSeal.from_stat(initial_status):
            raise IndexingToolchainAttestationError(f"{kind}_not_stable")
        if before.size < 1:
            raise IndexingToolchainAttestationError(f"{kind}_invalid")
        if before.size > max_bytes:
            raise IndexingToolchainAttestationError(f"{kind}_too_large")

        content_hash = hashlib.sha256()
        chunks: list[bytes] | None = [] if retain_content else None
        remaining = before.size
        while remaining:
            chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, remaining))
            if not chunk:
                raise IndexingToolchainAttestationError(f"{kind}_not_stable")
            content_hash.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise IndexingToolchainAttestationError(f"{kind}_not_stable")
        after = _FileSeal.from_stat(os.fstat(descriptor))
    finally:
        os.close(descriptor)

    try:
        final_status = _FileSeal.from_stat(os.lstat(resolved))
    except OSError:
        raise IndexingToolchainAttestationError(f"{kind}_not_stable") from None
    if before != after or after != final_status:
        raise IndexingToolchainAttestationError(f"{kind}_not_stable")
    return _StableFile(
        path=resolved,
        content_sha256="sha256:" + content_hash.hexdigest(),
        content=b"".join(chunks) if chunks is not None else None,
        seal=after,
    )


def _resolve_executable(value: str | os.PathLike[str]) -> _StableFile:
    path = _resolve_executable_path(value)
    return _stable_regular_file(
        path,
        max_bytes=MAX_ATTESTED_EXECUTABLE_BYTES,
        kind="executable",
        require_executable=True,
        reject_input_symlink=False,
        retain_content=False,
    )


def _terminate_version_process(process: subprocess.Popen[bytes]) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except (OSError, ProcessLookupError):
        pass
    try:
        process.wait(timeout=1.0)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _run_bounded_version(path: Path, *, timeout: float) -> bytes:
    try:
        process = subprocess.Popen(
            [str(path), "-version"],
            executable=str(path),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={},
            cwd=path.anchor,
            close_fds=True,
            start_new_session=True,
            shell=False,
            bufsize=0,
        )
    except (OSError, ValueError):
        raise IndexingToolchainAttestationError("version_command_unavailable") from None
    if process.stdout is None or process.stderr is None:
        _terminate_version_process(process)
        raise IndexingToolchainAttestationError("version_command_unavailable")

    buffers = (bytearray(), bytearray())
    overflow = Event()
    read_failed = Event()
    budget_lock = Lock()
    captured = 0

    def drain(stream, destination: bytearray) -> None:
        nonlocal captured
        try:
            while True:
                chunk = stream.read(_VERSION_READ_CHUNK_BYTES)
                if not chunk:
                    return
                with budget_lock:
                    room = max(0, MAX_ATTESTED_VERSION_OUTPUT_BYTES - captured)
                    if room:
                        selected = chunk[:room]
                        destination.extend(selected)
                        captured += len(selected)
                    if len(chunk) > room:
                        overflow.set()
        except OSError:
            read_failed.set()
        finally:
            try:
                stream.close()
            except OSError:
                pass

    threads = (
        Thread(target=drain, args=(process.stdout, buffers[0]), daemon=True),
        Thread(target=drain, args=(process.stderr, buffers[1]), daemon=True),
    )
    for thread in threads:
        thread.start()
    try:
        return_code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _terminate_version_process(process)
        for thread in threads:
            thread.join(timeout=1.0)
        raise IndexingToolchainAttestationError("version_command_timeout") from None
    for thread in threads:
        thread.join(timeout=1.0)
    if any(thread.is_alive() for thread in threads) or read_failed.is_set():
        _terminate_version_process(process)
        raise IndexingToolchainAttestationError("version_command_unavailable")
    if overflow.is_set():
        raise IndexingToolchainAttestationError("version_output_too_large")
    if return_code != 0:
        raise IndexingToolchainAttestationError("version_command_failed")
    stdout = bytes(buffers[0])
    stderr = bytes(buffers[1])
    if not stdout and not stderr:
        raise IndexingToolchainAttestationError("version_output_invalid")
    return (
        len(stdout).to_bytes(8, byteorder="big")
        + stdout
        + len(stderr).to_bytes(8, byteorder="big")
        + stderr
    )


def _attest_executable(
    value: str | os.PathLike[str],
    *,
    version_timeout: float,
    expected: _ExecutableAttestation | None = None,
) -> _ExecutableAttestation:
    initial = _resolve_executable(value)
    if expected is not None and (
        initial.path != expected.path
        or initial.seal != expected.seal
        or initial.content_sha256 != expected.content_sha256
    ):
        raise IndexingToolchainAttestationError("executable_drift")
    version_output = _run_bounded_version(initial.path, timeout=version_timeout)
    final = _stable_regular_file(
        initial.path,
        max_bytes=MAX_ATTESTED_EXECUTABLE_BYTES,
        kind="executable",
        require_executable=True,
        reject_input_symlink=True,
        retain_content=False,
    )
    if initial.seal != final.seal or initial.content_sha256 != final.content_sha256:
        raise IndexingToolchainAttestationError("executable_not_stable")
    return _ExecutableAttestation(
        content_sha256=initial.content_sha256,
        version_output_sha256="sha256:" + hashlib.sha256(version_output).hexdigest(),
        version_output_size=len(version_output),
        path=initial.path,
        seal=initial.seal,
    )


def _load_lock_contract(
    value: str | os.PathLike[str],
    *,
    reviewed_lock_sha256: str,
) -> _LockContract:
    if (
        type(reviewed_lock_sha256) is not str
        or not _SHA256_IDENTITY_PATTERN.fullmatch(reviewed_lock_sha256)
    ):
        raise _configuration_error()
    try:
        raw_path = os.fspath(value)
    except TypeError:
        raise _configuration_error() from None
    if (
        type(raw_path) is not str
        or not raw_path
        or "\x00" in raw_path
        or len(raw_path) > 4096
    ):
        raise _configuration_error()
    path = Path(raw_path)
    if not path.is_absolute():
        raise _configuration_error()
    stable = _stable_regular_file(
        path,
        max_bytes=_MAX_LOCK_BYTES,
        kind="lock",
        require_executable=False,
        reject_input_symlink=True,
        retain_content=True,
    )
    identity = stable.content_sha256
    if identity != reviewed_lock_sha256:
        raise IndexingToolchainAttestationError("lock_identity_mismatch")
    if stable.content is None:
        raise IndexingToolchainAttestationError("lock_invalid")
    try:
        payload = tomllib.loads(stable.content.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError, RecursionError):
        raise IndexingToolchainAttestationError("lock_invalid") from None
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise IndexingToolchainAttestationError("lock_invalid")
    packages = payload.get("package")
    if not isinstance(packages, list):
        raise IndexingToolchainAttestationError("lock_invalid")

    distributions: list[tuple[str, str]] = []
    for expected_name in _LOCKED_DISTRIBUTIONS:
        matches = [
            item
            for item in packages
            if isinstance(item, dict) and item.get("name") == expected_name
        ]
        if len(matches) != 1:
            raise IndexingToolchainAttestationError("lock_invalid")
        version = matches[0].get("version")
        if type(version) is not str or not _VERSION_PATTERN.fullmatch(version):
            raise IndexingToolchainAttestationError("lock_invalid")
        distributions.append((expected_name, version))
    return _LockContract(
        identity=identity,
        distributions=tuple(distributions),
        path=stable.path,
        seal=stable.seal,
    )


def _attest_distribution_versions(
    expected: tuple[tuple[str, str], ...],
    *,
    distribution_version: Callable[[str], str],
) -> None:
    for name, expected_version in expected:
        try:
            actual_version = distribution_version(name)
        except Exception:
            raise IndexingToolchainAttestationError("distribution_unavailable") from None
        if type(actual_version) is not str:
            raise IndexingToolchainAttestationError("distribution_unavailable")
        if actual_version != expected_version:
            raise IndexingToolchainAttestationError("distribution_version_mismatch")


def attest_indexing_toolchain(
    *,
    ffmpeg_binary: str | os.PathLike[str] = "ffmpeg",
    ffprobe_binary: str | os.PathLike[str] = "ffprobe",
    lock_path: str | os.PathLike[str] | None = None,
    reviewed_lock_sha256: str = DEFAULT_REVIEWED_UV_LOCK_SHA256,
    distribution_version: Callable[[str], str] = importlib_metadata.version,
    python_runtime_identity: Callable[
        [], Mapping[str, str]
    ] = _current_python_runtime_identity,
    version_timeout: float = 5.0,
) -> AttestedIndexingToolchain:
    """Pin the local indexing executor to reviewed, pathless identities."""
    if not callable(distribution_version):
        raise _configuration_error()
    if not callable(python_runtime_identity):
        raise _configuration_error()
    if (
        isinstance(version_timeout, bool)
        or not isinstance(version_timeout, (int, float))
        or not 0.01 <= float(version_timeout) <= 30.0
    ):
        raise _configuration_error()
    selected_lock_path = lock_path if lock_path is not None else _default_lock_path()
    lock = _load_lock_contract(
        selected_lock_path,
        reviewed_lock_sha256=reviewed_lock_sha256,
    )
    _attest_distribution_versions(
        lock.distributions,
        distribution_version=distribution_version,
    )
    python_runtime = _attest_python_runtime(python_runtime_identity)
    timeout = float(version_timeout)
    ffmpeg = _attest_executable(
        ffmpeg_binary,
        version_timeout=timeout,
    )
    ffprobe = _attest_executable(
        ffprobe_binary,
        version_timeout=timeout,
    )
    return AttestedIndexingToolchain(
        _ffmpeg=ffmpeg,
        _ffprobe=ffprobe,
        _lock=lock,
        _python_runtime=python_runtime,
        _distribution_version=distribution_version,
        _python_runtime_identity=python_runtime_identity,
        _version_timeout=timeout,
    )
