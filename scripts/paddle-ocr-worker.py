#!/usr/bin/env python3
from __future__ import annotations

from contextlib import redirect_stdout
from hashlib import sha256
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import stat
import sys
import tempfile
from typing import Any, BinaryIO
import zlib


OCR_WORKER_PROTOCOL = "videoscope.paddleocr-jsonl.v2"
OCR_WORKER_DEPENDENCY_IDENTITY = (
    "paddleocr-deps-v1:sha256:"
    "f1d9979471ad0a051576e9cdd0e5de25ca20f1e67ed48d284f0f904353c356a0"
)
OCR_WORKER_RUNTIME_IDENTITY = (
    "videoscope-paddleocr-worker-v1|python==3.12.13|"
    "platform==aarch64-apple-darwin-macos14plus|lock-sha256:"
    "f1d9979471ad0a051576e9cdd0e5de25ca20f1e67ed48d284f0f904353c356a0"
)
OCR_MODEL_ARTIFACT_IDENTITY = (
    "paddleocr-models-v1:sha256:"
    "623ad4bd1fa38f07332e5b1a6fd0a6f691b98c8d38153b9d4f32e1b9df6c9b80"
)

MAX_OCR_REQUEST_BYTES = 16 * 1024
MAX_OCR_RESPONSE_BYTES = 256 * 1024
MAX_OCR_CONTROL_FILE_BYTES = 1024 * 1024
MAX_OCR_IMAGE_BYTES = 64 * 1024 * 1024
MAX_OCR_IMAGE_DIMENSION = 8192
MAX_OCR_IMAGE_PIXELS = 32 * 1024 * 1024
MAX_OCR_DECODED_IMAGE_BYTES = 128 * 1024 * 1024
MAX_OCR_PATH_CHARACTERS = 4096
MAX_OCR_ITEMS = 2048
MAX_OCR_TEXT_CHARACTERS = 512
MAX_OCR_TOTAL_TEXT_CHARACTERS = 64 * 1024

_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUEST_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_FRAME_NAME_RE = re.compile(r"^([0-9a-f]{32})\.frame$")
_LOCK_PACKAGE_RE = re.compile(
    r"^([A-Za-z0-9_.-]+)==([^ ;\\]+)(?:\s*;[^\\]+)?\s*\\?$"
)
_ALLOWED_ERROR_CODES = {
    "invalid_request",
    "model_failure",
    "output_limit",
    "protocol_limit",
    "startup_failed",
}


class ProtocolError(ValueError):
    pass


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate JSON key")
        output[key] = value
    return output


def _strict_json_bytes(payload: bytes) -> object:
    try:
        return json.loads(
            payload,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError, RecursionError) as error:
        raise ProtocolError("invalid JSON") from error


def _canonical_json_bytes(payload: object) -> bytes:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as error:
        raise ProtocolError("invalid response") from error


def _result_payload(result: object) -> dict[str, Any]:
    payload: object = getattr(result, "json", result)
    if callable(payload):
        payload = payload()
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        return {}
    nested = payload.get("res")
    return nested if isinstance(nested, dict) else payload


def _fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _open_parent_without_symlinks(path: Path) -> tuple[Path, int]:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if not nofollow or not directory:
        raise ProtocolError("reviewed file access unavailable")
    absolute = Path(os.path.abspath(path))
    if not absolute.name:
        raise ProtocolError("reviewed file missing")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | directory | nofollow
    try:
        descriptor = os.open(absolute.anchor, flags)
        try:
            for component in absolute.parts[1:-1]:
                child = os.open(component, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
        except BaseException:
            os.close(descriptor)
            raise
    except OSError as error:
        raise ProtocolError("reviewed file unavailable") from error
    return absolute, descriptor


def _read_stable_file_with_fingerprint(
    path: Path,
    *,
    maximum_bytes: int,
    expected_size: int | None = None,
) -> tuple[bytes, tuple[int, ...]]:
    absolute, parent_descriptor = _open_parent_without_symlinks(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    after: os.stat_result | None = None
    try:
        lexical = os.stat(
            absolute.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(lexical.st_mode)
            or lexical.st_nlink != 1
            or not 1 <= lexical.st_size <= maximum_bytes
            or (expected_size is not None and lexical.st_size != expected_size)
        ):
            raise ProtocolError("invalid reviewed file")
        descriptor = os.open(absolute.name, flags, dir_fd=parent_descriptor)
        try:
            before = os.fstat(descriptor)
            if _fingerprint(before) != _fingerprint(lexical):
                raise ProtocolError("reviewed file changed")
            digest_input: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(descriptor, min(64 * 1024, maximum_bytes + 1))
                if not chunk:
                    break
                total += len(chunk)
                if total > maximum_bytes:
                    raise ProtocolError("reviewed file too large")
                digest_input.append(chunk)
            after = os.fstat(descriptor)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
                descriptor = -1
        current = os.stat(
            absolute.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except ProtocolError:
        raise
    except OSError as error:
        raise ProtocolError("reviewed file unavailable") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)
    assert after is not None
    if (
        _fingerprint(before) != _fingerprint(after)
        or _fingerprint(after) != _fingerprint(current)
    ):
        raise ProtocolError("reviewed file changed")
    return b"".join(digest_input), _fingerprint(after)


def _read_stable_file(
    path: Path,
    *,
    maximum_bytes: int,
    expected_size: int | None = None,
) -> bytes:
    raw, _fingerprint_value = _read_stable_file_with_fingerprint(
        path,
        maximum_bytes=maximum_bytes,
        expected_size=expected_size,
    )
    return raw


def _parse_dependency_lock(raw: bytes) -> dict[str, str]:
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ProtocolError("invalid dependency lock") from error
    packages: dict[str, str] = {}
    current_name: str | None = None
    current_has_hash = False
    for line in lines:
        match = _LOCK_PACKAGE_RE.fullmatch(line)
        if match is not None:
            if current_name is not None and not current_has_hash:
                raise ProtocolError("dependency lock entry is not hashed")
            name, version = match.groups()
            canonical_name = re.sub(r"[-_.]+", "-", name).lower()
            if canonical_name in packages:
                raise ProtocolError("duplicate dependency lock entry")
            packages[canonical_name] = version
            current_name = canonical_name
            current_has_hash = False
            continue
        if "--hash=" in line:
            if (
                current_name is None
                or re.search(r"--hash=sha256:[0-9a-f]{64}(?:\s*\\)?$", line)
                is None
            ):
                raise ProtocolError("invalid dependency lock hash")
            current_has_hash = True
    if current_name is not None and not current_has_hash:
        raise ProtocolError("dependency lock entry is not hashed")
    if not 1 <= len(packages) <= 256:
        raise ProtocolError("dependency lock is empty")
    return packages


def _verify_installed_dependencies(packages: dict[str, str]) -> None:
    for name, expected_version in packages.items():
        try:
            installed_version = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError as error:
            raise ProtocolError("dependency is missing") from error
        if installed_version != expected_version:
            raise ProtocolError("dependency version mismatch")
    observed: dict[str, str] = {}
    try:
        distributions = importlib_metadata.distributions()
        for distribution in distributions:
            name = distribution.metadata.get("Name")
            version = distribution.version
            if type(name) is not str or type(version) is not str:
                raise ProtocolError("invalid installed dependency set")
            canonical_name = re.sub(r"[-_.]+", "-", name).lower()
            if canonical_name in observed:
                raise ProtocolError("duplicate installed dependency")
            observed[canonical_name] = version
    except ProtocolError:
        raise
    except Exception as error:
        raise ProtocolError("installed dependency set is unavailable") from error
    if observed != packages:
        raise ProtocolError("unreviewed installed dependency set")


def _parse_model_manifest(raw: bytes) -> dict[str, object]:
    payload = _strict_json_bytes(raw)
    if not isinstance(payload, dict) or set(payload) != {
        "engine",
        "models",
        "profile",
        "schema_version",
    }:
        raise ProtocolError("invalid model manifest")
    profile = payload.get("profile")
    models = payload.get("models")
    if (
        payload.get("schema_version") != 1
        or payload.get("engine") != "transformers"
        or type(profile) is not str
        or not 1 <= len(profile) <= 128
        or not isinstance(models, list)
        or not 1 <= len(models) <= 8
    ):
        raise ProtocolError("invalid model manifest")
    roles: set[str] = set()
    total_files = 0
    total_size = 0
    for model in models:
        if not isinstance(model, dict) or set(model) != {
            "artifacts",
            "directory",
            "role",
        }:
            raise ProtocolError("invalid model manifest")
        directory = model.get("directory")
        role = model.get("role")
        artifacts = model.get("artifacts")
        if (
            type(directory) is not str
            or not directory
            or Path(directory).name != directory
            or directory in {".", ".."}
            or type(role) is not str
            or not role
            or role in roles
            or not isinstance(artifacts, list)
            or not 1 <= len(artifacts) <= 32
        ):
            raise ProtocolError("invalid model manifest")
        roles.add(role)
        names: set[str] = set()
        for artifact in artifacts:
            if not isinstance(artifact, dict) or set(artifact) != {
                "name",
                "sha256",
                "size",
            }:
                raise ProtocolError("invalid model manifest")
            name = artifact.get("name")
            digest = artifact.get("sha256")
            size = artifact.get("size")
            if (
                type(name) is not str
                or not name
                or Path(name).name != name
                or name in {".", ".."}
                or name in names
                or type(digest) is not str
                or _HEX_64_RE.fullmatch(digest) is None
                or type(size) is not int
                or not 1 <= size <= 2 * 1024 * 1024 * 1024
            ):
                raise ProtocolError("invalid model manifest")
            names.add(name)
            total_files += 1
            total_size += size
    if total_files > 64 or total_size > 4 * 1024 * 1024 * 1024:
        raise ProtocolError("invalid model manifest")
    return payload


def _verify_models(
    root: Path,
    manifest: dict[str, object],
) -> dict[str, Path]:
    directories: dict[str, Path] = {}
    models = manifest["models"]
    assert isinstance(models, list)
    for model in models:
        assert isinstance(model, dict)
        directory = root / str(model["directory"])
        directories[str(model["role"])] = directory
        artifacts = model["artifacts"]
        assert isinstance(artifacts, list)
        for artifact in artifacts:
            assert isinstance(artifact, dict)
            raw = _read_stable_file(
                directory / str(artifact["name"]),
                maximum_bytes=int(artifact["size"]),
                expected_size=int(artifact["size"]),
            )
            if sha256(raw).hexdigest() != artifact["sha256"]:
                raise ProtocolError("model artifact mismatch")
    return directories


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("private file write failed")
        view = view[written:]


def _copy_verified_model_artifact(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_size: int,
) -> tuple[int, ...]:
    absolute, parent_descriptor = _open_parent_without_symlinks(source)
    source_descriptor = -1
    destination_descriptor = -1
    destination_created = False
    try:
        lexical = os.stat(
            absolute.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(lexical.st_mode)
            or lexical.st_nlink != 1
            or lexical.st_size != expected_size
        ):
            raise ProtocolError("invalid model artifact")
        source_descriptor = os.open(
            absolute.name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        before = os.fstat(source_descriptor)
        if _fingerprint(before) != _fingerprint(lexical):
            raise ProtocolError("model artifact changed")
        destination_descriptor = os.open(
            destination,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o400,
        )
        destination_created = True
        os.fchmod(destination_descriptor, 0o400)
        digest = sha256()
        total = 0
        while True:
            chunk = os.read(source_descriptor, min(1024 * 1024, expected_size + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > expected_size:
                raise ProtocolError("model artifact changed")
            digest.update(chunk)
            _write_all(destination_descriptor, chunk)
        os.fsync(destination_descriptor)
        after = os.fstat(source_descriptor)
        copied = os.fstat(destination_descriptor)
        current = os.stat(
            absolute.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            total != expected_size
            or digest.hexdigest() != expected_sha256
            or _fingerprint(before) != _fingerprint(after)
            or _fingerprint(after) != _fingerprint(current)
            or not stat.S_ISREG(copied.st_mode)
            or copied.st_nlink != 1
            or copied.st_size != expected_size
        ):
            raise ProtocolError("model artifact mismatch")
        return _fingerprint(copied)
    except ProtocolError:
        if destination_created:
            try:
                destination.unlink()
            except OSError:
                pass
        raise
    except OSError as error:
        if destination_created:
            try:
                destination.unlink()
            except OSError:
                pass
        raise ProtocolError("model artifact unavailable") from error
    finally:
        if destination_descriptor >= 0:
            os.close(destination_descriptor)
        if source_descriptor >= 0:
            os.close(source_descriptor)
        os.close(parent_descriptor)


class _PrivateModelBundle:
    __slots__ = ("artifacts", "directories", "fingerprints", "root")

    def __init__(
        self,
        *,
        root: Path,
        directories: dict[str, Path],
        artifacts: tuple[Path, ...],
        fingerprints: dict[Path, tuple[int, ...]],
    ) -> None:
        self.root = root
        self.directories = directories
        self.artifacts = artifacts
        self.fingerprints = fingerprints

    def assert_current(self) -> None:
        for path, fingerprint in self.fingerprints.items():
            try:
                metadata = path.stat(follow_symlinks=False)
            except OSError as error:
                raise ProtocolError("private model artifact unavailable") from error
            if (
                _fingerprint(metadata) != fingerprint
                or not stat.S_ISREG(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != 0o400
            ):
                raise ProtocolError("private model artifact changed")

    def cleanup(self) -> None:
        for artifact in reversed(self.artifacts):
            try:
                artifact.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass
        for directory in reversed(tuple(self.directories.values())):
            try:
                directory.rmdir()
            except OSError:
                pass
        try:
            self.root.rmdir()
        except OSError:
            pass


def _create_private_model_bundle(
    *,
    model_root: Path,
    model_manifest_path: Path,
) -> _PrivateModelBundle:
    manifest_raw = _read_stable_file(
        model_manifest_path,
        maximum_bytes=MAX_OCR_CONTROL_FILE_BYTES,
    )
    manifest = _parse_model_manifest(manifest_raw)
    identity = "paddleocr-models-v1:sha256:" + sha256(
        _canonical_json_bytes(manifest)
    ).hexdigest()
    if identity != OCR_MODEL_ARTIFACT_IDENTITY:
        raise ProtocolError("unreviewed model manifest")
    root = Path(tempfile.mkdtemp(prefix="videoscope-ocr-models-")).resolve(
        strict=True
    )
    os.chmod(root, 0o700)
    directories: dict[str, Path] = {}
    artifacts: list[Path] = []
    fingerprints: dict[Path, tuple[int, ...]] = {}
    bundle = _PrivateModelBundle(
        root=root,
        directories=directories,
        artifacts=(),
        fingerprints=fingerprints,
    )
    try:
        models = manifest["models"]
        assert isinstance(models, list)
        for model in models:
            assert isinstance(model, dict)
            role = str(model["role"])
            source_directory = model_root / str(model["directory"])
            private_directory = root / str(model["directory"])
            private_directory.mkdir(mode=0o700)
            os.chmod(private_directory, 0o700)
            directories[role] = private_directory
            model_artifacts = model["artifacts"]
            assert isinstance(model_artifacts, list)
            for artifact in model_artifacts:
                assert isinstance(artifact, dict)
                destination = private_directory / str(artifact["name"])
                fingerprint = _copy_verified_model_artifact(
                    source_directory / str(artifact["name"]),
                    destination,
                    expected_sha256=str(artifact["sha256"]),
                    expected_size=int(artifact["size"]),
                )
                artifacts.append(destination)
                fingerprints[destination] = fingerprint
        if set(directories) != {"detection", "recognition"}:
            raise ProtocolError("incomplete OCR model profile")
        bundle.artifacts = tuple(artifacts)
        bundle.assert_current()
        return bundle
    except BaseException:
        bundle.artifacts = tuple(artifacts)
        bundle.cleanup()
        raise


def _runtime_identity(lock_sha256: str) -> str:
    if sys.version_info[:3] != (3, 12, 13):
        raise ProtocolError("unreviewed Python runtime")
    if platform.python_implementation() != "CPython":
        raise ProtocolError("unreviewed CPython runtime")
    machine = platform.machine().lower()
    if machine == "arm64":
        machine = "aarch64"
    if sys.platform != "darwin" or machine != "aarch64":
        raise ProtocolError("unreviewed platform runtime")
    try:
        macos_major = int(platform.mac_ver()[0].split(".", 1)[0])
    except (TypeError, ValueError) as error:
        raise ProtocolError("unreviewed macOS runtime") from error
    if macos_major < 14:
        raise ProtocolError("unreviewed macOS runtime")
    return (
        "videoscope-paddleocr-worker-v1|python==3.12.13|"
        "platform==aarch64-apple-darwin-macos14plus|"
        f"lock-sha256:{lock_sha256}"
    )


def _startup_attestation(
    *,
    dependency_lock_path: Path,
    model_manifest_path: Path,
    model_root: Path,
    script_path: Path,
) -> tuple[dict[str, str], dict[str, Path]]:
    dependency_lock = _read_stable_file(
        dependency_lock_path,
        maximum_bytes=MAX_OCR_CONTROL_FILE_BYTES,
    )
    dependency_lock_sha = sha256(dependency_lock).hexdigest()
    dependency_identity = "paddleocr-deps-v1:sha256:" + dependency_lock_sha
    if dependency_identity != OCR_WORKER_DEPENDENCY_IDENTITY:
        raise ProtocolError("unreviewed dependency lock")
    packages = _parse_dependency_lock(dependency_lock)
    _verify_installed_dependencies(packages)

    model_manifest_raw = _read_stable_file(
        model_manifest_path,
        maximum_bytes=MAX_OCR_CONTROL_FILE_BYTES,
    )
    model_manifest = _parse_model_manifest(model_manifest_raw)
    model_identity = "paddleocr-models-v1:sha256:" + sha256(
        _canonical_json_bytes(model_manifest)
    ).hexdigest()
    if model_identity != OCR_MODEL_ARTIFACT_IDENTITY:
        raise ProtocolError("unreviewed model manifest")
    model_directories = _verify_models(model_root, model_manifest)
    if set(model_directories) != {"detection", "recognition"}:
        raise ProtocolError("incomplete OCR model profile")

    script = _read_stable_file(
        script_path,
        maximum_bytes=MAX_OCR_CONTROL_FILE_BYTES,
    )
    runtime_identity = _runtime_identity(dependency_lock_sha)
    if runtime_identity != OCR_WORKER_RUNTIME_IDENTITY:
        raise ProtocolError("unreviewed OCR runtime")
    return (
        {
            "dependency_identity": dependency_identity,
            "model_identity": model_identity,
            "protocol": OCR_WORKER_PROTOCOL,
            "runtime_identity": runtime_identity,
            "script_sha256": sha256(script).hexdigest(),
        },
        model_directories,
    )


def _jpeg_dimensions(raw: bytes) -> tuple[int, int, int]:
    offset = 2
    start_of_frame = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    while offset < len(raw):
        if raw[offset] != 0xFF:
            raise ProtocolError("invalid frame JPEG header")
        while offset < len(raw) and raw[offset] == 0xFF:
            offset += 1
        if offset >= len(raw):
            break
        marker = raw[offset]
        offset += 1
        if marker == 0xD9:
            break
        if marker == 0x00 or marker == 0xDA:
            break
        if marker == 0x01 or 0xD0 <= marker <= 0xD8:
            continue
        if offset + 2 > len(raw):
            break
        segment_length = int.from_bytes(raw[offset : offset + 2], "big")
        if segment_length < 2 or offset + segment_length > len(raw):
            raise ProtocolError("invalid frame JPEG header")
        if marker in start_of_frame:
            if segment_length < 8:
                raise ProtocolError("invalid frame JPEG header")
            precision = raw[offset + 2]
            height = int.from_bytes(raw[offset + 3 : offset + 5], "big")
            width = int.from_bytes(raw[offset + 5 : offset + 7], "big")
            components = raw[offset + 7]
            if (
                precision not in {8, 12}
                or not 1 <= components <= 4
                or segment_length != 8 + (3 * components)
            ):
                raise ProtocolError("invalid frame JPEG header")
            return width, height, components
        offset += segment_length
    raise ProtocolError("missing frame JPEG dimensions")


def _bounded_image_dimensions(raw: bytes) -> tuple[int, int]:
    if not 1 <= len(raw) <= MAX_OCR_IMAGE_BYTES:
        raise ProtocolError("invalid frame compressed size")
    width: int
    height: int
    decoded_channels: int
    bytes_per_channel = 1
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        if len(raw) < 33 or raw[8:16] != b"\x00\x00\x00\rIHDR":
            raise ProtocolError("invalid frame PNG header")
        header = raw[16:29]
        expected_crc = int.from_bytes(raw[29:33], "big")
        if zlib.crc32(b"IHDR" + header) & 0xFFFFFFFF != expected_crc:
            raise ProtocolError("invalid frame PNG header")
        width = int.from_bytes(header[0:4], "big")
        height = int.from_bytes(header[4:8], "big")
        bit_depth = header[8]
        color_type = header[9]
        if (
            header[10] != 0
            or header[11] != 0
            or header[12] not in {0, 1}
            or (color_type, bit_depth)
            not in {
                (0, 1),
                (0, 2),
                (0, 4),
                (0, 8),
                (0, 16),
                (2, 8),
                (2, 16),
                (3, 1),
                (3, 2),
                (3, 4),
                (3, 8),
                (4, 8),
                (4, 16),
                (6, 8),
                (6, 16),
            }
        ):
            raise ProtocolError("unsupported frame PNG header")
        decoded_channels = {0: 1, 2: 3, 3: 4, 4: 2, 6: 4}[color_type]
        bytes_per_channel = 2 if bit_depth == 16 else 1
    elif raw.startswith(b"\xff\xd8"):
        width, height, decoded_channels = _jpeg_dimensions(raw)
    else:
        raise ProtocolError("unsupported frame format")
    pixels = width * height
    decoded_bytes = pixels * max(3, decoded_channels * bytes_per_channel)
    if (
        not 1 <= width <= MAX_OCR_IMAGE_DIMENSION
        or not 1 <= height <= MAX_OCR_IMAGE_DIMENSION
        or not 1 <= pixels <= MAX_OCR_IMAGE_PIXELS
        or decoded_bytes > MAX_OCR_DECODED_IMAGE_BYTES
    ):
        raise ProtocolError("frame decode limit exceeded")
    return width, height


def _unlink_verified_frame(path: Path, fingerprint: tuple[int, ...]) -> None:
    absolute, parent_descriptor = _open_parent_without_symlinks(path)
    try:
        current = os.stat(
            absolute.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if _fingerprint(current) != fingerprint:
            raise ProtocolError("private frame changed")
        os.unlink(absolute.name, dir_fd=parent_descriptor)
    except ProtocolError:
        raise
    except OSError as error:
        raise ProtocolError("private frame cleanup failed") from error
    finally:
        os.close(parent_descriptor)


def _read_bound_frame(
    frame_value: object,
    *,
    frame_root: Path,
    request_id: str,
) -> tuple[bytes, int, int]:
    if not isinstance(frame_value, dict) or set(frame_value) != {
        "compressed_bytes",
        "height",
        "name",
        "sha256",
        "width",
    }:
        raise ProtocolError("invalid frame contract")
    compressed_bytes = frame_value.get("compressed_bytes")
    height = frame_value.get("height")
    name = frame_value.get("name")
    expected_sha256 = frame_value.get("sha256")
    width = frame_value.get("width")
    if (
        type(compressed_bytes) is not int
        or not 1 <= compressed_bytes <= MAX_OCR_IMAGE_BYTES
        or type(width) is not int
        or not 1 <= width <= MAX_OCR_IMAGE_DIMENSION
        or type(height) is not int
        or not 1 <= height <= MAX_OCR_IMAGE_DIMENSION
        or type(name) is not str
        or _FRAME_NAME_RE.fullmatch(name) is None
        or name != f"{request_id}.frame"
        or type(expected_sha256) is not str
        or _HEX_64_RE.fullmatch(expected_sha256) is None
    ):
        raise ProtocolError("invalid frame contract")
    path = frame_root / name
    try:
        raw, fingerprint = _read_stable_file_with_fingerprint(
            path,
            maximum_bytes=MAX_OCR_IMAGE_BYTES,
            expected_size=compressed_bytes,
        )
    except ProtocolError as error:
        raise ProtocolError("private frame is unavailable or changed") from error
    actual_width, actual_height = _bounded_image_dimensions(raw)
    if (
        sha256(raw).hexdigest() != expected_sha256
        or actual_width != width
        or actual_height != height
    ):
        raise ProtocolError("private frame binding mismatch")
    _unlink_verified_frame(path, fingerprint)
    return raw, width, height


def _decode_image_bytes(raw: bytes, *, width: int, height: int) -> object:
    try:
        import cv2
        import numpy as np

        encoded = np.frombuffer(raw, dtype=np.uint8)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except Exception as error:
        raise ProtocolError("frame decode failed") from error
    if (
        not isinstance(decoded, np.ndarray)
        or decoded.dtype != np.uint8
        or decoded.ndim != 3
        or decoded.shape != (height, width, 3)
        or decoded.nbytes != width * height * 3
        or decoded.nbytes > MAX_OCR_DECODED_IMAGE_BYTES
        or not decoded.flags.c_contiguous
    ):
        raise ProtocolError("invalid decoded frame")
    return decoded


def _handle_request(
    request: object,
    model: object,
    attestation: dict[str, str],
    *,
    frame_root: Path,
) -> dict[str, object]:
    if not isinstance(request, dict) or set(request) != {
        "attestation",
        "frame",
        "request_id",
        "type",
    }:
        raise ProtocolError("invalid request envelope")
    request_id = request.get("request_id")
    if (
        request.get("type") != "read"
        or type(request_id) is not str
        or _REQUEST_ID_RE.fullmatch(request_id) is None
        or request.get("attestation") != attestation
    ):
        raise ProtocolError("invalid request binding")
    raw, width, height = _read_bound_frame(
        request.get("frame"),
        frame_root=frame_root,
        request_id=request_id,
    )
    image = _decode_image_bytes(raw, width=width, height=height)
    predict = getattr(model, "predict", None)
    if not callable(predict):
        raise ProtocolError("model unavailable")
    try:
        with redirect_stdout(sys.stderr):
            results = predict(image)
    except Exception as error:
        raise ProtocolError("model failure") from error
    if not isinstance(results, (list, tuple)):
        raise ProtocolError("invalid model output")
    items: list[list[object]] = []
    total_characters = 0
    for result in results:
        try:
            payload = _result_payload(result)
        except (json.JSONDecodeError, TypeError, ValueError, RecursionError) as error:
            raise ProtocolError("invalid model output") from error
        texts = payload.get("rec_texts") or []
        scores = payload.get("rec_scores") or []
        if not isinstance(texts, (list, tuple)) or not isinstance(scores, (list, tuple)):
            raise ProtocolError("invalid model output")
        for text, score in zip(texts, scores, strict=False):
            if type(text) is not str or isinstance(score, bool) or not isinstance(
                score, (int, float)
            ):
                raise ProtocolError("invalid model output")
            normalized = text.strip()
            confidence = float(score)
            if not normalized:
                continue
            if (
                len(normalized) > MAX_OCR_TEXT_CHARACTERS
                or not math.isfinite(confidence)
                or not 0 <= confidence <= 1
            ):
                raise ProtocolError("invalid model output")
            total_characters += len(normalized)
            if (
                len(items) >= MAX_OCR_ITEMS
                or total_characters > MAX_OCR_TOTAL_TEXT_CHARACTERS
            ):
                raise ProtocolError("OCR output limit exceeded")
            items.append([normalized, confidence])
    return {
        "attestation": attestation,
        "items": items,
        "ok": True,
        "request_id": request_id,
        "type": "result",
    }


def _error_response(
    code: str,
    *,
    request_id: str | None = None,
    attestation: dict[str, str] | None = None,
) -> dict[str, object]:
    normalized_code = code if code in _ALLOWED_ERROR_CODES else "model_failure"
    response: dict[str, object] = {
        "error_code": normalized_code,
        "ok": False,
        "type": "error",
    }
    if request_id is not None and _REQUEST_ID_RE.fullmatch(request_id):
        response["request_id"] = request_id
    if attestation is not None:
        response["attestation"] = attestation
    return response


def _encode_response(response: dict[str, object]) -> bytes:
    encoded = _canonical_json_bytes(response) + b"\n"
    if len(encoded) > MAX_OCR_RESPONSE_BYTES:
        raise ProtocolError("response limit exceeded")
    return encoded


def _emit(output: BinaryIO, response: dict[str, object]) -> None:
    encoded = _encode_response(response)
    written = output.write(encoded)
    if written is not None and written != len(encoded):
        raise OSError("short protocol write")
    output.flush()


def _environment_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value or len(value) > MAX_OCR_PATH_CHARACTERS:
        raise ProtocolError("missing worker configuration")
    return Path(value)


def main() -> int:
    protocol_input = sys.stdin.buffer
    protocol_output = sys.stdout.buffer
    attestation: dict[str, str] | None = None
    private_models: _PrivateModelBundle | None = None
    try:
        if sys.argv[1:] not in ([], ["--attest-only"]):
            raise ProtocolError("invalid worker arguments")
        dependency_lock_path = _environment_path(
            "VIDEOSCOPE_OCR_DEPENDENCY_LOCK"
        )
        model_manifest_path = _environment_path("VIDEOSCOPE_OCR_MODEL_MANIFEST")
        model_root = _environment_path("VIDEOSCOPE_OCR_MODEL_ROOT")
        script_path = Path(__file__)
        attestation, source_model_directories = _startup_attestation(
            dependency_lock_path=dependency_lock_path,
            model_manifest_path=model_manifest_path,
            model_root=model_root,
            script_path=script_path,
        )
        if sys.argv[1:] == ["--attest-only"]:
            _emit(
                protocol_output,
                {"attestation": attestation, "ok": True, "type": "hello"},
            )
            return 0
        if set(source_model_directories) != {"detection", "recognition"}:
            raise ProtocolError("incomplete OCR model profile")
        frame_root = _environment_path("VIDEOSCOPE_OCR_FRAME_ROOT")
        private_models = _create_private_model_bundle(
            model_root=model_root,
            model_manifest_path=model_manifest_path,
        )
        with redirect_stdout(sys.stderr):
            from paddleocr import PaddleOCR

            model = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                engine="transformers",
                text_detection_model_dir=str(
                    private_models.directories["detection"]
                ),
                text_recognition_model_dir=str(
                    private_models.directories["recognition"]
                ),
            )
        private_models.assert_current()
        _emit(
            protocol_output,
            {"attestation": attestation, "ok": True, "type": "hello"},
        )
    except Exception:
        if private_models is not None:
            private_models.cleanup()
        try:
            _emit(protocol_output, _error_response("startup_failed"))
        except Exception:
            pass
        return 2

    try:
        while True:
            raw = protocol_input.readline(MAX_OCR_REQUEST_BYTES + 1)
            if not raw:
                return 0
            if len(raw) > MAX_OCR_REQUEST_BYTES or not raw.endswith(b"\n"):
                try:
                    _emit(
                        protocol_output,
                        _error_response("protocol_limit", attestation=attestation),
                    )
                except Exception:
                    pass
                return 3
            request_id: str | None = None
            try:
                request = _strict_json_bytes(raw[:-1])
                if isinstance(request, dict) and type(request.get("request_id")) is str:
                    request_id = request["request_id"]
                response = _handle_request(
                    request,
                    model,
                    attestation,
                    frame_root=frame_root,
                )
                _emit(protocol_output, response)
            except ProtocolError as error:
                error_code = (
                    "output_limit"
                    if "limit" in str(error).lower()
                    else "invalid_request"
                )
                try:
                    _emit(
                        protocol_output,
                        _error_response(
                            error_code,
                            request_id=request_id,
                            attestation=attestation,
                        ),
                    )
                except Exception:
                    pass
                return 3
            except Exception:
                try:
                    _emit(
                        protocol_output,
                        _error_response(
                            "model_failure",
                            request_id=request_id,
                            attestation=attestation,
                        ),
                    )
                except Exception:
                    pass
                return 4
    finally:
        assert private_models is not None
        private_models.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
