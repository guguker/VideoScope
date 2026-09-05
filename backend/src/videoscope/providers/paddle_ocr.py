from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import secrets
import selectors
import stat
import subprocess
import tempfile
import threading
from time import monotonic
from typing import Any, BinaryIO
import zlib

from videoscope.providers.base import ProviderState, ProviderStatus


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
_WORKER_ENVIRONMENT_PASSTHROUGH = frozenset(
    {
        "HOME",
        "HF_HOME",
        "LANG",
        "LC_ALL",
        "PATH",
        "SSL_CERT_FILE",
        "TMPDIR",
        "XDG_CACHE_HOME",
    }
)
_DEPENDENCY_IDENTITY_RE = re.compile(r"^paddleocr-deps-v1:sha256:([0-9a-f]{64})$")
_MODEL_IDENTITY_RE = re.compile(r"^paddleocr-models-v1:sha256:[0-9a-f]{64}$")
_RUNTIME_IDENTITY_RE = re.compile(
    r"^videoscope-paddleocr-worker-v1\|python==3\.12\.13\|"
    r"platform==aarch64-apple-darwin-macos14plus\|"
    r"lock-sha256:([0-9a-f]{64})$"
)


def _project_path(*parts: str) -> Path:
    source = Path(__file__).resolve()
    candidates = [Path.cwd().joinpath(*parts)]
    if len(source.parents) > 4:
        candidates.append(source.parents[4].joinpath(*parts))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[-1]


DEFAULT_OCR_DEPENDENCY_LOCK = _project_path("workers", "ocr", "requirements.lock")
DEFAULT_OCR_MODEL_MANIFEST = _project_path(
    "workers", "ocr", "model-artifacts.lock.json"
)
DEFAULT_OCR_MODEL_ROOT = Path.home() / ".paddlex" / "official_models"


def _reject_non_finite_json(value: str) -> None:
    raise ValueError(f"non-finite worker JSON: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate worker JSON key")
        output[key] = value
    return output


def _strict_json_bytes(payload: bytes) -> object:
    try:
        return json.loads(
            payload,
            parse_constant=_reject_non_finite_json,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError, RecursionError) as error:
        raise ValueError("invalid worker JSON") from error


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
        raise ValueError("invalid canonical JSON") from error


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


def _bounded_image_dimensions(raw: bytes) -> tuple[int, int]:
    """Return reviewed dimensions without invoking a compressed-image decoder."""
    if not 1 <= len(raw) <= MAX_OCR_IMAGE_BYTES:
        raise ValueError("reviewed OCR frame has invalid compressed size")
    width: int
    height: int
    decoded_channels: int
    bytes_per_channel = 1
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        if len(raw) < 33 or raw[8:16] != b"\x00\x00\x00\rIHDR":
            raise ValueError("reviewed OCR frame has invalid PNG header")
        header = raw[16:29]
        expected_crc = int.from_bytes(raw[29:33], "big")
        if zlib.crc32(b"IHDR" + header) & 0xFFFFFFFF != expected_crc:
            raise ValueError("reviewed OCR frame has invalid PNG header")
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
            raise ValueError("reviewed OCR frame has unsupported PNG header")
        decoded_channels = {0: 1, 2: 3, 3: 4, 4: 2, 6: 4}[color_type]
        bytes_per_channel = 2 if bit_depth == 16 else 1
    elif raw.startswith(b"\xff\xd8"):
        width, height, decoded_channels = _jpeg_dimensions(raw)
    else:
        raise ValueError("reviewed OCR frame format is unsupported")
    pixels = width * height
    decoded_bytes = pixels * max(3, decoded_channels * bytes_per_channel)
    if (
        not 1 <= width <= MAX_OCR_IMAGE_DIMENSION
        or not 1 <= height <= MAX_OCR_IMAGE_DIMENSION
        or not 1 <= pixels <= MAX_OCR_IMAGE_PIXELS
        or decoded_bytes > MAX_OCR_DECODED_IMAGE_BYTES
    ):
        raise ValueError("reviewed OCR frame exceeds decode limits")
    return width, height


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
            raise ValueError("reviewed OCR frame has invalid JPEG header")
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
            raise ValueError("reviewed OCR frame has invalid JPEG header")
        if marker in start_of_frame:
            if segment_length < 8:
                raise ValueError("reviewed OCR frame has invalid JPEG header")
            precision = raw[offset + 2]
            height = int.from_bytes(raw[offset + 3 : offset + 5], "big")
            width = int.from_bytes(raw[offset + 5 : offset + 7], "big")
            components = raw[offset + 7]
            if (
                precision not in {8, 12}
                or not 1 <= components <= 4
                or segment_length != 8 + (3 * components)
            ):
                raise ValueError("reviewed OCR frame has invalid JPEG header")
            return width, height, components
        offset += segment_length
    raise ValueError("reviewed OCR frame is missing JPEG dimensions")


def _decode_image_bytes(raw: bytes, *, width: int, height: int) -> object:
    try:
        import cv2
        import numpy as np

        encoded = np.frombuffer(raw, dtype=np.uint8)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except Exception as error:
        raise ValueError("reviewed OCR frame decode failed") from error
    if (
        not isinstance(decoded, np.ndarray)
        or decoded.dtype != np.uint8
        or decoded.ndim != 3
        or decoded.shape != (height, width, 3)
        or decoded.nbytes != width * height * 3
        or decoded.nbytes > MAX_OCR_DECODED_IMAGE_BYTES
        or not decoded.flags.c_contiguous
    ):
        raise ValueError("reviewed OCR frame decode is invalid")
    return decoded


def _open_parent_without_symlinks(path: Path) -> tuple[Path, int]:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if not nofollow or not directory:
        raise ValueError("reviewed OCR file access is unavailable")
    absolute = Path(os.path.abspath(path))
    if not absolute.name:
        raise ValueError("reviewed OCR file is missing")
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
        raise ValueError("reviewed OCR file is unavailable") from error
    return absolute, descriptor


def _read_stable_regular_file(
    path: Path,
    *,
    maximum_bytes: int,
    expected_size: int | None = None,
) -> tuple[bytes, tuple[int, ...]]:
    absolute, parent_descriptor = _open_parent_without_symlinks(path)
    file_descriptor = -1
    after: os.stat_result | None = None
    try:
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
                raise ValueError("reviewed OCR file is invalid")
            file_descriptor = os.open(
                absolute.name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_descriptor,
            )
            before = os.fstat(file_descriptor)
            if _fingerprint(before) != _fingerprint(lexical):
                raise ValueError("reviewed OCR file changed")
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(file_descriptor, min(64 * 1024, maximum_bytes + 1))
                if not chunk:
                    break
                total += len(chunk)
                if total > maximum_bytes:
                    raise ValueError("reviewed OCR file is too large")
                chunks.append(chunk)
            after = os.fstat(file_descriptor)
            current = os.stat(
                absolute.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                _fingerprint(before) != _fingerprint(after)
                or _fingerprint(after) != _fingerprint(current)
            ):
                raise ValueError("reviewed OCR file changed")
        except ValueError:
            raise
        except OSError as error:
            raise ValueError("reviewed OCR file is unavailable") from error
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        os.close(parent_descriptor)
    assert after is not None
    return b"".join(chunks), _fingerprint(after)


def _manifest_identity(payload: object) -> str:
    return "paddleocr-models-v1:sha256:" + sha256(
        _canonical_json_bytes(payload)
    ).hexdigest()


def _parse_model_manifest(raw: bytes) -> dict[str, object]:
    payload = _strict_json_bytes(raw)
    if not isinstance(payload, dict) or set(payload) != {
        "engine",
        "models",
        "profile",
        "schema_version",
    }:
        raise ValueError("reviewed OCR model manifest is invalid")
    profile = payload.get("profile")
    if (
        payload.get("schema_version") != 1
        or payload.get("engine") != "transformers"
        or type(profile) is not str
        or not 1 <= len(profile) <= 128
    ):
        raise ValueError("reviewed OCR model manifest is invalid")
    models = payload.get("models")
    if not isinstance(models, list) or not 1 <= len(models) <= 8:
        raise ValueError("reviewed OCR model manifest is invalid")
    roles: set[str] = set()
    total_artifacts = 0
    total_bytes = 0
    for model in models:
        if not isinstance(model, dict) or set(model) != {
            "artifacts",
            "directory",
            "role",
        }:
            raise ValueError("reviewed OCR model manifest is invalid")
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
            raise ValueError("reviewed OCR model manifest is invalid")
        roles.add(role)
        names: set[str] = set()
        for artifact in artifacts:
            if not isinstance(artifact, dict) or set(artifact) != {
                "name",
                "sha256",
                "size",
            }:
                raise ValueError("reviewed OCR model manifest is invalid")
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
                raise ValueError("reviewed OCR model manifest is invalid")
            names.add(name)
            total_artifacts += 1
            total_bytes += size
    if total_artifacts > 64 or total_bytes > 4 * 1024 * 1024 * 1024:
        raise ValueError("reviewed OCR model manifest is invalid")
    return payload


def _verify_model_artifacts(
    root: Path,
    manifest: dict[str, object],
) -> tuple[dict[str, Path], dict[Path, tuple[int, ...]]]:
    model_directories: dict[str, Path] = {}
    fingerprints: dict[Path, tuple[int, ...]] = {}
    models = manifest["models"]
    assert isinstance(models, list)
    for model in models:
        assert isinstance(model, dict)
        directory = root / str(model["directory"])
        role = str(model["role"])
        model_directories[role] = directory
        artifacts = model["artifacts"]
        assert isinstance(artifacts, list)
        for artifact in artifacts:
            assert isinstance(artifact, dict)
            path = directory / str(artifact["name"])
            raw, fingerprint = _read_stable_regular_file(
                path,
                maximum_bytes=int(artifact["size"]),
                expected_size=int(artifact["size"]),
            )
            if sha256(raw).hexdigest() != artifact["sha256"]:
                raise ValueError("reviewed OCR model artifact is invalid")
            fingerprints[path] = fingerprint
    return model_directories, fingerprints


def _assert_fingerprints_current(expected: dict[Path, tuple[int, ...]]) -> None:
    for path, fingerprint in expected.items():
        try:
            metadata = path.stat(follow_symlinks=False)
        except OSError as error:
            raise ValueError("reviewed OCR model artifact is unavailable") from error
        if _fingerprint(metadata) != fingerprint or not stat.S_ISREG(metadata.st_mode):
            raise ValueError("reviewed OCR model artifact changed")


@dataclass(frozen=True, slots=True)
class OCRWorkerAttestation:
    dependency_identity: str
    model_identity: str
    protocol: str
    runtime_identity: str
    script_sha256: str

    def __post_init__(self) -> None:
        dependency = _DEPENDENCY_IDENTITY_RE.fullmatch(self.dependency_identity)
        runtime = _RUNTIME_IDENTITY_RE.fullmatch(self.runtime_identity)
        if (
            dependency is None
            or runtime is None
            or dependency.group(1) != runtime.group(1)
            or _MODEL_IDENTITY_RE.fullmatch(self.model_identity) is None
            or self.protocol != OCR_WORKER_PROTOCOL
            or _HEX_64_RE.fullmatch(self.script_sha256) is None
        ):
            raise ValueError("reviewed OCR worker attestation is invalid")

    def as_dict(self) -> dict[str, str]:
        return {
            "dependency_identity": self.dependency_identity,
            "model_identity": self.model_identity,
            "protocol": self.protocol,
            "runtime_identity": self.runtime_identity,
            "script_sha256": self.script_sha256,
        }


@dataclass(frozen=True, slots=True)
class _PreparedOCRContract:
    attestation: OCRWorkerAttestation
    dependency_lock: bytes
    model_manifest: bytes
    model_directories: dict[str, Path]
    model_fingerprints: dict[Path, tuple[int, ...]]
    worker_script: bytes


@dataclass(frozen=True, slots=True)
class _PreparedOCRFrame:
    compressed_bytes: bytes
    height: int
    sha256: str
    width: int

    def request_payload(self, request_id: str) -> dict[str, object]:
        return {
            "compressed_bytes": len(self.compressed_bytes),
            "height": self.height,
            "name": f"{request_id}.frame",
            "sha256": self.sha256,
            "width": self.width,
        }


class PaddleOCRReader:
    id = "paddleocr"

    def __init__(
        self,
        *,
        minimum_confidence: float = 0.55,
        worker_python: Path | None = None,
        worker_script: Path | None = None,
        worker_dependency_lock: Path | None = None,
        worker_model_manifest: Path | None = None,
        worker_model_root: Path | None = None,
        expected_dependency_identity: str = OCR_WORKER_DEPENDENCY_IDENTITY,
        expected_runtime_identity: str = OCR_WORKER_RUNTIME_IDENTITY,
        expected_model_identity: str = OCR_MODEL_ARTIFACT_IDENTITY,
        expected_script_sha256: str | None = None,
        worker_environment: Mapping[str, str] | None = None,
        worker_startup_timeout_seconds: float = 180.0,
        worker_response_timeout_seconds: float = 60.0,
    ) -> None:
        if (
            not math.isfinite(minimum_confidence)
            or not 0 <= minimum_confidence <= 1
            or not math.isfinite(worker_startup_timeout_seconds)
            or not 0 < worker_startup_timeout_seconds <= 600
            or not math.isfinite(worker_response_timeout_seconds)
            or not 0 < worker_response_timeout_seconds <= 600
        ):
            raise ValueError("invalid PaddleOCR reader limits")
        self.minimum_confidence = minimum_confidence
        self.worker_python = Path(worker_python) if worker_python else None
        self.worker_script = Path(worker_script) if worker_script else None
        self.worker_dependency_lock = Path(
            worker_dependency_lock or DEFAULT_OCR_DEPENDENCY_LOCK
        )
        self.worker_model_manifest = Path(
            worker_model_manifest or DEFAULT_OCR_MODEL_MANIFEST
        )
        configured_model_root: Path | str = (
            worker_model_root
            if worker_model_root is not None
            else os.environ.get("VIDEOSCOPE_OCR_MODEL_ROOT")
            or DEFAULT_OCR_MODEL_ROOT
        )
        self.worker_model_root = Path(configured_model_root)
        self.expected_dependency_identity = expected_dependency_identity
        self.expected_runtime_identity = expected_runtime_identity
        self.expected_model_identity = expected_model_identity
        if expected_script_sha256 is not None and _HEX_64_RE.fullmatch(
            expected_script_sha256
        ) is None:
            raise ValueError("invalid reviewed OCR script identity")
        self._expected_script_sha256 = expected_script_sha256
        if worker_environment is None:
            self._explicit_worker_environment: dict[str, str] | None = None
        else:
            if not isinstance(worker_environment, Mapping):
                raise ValueError("invalid PaddleOCR worker environment")
            explicit_environment: dict[str, str] = {}
            for name, value in worker_environment.items():
                if (
                    type(name) is not str
                    or name not in _WORKER_ENVIRONMENT_PASSTHROUGH
                    or type(value) is not str
                    or not value
                    or len(value) > 16_384
                    or "\x00" in value
                ):
                    raise ValueError("invalid PaddleOCR worker environment")
                explicit_environment[name] = value
            self._explicit_worker_environment = explicit_environment
        self.worker_startup_timeout_seconds = worker_startup_timeout_seconds
        self.worker_response_timeout_seconds = worker_response_timeout_seconds
        self._model = None
        self._prepared_contract: _PreparedOCRContract | None = None
        self._worker_process: subprocess.Popen[bytes] | None = None
        self._worker_retirement_pending = False
        self._worker_bundle_directory: Path | None = None
        self._worker_frame_directory: Path | None = None
        self._worker_lock = threading.Lock()

    @property
    def _uses_worker(self) -> bool:
        return self.worker_python is not None or self.worker_script is not None

    @property
    def worker_attestation(self) -> dict[str, str]:
        return self._ensure_contract_current().attestation.as_dict()

    def _prepare_contract(self) -> _PreparedOCRContract:
        if self.worker_script is None:
            raise ValueError("reviewed OCR worker script is missing")
        if self._expected_script_sha256 is None:
            raise ValueError("reviewed OCR script identity is missing")
        worker_script, _ = _read_stable_regular_file(
            self.worker_script,
            maximum_bytes=MAX_OCR_CONTROL_FILE_BYTES,
        )
        script_digest = sha256(worker_script).hexdigest()
        if script_digest != self._expected_script_sha256:
            raise ValueError("reviewed OCR worker script changed")

        dependency_lock, _ = _read_stable_regular_file(
            self.worker_dependency_lock,
            maximum_bytes=MAX_OCR_CONTROL_FILE_BYTES,
        )
        dependency_identity = (
            "paddleocr-deps-v1:sha256:" + sha256(dependency_lock).hexdigest()
        )
        if dependency_identity != self.expected_dependency_identity:
            raise ValueError("reviewed OCR dependency lock changed")

        model_manifest, _ = _read_stable_regular_file(
            self.worker_model_manifest,
            maximum_bytes=MAX_OCR_CONTROL_FILE_BYTES,
        )
        manifest_payload = _parse_model_manifest(model_manifest)
        if _manifest_identity(manifest_payload) != self.expected_model_identity:
            raise ValueError("reviewed OCR model manifest changed")
        model_directories, model_fingerprints = _verify_model_artifacts(
            self.worker_model_root,
            manifest_payload,
        )
        attestation = OCRWorkerAttestation(
            dependency_identity=dependency_identity,
            model_identity=self.expected_model_identity,
            protocol=OCR_WORKER_PROTOCOL,
            runtime_identity=self.expected_runtime_identity,
            script_sha256=script_digest,
        )
        return _PreparedOCRContract(
            attestation=attestation,
            dependency_lock=dependency_lock,
            model_manifest=model_manifest,
            model_directories=model_directories,
            model_fingerprints=model_fingerprints,
            worker_script=worker_script,
        )

    def _ensure_contract_current(self) -> _PreparedOCRContract:
        prepared = self._prepared_contract
        if prepared is None:
            prepared = self._prepare_contract()
            self._prepared_contract = prepared
            return prepared
        assert self.worker_script is not None
        script, _ = _read_stable_regular_file(
            self.worker_script,
            maximum_bytes=MAX_OCR_CONTROL_FILE_BYTES,
        )
        dependency_lock, _ = _read_stable_regular_file(
            self.worker_dependency_lock,
            maximum_bytes=MAX_OCR_CONTROL_FILE_BYTES,
        )
        model_manifest, _ = _read_stable_regular_file(
            self.worker_model_manifest,
            maximum_bytes=MAX_OCR_CONTROL_FILE_BYTES,
        )
        if (
            script != prepared.worker_script
            or dependency_lock != prepared.dependency_lock
            or model_manifest != prepared.model_manifest
        ):
            raise ValueError("reviewed OCR worker script or contract changed")
        _assert_fingerprints_current(prepared.model_fingerprints)
        return prepared

    def status(self) -> ProviderStatus:
        if self._uses_worker:
            if self.worker_python is None or self.worker_script is None:
                ready = False
            else:
                try:
                    ready = self.worker_python.is_file()
                    if ready:
                        self._ensure_contract_current()
                except (OSError, ValueError):
                    ready = False
            if not ready:
                return ProviderStatus(
                    self.id,
                    "PaddleOCR",
                    ProviderState.UNAVAILABLE,
                    "Изолированный OCR worker не аттестован; выполните make install-ocr",
                    optional=True,
                )
            return ProviderStatus(
                self.id,
                "PaddleOCR",
                ProviderState.READY,
                "OCR contract проверен; runtime аттестуется при запуске worker",
                optional=True,
            )
        try:
            import paddleocr  # noqa: F401
        except ImportError:
            return ProviderStatus(
                self.id,
                "PaddleOCR",
                ProviderState.UNAVAILABLE,
                "PaddleOCR не установлен",
                optional=True,
            )
        return ProviderStatus(
            self.id,
            "PaddleOCR",
            ProviderState.READY,
            "PP-OCR с модельным движком Transformers",
            optional=True,
        )

    def _load(self):  # type: ignore[no-untyped-def]
        if self._model is None:
            from paddleocr import PaddleOCR

            self._model = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                engine="transformers",
            )
        return self._model

    @staticmethod
    def _write_private_file(directory: Path, name: str, payload: bytes) -> Path:
        path = directory / name
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o400,
        )
        try:
            os.fchmod(descriptor, 0o400)
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("private OCR file write failed")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        copied, _ = _read_stable_regular_file(path, maximum_bytes=len(payload))
        if copied != payload:
            raise ValueError("private OCR file verification failed")
        return path

    def _create_private_bundle(
        self, prepared: _PreparedOCRContract
    ) -> tuple[Path, Path, Path]:
        private_parent: Path | None = None
        if self._explicit_worker_environment is not None:
            raw_parent = self._explicit_worker_environment.get("TMPDIR")
            try:
                candidate = Path(raw_parent) if raw_parent is not None else None
                if (
                    candidate is None
                    or not candidate.is_absolute()
                    or Path(os.path.abspath(candidate)) != candidate
                ):
                    raise OSError
                metadata = os.lstat(candidate)
                resolved = candidate.resolve(strict=True)
                if (
                    not stat.S_ISDIR(metadata.st_mode)
                    or stat.S_ISLNK(metadata.st_mode)
                    or resolved != candidate
                    or metadata.st_uid != os.geteuid()
                    or stat.S_IMODE(metadata.st_mode) != 0o700
                ):
                    raise OSError
            except (OSError, TypeError, ValueError) as error:
                raise ValueError("invalid PaddleOCR worker scratch root") from error
            private_parent = candidate
        directory = Path(
            tempfile.mkdtemp(
                prefix="videoscope-ocr-",
                dir=os.fspath(private_parent) if private_parent is not None else None,
            )
        ).resolve(strict=True)
        if private_parent is not None and directory.parent != private_parent:
            self._remove_private_bundle(directory)
            raise ValueError("invalid PaddleOCR worker scratch root")
        os.chmod(directory, 0o700)
        try:
            script = self._write_private_file(
                directory, "paddle-ocr-worker.py", prepared.worker_script
            )
            dependency_lock = self._write_private_file(
                directory, "requirements.lock", prepared.dependency_lock
            )
            model_manifest = self._write_private_file(
                directory, "model-artifacts.lock.json", prepared.model_manifest
            )
            frame_directory = directory / "frames"
            frame_directory.mkdir(mode=0o700)
            os.chmod(frame_directory, 0o700)
        except BaseException:
            self._remove_private_bundle(directory)
            raise
        self._worker_bundle_directory = directory
        self._worker_frame_directory = frame_directory
        return script, dependency_lock, model_manifest

    @staticmethod
    def _remove_private_bundle(directory: Path) -> None:
        try:
            (directory / "frames").rmdir()
        except OSError:
            pass
        for name in (
            "paddle-ocr-worker.py",
            "requirements.lock",
            "model-artifacts.lock.json",
        ):
            try:
                (directory / name).unlink()
            except FileNotFoundError:
                pass
        try:
            directory.rmdir()
        except OSError:
            # Never let cleanup of a private, already-detached bundle mask the
            # protocol or attestation failure that caused the worker discard.
            pass

    def _cleanup_private_bundle(self) -> None:
        directory = self._worker_bundle_directory
        self._worker_bundle_directory = None
        self._worker_frame_directory = None
        if directory is not None:
            self._remove_private_bundle(directory)

    @staticmethod
    def _prepare_frame(image: Path) -> _PreparedOCRFrame:
        path_text = str(image)
        if not 1 <= len(path_text) <= MAX_OCR_PATH_CHARACTERS:
            raise RuntimeError("PaddleOCR frame path is invalid")
        try:
            raw, _fingerprint_value = _read_stable_regular_file(
                image,
                maximum_bytes=MAX_OCR_IMAGE_BYTES,
            )
            width, height = _bounded_image_dimensions(raw)
        except ValueError as error:
            raise RuntimeError("PaddleOCR frame is invalid") from error
        return _PreparedOCRFrame(
            compressed_bytes=raw,
            height=height,
            sha256=sha256(raw).hexdigest(),
            width=width,
        )

    def _write_private_frame(
        self,
        request_id: str,
        frame: _PreparedOCRFrame,
    ) -> Path:
        directory = self._worker_frame_directory
        if directory is None:
            raise RuntimeError("PaddleOCR private frame directory is unavailable")
        return self._write_private_file(
            directory,
            f"{request_id}.frame",
            frame.compressed_bytes,
        )

    def _worker_environment(
        self,
        dependency_lock: Path,
        model_manifest: Path,
    ) -> dict[str, str]:
        environment: dict[str, str] = {
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "UV_OFFLINE": "1",
            "VIDEOSCOPE_OCR_DEPENDENCY_LOCK": str(dependency_lock),
            "VIDEOSCOPE_OCR_MODEL_MANIFEST": str(model_manifest),
            "VIDEOSCOPE_OCR_MODEL_ROOT": str(self.worker_model_root),
        }
        if self._worker_frame_directory is None:
            raise RuntimeError("PaddleOCR private frame directory is unavailable")
        environment["VIDEOSCOPE_OCR_FRAME_ROOT"] = str(
            self._worker_frame_directory
        )
        inherited = self._explicit_worker_environment
        if inherited is None:
            inherited = {
                name: value
                for name in _WORKER_ENVIRONMENT_PASSTHROUGH
                if (value := os.environ.get(name))
            }
        environment.update(inherited)
        hf_home = environment.get("HF_HOME")
        if hf_home:
            hub = os.fspath(Path(hf_home) / "hub")
            environment["HF_HUB_CACHE"] = hub
            environment["TRANSFORMERS_CACHE"] = hub
        return environment

    @staticmethod
    def _readline_bounded(
        stream: BinaryIO,
        *,
        maximum_bytes: int,
        timeout_seconds: float,
    ) -> bytes:
        try:
            file_descriptor = stream.fileno()
        except (AttributeError, OSError):
            file_descriptor = -1
        if file_descriptor >= 0:
            deadline = monotonic() + timeout_seconds
            received = bytearray()
            with selectors.DefaultSelector() as selector:
                selector.register(file_descriptor, selectors.EVENT_READ)
                while len(received) <= maximum_bytes:
                    remaining = deadline - monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        raise TimeoutError("OCR worker response timed out")
                    chunk = os.read(
                        file_descriptor,
                        min(64 * 1024, maximum_bytes + 1 - len(received)),
                    )
                    if not chunk:
                        break
                    received.extend(chunk)
                    newline = received.find(b"\n")
                    if newline >= 0:
                        if newline != len(received) - 1 or newline > maximum_bytes:
                            raise ValueError("invalid bounded OCR worker response")
                        return bytes(received[:newline])
            raise ValueError("invalid bounded OCR worker response")
        line = stream.readline(maximum_bytes + 1)
        if isinstance(line, str):
            line = line.encode("utf-8")
        if not line or len(line) > maximum_bytes or not line.endswith(b"\n"):
            raise ValueError("invalid bounded OCR worker response")
        return line[:-1]

    @staticmethod
    def _decode_worker_message(raw: bytes) -> dict[str, object]:
        payload = _strict_json_bytes(raw)
        if not isinstance(payload, dict):
            raise ValueError("invalid OCR worker message")
        return payload

    def _discard_worker(
        self,
        process: subprocess.Popen[bytes] | None = None,
    ) -> None:
        active = process or self._worker_process
        if active is not None:
            # Keep ownership until exit is confirmed. A failed kill must not
            # leave an untracked child or allow reuse of its closed protocol.
            self._worker_process = active
            self._worker_retirement_pending = True
            for pipe_name in ("stdin", "stdout"):
                pipe = getattr(active, pipe_name, None)
                close = getattr(pipe, "close", None)
                if callable(close):
                    try:
                        close()
                    except OSError:
                        pass
            try:
                if active.poll() is None:
                    try:
                        active.terminate()
                        active.wait(timeout=2)
                    except (OSError, subprocess.SubprocessError):
                        if active.poll() is None:
                            active.kill()
                            active.wait(timeout=2)
                if active.poll() is None:
                    raise RuntimeError("PaddleOCR worker retirement failed")
            except (OSError, subprocess.SubprocessError):
                raise RuntimeError("PaddleOCR worker retirement failed") from None
        self._worker_process = None
        self._worker_retirement_pending = False
        self._cleanup_private_bundle()

    def _load_worker(self) -> subprocess.Popen[bytes]:
        if self._worker_retirement_pending:
            self._discard_worker()
        try:
            prepared = self._ensure_contract_current()
        except ValueError as error:
            self._discard_worker()
            raise RuntimeError("PaddleOCR worker script or attestation changed") from error
        if self._worker_process is not None and self._worker_process.poll() is None:
            return self._worker_process
        if self.worker_python is None or self.worker_script is None:
            raise RuntimeError("PaddleOCR worker is not configured")
        if not self.worker_python.is_file():
            raise RuntimeError("PaddleOCR worker runtime is unavailable")
        self._discard_worker()
        private_script, dependency_lock, model_manifest = self._create_private_bundle(
            prepared
        )
        try:
            process = subprocess.Popen(
                [str(self.worker_python), "-I", "-u", str(private_script)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
                text=False,
                close_fds=True,
                env=self._worker_environment(dependency_lock, model_manifest),
            )
        except OSError as error:
            self._cleanup_private_bundle()
            raise RuntimeError("PaddleOCR worker could not start") from error
        if process.stdin is None or process.stdout is None:
            self._discard_worker(process)
            raise RuntimeError("PaddleOCR worker pipes are unavailable")
        try:
            hello = self._decode_worker_message(
                self._readline_bounded(
                    process.stdout,
                    maximum_bytes=MAX_OCR_RESPONSE_BYTES,
                    timeout_seconds=self.worker_startup_timeout_seconds,
                )
            )
            if (
                set(hello) != {"attestation", "ok", "type"}
                or hello.get("type") != "hello"
                or hello.get("ok") is not True
                or hello.get("attestation") != prepared.attestation.as_dict()
            ):
                raise ValueError("OCR worker startup attestation mismatch")
        except (TimeoutError, ValueError) as error:
            self._discard_worker(process)
            raise RuntimeError("PaddleOCR worker attestation failed") from error
        self._worker_process = process
        return process

    @staticmethod
    def _write_worker_message(stream: BinaryIO, payload: dict[str, object]) -> None:
        encoded = _canonical_json_bytes(payload) + b"\n"
        if len(encoded) > MAX_OCR_REQUEST_BYTES:
            raise ValueError("OCR worker request is too large")
        written = stream.write(encoded)
        if written is not None and written != len(encoded):
            raise OSError("short OCR worker write")
        stream.flush()

    def close(self) -> None:
        with self._worker_lock:
            self._discard_worker()

    def release_ingestion_resources(self) -> None:
        """Retire the stage's child; the next indexing stage may load afresh."""
        self.close()

    def _validated_worker_items(self, items: object) -> list[tuple[str, float]]:
        if not isinstance(items, list) or len(items) > MAX_OCR_ITEMS:
            raise ValueError("invalid OCR worker items")
        output: list[tuple[str, float]] = []
        total_characters = 0
        for item in items:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError("invalid OCR worker item")
            text, score = item
            if (
                type(text) is not str
                or not 1 <= len(text) <= MAX_OCR_TEXT_CHARACTERS
                or isinstance(score, bool)
                or not isinstance(score, (int, float))
            ):
                raise ValueError("invalid OCR worker item")
            confidence = float(score)
            normalized = text.strip()
            total_characters += len(text)
            if (
                total_characters > MAX_OCR_TOTAL_TEXT_CHARACTERS
                or not normalized
                or not math.isfinite(confidence)
                or not 0 <= confidence <= 1
            ):
                raise ValueError("invalid OCR worker item")
            if confidence >= self.minimum_confidence:
                output.append((normalized, confidence))
        return output

    def _read_worker(self, image: Path) -> list[tuple[str, float]]:
        frame = self._prepare_frame(image)
        with self._worker_lock:
            process = self._load_worker()
            assert process.stdin is not None
            assert process.stdout is not None
            request_id = secrets.token_hex(16)
            private_frame: Path | None = None
            try:
                private_frame = self._write_private_frame(request_id, frame)
                request_attestation = (
                    self._ensure_contract_current().attestation.as_dict()
                )
                request = {
                    "attestation": request_attestation,
                    "frame": frame.request_payload(request_id),
                    "request_id": request_id,
                    "type": "read",
                }
                self._write_worker_message(process.stdin, request)
                payload = self._decode_worker_message(
                    self._readline_bounded(
                        process.stdout,
                        maximum_bytes=MAX_OCR_RESPONSE_BYTES,
                        timeout_seconds=self.worker_response_timeout_seconds,
                    )
                )
                current_attestation = (
                    self._ensure_contract_current().attestation.as_dict()
                )
                if (
                    set(payload)
                    != {"attestation", "items", "ok", "request_id", "type"}
                    or payload.get("type") != "result"
                    or payload.get("ok") is not True
                    or payload.get("request_id") != request_id
                    or payload.get("attestation") != request_attestation
                    or current_attestation != request_attestation
                ):
                    raise ValueError("invalid OCR worker result envelope")
                return self._validated_worker_items(payload.get("items"))
            except (BrokenPipeError, OSError, TimeoutError, ValueError) as error:
                self._discard_worker(process)
                raise RuntimeError("PaddleOCR worker returned invalid data") from error
            finally:
                if private_frame is not None:
                    try:
                        private_frame.unlink()
                    except FileNotFoundError:
                        pass
                    except OSError as error:
                        self._discard_worker(process)
                        raise RuntimeError(
                            "PaddleOCR private frame cleanup failed"
                        ) from error

    def _normalized_items(self, items: object) -> list[tuple[str, float]]:
        if not isinstance(items, list):
            return []
        output: list[tuple[str, float]] = []
        for item in items[:MAX_OCR_ITEMS]:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            try:
                normalized = str(item[0]).strip()[:MAX_OCR_TEXT_CHARACTERS]
                confidence = float(item[1])
            except (TypeError, ValueError):
                continue
            if (
                normalized
                and math.isfinite(confidence)
                and self.minimum_confidence <= confidence <= 1
            ):
                output.append((normalized, confidence))
        return output

    def read(self, image: Path) -> list[tuple[str, float]]:
        if self._uses_worker:
            return self._read_worker(image)
        frame = self._prepare_frame(image)
        try:
            decoded = _decode_image_bytes(
                frame.compressed_bytes,
                width=frame.width,
                height=frame.height,
            )
        except ValueError as error:
            raise RuntimeError("PaddleOCR frame is invalid") from error
        items: list[list[object]] = []
        for result in self._load().predict(decoded):
            payload = _result_payload(result)
            texts = payload.get("rec_texts") or []
            scores = payload.get("rec_scores") or []
            for text, score in zip(texts, scores, strict=False):
                items.append([text, score])
        return self._normalized_items(items)
