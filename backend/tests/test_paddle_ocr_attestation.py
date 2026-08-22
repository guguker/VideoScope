from __future__ import annotations

from collections import deque
from hashlib import sha256
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import stat
import struct
import subprocess
import sys
import threading
import time
import types
import zlib

import pytest

from videoscope.providers import paddle_ocr as paddle_module
from videoscope.providers.paddle_ocr import (
    MAX_OCR_RESPONSE_BYTES,
    OCR_MODEL_ARTIFACT_IDENTITY,
    OCR_WORKER_DEPENDENCY_IDENTITY,
    OCR_WORKER_RUNTIME_IDENTITY,
    PaddleOCRReader,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKER_SCRIPT = PROJECT_ROOT / "scripts" / "paddle-ocr-worker.py"
DEPENDENCY_LOCK = PROJECT_ROOT / "workers" / "ocr" / "requirements.lock"
MODEL_MANIFEST = PROJECT_ROOT / "workers" / "ocr" / "model-artifacts.lock.json"


def _png_bytes(
    *,
    width: int = 2,
    height: int = 2,
    bit_depth: int = 8,
    color_type: int = 2,
) -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)

    header = struct.pack(
        ">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0
    )
    if width * height <= 1024 and (bit_depth, color_type) == (8, 2):
        scanline = b"\x00" + (b"\x00\x00\x00" * width)
        image_data = zlib.compress(scanline * height)
    else:
        # The oversized-header test must be rejected before any decoder sees it.
        image_data = zlib.compress(b"")
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", image_data)
        + chunk(b"IEND", b"")
    )


def _jpeg_header_bytes(*, width: int = 3, height: int = 2) -> bytes:
    components = b"\x01\x11\x00\x02\x11\x01\x03\x11\x01"
    start_of_frame = (
        b"\xff\xc0"
        + struct.pack(">H", 17)
        + b"\x08"
        + struct.pack(">HH", height, width)
        + b"\x03"
        + components
    )
    return b"\xff\xd8\xff\xe0\x00\x04JF" + start_of_frame


def _canonical_sha256(payload: object) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(canonical).hexdigest()


def _assert_hash_pinned_lock(lock: bytes) -> None:
    text = lock.decode("utf-8")
    records = re.split(r"(?m)(?=^[A-Za-z0-9_.-]+==)", text)
    package_records = [record for record in records if "==" in record.splitlines()[0]]
    assert package_records
    for record in package_records:
        first_line = record.splitlines()[0]
        assert re.fullmatch(
            r"[A-Za-z0-9_.-]+==[^ ;\\]+(?:\s*;[^\\]+)?\s*\\?",
            first_line,
        )
        assert re.search(r"--hash=sha256:[0-9a-f]{64}", record)


def test_reviewed_ocr_packaging_is_exact_hash_pinned_and_pathless() -> None:
    lock = DEPENDENCY_LOCK.read_bytes()
    _assert_hash_pinned_lock(lock)
    lock_sha = sha256(lock).hexdigest()
    assert OCR_WORKER_DEPENDENCY_IDENTITY == (
        f"paddleocr-deps-v1:sha256:{lock_sha}"
    )

    manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    assert set(manifest) == {"engine", "models", "profile", "schema_version"}
    assert manifest["schema_version"] == 1
    assert manifest["engine"] == "transformers"
    assert manifest["models"]
    for model in manifest["models"]:
        assert set(model) == {"artifacts", "directory", "role"}
        assert "/" not in model["directory"] and "\\" not in model["directory"]
        for artifact in model["artifacts"]:
            assert set(artifact) == {"name", "sha256", "size"}
            assert "/" not in artifact["name"] and "\\" not in artifact["name"]
            assert re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"])
            assert type(artifact["size"]) is int and artifact["size"] > 0
    assert OCR_MODEL_ARTIFACT_IDENTITY == (
        f"paddleocr-models-v1:sha256:{_canonical_sha256(manifest)}"
    )
    assert re.fullmatch(
        r"videoscope-paddleocr-worker-v1\|python==3\.12\.13\|"
        r"platform==aarch64-apple-darwin-macos14plus\|"
        r"lock-sha256:[0-9a-f]{64}",
        OCR_WORKER_RUNTIME_IDENTITY,
    )
    public_identity = "|".join(
        (
            OCR_WORKER_DEPENDENCY_IDENTITY,
            OCR_WORKER_RUNTIME_IDENTITY,
            OCR_MODEL_ARTIFACT_IDENTITY,
        )
    )
    assert str(PROJECT_ROOT) not in public_identity
    assert "secret" not in public_identity.lower()

    worker_module = _load_worker_script_module()
    assert worker_module.OCR_WORKER_DEPENDENCY_IDENTITY == (
        OCR_WORKER_DEPENDENCY_IDENTITY
    )
    assert worker_module.OCR_WORKER_RUNTIME_IDENTITY == OCR_WORKER_RUNTIME_IDENTITY
    assert worker_module.OCR_MODEL_ARTIFACT_IDENTITY == OCR_MODEL_ARTIFACT_IDENTITY

    installer = (PROJECT_ROOT / "scripts" / "install-ocr.sh").read_text(
        encoding="utf-8"
    )
    assert "--require-hashes" in installer
    assert "--only-binary=:all:" in installer
    assert '!= "3.12.13"' in installer
    assert '[[ -e "$OCR_VENV" || -L "$OCR_VENV" ]]' in installer
    assert 'python3.12 -I -m venv "$OCR_VENV"' in installer
    assert (
        '"$OCR_VENV/bin/python" -I -m pip --isolated '
        '--disable-pip-version-check install'
    ) in installer
    assert "transformers>=" not in installer
    assert "torch>=" not in installer


def test_ocr_lock_targets_have_a_nonmutating_exact_host_dry_run() -> None:
    lock_before = DEPENDENCY_LOCK.read_bytes()

    lock_dry_run = subprocess.run(
        ["make", "-n", "lock-ocr"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    check_dry_run = subprocess.run(
        ["make", "-n", "lock-ocr-check"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert lock_dry_run.returncode == 0, lock_dry_run.stderr
    assert check_dry_run.returncode == 0, check_dry_run.stderr
    assert DEPENDENCY_LOCK.read_bytes() == lock_before
    assert "scripts/check-worker-platform.py" in lock_dry_run.stdout
    assert "workers/ocr/pyproject.toml" in lock_dry_run.stdout
    assert "--python .venv/bin/python" in lock_dry_run.stdout
    assert "--generate-hashes" in lock_dry_run.stdout
    assert "--exclude-newer 2026-08-18T00:00:00Z" in lock_dry_run.stdout
    assert "--custom-compile-command 'make lock-ocr'" in lock_dry_run.stdout
    assert "--no-python-downloads" in lock_dry_run.stdout
    assert "--python-platform" not in lock_dry_run.stdout
    assert "mktemp -t videoscope-ocr-lock.XXXXXX" in check_dry_run.stdout
    assert "cmp -s workers/ocr/requirements.lock" in check_dry_run.stdout


def test_ocr_installer_rejects_unsupported_host_before_venv_or_pip() -> None:
    installer = (PROJECT_ROOT / "scripts" / "install-ocr.sh").read_text(
        encoding="utf-8"
    )
    version_check = installer.index("python3.12 -I -c")
    platform_check = installer.index(
        "python3.12 -I scripts/check-worker-platform.py"
    )
    create_environment = installer.index('python3.12 -I -m venv "$OCR_VENV"')
    package_install = installer.index(
        '"$OCR_VENV/bin/python" -I -m pip --isolated '
        '--disable-pip-version-check install'
    )

    assert version_check < platform_check < create_environment < package_install
    assert "PYTHONNOUSERSITE=1" in installer


def test_ocr_docs_do_not_claim_an_unimplemented_clean_model_install() -> None:
    readme = (PROJECT_ROOT / "workers" / "ocr" / "README.md").read_text(
        encoding="utf-8"
    )
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")

    assert "not** a complete clean-install" in readme
    assert "neither a `models-ocr` acquisition" in readme
    assert "command nor a reviewed upstream source/revision" in readme
    assert "source/revision evidence" in readme
    assert "models-ocr:" not in makefile


def _write_fixture_contract(tmp_path: Path) -> tuple[Path, Path, Path, str, str]:
    lock = tmp_path / "requirements.lock"
    lock.write_text(
        "demo-runtime==1.0.0 \\\n"
        "    --hash=sha256:" + "a" * 64 + "\n",
        encoding="utf-8",
    )
    dependency_identity = f"paddleocr-deps-v1:sha256:{sha256(lock.read_bytes()).hexdigest()}"

    model_root = tmp_path / "models"
    model_dir = model_root / "detector"
    model_dir.mkdir(parents=True)
    artifact = model_dir / "model.bin"
    artifact.write_bytes(b"reviewed-model")
    manifest_payload = {
        "engine": "transformers",
        "models": [
            {
                "artifacts": [
                    {
                        "name": artifact.name,
                        "sha256": sha256(artifact.read_bytes()).hexdigest(),
                        "size": artifact.stat().st_size,
                    }
                ],
                "directory": model_dir.name,
                "role": "detection",
            }
        ],
        "profile": "test-profile",
        "schema_version": 1,
    }
    manifest = tmp_path / "model-artifacts.lock.json"
    manifest.write_text(
        json.dumps(manifest_payload, sort_keys=True),
        encoding="utf-8",
    )
    model_identity = (
        f"paddleocr-models-v1:sha256:{_canonical_sha256(manifest_payload)}"
    )
    return lock, manifest, model_root, dependency_identity, model_identity


class _FakeInput:
    def __init__(self, process: "_FakeProcess") -> None:
        self._process = process
        self.writes: list[bytes] = []

    def write(self, value: bytes) -> int:
        self.writes.append(value)
        request = json.loads(value)
        response = self._process.response_factory(request)
        self._process.stdout.lines.append(
            json.dumps(response, separators=(",", ":")).encode("utf-8") + b"\n"
        )
        return len(value)

    def flush(self) -> None:
        pass


class _FakeOutput:
    def __init__(self, lines: list[bytes]) -> None:
        self.lines = deque(lines)

    def readline(self, limit: int = -1) -> bytes:
        if not self.lines:
            return b""
        line = self.lines.popleft()
        return line if limit < 0 else line[:limit]


class _FakeProcess:
    def __init__(self, hello: dict[str, object]) -> None:
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False
        self.stdout = _FakeOutput(
            [json.dumps(hello, separators=(",", ":")).encode("utf-8") + b"\n"]
        )
        self.stdin = _FakeInput(self)
        self.response_factory = lambda request: {
            "attestation": request["attestation"],
            "items": [[" SCORE 90 ", 0.96]],
            "ok": True,
            "request_id": request["request_id"],
            "type": "result",
        }

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def wait(self, timeout: float | None = None) -> int:
        assert timeout is None or timeout >= 0
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


def _reader_fixture(
    tmp_path: Path,
    *,
    include_script_identity: bool = True,
) -> tuple[PaddleOCRReader, Path, Path]:
    python = tmp_path / "python"
    python.write_bytes(b"python-fixture")
    script = tmp_path / "worker.py"
    script.write_bytes(b"print('reviewed worker')\n")
    lock, manifest, model_root, dependency_identity, model_identity = (
        _write_fixture_contract(tmp_path)
    )
    reader = PaddleOCRReader(
        minimum_confidence=0.5,
        worker_python=python,
        worker_script=script,
        worker_dependency_lock=lock,
        worker_model_manifest=manifest,
        worker_model_root=model_root,
        expected_dependency_identity=dependency_identity,
        expected_runtime_identity=(
            "videoscope-paddleocr-worker-v1|python==3.12.13|"
            "platform==aarch64-apple-darwin-macos14plus|lock-sha256:"
            + dependency_identity.rsplit(":", 1)[-1]
        ),
        expected_model_identity=model_identity,
        expected_script_sha256=(
            sha256(script.read_bytes()).hexdigest()
            if include_script_identity
            else None
        ),
    )
    return reader, script, python


def test_reader_uses_reviewed_model_root_from_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configured_root = tmp_path / "configured-models"
    explicit_root = tmp_path / "explicit-models"
    monkeypatch.setenv("VIDEOSCOPE_OCR_MODEL_ROOT", str(configured_root))

    assert PaddleOCRReader().worker_model_root == configured_root
    assert (
        PaddleOCRReader(worker_model_root=explicit_root).worker_model_root
        == explicit_root
    )


def test_worker_mode_requires_an_explicit_reviewed_script_identity(
    tmp_path: Path,
) -> None:
    reader, _script, _python = _reader_fixture(
        tmp_path,
        include_script_identity=False,
    )

    with pytest.raises(ValueError, match="script identity"):
        _ = reader.worker_attestation


def test_reader_response_timeout_covers_a_partial_jsonl_line() -> None:
    read_descriptor, write_descriptor = os.pipe()

    def write_partial_line() -> None:
        try:
            os.write(write_descriptor, b"{")
            time.sleep(0.15)
        finally:
            os.close(write_descriptor)

    writer = threading.Thread(target=write_partial_line)
    writer.start()
    try:
        with os.fdopen(read_descriptor, "rb", buffering=0) as stream:
            with pytest.raises(TimeoutError, match="timed out"):
                PaddleOCRReader._readline_bounded(
                    stream,
                    maximum_bytes=1024,
                    timeout_seconds=0.02,
                )
    finally:
        writer.join()


def test_private_bundle_cleanup_does_not_mask_a_protocol_failure(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "private-bundle"
    bundle.mkdir()
    (bundle / "paddle-ocr-worker.py").write_bytes(b"worker")
    unexpected = bundle / "unexpected"
    unexpected.write_bytes(b"leave-private")

    PaddleOCRReader._remove_private_bundle(bundle)

    assert bundle.is_dir()
    assert unexpected.read_bytes() == b"leave-private"
    assert not (bundle / "paddle-ocr-worker.py").exists()


def _hello(reader: PaddleOCRReader, **changes: object) -> dict[str, object]:
    hello: dict[str, object] = {
        "attestation": reader.worker_attestation,
        "ok": True,
        "type": "hello",
    }
    hello.update(changes)
    return hello


def test_reader_executes_private_verified_script_and_binds_every_message(
    monkeypatch, tmp_path
) -> None:
    reader, source_script, _python = _reader_fixture(tmp_path)
    captured: dict[str, object] = {}
    process = _FakeProcess(_hello(reader))

    def popen(args, **kwargs):  # type: ignore[no-untyped-def]
        captured["args"] = args
        captured["kwargs"] = kwargs
        private_script = Path(args[-1])
        assert private_script != source_script
        assert private_script.read_bytes() == source_script.read_bytes()
        return process

    monkeypatch.setenv("VIDEOSCOPE_TEST_SECRET", "must-not-leak")
    monkeypatch.setattr(paddle_module.subprocess, "Popen", popen)
    image = tmp_path / "frame.jpg"
    image.write_bytes(_png_bytes())

    assert reader.read(image) == [("SCORE 90", 0.96)]
    request = json.loads(process.stdin.writes[0])
    assert set(request) == {"attestation", "frame", "request_id", "type"}
    assert request["attestation"] == reader.worker_attestation
    assert re.fullmatch(r"[0-9a-f]{32}", request["request_id"])
    assert request["type"] == "read"
    assert captured["args"][:2] == [str(reader.worker_python), "-I"]
    environment = captured["kwargs"]["env"]
    assert "VIDEOSCOPE_TEST_SECRET" not in environment
    private_script = Path(captured["args"][-1])

    reader.close()

    assert process.terminated
    assert not private_script.exists()


def test_reader_discards_worker_when_contract_changes_during_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    reader, source_script, _python = _reader_fixture(tmp_path)
    process = _FakeProcess(_hello(reader))

    def response_factory(request: dict[str, object]) -> dict[str, object]:
        source_script.write_bytes(b"tampered-during-request\n")
        return {
            "attestation": request["attestation"],
            "items": [["stale", 0.99]],
            "ok": True,
            "request_id": request["request_id"],
            "type": "result",
        }

    process.response_factory = response_factory
    monkeypatch.setattr(
        paddle_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )
    image = tmp_path / "frame.jpg"
    image.write_bytes(_png_bytes())

    with pytest.raises(RuntimeError, match="invalid data"):
        reader.read(image)
    assert process.terminated
    assert reader._worker_process is None


def test_reader_destroys_mismatched_startup_and_never_reuses_it(
    monkeypatch, tmp_path
) -> None:
    reader, _script, _python = _reader_fixture(tmp_path)
    bad_attestation = dict(reader.worker_attestation)
    bad_attestation["model_identity"] = "paddleocr-models-v1:sha256:" + "0" * 64
    first = _FakeProcess(_hello(reader, attestation=bad_attestation))
    second = _FakeProcess(_hello(reader))
    processes = deque([first, second])
    monkeypatch.setattr(
        paddle_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: processes.popleft(),
    )
    image = tmp_path / "frame.jpg"
    image.write_bytes(_png_bytes())

    with pytest.raises(RuntimeError, match="attestation"):
        reader.read(image)
    assert first.terminated
    assert reader._worker_process is None

    assert reader.read(image) == [("SCORE 90", 0.96)]
    assert not second.terminated


def test_reader_destroys_process_on_response_attestation_or_request_mismatch(
    monkeypatch, tmp_path
) -> None:
    reader, _script, _python = _reader_fixture(tmp_path)
    process = _FakeProcess(_hello(reader))

    def mismatched_response(request):  # type: ignore[no-untyped-def]
        return {
            "attestation": request["attestation"],
            "items": [],
            "ok": True,
            "request_id": "0" * 32,
            "type": "result",
        }

    process.response_factory = mismatched_response
    monkeypatch.setattr(
        paddle_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )
    image = tmp_path / "frame.jpg"
    image.write_bytes(_png_bytes())

    with pytest.raises(RuntimeError, match="invalid data"):
        reader.read(image)

    assert process.terminated
    assert reader._worker_process is None


def test_reader_destroys_process_on_response_attestation_mismatch(
    monkeypatch, tmp_path
) -> None:
    reader, _script, _python = _reader_fixture(tmp_path)
    process = _FakeProcess(_hello(reader))

    def mismatched_response(request):  # type: ignore[no-untyped-def]
        attestation = dict(request["attestation"])
        attestation["dependency_identity"] = (
            "paddleocr-deps-v1:sha256:" + "0" * 64
        )
        return {
            "attestation": attestation,
            "items": [],
            "ok": True,
            "request_id": request["request_id"],
            "type": "result",
        }

    process.response_factory = mismatched_response
    monkeypatch.setattr(
        paddle_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )
    image = tmp_path / "frame.jpg"
    image.write_bytes(_png_bytes())

    with pytest.raises(RuntimeError, match="invalid data"):
        reader.read(image)

    assert process.terminated
    assert reader._worker_process is None


def test_reader_rejects_stale_source_script_before_reusing_process(
    monkeypatch, tmp_path
) -> None:
    reader, script, _python = _reader_fixture(tmp_path)
    process = _FakeProcess(_hello(reader))
    monkeypatch.setattr(
        paddle_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )
    image = tmp_path / "frame.jpg"
    image.write_bytes(_png_bytes())
    assert reader.read(image)
    writes_before = len(process.stdin.writes)

    script.write_bytes(b"print('changed')\n")

    with pytest.raises(RuntimeError, match="script"):
        reader.read(image)
    assert process.terminated
    assert len(process.stdin.writes) == writes_before


@pytest.mark.parametrize("changed_asset", ["dependency-lock", "model-artifact"])
def test_reader_rejects_stale_attested_assets_before_reusing_process(
    monkeypatch, tmp_path, changed_asset
) -> None:
    reader, _script, _python = _reader_fixture(tmp_path)
    process = _FakeProcess(_hello(reader))
    monkeypatch.setattr(
        paddle_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )
    image = tmp_path / "frame.jpg"
    image.write_bytes(_png_bytes())
    assert reader.read(image)

    if changed_asset == "dependency-lock":
        reader.worker_dependency_lock.write_bytes(
            reader.worker_dependency_lock.read_bytes() + b"\n"
        )
    else:
        artifact = reader.worker_model_root / "detector" / "model.bin"
        artifact.write_bytes(b"tampered-model")

    with pytest.raises(RuntimeError, match="attestation"):
        reader.read(image)

    assert process.terminated
    assert reader._worker_process is None


def test_reader_rejects_symlinked_worker_source(tmp_path) -> None:
    reader, script, _python = _reader_fixture(tmp_path)
    linked = tmp_path / "linked-worker.py"
    linked.symlink_to(script)
    reader.worker_script = linked

    with pytest.raises(ValueError, match="reviewed OCR file"):
        _ = reader.worker_attestation


def test_reader_rejects_oversized_or_unterminated_response(
    monkeypatch, tmp_path
) -> None:
    reader, _script, _python = _reader_fixture(tmp_path)
    process = _FakeProcess(_hello(reader))
    process.response_factory = lambda _request: pytest.fail("response not expected")
    process.stdin.write = lambda value: len(value)  # type: ignore[method-assign]
    process.stdin.flush = lambda: None  # type: ignore[method-assign]
    process.stdout.lines.append(b"x" * (MAX_OCR_RESPONSE_BYTES + 1))
    monkeypatch.setattr(
        paddle_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )
    image = tmp_path / "frame.jpg"
    image.write_bytes(_png_bytes())

    with pytest.raises(RuntimeError, match="invalid data"):
        reader.read(image)

    assert process.terminated


def _load_worker_script_module() -> types.ModuleType:
    specification = importlib.util.spec_from_file_location(
        "videoscope_test_paddle_worker", WORKER_SCRIPT
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_worker_request_contract_is_exact_bounded_and_pathless_on_error(tmp_path) -> None:
    worker = _load_worker_script_module()
    attestation = {
        "dependency_identity": "paddleocr-deps-v1:sha256:" + "a" * 64,
        "model_identity": "paddleocr-models-v1:sha256:" + "b" * 64,
        "protocol": "videoscope.paddleocr-jsonl.v2",
        "runtime_identity": "runtime",
        "script_sha256": "c" * 64,
    }
    payload = _png_bytes()
    image = tmp_path / f"{'d' * 32}.frame"
    image.write_bytes(payload)

    class FakeResult:
        json = {
            "res": {
                "rec_scores": [0.9],
                "rec_texts": [" scoreboard "],
            }
        }

    model = types.SimpleNamespace(predict=lambda _decoded: [FakeResult()])
    request = _bound_frame_request(payload)
    request["attestation"] = attestation
    worker._decode_image_bytes = lambda _raw, **_kwargs: object()

    assert worker._handle_request(
        request, model, attestation, frame_root=tmp_path
    ) == {
        "attestation": attestation,
        "items": [["scoreboard", 0.9]],
        "ok": True,
        "request_id": "d" * 32,
        "type": "result",
    }
    with pytest.raises(worker.ProtocolError):
        worker._handle_request(
            {**request, "extra": True},
            model,
            attestation,
            frame_root=tmp_path,
        )
    with pytest.raises(worker.ProtocolError):
        worker._handle_request(
            {**request, "attestation": {}},
            model,
            attestation,
            frame_root=tmp_path,
        )

    error = worker._error_response("invalid_request", request_id="d" * 32)
    serialized = worker._encode_response(error)
    assert len(serialized) <= MAX_OCR_RESPONSE_BYTES
    assert str(image).encode("utf-8") not in serialized
    assert b"Traceback" not in serialized


def test_worker_dependency_lock_parser_requires_exact_hashed_records() -> None:
    worker = _load_worker_script_module()
    packages = worker._parse_dependency_lock(DEPENDENCY_LOCK.read_bytes())

    assert len(packages) == 77
    assert packages["paddleocr"] == "3.7.0"
    assert packages["paddlex"] == "3.7.2"
    assert packages["torch"] == "2.13.0"
    assert packages["transformers"] == "5.14.1"
    with pytest.raises(worker.ProtocolError, match="not hashed"):
        worker._parse_dependency_lock(b"demo==1.0\n")


def test_worker_fails_closed_instead_of_truncating_model_output(
    monkeypatch, tmp_path
) -> None:
    worker = _load_worker_script_module()
    monkeypatch.setattr(worker, "MAX_OCR_ITEMS", 1)
    attestation = {
        "dependency_identity": "paddleocr-deps-v1:sha256:" + "a" * 64,
        "model_identity": "paddleocr-models-v1:sha256:" + "b" * 64,
        "protocol": "videoscope.paddleocr-jsonl.v2",
        "runtime_identity": "runtime",
        "script_sha256": "c" * 64,
    }
    payload = _png_bytes()
    image = tmp_path / f"{'d' * 32}.frame"
    image.write_bytes(payload)
    result = types.SimpleNamespace(
        json={"res": {"rec_scores": [0.9, 0.8], "rec_texts": ["one", "two"]}}
    )
    model = types.SimpleNamespace(predict=lambda _decoded: [result])
    request = _bound_frame_request(payload)
    request["attestation"] = attestation
    monkeypatch.setattr(worker, "_decode_image_bytes", lambda _raw, **_kwargs: object())

    with pytest.raises(worker.ProtocolError, match="limit"):
        worker._handle_request(
            request,
            model,
            attestation,
            frame_root=tmp_path,
        )


def _write_worker_startup_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, str, str]:
    lock = tmp_path / "worker-requirements.lock"
    lock.write_text(
        "demo-runtime==1.0.0 \\\n"
        "    --hash=sha256:" + "a" * 64 + "\n",
        encoding="utf-8",
    )
    model_root = tmp_path / "worker-models"
    models: list[dict[str, object]] = []
    for role in ("detection", "recognition"):
        directory = model_root / role
        directory.mkdir(parents=True)
        artifact = directory / "model.bin"
        artifact.write_bytes(f"{role}-model".encode("utf-8"))
        models.append(
            {
                "artifacts": [
                    {
                        "name": artifact.name,
                        "sha256": sha256(artifact.read_bytes()).hexdigest(),
                        "size": artifact.stat().st_size,
                    }
                ],
                "directory": directory.name,
                "role": role,
            }
        )
    manifest_payload = {
        "engine": "transformers",
        "models": models,
        "profile": "fixture-profile",
        "schema_version": 1,
    }
    manifest = tmp_path / "worker-models.lock.json"
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    script = tmp_path / "attested-worker.py"
    script.write_bytes(b"# reviewed worker\n")
    dependency_hash = sha256(lock.read_bytes()).hexdigest()
    model_hash = _canonical_sha256(manifest_payload)
    return lock, manifest, model_root, script, dependency_hash, model_hash


def test_worker_startup_attestation_verifies_lock_runtime_script_and_models(
    monkeypatch, tmp_path
) -> None:
    worker = _load_worker_script_module()
    lock, manifest, model_root, script, dependency_hash, model_hash = (
        _write_worker_startup_fixture(tmp_path)
    )
    monkeypatch.setattr(
        worker,
        "OCR_WORKER_DEPENDENCY_IDENTITY",
        f"paddleocr-deps-v1:sha256:{dependency_hash}",
    )
    monkeypatch.setattr(
        worker,
        "OCR_MODEL_ARTIFACT_IDENTITY",
        f"paddleocr-models-v1:sha256:{model_hash}",
    )
    runtime_identity = (
        "videoscope-paddleocr-worker-v1|python==3.12.13|"
        "platform==aarch64-apple-darwin-macos14plus|"
        f"lock-sha256:{dependency_hash}"
    )
    monkeypatch.setattr(worker, "OCR_WORKER_RUNTIME_IDENTITY", runtime_identity)
    captured: dict[str, str] = {}

    def verify_installed(packages):  # type: ignore[no-untyped-def]
        captured.update(packages)

    monkeypatch.setattr(worker, "_verify_installed_dependencies", verify_installed)

    attestation, directories = worker._startup_attestation(
        dependency_lock_path=lock,
        model_manifest_path=manifest,
        model_root=model_root,
        script_path=script,
    )

    assert captured == {"demo-runtime": "1.0.0"}
    assert directories == {
        "detection": model_root / "detection",
        "recognition": model_root / "recognition",
    }
    assert attestation == {
        "dependency_identity": f"paddleocr-deps-v1:sha256:{dependency_hash}",
        "model_identity": f"paddleocr-models-v1:sha256:{model_hash}",
        "protocol": worker.OCR_WORKER_PROTOCOL,
        "runtime_identity": runtime_identity,
        "script_sha256": sha256(script.read_bytes()).hexdigest(),
    }


def test_worker_startup_rejects_tampered_model_artifact(monkeypatch, tmp_path) -> None:
    worker = _load_worker_script_module()
    lock, manifest, model_root, script, dependency_hash, model_hash = (
        _write_worker_startup_fixture(tmp_path)
    )
    monkeypatch.setattr(
        worker,
        "OCR_WORKER_DEPENDENCY_IDENTITY",
        f"paddleocr-deps-v1:sha256:{dependency_hash}",
    )
    monkeypatch.setattr(
        worker,
        "OCR_MODEL_ARTIFACT_IDENTITY",
        f"paddleocr-models-v1:sha256:{model_hash}",
    )
    monkeypatch.setattr(worker, "_verify_installed_dependencies", lambda _packages: None)
    (model_root / "recognition" / "model.bin").write_bytes(b"tampered-modelxxx")

    with pytest.raises(worker.ProtocolError, match="model artifact"):
        worker._startup_attestation(
            dependency_lock_path=lock,
            model_manifest_path=manifest,
            model_root=model_root,
            script_path=script,
        )


def test_worker_dependency_and_runtime_validation_fail_closed(monkeypatch) -> None:
    worker = _load_worker_script_module()

    class Distribution:
        def __init__(self, name: str, version: str) -> None:
            self.metadata = {"Name": name}
            self.version = version

    monkeypatch.setattr(
        worker.importlib_metadata,
        "version",
        lambda name: "1.0" if name == "good" else "0.9",
    )
    monkeypatch.setattr(
        worker.importlib_metadata,
        "distributions",
        lambda: [Distribution("good", "1.0"), Distribution("pip", "25.0.1")],
    )
    worker._verify_installed_dependencies({"good": "1.0"})
    with pytest.raises(worker.ProtocolError, match="version mismatch"):
        worker._verify_installed_dependencies({"bad": "1.0"})

    monkeypatch.setattr(
        worker.importlib_metadata,
        "distributions",
        lambda: [
            Distribution("good", "1.0"),
            Distribution("pip", "25.0.1"),
            Distribution("unreviewed", "9.9"),
        ],
    )
    with pytest.raises(worker.ProtocolError, match="dependency set"):
        worker._verify_installed_dependencies({"good": "1.0"})

    monkeypatch.setattr(worker.sys, "version_info", (3, 11, 9))
    with pytest.raises(worker.ProtocolError, match="Python"):
        worker._runtime_identity("a" * 64)

    monkeypatch.setattr(worker.sys, "version_info", (3, 12, 13))
    monkeypatch.setattr(worker.platform, "python_implementation", lambda: "PyPy")
    with pytest.raises(worker.ProtocolError, match="CPython"):
        worker._runtime_identity("a" * 64)

    monkeypatch.setattr(worker.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(worker.sys, "platform", "darwin")
    monkeypatch.setattr(worker.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(worker.platform, "mac_ver", lambda: ("13.6.9", (), ""))
    with pytest.raises(worker.ProtocolError, match="macOS"):
        worker._runtime_identity("a" * 64)
    monkeypatch.setattr(worker.platform, "mac_ver", lambda: ("14.0", (), ""))
    assert worker._runtime_identity("a" * 64) == (
        "videoscope-paddleocr-worker-v1|python==3.12.13|"
        "platform==aarch64-apple-darwin-macos14plus|lock-sha256:"
        + "a" * 64
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update({"extra": True}),
        lambda payload: payload.update({"schema_version": 2}),
        lambda payload: payload.update({"engine": "paddle"}),
        lambda payload: payload.update({"models": []}),
        lambda payload: payload["models"][0].update({"directory": "../escape"}),
        lambda payload: payload["models"][0].update({"directory": ".."}),
        lambda payload: payload["models"][0]["artifacts"][0].update(
            {"sha256": "invalid"}
        ),
    ],
)
def test_worker_model_manifest_rejects_unreviewed_shapes(mutation) -> None:
    worker = _load_worker_script_module()
    payload = {
        "engine": "transformers",
        "models": [
            {
                "artifacts": [
                    {"name": "model.bin", "sha256": "a" * 64, "size": 1}
                ],
                "directory": "detector",
                "role": "detection",
            }
        ],
        "profile": "fixture",
        "schema_version": 1,
    }
    mutation(payload)
    encoded = json.dumps(payload).encode("utf-8")

    with pytest.raises(worker.ProtocolError, match="model manifest"):
        worker._parse_model_manifest(encoded)
    with pytest.raises(ValueError, match="model manifest"):
        paddle_module._parse_model_manifest(encoded)


def test_worker_strict_json_and_response_envelope_reject_ambiguous_values() -> None:
    worker = _load_worker_script_module()
    with pytest.raises(worker.ProtocolError, match="invalid JSON"):
        worker._strict_json_bytes(b'{"x":1,"x":2}')
    with pytest.raises(worker.ProtocolError, match="invalid JSON"):
        worker._strict_json_bytes(b'{"x":NaN}')
    with pytest.raises(worker.ProtocolError, match="invalid response"):
        worker._canonical_json_bytes({"x": float("nan")})

    fallback = worker._error_response("private/path/leak", request_id="bad")
    assert fallback == {"error_code": "model_failure", "ok": False, "type": "error"}


class _BinaryConsole:
    def __init__(self, payload: bytes = b"") -> None:
        self.buffer = io.BytesIO(payload)


class _FakePrivateModelBundle:
    def __init__(
        self,
        tmp_path: Path,
        *,
        current_error: Exception | None = None,
    ) -> None:
        self.directories = {
            "detection": tmp_path / "private-det",
            "recognition": tmp_path / "private-rec",
        }
        self.current_error = current_error
        self.cleaned = False

    def assert_current(self) -> None:
        if self.current_error is not None:
            raise self.current_error

    def cleanup(self) -> None:
        self.cleaned = True


def _use_fake_private_models(
    monkeypatch: pytest.MonkeyPatch,
    worker: types.ModuleType,
    tmp_path: Path,
    *,
    current_error: Exception | None = None,
) -> _FakePrivateModelBundle:
    bundle = _FakePrivateModelBundle(tmp_path, current_error=current_error)
    monkeypatch.setattr(
        worker,
        "_create_private_model_bundle",
        lambda **_kwargs: bundle,
    )
    return bundle


def test_worker_main_attest_only_emits_one_pathless_hello(monkeypatch) -> None:
    worker = _load_worker_script_module()
    attestation = {
        "dependency_identity": "paddleocr-deps-v1:sha256:" + "a" * 64,
        "model_identity": "paddleocr-models-v1:sha256:" + "b" * 64,
        "protocol": worker.OCR_WORKER_PROTOCOL,
        "runtime_identity": "runtime",
        "script_sha256": "c" * 64,
    }
    output = _BinaryConsole()
    monkeypatch.setattr(worker.sys, "stdin", _BinaryConsole())
    monkeypatch.setattr(worker.sys, "stdout", output)
    monkeypatch.setattr(worker.sys, "argv", ["worker.py", "--attest-only"])
    monkeypatch.setattr(
        worker,
        "_startup_attestation",
        lambda **_kwargs: (attestation, {}),
    )
    monkeypatch.setattr(worker, "_environment_path", lambda name: Path(name))

    assert worker.main() == 0
    assert json.loads(output.buffer.getvalue()) == {
        "attestation": attestation,
        "ok": True,
        "type": "hello",
    }


def test_worker_main_processes_one_bound_request_without_real_model(
    monkeypatch, tmp_path
) -> None:
    worker = _load_worker_script_module()
    attestation = {
        "dependency_identity": "paddleocr-deps-v1:sha256:" + "a" * 64,
        "model_identity": "paddleocr-models-v1:sha256:" + "b" * 64,
        "protocol": worker.OCR_WORKER_PROTOCOL,
        "runtime_identity": "runtime",
        "script_sha256": "c" * 64,
    }
    payload = _png_bytes()
    image = tmp_path / f"{'d' * 32}.frame"
    image.write_bytes(payload)
    request = _bound_frame_request(payload)
    request["attestation"] = attestation
    input_console = _BinaryConsole(
        json.dumps(request, separators=(",", ":")).encode("utf-8") + b"\n"
    )
    output_console = _BinaryConsole()
    monkeypatch.setattr(worker.sys, "stdin", input_console)
    monkeypatch.setattr(worker.sys, "stdout", output_console)
    monkeypatch.setattr(worker.sys, "argv", ["worker.py"])
    monkeypatch.setattr(worker, "_environment_path", lambda _name: tmp_path)
    monkeypatch.setattr(
        worker,
        "_startup_attestation",
        lambda **_kwargs: (
            attestation,
            {"detection": tmp_path / "det", "recognition": tmp_path / "rec"},
        ),
    )
    bundle = _use_fake_private_models(monkeypatch, worker, tmp_path)
    decoded = object()
    monkeypatch.setattr(
        worker,
        "_decode_image_bytes",
        lambda _raw, **_kwargs: decoded,
    )

    class FakeResult:
        json = {"res": {"rec_scores": [0.95], "rec_texts": [" SCORE "]}}

    class FakePaddleOCR:
        def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
            assert kwargs["engine"] == "transformers"

        def predict(self, value: object):
            assert value is decoded
            return [FakeResult()]

    fake_module = types.ModuleType("paddleocr")
    fake_module.PaddleOCR = FakePaddleOCR  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "paddleocr", fake_module)

    assert worker.main() == 0
    messages = [json.loads(line) for line in output_console.buffer.getvalue().splitlines()]
    assert messages[0] == {"attestation": attestation, "ok": True, "type": "hello"}
    assert messages[1] == {
        "attestation": attestation,
        "items": [["SCORE", 0.95]],
        "ok": True,
        "request_id": "d" * 32,
        "type": "result",
    }
    assert bundle.cleaned
    assert not image.exists()


def test_worker_rechecks_private_artifacts_after_model_initialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    worker = _load_worker_script_module()
    attestation = {
        "dependency_identity": "paddleocr-deps-v1:sha256:" + "a" * 64,
        "model_identity": "paddleocr-models-v1:sha256:" + "b" * 64,
        "protocol": worker.OCR_WORKER_PROTOCOL,
        "runtime_identity": "runtime",
        "script_sha256": "c" * 64,
    }
    output = _BinaryConsole()
    monkeypatch.setattr(worker.sys, "stdin", _BinaryConsole())
    monkeypatch.setattr(worker.sys, "stdout", output)
    monkeypatch.setattr(worker.sys, "argv", ["worker.py"])
    monkeypatch.setattr(worker, "_environment_path", lambda name: Path(name))
    monkeypatch.setattr(
        worker,
        "_startup_attestation",
        lambda **_kwargs: (
            attestation,
            {"detection": tmp_path / "det", "recognition": tmp_path / "rec"},
        ),
    )
    bundle = _use_fake_private_models(
        monkeypatch,
        worker,
        tmp_path,
        current_error=worker.ProtocolError("private model artifact changed"),
    )
    fake_module = types.ModuleType("paddleocr")
    fake_module.PaddleOCR = lambda **_kwargs: object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "paddleocr", fake_module)

    assert worker.main() == 2
    assert bundle.cleaned
    assert json.loads(output.buffer.getvalue()) == {
        "error_code": "startup_failed",
        "ok": False,
        "type": "error",
    }


def test_worker_main_returns_bounded_generic_errors(monkeypatch) -> None:
    worker = _load_worker_script_module()
    output = _BinaryConsole()
    monkeypatch.setattr(worker.sys, "stdin", _BinaryConsole())
    monkeypatch.setattr(worker.sys, "stdout", output)
    monkeypatch.setattr(worker.sys, "argv", ["worker.py", "--unexpected"])

    assert worker.main() == 2
    assert json.loads(output.buffer.getvalue()) == {
        "error_code": "startup_failed",
        "ok": False,
        "type": "error",
    }


def test_worker_main_exits_after_one_invalid_bound_request(monkeypatch, tmp_path) -> None:
    worker = _load_worker_script_module()
    attestation = {
        "dependency_identity": "paddleocr-deps-v1:sha256:" + "a" * 64,
        "model_identity": "paddleocr-models-v1:sha256:" + "b" * 64,
        "protocol": worker.OCR_WORKER_PROTOCOL,
        "runtime_identity": "runtime",
        "script_sha256": "c" * 64,
    }
    output = _BinaryConsole()
    monkeypatch.setattr(worker.sys, "stdin", _BinaryConsole(b"{}\n"))
    monkeypatch.setattr(worker.sys, "stdout", output)
    monkeypatch.setattr(worker.sys, "argv", ["worker.py"])
    monkeypatch.setattr(worker, "_environment_path", lambda name: Path(name))
    monkeypatch.setattr(
        worker,
        "_startup_attestation",
        lambda **_kwargs: (
            attestation,
            {"detection": tmp_path / "det", "recognition": tmp_path / "rec"},
        ),
    )
    bundle = _use_fake_private_models(monkeypatch, worker, tmp_path)
    fake_module = types.ModuleType("paddleocr")
    fake_module.PaddleOCR = lambda **_kwargs: types.SimpleNamespace(  # type: ignore[attr-defined]
        predict=lambda _path: []
    )
    monkeypatch.setitem(sys.modules, "paddleocr", fake_module)

    assert worker.main() == 3
    messages = [json.loads(line) for line in output.buffer.getvalue().splitlines()]
    assert messages == [
        {"attestation": attestation, "ok": True, "type": "hello"},
        {
            "attestation": attestation,
            "error_code": "invalid_request",
            "ok": False,
            "type": "error",
        },
    ]
    assert bundle.cleaned


def test_worker_main_exits_on_oversized_jsonl_before_parsing(monkeypatch, tmp_path) -> None:
    worker = _load_worker_script_module()
    attestation = {
        "dependency_identity": "paddleocr-deps-v1:sha256:" + "a" * 64,
        "model_identity": "paddleocr-models-v1:sha256:" + "b" * 64,
        "protocol": worker.OCR_WORKER_PROTOCOL,
        "runtime_identity": "runtime",
        "script_sha256": "c" * 64,
    }
    output = _BinaryConsole()
    monkeypatch.setattr(
        worker.sys,
        "stdin",
        _BinaryConsole(b"x" * (worker.MAX_OCR_REQUEST_BYTES + 1)),
    )
    monkeypatch.setattr(worker.sys, "stdout", output)
    monkeypatch.setattr(worker.sys, "argv", ["worker.py"])
    monkeypatch.setattr(worker, "_environment_path", lambda name: Path(name))
    monkeypatch.setattr(
        worker,
        "_startup_attestation",
        lambda **_kwargs: (
            attestation,
            {"detection": tmp_path / "det", "recognition": tmp_path / "rec"},
        ),
    )
    bundle = _use_fake_private_models(monkeypatch, worker, tmp_path)
    fake_module = types.ModuleType("paddleocr")
    fake_module.PaddleOCR = lambda **_kwargs: types.SimpleNamespace(  # type: ignore[attr-defined]
        predict=lambda _path: []
    )
    monkeypatch.setitem(sys.modules, "paddleocr", fake_module)

    assert worker.main() == 3
    messages = [json.loads(line) for line in output.buffer.getvalue().splitlines()]
    assert messages[-1] == {
        "attestation": attestation,
        "error_code": "protocol_limit",
        "ok": False,
        "type": "error",
    }
    assert bundle.cleaned


def test_reader_sends_an_exact_pathless_private_frame_contract_and_cleans_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    reader, _script, _python = _reader_fixture(tmp_path)
    process = _FakeProcess(_hello(reader))
    captured: dict[str, object] = {}

    def popen(args, **kwargs):  # type: ignore[no-untyped-def]
        captured["environment"] = kwargs["env"]
        return process

    def inspect_request(request: dict[str, object]) -> dict[str, object]:
        assert set(request) == {"attestation", "frame", "request_id", "type"}
        assert str(image) not in json.dumps(request)
        frame = request["frame"]
        assert isinstance(frame, dict)
        assert set(frame) == {
            "compressed_bytes",
            "height",
            "name",
            "sha256",
            "width",
        }
        assert frame["name"] == f"{request['request_id']}.frame"
        assert frame["compressed_bytes"] == len(payload)
        assert frame["sha256"] == sha256(payload).hexdigest()
        assert (frame["width"], frame["height"]) == (2, 2)
        environment = captured["environment"]
        assert isinstance(environment, dict)
        frame_root = Path(environment["VIDEOSCOPE_OCR_FRAME_ROOT"])
        private_frame = frame_root / str(frame["name"])
        captured["private_frame"] = private_frame
        assert private_frame.read_bytes() == payload
        assert stat.S_IMODE(private_frame.stat().st_mode) == 0o400
        return {
            "attestation": request["attestation"],
            "items": [["private", 0.99]],
            "ok": True,
            "request_id": request["request_id"],
            "type": "result",
        }

    process.response_factory = inspect_request
    monkeypatch.setattr(paddle_module.subprocess, "Popen", popen)
    image = tmp_path / "source-frame.png"
    payload = _png_bytes()
    image.write_bytes(payload)

    assert reader.read(image) == [("private", 0.99)]
    private_frame = captured["private_frame"]
    assert isinstance(private_frame, Path)
    assert not private_frame.exists()

    bundle = reader._worker_bundle_directory
    reader.close()
    assert bundle is not None and not bundle.exists()


def test_reader_rejects_source_path_swap_during_descriptor_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    reader, _script, _python = _reader_fixture(tmp_path)
    process = _FakeProcess(_hello(reader))
    monkeypatch.setattr(
        paddle_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )
    image = tmp_path / "frame.png"
    image.write_bytes(_png_bytes() + b"x" * (70 * 1024))
    original_inode = image.stat().st_ino
    replacement = tmp_path / "replacement.png"
    replacement.write_bytes(_png_bytes(width=3, height=3))
    real_read = paddle_module.os.read
    swapped = False

    def swapping_read(descriptor: int, count: int) -> bytes:
        nonlocal swapped
        chunk = real_read(descriptor, count)
        if not swapped and os.fstat(descriptor).st_ino == original_inode:
            swapped = True
            os.replace(replacement, image)
        return chunk

    monkeypatch.setattr(paddle_module.os, "read", swapping_read)

    with pytest.raises(RuntimeError, match="frame"):
        reader.read(image)

    assert swapped
    assert process.stdin.writes == []


def test_reader_rejects_decompression_bomb_before_starting_worker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    reader, _script, _python = _reader_fixture(tmp_path)
    popen_called = False

    def popen(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        nonlocal popen_called
        popen_called = True
        return _FakeProcess(_hello(reader))

    monkeypatch.setattr(paddle_module.subprocess, "Popen", popen)
    image = tmp_path / "bomb.png"
    image.write_bytes(_png_bytes(width=100_000, height=100_000))

    with pytest.raises(RuntimeError, match="frame"):
        reader.read(image)

    assert not popen_called


def test_reader_rejects_hardlinked_frame_before_starting_worker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    reader, _script, _python = _reader_fixture(tmp_path)
    original = tmp_path / "original.png"
    original.write_bytes(_png_bytes())
    linked = tmp_path / "linked.png"
    os.link(original, linked)
    monkeypatch.setattr(
        paddle_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("worker must not start"),
    )

    with pytest.raises(RuntimeError, match="frame"):
        reader.read(linked)


@pytest.mark.parametrize("kind", ["symlink", "fifo", "directory"])
def test_reader_rejects_symlink_or_special_frame(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    kind: str,
) -> None:
    reader, _script, _python = _reader_fixture(tmp_path)
    frame = tmp_path / "unsafe-frame"
    if kind == "symlink":
        target = tmp_path / "target.png"
        target.write_bytes(_png_bytes())
        frame.symlink_to(target)
    elif kind == "fifo":
        os.mkfifo(frame)
    else:
        frame.mkdir()
    monkeypatch.setattr(
        paddle_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("worker must not start"),
    )

    with pytest.raises(RuntimeError, match="frame"):
        reader.read(frame)


def _bound_frame_request(
    payload: bytes,
    *,
    request_id: str = "d" * 32,
    width: int = 2,
    height: int = 2,
) -> dict[str, object]:
    return {
        "attestation": {
            "dependency_identity": "paddleocr-deps-v1:sha256:" + "a" * 64,
            "model_identity": "paddleocr-models-v1:sha256:" + "b" * 64,
            "protocol": "videoscope.paddleocr-jsonl.v2",
            "runtime_identity": "runtime",
            "script_sha256": "c" * 64,
        },
        "frame": {
            "compressed_bytes": len(payload),
            "height": height,
            "name": f"{request_id}.frame",
            "sha256": sha256(payload).hexdigest(),
            "width": width,
        },
        "request_id": request_id,
        "type": "read",
    }


def test_provider_and_worker_accept_bounded_png_and_jpeg_headers() -> None:
    worker = _load_worker_script_module()

    assert paddle_module._bounded_image_dimensions(_png_bytes()) == (2, 2)
    assert worker._bounded_image_dimensions(_png_bytes()) == (2, 2)
    assert paddle_module._bounded_image_dimensions(
        _jpeg_header_bytes()
    ) == (3, 2)
    assert worker._bounded_image_dimensions(_jpeg_header_bytes()) == (3, 2)


@pytest.mark.parametrize(
    "payload",
    [
        _png_bytes(width=8192, height=4097),
        _png_bytes(
            width=4096,
            height=4097,
            bit_depth=16,
            color_type=6,
        ),
    ],
)
def test_provider_and_worker_reject_pixel_or_decoded_ram_bombs(
    payload: bytes,
) -> None:
    worker = _load_worker_script_module()

    with pytest.raises(ValueError, match="decode limits"):
        paddle_module._bounded_image_dimensions(payload)
    with pytest.raises(worker.ProtocolError, match="decode limit"):
        worker._bounded_image_dimensions(payload)


def test_worker_decodes_a_bounded_png_to_the_exact_declared_shape() -> None:
    worker = _load_worker_script_module()

    decoded = worker._decode_image_bytes(_png_bytes(), width=2, height=2)

    assert decoded.shape == (2, 2, 3)
    assert decoded.nbytes == 12

    with pytest.raises(worker.ProtocolError, match="decoded frame"):
        worker._decode_image_bytes(_png_bytes(), width=3, height=2)
    with pytest.raises(worker.ProtocolError, match="decoded frame"):
        worker._decode_image_bytes(b"not-an-image", width=2, height=2)


@pytest.mark.parametrize(
    "field,value",
    [
        ("compressed_bytes", 1),
        ("height", 3),
        ("sha256", "0" * 64),
        ("width", 3),
    ],
)
def test_worker_rejects_every_private_frame_binding_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    worker = _load_worker_script_module()
    payload = _png_bytes()
    request = _bound_frame_request(payload)
    frame_contract = request["frame"]
    assert isinstance(frame_contract, dict)
    frame_contract[field] = value
    attestation = request["attestation"]
    assert isinstance(attestation, dict)
    frame = tmp_path / f"{'d' * 32}.frame"
    frame.write_bytes(payload)
    monkeypatch.setattr(
        worker,
        "_decode_image_bytes",
        lambda *_args, **_kwargs: pytest.fail("mismatched frame must not decode"),
    )

    with pytest.raises(worker.ProtocolError, match="frame"):
        worker._handle_request(
            request,
            types.SimpleNamespace(predict=lambda _decoded: []),
            attestation,
            frame_root=tmp_path,
        )


def test_worker_decodes_verified_bytes_without_passing_a_path_to_the_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    worker = _load_worker_script_module()
    payload = _png_bytes()
    request = _bound_frame_request(payload)
    attestation = request["attestation"]
    assert isinstance(attestation, dict)
    frame = tmp_path / f"{'d' * 32}.frame"
    frame.write_bytes(payload)
    decoded = object()
    monkeypatch.setattr(
        worker,
        "_decode_image_bytes",
        lambda raw, *, width, height: (
            decoded
            if raw == payload and (width, height) == (2, 2)
            else pytest.fail("unexpected decode input")
        ),
    )
    observed: list[object] = []
    model = types.SimpleNamespace(
        predict=lambda value: observed.append(value) or []
    )

    response = worker._handle_request(
        request,
        model,
        attestation,
        frame_root=tmp_path,
    )

    assert response["ok"] is True
    assert observed == [decoded]
    assert not frame.exists()


def test_worker_rejects_private_frame_swap_during_stable_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    worker = _load_worker_script_module()
    payload = _png_bytes() + b"x" * (70 * 1024)
    request = _bound_frame_request(payload)
    attestation = request["attestation"]
    assert isinstance(attestation, dict)
    frame = tmp_path / f"{'d' * 32}.frame"
    frame.write_bytes(payload)
    original_inode = frame.stat().st_ino
    replacement = tmp_path / "replacement.frame"
    replacement.write_bytes(_png_bytes(width=3, height=3))
    real_read = worker.os.read
    swapped = False

    def swapping_read(descriptor: int, count: int) -> bytes:
        nonlocal swapped
        chunk = real_read(descriptor, count)
        if not swapped and os.fstat(descriptor).st_ino == original_inode:
            swapped = True
            os.replace(replacement, frame)
        return chunk

    monkeypatch.setattr(worker.os, "read", swapping_read)

    with pytest.raises(worker.ProtocolError, match="frame"):
        worker._handle_request(
            request,
            types.SimpleNamespace(predict=lambda _value: []),
            attestation,
            frame_root=tmp_path,
        )

    assert swapped


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo", "directory"])
def test_worker_rejects_non_private_frame_nodes(tmp_path: Path, kind: str) -> None:
    worker = _load_worker_script_module()
    payload = _png_bytes()
    request = _bound_frame_request(payload)
    attestation = request["attestation"]
    assert isinstance(attestation, dict)
    frame = tmp_path / f"{'d' * 32}.frame"
    if kind in {"symlink", "hardlink"}:
        target = tmp_path / "target.frame"
        target.write_bytes(payload)
        if kind == "symlink":
            frame.symlink_to(target)
        else:
            os.link(target, frame)
    elif kind == "fifo":
        os.mkfifo(frame)
    else:
        frame.mkdir()

    with pytest.raises(worker.ProtocolError, match="frame"):
        worker._handle_request(
            request,
            types.SimpleNamespace(predict=lambda _value: []),
            attestation,
            frame_root=tmp_path,
        )


def test_worker_creates_verified_private_model_copies_and_cleans_them(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    worker = _load_worker_script_module()
    _lock, manifest, model_root, _script, _dependency_hash, model_hash = (
        _write_worker_startup_fixture(tmp_path)
    )
    monkeypatch.setattr(
        worker,
        "OCR_MODEL_ARTIFACT_IDENTITY",
        f"paddleocr-models-v1:sha256:{model_hash}",
    )

    bundle = worker._create_private_model_bundle(
        model_root=model_root,
        model_manifest_path=manifest,
    )
    private_root = bundle.root
    try:
        assert private_root != model_root
        assert stat.S_IMODE(private_root.stat().st_mode) == 0o700
        for role in ("detection", "recognition"):
            private_artifact = bundle.directories[role] / "model.bin"
            source_artifact = model_root / role / "model.bin"
            assert private_artifact != source_artifact
            assert private_artifact.read_bytes() == source_artifact.read_bytes()
            assert stat.S_IMODE(private_artifact.stat().st_mode) == 0o400
        (model_root / "detection" / "model.bin").write_bytes(b"mutated-shared-cache")
        assert (
            bundle.directories["detection"] / "model.bin"
        ).read_bytes() == b"detection-model"
    finally:
        bundle.cleanup()

    assert not private_root.exists()


def test_worker_rejects_shared_model_swap_during_private_copy_and_cleans_partial_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    worker = _load_worker_script_module()
    _lock, manifest, model_root, _script, _dependency_hash, model_hash = (
        _write_worker_startup_fixture(tmp_path)
    )
    monkeypatch.setattr(
        worker,
        "OCR_MODEL_ARTIFACT_IDENTITY",
        f"paddleocr-models-v1:sha256:{model_hash}",
    )
    source = model_root / "detection" / "model.bin"
    original_inode = source.stat().st_ino
    replacement = tmp_path / "replacement-model.bin"
    replacement.write_bytes(source.read_bytes())
    private_root = tmp_path / "worker-private-models"

    def mkdtemp(*, prefix: str) -> str:
        assert prefix == "videoscope-ocr-models-"
        private_root.mkdir()
        return str(private_root)

    monkeypatch.setattr(worker.tempfile, "mkdtemp", mkdtemp)
    real_read = worker.os.read
    swapped = False

    def swapping_read(descriptor: int, count: int) -> bytes:
        nonlocal swapped
        chunk = real_read(descriptor, count)
        if not swapped and os.fstat(descriptor).st_ino == original_inode:
            swapped = True
            os.replace(replacement, source)
        return chunk

    monkeypatch.setattr(worker.os, "read", swapping_read)

    with pytest.raises(worker.ProtocolError, match="model artifact"):
        worker._create_private_model_bundle(
            model_root=model_root,
            model_manifest_path=manifest,
        )

    assert swapped
    assert not private_root.exists()


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo", "directory"])
def test_worker_rejects_non_regular_or_linked_shared_model_artifact_and_cleans(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    kind: str,
) -> None:
    worker = _load_worker_script_module()
    _lock, manifest, model_root, _script, _dependency_hash, model_hash = (
        _write_worker_startup_fixture(tmp_path)
    )
    monkeypatch.setattr(
        worker,
        "OCR_MODEL_ARTIFACT_IDENTITY",
        f"paddleocr-models-v1:sha256:{model_hash}",
    )
    source = model_root / "detection" / "model.bin"
    original = source.read_bytes()
    source.unlink()
    if kind in {"symlink", "hardlink"}:
        target = tmp_path / "shared-model-target.bin"
        target.write_bytes(original)
        if kind == "symlink":
            source.symlink_to(target)
        else:
            os.link(target, source)
    elif kind == "fifo":
        os.mkfifo(source)
    else:
        source.mkdir()
    private_root = tmp_path / f"private-models-{kind}"

    def mkdtemp(*, prefix: str) -> str:
        assert prefix == "videoscope-ocr-models-"
        private_root.mkdir()
        return str(private_root)

    monkeypatch.setattr(worker.tempfile, "mkdtemp", mkdtemp)

    with pytest.raises(worker.ProtocolError, match="model artifact"):
        worker._create_private_model_bundle(
            model_root=model_root,
            model_manifest_path=manifest,
        )

    assert not private_root.exists()


def test_worker_main_cleans_private_models_when_paddle_initialization_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    worker = _load_worker_script_module()
    attestation = {
        "dependency_identity": "paddleocr-deps-v1:sha256:" + "a" * 64,
        "model_identity": "paddleocr-models-v1:sha256:" + "b" * 64,
        "protocol": worker.OCR_WORKER_PROTOCOL,
        "runtime_identity": "runtime",
        "script_sha256": "c" * 64,
    }
    output = _BinaryConsole()
    monkeypatch.setattr(worker.sys, "stdin", _BinaryConsole())
    monkeypatch.setattr(worker.sys, "stdout", output)
    monkeypatch.setattr(worker.sys, "argv", ["worker.py"])
    monkeypatch.setattr(worker, "_environment_path", lambda name: Path(name))
    monkeypatch.setattr(
        worker,
        "_startup_attestation",
        lambda **_kwargs: (
            attestation,
            {"detection": tmp_path / "det", "recognition": tmp_path / "rec"},
        ),
    )
    cleaned = False

    class Bundle:
        directories = {
            "detection": tmp_path / "private-det",
            "recognition": tmp_path / "private-rec",
        }

        def cleanup(self) -> None:
            nonlocal cleaned
            cleaned = True

    monkeypatch.setattr(
        worker,
        "_create_private_model_bundle",
        lambda **_kwargs: Bundle(),
    )

    def fail_initialization(**_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("private model failure details")

    fake_module = types.ModuleType("paddleocr")
    fake_module.PaddleOCR = fail_initialization  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "paddleocr", fake_module)

    assert worker.main() == 2
    assert cleaned
    assert json.loads(output.buffer.getvalue()) == {
        "error_code": "startup_failed",
        "ok": False,
        "type": "error",
    }


def test_worker_paddle_initialization_reads_only_verified_private_model_copies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    worker = _load_worker_script_module()
    _lock, manifest, model_root, script, _dependency_hash, model_hash = (
        _write_worker_startup_fixture(tmp_path)
    )
    monkeypatch.setattr(
        worker,
        "OCR_MODEL_ARTIFACT_IDENTITY",
        f"paddleocr-models-v1:sha256:{model_hash}",
    )
    attestation = {
        "dependency_identity": "paddleocr-deps-v1:sha256:" + "a" * 64,
        "model_identity": f"paddleocr-models-v1:sha256:{model_hash}",
        "protocol": worker.OCR_WORKER_PROTOCOL,
        "runtime_identity": "runtime",
        "script_sha256": "c" * 64,
    }
    output = _BinaryConsole()
    monkeypatch.setattr(worker.sys, "stdin", _BinaryConsole())
    monkeypatch.setattr(worker.sys, "stdout", output)
    monkeypatch.setattr(worker.sys, "argv", ["worker.py"])
    paths = {
        "VIDEOSCOPE_OCR_DEPENDENCY_LOCK": tmp_path / "unused.lock",
        "VIDEOSCOPE_OCR_FRAME_ROOT": tmp_path,
        "VIDEOSCOPE_OCR_MODEL_MANIFEST": manifest,
        "VIDEOSCOPE_OCR_MODEL_ROOT": model_root,
    }
    monkeypatch.setattr(worker, "_environment_path", lambda name: paths[name])
    monkeypatch.setattr(
        worker,
        "_startup_attestation",
        lambda **_kwargs: (
            attestation,
            {
                "detection": model_root / "detection",
                "recognition": model_root / "recognition",
            },
        ),
    )
    private_directories: list[Path] = []

    class FakePaddleOCR:
        def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
            detection = Path(kwargs["text_detection_model_dir"])
            recognition = Path(kwargs["text_recognition_model_dir"])
            private_directories.extend((detection, recognition))
            assert detection != model_root / "detection"
            assert recognition != model_root / "recognition"
            (model_root / "detection" / "model.bin").write_bytes(
                b"mutate-load-restore-attempt"
            )
            assert (detection / "model.bin").read_bytes() == b"detection-model"
            assert (recognition / "model.bin").read_bytes() == b"recognition-model"

        def predict(self, _decoded: object) -> list[object]:
            return []

    fake_module = types.ModuleType("paddleocr")
    fake_module.PaddleOCR = FakePaddleOCR  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "paddleocr", fake_module)
    monkeypatch.setattr(worker, "__file__", str(script))

    assert worker.main() == 0
    assert private_directories
    assert all(not path.exists() for path in private_directories)
    assert json.loads(output.buffer.getvalue()) == {
        "attestation": attestation,
        "ok": True,
        "type": "hello",
    }
