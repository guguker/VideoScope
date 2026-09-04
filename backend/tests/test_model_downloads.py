from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import stat
import subprocess
import sys
from types import ModuleType

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_downloader() -> ModuleType:
    path = PROJECT_ROOT / "scripts" / "download-models.py"
    specification = importlib.util.spec_from_file_location(
        "videoscope_test_download_models",
        path,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _load_ocr_downloader() -> ModuleType:
    path = PROJECT_ROOT / "scripts" / "download-ocr-models.py"
    specification = importlib.util.spec_from_file_location(
        "videoscope_test_download_ocr_models",
        path,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _ocr_contracts(tmp_path: Path) -> tuple[Path, Path, dict[str, bytes]]:
    files = {
        "config.json": b"config",
        "model.safetensors": b"safe-weights",
        "README.md": b"---\nlicense: apache-2.0\n---\n",
    }
    artifact_manifest = {
        "engine": "transformers",
        "models": [
            {
                "artifacts": [
                    {
                        "name": name,
                        "sha256": sha256(body).hexdigest(),
                        "size": len(body),
                    }
                    for name, body in files.items()
                    if name != "README.md"
                ],
                "directory": "detector",
                "role": "detection",
            }
        ],
        "profile": "fixture",
        "schema_version": 1,
    }
    source_manifest = {
        "provider": "huggingface_hub",
        "schema_version": 1,
        "models": [
            {
                "directory": "detector",
                "license": "apache-2.0",
                "license_evidence": {
                    "name": "README.md",
                    "sha256": sha256(files["README.md"]).hexdigest(),
                    "size": len(files["README.md"]),
                },
                "repository": "org/detector",
                "revision": "a" * 40,
                "role": "detection",
            }
        ],
    }
    artifact_path = tmp_path / "artifact.json"
    source_path = tmp_path / "source.json"
    artifact_path.write_text(json.dumps(artifact_manifest), encoding="utf-8")
    source_path.write_text(json.dumps(source_manifest), encoding="utf-8")
    return source_path, artifact_path, files


def test_rfdetr_download_replaces_checkpoint_only_after_hash_validation(
    tmp_path: Path,
) -> None:
    script = _load_downloader()
    checkpoint = tmp_path / "models" / "rf-detr-small.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"previous-verified-generation")
    expected = b"new-verified-generation"
    downloaded_paths: list[Path] = []

    def download(path: str) -> None:
        destination = Path(path)
        downloaded_paths.append(destination)
        destination.write_bytes(expected)

    script.install_rfdetr_checkpoint(
        checkpoint,
        downloader=download,
        expected_sha256=sha256(expected).hexdigest(),
    )

    assert checkpoint.read_bytes() == expected
    assert downloaded_paths[0].name == checkpoint.name
    assert downloaded_paths[0].parent != checkpoint.parent
    assert stat.S_IMODE(checkpoint.stat().st_mode) & 0o077 == 0
    assert list(checkpoint.parent.iterdir()) == [checkpoint]


def test_rfdetr_download_failure_preserves_previous_checkpoint(
    tmp_path: Path,
) -> None:
    script = _load_downloader()
    checkpoint = tmp_path / "models" / "rf-detr-small.pth"
    checkpoint.parent.mkdir(parents=True)
    previous = b"previous-verified-generation"
    checkpoint.write_bytes(previous)

    def download(path: str) -> None:
        Path(path).write_bytes(b"wrong-download")

    with pytest.raises(SystemExit, match="pinned SHA-256"):
        script.install_rfdetr_checkpoint(
            checkpoint,
            downloader=download,
            expected_sha256=sha256(b"expected-download").hexdigest(),
        )

    assert checkpoint.read_bytes() == previous
    assert list(checkpoint.parent.iterdir()) == [checkpoint]


def test_existing_verified_rfdetr_checkpoint_skips_network_download(
    tmp_path: Path,
) -> None:
    script = _load_downloader()
    checkpoint = tmp_path / "models" / "rf-detr-small.pth"
    checkpoint.parent.mkdir(parents=True)
    body = b"already-verified"
    checkpoint.write_bytes(body)

    script.install_rfdetr_checkpoint(
        checkpoint,
        downloader=lambda _path: pytest.fail("verified checkpoint was downloaded again"),
        expected_sha256=sha256(body).hexdigest(),
    )

    assert checkpoint.read_bytes() == body


def test_ocr_download_is_revision_pinned_and_published_after_verification(
    tmp_path: Path,
) -> None:
    script = _load_ocr_downloader()
    source, artifacts, files = _ocr_contracts(tmp_path)
    destination = tmp_path / "models"
    calls: list[tuple[str, str, tuple[str, ...]]] = []

    def download(repository, revision, includes, target):  # type: ignore[no-untyped-def]
        calls.append((repository, revision, tuple(includes)))
        target.mkdir(parents=True)
        for name in includes:
            (target / name).write_bytes(files[name])

    installed = script.install_ocr_models(
        destination,
        source_manifest_path=source,
        artifact_manifest_path=artifacts,
        downloader=download,
    )

    assert installed == ("detector",)
    assert calls == [
        (
            "org/detector",
            "a" * 40,
            ("README.md", "config.json", "model.safetensors"),
        )
    ]
    assert (destination / "detector" / "model.safetensors").read_bytes() == (
        b"safe-weights"
    )
    assert not list(destination.glob(".ocr-download-*"))


def test_ocr_hub_download_uses_current_python_and_ignores_ambient_hf_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _load_ocr_downloader()
    hostile_bin = tmp_path / "hostile-bin"
    hostile_bin.mkdir()
    hostile_hf = hostile_bin / "hf"
    hostile_hf.write_text("#!/bin/sh\nexit 91\n", encoding="utf-8")
    hostile_hf.chmod(0o755)
    hostile_environment = {
        "PATH": str(hostile_bin),
        "HF_ENDPOINT": "https://attacker.invalid",
        "HF_TOKEN": "ambient-secret",
        "HUGGING_FACE_HUB_TOKEN": "ambient-secret-two",
        "HTTP_PROXY": "http://attacker.invalid:8080",
        "HTTPS_PROXY": "http://attacker.invalid:8080",
        "ALL_PROXY": "socks5://attacker.invalid:1080",
        "NO_PROXY": "",
        "http_proxy": "http://attacker.invalid:8080",
        "https_proxy": "http://attacker.invalid:8080",
        "all_proxy": "socks5://attacker.invalid:1080",
        "no_proxy": "",
    }
    for name, value in hostile_environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        script.shutil,
        "which",
        lambda _name: pytest.fail("ambient Hugging Face CLI must not be resolved"),
    )
    observed: dict[str, object] = {}

    def run(command, **kwargs):  # type: ignore[no-untyped-def]
        observed["command"] = command
        observed.update(kwargs)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(script.subprocess, "run", run)
    destination = tmp_path / "download"

    script._hf_download(  # noqa: SLF001 - acquisition boundary contract
        "org/detector",
        "a" * 40,
        ("README.md", "config.json", "model.safetensors"),
        destination,
    )

    command = observed["command"]
    assert isinstance(command, list)
    assert command[:3] == [sys.executable, "-I", "-c"]
    assert str(hostile_hf) not in command
    assert command[4:] == [
        "org/detector",
        "a" * 40,
        str(destination),
        "README.md",
        "config.json",
        "model.safetensors",
    ]
    helper = command[3]
    assert "hf_hub_download" in helper
    assert "token=False" in helper
    assert 'endpoint="https://huggingface.co"' in helper
    assert observed["check"] is False
    assert observed["stdin"] is subprocess.DEVNULL
    assert observed["stdout"] is subprocess.DEVNULL
    assert observed["stderr"] is subprocess.DEVNULL
    child_environment = observed["env"]
    assert child_environment == {
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_XET": "1",
        "NO_PROXY": "*",
        "no_proxy": "*",
    }
    for name in (
        "PATH",
        "HF_ENDPOINT",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        assert name not in child_environment


def test_existing_verified_ocr_models_are_checked_without_mutating_root_permissions(
    tmp_path: Path,
) -> None:
    script = _load_ocr_downloader()
    source, artifacts, files = _ocr_contracts(tmp_path)
    destination = tmp_path / "models"
    existing = destination / "detector"
    existing.mkdir(parents=True)
    for name, body in files.items():
        (existing / name).write_bytes(body)
    destination.chmod(0o755)

    installed = script.install_ocr_models(
        destination,
        source_manifest_path=source,
        artifact_manifest_path=artifacts,
        downloader=lambda *_args: pytest.fail("verified model was downloaded again"),
    )

    assert installed == ("detector",)
    assert stat.S_IMODE(destination.stat().st_mode) == 0o755


def test_ocr_download_failure_never_replaces_existing_or_publishes_partial(
    tmp_path: Path,
) -> None:
    script = _load_ocr_downloader()
    source, artifacts, files = _ocr_contracts(tmp_path)
    destination = tmp_path / "models"
    existing = destination / "detector"
    existing.mkdir(parents=True)
    (existing / "model.safetensors").write_bytes(b"user-state")

    with pytest.raises(SystemExit, match="refusing to replace"):
        script.install_ocr_models(
            destination,
            source_manifest_path=source,
            artifact_manifest_path=artifacts,
            downloader=lambda *_args: pytest.fail("must not download"),
        )

    assert (existing / "model.safetensors").read_bytes() == b"user-state"
    assert not list(destination.glob(".ocr-download-*"))

    clean_destination = tmp_path / "clean-models"

    def corrupt_download(_repository, _revision, includes, target):  # type: ignore[no-untyped-def]
        target.mkdir(parents=True)
        for name in includes:
            (target / name).write_bytes(
                b"evil-weights" if name == "model.safetensors" else files[name]
            )

    with pytest.raises(SystemExit, match="SHA-256"):
        script.install_ocr_models(
            clean_destination,
            source_manifest_path=source,
            artifact_manifest_path=artifacts,
            downloader=corrupt_download,
        )

    assert not (clean_destination / "detector").exists()
    assert not list(clean_destination.glob(".ocr-download-*"))
