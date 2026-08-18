from __future__ import annotations

from hashlib import sha256
import importlib.util
from pathlib import Path
import stat
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
