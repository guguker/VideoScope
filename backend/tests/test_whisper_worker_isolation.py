from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_importing_whisper_contract_does_not_import_heavy_runtime() -> None:
    script = """
import sys
import videoscope.providers.whisper_worker
for name in ('mlx', 'mlx_whisper', 'torch', 'numba'):
    assert name not in sys.modules, name
"""

    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=ROOT / "backend",
        env={"PYTHONPATH": str(ROOT / "backend" / "src")},
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_whisper_worker_project_has_hashed_immutable_install_lock() -> None:
    project = (ROOT / "workers" / "whisper" / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    lock = (ROOT / "workers" / "whisper" / "requirements.lock").read_text(
        encoding="utf-8"
    )

    assert 'requires-python = ">=3.12,<3.13"' in project
    for requirement in (
        "fastapi==0.141.1",
        "huggingface-hub==1.27.0",
        "mlx==0.32.0",
        "mlx-whisper==0.4.3",
        "pydantic==2.13.4",
        "pydantic-settings==2.15.0",
        "uvicorn==0.52.1",
    ):
        assert f'"{requirement}"' in project
        assert requirement in lock
    assert "--hash=sha256:" in lock
    assert "exclude-newer" in project

    from videoscope.providers.whisper_worker import (
        WHISPER_EXACT_DEPENDENCIES,
        WHISPER_WORKER_LOCK_SHA256,
    )

    assert sha256(lock.encode("utf-8")).hexdigest() == WHISPER_WORKER_LOCK_SHA256
    locked_versions = dict(
        re.findall(r"^([A-Za-z0-9_.-]+)==([^ \\\n]+)", lock, flags=re.MULTILINE)
    )
    assert locked_versions == dict(WHISPER_EXACT_DEPENDENCIES)
