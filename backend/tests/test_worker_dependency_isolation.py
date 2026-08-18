from __future__ import annotations

from pathlib import Path
import importlib.util
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_default_runtime_worker_factories_do_not_import_ml_frameworks() -> None:
    script = """
import sys
from pathlib import Path
from videoscope.config import AppSettings
from videoscope.runtime import (
    create_object_detector,
    create_visual_index,
    create_whisper_transcriber,
)

settings = AppSettings(_env_file=None, data_dir=Path('isolated-data'))
visual = create_visual_index(settings)
objects = create_object_detector(settings)
speech = create_whisper_transcriber(settings)
visual.status(check_index=False)
assert objects is None
assert speech is None
for name in ('mlx', 'mlx_whisper', 'rfdetr', 'supervision', 'torch', 'torchvision', 'transformers'):
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


def test_backend_project_keeps_test_torch_but_not_worker_extras() -> None:
    project = (ROOT / "backend" / "pyproject.toml").read_text(encoding="utf-8")

    assert "apple = [" not in project
    assert "roboflow = [" not in project
    assert "vision = [" not in project
    dev_section = project.split("dev = [", 1)[1].split("]", 1)[0]
    assert '"torch>=2.6,<3"' in dev_section


def test_worker_install_and_model_commands_use_their_dedicated_environments() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    downloader = (ROOT / "scripts/download-models.py").read_text(encoding="utf-8")

    for worker in ("vision", "whisper"):
        assert (
            f"uv pip sync --python .venv-{worker}/bin/python --require-hashes "
            f"workers/{worker}/requirements.lock"
        ) in makefile
        assert (
            f'.venv-{worker}/bin/python scripts/download-models.py --profile {worker}'
            in makefile
        )
    assert '"vision": (' in downloader
    assert '"whisper": (' in downloader
    assert "RFDETRSmall()" not in downloader


def test_worker_install_checks_the_managed_runtime_before_syncing_packages() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    for worker in ("vision", "whisper"):
        target = makefile.split(f"install-{worker}:\n", 1)[1].split("\n\n", 1)[0]
        create = target.index(
            f"uv venv --python 3.12.13 --managed-python .venv-{worker}"
        )
        check = target.index(
            f".venv-{worker}/bin/python scripts/check-worker-platform.py"
        )
        sync = target.index(
            f"uv pip sync --python .venv-{worker}/bin/python --require-hashes"
        )
        assert create < check < sync


def test_worker_lock_commands_are_guarded_by_the_exact_build_platform() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    checker_path = ROOT / "scripts" / "check-worker-platform.py"
    specification = importlib.util.spec_from_file_location(
        "videoscope_test_worker_platform",
        checker_path,
    )
    assert specification is not None and specification.loader is not None
    checker = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(checker)

    assert "lock-vision: worker-platform-check" in makefile
    assert "lock-whisper: worker-platform-check" in makefile
    assert checker.worker_platform_is_supported(
        python_version=(3, 12, 13),
        system="Darwin",
        machine="arm64",
        macos_version="14.0",
    )
    assert not checker.worker_platform_is_supported(
        python_version=(3, 12, 12),
        system="Darwin",
        machine="arm64",
        macos_version="14.0",
    )
    assert not checker.worker_platform_is_supported(
        python_version=(3, 12, 13),
        system="Linux",
        machine="aarch64",
        macos_version="",
    )
