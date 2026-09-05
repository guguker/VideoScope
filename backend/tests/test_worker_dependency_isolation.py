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
            "--only-binary=:all: --no-python-downloads "
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
            f".venv-{worker}/bin/python -I scripts/check-worker-platform.py"
        )
        sync = target.index(
            f"uv pip sync --python .venv-{worker}/bin/python --require-hashes"
        )
        assert create < check < sync
        assert f'[ -e ".venv-{worker}" ] || [ -L ".venv-{worker}" ]' in target
        assert "--only-binary=:all:" in target
        assert "--no-python-downloads" in target
        assert f"uv pip check --python .venv-{worker}/bin/python" in target


def test_base_bootstrap_creates_an_exact_managed_runtime_without_self_bootstrap() -> None:
    bootstrap = (ROOT / "scripts" / "bootstrap.sh").read_text(encoding="utf-8")

    assert 'UV_VERSION="0.12.3"' in bootstrap
    assert 'require_command uv "uv $UV_VERSION"' in bootstrap
    assert 'uv venv --python 3.12.13 --managed-python .venv' in bootstrap
    assert '.venv/bin/python -I scripts/check-worker-platform.py' in bootstrap
    assert "--python .venv/bin/python" in bootstrap
    assert "--no-python-downloads" in bootstrap
    assert 'python3.12' not in bootstrap
    assert "-m pip install" not in bootstrap


def test_backend_only_bootstrap_keeps_phase_zero_independent_of_frontend_install() -> None:
    bootstrap = (ROOT / "scripts" / "bootstrap.sh").read_text(encoding="utf-8")
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "install-backend:" in makefile
    assert "./scripts/bootstrap.sh --backend-only" in makefile
    assert "BACKEND_ONLY=0" in bootstrap
    assert "--backend-only) BACKEND_ONLY=1" in bootstrap
    assert 'if [[ "$BACKEND_ONLY" -eq 0 ]]; then\n  require_command pnpm "pnpm"\nfi' in bootstrap
    assert 'if [[ "$BACKEND_ONLY" -eq 0 ]]; then\n  pnpm --dir frontend install --frozen-lockfile\nfi' in bootstrap
    assert 'Usage: $0 [--backend-only]' in bootstrap


def test_phase_zero_evidence_commands_use_isolated_python_entrypoints() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    phase_zero = (
        ROOT / "docs" / "benchmarks" / "phase0" / "README.md"
    ).read_text(encoding="utf-8")
    product = (
        ROOT / "docs" / "benchmarks" / "product-retrieval" / "README.md"
    ).read_text(encoding="utf-8")
    verifier = (
        ROOT / "docs" / "benchmarks" / "video-verifier" / "README.md"
    ).read_text(encoding="utf-8")

    assert (
        '"$(CURDIR)/.venv/bin/python" -I '
        '"$(CURDIR)/scripts/full-ml-smoke-diagnostic.py"'
    ) in makefile
    assert (
        '"$(CURDIR)/.venv/bin/python" -I '
        '"$(CURDIR)/scripts/full-ml-smoke.py"'
    ) in makefile
    assert ".venv/bin/python -I scripts/phase0-rollback-proof.py" in phase_zero
    assert ".venv/bin/python -I scripts/phase0-evidence.py" in phase_zero
    assert "../.venv/bin/python -I -m videoscope.benchmark.regression_fixture" in product
    assert ".venv/bin/python -I \\\n  -m videoscope.benchmark.video_verifier_runner" in verifier


def test_phase_zero_entrypoints_ignore_hostile_pythonpath(tmp_path: Path) -> None:
    hostile = tmp_path / "hostile"
    package = hostile / "videoscope"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        'raise RuntimeError("hostile PYTHONPATH imported")\n',
        encoding="utf-8",
    )
    python = ROOT / ".venv" / "bin" / "python"
    commands = (
        [python, "-I", ROOT / "scripts" / "full-ml-smoke.py", "--help"],
        [python, "-I", ROOT / "scripts" / "full-ml-smoke-diagnostic.py", "--help"],
        [python, "-I", ROOT / "scripts" / "phase0-evidence.py", "--help"],
        [
            python,
            "-I",
            "-m",
            "videoscope.benchmark.regression_fixture",
            "--help",
        ],
        [
            python,
            "-I",
            "-m",
            "videoscope.benchmark.video_verifier_runner",
            "--help",
        ],
    )

    for command in commands:
        result = subprocess.run(
            [str(item) for item in command],
            cwd=tmp_path,
            env={"LC_ALL": "C", "PYTHONPATH": str(hostile)},
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "hostile PYTHONPATH imported" not in result.stderr


def test_base_bootstrap_rejects_external_venv_symlink_before_uv_sync(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    scripts = checkout / "scripts"
    scripts.mkdir(parents=True)
    bootstrap = scripts / "bootstrap.sh"
    bootstrap.write_bytes((ROOT / "scripts" / "bootstrap.sh").read_bytes())
    external = tmp_path / "external-venv"
    (external / "bin").mkdir(parents=True)
    python = external / "bin" / "python"
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.chmod(0o700)
    (checkout / ".venv").symlink_to(external, target_is_directory=True)

    result = subprocess.run(
        ["/bin/bash", str(bootstrap), "--backend-only"],
        cwd=checkout,
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "not a symbolic link" in result.stderr
    assert "uv sync" not in result.stderr


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
    assert checker.worker_platform_is_supported(
        python_version=(3, 11, 14),
        expected_python_version=(3, 11, 14),
        system="Darwin",
        machine="arm64",
        macos_version="14.0",
    )
    assert not checker.worker_platform_is_supported(
        python_version=(3, 11, 15),
        expected_python_version=(3, 11, 14),
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
    assert checker.worker_host_is_supported(
        system="Darwin",
        machine="arm64",
        macos_version="14.0",
    )
    assert not checker.worker_host_is_supported(
        system="Linux",
        machine="aarch64",
        macos_version="",
    )
