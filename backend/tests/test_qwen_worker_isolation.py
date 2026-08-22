from pathlib import Path

from videoscope.providers.qwen_worker import QWEN_INFERENCE_RUNTIME_IDENTITY


ROOT = Path(__file__).resolve().parents[2]


def test_video_profile_uses_dedicated_reproducible_environment() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    install_video = makefile.split("install-video:", 1)[1].split("\n\n", 1)[0]

    assert "install-video: worker-platform-check" in makefile
    assert 'UV_PROJECT_ENVIRONMENT="$(CURDIR)/.venv-qwen"' in install_video
    assert "--project backend --locked --extra video" in install_video
    assert "--python .venv/bin/python" in install_video
    assert "--no-python-downloads" in install_video
    assert "--inexact" not in install_video
    assert ".venv-qwen/bin/python scripts/download-models.py --profile video" in makefile
    assert ".venv-qwen/bin/python -m videoscope.providers.qwen_worker" in makefile
    assert ".venv-qwen/" in (ROOT / ".gitignore").read_text(encoding="utf-8")


def test_qwen_dependency_remains_pinned_in_project_and_lock() -> None:
    project = (ROOT / "backend" / "pyproject.toml").read_text(encoding="utf-8")
    lock = (ROOT / "backend" / "uv.lock").read_text(encoding="utf-8")

    assert '"mlx-vlm==0.6.7"' in project
    assert QWEN_INFERENCE_RUNTIME_IDENTITY == "mlx-vlm==0.6.7"
    assert 'name = "mlx-vlm"' in lock
    assert 'version = "0.6.7"' in lock
