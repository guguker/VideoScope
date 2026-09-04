from hashlib import sha256
import json
from pathlib import Path

from videoscope.providers.qwen_worker import QWEN_INFERENCE_RUNTIME_IDENTITY


ROOT = Path(__file__).resolve().parents[2]


def test_video_profile_uses_dedicated_reproducible_environment() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    install_video = makefile.split("install-video:", 1)[1].split("\n\n", 1)[0]

    assert "install-video: worker-platform-check" in makefile
    assert 'UV_PROJECT_ENVIRONMENT="$(CURDIR)/.venv-qwen"' in install_video
    assert "--project backend --locked --extra video" in install_video
    assert "uv venv --python 3.12.13 --managed-python .venv-qwen" in install_video
    assert ".venv-qwen/bin/python -I scripts/check-worker-platform.py" in install_video
    assert "--python .venv-qwen/bin/python" in install_video
    assert "--no-python-downloads" in install_video
    assert "uv pip check --python .venv-qwen/bin/python" in install_video
    assert '[ -e ".venv-qwen" ] || [ -L ".venv-qwen" ]' in install_video
    assert "--inexact" not in install_video
    assert ".venv-qwen/bin/python scripts/download-models.py --profile video" in makefile
    assert ".venv-qwen/bin/python -m videoscope.providers.qwen_worker" in makefile
    assert ".venv-qwen/" in (ROOT / ".gitignore").read_text(encoding="utf-8")


def test_qwen_dependency_remains_pinned_in_project_and_lock() -> None:
    project = (ROOT / "backend" / "pyproject.toml").read_text(encoding="utf-8")
    lock = (ROOT / "backend" / "uv.lock").read_text(encoding="utf-8")

    assert '"mlx-vlm==0.6.7"' in project
    runtime_manifest = json.loads(
        (ROOT / "workers/qwen/runtime.lock.json").read_text(encoding="utf-8")
    )
    model_manifest = json.loads(
        (ROOT / "workers/qwen/model-artifacts.lock.json").read_text(encoding="utf-8")
    )
    runtime_hash = sha256(
        json.dumps(runtime_manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    model_hash = sha256(
        json.dumps(model_manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert QWEN_INFERENCE_RUNTIME_IDENTITY == (
        "videoscope-qwen-worker-v4|python==3.12.13|"
        "platform==aarch64-apple-darwin-macos14plus|"
        f"runtime-manifest-sha256:{runtime_hash}|model-manifest-sha256:{model_hash}"
    )
    assert runtime_manifest["backend_lock_sha256"] == sha256(
        (ROOT / "backend/uv.lock").read_bytes()
    ).hexdigest()
    assert runtime_manifest["distributions"]["mlx-vlm"] == "0.6.7"
    assert model_manifest["model"] == "mlx-community/Qwen3.5-9B-MLX-4bit"
    assert model_manifest["revision"] == "938d8919941c6e7efd3c7150eff7fe9d12afa631"
    assert len(model_manifest["artifacts"]) == 13
    assert 'name = "mlx-vlm"' in lock
    assert 'version = "0.6.7"' in lock
