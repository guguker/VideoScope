from __future__ import annotations

import ast
from hashlib import sha256
from pathlib import Path

from videoscope.providers.lighthouse_worker import (
    LIGHTHOUSE_WORKER_CORE_DISTRIBUTIONS,
    LIGHTHOUSE_WORKER_LOCK_SHA256,
    LIGHTHOUSE_WORKER_RUNTIME_IDENTITY,
)


ROOT = Path(__file__).resolve().parents[2]


def test_lighthouse_profile_uses_dedicated_locked_environment() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    install = makefile.split("install-lighthouse:", 1)[1].split("\n\n", 1)[0]
    project = (ROOT / "workers/lighthouse/pyproject.toml").read_text(encoding="utf-8")
    overrides = (ROOT / "workers/lighthouse/overrides.txt").read_text(encoding="utf-8")
    lock_path = ROOT / "workers/lighthouse/requirements.lock"
    lock = lock_path.read_text(encoding="utf-8")
    backend_project = (ROOT / "backend/pyproject.toml").read_text(encoding="utf-8")

    assert "uv venv --python 3.11.14 --managed-python" in install
    assert "uv pip sync --python .venv-lighthouse/bin/python" in install
    assert "workers/lighthouse/requirements.lock" in install
    assert "--require-hashes" in install
    assert "--project backend" not in install
    assert "--inexact" not in install
    assert 'PYTHONPATH="$(CURDIR)/backend/src"' in makefile
    assert ".venv-lighthouse/bin/python -m videoscope.providers.lighthouse_worker" in makefile
    assert ".venv-lighthouse/" in (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.11,<3.12"' in project
    assert (
        "d095eaa552cecef240897a8b750306b3b2a08740.tar.gz#sha256="
        "0941b57be663ae603b2c2443bf05de4938d5e35f183770190d803710bcf0db3f"
        in project
    )
    assert (
        "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6.tar.gz#sha256="
        "553c80a6364ac7350dd7800f535531175f22c11fa742bda1b50b62121eb5f40f"
        in project
    )
    assert overrides.strip() in project
    assert "d095eaa552cecef240897a8b750306b3b2a08740.tar.gz" in lock
    assert "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6.tar.gz" in lock
    assert "numpy==1.23.5" in lock
    assert "transformers==4.51.3" in lock
    for distribution, version in LIGHTHOUSE_WORKER_CORE_DISTRIBUTIONS.items():
        assert f"{distribution}=={version}" in lock
    assert "line/lighthouse" not in backend_project
    assert "openai/CLIP" not in backend_project
    assert LIGHTHOUSE_WORKER_LOCK_SHA256 == sha256(lock_path.read_bytes()).hexdigest()
    assert f"lock-sha256:{LIGHTHOUSE_WORKER_LOCK_SHA256}" in LIGHTHOUSE_WORKER_RUNTIME_IDENTITY


def test_backend_runtime_does_not_import_lighthouse_clip_or_torch() -> None:
    runtime_source = (ROOT / "backend/src/videoscope/runtime.py").read_text(encoding="utf-8")
    worker_source = (ROOT / "backend/src/videoscope/providers/lighthouse_worker.py").read_text(
        encoding="utf-8"
    )
    runtime_tree = ast.parse(runtime_source)
    worker_tree = ast.parse(worker_source)
    forbidden = {"clip", "lighthouse", "torch", "torchvision"}

    def imported_roots(tree: ast.AST) -> set[str]:
        roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".", 1)[0])
        return roots

    assert imported_roots(runtime_tree).isdisjoint(forbidden)
    # Contract import is safe; heavy imports are allowed only inside worker runtime methods.
    top_level = ast.Module(
        body=[
            node
            for node in worker_tree.body
            if not isinstance(node, (ast.FunctionDef, ast.ClassDef))
        ],
        type_ignores=[],
    )
    assert imported_roots(top_level).isdisjoint(forbidden)
