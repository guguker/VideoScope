#!/usr/bin/env python3
"""Run the exact persisted-generation rollback drill and emit a sealed receipt."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_BACKEND_SOURCE_ROOT = os.fspath(_PROJECT_ROOT / "backend" / "src")
sys.path[:] = [
    _BACKEND_SOURCE_ROOT,
    *(entry for entry in sys.path if entry != _BACKEND_SOURCE_ROOT),
]

ROLLBACK_TEST_NODE_ID = (
    "backend/tests/test_video_index_jobs_repository.py::"
    "test_failed_full_job_publish_keeps_prior_release_searchable_after_restart"
)
ROLLBACK_TEST_SOURCE = Path("backend/tests/test_video_index_jobs_repository.py")
ROLLBACK_PROOF_SCHEMA_VERSION = 1
ROLLBACK_PROOF_ID = "phase0-persisted-generation-rollback-v1"
ROLLBACK_FORCED_FAILURE = "injected_pre_commit_publication_failure"
ROLLBACK_SEMANTIC_ASSERTIONS = (
    "candidate_release_not_activated_after_forced_failure",
    "prior_release_active_after_process_restart",
    "prior_release_searchable_after_process_restart",
    "candidate_generation_addressable_but_inactive",
    "source_asset_identity_unchanged",
    "source_reindex_not_required",
)
_CODE_SHA_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


class RollbackProofError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _source_bytes(project_root: Path) -> bytes:
    path = project_root / ROLLBACK_TEST_SOURCE
    try:
        metadata = path.lstat()
        if path.is_symlink() or not path.is_file() or metadata.st_size > 4 * 1024**2:
            raise RollbackProofError("test_source_invalid")
        raw = path.read_bytes()
    except RollbackProofError:
        raise
    except OSError as error:
        raise RollbackProofError("test_source_unavailable") from error
    if not raw or len(raw) != metadata.st_size:
        raise RollbackProofError("test_source_invalid")
    return raw


def _default_test_runner(project_root: Path, node_id: str) -> int:
    python = project_root / ".venv" / "bin" / "python"
    environment = {
        "LC_ALL": "C",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }
    try:
        completed = subprocess.run(
            [os.fspath(python), "-I", "-m", "pytest", node_id, "-q"],
            cwd=project_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=180.0,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RollbackProofError("rollback_test_unavailable") from error
    return completed.returncode


def execute(
    project_root: Path,
    *,
    code_identity_resolver: Callable[[], str],
    test_runner: Callable[[Path, str], int] = _default_test_runner,
) -> dict[str, object]:
    """Execute one exact test under one unchanged clean Git identity."""

    root = Path(project_root)
    try:
        code_before = code_identity_resolver()
    except Exception as error:
        raise RollbackProofError("code_identity_unavailable") from error
    if not isinstance(code_before, str) or _CODE_SHA_RE.fullmatch(code_before) is None:
        raise RollbackProofError("code_identity_invalid")
    source_before = _source_bytes(root)
    source_identity = sha256(source_before).hexdigest()
    try:
        return_code = test_runner(root, ROLLBACK_TEST_NODE_ID)
    except RollbackProofError:
        raise
    except Exception as error:
        raise RollbackProofError("rollback_test_unavailable") from error
    if return_code != 0:
        raise RollbackProofError("rollback_test_failed")
    source_after = _source_bytes(root)
    if source_after != source_before:
        raise RollbackProofError("test_source_changed")
    try:
        code_after = code_identity_resolver()
    except Exception as error:
        raise RollbackProofError("code_identity_unavailable") from error
    if not isinstance(code_after, str) or _CODE_SHA_RE.fullmatch(code_after) is None:
        raise RollbackProofError("code_identity_invalid")
    if code_before != code_after:
        raise RollbackProofError("code_identity_changed")
    return {
        "code_sha": code_before,
        "forced_failure": ROLLBACK_FORCED_FAILURE,
        "proof_id": ROLLBACK_PROOF_ID,
        "schema_version": ROLLBACK_PROOF_SCHEMA_VERSION,
        "semantic_assertions": list(ROLLBACK_SEMANTIC_ASSERTIONS),
        "status": "verified",
        "test_node_id": ROLLBACK_TEST_NODE_ID,
        "test_result": "passed",
        "test_source_sha256": source_identity,
    }


def _project_root() -> Path:
    return _PROJECT_ROOT


def _write_json(stream: object, value: object) -> None:
    stream.write(  # type: ignore[attr-defined]
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    if argv:
        _write_json(
            sys.stderr,
            {"error": "usage_error", "message": "rollback proof takes no arguments"},
        )
        return 2
    try:
        from videoscope.benchmark.cli import _current_code_sha

        receipt = execute(
            _project_root(),
            code_identity_resolver=_current_code_sha,
        )
    except RollbackProofError:
        _write_json(
            sys.stderr,
            {"error": "proof_failed", "message": "rollback proof failed"},
        )
        return 3
    except KeyboardInterrupt:
        _write_json(
            sys.stderr,
            {"error": "interrupted", "message": "rollback proof was interrupted"},
        )
        return 130
    except Exception:
        _write_json(
            sys.stderr,
            {"error": "internal_error", "message": "rollback proof failed"},
        )
        return 70
    _write_json(sys.stdout, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
