from __future__ import annotations

from hashlib import sha256
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_script() -> ModuleType:
    path = PROJECT_ROOT / "scripts" / "phase0-rollback-proof.py"
    specification = importlib.util.spec_from_file_location(
        "videoscope_test_phase0_rollback_proof",
        path,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _fixture_root(tmp_path: Path, script: ModuleType) -> Path:
    root = tmp_path / "checkout"
    source = root / script.ROLLBACK_TEST_SOURCE
    source.parent.mkdir(parents=True)
    source.write_text("def test_fixture():\n    assert True\n", encoding="utf-8")
    return root


def test_executes_only_exact_node_and_emits_path_free_receipt(tmp_path: Path) -> None:
    script = _load_script()
    root = _fixture_root(tmp_path, script)
    calls: list[tuple[Path, str]] = []

    def runner(project_root: Path, node_id: str) -> int:
        calls.append((project_root, node_id))
        return 0

    receipt = script.execute(
        root,
        code_identity_resolver=lambda: "a" * 40,
        test_runner=runner,
    )

    assert calls == [(root, script.ROLLBACK_TEST_NODE_ID)]
    assert receipt == {
        "code_sha": "a" * 40,
        "forced_failure": script.ROLLBACK_FORCED_FAILURE,
        "proof_id": script.ROLLBACK_PROOF_ID,
        "schema_version": 1,
        "semantic_assertions": list(script.ROLLBACK_SEMANTIC_ASSERTIONS),
        "status": "verified",
        "test_node_id": script.ROLLBACK_TEST_NODE_ID,
        "test_result": "passed",
        "test_source_sha256": sha256(
            (root / script.ROLLBACK_TEST_SOURCE).read_bytes()
        ).hexdigest(),
    }
    assert str(root) not in repr(receipt)


def test_rejects_failed_test_without_exposing_test_output(tmp_path: Path) -> None:
    script = _load_script()
    root = _fixture_root(tmp_path, script)

    with pytest.raises(script.RollbackProofError) as captured:
        script.execute(
            root,
            code_identity_resolver=lambda: "a" * 40,
            test_runner=lambda _root, _node: 1,
        )

    assert captured.value.code == "rollback_test_failed"


def test_rejects_code_or_test_source_drift(tmp_path: Path) -> None:
    script = _load_script()
    root = _fixture_root(tmp_path, script)
    identities = iter(("a" * 40, "b" * 40))
    with pytest.raises(script.RollbackProofError) as captured:
        script.execute(
            root,
            code_identity_resolver=lambda: next(identities),
            test_runner=lambda _root, _node: 0,
        )
    assert captured.value.code == "code_identity_changed"

    def mutate_source(project_root: Path, _node: str) -> int:
        (project_root / script.ROLLBACK_TEST_SOURCE).write_text(
            "def test_fixture():\n    assert False\n",
            encoding="utf-8",
        )
        return 0

    with pytest.raises(script.RollbackProofError) as captured:
        script.execute(
            root,
            code_identity_resolver=lambda: "a" * 40,
            test_runner=mutate_source,
        )
    assert captured.value.code == "test_source_changed"


def test_rejects_non_commit_code_identity(tmp_path: Path) -> None:
    script = _load_script()
    root = _fixture_root(tmp_path, script)

    with pytest.raises(script.RollbackProofError) as captured:
        script.execute(
            root,
            code_identity_resolver=lambda: "not-a-commit",
            test_runner=lambda _root, _node: pytest.fail(
                "invalid identity must fail before the test"
            ),
        )

    assert captured.value.code == "code_identity_invalid"
