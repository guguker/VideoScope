from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import videoscope.benchmark.cli as cli_module
from videoscope.benchmark import (
    AssetProvenance,
    BenchmarkAsset,
    BenchmarkDataset,
    BenchmarkInterval,
    BenchmarkRunRegistry,
    ComponentIdentity,
    HardwareProfile,
    LocalAssetResolver,
    QueryCase,
    dataset_revision,
    write_dataset,
)
from videoscope.benchmark.cli import (
    EXIT_AUDIT,
    EXIT_CONFLICT,
    EXIT_DATA,
    EXIT_INTERNAL,
    EXIT_IO,
    EXIT_NOT_FOUND,
    EXIT_SUCCESS,
    EXIT_USAGE,
    main,
)
from videoscope.benchmark.policy import MAX_POLICY_BYTES, load_policy
from videoscope.benchmark.runner import (
    BenchmarkRunner,
    BenchmarkSearchHit,
    ExecutionIdentities,
)


def _dataset(*, description: str = "CLI fixture") -> BenchmarkDataset:
    asset = BenchmarkAsset(
        asset_id="fixture-a",
        sha256="a" * 64,
        byte_size=1_024,
        duration_seconds=60.0,
        provenance=AssetProvenance(
            source="Public CLI fixture",
            source_uri="https://example.test/fixture-a.mp4",
            license_id="CC-BY-4.0",
        ),
    )
    return BenchmarkDataset(
        schema_version=1,
        dataset_id="cli-core",
        dataset_version="1.0.0",
        description=description,
        assets=(asset,),
        cases=(
            QueryCase(
                case_id="case-a",
                query="made basket",
                asset_ids=(asset.asset_id,),
                domain="basketball",
                modalities=("sports", "visual"),
                label_quality="gold",
                split_group="match-a",
                relevant_intervals=(
                    BenchmarkInterval(asset.asset_id, 10.0, 12.0),
                ),
            ),
        ),
    )


@dataclass(frozen=True)
class _RepositoryAsset:
    id: str
    sha256: str
    byte_size: int
    duration_seconds: float
    video_id: str


class _Repository:
    def __init__(self, dataset: BenchmarkDataset) -> None:
        asset = dataset.assets[0]
        self.asset = _RepositoryAsset(
            id=f"sha256:{asset.sha256}",
            sha256=asset.sha256,
            byte_size=asset.byte_size,
            duration_seconds=asset.duration_seconds,
            video_id="private-video-row",
        )

    def find_assets_by_sha256(self, digest: str) -> tuple[_RepositoryAsset, ...]:
        return (self.asset,) if digest == self.asset.sha256 else ()


class _SearchAdapter:
    def __init__(self, hits: tuple[BenchmarkSearchHit, ...]) -> None:
        self.hits = hits

    def identities(self, profile) -> ExecutionIdentities:  # type: ignore[no-untyped-def]
        return ExecutionIdentities(
            model_identities=(ComponentIdentity("model", "model@revision"),),
            index_identities=(ComponentIdentity("index", "generation-1"),),
            config_identities=(
                ComponentIdentity("search", f"search-for-{profile.profile_id}"),
            ),
        )

    def capability_state(self, profile, asset, capability):  # type: ignore[no-untyped-def]
        del profile, asset, capability
        return "complete"

    def search(self, profile, query, assets, *, limit):  # type: ignore[no-untyped-def]
        del profile, query, assets
        return self.hits[:limit]


class _Timer:
    def __init__(self) -> None:
        self.value = -0.01

    def __call__(self) -> float:
        self.value += 0.01
        return self.value


def _create_registry(root: Path, dataset: BenchmarkDataset) -> BenchmarkRunRegistry:
    registry = BenchmarkRunRegistry(root)
    common = {
        "registry": registry,
        "asset_resolver": LocalAssetResolver(_Repository(dataset)),
        "hardware": HardwareProfile("macOS", "arm64", "Apple M4", 1024, "Metal"),
        "clock": lambda: datetime(2026, 8, 18, 12, 0, tzinfo=UTC),
    }
    BenchmarkRunner(
        **common,
        search=_SearchAdapter(
            (BenchmarkSearchHit("fixture-a", 30.0, 31.0, 0.4),)
        ),
        code_sha="b" * 40,
        timer=_Timer(),
    ).run(
        dataset,
        profile_id="dense_siglip",
        run_id="run-baseline",
        execution_mode="warm",
    )
    BenchmarkRunner(
        **common,
        search=_SearchAdapter(
            (BenchmarkSearchHit("fixture-a", 10.0, 12.0, 0.9),)
        ),
        code_sha="c" * 40,
        timer=_Timer(),
    ).run(
        dataset,
        profile_id="dense_siglip",
        run_id="run-candidate",
        execution_mode="warm",
    )
    return registry


def _write_policy(path: Path, **overrides: object) -> None:
    value: dict[str, object] = {
        "schema_version": 1,
        "policy_id": "cli-promotion",
        "minimum_completed_cases": 1,
        "require_no_errors": True,
        "guardrails": [
            {
                "metric_name": "recall_at_5",
                "direction": "higher_is_better",
                "allowed_regression": 0.0,
                "absolute_threshold": 0.8,
            },
            {
                "metric_name": "false_positive_rate",
                "direction": "lower_is_better",
                "allowed_regression": 0.0,
                "absolute_threshold": 0.2,
            },
        ],
    }
    value.update(overrides)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def cli_workspace(tmp_path):  # type: ignore[no-untyped-def]
    dataset = _dataset()
    dataset_path = tmp_path / "dataset.json"
    write_dataset(dataset_path, dataset)
    registry_root = tmp_path / "registry"
    _create_registry(registry_root, dataset)
    policy_path = tmp_path / "policy.json"
    _write_policy(policy_path)
    return dataset, dataset_path, registry_root, policy_path


def test_dataset_validate_emits_canonical_summary_json(
    cli_workspace,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    dataset, dataset_path, _registry, _policy = cli_workspace

    exit_code = main(["dataset", "validate", str(dataset_path)])
    captured = capsys.readouterr()
    value = json.loads(captured.out)

    assert exit_code == EXIT_SUCCESS
    assert captured.err == ""
    assert value == {
        "dataset": {
            "asset_count": 1,
            "case_count": 1,
            "dataset_id": "cli-core",
            "dataset_version": "1.0.0",
            "hard_negative_count": 0,
            "negative_case_count": 0,
            "positive_case_count": 1,
            "relevant_interval_count": 1,
            "revision": dataset_revision(dataset),
            "schema_version": 1,
        },
        "status": "valid",
    }
    assert captured.out == json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"


def test_dataset_import_is_the_only_command_that_creates_catalog_state(
    cli_workspace,
    tmp_path,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    dataset, dataset_path, _registry, _policy = cli_workspace
    catalog_root = tmp_path / "catalog"

    exit_code = main(
        ["dataset", "import", str(dataset_path), "--catalog", str(catalog_root)]
    )
    value = json.loads(capsys.readouterr().out)

    assert exit_code == EXIT_SUCCESS
    assert value["catalog_entry"] == (
        f"{dataset.dataset_id}/{dataset_revision(dataset)}.json"
    )
    assert (
        catalog_root
        / dataset.dataset_id
        / f"{dataset_revision(dataset)}.json"
    ).is_file()


def test_runs_list_and_show_are_deterministic_and_portable(
    cli_workspace,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    _dataset_value, _dataset_path, registry_root, _policy = cli_workspace

    assert main(["runs", "list", "--registry", str(registry_root)]) == EXIT_SUCCESS
    first = capsys.readouterr().out
    assert main(["runs", "list", "--registry", str(registry_root)]) == EXIT_SUCCESS
    second = capsys.readouterr().out
    listing = json.loads(first)

    assert first == second
    assert [item["run_id"] for item in listing["runs"]] == [
        "run-baseline",
        "run-candidate",
    ]

    assert main(
        [
            "runs",
            "show",
            "run-candidate",
            "--registry",
            str(registry_root),
        ]
    ) == EXIT_SUCCESS
    shown = capsys.readouterr().out
    manifest = json.loads(shown)
    assert manifest["run_id"] == "run-candidate"
    assert manifest["case_outcomes"][0]["result_evidence"][0]["rank"] == 1
    assert "private-video-row" not in shown


def test_runs_audit_recomputes_the_immutable_manifest(
    cli_workspace,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    dataset, dataset_path, registry_root, _policy = cli_workspace

    exit_code = main(
        [
            "runs",
            "audit",
            "run-candidate",
            "--registry",
            str(registry_root),
            "--dataset",
            str(dataset_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == EXIT_SUCCESS
    assert json.loads(captured.out) == {
        "case_count": 1,
        "dataset_revision": dataset_revision(dataset),
        "run_id": "run-candidate",
        "status": "valid",
    }
    assert captured.err == ""


def test_compare_loads_strict_policy_and_emits_explicit_guardrails(
    cli_workspace,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    _dataset_value, _dataset_path, registry_root, policy_path = cli_workspace

    exit_code = main(
        [
            "compare",
            "run-baseline",
            "run-candidate",
            "--registry",
            str(registry_root),
            "--policy",
            str(policy_path),
        ]
    )
    captured = capsys.readouterr()
    value = json.loads(captured.out)

    assert exit_code == EXIT_SUCCESS
    assert value["status"] == "eligible"
    assert value["minimum_completed_cases"] == 1
    assert value["statistical_significance"] == "not_assessed"
    assert [item["metric_name"] for item in value["guardrails"]] == [
        "false_positive_rate",
        "recall_at_5",
    ]
    assert captured.err == ""


@pytest.mark.parametrize(
    "invalid_policy",
    [
        {
            "schema_version": 1,
            "policy_id": "policy",
            "minimum_completed_cases": 1,
            "require_no_errors": True,
            "guardrails": [],
            "unknown": "rejected",
        },
        {
            "schema_version": 999,
            "policy_id": "policy",
            "minimum_completed_cases": 1,
            "require_no_errors": True,
            "guardrails": [],
        },
    ],
)
def test_policy_loader_rejects_unknown_fields_and_versions(
    tmp_path,
    invalid_policy: dict[str, object],
) -> None:
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(invalid_policy), encoding="utf-8")

    with pytest.raises(ValueError):
        load_policy(path)


@pytest.mark.parametrize(
    "payload",
    [
        '{"schema_version":1,"schema_version":1}',
        '{"schema_version":1,"policy_id":"p","minimum_completed_cases":1,'
        '"require_no_errors":true,"guardrails":[],"value":NaN}',
    ],
)
def test_policy_loader_rejects_duplicate_keys_and_nonfinite_json(
    tmp_path,
    payload: str,
) -> None:
    path = tmp_path / "policy.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError):
        load_policy(path)


def test_policy_loader_bounds_regular_file_input_and_rejects_symlinks(tmp_path) -> None:
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * (MAX_POLICY_BYTES + 1))
    target = tmp_path / "target.json"
    _write_policy(target)
    linked = tmp_path / "linked.json"
    linked.symlink_to(target)

    with pytest.raises(ValueError, match="limit"):
        load_policy(oversized)
    with pytest.raises(ValueError, match="symbolic link"):
        load_policy(linked)


def test_policy_loader_rejects_unknown_guardrail_fields_and_excess_count(
    tmp_path,
) -> None:
    unknown = tmp_path / "unknown-guardrail.json"
    _write_policy(unknown)
    value = json.loads(unknown.read_text(encoding="utf-8"))
    value["guardrails"][0]["private_path"] = "/Users/person/model"  # type: ignore[index]
    unknown.write_text(json.dumps(value), encoding="utf-8")
    excessive = tmp_path / "excessive.json"
    guardrail = {
        "metric_name": "recall_at_5",
        "direction": "higher_is_better",
        "allowed_regression": 0.0,
        "absolute_threshold": 0.8,
    }
    _write_policy(excessive, guardrails=[guardrail] * 257)

    with pytest.raises(ValueError, match="unexpected"):
        load_policy(unknown)
    with pytest.raises(ValueError, match="limit"):
        load_policy(excessive)


def test_invalid_input_has_stable_exit_and_sanitized_machine_error(
    tmp_path,
    capsys,
) -> None:
    private_path = tmp_path / "private-secret-dataset.json"
    private_path.write_text("{broken", encoding="utf-8")

    exit_code = main(["dataset", "validate", str(private_path)])
    captured = capsys.readouterr()
    error = json.loads(captured.err)

    assert exit_code == EXIT_DATA
    assert captured.out == ""
    assert error == {
        "error": {
            "code": "invalid_data",
            "message": "benchmark input or stored data is invalid",
        }
    }
    assert str(private_path) not in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    ("failure", "exit_code", "error_code"),
    [
        (FileExistsError("private destination"), EXIT_CONFLICT, "conflict"),
        (PermissionError("private path /Users/person"), EXIT_IO, "io_error"),
        (RuntimeError("private provider token"), EXIT_INTERNAL, "internal_error"),
    ],
)
def test_storage_and_unexpected_failures_have_stable_sanitized_exits(
    tmp_path,
    capsys,
    monkeypatch,
    failure: Exception,
    exit_code: int,
    error_code: str,
) -> None:
    def fail(_path):  # type: ignore[no-untyped-def]
        raise failure

    monkeypatch.setattr(cli_module, "load_dataset", fail)

    actual = main(["dataset", "validate", str(tmp_path / "private.json")])
    captured = capsys.readouterr()

    assert actual == exit_code
    assert captured.out == ""
    assert json.loads(captured.err)["error"]["code"] == error_code
    assert "private" not in captured.err
    assert "Traceback" not in captured.err


def test_missing_registry_is_read_only_and_does_not_create_state(tmp_path, capsys) -> None:
    missing = tmp_path / "missing-registry"

    exit_code = main(["runs", "list", "--registry", str(missing)])
    captured = capsys.readouterr()

    assert exit_code == EXIT_NOT_FOUND
    assert captured.out == ""
    assert not missing.exists()

    empty = tmp_path / "empty-registry"
    empty.mkdir()
    exit_code = main(["runs", "list", "--registry", str(empty)])
    capsys.readouterr()

    assert exit_code == EXIT_NOT_FOUND
    assert tuple(empty.iterdir()) == ()


def test_tampered_run_and_wrong_dataset_fail_with_distinct_sanitized_codes(
    cli_workspace,
    tmp_path,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    _dataset_value, _dataset_path, registry_root, _policy = cli_workspace
    wrong_dataset_path = tmp_path / "wrong-private-dataset.json"
    write_dataset(wrong_dataset_path, _dataset(description="different revision"))

    audit_code = main(
        [
            "runs",
            "audit",
            "run-candidate",
            "--registry",
            str(registry_root),
            "--dataset",
            str(wrong_dataset_path),
        ]
    )
    audit_error = capsys.readouterr()

    assert audit_code == EXIT_AUDIT
    assert "wrong-private-dataset" not in audit_error.err

    manifest = registry_root / "runs" / "run-candidate" / "manifest.json"
    manifest.write_bytes(manifest.read_bytes() + b" ")
    show_code = main(
        [
            "runs",
            "show",
            "run-candidate",
            "--registry",
            str(registry_root),
        ]
    )
    show_error = capsys.readouterr()

    assert show_code == EXIT_DATA
    assert "manifest.json" not in show_error.err
    assert "Traceback" not in show_error.err


def test_read_commands_do_not_mutate_registry_files(
    cli_workspace,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    _dataset_value, dataset_path, registry_root, policy_path = cli_workspace

    before = {
        path.relative_to(registry_root): path.read_bytes()
        for path in registry_root.rglob("*")
        if path.is_file()
    }
    commands = (
        ["runs", "list", "--registry", str(registry_root)],
        [
            "runs",
            "show",
            "run-candidate",
            "--registry",
            str(registry_root),
        ],
        [
            "runs",
            "audit",
            "run-candidate",
            "--registry",
            str(registry_root),
            "--dataset",
            str(dataset_path),
        ],
        [
            "compare",
            "run-baseline",
            "run-candidate",
            "--registry",
            str(registry_root),
            "--policy",
            str(policy_path),
        ],
    )
    for command in commands:
        assert main(command) == EXIT_SUCCESS
        capsys.readouterr()
    after = {
        path.relative_to(registry_root): path.read_bytes()
        for path in registry_root.rglob("*")
        if path.is_file()
    }

    assert after == before


def test_cli_has_no_unbacked_run_command_and_usage_errors_are_machine_readable(
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    exit_code = main(["run"])
    captured = capsys.readouterr()

    assert exit_code == EXIT_USAGE
    assert captured.out == ""
    assert json.loads(captured.err)["error"]["code"] == "usage_error"


def test_python_module_entrypoint_emits_parseable_json(
    cli_workspace,
) -> None:  # type: ignore[no-untyped-def]
    dataset, dataset_path, _registry, _policy = cli_workspace

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "videoscope.benchmark",
            "dataset",
            "validate",
            str(dataset_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == EXIT_SUCCESS
    assert json.loads(completed.stdout)["dataset"]["revision"] == dataset_revision(
        dataset
    )
    assert completed.stderr == ""


def test_cli_rejects_fifo_input_without_waiting_for_a_writer(tmp_path) -> None:
    fifo = tmp_path / "private-dataset.pipe"
    os.mkfifo(fifo)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "videoscope.benchmark",
            "dataset",
            "validate",
            str(fifo),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
    )

    assert completed.returncode == EXIT_DATA
    assert completed.stdout == ""
    assert json.loads(completed.stderr)["error"]["code"] == "invalid_data"
    assert str(fifo) not in completed.stderr
