from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

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
    EXIT_EXECUTION,
    EXIT_INTERNAL,
    EXIT_IO,
    EXIT_MEASUREMENT,
    EXIT_NOT_FOUND,
    EXIT_SUCCESS,
    EXIT_TERMINATED,
    EXIT_USAGE,
    main,
)
from videoscope.benchmark.policy import (
    MAX_POLICY_BYTES,
    POLICY_SCHEMA_VERSION,
    load_policy,
)
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
        self.profile = None
        self.execution_mode = None

    def open_session(  # type: ignore[no-untyped-def]
        self,
        profile,
        assets,
        *,
        execution_mode,
    ):
        del assets
        self.profile = profile
        self.execution_mode = execution_mode
        return self

    def identities(self) -> ExecutionIdentities:
        assert self.profile is not None
        return ExecutionIdentities(
            model_identities=(ComponentIdentity("model", "model@revision"),),
            index_identities=(ComponentIdentity("index", "generation-1"),),
            config_identities=(
                ComponentIdentity(
                    "search",
                    f"search-for-{self.profile.profile_id}",
                ),
            ),
        )

    def lifecycle_identity(self) -> ComponentIdentity:
        assert self.execution_mode in {"cold", "warm"}
        return ComponentIdentity(
            "benchmark_execution_lifecycle",
            f"{self.execution_mode}:test-cache-policy@1",
        )

    def capability_state(self, asset, capability):  # type: ignore[no-untyped-def]
        del asset, capability
        return "complete"

    def search(self, query, assets, *, limit):  # type: ignore[no-untyped-def]
        del query, assets
        return self.hits[:limit]

    def close(self) -> None:
        pass


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


def _allowed_differences(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "allow_code_sha_difference": True,
        "allow_benchmark_profile_difference": False,
        "allow_benchmark_search_plan_difference": False,
        "model_component_ids": [],
        "index_component_ids": [],
        "config_component_ids": [],
    }
    value.update(overrides)
    return value


def _write_policy(path: Path, **overrides: object) -> None:
    value: dict[str, object] = {
        "schema_version": 2,
        "policy_id": "cli-promotion",
        "minimum_completed_cases": 1,
        "require_no_errors": True,
        "allowed_differences": _allowed_differences(),
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
    _dataset_value, dataset_path, registry_root, policy_path = cli_workspace

    exit_code = main(
        [
            "compare",
            "run-baseline",
            "run-candidate",
            "--registry",
            str(registry_root),
            "--policy",
            str(policy_path),
            "--dataset",
            str(dataset_path),
        ]
    )
    captured = capsys.readouterr()
    value = json.loads(captured.out)

    assert exit_code == EXIT_SUCCESS
    assert value["status"] == "eligible"
    assert value["minimum_completed_cases"] == 1
    assert value["statistical_significance"] == "not_assessed"
    assert value["baseline_code_sha"] == "b" * 40
    assert value["candidate_code_sha"] == "c" * 40
    assert value["code_sha_difference_allowed"] is True
    assert value["allowed_differences"] == _allowed_differences()
    assert [item["metric_name"] for item in value["guardrails"]] == [
        "false_positive_rate",
        "recall_at_5",
    ]
    assert captured.err == ""


def test_compare_rejects_forged_aggregates_that_contradict_ranked_evidence(
    cli_workspace,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    _dataset_value, dataset_path, registry_root, policy_path = cli_workspace
    registry = BenchmarkRunRegistry(registry_root)
    missed = registry.read("run-baseline")
    forged_metrics = tuple(
        replace(metric, value=1.0)
        if metric.name == "recall_at_5"
        else replace(metric, value=0.0)
        if metric.name == "false_positive_rate"
        else metric
        for metric in missed.quality_metrics
    )
    registry.add(
        replace(
            missed,
            run_id="run-forged",
            code_sha="c" * 40,
            quality_metrics=forged_metrics,
        )
    )

    exit_code = main(
        [
            "compare",
            "run-baseline",
            "run-forged",
            "--registry",
            str(registry_root),
            "--policy",
            str(policy_path),
            "--dataset",
            str(dataset_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == EXIT_AUDIT
    assert captured.out == ""
    assert json.loads(captured.err)["error"]["code"] == "audit_failed"


@pytest.mark.parametrize(
    "invalid_policy",
    [
        {
            "schema_version": 2,
            "policy_id": "policy",
            "minimum_completed_cases": 1,
            "require_no_errors": True,
            "allowed_differences": _allowed_differences(),
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


def test_policy_v2_requires_an_explicit_migration_from_legacy_v1(tmp_path) -> None:
    path = tmp_path / "legacy-policy.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "policy_id": "legacy",
                "minimum_completed_cases": 1,
                "require_no_errors": True,
                "guardrails": [
                    {
                        "metric_name": "recall_at_5",
                        "direction": "higher_is_better",
                        "allowed_regression": 0.0,
                        "absolute_threshold": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema_version must be 2"):
        load_policy(path)
    assert POLICY_SCHEMA_VERSION == 2


def test_policy_loader_parses_the_explicit_v2_difference_allowlist(tmp_path) -> None:
    path = tmp_path / "policy.json"
    _write_policy(
        path,
        allowed_differences=_allowed_differences(
            allow_benchmark_profile_difference=True,
            model_component_ids=["reranker", "embedding"],
            index_component_ids=["visual_index"],
            config_component_ids=["fusion"],
        ),
    )

    policy = load_policy(path)

    assert policy.allowed_differences.allow_code_sha_difference is True
    assert policy.allowed_differences.allow_benchmark_profile_difference is True
    assert policy.allowed_differences.model_component_ids == (
        "embedding",
        "reranker",
    )
    assert policy.allowed_differences.index_component_ids == ("visual_index",)
    assert policy.allowed_differences.config_component_ids == ("fusion",)


@pytest.mark.parametrize(
    "allowed_differences",
    (
        _allowed_differences(unknown=True),
        _allowed_differences(model_component_ids=["model", "model"]),
        _allowed_differences(
            config_component_ids=["benchmark_execution_lifecycle"],
        ),
        _allowed_differences(allow_code_sha_difference=1),
    ),
)
def test_policy_loader_rejects_unsafe_v2_difference_declarations(
    tmp_path,
    allowed_differences: dict[str, object],
) -> None:
    path = tmp_path / "policy.json"
    _write_policy(path, allowed_differences=allowed_differences)

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


def test_termination_has_a_stable_shell_exit_and_sanitized_error(
    capsys,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    def terminate(_arguments) -> object:  # type: ignore[no-untyped-def]
        raise cli_module._TerminationRequested

    monkeypatch.setattr(cli_module, "_dispatch", terminate)

    exit_code = main(["dataset", "validate", "unused.json"])
    captured = capsys.readouterr()

    assert exit_code == EXIT_TERMINATED == 143
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "error": {
            "code": "terminated",
            "message": "benchmark command was terminated",
        }
    }


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
            "--dataset",
            str(dataset_path),
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


def test_run_command_requires_the_complete_read_only_execution_contract(
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    exit_code = main(["run"])
    captured = capsys.readouterr()

    assert exit_code == EXIT_USAGE
    assert captured.out == ""
    assert json.loads(captured.err)["error"]["code"] == "usage_error"


class _CliEnvironment:
    def __init__(self, dataset: BenchmarkDataset, events: list[str]) -> None:
        self.asset_resolver = LocalAssetResolver(_Repository(dataset))
        self.search_adapter = object()
        self.identity = SimpleNamespace(identity="benchmark-product-environment@1:" + "e" * 64)
        self.is_closed = False
        self._events = events

    def close(self) -> None:
        self._events.append("environment.close")
        self.is_closed = True


def _run_arguments(
    *,
    dataset_path: Path,
    registry_root: Path,
    data_dir: Path,
    scratch_parent: Path,
    preflight: bool = False,
) -> list[str]:
    value = [
        "run",
        "--dataset",
        str(dataset_path),
        "--registry",
        str(registry_root),
        "--data-dir",
        str(data_dir),
        "--scratch-parent",
        str(scratch_parent),
        "--run-id",
        "run-new",
    ]
    if preflight:
        value.append("--preflight")
    return value


def test_run_prepares_closes_then_publishes_complete_manifest(
    cli_workspace,
    tmp_path,
    capsys,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    dataset, dataset_path, registry_root, _policy = cli_workspace
    events: list[str] = []
    environment = _CliEnvironment(dataset, events)
    prepared = SimpleNamespace(
        run_id="run-new",
        run_status="complete",
        measurement_status="not_measured",
        dataset_revision=dataset_revision(dataset),
        case_outcomes=(object(),),
    )

    class FakeRunner:
        def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
            assert kwargs["asset_resolver"] is environment.asset_resolver
            assert kwargs["search"] is environment.search_adapter
            assert kwargs["code_sha"] == "d" * 40
            events.append("runner.init")

        def run(self, supplied, **kwargs):  # type: ignore[no-untyped-def]
            assert supplied == dataset
            assert kwargs == {
                "profile_id": "lexical_qdrant",
                "run_id": "run-new",
                "execution_mode": "warm",
                "publish": False,
            }
            events.append("runner.run")
            return prepared

        def publish_prepared_run(self, run) -> None:  # type: ignore[no-untyped-def]
            assert run is prepared
            assert environment.is_closed
            events.append("runner.publish")

    monkeypatch.setattr(cli_module, "BenchmarkRunner", FakeRunner)
    monkeypatch.setattr(
        cli_module,
        "audit_run_manifest",
        lambda supplied, run: (
            events.append("runner.audit")
            if supplied == dataset and run is prepared
            else pytest.fail("unexpected audit input")
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "open_product_benchmark_environment",
        lambda *_args, **_kwargs: environment,
    )
    monkeypatch.setattr(cli_module, "_current_code_sha", lambda: "d" * 40)
    monkeypatch.setattr(
        cli_module,
        "_current_hardware_profile",
        lambda: HardwareProfile("test-os", "arm64", "test-cpu", 1024),
    )
    data_dir = tmp_path / "product"
    data_dir.mkdir()
    scratch_parent = tmp_path / "scratch"
    scratch_parent.mkdir(mode=0o700)

    exit_code = main(
        _run_arguments(
            dataset_path=dataset_path,
            registry_root=registry_root,
            data_dir=data_dir,
            scratch_parent=scratch_parent,
        )
    )
    captured = capsys.readouterr()

    assert exit_code == EXIT_SUCCESS
    assert events == [
        "runner.init",
        "runner.run",
        "environment.close",
        "runner.audit",
        "runner.publish",
    ]
    assert json.loads(captured.out) == {
        "case_count": 1,
        "dataset_revision": dataset_revision(dataset),
        "execution_mode": "warm",
        "profile_id": "lexical_qdrant",
        "run_id": "run-new",
        "run_status": "complete",
        "status": "published",
    }
    assert captured.err == ""
    assert EXIT_EXECUTION == 8
    assert EXIT_MEASUREMENT == 9


@pytest.mark.parametrize(
    ("run_status", "measurement_status"),
    [
        ("partial", "not_measured"),
        ("failed", "not_measured"),
        ("cancelled", "not_measured"),
        ("complete", "failed"),
    ],
)
def test_run_never_publishes_an_untrustworthy_prepared_manifest(
    cli_workspace,
    tmp_path,
    capsys,
    monkeypatch,
    run_status: str,
    measurement_status: str,
) -> None:  # type: ignore[no-untyped-def]
    dataset, dataset_path, registry_root, _policy = cli_workspace
    events: list[str] = []
    environment = _CliEnvironment(dataset, events)

    class FakeRunner:
        def __init__(self, **_kwargs) -> None:  # type: ignore[no-untyped-def]
            pass

        def run(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(
                run_id="run-new",
                run_status=run_status,
                measurement_status=measurement_status,
                dataset_revision=dataset_revision(dataset),
                case_outcomes=(),
            )

        def publish_prepared_run(self, _run) -> None:  # type: ignore[no-untyped-def]
            raise AssertionError("an incomplete run must not be published")

    monkeypatch.setattr(cli_module, "BenchmarkRunner", FakeRunner)
    monkeypatch.setattr(
        cli_module,
        "open_product_benchmark_environment",
        lambda *_args, **_kwargs: environment,
    )
    monkeypatch.setattr(cli_module, "_current_code_sha", lambda: "d" * 40)
    monkeypatch.setattr(
        cli_module,
        "_current_hardware_profile",
        lambda: HardwareProfile("test-os", "arm64", "test-cpu", 1024),
    )
    data_dir = tmp_path / "product"
    data_dir.mkdir()
    scratch_parent = tmp_path / "scratch"
    scratch_parent.mkdir(mode=0o700)
    before = registry_root.joinpath("registry.json").read_bytes()

    exit_code = main(
        _run_arguments(
            dataset_path=dataset_path,
            registry_root=registry_root,
            data_dir=data_dir,
            scratch_parent=scratch_parent,
        )
    )
    captured = capsys.readouterr()

    assert exit_code == EXIT_EXECUTION
    assert events == ["environment.close"]
    assert registry_root.joinpath("registry.json").read_bytes() == before
    assert json.loads(captured.err)["error"]["code"] == "execution_failed"


def test_cleanup_failure_prevents_publication(
    cli_workspace,
    tmp_path,
    capsys,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    dataset, dataset_path, registry_root, _policy = cli_workspace
    environment = _CliEnvironment(dataset, [])
    close_calls = 0

    def fail_close() -> None:
        nonlocal close_calls
        close_calls += 1
        raise RuntimeError("private cleanup path")

    environment.close = fail_close  # type: ignore[method-assign]

    class FakeRunner:
        def __init__(self, **_kwargs) -> None:  # type: ignore[no-untyped-def]
            pass

        def run(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(
                run_id="run-new",
                run_status="complete",
                measurement_status="not_measured",
                dataset_revision=dataset_revision(dataset),
                case_outcomes=(),
            )

        def publish_prepared_run(self, _run) -> None:  # type: ignore[no-untyped-def]
            raise AssertionError("cleanup failure must prevent publication")

    monkeypatch.setattr(cli_module, "BenchmarkRunner", FakeRunner)
    monkeypatch.setattr(
        cli_module,
        "open_product_benchmark_environment",
        lambda *_args, **_kwargs: environment,
    )
    monkeypatch.setattr(cli_module, "_current_code_sha", lambda: "d" * 40)
    monkeypatch.setattr(
        cli_module,
        "_current_hardware_profile",
        lambda: HardwareProfile("test-os", "arm64", "test-cpu", 1024),
    )
    data_dir = tmp_path / "product"
    data_dir.mkdir()
    scratch_parent = tmp_path / "scratch"
    scratch_parent.mkdir(mode=0o700)
    before = registry_root.joinpath("registry.json").read_bytes()

    exit_code = main(
        _run_arguments(
            dataset_path=dataset_path,
            registry_root=registry_root,
            data_dir=data_dir,
            scratch_parent=scratch_parent,
        )
    )
    captured = capsys.readouterr()

    assert exit_code == EXIT_EXECUTION
    assert close_calls > 1
    assert registry_root.joinpath("registry.json").read_bytes() == before
    assert "private cleanup path" not in captured.err


def test_code_identity_drift_after_cleanup_prevents_publication(
    cli_workspace,
    tmp_path,
    capsys,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    dataset, dataset_path, registry_root, _policy = cli_workspace
    environment = _CliEnvironment(dataset, [])
    published = False

    class FakeRunner:
        def __init__(self, **_kwargs) -> None:  # type: ignore[no-untyped-def]
            pass

        def run(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(
                run_id="run-new",
                run_status="complete",
                measurement_status="not_measured",
                dataset_revision=dataset_revision(dataset),
                case_outcomes=(),
            )

        def publish_prepared_run(self, _run) -> None:  # type: ignore[no-untyped-def]
            nonlocal published
            published = True

    code_identities = iter(("d" * 40, "e" * 40))
    monkeypatch.setattr(cli_module, "BenchmarkRunner", FakeRunner)
    monkeypatch.setattr(cli_module, "audit_run_manifest", lambda *_args: None)
    monkeypatch.setattr(
        cli_module,
        "open_product_benchmark_environment",
        lambda *_args, **_kwargs: environment,
    )
    monkeypatch.setattr(
        cli_module,
        "_current_code_sha",
        lambda: next(code_identities),
    )
    monkeypatch.setattr(
        cli_module,
        "_current_hardware_profile",
        lambda: HardwareProfile("test-os", "arm64", "test-cpu", 1024),
    )
    data_dir = tmp_path / "product"
    data_dir.mkdir()
    scratch_parent = tmp_path / "scratch"
    scratch_parent.mkdir(mode=0o700)
    before = registry_root.joinpath("registry.json").read_bytes()

    exit_code = main(
        _run_arguments(
            dataset_path=dataset_path,
            registry_root=registry_root,
            data_dir=data_dir,
            scratch_parent=scratch_parent,
        )
    )
    captured = capsys.readouterr()

    assert exit_code == EXIT_EXECUTION
    assert environment.is_closed
    assert published is False
    assert registry_root.joinpath("registry.json").read_bytes() == before
    assert json.loads(captured.err)["error"]["code"] == "execution_failed"


def test_preflight_checks_every_capability_without_search_or_registry_mutation(
    cli_workspace,
    tmp_path,
    capsys,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    dataset, dataset_path, registry_root, _policy = cli_workspace
    events: list[str] = []
    environment = _CliEnvironment(dataset, events)
    resolved = environment.asset_resolver.resolve(dataset.assets[0])

    class Session:
        def identities(self) -> ExecutionIdentities:
            events.append("session.identities")
            return ExecutionIdentities(
                model_identities=(
                    ComponentIdentity("text_embedding", "fastembed@revision"),
                ),
                index_identities=(
                    ComponentIdentity("text_vector_index", "1" * 64),
                    ComponentIdentity(
                        "text_vector_generations",
                        "sha256:" + "2" * 64,
                    ),
                ),
                config_identities=(
                    ComponentIdentity(
                        "benchmark_product_environment",
                        environment.identity.identity,
                    ),
                    ComponentIdentity(
                        "evaluation_search_configuration",
                        cli_module.get_profile("lexical_qdrant")
                        .search_plan.identity.replace(
                            "evaluation-search-plan",
                            "evaluation-search-configuration",
                            1,
                        ),
                    ),
                    ComponentIdentity(
                        "product_search_lifecycle",
                        "warm:process-cache-preserved@1",
                    ),
                    ComponentIdentity(
                        "product_search_runtime",
                        "sha256:" + "3" * 64,
                    ),
                ),
            )

        def lifecycle_identity(self) -> ComponentIdentity:
            events.append("session.lifecycle")
            return ComponentIdentity(
                "benchmark_execution_lifecycle",
                "warm:process-cache-preserved@1",
            )

        def capability_state(self, asset, capability):  # type: ignore[no-untyped-def]
            assert asset == resolved
            assert capability == "text_vectors"
            events.append("session.capability")
            return "complete"

        def search(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("preflight must not execute search")

        def close(self) -> None:
            events.append("session.close")

    session = Session()

    class Adapter:
        def open_session(self, profile, assets, *, execution_mode):  # type: ignore[no-untyped-def]
            assert profile.profile_id == "lexical_qdrant"
            assert assets == (resolved,)
            assert execution_mode == "warm"
            events.append("adapter.open_session")
            return session

    environment.search_adapter = Adapter()
    monkeypatch.setattr(
        cli_module,
        "open_product_benchmark_environment",
        lambda *_args, **_kwargs: environment,
    )
    monkeypatch.setattr(cli_module, "_current_code_sha", lambda: "d" * 40)
    monkeypatch.setattr(
        cli_module,
        "_current_hardware_profile",
        lambda: HardwareProfile("test-os", "arm64", "test-cpu", 1024),
    )
    data_dir = tmp_path / "product"
    data_dir.mkdir()
    scratch_parent = tmp_path / "scratch"
    scratch_parent.mkdir(mode=0o700)
    before = {
        path.relative_to(registry_root): path.read_bytes()
        for path in registry_root.rglob("*")
        if path.is_file()
    }

    exit_code = main(
        _run_arguments(
            dataset_path=dataset_path,
            registry_root=registry_root,
            data_dir=data_dir,
            scratch_parent=scratch_parent,
            preflight=True,
        )
    )
    captured = capsys.readouterr()
    after = {
        path.relative_to(registry_root): path.read_bytes()
        for path in registry_root.rglob("*")
        if path.is_file()
    }

    assert exit_code == EXIT_SUCCESS
    assert before == after
    assert events == [
        "adapter.open_session",
        "session.identities",
        "session.lifecycle",
        "session.capability",
        "session.close",
        "environment.close",
    ]
    assert json.loads(captured.out) == {
        "asset_count": 1,
        "capability_count": 1,
        "code_sha": "d" * 40,
        "dataset_revision": dataset_revision(dataset),
        "environment_identity": "benchmark-product-environment@1:" + "e" * 64,
        "execution_mode": "warm",
        "profile_id": "lexical_qdrant",
        "run_id": "run-new",
        "status": "ready",
    }
    assert captured.err == ""


def test_bindings_must_be_an_exact_bounded_alias_to_video_object(
    cli_workspace,
    tmp_path,
    capsys,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    _dataset_value, dataset_path, registry_root, _policy = cli_workspace
    bindings = tmp_path / "bindings.json"
    bindings.write_text('{"unknown-alias":"private-video"}', encoding="utf-8")
    monkeypatch.setattr(
        cli_module,
        "open_product_benchmark_environment",
        lambda *_args, **_kwargs: pytest.fail("invalid bindings must fail before open"),
    )
    data_dir = tmp_path / "product"
    data_dir.mkdir()
    scratch_parent = tmp_path / "scratch"
    scratch_parent.mkdir(mode=0o700)
    arguments = _run_arguments(
        dataset_path=dataset_path,
        registry_root=registry_root,
        data_dir=data_dir,
        scratch_parent=scratch_parent,
        preflight=True,
    )
    arguments.extend(["--bindings", str(bindings)])

    exit_code = main(arguments)
    captured = capsys.readouterr()

    assert exit_code == EXIT_DATA
    assert json.loads(captured.err)["error"]["code"] == "invalid_data"
    assert str(bindings) not in captured.err


@pytest.mark.parametrize(
    "status",
    (
        b" M docs/benchmark-core.md\0",
        b"?? backend/src/videoscope/private.py\0",
        b"?? backend/uv.lock\0",
    ),
)
def test_code_identity_rejects_tracked_or_relevant_untracked_changes(
    monkeypatch,
    status: bytes,
) -> None:
    def git_bytes(_root, *arguments):  # type: ignore[no-untyped-def]
        if arguments[0] == "rev-parse":
            return b"a" * 40 + b"\n"
        return status

    monkeypatch.setattr(cli_module, "_git_bytes", git_bytes)

    with pytest.raises(cli_module._CodeIdentityError):
        cli_module._current_code_sha()


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
