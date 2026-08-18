from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
import json

import pytest

from videoscope.benchmark import (
    BenchmarkCaseOutcome,
    BenchmarkDataError,
    BenchmarkDurabilityError,
    BenchmarkRunManifest,
    BenchmarkRunRegistry,
    BenchmarkResultEvidence,
    ComponentIdentity,
    HardwareProfile,
    MetricValue,
)


def _run(run_id: str = "run-001") -> BenchmarkRunManifest:
    return BenchmarkRunManifest(
        schema_version=1,
        run_id=run_id,
        created_at="2026-08-18T12:30:00Z",
        code_sha="a" * 40,
        dataset_revision="b" * 64,
        model_identities=(
            ComponentIdentity("siglip", "google/siglip@revision-1"),
        ),
        index_identities=(
            ComponentIdentity("visual_dense", "generation-7:spec-sha"),
        ),
        config_identities=(
            ComponentIdentity("search_profile", "dense-v1:config-sha"),
        ),
        hardware=HardwareProfile(
            operating_system="macOS 15.6",
            architecture="arm64",
            processor="Apple M4 Pro",
            memory_bytes=24 * 1024**3,
            accelerator="Metal",
        ),
        execution_mode="warm",
        metrics=(
            MetricValue("recall_at_5", 0.75, "ratio"),
            MetricValue("mean_latency_ms", 125.5, "milliseconds"),
        ),
        case_outcomes=(
            BenchmarkCaseOutcome(
                case_id="made-shot",
                status="complete",
                latency_ms=120.0,
                result_count=5,
                metrics=(MetricValue("reciprocal_rank", 1.0, "ratio"),),
                result_evidence=tuple(
                    BenchmarkResultEvidence(
                        rank=rank,
                        asset_id="asset-a",
                        start_seconds=float(rank),
                        end_seconds=float(rank + 1),
                        score=1.0 / rank,
                    )
                    for rank in range(1, 6)
                ),
            ),
            BenchmarkCaseOutcome(
                case_id="no-dunk",
                status="failed",
                latency_ms=10.0,
                result_count=0,
                diagnostic_code="required_index_stale",
            ),
        ),
    )


def test_run_contract_is_deeply_immutable() -> None:
    run = _run()

    with pytest.raises(FrozenInstanceError):
        run.run_id = "changed"  # type: ignore[misc]
    assert isinstance(run.metrics, tuple)
    assert isinstance(run.case_outcomes, tuple)


@pytest.mark.parametrize(
    "change",
    [
        {"code_sha": "not-a-commit"},
        {"dataset_revision": "a" * 40},
        {"execution_mode": "sometimes"},
    ],
)
def test_run_rejects_invalid_revisions_modes_and_metrics(
    change,
) -> None:  # type: ignore[no-untyped-def]
    values = _run_values()
    values.update(change)

    with pytest.raises(BenchmarkDataError):
        BenchmarkRunManifest(**values)


def test_metric_rejects_non_finite_values() -> None:
    with pytest.raises(BenchmarkDataError):
        MetricValue("latency", float("nan"))


def test_run_rejects_duplicate_identity_and_outcome_names() -> None:
    values = _run_values()
    identity = ComponentIdentity("visual", "generation-1")
    values["index_identities"] = (identity, identity)

    with pytest.raises(BenchmarkDataError, match="index_identities"):
        BenchmarkRunManifest(**values)

    values = _run_values()
    outcome = values["case_outcomes"][0]
    values["case_outcomes"] = (outcome, outcome)
    with pytest.raises(BenchmarkDataError, match="case_outcomes"):
        BenchmarkRunManifest(**values)


def test_outcome_requires_consistent_status_and_diagnostic_code() -> None:
    with pytest.raises(BenchmarkDataError, match="diagnostic_code"):
        BenchmarkCaseOutcome("case", "failed", 1.0, 0)
    with pytest.raises(BenchmarkDataError, match="diagnostic_code"):
        BenchmarkCaseOutcome("case", "complete", 1.0, 1, diagnostic_code="error")
    with pytest.raises(BenchmarkDataError):
        BenchmarkCaseOutcome("case", "complete", float("inf"), 1)


def test_outcome_requires_bounded_canonical_portable_result_evidence() -> None:
    evidence = BenchmarkResultEvidence(1, "asset-a", 1.0, 2.0, 0.9)

    with pytest.raises(BenchmarkDataError, match="result_count"):
        BenchmarkCaseOutcome("case", "complete", 1.0, 1)
    with pytest.raises(BenchmarkDataError, match="rank"):
        BenchmarkCaseOutcome(
            "case",
            "complete",
            1.0,
            1,
            result_evidence=(replace(evidence, rank=2),),
        )
    with pytest.raises(BenchmarkDataError, match="duplicate"):
        BenchmarkCaseOutcome(
            "case",
            "complete",
            1.0,
            2,
            result_evidence=(evidence, replace(evidence, rank=2, score=0.8)),
        )
    with pytest.raises(BenchmarkDataError, match="failed"):
        BenchmarkCaseOutcome(
            "case",
            "failed",
            1.0,
            1,
            result_evidence=(evidence,),
            diagnostic_code="provider_failed",
        )
    with pytest.raises(BenchmarkDataError, match="between zero and one"):
        BenchmarkResultEvidence(1, "asset-a", 1.0, 2.0, 1.1)


def test_registry_persists_runs_and_lists_them_in_canonical_order(tmp_path) -> None:
    registry = BenchmarkRunRegistry(tmp_path / "benchmark-runs")

    registry.add(_run("run-002"))
    registry.add(_run("run-001"))

    assert registry.read("run-001") == _run("run-001")
    assert [entry.run_id for entry in registry.list()] == ["run-001", "run-002"]
    assert all(len(entry.manifest_sha256) == 64 for entry in registry.list())
    assert (tmp_path / "benchmark-runs" / "runs" / "run-001" / "manifest.json").is_file()


def test_registry_refuses_to_overwrite_an_existing_run(tmp_path) -> None:
    registry = BenchmarkRunRegistry(tmp_path / "benchmark-runs")
    original = _run()
    registry.add(original)
    manifest_path = tmp_path / "benchmark-runs" / "runs" / original.run_id / "manifest.json"
    original_bytes = manifest_path.read_bytes()

    with pytest.raises(FileExistsError):
        registry.add(original)

    assert manifest_path.read_bytes() == original_bytes
    assert registry.read(original.run_id) == original


def test_registry_detects_manifest_tampering(tmp_path) -> None:
    registry = BenchmarkRunRegistry(tmp_path / "benchmark-runs")
    registry.add(_run())
    path = tmp_path / "benchmark-runs" / "runs" / "run-001" / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["execution_mode"] = "cold"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(BenchmarkDataError, match="digest"):
        registry.read("run-001")


def test_registry_detects_metadata_that_disagrees_with_the_manifest(tmp_path) -> None:
    registry = BenchmarkRunRegistry(tmp_path / "benchmark-runs")
    registry.add(_run())
    path = tmp_path / "benchmark-runs" / "registry.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["runs"][0]["dataset_revision"] = "c" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(BenchmarkDataError, match="metadata"):
        registry.read("run-001")


def test_registry_update_failure_preserves_existing_registry(
    tmp_path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    registry = BenchmarkRunRegistry(tmp_path / "benchmark-runs")
    registry.add(_run("run-001"))
    original_entries = registry.list()

    from videoscope.benchmark import storage

    real_replace = storage.os.replace

    def fail_registry_replace(source, destination):  # type: ignore[no-untyped-def]
        if str(destination).endswith("registry.json"):
            raise OSError("simulated registry storage failure")
        return real_replace(source, destination)

    monkeypatch.setattr(storage.os, "replace", fail_registry_replace)

    with pytest.raises(OSError, match="simulated"):
        registry.add(_run("run-002"))

    assert registry.list() == original_entries
    assert not (tmp_path / "benchmark-runs" / "runs" / "run-002").exists()
    assert not list((tmp_path / "benchmark-runs").glob(".*.tmp"))


def test_registry_loader_rejects_unknown_registry_fields(tmp_path) -> None:
    registry = BenchmarkRunRegistry(tmp_path / "benchmark-runs")
    registry.add(_run())
    path = tmp_path / "benchmark-runs" / "registry.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["unexpected"] = []
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(BenchmarkDataError, match="unexpected fields"):
        registry.list()


def test_registry_rejects_a_symlinked_runs_directory(tmp_path) -> None:
    root = tmp_path / "registry"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "runs").symlink_to(outside, target_is_directory=True)

    with pytest.raises(BenchmarkDataError, match="symbolic link"):
        BenchmarkRunRegistry(root).add(_run())

    assert not list(outside.iterdir())


def test_registry_rebuilds_a_corrupt_derived_index_from_immutable_runs(tmp_path) -> None:
    registry = BenchmarkRunRegistry(tmp_path / "benchmark-runs")
    registry.add(_run("run-002"))
    registry.add(_run("run-001"))
    registry.registry_path.write_text("{broken", encoding="utf-8")

    with pytest.raises(BenchmarkDataError):
        registry.list()

    rebuilt = registry.rebuild()

    assert [entry.run_id for entry in rebuilt] == ["run-001", "run-002"]
    assert registry.read("run-001") == _run("run-001")
    assert registry.read("run-002") == _run("run-002")


def test_registry_rebuild_fails_closed_on_a_corrupt_run_manifest(tmp_path) -> None:
    registry = BenchmarkRunRegistry(tmp_path / "benchmark-runs")
    registry.add(_run("run-001"))
    registry.add(_run("run-002"))
    registry_bytes = registry.registry_path.read_bytes()
    manifest = registry.runs_path / "run-002" / "manifest.json"
    manifest.write_text("{broken", encoding="utf-8")

    with pytest.raises(BenchmarkDataError):
        registry.rebuild()

    assert registry.registry_path.read_bytes() == registry_bytes


def test_registry_rebuild_does_not_bless_a_tampered_valid_manifest(tmp_path) -> None:
    registry = BenchmarkRunRegistry(tmp_path / "benchmark-runs")
    registry.add(_run())
    registry_bytes = registry.registry_path.read_bytes()
    manifest = registry.runs_path / "run-001" / "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["execution_mode"] = "cold"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(BenchmarkDataError, match="immutable"):
        registry.rebuild()

    assert registry.registry_path.read_bytes() == registry_bytes


def test_registry_rebuild_does_not_drop_a_registered_missing_run(tmp_path) -> None:
    registry = BenchmarkRunRegistry(tmp_path / "benchmark-runs")
    registry.add(_run("run-001"))
    registry.add(_run("run-002"))
    registry_bytes = registry.registry_path.read_bytes()
    missing_run = registry.runs_path / "run-002"
    (missing_run / "manifest.json").unlink()
    missing_run.rmdir()

    with pytest.raises(BenchmarkDataError, match="missing immutable runs"):
        registry.rebuild()

    assert registry.registry_path.read_bytes() == registry_bytes


def test_concurrent_registry_writers_are_serialized_without_lost_entries(tmp_path) -> None:
    registry = BenchmarkRunRegistry(tmp_path / "benchmark-runs")

    with ThreadPoolExecutor(max_workers=8) as pool:
        entries = list(pool.map(lambda index: registry.add(_run(f"run-{index:03d}")), range(30)))

    assert len(entries) == 30
    assert [entry.run_id for entry in registry.list()] == [
        f"run-{index:03d}" for index in range(30)
    ]


def test_concurrent_duplicate_run_id_has_exactly_one_winner(tmp_path) -> None:
    registry = BenchmarkRunRegistry(tmp_path / "benchmark-runs")

    def add_once(_index: int) -> str:
        try:
            registry.add(_run())
            return "written"
        except FileExistsError:
            return "exists"

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(add_once, range(20)))

    assert results.count("written") == 1
    assert results.count("exists") == 19
    assert registry.read("run-001") == _run()


def test_registry_keeps_post_commit_state_consistent_when_directory_fsync_fails(
    tmp_path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    registry = BenchmarkRunRegistry(tmp_path / "benchmark-runs")
    registry.add(_run("run-001"))

    from videoscope.benchmark import storage

    real_fsync_directory = storage._fsync_directory

    def fail_registry_directory_sync(path):  # type: ignore[no-untyped-def]
        if path == registry.root:
            raise OSError("simulated registry directory fsync failure")
        real_fsync_directory(path)

    monkeypatch.setattr(storage, "_fsync_directory", fail_registry_directory_sync)

    with pytest.raises(BenchmarkDurabilityError, match="committed"):
        registry.add(_run("run-002"))

    assert [entry.run_id for entry in registry.list()] == ["run-001", "run-002"]
    assert registry.read("run-002") == _run("run-002")


def _run_values() -> dict[str, object]:
    run = _run()
    return {
        "schema_version": run.schema_version,
        "run_id": run.run_id,
        "created_at": run.created_at,
        "code_sha": run.code_sha,
        "dataset_revision": run.dataset_revision,
        "model_identities": run.model_identities,
        "index_identities": run.index_identities,
        "config_identities": run.config_identities,
        "hardware": run.hardware,
        "execution_mode": run.execution_mode,
        "metrics": run.metrics,
        "case_outcomes": run.case_outcomes,
    }
