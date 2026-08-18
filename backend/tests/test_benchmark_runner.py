from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
import json

import pytest

from videoscope.benchmark import (
    AssetProvenance,
    BenchmarkAsset,
    BenchmarkDataError,
    BenchmarkDataset,
    BenchmarkInterval,
    BenchmarkRunRegistry,
    ComponentIdentity,
    HardNegative,
    HardwareProfile,
    LocalAssetResolver,
    MetricValue,
    QueryCase,
    dataset_revision,
)
from videoscope.benchmark.runner import (
    BenchmarkExecutionError,
    BenchmarkRunner,
    BenchmarkSearchHit,
    ExecutionIdentities,
    audit_run_manifest,
)


def _asset(asset_id: str, digest: str, duration: float) -> BenchmarkAsset:
    return BenchmarkAsset(
        asset_id=asset_id,
        sha256=digest,
        byte_size=1_024,
        duration_seconds=duration,
        provenance=AssetProvenance(
            source=f"Fixture {asset_id}",
            source_uri=f"https://example.test/{asset_id}.mp4",
            license_id="CC-BY-4.0",
        ),
    )


def _dataset() -> BenchmarkDataset:
    first = _asset("asset-a", "a" * 64, 100.0)
    second = _asset("asset-b", "b" * 64, 80.0)
    return BenchmarkDataset(
        schema_version=1,
        dataset_id="runner-core",
        dataset_version="1.0.0",
        description="Runner fixture",
        assets=(first, second),
        cases=(
            QueryCase(
                case_id="multi-positive",
                query="made basket",
                asset_ids=(first.asset_id,),
                domain="basketball",
                modalities=("sports", "visual"),
                label_quality="gold",
                split_group="match-a",
                relevant_intervals=(
                    BenchmarkInterval(first.asset_id, 10.0, 12.0),
                    BenchmarkInterval(first.asset_id, 40.0, 42.0),
                ),
                hard_negatives=(
                    HardNegative(first.asset_id, 20.0, 22.0, "missed shot"),
                ),
            ),
            QueryCase(
                case_id="negative",
                query="a dunk occurs",
                asset_ids=(second.asset_id,),
                domain="basketball",
                modalities=("sports", "visual"),
                label_quality="silver",
                split_group="match-b",
                relevant_intervals=(),
                hard_negatives=(
                    HardNegative(second.asset_id, 30.0, 32.0, "layup"),
                ),
            ),
            QueryCase(
                case_id="one-positive",
                query="scoreboard changes",
                asset_ids=(second.asset_id,),
                domain="basketball",
                modalities=("ocr",),
                label_quality="gold",
                split_group="match-b",
                relevant_intervals=(
                    BenchmarkInterval(second.asset_id, 50.0, 51.0),
                ),
            ),
        ),
    )


@dataclass(frozen=True)
class FakeRepositoryAsset:
    id: str
    sha256: str
    byte_size: int
    duration_seconds: float
    video_id: str


class FakeAssetRepository:
    def __init__(self, dataset: BenchmarkDataset) -> None:
        self.assets = {
            asset.asset_id: FakeRepositoryAsset(
                id=f"sha256:{asset.sha256}",
                sha256=asset.sha256,
                byte_size=asset.byte_size,
                duration_seconds=asset.duration_seconds,
                video_id=f"video-{asset.asset_id}",
            )
            for asset in dataset.assets
        }

    def find_assets_by_sha256(self, digest: str) -> tuple[FakeRepositoryAsset, ...]:
        return tuple(
            asset
            for asset in self.assets.values()
            if asset.id == f"sha256:{digest}"
        )


class FakeSearchAdapter:
    def __init__(self) -> None:
        self.states: dict[tuple[str, str], str | Exception] = {}
        self.results: dict[str, tuple[BenchmarkSearchHit, ...] | Exception] = {}
        self.calls: list[tuple[str, str, tuple[str, ...]]] = []

    def identities(self, profile) -> ExecutionIdentities:  # type: ignore[no-untyped-def]
        return ExecutionIdentities(
            model_identities=(ComponentIdentity("siglip", "siglip@revision"),),
            index_identities=(
                ComponentIdentity("visual_dense", "generation-7@specification"),
            ),
            config_identities=(
                ComponentIdentity("search_stack", f"stack-for-{profile.profile_id}"),
            ),
        )

    def capability_state(self, profile, asset, capability):  # type: ignore[no-untyped-def]
        del profile
        state = self.states.get((asset.asset_id, capability), "complete")
        if isinstance(state, Exception):
            raise state
        return state

    def search(self, profile, query, assets, *, limit):  # type: ignore[no-untyped-def]
        self.calls.append(
            (profile.profile_id, query, tuple(asset.asset_id for asset in assets))
        )
        result = self.results.get(query, ())
        if isinstance(result, Exception):
            raise result
        return result[:limit]


class DeterministicTimer:
    def __init__(self) -> None:
        self.value = -0.1

    def __call__(self) -> float:
        self.value += 0.1
        return self.value


def _hardware() -> HardwareProfile:
    return HardwareProfile("macOS 15.6", "arm64", "Apple M4 Pro", 24 * 1024**3, "Metal")


def _runner(tmp_path, adapter: FakeSearchAdapter, repository=None) -> BenchmarkRunner:  # type: ignore[no-untyped-def]
    dataset = _dataset()
    repository = repository or FakeAssetRepository(dataset)
    return BenchmarkRunner(
        registry=BenchmarkRunRegistry(tmp_path / "runs"),
        asset_resolver=LocalAssetResolver(repository),
        search=adapter,
        hardware=_hardware(),
        code_sha="c" * 40,
        clock=lambda: datetime(2026, 8, 18, 12, 0, tzinfo=UTC),
        timer=DeterministicTimer(),
    )


def _metric_map(run) -> dict[str, float]:  # type: ignore[no-untyped-def]
    return {metric.name: metric.value for metric in run.metrics}


def _outcome_metric_map(outcome) -> dict[str, float]:  # type: ignore[no-untyped-def]
    return {metric.name: metric.value for metric in outcome.metrics}


def test_runner_scores_multi_interval_positive_and_zero_interval_negative_cases(
    tmp_path,
) -> None:
    adapter = FakeSearchAdapter()
    adapter.results = {
        "made basket": (
            BenchmarkSearchHit("asset-a", 20.0, 22.0, 0.95),
            BenchmarkSearchHit("asset-a", 40.0, 42.0, 0.90),
            BenchmarkSearchHit("asset-a", 70.0, 71.0, 0.50),
        ),
        "a dunk occurs": (),
        "scoreboard changes": (
            BenchmarkSearchHit("asset-b", 50.0, 51.0, 0.80),
        ),
    }
    runner = _runner(tmp_path, adapter)

    run = runner.run(
        _dataset(),
        profile_id="dense_siglip",
        run_id="run-001",
        execution_mode="warm",
    )
    metrics = _metric_map(run)
    outcomes = {outcome.case_id: outcome for outcome in run.case_outcomes}

    assert run.dataset_revision == dataset_revision(_dataset())
    assert run.execution_mode == "warm"
    assert metrics["total_case_count"] == 3
    assert metrics["completed_case_count"] == 3
    assert metrics["error_count"] == 0
    assert metrics["recall_at_1"] == pytest.approx(0.5)
    assert metrics["recall_at_3"] == 1
    assert metrics["recall_at_5"] == 1
    assert metrics["mrr"] == pytest.approx(0.75)
    assert metrics["mean_temporal_iou"] == 1
    assert metrics["false_positive_rate"] == pytest.approx(0.5)
    assert metrics["negative_case_false_positive_rate"] == 0
    assert metrics["hard_negative_hit_rate"] == pytest.approx(0.5)
    assert metrics["mean_latency_ms"] == pytest.approx(100.0)
    assert metrics["slice.modality.visual.recall_at_5"] == 1
    assert metrics["slice.label_quality.gold.recall_at_1"] == pytest.approx(0.5)
    assert outcomes["negative"].status == "complete"
    assert _outcome_metric_map(outcomes["multi-positive"])["reciprocal_rank"] == 0.5
    assert _outcome_metric_map(outcomes["multi-positive"])["hard_negative_hit"] == 1
    assert runner.registry.read("run-001") == run
    assert ComponentIdentity(
        "benchmark_profile",
        "dense_siglip@1",
    ) in run.config_identities
    assert outcomes["multi-positive"].result_count == 3
    assert tuple(
        (item.rank, item.asset_id, item.start_seconds, item.end_seconds, item.score)
        for item in outcomes["multi-positive"].result_evidence
    ) == (
        (1, "asset-a", 20.0, 22.0, 0.95),
        (2, "asset-a", 40.0, 42.0, 0.90),
        (3, "asset-a", 70.0, 71.0, 0.50),
    )
    audit_run_manifest(_dataset(), run)


def test_runner_persists_capability_failures_as_errors_not_model_misses(tmp_path) -> None:
    adapter = FakeSearchAdapter()
    adapter.states[("asset-b", "visual_dense")] = "stale"
    adapter.results["made basket"] = (
        BenchmarkSearchHit("asset-a", 40.0, 42.0, 0.9),
    )
    runner = _runner(tmp_path, adapter)

    run = runner.run(
        _dataset(),
        profile_id="dense_siglip",
        run_id="run-stale",
        execution_mode="cold",
    )
    outcomes = {outcome.case_id: outcome for outcome in run.case_outcomes}
    metrics = _metric_map(run)

    assert outcomes["multi-positive"].status == "complete"
    assert outcomes["negative"].status == "failed"
    assert outcomes["negative"].diagnostic_code == "capability_stale"
    assert outcomes["one-positive"].diagnostic_code == "capability_stale"
    assert metrics["completed_case_count"] == 1
    assert metrics["error_count"] == 2
    assert metrics["recall_at_1"] == 1
    assert "a dunk occurs" not in {call[1] for call in adapter.calls}
    assert runner.registry.read("run-stale") == run


def test_runner_fails_closed_on_missing_required_capability(tmp_path) -> None:
    adapter = FakeSearchAdapter()
    adapter.states[("asset-a", "visual_dense")] = "missing"
    runner = _runner(tmp_path, adapter)

    run = runner.run(
        _dataset(),
        profile_id="dense_siglip",
        run_id="run-missing-capability",
        execution_mode="warm",
    )
    outcome = next(
        item for item in run.case_outcomes if item.case_id == "multi-positive"
    )

    assert outcome.status == "failed"
    assert outcome.diagnostic_code == "capability_missing"


def test_runner_sanitizes_provider_and_capability_exceptions(tmp_path) -> None:
    adapter = FakeSearchAdapter()
    adapter.results["made basket"] = RuntimeError(
        "private model at /Users/person/models/siglip"
    )
    adapter.states[("asset-b", "visual_dense")] = RuntimeError(
        "private index at /Users/person/index"
    )
    runner = _runner(tmp_path, adapter)

    run = runner.run(
        _dataset(),
        profile_id="dense_siglip",
        run_id="run-errors",
        execution_mode="warm",
    )
    diagnostics = {
        outcome.case_id: outcome.diagnostic_code for outcome in run.case_outcomes
    }
    manifest_text = json.dumps(
        {
            "diagnostics": diagnostics,
            "metrics": [metric.name for metric in run.metrics],
        }
    )

    assert diagnostics == {
        "multi-positive": "provider_failed",
        "negative": "capability_check_failed",
        "one-positive": "capability_check_failed",
    }
    assert "/Users/person" not in manifest_text
    assert _metric_map(run)["error_count"] == 3
    assert "recall_at_5" not in _metric_map(run)


def test_runner_persists_asset_resolution_failure_without_searching(tmp_path) -> None:
    dataset = _dataset()
    repository = FakeAssetRepository(dataset)
    repository.assets["asset-b"] = replace(
        repository.assets["asset-b"],
        sha256="d" * 64,
    )
    adapter = FakeSearchAdapter()
    adapter.results["made basket"] = (
        BenchmarkSearchHit("asset-a", 40.0, 42.0, 0.9),
    )
    runner = _runner(tmp_path, adapter, repository)

    run = runner.run(
        dataset,
        profile_id="dense_siglip",
        run_id="run-assets",
        execution_mode="warm",
    )
    outcomes = {outcome.case_id: outcome for outcome in run.case_outcomes}

    assert outcomes["negative"].diagnostic_code == "asset_sha_mismatch"
    assert outcomes["one-positive"].diagnostic_code == "asset_sha_mismatch"
    assert {call[1] for call in adapter.calls} == {"made basket"}


@pytest.mark.parametrize(
    "bad_result",
    [
        BenchmarkSearchHit("outside", 1.0, 2.0, 0.5),
        BenchmarkSearchHit("asset-a", 99.0, 101.0, 0.5),
    ],
)
def test_runner_rejects_out_of_scope_or_out_of_duration_results(
    tmp_path,
    bad_result: BenchmarkSearchHit,
) -> None:
    adapter = FakeSearchAdapter()
    adapter.results["made basket"] = (bad_result,)
    runner = _runner(tmp_path, adapter)

    run = runner.run(
        _dataset(),
        profile_id="dense_siglip",
        run_id=f"run-invalid-{bad_result.asset_id}",
        execution_mode="warm",
    )
    outcome = next(
        item for item in run.case_outcomes if item.case_id == "multi-positive"
    )

    assert outcome.status == "failed"
    assert outcome.diagnostic_code == "invalid_search_result"


def test_runner_requires_a_stable_execution_identity_before_search(tmp_path) -> None:
    class FailingIdentityAdapter(FakeSearchAdapter):
        def identities(self, profile):  # type: ignore[no-untyped-def]
            del profile
            raise RuntimeError("identity unavailable")

    runner = _runner(tmp_path, FailingIdentityAdapter())

    with pytest.raises(BenchmarkExecutionError, match="identity"):
        runner.run(
            _dataset(),
            profile_id="dense_siglip",
            run_id="run-no-identity",
            execution_mode="warm",
        )

    assert runner.registry.list() == ()


def test_execution_identity_requires_model_index_and_config_snapshots() -> None:
    identity = ComponentIdentity("component", "revision")

    with pytest.raises(BenchmarkDataError, match="model_identities"):
        ExecutionIdentities((), (identity,), (identity,))
    with pytest.raises(BenchmarkDataError, match="index_identities"):
        ExecutionIdentities((identity,), (), (identity,))
    with pytest.raises(BenchmarkDataError, match="config_identities"):
        ExecutionIdentities((identity,), (identity,), ())


def test_runner_refuses_an_empty_case_set(tmp_path) -> None:
    dataset = replace(_dataset(), cases=())
    adapter = FakeSearchAdapter()
    runner = _runner(tmp_path, adapter)

    with pytest.raises(BenchmarkExecutionError, match="at least one"):
        runner.run(
            dataset,
            profile_id="dense_siglip",
            run_id="run-empty",
            execution_mode="warm",
        )

    assert adapter.calls == []
    assert runner.registry.list() == ()


def test_runner_refuses_registered_run_id_before_provider_work(tmp_path) -> None:
    adapter = FakeSearchAdapter()
    runner = _runner(tmp_path, adapter)
    runner.run(
        _dataset(),
        profile_id="dense_siglip",
        run_id="run-existing",
        execution_mode="warm",
    )
    calls_before = tuple(adapter.calls)

    with pytest.raises(FileExistsError):
        runner.run(
            _dataset(),
            profile_id="dense_siglip",
            run_id="run-existing",
            execution_mode="warm",
        )

    assert tuple(adapter.calls) == calls_before


def test_runner_bounds_provider_iterables_before_rejecting_excess_results(
    tmp_path,
) -> None:
    class GeneratorAdapter(FakeSearchAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.yield_count = 0

        def search(self, profile, query, assets, *, limit):  # type: ignore[no-untyped-def]
            del profile, query, assets, limit

            def results():  # type: ignore[no-untyped-def]
                for index in range(1_000):
                    self.yield_count += 1
                    yield BenchmarkSearchHit(
                        "asset-a",
                        float(index % 90),
                        float(index % 90 + 0.5),
                        1.0,
                    )

            return results()

    adapter = GeneratorAdapter()
    runner = _runner(tmp_path, adapter)

    run = runner.run(
        _dataset(),
        profile_id="dense_siglip",
        run_id="run-bounded-provider",
        execution_mode="warm",
    )
    outcome = next(
        item for item in run.case_outcomes if item.case_id == "multi-positive"
    )

    assert outcome.diagnostic_code == "invalid_search_result"
    assert adapter.yield_count == 21 + 21 + 21


def test_runner_rejects_duplicate_exact_ranked_hits(tmp_path) -> None:
    adapter = FakeSearchAdapter()
    adapter.results["made basket"] = (
        BenchmarkSearchHit("asset-a", 40.0, 42.0, 0.9),
        BenchmarkSearchHit("asset-a", 40.0, 42.0, 0.8),
    )
    runner = _runner(tmp_path, adapter)

    run = runner.run(
        _dataset(),
        profile_id="dense_siglip",
        run_id="run-duplicate-hit",
        execution_mode="warm",
    )
    outcome = next(
        item for item in run.case_outcomes if item.case_id == "multi-positive"
    )

    assert outcome.diagnostic_code == "invalid_search_result"


def test_audit_recomputes_metrics_from_ranked_evidence(tmp_path) -> None:
    adapter = FakeSearchAdapter()
    adapter.results["made basket"] = (
        BenchmarkSearchHit("asset-a", 40.0, 42.0, 0.9),
    )
    adapter.results["scoreboard changes"] = (
        BenchmarkSearchHit("asset-b", 50.0, 51.0, 0.8),
    )
    run = _runner(tmp_path, adapter).run(
        _dataset(),
        profile_id="dense_siglip",
        run_id="run-audit",
        execution_mode="warm",
    )
    target = next(
        outcome for outcome in run.case_outcomes if outcome.case_id == "multi-positive"
    )
    tampered_outcome = replace(
        target,
        metrics=tuple(
            MetricValue(metric.name, 0.0, metric.unit)
            if metric.name == "reciprocal_rank"
            else metric
            for metric in target.metrics
        ),
    )
    tampered_run = replace(
        run,
        case_outcomes=tuple(
            tampered_outcome if item.case_id == target.case_id else item
            for item in run.case_outcomes
        ),
    )

    with pytest.raises(BenchmarkExecutionError, match="audit"):
        audit_run_manifest(_dataset(), tampered_run)
