from __future__ import annotations

from dataclasses import replace

import pytest

from videoscope.benchmark import (
    BenchmarkCaseOutcome,
    BenchmarkDataError,
    BenchmarkRunManifest,
    ComponentIdentity,
    HardwareProfile,
    MetricValue,
)
from videoscope.benchmark.comparison import (
    MetricGuardrail,
    PromotionPolicy,
    compare_runs,
    comparison_json,
)


def _run(
    run_id: str,
    *,
    case_count: int,
    recall_at_5: float,
    false_positive_rate: float,
    mean_latency_ms: float,
    failed_cases: int = 0,
    dataset_revision: str = "d" * 64,
    execution_mode: str = "warm",
) -> BenchmarkRunManifest:
    outcomes = tuple(
        BenchmarkCaseOutcome(
            case_id=f"case-{index:03d}",
            status="failed" if index < failed_cases else "complete",
            latency_ms=10.0,
            result_count=0,
            diagnostic_code="provider_failed" if index < failed_cases else None,
        )
        for index in range(case_count)
    )
    return BenchmarkRunManifest(
        schema_version=1,
        run_id=run_id,
        created_at="2026-08-18T12:30:00Z",
        code_sha="a" * 40,
        dataset_revision=dataset_revision,
        model_identities=(ComponentIdentity("model", f"model-for-{run_id}"),),
        index_identities=(ComponentIdentity("index", "generation-1"),),
        config_identities=(
            ComponentIdentity("benchmark_profile", f"{run_id}@1"),
            ComponentIdentity("benchmark_methodology", "portable-retrieval@1"),
        ),
        hardware=HardwareProfile("macOS", "arm64", "Apple M4 Pro", 1024, "Metal"),
        execution_mode=execution_mode,  # type: ignore[arg-type]
        metrics=(
            MetricValue("completed_case_count", case_count - failed_cases, "count"),
            MetricValue("error_count", failed_cases, "count"),
            MetricValue("recall_at_5", recall_at_5, "ratio"),
            MetricValue("false_positive_rate", false_positive_rate, "ratio"),
            MetricValue("mean_latency_ms", mean_latency_ms, "milliseconds"),
        ),
        case_outcomes=outcomes,
    )


def _policy(minimum_cases: int = 10) -> PromotionPolicy:
    return PromotionPolicy(
        policy_id="retrieval-core",
        minimum_completed_cases=minimum_cases,
        require_no_errors=True,
        guardrails=(
            MetricGuardrail(
                "recall_at_5",
                "higher_is_better",
                allowed_regression=0.0,
                absolute_threshold=0.80,
            ),
            MetricGuardrail(
                "false_positive_rate",
                "lower_is_better",
                allowed_regression=0.02,
                absolute_threshold=0.20,
            ),
            MetricGuardrail(
                "mean_latency_ms",
                "lower_is_better",
                allowed_regression=10.0,
                absolute_threshold=200.0,
            ),
        ),
    )


def test_small_sample_never_claims_promotion_or_statistical_significance() -> None:
    baseline = _run(
        "baseline",
        case_count=3,
        recall_at_5=0.80,
        false_positive_rate=0.15,
        mean_latency_ms=100.0,
    )
    candidate = _run(
        "candidate",
        case_count=3,
        recall_at_5=1.0,
        false_positive_rate=0.0,
        mean_latency_ms=90.0,
    )

    comparison = compare_runs(baseline, candidate, _policy(minimum_cases=10))

    assert comparison.status == "insufficient_evidence"
    assert comparison.completed_case_count == 3
    assert comparison.statistical_significance == "not_assessed"
    assert "3" in comparison.caveat
    assert all(check.passed for check in comparison.guardrails)


def test_candidate_is_only_eligible_after_explicit_guardrails_pass() -> None:
    baseline = _run(
        "baseline",
        case_count=20,
        recall_at_5=0.80,
        false_positive_rate=0.15,
        mean_latency_ms=100.0,
    )
    candidate = _run(
        "candidate",
        case_count=20,
        recall_at_5=0.85,
        false_positive_rate=0.16,
        mean_latency_ms=108.0,
    )

    comparison = compare_runs(baseline, candidate, _policy())

    assert comparison.status == "eligible"
    assert comparison.statistical_significance == "not_assessed"
    assert "guardrails" in comparison.caveat.lower()
    assert all(check.passed for check in comparison.guardrails)


@pytest.mark.parametrize(
    "candidate",
    [
        _run(
            "candidate-regression",
            case_count=20,
            recall_at_5=0.70,
            false_positive_rate=0.15,
            mean_latency_ms=100.0,
        ),
        _run(
            "candidate-errors",
            case_count=20,
            recall_at_5=0.85,
            false_positive_rate=0.15,
            mean_latency_ms=100.0,
            failed_cases=1,
        ),
    ],
)
def test_regression_or_infrastructure_error_rejects_candidate(
    candidate: BenchmarkRunManifest,
) -> None:
    baseline = _run(
        "baseline",
        case_count=20,
        recall_at_5=0.80,
        false_positive_rate=0.15,
        mean_latency_ms=100.0,
    )

    comparison = compare_runs(baseline, candidate, _policy())

    assert comparison.status == "reject"


@pytest.mark.parametrize(
    "candidate",
    [
        _run(
            "different-dataset",
            case_count=20,
            recall_at_5=0.9,
            false_positive_rate=0.1,
            mean_latency_ms=90,
            dataset_revision="e" * 64,
        ),
        _run(
            "different-mode",
            case_count=20,
            recall_at_5=0.9,
            false_positive_rate=0.1,
            mean_latency_ms=90,
            execution_mode="cold",
        ),
    ],
)
def test_incompatible_runs_are_not_compared(candidate: BenchmarkRunManifest) -> None:
    baseline = _run(
        "baseline",
        case_count=20,
        recall_at_5=0.8,
        false_positive_rate=0.15,
        mean_latency_ms=100,
    )

    comparison = compare_runs(baseline, candidate, _policy())

    assert comparison.status == "incomparable"
    assert comparison.guardrails == ()


def test_missing_required_metric_fails_its_guardrail() -> None:
    baseline = _run(
        "baseline",
        case_count=20,
        recall_at_5=0.8,
        false_positive_rate=0.15,
        mean_latency_ms=100,
    )
    candidate = _run(
        "candidate",
        case_count=20,
        recall_at_5=0.85,
        false_positive_rate=0.1,
        mean_latency_ms=90,
    )
    candidate = replace(
        candidate,
        metrics=tuple(
            metric for metric in candidate.metrics if metric.name != "recall_at_5"
        ),
    )

    comparison = compare_runs(baseline, candidate, _policy())
    recall_check = next(
        check for check in comparison.guardrails if check.metric_name == "recall_at_5"
    )

    assert comparison.status == "reject"
    assert recall_check.passed is False
    assert recall_check.reason == "missing_metric"


def test_comparison_json_is_deterministic_for_guardrail_order() -> None:
    baseline = _run(
        "baseline",
        case_count=20,
        recall_at_5=0.8,
        false_positive_rate=0.15,
        mean_latency_ms=100,
    )
    candidate = _run(
        "candidate",
        case_count=20,
        recall_at_5=0.85,
        false_positive_rate=0.1,
        mean_latency_ms=90,
    )
    policy = _policy()
    reordered = replace(policy, guardrails=tuple(reversed(policy.guardrails)))

    first = comparison_json(compare_runs(baseline, candidate, policy))
    second = comparison_json(compare_runs(baseline, candidate, reordered))

    assert first == second
    assert '"statistical_significance":"not_assessed"' in first
    assert '"minimum_completed_cases":10' in first
    assert '"allowed_regression":0.0' in first
    assert '"absolute_threshold":0.8' in first


def test_promotion_policy_requires_unique_explicit_guardrails() -> None:
    guardrail = MetricGuardrail("recall_at_5", "higher_is_better")

    with pytest.raises(BenchmarkDataError, match="unique"):
        PromotionPolicy("policy", 10, True, (guardrail, guardrail))
    with pytest.raises(BenchmarkDataError):
        PromotionPolicy("policy", 0, True, (guardrail,))
    with pytest.raises(BenchmarkDataError):
        PromotionPolicy("policy", 10, True, ())


def test_comparison_rejects_metric_unit_mismatch() -> None:
    baseline = _run(
        "baseline",
        case_count=20,
        recall_at_5=0.8,
        false_positive_rate=0.15,
        mean_latency_ms=100,
    )
    candidate = _run(
        "candidate",
        case_count=20,
        recall_at_5=0.85,
        false_positive_rate=0.1,
        mean_latency_ms=90,
    )
    candidate = replace(
        candidate,
        metrics=tuple(
            MetricValue(metric.name, metric.value, "seconds")
            if metric.name == "mean_latency_ms"
            else metric
            for metric in candidate.metrics
        ),
    )

    comparison = compare_runs(baseline, candidate, _policy())
    latency_check = next(
        check for check in comparison.guardrails if check.metric_name == "mean_latency_ms"
    )

    assert comparison.status == "reject"
    assert latency_check.reason == "unit_mismatch"


def test_comparison_requires_same_benchmark_methodology() -> None:
    baseline = _run(
        "baseline",
        case_count=20,
        recall_at_5=0.8,
        false_positive_rate=0.15,
        mean_latency_ms=100,
    )
    candidate = _run(
        "candidate",
        case_count=20,
        recall_at_5=0.9,
        false_positive_rate=0.1,
        mean_latency_ms=90,
    )
    candidate = replace(
        candidate,
        config_identities=tuple(
            replace(identity, identity="portable-retrieval@2")
            if identity.component_id == "benchmark_methodology"
            else identity
            for identity in candidate.config_identities
        ),
    )

    comparison = compare_runs(baseline, candidate, _policy())

    assert comparison.status == "incomparable"
    assert "methodolog" in comparison.caveat.lower()


def test_comparison_fails_closed_when_both_methodology_identities_are_missing() -> None:
    baseline = _run(
        "baseline",
        case_count=20,
        recall_at_5=0.8,
        false_positive_rate=0.15,
        mean_latency_ms=100,
    )
    candidate = _run(
        "candidate",
        case_count=20,
        recall_at_5=0.9,
        false_positive_rate=0.1,
        mean_latency_ms=90,
    )
    baseline = replace(
        baseline,
        config_identities=tuple(
            identity
            for identity in baseline.config_identities
            if identity.component_id != "benchmark_methodology"
        ),
    )
    candidate = replace(
        candidate,
        config_identities=tuple(
            identity
            for identity in candidate.config_identities
            if identity.component_id != "benchmark_methodology"
        ),
    )

    comparison = compare_runs(baseline, candidate, _policy())

    assert comparison.status == "incomparable"
