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
    RUN_SCHEMA_VERSION,
)
import videoscope.benchmark.comparison as comparison_module
from videoscope.benchmark.comparison import (
    AllowedRunDifferences,
    MetricGuardrail,
    PromotionPolicy,
    _compare_verified_runs as compare_runs,
    compare_runs as compare_promotion_runs,
    comparison_json,
)
from videoscope.benchmark.runner import BenchmarkExecutionError


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
    code_sha: str = "a" * 40,
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
        schema_version=RUN_SCHEMA_VERSION,
        run_id=run_id,
        created_at="2026-08-18T12:30:02Z",
        started_at="2026-08-18T12:30:00Z",
        finished_at="2026-08-18T12:30:01Z",
        run_status=(
            "complete"
            if failed_cases == 0
            else "failed"
            if failed_cases == case_count
            else "partial"
        ),
        code_sha=code_sha,
        dataset_revision=dataset_revision,
        model_identities=(ComponentIdentity("model", "model@1"),),
        index_identities=(ComponentIdentity("index", "generation-1"),),
        config_identities=(
            ComponentIdentity("benchmark_profile", "retrieval-core@1"),
            ComponentIdentity(
                "benchmark_search_plan",
                "evaluation-search-plan@1:stable",
            ),
            ComponentIdentity("benchmark_methodology", "portable-retrieval@1"),
            ComponentIdentity(
                "benchmark_execution_lifecycle",
                f"{execution_mode}:test-cache-policy@1",
            ),
            ComponentIdentity("product_search", "product-search@1"),
        ),
        hardware=HardwareProfile("macOS", "arm64", "Apple M4 Pro", 1024, "Metal"),
        execution_mode=execution_mode,  # type: ignore[arg-type]
        quality_metrics=(
            MetricValue("completed_case_count", case_count - failed_cases, "count"),
            MetricValue("error_count", failed_cases, "count"),
            MetricValue("recall_at_5", recall_at_5, "ratio"),
            MetricValue("false_positive_rate", false_positive_rate, "ratio"),
            MetricValue("mean_latency_ms", mean_latency_ms, "milliseconds"),
        ),
        system_metrics=(),
        measurement_protocol=ComponentIdentity(
            "benchmark_measurement_protocol",
            "not-measured@1",
        ),
        measurement_status="not_measured",
        measurement_started_at=None,
        measurement_finished_at=None,
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


def test_promotion_comparison_audits_both_runs_before_guardrails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        false_positive_rate=0.10,
        mean_latency_ms=90.0,
    )
    dataset = object()
    audited: list[str] = []

    def audit(value: object, run: BenchmarkRunManifest) -> None:
        assert value is dataset
        audited.append(run.run_id)
        if run.run_id == "candidate":
            raise RuntimeError("aggregate metrics mismatch")

    monkeypatch.setattr(comparison_module, "audit_run_manifest", audit)

    with pytest.raises(RuntimeError, match="aggregate metrics mismatch"):
        compare_promotion_runs(dataset, baseline, candidate, _policy())  # type: ignore[arg-type]

    assert audited == ["baseline", "candidate"]


def test_public_comparison_returns_incomparable_before_cross_dataset_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        false_positive_rate=0.10,
        mean_latency_ms=90.0,
        dataset_revision="e" * 64,
    )
    monkeypatch.setattr(
        comparison_module,
        "audit_run_manifest",
        lambda *_: pytest.fail("incompatible runs must not enter metric audit"),
    )

    comparison = compare_promotion_runs(  # type: ignore[arg-type]
        object(),
        baseline,
        candidate,
        _policy(),
    )

    assert comparison.status == "incomparable"
    assert comparison.guardrails == ()


def _replace_identity(
    run: BenchmarkRunManifest,
    field_name: str,
    component_id: str,
    identity_value: str,
) -> BenchmarkRunManifest:
    identities = tuple(
        replace(identity, identity=identity_value)
        if identity.component_id == component_id
        else identity
        for identity in getattr(run, field_name)
    )
    return replace(run, **{field_name: identities})


def _candidate_with_difference(
    candidate: BenchmarkRunManifest,
    difference: str,
) -> BenchmarkRunManifest:
    if difference == "code_sha":
        return replace(candidate, code_sha="b" * 40)
    if difference == "benchmark_profile":
        return _replace_identity(
            candidate,
            "config_identities",
            "benchmark_profile",
            "dense-siglip@1",
        )
    if difference == "benchmark_search_plan":
        return _replace_identity(
            candidate,
            "config_identities",
            "benchmark_search_plan",
            "evaluation-search-plan@1:changed",
        )
    field_name, component_id = {
        "model": ("model_identities", "model"),
        "index": ("index_identities", "index"),
        "config": ("config_identities", "product_search"),
    }[difference]
    return _replace_identity(
        candidate,
        field_name,
        component_id,
        f"{component_id}@changed",
    )


def _allow_difference(difference: str) -> AllowedRunDifferences:
    if difference == "code_sha":
        return AllowedRunDifferences(allow_code_sha_difference=True)
    if difference == "benchmark_profile":
        return AllowedRunDifferences(
            allow_benchmark_profile_difference=True,
        )
    if difference == "benchmark_search_plan":
        return AllowedRunDifferences(
            allow_benchmark_search_plan_difference=True,
        )
    field_name, component_id = {
        "model": ("model_component_ids", "model"),
        "index": ("index_component_ids", "index"),
        "config": ("config_component_ids", "product_search"),
    }[difference]
    return AllowedRunDifferences(**{field_name: (component_id,)})


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
    "difference",
    (
        "code_sha",
        "benchmark_profile",
        "benchmark_search_plan",
        "model",
        "index",
        "config",
    ),
)
def test_comparison_rejects_every_undeclared_execution_identity_difference(
    difference: str,
) -> None:
    baseline = _run(
        "baseline",
        case_count=20,
        recall_at_5=0.8,
        false_positive_rate=0.15,
        mean_latency_ms=100,
    )
    candidate = _candidate_with_difference(
        _run(
            "candidate",
            case_count=20,
            recall_at_5=0.9,
            false_positive_rate=0.1,
            mean_latency_ms=90,
        ),
        difference,
    )

    comparison = compare_runs(baseline, candidate, _policy())

    assert comparison.status == "incomparable"
    assert comparison.guardrails == ()


def test_product_environment_identity_is_always_equal_and_not_allowlistable() -> None:
    baseline = replace(
        _run(
            "baseline",
            case_count=20,
            recall_at_5=0.8,
            false_positive_rate=0.15,
            mean_latency_ms=100,
        ),
        config_identities=(
            *_run(
                "identity-source-a",
                case_count=20,
                recall_at_5=0.8,
                false_positive_rate=0.15,
                mean_latency_ms=100,
            ).config_identities,
            ComponentIdentity(
                "benchmark_product_environment",
                f"benchmark-product-environment@1:{'a' * 64}",
            ),
        ),
    )
    candidate = replace(
        baseline,
        run_id="candidate",
        config_identities=tuple(
            replace(
                identity,
                identity=f"benchmark-product-environment@1:{'b' * 64}",
            )
            if identity.component_id == "benchmark_product_environment"
            else identity
            for identity in baseline.config_identities
        ),
    )

    assert compare_runs(baseline, candidate, _policy()).status == "incomparable"
    with pytest.raises(BenchmarkDataError, match="reserved"):
        AllowedRunDifferences(
            config_component_ids=("benchmark_product_environment",),
        )


@pytest.mark.parametrize(
    "difference",
    (
        "code_sha",
        "benchmark_profile",
        "benchmark_search_plan",
        "model",
        "index",
        "config",
    ),
)
def test_comparison_allows_only_an_explicitly_declared_identity_difference(
    difference: str,
) -> None:
    baseline = _run(
        "baseline",
        case_count=20,
        recall_at_5=0.8,
        false_positive_rate=0.15,
        mean_latency_ms=100,
    )
    candidate = _candidate_with_difference(
        _run(
            "candidate",
            case_count=20,
            recall_at_5=0.9,
            false_positive_rate=0.1,
            mean_latency_ms=90,
        ),
        difference,
    )
    policy = replace(
        _policy(),
        allowed_differences=_allow_difference(difference),
    )

    comparison = compare_runs(baseline, candidate, policy)

    assert comparison.status == "eligible"


@pytest.mark.parametrize(
    ("field_name", "extra_identity", "allowed_differences"),
    (
        (
            "model_identities",
            ComponentIdentity("added_model", "added@1"),
            AllowedRunDifferences(model_component_ids=("added_model",)),
        ),
        (
            "index_identities",
            ComponentIdentity("added_index", "added@1"),
            AllowedRunDifferences(index_component_ids=("added_index",)),
        ),
        (
            "config_identities",
            ComponentIdentity("added_config", "added@1"),
            AllowedRunDifferences(config_component_ids=("added_config",)),
        ),
    ),
)
def test_comparison_allows_only_explicitly_allowlisted_component_membership_changes(
    field_name: str,
    extra_identity: ComponentIdentity,
    allowed_differences: AllowedRunDifferences,
) -> None:
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
        **{field_name: (*getattr(candidate, field_name), extra_identity)},
    )

    comparison = compare_runs(
        baseline,
        candidate,
        replace(_policy(), allowed_differences=allowed_differences),
    )

    assert comparison.status == "eligible"


@pytest.mark.parametrize(
    ("field_name", "extra_identity", "allowed_differences"),
    (
        (
            "model_identities",
            ComponentIdentity("removed_model", "removed@1"),
            AllowedRunDifferences(model_component_ids=("removed_model",)),
        ),
        (
            "index_identities",
            ComponentIdentity("removed_index", "removed@1"),
            AllowedRunDifferences(index_component_ids=("removed_index",)),
        ),
        (
            "config_identities",
            ComponentIdentity("removed_config", "removed@1"),
            AllowedRunDifferences(config_component_ids=("removed_config",)),
        ),
    ),
)
def test_comparison_allows_explicitly_allowlisted_component_removal(
    field_name: str,
    extra_identity: ComponentIdentity,
    allowed_differences: AllowedRunDifferences,
) -> None:
    baseline = _run(
        "baseline",
        case_count=20,
        recall_at_5=0.8,
        false_positive_rate=0.15,
        mean_latency_ms=100,
    )
    baseline = replace(
        baseline,
        **{field_name: (*getattr(baseline, field_name), extra_identity)},
    )
    candidate = _run(
        "candidate",
        case_count=20,
        recall_at_5=0.9,
        false_positive_rate=0.1,
        mean_latency_ms=90,
    )

    comparison = compare_runs(
        baseline,
        candidate,
        replace(_policy(), allowed_differences=allowed_differences),
    )

    assert comparison.status == "eligible"


@pytest.mark.parametrize(
    ("field_name", "extra_identity"),
    (
        ("model_identities", ComponentIdentity("added_model", "added@1")),
        ("index_identities", ComponentIdentity("added_index", "added@1")),
        ("config_identities", ComponentIdentity("added_config", "added@1")),
    ),
)
def test_comparison_rejects_undeclared_component_membership_changes(
    field_name: str,
    extra_identity: ComponentIdentity,
) -> None:
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
        **{field_name: (*getattr(candidate, field_name), extra_identity)},
    )

    comparison = compare_runs(baseline, candidate, _policy())

    assert comparison.status == "incomparable"
    assert "component" in comparison.caveat.lower()


def test_policy_can_declare_a_cumulative_selected_only_profile_ablation() -> None:
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
    changed_candidate = _candidate_with_difference(
        _candidate_with_difference(candidate, "benchmark_profile"),
        "benchmark_search_plan",
    )
    candidate = replace(
        changed_candidate,
        model_identities=(
            *changed_candidate.model_identities,
            ComponentIdentity("siglip", "siglip@revision"),
        ),
        index_identities=(
            *changed_candidate.index_identities,
            ComponentIdentity("visual_dense", "visual-generation@1"),
        ),
        config_identities=(
            *changed_candidate.config_identities,
            ComponentIdentity("visual_search", "visual-search@1"),
        ),
    )
    policy = replace(
        _policy(),
        allowed_differences=AllowedRunDifferences(
            allow_benchmark_profile_difference=True,
            allow_benchmark_search_plan_difference=True,
            model_component_ids=("siglip",),
            index_component_ids=("visual_dense",),
            config_component_ids=("visual_search",),
        ),
    )

    comparison = compare_runs(baseline, candidate, policy)

    assert comparison.status == "eligible"


@pytest.mark.parametrize(
    ("candidate", "expected_status"),
    [
        (
            _run(
                "candidate-regression",
                case_count=20,
                recall_at_5=0.70,
                false_positive_rate=0.15,
                mean_latency_ms=100.0,
            ),
            "reject",
        ),
        (
            _run(
                "candidate-errors",
                case_count=20,
                recall_at_5=0.85,
                false_positive_rate=0.15,
                mean_latency_ms=100.0,
                failed_cases=1,
            ),
            "incomparable",
        ),
    ],
)
def test_regression_rejects_and_cohort_changing_error_is_incomparable(
    candidate: BenchmarkRunManifest,
    expected_status: str,
) -> None:
    baseline = _run(
        "baseline",
        case_count=20,
        recall_at_5=0.80,
        false_positive_rate=0.15,
        mean_latency_ms=100.0,
    )

    comparison = compare_runs(baseline, candidate, _policy())

    assert comparison.status == expected_status


def test_comparison_requires_the_exact_same_completed_case_cohort() -> None:
    baseline = _run(
        "baseline",
        case_count=20,
        failed_cases=1,
        recall_at_5=0.8,
        false_positive_rate=0.15,
        mean_latency_ms=100,
    )
    candidate = _run(
        "candidate",
        case_count=20,
        failed_cases=1,
        recall_at_5=0.9,
        false_positive_rate=0.1,
        mean_latency_ms=90,
    )
    first = candidate.case_outcomes[0]
    last = candidate.case_outcomes[-1]
    candidate = replace(
        candidate,
        case_outcomes=(
            replace(
                first,
                status="complete",
                diagnostic_code=None,
            ),
            *candidate.case_outcomes[1:-1],
            replace(
                last,
                status="failed",
                diagnostic_code="provider_failed",
            ),
        ),
    )
    policy = replace(_policy(), require_no_errors=False)

    comparison = compare_runs(baseline, candidate, policy)

    assert comparison.status == "incomparable"
    assert "completed" in comparison.caveat.lower()
    assert "cohort" in comparison.caveat.lower()


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
        quality_metrics=tuple(
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


def test_comparison_json_surfaces_both_code_shas_when_change_is_allowed() -> None:
    baseline = _run(
        "baseline",
        case_count=20,
        recall_at_5=0.8,
        false_positive_rate=0.15,
        mean_latency_ms=100,
        code_sha="a" * 40,
    )
    candidate = _run(
        "candidate",
        case_count=20,
        recall_at_5=0.9,
        false_positive_rate=0.1,
        mean_latency_ms=90,
        code_sha="b" * 40,
    )
    policy = replace(
        _policy(),
        allowed_differences=AllowedRunDifferences(
            allow_code_sha_difference=True,
        ),
    )

    value = comparison_json(compare_runs(baseline, candidate, policy))

    assert '"baseline_code_sha":"' + "a" * 40 + '"' in value
    assert '"candidate_code_sha":"' + "b" * 40 + '"' in value
    assert '"code_sha_difference_allowed":true' in value


def test_promotion_policy_requires_unique_explicit_guardrails() -> None:
    guardrail = MetricGuardrail("recall_at_5", "higher_is_better")

    with pytest.raises(BenchmarkDataError, match="unique"):
        PromotionPolicy("policy", 10, True, (guardrail, guardrail))
    with pytest.raises(BenchmarkDataError):
        PromotionPolicy("policy", 0, True, (guardrail,))
    with pytest.raises(BenchmarkDataError):
        PromotionPolicy("policy", 10, True, ())


def test_allowed_run_differences_are_canonical_bounded_and_cannot_override_reserved_ids(
) -> None:
    canonical = AllowedRunDifferences(
        model_component_ids=("z_model", "a_model"),
        index_component_ids=("z_index", "a_index"),
        config_component_ids=("z_config", "a_config"),
    )

    assert canonical.model_component_ids == ("a_model", "z_model")
    assert canonical.index_component_ids == ("a_index", "z_index")
    assert canonical.config_component_ids == ("a_config", "z_config")
    with pytest.raises(BenchmarkDataError, match="unique"):
        AllowedRunDifferences(model_component_ids=("model", "model"))
    with pytest.raises(BenchmarkDataError, match="reserved"):
        AllowedRunDifferences(
            config_component_ids=("benchmark_execution_lifecycle",),
        )
    with pytest.raises(BenchmarkDataError, match="reserved"):
        AllowedRunDifferences(config_component_ids=("benchmark_methodology",))
    with pytest.raises(BenchmarkDataError, match="reserved"):
        AllowedRunDifferences(
            model_component_ids=("benchmark_execution_lifecycle",),
        )
    with pytest.raises(BenchmarkDataError, match="boolean"):
        AllowedRunDifferences(allow_code_sha_difference=1)  # type: ignore[arg-type]


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
        quality_metrics=tuple(
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


def test_comparison_requires_the_same_attested_execution_lifecycle() -> None:
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
    mismatched = replace(
        candidate,
        config_identities=tuple(
            replace(identity, identity="warm:different-cache-policy@1")
            if identity.component_id == "benchmark_execution_lifecycle"
            else identity
            for identity in candidate.config_identities
        ),
    )
    missing = replace(
        candidate,
        config_identities=tuple(
            identity
            for identity in candidate.config_identities
            if identity.component_id != "benchmark_execution_lifecycle"
        ),
    )

    assert compare_runs(baseline, mismatched, _policy()).status == "incomparable"
    assert compare_runs(baseline, missing, _policy()).status == "incomparable"


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


def test_comparison_requires_declared_search_plan_and_explicit_ablation_permission() -> None:
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

    changed = _replace_identity(
        candidate,
        "config_identities",
        "benchmark_search_plan",
        "evaluation-search-plan@1:ablation",
    )

    assert compare_runs(baseline, changed, _policy()).status == "incomparable"
    allowed = replace(
        _policy(),
        allowed_differences=AllowedRunDifferences(
            allow_benchmark_search_plan_difference=True,
        ),
    )
    assert compare_runs(baseline, changed, allowed).status == "eligible"

    candidate = replace(
        candidate,
        config_identities=tuple(
            identity
            for identity in candidate.config_identities
            if identity.component_id != "benchmark_search_plan"
        ),
    )
    comparison = compare_runs(baseline, candidate, allowed)

    assert comparison.status == "incomparable"
    assert "search-plan" in comparison.caveat


def test_comparison_requires_same_measurement_protocol_and_status() -> None:
    baseline = replace(
        _run(
        "baseline",
        case_count=20,
        recall_at_5=0.8,
        false_positive_rate=0.15,
            mean_latency_ms=100,
        ),
        measurement_protocol=ComponentIdentity(
            "benchmark_measurement_protocol",
            "process-tree-sampling@1",
        ),
        measurement_status="complete",
        measurement_started_at="2026-08-18T12:30:00Z",
        measurement_finished_at="2026-08-18T12:30:01Z",
        system_metrics=(MetricValue("peak_rss_bytes", 1_000, "bytes"),),
        measurement_evidence_status="legacy_unavailable",
    )
    candidate = replace(
        _run(
        "candidate",
        case_count=20,
        recall_at_5=0.9,
        false_positive_rate=0.1,
            mean_latency_ms=90,
        ),
        measurement_protocol=baseline.measurement_protocol,
        measurement_status="complete",
        measurement_started_at="2026-08-18T12:30:00Z",
        measurement_finished_at="2026-08-18T12:30:01Z",
        system_metrics=(MetricValue("peak_rss_bytes", 900, "bytes"),),
        measurement_evidence_status="legacy_unavailable",
    )

    protocol_mismatch = replace(
        candidate,
        measurement_protocol=ComponentIdentity(
            "benchmark_measurement_protocol",
            "different-protocol@1",
        ),
    )
    status_mismatch = replace(
        candidate,
        measurement_status="failed",
        system_metrics=(),
        measurement_evidence_status="not_applicable",
    )

    assert compare_runs(baseline, protocol_mismatch, _policy()).status == "incomparable"
    assert compare_runs(baseline, status_mismatch, _policy()).status == "incomparable"


def test_legacy_unmeasured_runs_are_readable_but_not_promotion_comparable() -> None:
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
    legacy_protocol = ComponentIdentity(
        "benchmark_measurement_protocol",
        "legacy-unmeasured@1",
    )

    comparison = compare_runs(
        replace(baseline, measurement_protocol=legacy_protocol),
        replace(candidate, measurement_protocol=legacy_protocol),
        _policy(),
    )

    assert comparison.status == "incomparable"
    assert "Legacy v1" in comparison.caveat


def test_verified_system_metrics_remain_descriptively_addressable() -> None:
    baseline = replace(
        _run(
            "baseline",
            case_count=20,
            recall_at_5=0.8,
            false_positive_rate=0.15,
            mean_latency_ms=100,
        ),
        measurement_status="complete",
        measurement_protocol=ComponentIdentity(
            "benchmark_measurement_protocol",
            "process-tree-sampling@1",
        ),
        measurement_started_at="2026-08-18T12:30:00Z",
        measurement_finished_at="2026-08-18T12:30:01Z",
        system_metrics=(
            MetricValue("sampled_peak_process_tree_rss_bytes", 1_000, "bytes"),
        ),
        measurement_evidence_status="legacy_unavailable",
    )
    candidate = replace(
        _run(
            "candidate",
            case_count=20,
            recall_at_5=0.9,
            false_positive_rate=0.1,
            mean_latency_ms=90,
        ),
        measurement_status="complete",
        measurement_protocol=ComponentIdentity(
            "benchmark_measurement_protocol",
            "process-tree-sampling@1",
        ),
        measurement_started_at="2026-08-18T12:30:00Z",
        measurement_finished_at="2026-08-18T12:30:01Z",
        system_metrics=(
            MetricValue("sampled_peak_process_tree_rss_bytes", 900, "bytes"),
        ),
        measurement_evidence_status="legacy_unavailable",
    )
    policy = PromotionPolicy(
        "memory",
        10,
        True,
        (
            MetricGuardrail(
                "sampled_peak_process_tree_rss_bytes",
                "lower_is_better",
            ),
        ),
    )

    comparison = compare_runs(baseline, candidate, policy)

    assert comparison.status == "eligible"
    assert comparison.guardrails[0].candidate_value == 900

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(comparison_module, "audit_run_manifest", lambda *_: None)
    try:
        with pytest.raises(
            BenchmarkExecutionError,
            match="cannot audit observed system metric guardrails",
        ):
            compare_promotion_runs(
                object(),  # type: ignore[arg-type]
                baseline,
                candidate,
                policy,
            )
    finally:
        monkeypatch.undo()
