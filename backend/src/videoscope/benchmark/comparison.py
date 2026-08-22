from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from typing import Literal

from .schema import (
    LEGACY_UNMEASURED_PROTOCOL_IDENTITY,
    BenchmarkDataError,
    BenchmarkDataset,
    BenchmarkRunManifest,
    ComponentIdentity,
    MetricValue,
    _require_finite_float,
    _require_id,
    _require_positive_int,
    _require_string,
    _require_tuple,
)
from .runner import BenchmarkExecutionError, audit_run_manifest


GuardrailDirection = Literal["higher_is_better", "lower_is_better"]
ComparisonStatus = Literal[
    "eligible",
    "reject",
    "insufficient_evidence",
    "incomparable",
]

MAX_ALLOWED_COMPONENT_IDS = 256
_RESERVED_CONFIG_COMPONENT_IDS = frozenset(
    {
        "benchmark_execution_lifecycle",
        "benchmark_measurement_protocol",
        "benchmark_methodology",
        "benchmark_product_environment",
        "benchmark_profile",
        "benchmark_search_plan",
    }
)


def _canonical_component_ids(
    values: object,
    field_name: str,
    *,
    reject_reserved: bool = False,
) -> tuple[str, ...]:
    resolved = _require_tuple(values, field_name)
    if len(resolved) > MAX_ALLOWED_COMPONENT_IDS:
        raise BenchmarkDataError(f"{field_name} exceeds the supported limit")
    for value in resolved:
        _require_id(value, field_name)
    if len(resolved) != len(set(resolved)):
        raise BenchmarkDataError(f"{field_name} must contain unique component IDs")
    if reject_reserved and set(resolved) & _RESERVED_CONFIG_COMPONENT_IDS:
        raise BenchmarkDataError(
            f"{field_name} must not contain reserved benchmark component IDs"
        )
    return tuple(sorted(resolved))


@dataclass(frozen=True, slots=True)
class AllowedRunDifferences:
    """The only execution identity values a policy permits to change."""

    allow_code_sha_difference: bool = False
    allow_benchmark_profile_difference: bool = False
    allow_benchmark_search_plan_difference: bool = False
    model_component_ids: tuple[str, ...] = ()
    index_component_ids: tuple[str, ...] = ()
    config_component_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "allow_code_sha_difference",
            "allow_benchmark_profile_difference",
            "allow_benchmark_search_plan_difference",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise BenchmarkDataError(
                    f"allowed differences {field_name} must be boolean"
                )
        for field_name in (
            "model_component_ids",
            "index_component_ids",
            "config_component_ids",
        ):
            object.__setattr__(
                self,
                field_name,
                _canonical_component_ids(
                    getattr(self, field_name),
                    f"allowed differences {field_name}",
                    reject_reserved=True,
                ),
            )


@dataclass(frozen=True, slots=True)
class MetricGuardrail:
    metric_name: str
    direction: GuardrailDirection
    allowed_regression: float = 0.0
    absolute_threshold: float | None = None

    def __post_init__(self) -> None:
        _require_id(self.metric_name, "guardrail metric_name")
        if self.direction not in {"higher_is_better", "lower_is_better"}:
            raise BenchmarkDataError(
                "guardrail direction must be higher_is_better or lower_is_better"
            )
        regression = _require_finite_float(
            self.allowed_regression,
            "guardrail allowed_regression",
            minimum=0,
        )
        object.__setattr__(self, "allowed_regression", regression)
        if self.absolute_threshold is not None:
            threshold = _require_finite_float(
                self.absolute_threshold,
                "guardrail absolute_threshold",
            )
            object.__setattr__(self, "absolute_threshold", threshold)


@dataclass(frozen=True, slots=True)
class PromotionPolicy:
    policy_id: str
    minimum_completed_cases: int
    require_no_errors: bool
    guardrails: tuple[MetricGuardrail, ...]
    allowed_differences: AllowedRunDifferences = field(
        default_factory=AllowedRunDifferences
    )

    def __post_init__(self) -> None:
        _require_id(self.policy_id, "promotion policy id")
        _require_positive_int(
            self.minimum_completed_cases,
            "promotion policy minimum_completed_cases",
        )
        if type(self.require_no_errors) is not bool:
            raise BenchmarkDataError("promotion policy require_no_errors must be boolean")
        guardrails = _require_tuple(self.guardrails, "promotion policy guardrails")
        if not guardrails:
            raise BenchmarkDataError("promotion policy guardrails must not be empty")
        if any(not isinstance(item, MetricGuardrail) for item in guardrails):
            raise BenchmarkDataError(
                "promotion policy guardrails must contain MetricGuardrail values"
            )
        names = tuple(item.metric_name for item in guardrails)
        if len(names) != len(set(names)):
            raise BenchmarkDataError(
                "promotion policy guardrails must contain unique metric names"
            )
        if not isinstance(self.allowed_differences, AllowedRunDifferences):
            raise BenchmarkDataError(
                "promotion policy allowed_differences must be explicit"
            )


@dataclass(frozen=True, slots=True)
class GuardrailCheck:
    metric_name: str
    direction: GuardrailDirection
    allowed_regression: float
    absolute_threshold: float | None
    baseline_value: float | None
    candidate_value: float | None
    delta: float | None
    passed: bool
    reason: str

    def __post_init__(self) -> None:
        _require_id(self.metric_name, "guardrail check metric_name")
        if self.direction not in {"higher_is_better", "lower_is_better"}:
            raise BenchmarkDataError("guardrail check direction is invalid")
        _require_finite_float(
            self.allowed_regression,
            "guardrail check allowed_regression",
            minimum=0,
        )
        if self.absolute_threshold is not None:
            _require_finite_float(
                self.absolute_threshold,
                "guardrail check absolute_threshold",
            )
        for field_name in ("baseline_value", "candidate_value", "delta"):
            value = getattr(self, field_name)
            if value is not None and not math.isfinite(value):
                raise BenchmarkDataError(f"guardrail check {field_name} must be finite")
        if type(self.passed) is not bool:
            raise BenchmarkDataError("guardrail check passed must be boolean")
        _require_id(self.reason, "guardrail check reason")


@dataclass(frozen=True, slots=True)
class RunComparison:
    baseline_run_id: str
    candidate_run_id: str
    baseline_code_sha: str
    candidate_code_sha: str
    code_sha_difference_allowed: bool
    allowed_differences: AllowedRunDifferences
    policy_id: str
    minimum_completed_cases: int
    require_no_errors: bool
    status: ComparisonStatus
    completed_case_count: int
    statistical_significance: Literal["not_assessed"]
    caveat: str
    guardrails: tuple[GuardrailCheck, ...]

    def __post_init__(self) -> None:
        _require_id(self.baseline_run_id, "comparison baseline_run_id")
        _require_id(self.candidate_run_id, "comparison candidate_run_id")
        _require_string(
            self.baseline_code_sha,
            "comparison baseline_code_sha",
            max_length=64,
        )
        _require_string(
            self.candidate_code_sha,
            "comparison candidate_code_sha",
            max_length=64,
        )
        if type(self.code_sha_difference_allowed) is not bool:
            raise BenchmarkDataError(
                "comparison code_sha_difference_allowed must be boolean"
            )
        if not isinstance(self.allowed_differences, AllowedRunDifferences):
            raise BenchmarkDataError(
                "comparison allowed_differences must be explicit"
            )
        _require_id(self.policy_id, "comparison policy_id")
        _require_positive_int(
            self.minimum_completed_cases,
            "comparison minimum_completed_cases",
        )
        if type(self.require_no_errors) is not bool:
            raise BenchmarkDataError("comparison require_no_errors must be boolean")
        if self.status not in {
            "eligible",
            "reject",
            "insufficient_evidence",
            "incomparable",
        }:
            raise BenchmarkDataError("comparison status is invalid")
        if type(self.completed_case_count) is not int or self.completed_case_count < 0:
            raise BenchmarkDataError(
                "comparison completed_case_count must be non-negative"
            )
        if self.statistical_significance != "not_assessed":
            raise BenchmarkDataError(
                "comparison statistical significance is not assessed by this core"
            )
        _require_string(self.caveat, "comparison caveat", max_length=2_000)
        guardrails = _require_tuple(self.guardrails, "comparison guardrails")
        if any(not isinstance(item, GuardrailCheck) for item in guardrails):
            raise BenchmarkDataError(
                "comparison guardrails must contain GuardrailCheck values"
            )


def compare_runs(
    dataset: BenchmarkDataset,
    baseline: BenchmarkRunManifest,
    candidate: BenchmarkRunManifest,
    policy: PromotionPolicy,
) -> RunComparison:
    """Audit portable evidence before evaluating promotion guardrails."""

    # A deterministic incompatibility can never promote either run and does not
    # require pretending that one dataset can audit two different revisions.
    # Every path that reaches metric guardrails is audited below.
    if _incompatibility(baseline, candidate, policy) is not None:
        return _compare_verified_runs(baseline, candidate, policy)
    audit_run_manifest(dataset, baseline)
    audit_run_manifest(dataset, candidate)
    system_metric_names = {
        metric.name
        for run in (baseline, candidate)
        for metric in run.system_metrics
    }
    unauditable_guardrails = sorted(
        guardrail.metric_name
        for guardrail in policy.guardrails
        if guardrail.metric_name in system_metric_names
    )
    if unauditable_guardrails:
        raise BenchmarkExecutionError(
            "benchmark promotion cannot audit observed system metric guardrails"
        )
    return _compare_verified_runs(baseline, candidate, policy)


def _compare_verified_runs(
    baseline: BenchmarkRunManifest,
    candidate: BenchmarkRunManifest,
    policy: PromotionPolicy,
) -> RunComparison:
    """Compare runs whose portable quality evidence was already audited."""

    incompatibility = _incompatibility(baseline, candidate, policy)
    completed_count = sum(
        outcome.status == "complete" for outcome in candidate.case_outcomes
    )
    if incompatibility is not None:
        return RunComparison(
            baseline_run_id=baseline.run_id,
            candidate_run_id=candidate.run_id,
            baseline_code_sha=baseline.code_sha,
            candidate_code_sha=candidate.code_sha,
            code_sha_difference_allowed=(
                policy.allowed_differences.allow_code_sha_difference
            ),
            allowed_differences=policy.allowed_differences,
            policy_id=policy.policy_id,
            minimum_completed_cases=policy.minimum_completed_cases,
            require_no_errors=policy.require_no_errors,
            status="incomparable",
            completed_case_count=completed_count,
            statistical_significance="not_assessed",
            caveat=incompatibility,
            guardrails=(),
        )

    baseline_metrics = {
        metric.name: metric
        for metric in (*baseline.quality_metrics, *baseline.system_metrics)
    }
    candidate_metrics = {
        metric.name: metric
        for metric in (*candidate.quality_metrics, *candidate.system_metrics)
    }
    checks = tuple(
        _check_guardrail(guardrail, baseline_metrics, candidate_metrics)
        for guardrail in sorted(policy.guardrails, key=lambda item: item.metric_name)
    )
    candidate_errors = sum(
        outcome.status != "complete" for outcome in candidate.case_outcomes
    )
    baseline_completed = sum(
        outcome.status == "complete" for outcome in baseline.case_outcomes
    )
    baseline_errors = len(baseline.case_outcomes) - baseline_completed
    if policy.require_no_errors and baseline_errors:
        return RunComparison(
            baseline_run_id=baseline.run_id,
            candidate_run_id=candidate.run_id,
            baseline_code_sha=baseline.code_sha,
            candidate_code_sha=candidate.code_sha,
            code_sha_difference_allowed=(
                policy.allowed_differences.allow_code_sha_difference
            ),
            allowed_differences=policy.allowed_differences,
            policy_id=policy.policy_id,
            minimum_completed_cases=policy.minimum_completed_cases,
            require_no_errors=policy.require_no_errors,
            status="incomparable",
            completed_case_count=completed_count,
            statistical_significance="not_assessed",
            caveat=(
                "Baseline contains infrastructure errors under a no-errors policy "
                "and cannot support a promotion comparison."
            ),
            guardrails=(),
        )
    if (policy.require_no_errors and candidate_errors) or not all(
        check.passed for check in checks
    ):
        status: ComparisonStatus = "reject"
        caveat = (
            "Candidate rejected by explicit guardrails or infrastructure-error policy; "
            "statistical significance was not assessed."
        )
    elif (
        completed_count < policy.minimum_completed_cases
        or baseline_completed < policy.minimum_completed_cases
    ):
        status = "insufficient_evidence"
        caveat = (
            f"Only {completed_count} candidate and {baseline_completed} baseline "
            f"cases completed; policy requires {policy.minimum_completed_cases}. "
            "Guardrail checks are descriptive and statistical significance was not assessed."
        )
    else:
        status = "eligible"
        caveat = (
            "Candidate is eligible under the declared deterministic guardrails only; "
            "this is not a claim of statistical significance."
        )
    return RunComparison(
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        baseline_code_sha=baseline.code_sha,
        candidate_code_sha=candidate.code_sha,
        code_sha_difference_allowed=(
            policy.allowed_differences.allow_code_sha_difference
        ),
        allowed_differences=policy.allowed_differences,
        policy_id=policy.policy_id,
        minimum_completed_cases=policy.minimum_completed_cases,
        require_no_errors=policy.require_no_errors,
        status=status,
        completed_case_count=completed_count,
        statistical_significance="not_assessed",
        caveat=caveat,
        guardrails=checks,
    )


def comparison_json(comparison: RunComparison) -> str:
    value = {
        "baseline_run_id": comparison.baseline_run_id,
        "candidate_run_id": comparison.candidate_run_id,
        "baseline_code_sha": comparison.baseline_code_sha,
        "candidate_code_sha": comparison.candidate_code_sha,
        "code_sha_difference_allowed": comparison.code_sha_difference_allowed,
        "allowed_differences": {
            "allow_code_sha_difference": (
                comparison.allowed_differences.allow_code_sha_difference
            ),
            "allow_benchmark_profile_difference": (
                comparison.allowed_differences.allow_benchmark_profile_difference
            ),
            "allow_benchmark_search_plan_difference": (
                comparison.allowed_differences.allow_benchmark_search_plan_difference
            ),
            "model_component_ids": list(
                comparison.allowed_differences.model_component_ids
            ),
            "index_component_ids": list(
                comparison.allowed_differences.index_component_ids
            ),
            "config_component_ids": list(
                comparison.allowed_differences.config_component_ids
            ),
        },
        "policy_id": comparison.policy_id,
        "minimum_completed_cases": comparison.minimum_completed_cases,
        "require_no_errors": comparison.require_no_errors,
        "status": comparison.status,
        "completed_case_count": comparison.completed_case_count,
        "statistical_significance": comparison.statistical_significance,
        "caveat": comparison.caveat,
        "guardrails": [
            {
                "metric_name": check.metric_name,
                "direction": check.direction,
                "allowed_regression": check.allowed_regression,
                "absolute_threshold": check.absolute_threshold,
                "baseline_value": check.baseline_value,
                "candidate_value": check.candidate_value,
                "delta": check.delta,
                "passed": check.passed,
                "reason": check.reason,
            }
            for check in comparison.guardrails
        ],
    }
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _incompatibility(
    baseline: BenchmarkRunManifest,
    candidate: BenchmarkRunManifest,
    policy: PromotionPolicy,
) -> str | None:
    if baseline.dataset_revision != candidate.dataset_revision:
        return "Runs use different dataset revisions and are not comparable."
    if baseline.execution_mode != candidate.execution_mode:
        return "Runs use different cold/warm execution modes and are not comparable."
    if baseline.hardware != candidate.hardware:
        return "Runs use different hardware profiles and are not comparable."
    if baseline.measurement_protocol != candidate.measurement_protocol:
        return "Runs use different measurement protocols and are not comparable."
    if baseline.measurement_status != candidate.measurement_status:
        return "Runs have different measurement statuses and are not comparable."
    if baseline.measurement_protocol.identity == LEGACY_UNMEASURED_PROTOCOL_IDENTITY:
        return (
            "Legacy v1 runs lack an explicit measurement contract and are not "
            "promotion-comparable."
        )
    if baseline.measurement_status == "failed":
        return "Runs contain failed system measurements and are not comparable."
    if baseline.run_status in {"cancelled", "failed"}:
        return "The baseline run did not produce a comparable result set."
    if candidate.run_status in {"cancelled", "failed"}:
        return "The candidate run did not produce a comparable result set."
    allowed = policy.allowed_differences
    if (
        baseline.code_sha != candidate.code_sha
        and not allowed.allow_code_sha_difference
    ):
        return (
            "Runs use different code SHAs without explicit policy permission "
            "and are not comparable."
        )
    if (
        _config_identity(baseline, "benchmark_profile") is None
        or _config_identity(candidate, "benchmark_profile") is None
    ):
        return "Runs must declare named benchmark profiles to be comparable."
    if (
        _config_identity(baseline, "benchmark_search_plan") is None
        or _config_identity(candidate, "benchmark_search_plan") is None
    ):
        return "Runs must declare frozen search-plan identities to be comparable."
    baseline_methodology = _config_identity(baseline, "benchmark_methodology")
    candidate_methodology = _config_identity(candidate, "benchmark_methodology")
    if (
        baseline_methodology is None
        or candidate_methodology is None
        or baseline_methodology != candidate_methodology
    ):
        return "Runs use different benchmark methodologies and are not comparable."
    baseline_lifecycle = _config_identity(
        baseline,
        "benchmark_execution_lifecycle",
    )
    candidate_lifecycle = _config_identity(
        candidate,
        "benchmark_execution_lifecycle",
    )
    if (
        baseline_lifecycle is None
        or candidate_lifecycle is None
        or baseline_lifecycle != candidate_lifecycle
        or not baseline_lifecycle.startswith(f"{baseline.execution_mode}:")
        or not candidate_lifecycle.startswith(f"{candidate.execution_mode}:")
    ):
        return (
            "Runs use different or unverified execution lifecycle policies and "
            "are not comparable."
        )
    allowed_config_ids = set(allowed.config_component_ids)
    if allowed.allow_benchmark_profile_difference:
        allowed_config_ids.add("benchmark_profile")
    if allowed.allow_benchmark_search_plan_difference:
        allowed_config_ids.add("benchmark_search_plan")
    for label, baseline_values, candidate_values, allowed_ids in (
        (
            "model",
            baseline.model_identities,
            candidate.model_identities,
            set(allowed.model_component_ids),
        ),
        (
            "index",
            baseline.index_identities,
            candidate.index_identities,
            set(allowed.index_component_ids),
        ),
        (
            "config",
            baseline.config_identities,
            candidate.config_identities,
            allowed_config_ids,
        ),
    ):
        identity_incompatibility = _identity_incompatibility(
            label,
            baseline_values,
            candidate_values,
            allowed_ids,
        )
        if identity_incompatibility is not None:
            return identity_incompatibility
    baseline_cases = tuple(sorted(item.case_id for item in baseline.case_outcomes))
    candidate_cases = tuple(sorted(item.case_id for item in candidate.case_outcomes))
    if baseline_cases != candidate_cases:
        return "Runs contain different case identities and are not comparable."
    baseline_completed = tuple(
        sorted(
            item.case_id
            for item in baseline.case_outcomes
            if item.status == "complete"
        )
    )
    candidate_completed = tuple(
        sorted(
            item.case_id
            for item in candidate.case_outcomes
            if item.status == "complete"
        )
    )
    if baseline_completed != candidate_completed:
        return (
            "Runs use different completed case cohorts and denominators and are "
            "not comparable."
        )
    for label, run in (("Baseline", baseline), ("Candidate", candidate)):
        inconsistency = _aggregate_count_inconsistency(run)
        if inconsistency is not None:
            return f"{label} {inconsistency} and is not comparable."
    return None


def _identity_incompatibility(
    label: str,
    baseline_values: tuple[ComponentIdentity, ...],
    candidate_values: tuple[ComponentIdentity, ...],
    allowed_ids: set[str],
) -> str | None:
    baseline = {item.component_id: item.identity for item in baseline_values}
    candidate = {item.component_id: item.identity for item in candidate_values}
    undeclared_membership = tuple(
        sorted((set(baseline) ^ set(candidate)) - allowed_ids)
    )
    if undeclared_membership:
        return (
            f"Runs changed undeclared {label} component membership "
            f"({', '.join(undeclared_membership)}) and are not comparable."
        )
    undeclared = tuple(
        sorted(
            component_id
            for component_id in set(baseline) & set(candidate)
            if baseline[component_id] != candidate[component_id]
            and component_id not in allowed_ids
        )
    )
    if undeclared:
        return (
            f"Runs changed undeclared {label} component identities "
            f"({', '.join(undeclared)}) and are not comparable."
        )
    return None


def _check_guardrail(
    guardrail: MetricGuardrail,
    baseline_metrics: dict[str, MetricValue],
    candidate_metrics: dict[str, MetricValue],
) -> GuardrailCheck:
    baseline_metric = baseline_metrics.get(guardrail.metric_name)
    candidate_metric = candidate_metrics.get(guardrail.metric_name)
    if baseline_metric is None or candidate_metric is None:
        return GuardrailCheck(
            metric_name=guardrail.metric_name,
            direction=guardrail.direction,
            allowed_regression=guardrail.allowed_regression,
            absolute_threshold=guardrail.absolute_threshold,
            baseline_value=(baseline_metric.value if baseline_metric else None),
            candidate_value=(candidate_metric.value if candidate_metric else None),
            delta=None,
            passed=False,
            reason="missing_metric",
        )
    baseline = baseline_metric.value
    candidate = candidate_metric.value
    if baseline_metric.unit != candidate_metric.unit:
        return GuardrailCheck(
            metric_name=guardrail.metric_name,
            direction=guardrail.direction,
            allowed_regression=guardrail.allowed_regression,
            absolute_threshold=guardrail.absolute_threshold,
            baseline_value=baseline,
            candidate_value=candidate,
            delta=None,
            passed=False,
            reason="unit_mismatch",
        )
    delta = candidate - baseline
    if guardrail.direction == "higher_is_better":
        regression_passed = candidate >= baseline - guardrail.allowed_regression
        threshold_passed = (
            guardrail.absolute_threshold is None
            or candidate >= guardrail.absolute_threshold
        )
    else:
        regression_passed = candidate <= baseline + guardrail.allowed_regression
        threshold_passed = (
            guardrail.absolute_threshold is None
            or candidate <= guardrail.absolute_threshold
        )
    if regression_passed and threshold_passed:
        reason = "passed"
    elif not regression_passed and not threshold_passed:
        reason = "regression_and_threshold"
    elif not regression_passed:
        reason = "regression"
    else:
        reason = "absolute_threshold"
    return GuardrailCheck(
        metric_name=guardrail.metric_name,
        direction=guardrail.direction,
        allowed_regression=guardrail.allowed_regression,
        absolute_threshold=guardrail.absolute_threshold,
        baseline_value=baseline,
        candidate_value=candidate,
        delta=delta,
        passed=regression_passed and threshold_passed,
        reason=reason,
    )


def _config_identity(run: BenchmarkRunManifest, component_id: str) -> str | None:
    return next(
        (
            identity.identity
            for identity in run.config_identities
            if identity.component_id == component_id
        ),
        None,
    )


def _aggregate_count_inconsistency(run: BenchmarkRunManifest) -> str | None:
    metrics = {metric.name: metric.value for metric in run.quality_metrics}
    completed = sum(outcome.status == "complete" for outcome in run.case_outcomes)
    errors = sum(outcome.status != "complete" for outcome in run.case_outcomes)
    expected = {
        "total_case_count": len(run.case_outcomes),
        "completed_case_count": completed,
        "error_count": errors,
    }
    for name, value in expected.items():
        persisted = metrics.get(name)
        if persisted is not None and persisted != value:
            return f"has inconsistent {name}"
    return None
