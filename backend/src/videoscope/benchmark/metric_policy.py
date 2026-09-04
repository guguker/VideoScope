"""Frozen Phase-0 metric policy and its fail-closed validation CLI.

This contract deliberately does not make promotion decisions.  The only
committed Phase-0 data is already-seen regression evidence, so promotion stays
ineligible until an independent holdout exists.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from typing import Literal, Sequence

from .measurements import MEASUREMENT_PROTOCOL_IDENTITY_PREFIX
from .policy import POLICY_SCHEMA_VERSION
from .profiles import EVALUATION_SEARCH_PLAN_SCHEMA_VERSION, FROZEN_PROFILES
from .runner import BENCHMARK_METHODOLOGY_VERSION, _methodology_identity
from .schema import (
    DATASET_SCHEMA_VERSION,
    MEASUREMENT_EVIDENCE_SCHEMA_VERSION,
    RUN_SCHEMA_VERSION,
    BenchmarkDataError,
    BenchmarkDataset,
    _require_finite_float,
    _require_id,
    _require_positive_int,
    _require_string,
)
from .serialization import (
    JsonObject,
    dataset_revision,
    expect_fields,
    expect_list,
    expect_object,
    parse_json_object,
)
from .storage import _read_bounded_file, load_dataset
from .video_verifier_runner import (
    VIDEO_VERIFIER_CANDIDATE_SCHEMA_VERSION,
    VIDEO_VERIFIER_RUN_SCHEMA_VERSION,
)
from .video_verifier_schema import (
    MAX_VIDEO_VERIFIER_MANIFEST_BYTES,
    VIDEO_VERIFIER_DATASET_SCHEMA_VERSION,
    VideoVerifierDataset,
    video_verifier_dataset_from_json,
    video_verifier_dataset_revision,
)


FROZEN_METRIC_POLICY_SCHEMA_VERSION = 1
MAX_FROZEN_METRIC_POLICY_BYTES = 1024 * 1024
_MAX_POLICY_ITEMS = 256
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

MetricDirection = Literal["higher_is_better", "lower_is_better"]
MetricLevel = Literal["proposal", "component", "product", "reliability", "system"]
MetricUnit = Literal["fraction", "count", "seconds", "milliseconds", "bytes"]
ProfileRequirement = Literal[
    "must_execute_or_report_not_configured",
    "optional_execute_or_report_not_configured",
]

_PRIMARY_METRICS = {
    "candidate_recall_at_50": ("proposal", "higher_is_better", "fraction"),
    "error_count": ("reliability", "lower_is_better", "count"),
    "hard_negative_hit_rate": ("product", "lower_is_better", "fraction"),
    "infrastructure_error_count": (
        "reliability",
        "lower_is_better",
        "count",
    ),
    "mean_boundary_error_seconds": ("product", "lower_is_better", "seconds"),
    "model_miss_count": ("component", "lower_is_better", "count"),
    "ndcg_at_10": ("product", "higher_is_better", "fraction"),
    "p95_latency_ms": ("system", "lower_is_better", "milliseconds"),
    "precision_at_5": ("product", "higher_is_better", "fraction"),
    "recall_at_10": ("product", "higher_is_better", "fraction"),
    "recall_at_20": ("product", "higher_is_better", "fraction"),
    "sampled_peak_process_tree_rss_bytes": (
        "system",
        "lower_is_better",
        "bytes",
    ),
}

_CRITICAL_SLICES = {
    "advertisement_non_game": ("event_class", "advertisement_non_game"),
    "different_camera_or_league": (
        "distribution_shift",
        "different_camera_or_league",
    ),
    "free_throw": ("event_class", "free_throw"),
    "low_resolution": ("capture_condition", "low_resolution"),
    "made_2": ("event_class", "made_2"),
    "made_3": ("event_class", "made_3"),
    "miss": ("event_class", "miss"),
    "ood_non_basketball": ("domain", "ood_non_basketball"),
    "replay": ("event_class", "replay"),
    "scoreboard_hidden": ("capture_condition", "scoreboard_hidden"),
    "universal_non_sports": ("domain", "universal_non_sports"),
}

_PROFILE_REQUIREMENTS = {
    "dense_siglip": "must_execute_or_report_not_configured",
    "internvideo": "optional_execute_or_report_not_configured",
    "lexical_qdrant": "must_execute_or_report_not_configured",
    "lighthouse": "must_execute_or_report_not_configured",
    "qwen_verification": "must_execute_or_report_not_configured",
    "temporal_refinement": "must_execute_or_report_not_configured",
}

_HARD_CONDITIONS = frozenset(
    {
        "advertisement_never_confirmed_as_three",
        "evidence_or_abstain_preserved",
        "free_throw_never_confirmed_as_three",
        "known_made_threes_in_top_10",
        "no_metal_oom",
        "no_sustained_swap",
        "regression_seen_never_promotion_evidence",
        "train_serve_preprocessing_parity",
        "zero_infrastructure_errors",
        "jersey_requires_multiframe_corroboration",
    }
)


def _require_choice(value: object, choices: frozenset[str], field: str) -> str:
    result = _require_id(value, field)
    if result not in choices:
        raise BenchmarkDataError(
            f"{field} must be one of: {', '.join(sorted(choices))}"
        )
    return result


def _require_bool(value: object, field: str) -> bool:
    if type(value) is not bool:
        raise BenchmarkDataError(f"{field} must be a boolean")
    return value


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise BenchmarkDataError(
            f"{field} must be a lowercase 64-character SHA-256"
        )
    return value


def _require_optional_nonnegative_float(value: object, field: str) -> float | None:
    if value is None:
        return None
    return _require_finite_float(value, field, minimum=0)


def _require_string_tuple(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise BenchmarkDataError(f"{field} must be an immutable tuple")
    items = tuple(_require_id(item, field) for item in value)
    if len(items) != len(set(items)):
        raise BenchmarkDataError(f"{field} must contain unique values")
    return tuple(sorted(items))


@dataclass(frozen=True, slots=True)
class VerifierDatasetBinding:
    dataset_id: str
    dataset_version: str
    dataset_revision: str
    dataset_schema_version: int
    case_count: int
    source_group_count: int
    splits: tuple[str, ...]
    label_qualities: tuple[str, ...]
    strata: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_id(self.dataset_id, "verifier_dataset.dataset_id")
        _require_id(self.dataset_version, "verifier_dataset.dataset_version")
        _require_sha256(self.dataset_revision, "verifier_dataset.dataset_revision")
        _require_positive_int(
            self.dataset_schema_version,
            "verifier_dataset.dataset_schema_version",
        )
        _require_positive_int(self.case_count, "verifier_dataset.case_count")
        _require_positive_int(
            self.source_group_count,
            "verifier_dataset.source_group_count",
        )
        for field_name in ("splits", "label_qualities", "strata"):
            object.__setattr__(
                self,
                field_name,
                _require_string_tuple(
                    getattr(self, field_name),
                    f"verifier_dataset.{field_name}",
                ),
            )


@dataclass(frozen=True, slots=True)
class ProductDatasetBinding:
    dataset_id: str
    dataset_version: str
    dataset_revision: str
    dataset_schema_version: int
    case_count: int
    source_group_count: int
    evidence_use: str

    def __post_init__(self) -> None:
        _require_id(self.dataset_id, "product_dataset.dataset_id")
        _require_id(self.dataset_version, "product_dataset.dataset_version")
        _require_sha256(self.dataset_revision, "product_dataset.dataset_revision")
        _require_positive_int(
            self.dataset_schema_version,
            "product_dataset.dataset_schema_version",
        )
        _require_positive_int(self.case_count, "product_dataset.case_count")
        _require_positive_int(
            self.source_group_count,
            "product_dataset.source_group_count",
        )
        _require_choice(
            self.evidence_use,
            frozenset({"regression_only"}),
            "product_dataset.evidence_use",
        )


@dataclass(frozen=True, slots=True)
class EvaluationDataPolicy:
    status: str
    evidence_use: str
    promotion_evidence_status: str
    promotion_eligible: bool
    promotion_blocker: str
    product_dataset_status: str
    product_dataset: ProductDatasetBinding
    verifier_dataset: VerifierDatasetBinding

    def __post_init__(self) -> None:
        _require_choice(
            self.status,
            frozenset({"regression_seen_only"}),
            "evaluation_data.status",
        )
        _require_choice(
            self.evidence_use,
            frozenset({"regression_only"}),
            "evaluation_data.evidence_use",
        )
        _require_choice(
            self.promotion_evidence_status,
            frozenset({"not_promotion_evidence"}),
            "evaluation_data.promotion_evidence_status",
        )
        _require_bool(
            self.promotion_eligible,
            "evaluation_data.promotion_eligible",
        )
        if self.promotion_eligible:
            raise BenchmarkDataError(
                "regression-only evaluation data cannot be promotion eligible"
            )
        _require_choice(
            self.promotion_blocker,
            frozenset({"independent_holdout_not_available"}),
            "evaluation_data.promotion_blocker",
        )
        _require_choice(
            self.product_dataset_status,
            frozenset({"frozen_regression_seen"}),
            "evaluation_data.product_dataset_status",
        )
        if not isinstance(self.product_dataset, ProductDatasetBinding):
            raise BenchmarkDataError(
                "evaluation_data.product_dataset must be a dataset binding"
            )
        if self.product_dataset.evidence_use != self.evidence_use:
            raise BenchmarkDataError(
                "product dataset evidence use must match the regression policy"
            )
        if not isinstance(self.verifier_dataset, VerifierDatasetBinding):
            raise BenchmarkDataError(
                "evaluation_data.verifier_dataset must be a dataset binding"
            )
        if self.verifier_dataset.splits != ("regression_seen",):
            raise BenchmarkDataError(
                "regression policy verifier splits must be exactly regression_seen"
            )
        if self.verifier_dataset.label_qualities != ("gold",):
            raise BenchmarkDataError(
                "regression policy verifier labels must be exactly gold"
            )


@dataclass(frozen=True, slots=True)
class MetricPolicyContracts:
    benchmark_methodology_version: int
    benchmark_methodology_identity: str
    benchmark_dataset_schema_version: int
    benchmark_run_schema_version: int
    video_verifier_dataset_schema_version: int
    video_verifier_candidate_schema_version: int
    video_verifier_run_schema_version: int
    promotion_policy_schema_version: int
    evaluation_search_plan_schema_version: int
    measurement_evidence_schema_version: int
    measurement_protocol_identity_prefix: str

    def __post_init__(self) -> None:
        for field_name in (
            "benchmark_methodology_version",
            "benchmark_dataset_schema_version",
            "benchmark_run_schema_version",
            "video_verifier_dataset_schema_version",
            "video_verifier_candidate_schema_version",
            "video_verifier_run_schema_version",
            "promotion_policy_schema_version",
            "evaluation_search_plan_schema_version",
            "measurement_evidence_schema_version",
        ):
            _require_positive_int(getattr(self, field_name), f"contracts.{field_name}")
        _require_string(
            self.benchmark_methodology_identity,
            "contracts.benchmark_methodology_identity",
            max_length=2_000,
        )
        _require_string(
            self.measurement_protocol_identity_prefix,
            "contracts.measurement_protocol_identity_prefix",
            max_length=512,
        )


@dataclass(frozen=True, slots=True)
class ProfileRevision:
    profile_id: str
    profile_schema_version: int
    profile_identity: str
    search_plan_identity: str
    phase0_requirement: ProfileRequirement

    def __post_init__(self) -> None:
        _require_id(self.profile_id, "profile.profile_id")
        _require_positive_int(
            self.profile_schema_version,
            "profile.profile_schema_version",
        )
        _require_string(self.profile_identity, "profile.profile_identity")
        _require_string(self.search_plan_identity, "profile.search_plan_identity")
        _require_choice(
            self.phase0_requirement,
            frozenset(
                {
                    "must_execute_or_report_not_configured",
                    "optional_execute_or_report_not_configured",
                }
            ),
            "profile.phase0_requirement",
        )


@dataclass(frozen=True, slots=True)
class PrimaryMetric:
    metric_name: str
    level: MetricLevel
    direction: MetricDirection
    unit: MetricUnit

    def __post_init__(self) -> None:
        _require_id(self.metric_name, "primary_metric.metric_name")
        _require_choice(
            self.level,
            frozenset({"proposal", "component", "product", "reliability", "system"}),
            "primary_metric.level",
        )
        _require_choice(
            self.direction,
            frozenset({"higher_is_better", "lower_is_better"}),
            "primary_metric.direction",
        )
        _require_choice(
            self.unit,
            frozenset({"fraction", "count", "seconds", "milliseconds", "bytes"}),
            "primary_metric.unit",
        )


@dataclass(frozen=True, slots=True)
class CriticalSlice:
    slice_id: str
    dimension: str
    value: str

    def __post_init__(self) -> None:
        _require_id(self.slice_id, "critical_slice.slice_id")
        _require_id(self.dimension, "critical_slice.dimension")
        _require_id(self.value, "critical_slice.value")


@dataclass(frozen=True, slots=True)
class MetricGuardrail:
    guardrail_id: str
    metric_name: str
    scope: str
    direction: MetricDirection
    absolute_threshold: float | None
    maximum_absolute_regression: float | None
    maximum_relative_regression: float | None

    def __post_init__(self) -> None:
        _require_id(self.guardrail_id, "metric_guardrail.guardrail_id")
        _require_id(self.metric_name, "metric_guardrail.metric_name")
        _require_choice(
            self.scope,
            frozenset({"overall", "critical_slices", "universal_non_sports"}),
            "metric_guardrail.scope",
        )
        _require_choice(
            self.direction,
            frozenset({"higher_is_better", "lower_is_better"}),
            "metric_guardrail.direction",
        )
        for field_name in (
            "absolute_threshold",
            "maximum_absolute_regression",
            "maximum_relative_regression",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_optional_nonnegative_float(
                    getattr(self, field_name),
                    f"metric_guardrail.{field_name}",
                ),
            )
        if (
            self.absolute_threshold is None
            and self.maximum_absolute_regression is None
            and self.maximum_relative_regression is None
        ):
            raise BenchmarkDataError(
                "metric guardrail must declare at least one threshold"
            )
        if (
            self.maximum_relative_regression is not None
            and self.maximum_relative_regression > 1
        ):
            raise BenchmarkDataError(
                "metric guardrail maximum_relative_regression must not exceed one"
            )


@dataclass(frozen=True, slots=True)
class HardGuardrails:
    metrics: tuple[MetricGuardrail, ...]
    conditions: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.metrics, tuple) or any(
            not isinstance(item, MetricGuardrail) for item in self.metrics
        ):
            raise BenchmarkDataError(
                "hard_guardrails.metrics must contain metric guardrails"
            )
        if not self.metrics or len(self.metrics) > _MAX_POLICY_ITEMS:
            raise BenchmarkDataError(
                "hard_guardrails.metrics must contain a bounded non-empty set"
            )
        ids = tuple(item.guardrail_id for item in self.metrics)
        if len(ids) != len(set(ids)):
            raise BenchmarkDataError(
                "hard_guardrails.metrics must have unique guardrail IDs"
            )
        object.__setattr__(
            self,
            "metrics",
            tuple(sorted(self.metrics, key=lambda item: item.guardrail_id)),
        )
        conditions = _require_string_tuple(
            self.conditions,
            "hard_guardrails.conditions",
        )
        if not conditions or len(conditions) > _MAX_POLICY_ITEMS:
            raise BenchmarkDataError(
                "hard_guardrails.conditions must contain a bounded non-empty set"
            )
        object.__setattr__(self, "conditions", conditions)


@dataclass(frozen=True, slots=True)
class MinimumEffect:
    metric_name: str
    direction: MetricDirection
    minimum_absolute_improvement: float

    def __post_init__(self) -> None:
        _require_id(self.metric_name, "minimum_effect.metric_name")
        _require_choice(
            self.direction,
            frozenset({"higher_is_better", "lower_is_better"}),
            "minimum_effect.direction",
        )
        improvement = _require_finite_float(
            self.minimum_absolute_improvement,
            "minimum_effect.minimum_absolute_improvement",
            strictly_positive=True,
        )
        if improvement > 1:
            raise BenchmarkDataError(
                "minimum effect absolute improvement must not exceed one"
            )
        object.__setattr__(self, "minimum_absolute_improvement", improvement)


@dataclass(frozen=True, slots=True)
class MinimumUsefulEffect:
    comparison_baseline: str
    combination: str
    effects: tuple[MinimumEffect, ...]

    def __post_init__(self) -> None:
        _require_choice(
            self.comparison_baseline,
            frozenset({"frozen_feature_control"}),
            "minimum_useful_effect.comparison_baseline",
        )
        _require_choice(
            self.combination,
            frozenset({"any"}),
            "minimum_useful_effect.combination",
        )
        if not isinstance(self.effects, tuple) or any(
            not isinstance(item, MinimumEffect) for item in self.effects
        ):
            raise BenchmarkDataError(
                "minimum_useful_effect.effects must contain effect rules"
            )
        ids = tuple(item.metric_name for item in self.effects)
        if len(ids) != len(set(ids)):
            raise BenchmarkDataError(
                "minimum_useful_effect.effects must have unique metric names"
            )
        object.__setattr__(
            self,
            "effects",
            tuple(sorted(self.effects, key=lambda item: item.metric_name)),
        )


@dataclass(frozen=True, slots=True)
class UncertaintyPolicy:
    method: str
    confidence_level: float
    resampling_unit: str
    minimum_completed_gold_cases_per_critical_slice: int
    minimum_source_groups_per_critical_slice: int
    interval_rule: str
    decision_when_unavailable: str

    def __post_init__(self) -> None:
        _require_choice(
            self.method,
            frozenset({"source_group_bootstrap"}),
            "uncertainty.method",
        )
        confidence = _require_finite_float(
            self.confidence_level,
            "uncertainty.confidence_level",
            strictly_positive=True,
        )
        if confidence >= 1:
            raise BenchmarkDataError("uncertainty.confidence_level must be below one")
        object.__setattr__(self, "confidence_level", confidence)
        _require_choice(
            self.resampling_unit,
            frozenset({"whole_source_group"}),
            "uncertainty.resampling_unit",
        )
        _require_positive_int(
            self.minimum_completed_gold_cases_per_critical_slice,
            "uncertainty.minimum_completed_gold_cases_per_critical_slice",
        )
        _require_positive_int(
            self.minimum_source_groups_per_critical_slice,
            "uncertainty.minimum_source_groups_per_critical_slice",
        )
        _require_choice(
            self.interval_rule,
            frozenset({"two_sided_interval_excludes_zero"}),
            "uncertainty.interval_rule",
        )
        _require_choice(
            self.decision_when_unavailable,
            frozenset({"insufficient_evidence"}),
            "uncertainty.decision_when_unavailable",
        )


@dataclass(frozen=True, slots=True)
class FrozenMetricPolicy:
    schema_version: int
    policy_id: str
    policy_version: str
    evaluation_data: EvaluationDataPolicy
    contracts: MetricPolicyContracts
    profiles: tuple[ProfileRevision, ...]
    primary_metrics: tuple[PrimaryMetric, ...]
    critical_slices: tuple[CriticalSlice, ...]
    hard_guardrails: HardGuardrails
    minimum_useful_effect: MinimumUsefulEffect
    uncertainty: UncertaintyPolicy

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != FROZEN_METRIC_POLICY_SCHEMA_VERSION
        ):
            raise BenchmarkDataError(
                "frozen metric policy schema_version must be "
                f"{FROZEN_METRIC_POLICY_SCHEMA_VERSION}"
            )
        _require_id(self.policy_id, "metric_policy.policy_id")
        _require_id(self.policy_version, "metric_policy.policy_version")
        for field_name, item_type in (
            ("profiles", ProfileRevision),
            ("primary_metrics", PrimaryMetric),
            ("critical_slices", CriticalSlice),
        ):
            values = getattr(self, field_name)
            if not isinstance(values, tuple) or any(
                not isinstance(item, item_type) for item in values
            ):
                raise BenchmarkDataError(
                    f"metric_policy.{field_name} contains invalid items"
                )
            if not values or len(values) > _MAX_POLICY_ITEMS:
                raise BenchmarkDataError(
                    f"metric_policy.{field_name} must be a bounded non-empty set"
                )
            key_name = {
                "profiles": "profile_id",
                "primary_metrics": "metric_name",
                "critical_slices": "slice_id",
            }[field_name]
            ids = tuple(getattr(item, key_name) for item in values)
            if len(ids) != len(set(ids)):
                raise BenchmarkDataError(
                    f"metric_policy.{field_name} must contain unique IDs"
                )
            object.__setattr__(
                self,
                field_name,
                tuple(sorted(values, key=lambda item: getattr(item, key_name))),
            )
        if not isinstance(self.evaluation_data, EvaluationDataPolicy):
            raise BenchmarkDataError("metric_policy.evaluation_data is invalid")
        if not isinstance(self.contracts, MetricPolicyContracts):
            raise BenchmarkDataError("metric_policy.contracts is invalid")
        if not isinstance(self.hard_guardrails, HardGuardrails):
            raise BenchmarkDataError("metric_policy.hard_guardrails is invalid")
        if not isinstance(self.minimum_useful_effect, MinimumUsefulEffect):
            raise BenchmarkDataError("metric_policy.minimum_useful_effect is invalid")
        if not isinstance(self.uncertainty, UncertaintyPolicy):
            raise BenchmarkDataError("metric_policy.uncertainty is invalid")
        _validate_phase0_semantics(self)


def load_frozen_metric_policy(
    path: Path,
    *,
    verify_current_contract: bool = True,
) -> FrozenMetricPolicy:
    """Load a bounded regular JSON policy and optionally reject runtime drift."""

    payload = _read_bounded_file(
        Path(path),
        MAX_FROZEN_METRIC_POLICY_BYTES,
        "frozen metric policy",
    )
    return frozen_metric_policy_from_dict(
        parse_json_object(payload, "frozen metric policy"),
        verify_current_contract=verify_current_contract,
    )


def frozen_metric_policy_from_dict(
    value: JsonObject,
    *,
    verify_current_contract: bool = True,
) -> FrozenMetricPolicy:
    """Parse one already-captured JSON object without rereading mutable storage."""

    policy = _policy_from_object(value)
    if verify_current_contract:
        _validate_current_contract(policy)
    return policy


def validate_frozen_metric_policy_dataset(
    policy: FrozenMetricPolicy,
    dataset: VideoVerifierDataset,
) -> None:
    """Prove that the supplied verifier dataset is the exact frozen fixture."""

    if not isinstance(policy, FrozenMetricPolicy):
        raise BenchmarkDataError("policy must be a FrozenMetricPolicy value")
    if not isinstance(dataset, VideoVerifierDataset):
        raise BenchmarkDataError("dataset must be a VideoVerifierDataset value")
    binding = policy.evaluation_data.verifier_dataset
    actual: dict[str, object] = {
        "dataset_id": dataset.dataset_id,
        "dataset_version": dataset.dataset_version,
        "dataset_revision": video_verifier_dataset_revision(dataset),
        "dataset_schema_version": dataset.schema_version,
        "case_count": len(dataset.cases),
        "source_group_count": len({case.split_group for case in dataset.cases}),
        "splits": tuple(sorted({case.split for case in dataset.cases})),
        "label_qualities": tuple(
            sorted({case.label_quality for case in dataset.cases})
        ),
        "strata": tuple(sorted({case.stratum for case in dataset.cases})),
    }
    for field_name, actual_value in actual.items():
        if actual_value != getattr(binding, field_name):
            raise BenchmarkDataError(
                f"verifier dataset {field_name} does not match the frozen policy"
            )


def validate_frozen_metric_policy_product_dataset(
    policy: FrozenMetricPolicy,
    dataset: BenchmarkDataset,
) -> None:
    """Prove that the supplied product dataset is the exact regression fixture."""

    if not isinstance(policy, FrozenMetricPolicy):
        raise BenchmarkDataError("policy must be a FrozenMetricPolicy value")
    if not isinstance(dataset, BenchmarkDataset):
        raise BenchmarkDataError("dataset must be a BenchmarkDataset value")
    binding = policy.evaluation_data.product_dataset
    actual: dict[str, object] = {
        "dataset_id": dataset.dataset_id,
        "dataset_version": dataset.dataset_version,
        "dataset_revision": dataset_revision(dataset),
        "dataset_schema_version": dataset.schema_version,
        "case_count": len(dataset.cases),
        "source_group_count": len({case.split_group for case in dataset.cases}),
        "evidence_use": "regression_only",
    }
    for field_name, actual_value in actual.items():
        if actual_value != getattr(binding, field_name):
            raise BenchmarkDataError(
                f"product dataset {field_name} does not match the frozen policy"
            )
    if any(
        case.label_quality != "gold"
        or case.critical_slices is None
        or "regression_seen" not in case.critical_slices.distribution_shift
        for case in dataset.cases
    ):
        raise BenchmarkDataError(
            "product dataset must contain only gold regression_seen critical slices"
        )


def frozen_metric_policy_revision(policy: FrozenMetricPolicy) -> str:
    if not isinstance(policy, FrozenMetricPolicy):
        raise BenchmarkDataError("policy must be a FrozenMetricPolicy value")
    payload = json.dumps(
        asdict(policy),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _policy_from_object(value: JsonObject) -> FrozenMetricPolicy:
    expect_fields(
        value,
        {
            "schema_version",
            "policy_id",
            "policy_version",
            "evaluation_data",
            "contracts",
            "profiles",
            "primary_metrics",
            "critical_slices",
            "hard_guardrails",
            "minimum_useful_effect",
            "uncertainty",
        },
        "frozen metric policy",
    )
    schema_version = value["schema_version"]
    if (
        type(schema_version) is not int
        or schema_version != FROZEN_METRIC_POLICY_SCHEMA_VERSION
    ):
        raise BenchmarkDataError(
            "frozen metric policy schema_version must be "
            f"{FROZEN_METRIC_POLICY_SCHEMA_VERSION}"
        )
    evaluation_data = expect_object(
        value["evaluation_data"],
        "frozen metric policy evaluation_data",
    )
    expect_fields(
        evaluation_data,
        {
            "status",
            "evidence_use",
            "promotion_evidence_status",
            "promotion_eligible",
            "promotion_blocker",
            "product_dataset_status",
            "product_dataset",
            "verifier_dataset",
        },
        "frozen metric policy evaluation_data",
    )
    verifier_dataset = expect_object(
        evaluation_data["verifier_dataset"],
        "frozen metric policy verifier_dataset",
    )
    product_dataset = expect_object(
        evaluation_data["product_dataset"],
        "frozen metric policy product_dataset",
    )
    expect_fields(
        product_dataset,
        {
            "dataset_id",
            "dataset_version",
            "dataset_revision",
            "dataset_schema_version",
            "case_count",
            "source_group_count",
            "evidence_use",
        },
        "frozen metric policy product_dataset",
    )
    expect_fields(
        verifier_dataset,
        {
            "dataset_id",
            "dataset_version",
            "dataset_revision",
            "dataset_schema_version",
            "case_count",
            "source_group_count",
            "splits",
            "label_qualities",
            "strata",
        },
        "frozen metric policy verifier_dataset",
    )
    contracts = expect_object(value["contracts"], "frozen metric policy contracts")
    expect_fields(
        contracts,
        {
            "benchmark_methodology_version",
            "benchmark_methodology_identity",
            "benchmark_dataset_schema_version",
            "benchmark_run_schema_version",
            "video_verifier_dataset_schema_version",
            "video_verifier_candidate_schema_version",
            "video_verifier_run_schema_version",
            "promotion_policy_schema_version",
            "evaluation_search_plan_schema_version",
            "measurement_evidence_schema_version",
            "measurement_protocol_identity_prefix",
        },
        "frozen metric policy contracts",
    )
    hard_guardrails = expect_object(
        value["hard_guardrails"],
        "frozen metric policy hard_guardrails",
    )
    expect_fields(
        hard_guardrails,
        {"metrics", "conditions"},
        "frozen metric policy hard_guardrails",
    )
    minimum_effect = expect_object(
        value["minimum_useful_effect"],
        "frozen metric policy minimum_useful_effect",
    )
    expect_fields(
        minimum_effect,
        {"comparison_baseline", "combination", "effects"},
        "frozen metric policy minimum_useful_effect",
    )
    uncertainty = expect_object(
        value["uncertainty"],
        "frozen metric policy uncertainty",
    )
    expect_fields(
        uncertainty,
        {
            "method",
            "confidence_level",
            "resampling_unit",
            "minimum_completed_gold_cases_per_critical_slice",
            "minimum_source_groups_per_critical_slice",
            "interval_rule",
            "decision_when_unavailable",
        },
        "frozen metric policy uncertainty",
    )
    return FrozenMetricPolicy(
        schema_version=schema_version,
        policy_id=value["policy_id"],  # type: ignore[arg-type]
        policy_version=value["policy_version"],  # type: ignore[arg-type]
        evaluation_data=EvaluationDataPolicy(
            status=evaluation_data["status"],  # type: ignore[arg-type]
            evidence_use=evaluation_data["evidence_use"],  # type: ignore[arg-type]
            promotion_evidence_status=evaluation_data[
                "promotion_evidence_status"
            ],  # type: ignore[arg-type]
            promotion_eligible=evaluation_data["promotion_eligible"],  # type: ignore[arg-type]
            promotion_blocker=evaluation_data["promotion_blocker"],  # type: ignore[arg-type]
            product_dataset_status=evaluation_data[
                "product_dataset_status"
            ],  # type: ignore[arg-type]
            product_dataset=ProductDatasetBinding(
                dataset_id=product_dataset["dataset_id"],  # type: ignore[arg-type]
                dataset_version=product_dataset["dataset_version"],  # type: ignore[arg-type]
                dataset_revision=product_dataset["dataset_revision"],  # type: ignore[arg-type]
                dataset_schema_version=product_dataset[
                    "dataset_schema_version"
                ],  # type: ignore[arg-type]
                case_count=product_dataset["case_count"],  # type: ignore[arg-type]
                source_group_count=product_dataset[
                    "source_group_count"
                ],  # type: ignore[arg-type]
                evidence_use=product_dataset["evidence_use"],  # type: ignore[arg-type]
            ),
            verifier_dataset=VerifierDatasetBinding(
                dataset_id=verifier_dataset["dataset_id"],  # type: ignore[arg-type]
                dataset_version=verifier_dataset["dataset_version"],  # type: ignore[arg-type]
                dataset_revision=verifier_dataset["dataset_revision"],  # type: ignore[arg-type]
                dataset_schema_version=verifier_dataset[
                    "dataset_schema_version"
                ],  # type: ignore[arg-type]
                case_count=verifier_dataset["case_count"],  # type: ignore[arg-type]
                source_group_count=verifier_dataset["source_group_count"],  # type: ignore[arg-type]
                splits=_string_values(
                    verifier_dataset["splits"],
                    "verifier_dataset.splits",
                ),
                label_qualities=_string_values(
                    verifier_dataset["label_qualities"],
                    "verifier_dataset.label_qualities",
                ),
                strata=_string_values(
                    verifier_dataset["strata"],
                    "verifier_dataset.strata",
                ),
            ),
        ),
        contracts=MetricPolicyContracts(**contracts),  # type: ignore[arg-type]
        profiles=tuple(
            _profile_from_object(item, index)
            for index, item in enumerate(
                _bounded_list(value["profiles"], "profiles")
            )
        ),
        primary_metrics=tuple(
            _primary_metric_from_object(item, index)
            for index, item in enumerate(
                _bounded_list(value["primary_metrics"], "primary_metrics")
            )
        ),
        critical_slices=tuple(
            _critical_slice_from_object(item, index)
            for index, item in enumerate(
                _bounded_list(value["critical_slices"], "critical_slices")
            )
        ),
        hard_guardrails=HardGuardrails(
            metrics=tuple(
                _guardrail_from_object(item, index)
                for index, item in enumerate(
                    _bounded_list(
                        hard_guardrails["metrics"],
                        "hard_guardrails.metrics",
                    )
                )
            ),
            conditions=_string_values(
                hard_guardrails["conditions"],
                "hard_guardrails.conditions",
            ),
        ),
        minimum_useful_effect=MinimumUsefulEffect(
            comparison_baseline=minimum_effect["comparison_baseline"],  # type: ignore[arg-type]
            combination=minimum_effect["combination"],  # type: ignore[arg-type]
            effects=tuple(
                _minimum_effect_from_object(item, index)
                for index, item in enumerate(
                    _bounded_list(
                        minimum_effect["effects"],
                        "minimum_useful_effect.effects",
                    )
                )
            ),
        ),
        uncertainty=UncertaintyPolicy(**uncertainty),  # type: ignore[arg-type]
    )


def _bounded_list(value: object, field: str) -> list[object]:
    values = expect_list(value, f"frozen metric policy {field}")
    if len(values) > _MAX_POLICY_ITEMS:
        raise BenchmarkDataError(
            f"frozen metric policy {field} exceeds the supported limit"
        )
    return values


def _string_values(value: object, field: str) -> tuple[str, ...]:
    return tuple(
        item  # type: ignore[misc]
        for item in _bounded_list(value, field)
    )


def _profile_from_object(value: object, index: int) -> ProfileRevision:
    item = expect_object(value, f"frozen metric policy profiles[{index}]")
    expect_fields(
        item,
        {
            "profile_id",
            "profile_schema_version",
            "profile_identity",
            "search_plan_identity",
            "phase0_requirement",
        },
        f"frozen metric policy profiles[{index}]",
    )
    return ProfileRevision(**item)  # type: ignore[arg-type]


def _primary_metric_from_object(value: object, index: int) -> PrimaryMetric:
    item = expect_object(value, f"frozen metric policy primary_metrics[{index}]")
    expect_fields(
        item,
        {"metric_name", "level", "direction", "unit"},
        f"frozen metric policy primary_metrics[{index}]",
    )
    return PrimaryMetric(**item)  # type: ignore[arg-type]


def _critical_slice_from_object(value: object, index: int) -> CriticalSlice:
    item = expect_object(value, f"frozen metric policy critical_slices[{index}]")
    expect_fields(
        item,
        {"slice_id", "dimension", "value"},
        f"frozen metric policy critical_slices[{index}]",
    )
    return CriticalSlice(**item)  # type: ignore[arg-type]


def _guardrail_from_object(value: object, index: int) -> MetricGuardrail:
    item = expect_object(value, f"frozen metric policy guardrails[{index}]")
    expect_fields(
        item,
        {
            "guardrail_id",
            "metric_name",
            "scope",
            "direction",
            "absolute_threshold",
            "maximum_absolute_regression",
            "maximum_relative_regression",
        },
        f"frozen metric policy guardrails[{index}]",
    )
    return MetricGuardrail(**item)  # type: ignore[arg-type]


def _minimum_effect_from_object(value: object, index: int) -> MinimumEffect:
    item = expect_object(value, f"frozen metric policy effects[{index}]")
    expect_fields(
        item,
        {"metric_name", "direction", "minimum_absolute_improvement"},
        f"frozen metric policy effects[{index}]",
    )
    return MinimumEffect(**item)  # type: ignore[arg-type]


def _validate_phase0_semantics(policy: FrozenMetricPolicy) -> None:
    if policy.policy_id != "phase0-regression-baseline":
        raise BenchmarkDataError(
            "frozen metric policy must use the required Phase-0 policy ID"
        )
    if policy.policy_version != "1.0.0":
        raise BenchmarkDataError(
            "frozen metric policy must use the required Phase-0 policy version"
        )

    profile_requirements = {
        item.profile_id: item.phase0_requirement for item in policy.profiles
    }
    if profile_requirements != _PROFILE_REQUIREMENTS:
        raise BenchmarkDataError(
            "frozen metric policy profiles must contain exactly the required Phase-0 set"
        )
    primary_metrics = {
        item.metric_name: (item.level, item.direction, item.unit)
        for item in policy.primary_metrics
    }
    if primary_metrics != _PRIMARY_METRICS:
        raise BenchmarkDataError(
            "frozen metric policy primary_metrics must contain exactly the required set"
        )
    critical_slices = {
        item.slice_id: (item.dimension, item.value)
        for item in policy.critical_slices
    }
    if critical_slices != _CRITICAL_SLICES:
        raise BenchmarkDataError(
            "frozen metric policy critical_slices must contain exactly the required set"
        )
    if frozenset(policy.hard_guardrails.conditions) != _HARD_CONDITIONS:
        raise BenchmarkDataError(
            "frozen metric policy conditions must contain exactly the required set"
        )

    guardrails = {item.guardrail_id: item for item in policy.hard_guardrails.metrics}
    required_guardrail_ids = {
        "candidate_recall_floor",
        "critical_ndcg_regression",
        "critical_recall_regression",
        "hard_negative_false_positive_ceiling",
        "infrastructure_error_ceiling",
        "interactive_latency_regression",
        "mean_boundary_error_ceiling",
        "peak_process_tree_rss_ceiling",
        "precision_at_5_floor",
        "recall_at_20_floor",
        "universal_ndcg_regression",
        "universal_recall_regression",
        "verifier_infrastructure_error_ceiling",
    }
    if set(guardrails) != required_guardrail_ids:
        raise BenchmarkDataError(
            "frozen metric policy guardrails must contain exactly the required set"
        )
    _require_guardrail(
        guardrails,
        "candidate_recall_floor",
        "candidate_recall_at_50",
        "overall",
        "higher_is_better",
        absolute_threshold=0.95,
    )
    _require_guardrail(
        guardrails,
        "precision_at_5_floor",
        "precision_at_5",
        "overall",
        "higher_is_better",
        absolute_threshold=0.60,
    )
    _require_guardrail(
        guardrails,
        "recall_at_20_floor",
        "recall_at_20",
        "overall",
        "higher_is_better",
        absolute_threshold=0.80,
    )
    _require_guardrail(
        guardrails,
        "mean_boundary_error_ceiling",
        "mean_boundary_error_seconds",
        "overall",
        "lower_is_better",
        absolute_threshold=1.0,
    )
    _require_guardrail(
        guardrails,
        "hard_negative_false_positive_ceiling",
        "hard_negative_hit_rate",
        "overall",
        "lower_is_better",
        absolute_threshold=0.0,
    )
    _require_guardrail(
        guardrails,
        "infrastructure_error_ceiling",
        "error_count",
        "overall",
        "lower_is_better",
        absolute_threshold=0.0,
    )
    _require_guardrail(
        guardrails,
        "verifier_infrastructure_error_ceiling",
        "infrastructure_error_count",
        "overall",
        "lower_is_better",
        absolute_threshold=0.0,
    )
    _require_guardrail(
        guardrails,
        "peak_process_tree_rss_ceiling",
        "sampled_peak_process_tree_rss_bytes",
        "overall",
        "lower_is_better",
        absolute_threshold=float(16 * 1024**3),
    )
    _require_guardrail(
        guardrails,
        "interactive_latency_regression",
        "p95_latency_ms",
        "overall",
        "lower_is_better",
        maximum_absolute_regression=2000.0,
        maximum_relative_regression=0.25,
    )
    _require_guardrail(
        guardrails,
        "critical_ndcg_regression",
        "ndcg_at_10",
        "critical_slices",
        "higher_is_better",
        maximum_absolute_regression=0.05,
    )
    _require_guardrail(
        guardrails,
        "critical_recall_regression",
        "recall_at_10",
        "critical_slices",
        "higher_is_better",
        maximum_absolute_regression=0.05,
    )
    _require_guardrail(
        guardrails,
        "universal_ndcg_regression",
        "ndcg_at_10",
        "universal_non_sports",
        "higher_is_better",
        maximum_absolute_regression=0.02,
    )
    _require_guardrail(
        guardrails,
        "universal_recall_regression",
        "recall_at_10",
        "universal_non_sports",
        "higher_is_better",
        maximum_absolute_regression=0.02,
    )

    effects = {
        item.metric_name: (item.direction, item.minimum_absolute_improvement)
        for item in policy.minimum_useful_effect.effects
    }
    if effects != {
        "ndcg_at_10": ("higher_is_better", 0.03),
        "recall_at_10": ("higher_is_better", 0.03),
    }:
        raise BenchmarkDataError(
            "minimum useful effects must contain exactly the required Phase-0 set"
        )
    if (
        policy.uncertainty.confidence_level != 0.95
        or policy.uncertainty.minimum_completed_gold_cases_per_critical_slice != 20
        or policy.uncertainty.minimum_source_groups_per_critical_slice != 2
    ):
        raise BenchmarkDataError(
            "uncertainty policy must use the required Phase-0 evidence minimums"
        )


def _require_guardrail(
    guardrails: dict[str, MetricGuardrail],
    guardrail_id: str,
    metric_name: str,
    scope: str,
    direction: str,
    *,
    absolute_threshold: float | None = None,
    maximum_absolute_regression: float | None = None,
    maximum_relative_regression: float | None = None,
) -> None:
    actual = guardrails[guardrail_id]
    expected = (
        metric_name,
        scope,
        direction,
        absolute_threshold,
        maximum_absolute_regression,
        maximum_relative_regression,
    )
    observed = (
        actual.metric_name,
        actual.scope,
        actual.direction,
        actual.absolute_threshold,
        actual.maximum_absolute_regression,
        actual.maximum_relative_regression,
    )
    if observed != expected:
        raise BenchmarkDataError(
            f"guardrail {guardrail_id!r} does not match the required Phase-0 rule"
        )


def _validate_current_contract(policy: FrozenMetricPolicy) -> None:
    expected_contracts = MetricPolicyContracts(
        benchmark_methodology_version=BENCHMARK_METHODOLOGY_VERSION,
        benchmark_methodology_identity=_methodology_identity(),
        benchmark_dataset_schema_version=DATASET_SCHEMA_VERSION,
        benchmark_run_schema_version=RUN_SCHEMA_VERSION,
        video_verifier_dataset_schema_version=VIDEO_VERIFIER_DATASET_SCHEMA_VERSION,
        video_verifier_candidate_schema_version=VIDEO_VERIFIER_CANDIDATE_SCHEMA_VERSION,
        video_verifier_run_schema_version=VIDEO_VERIFIER_RUN_SCHEMA_VERSION,
        promotion_policy_schema_version=POLICY_SCHEMA_VERSION,
        evaluation_search_plan_schema_version=EVALUATION_SEARCH_PLAN_SCHEMA_VERSION,
        measurement_evidence_schema_version=MEASUREMENT_EVIDENCE_SCHEMA_VERSION,
        measurement_protocol_identity_prefix=MEASUREMENT_PROTOCOL_IDENTITY_PREFIX,
    )
    if policy.contracts != expected_contracts:
        raise BenchmarkDataError(
            "frozen metric policy contracts do not match the current methodology or schema"
        )
    profiles = {item.profile_id: item for item in policy.profiles}
    if set(profiles) != set(FROZEN_PROFILES):
        raise BenchmarkDataError(
            "frozen metric policy profiles do not match the current profile set"
        )
    for profile_id, current in FROZEN_PROFILES.items():
        frozen = profiles[profile_id]
        if (
            frozen.profile_schema_version != current.schema_version
            or frozen.profile_identity != current.identity
            or frozen.search_plan_identity != current.search_plan.identity
        ):
            raise BenchmarkDataError(
                f"frozen profile {profile_id!r} identity does not match the current contract"
            )


class _CliUsageError(ValueError):
    pass


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        raise _CliUsageError("invalid command arguments")


def _parser() -> _ArgumentParser:
    parser = _ArgumentParser(prog="videoscope-frozen-metric-policy")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--policy", type=Path, required=True)
    validate.add_argument("--product-dataset", type=Path, required=True)
    validate.add_argument("--verifier-dataset", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        if arguments.command != "validate":
            raise _CliUsageError("invalid command")
        policy = load_frozen_metric_policy(arguments.policy)
        dataset_payload = _read_bounded_file(
            arguments.verifier_dataset,
            MAX_VIDEO_VERIFIER_MANIFEST_BYTES,
            "video verifier dataset",
        )
        dataset = video_verifier_dataset_from_json(dataset_payload)
        validate_frozen_metric_policy_dataset(policy, dataset)
        product_dataset = load_dataset(arguments.product_dataset)
        validate_frozen_metric_policy_product_dataset(policy, product_dataset)
        _write_json(
            sys.stdout,
            {
                "dataset_revision": video_verifier_dataset_revision(dataset),
                "product_dataset_revision": dataset_revision(product_dataset),
                "policy_id": policy.policy_id,
                "policy_revision": frozen_metric_policy_revision(policy),
                "promotion_eligible": policy.evaluation_data.promotion_eligible,
                "status": "valid",
            },
        )
        return 0
    except _CliUsageError:
        return _fail(2, "usage_error", "invalid metric policy arguments")
    except FileNotFoundError:
        return _fail(4, "not_found", "metric policy input was not found")
    except BenchmarkDataError:
        return _fail(3, "invalid_data", "metric policy data is invalid")
    except OSError:
        return _fail(7, "io_error", "metric policy storage failed")
    except KeyboardInterrupt:
        return _fail(130, "interrupted", "metric policy validation was interrupted")
    except Exception:
        return _fail(70, "internal_error", "metric policy validation failed")


def _write_json(stream, value: JsonObject) -> None:  # type: ignore[no-untyped-def]
    json.dump(
        value,
        stream,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    stream.write("\n")


def _fail(code: int, error: str, message: str) -> int:
    _write_json(sys.stderr, {"error": error, "message": message})
    return code


__all__ = [
    "FROZEN_METRIC_POLICY_SCHEMA_VERSION",
    "MAX_FROZEN_METRIC_POLICY_BYTES",
    "CriticalSlice",
    "EvaluationDataPolicy",
    "FrozenMetricPolicy",
    "HardGuardrails",
    "MetricGuardrail",
    "MetricPolicyContracts",
    "MinimumEffect",
    "MinimumUsefulEffect",
    "PrimaryMetric",
    "ProductDatasetBinding",
    "ProfileRevision",
    "UncertaintyPolicy",
    "VerifierDatasetBinding",
    "frozen_metric_policy_from_dict",
    "frozen_metric_policy_revision",
    "load_frozen_metric_policy",
    "main",
    "validate_frozen_metric_policy_dataset",
    "validate_frozen_metric_policy_product_dataset",
]


if __name__ == "__main__":
    raise SystemExit(main())
