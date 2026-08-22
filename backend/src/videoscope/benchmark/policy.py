from __future__ import annotations

from pathlib import Path

from .comparison import (
    MAX_ALLOWED_COMPONENT_IDS,
    AllowedRunDifferences,
    MetricGuardrail,
    PromotionPolicy,
)
from .schema import BenchmarkDataError
from .serialization import expect_fields, expect_list, expect_object, parse_json_object
from .storage import _read_bounded_file


POLICY_SCHEMA_VERSION = 2
MAX_POLICY_BYTES = 1024 * 1024
MAX_POLICY_GUARDRAILS = 256


def load_policy(path: Path) -> PromotionPolicy:
    """Load a strict, bounded promotion policy from a regular JSON file."""

    payload = _read_bounded_file(
        Path(path),
        MAX_POLICY_BYTES,
        "benchmark promotion policy",
    )
    value = parse_json_object(payload, "benchmark promotion policy")
    schema_version = value.get("schema_version")
    if type(schema_version) is not int or schema_version != POLICY_SCHEMA_VERSION:
        raise BenchmarkDataError(
            f"promotion policy schema_version must be {POLICY_SCHEMA_VERSION}"
        )
    expect_fields(
        value,
        {
            "schema_version",
            "policy_id",
            "minimum_completed_cases",
            "require_no_errors",
            "allowed_differences",
            "guardrails",
        },
        "benchmark promotion policy",
    )
    guardrail_values = expect_list(
        value["guardrails"],
        "benchmark promotion policy guardrails",
    )
    if len(guardrail_values) > MAX_POLICY_GUARDRAILS:
        raise BenchmarkDataError(
            "benchmark promotion policy guardrails exceed the supported limit"
        )
    guardrails = tuple(
        _guardrail_from_object(
            expect_object(item, f"promotion policy guardrails[{index}]")
        )
        for index, item in enumerate(guardrail_values)
    )
    return PromotionPolicy(
        policy_id=value["policy_id"],  # type: ignore[arg-type]
        minimum_completed_cases=value["minimum_completed_cases"],  # type: ignore[arg-type]
        require_no_errors=value["require_no_errors"],  # type: ignore[arg-type]
        guardrails=guardrails,
        allowed_differences=_allowed_differences_from_object(
            expect_object(
                value["allowed_differences"],
                "benchmark promotion policy allowed_differences",
            )
        ),
    )


def _allowed_differences_from_object(
    value: dict[str, object],
) -> AllowedRunDifferences:
    expect_fields(
        value,
        {
            "allow_code_sha_difference",
            "allow_benchmark_profile_difference",
            "allow_benchmark_search_plan_difference",
            "model_component_ids",
            "index_component_ids",
            "config_component_ids",
        },
        "benchmark promotion policy allowed_differences",
    )
    return AllowedRunDifferences(
        allow_code_sha_difference=value["allow_code_sha_difference"],  # type: ignore[arg-type]
        allow_benchmark_profile_difference=value[
            "allow_benchmark_profile_difference"
        ],  # type: ignore[arg-type]
        allow_benchmark_search_plan_difference=value[
            "allow_benchmark_search_plan_difference"
        ],  # type: ignore[arg-type]
        model_component_ids=_component_ids(
            value["model_component_ids"],
            "model_component_ids",
        ),
        index_component_ids=_component_ids(
            value["index_component_ids"],
            "index_component_ids",
        ),
        config_component_ids=_component_ids(
            value["config_component_ids"],
            "config_component_ids",
        ),
    )


def _component_ids(value: object, field_name: str) -> tuple[object, ...]:
    values = expect_list(
        value,
        f"benchmark promotion policy allowed_differences {field_name}",
    )
    if len(values) > MAX_ALLOWED_COMPONENT_IDS:
        raise BenchmarkDataError(
            "benchmark promotion policy allowed component IDs exceed the "
            "supported limit"
        )
    return tuple(values)


def _guardrail_from_object(value: dict[str, object]) -> MetricGuardrail:
    expect_fields(
        value,
        {
            "metric_name",
            "direction",
            "allowed_regression",
            "absolute_threshold",
        },
        "promotion policy guardrail",
    )
    return MetricGuardrail(
        metric_name=value["metric_name"],  # type: ignore[arg-type]
        direction=value["direction"],  # type: ignore[arg-type]
        allowed_regression=value["allowed_regression"],  # type: ignore[arg-type]
        absolute_threshold=value["absolute_threshold"],  # type: ignore[arg-type]
    )
