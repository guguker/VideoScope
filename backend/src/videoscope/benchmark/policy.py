from __future__ import annotations

from pathlib import Path

from .comparison import MetricGuardrail, PromotionPolicy
from .schema import BenchmarkDataError
from .serialization import expect_fields, expect_list, expect_object, parse_json_object
from .storage import _read_bounded_file


POLICY_SCHEMA_VERSION = 1
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
    expect_fields(
        value,
        {
            "schema_version",
            "policy_id",
            "minimum_completed_cases",
            "require_no_errors",
            "guardrails",
        },
        "benchmark promotion policy",
    )
    schema_version = value["schema_version"]
    if type(schema_version) is not int or schema_version != POLICY_SCHEMA_VERSION:
        raise BenchmarkDataError(
            f"promotion policy schema_version must be {POLICY_SCHEMA_VERSION}"
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
    )


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
