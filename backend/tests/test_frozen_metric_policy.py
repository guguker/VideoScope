from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pytest

from videoscope.benchmark.metric_policy import (
    MAX_FROZEN_METRIC_POLICY_BYTES,
    frozen_metric_policy_from_dict,
    frozen_metric_policy_revision,
    load_frozen_metric_policy,
    main,
    validate_frozen_metric_policy_dataset,
    validate_frozen_metric_policy_product_dataset,
)
from videoscope.benchmark.profiles import FROZEN_PROFILES
from videoscope.benchmark.serialization import dataset_revision
from videoscope.benchmark.storage import load_dataset
from videoscope.benchmark.video_verifier_schema import (
    video_verifier_dataset_from_json,
    video_verifier_dataset_revision,
)


ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "docs/benchmarks/policies/phase0-regression-v1.json"
DATASET_PATH = ROOT / "docs/benchmarks/video-verifier/seed-v1.json"
PRODUCT_DATASET_PATH = (
    ROOT / "docs/benchmarks/product-retrieval/seed-v1.json"
)


def _policy_value() -> dict[str, object]:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def _write_mutation(tmp_path: Path, mutate) -> Path:  # type: ignore[no-untyped-def]
    value = _policy_value()
    mutate(value)
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _change_guardrail(
    value: dict[str, object],
    guardrail_id: str,
    field: str,
    replacement: object,
) -> None:
    hard_guardrails = value["hard_guardrails"]
    assert isinstance(hard_guardrails, dict)
    metrics = hard_guardrails["metrics"]
    assert isinstance(metrics, list)
    guardrail = next(
        item
        for item in metrics
        if isinstance(item, dict) and item.get("guardrail_id") == guardrail_id
    )
    guardrail[field] = replacement


def test_committed_phase0_policy_is_current_but_never_promotion_evidence() -> None:
    policy = load_frozen_metric_policy(POLICY_PATH)
    dataset = video_verifier_dataset_from_json(DATASET_PATH.read_bytes())
    product_dataset = load_dataset(PRODUCT_DATASET_PATH)

    validate_frozen_metric_policy_dataset(policy, dataset)
    validate_frozen_metric_policy_product_dataset(policy, product_dataset)

    assert policy.policy_id == "phase0-regression-baseline"
    assert policy.policy_version == "1.0.0"
    assert policy.evaluation_data.status == "regression_seen_only"
    assert policy.evaluation_data.evidence_use == "regression_only"
    assert (
        policy.evaluation_data.promotion_evidence_status
        == "not_promotion_evidence"
    )
    assert policy.evaluation_data.promotion_eligible is False
    assert policy.evaluation_data.product_dataset_status == "frozen_regression_seen"
    assert policy.evaluation_data.product_dataset.evidence_use == "regression_only"
    assert policy.evaluation_data.product_dataset.case_count == 5
    assert policy.evaluation_data.product_dataset.source_group_count == 2
    assert (
        policy.evaluation_data.product_dataset.dataset_revision
        == dataset_revision(product_dataset)
    )
    assert policy.evaluation_data.verifier_dataset.splits == ("regression_seen",)
    assert policy.evaluation_data.verifier_dataset.case_count == 10
    assert policy.evaluation_data.verifier_dataset.source_group_count == 2
    assert (
        policy.evaluation_data.verifier_dataset.dataset_revision
        == video_verifier_dataset_revision(dataset)
    )
    assert {
        item.profile_id: (item.profile_identity, item.search_plan_identity)
        for item in policy.profiles
    } == {
        profile_id: (profile.identity, profile.search_plan.identity)
        for profile_id, profile in FROZEN_PROFILES.items()
    }
    assert {metric.metric_name for metric in policy.primary_metrics} == {
        "candidate_recall_at_50",
        "error_count",
        "hard_negative_hit_rate",
        "infrastructure_error_count",
        "mean_boundary_error_seconds",
        "model_miss_count",
        "ndcg_at_10",
        "p95_latency_ms",
        "precision_at_5",
        "recall_at_10",
        "recall_at_20",
        "sampled_peak_process_tree_rss_bytes",
    }
    assert {item.slice_id for item in policy.critical_slices} == {
        "advertisement_non_game",
        "different_camera_or_league",
        "free_throw",
        "low_resolution",
        "made_2",
        "made_3",
        "miss",
        "ood_non_basketball",
        "replay",
        "scoreboard_hidden",
        "universal_non_sports",
    }
    assert policy.minimum_useful_effect.combination == "any"
    assert {
        item.metric_name: item.minimum_absolute_improvement
        for item in policy.minimum_useful_effect.effects
    } == {"ndcg_at_10": 0.03, "recall_at_10": 0.03}
    assert policy.uncertainty.method == "source_group_bootstrap"
    assert policy.uncertainty.confidence_level == 0.95
    assert policy.uncertainty.decision_when_unavailable == "insufficient_evidence"
    assert frozen_metric_policy_revision(policy) == frozen_metric_policy_revision(
        load_frozen_metric_policy(POLICY_PATH)
    )
    with pytest.raises(FrozenInstanceError):
        policy.policy_id = "changed"  # type: ignore[misc]


def test_captured_policy_object_has_a_public_single_read_parser() -> None:
    captured = _policy_value()

    parsed = frozen_metric_policy_from_dict(captured)

    assert parsed == load_frozen_metric_policy(POLICY_PATH)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value["evaluation_data"].__setitem__(  # type: ignore[union-attr]
            "promotion_eligible", True
        ),
        lambda value: value["evaluation_data"]["verifier_dataset"].__setitem__(  # type: ignore[index,union-attr]
            "splits", ["promotion_holdout"]
        ),
        lambda value: value["evaluation_data"].__setitem__(  # type: ignore[union-attr]
            "evidence_use", "promotion"
        ),
        lambda value: value["evaluation_data"].__setitem__(  # type: ignore[union-attr]
            "promotion_evidence_status", "promotion_evidence"
        ),
    ),
)
def test_regression_policy_rejects_any_promotion_claim(
    tmp_path: Path,
    mutate,
) -> None:  # type: ignore[no-untyped-def]
    path = _write_mutation(tmp_path, mutate)

    with pytest.raises(ValueError, match="regression|promotion"):
        load_frozen_metric_policy(path)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value["contracts"].__setitem__(  # type: ignore[union-attr]
            "benchmark_methodology_identity", "stale-methodology"
        ),
        lambda value: value["contracts"].__setitem__(  # type: ignore[union-attr]
            "benchmark_run_schema_version", 999
        ),
        lambda value: value["profiles"][0].__setitem__(  # type: ignore[index,union-attr]
            "profile_identity", "stale-profile"
        ),
    ),
)
def test_loader_fails_closed_when_frozen_runtime_contract_is_stale(
    tmp_path: Path,
    mutate,
) -> None:  # type: ignore[no-untyped-def]
    path = _write_mutation(tmp_path, mutate)

    with pytest.raises(ValueError, match="current|identity|schema"):
        load_frozen_metric_policy(path)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value["primary_metrics"].pop(),  # type: ignore[union-attr]
        lambda value: value["critical_slices"].pop(),  # type: ignore[union-attr]
        lambda value: value["hard_guardrails"]["conditions"].pop(),  # type: ignore[index,union-attr]
        lambda value: value["minimum_useful_effect"]["effects"].pop(),  # type: ignore[index,union-attr]
    ),
)
def test_loader_rejects_silently_weakened_phase0_policy(
    tmp_path: Path,
    mutate,
) -> None:  # type: ignore[no-untyped-def]
    path = _write_mutation(tmp_path, mutate)

    with pytest.raises(ValueError, match="required|exactly"):
        load_frozen_metric_policy(path)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: _change_guardrail(
            value,
            "candidate_recall_floor",
            "absolute_threshold",
            0.94,
        ),
        lambda value: _change_guardrail(
            value,
            "critical_recall_regression",
            "maximum_absolute_regression",
            0.06,
        ),
        lambda value: value["uncertainty"].__setitem__(  # type: ignore[union-attr]
            "confidence_level", 0.90
        ),
        lambda value: value["uncertainty"].__setitem__(  # type: ignore[union-attr]
            "minimum_completed_gold_cases_per_critical_slice", 19
        ),
    ),
)
def test_loader_rejects_weakened_threshold_or_uncertainty_rule(
    tmp_path: Path,
    mutate,
) -> None:  # type: ignore[no-untyped-def]
    path = _write_mutation(tmp_path, mutate)

    with pytest.raises(ValueError, match="required|guardrail"):
        load_frozen_metric_policy(path)


def test_policy_dataset_binding_rejects_a_changed_fixture(tmp_path: Path) -> None:
    policy = load_frozen_metric_policy(POLICY_PATH)
    value = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    value["description"] += " changed"
    changed = video_verifier_dataset_from_json(json.dumps(value))

    with pytest.raises(ValueError, match="revision"):
        validate_frozen_metric_policy_dataset(policy, changed)


def test_policy_product_dataset_binding_rejects_a_changed_fixture(
    tmp_path: Path,
) -> None:
    policy = load_frozen_metric_policy(POLICY_PATH)
    value = json.loads(PRODUCT_DATASET_PATH.read_text(encoding="utf-8"))
    value["description"] += " changed"
    path = tmp_path / "changed-product.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="revision"):
        validate_frozen_metric_policy_product_dataset(policy, load_dataset(path))


def test_policy_loader_rejects_unknown_duplicate_nonfinite_and_unsafe_files(
    tmp_path: Path,
) -> None:
    unknown = _write_mutation(tmp_path, lambda value: value.update({"path": "/tmp"}))
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"schema_version":1,"value":NaN}', encoding="utf-8")
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * (MAX_FROZEN_METRIC_POLICY_BYTES + 1))
    linked = tmp_path / "linked.json"
    linked.symlink_to(POLICY_PATH)

    for path in (unknown, duplicate, nonfinite, oversized, linked):
        with pytest.raises(ValueError):
            load_frozen_metric_policy(path)


def test_policy_validation_cli_is_machine_readable_and_sanitized(
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    exit_code = main(
        [
            "validate",
            "--policy",
            str(POLICY_PATH),
            "--product-dataset",
            str(PRODUCT_DATASET_PATH),
            "--verifier-dataset",
            str(DATASET_PATH),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    result = json.loads(captured.out)
    assert result == {
        "dataset_revision": video_verifier_dataset_revision(
            video_verifier_dataset_from_json(DATASET_PATH.read_bytes())
        ),
        "product_dataset_revision": dataset_revision(
            load_dataset(PRODUCT_DATASET_PATH)
        ),
        "policy_id": "phase0-regression-baseline",
        "policy_revision": result["policy_revision"],
        "promotion_eligible": False,
        "status": "valid",
    }
    assert len(result["policy_revision"]) == 64
    assert str(ROOT) not in captured.out
