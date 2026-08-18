from dataclasses import FrozenInstanceError, replace

import pytest

from videoscope.benchmark import BenchmarkDataError
from videoscope.benchmark.profiles import (
    BenchmarkProfile,
    EvaluationModalityWeight,
    EvaluationSearchPlan,
    FROZEN_PROFILES,
    get_profile,
)


def test_approved_ablation_profiles_are_named_and_frozen() -> None:
    assert tuple(FROZEN_PROFILES) == (
        "lexical_qdrant",
        "dense_siglip",
        "temporal_refinement",
        "lighthouse",
        "qwen_verification",
        "internvideo",
    )
    assert get_profile("dense_siglip").required_capabilities == (
        "text_vectors",
        "visual_dense",
    )
    assert get_profile("lighthouse").required_capabilities == (
        "text_vectors",
        "visual_dense",
        "temporal_refinement",
        "lighthouse",
    )

    with pytest.raises(TypeError):
        FROZEN_PROFILES["new"] = get_profile("dense_siglip")  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        get_profile("dense_siglip").profile_id = "changed"  # type: ignore[misc]


def test_profile_identity_changes_with_contract_version() -> None:
    plan = get_profile("dense_siglip").search_plan
    first = BenchmarkProfile("profile", 1, ("visual_dense",), plan)
    second = BenchmarkProfile("profile", 2, ("visual_dense",), plan)

    assert first.identity != second.identity


@pytest.mark.parametrize(
    "profile",
    [
        BenchmarkProfile,
    ],
)
def test_profile_contract_rejects_ambiguous_values(profile) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(BenchmarkDataError):
        profile("bad/path", 1, (), 20)
    with pytest.raises(BenchmarkDataError):
        profile("profile", 0, (), 20)
    with pytest.raises(BenchmarkDataError):
        profile("profile", 1, ("visual", "visual"), 20)
    with pytest.raises(BenchmarkDataError):
        profile("profile", 1, (), 20)
    with pytest.raises(BenchmarkDataError):
        profile("profile", 1, (), 0)


def test_unknown_profile_fails_closed() -> None:
    with pytest.raises(BenchmarkDataError, match="unknown benchmark profile"):
        get_profile("unknown")


def test_frozen_profiles_publish_exact_search_plans() -> None:
    baseline = get_profile("lexical_qdrant")
    dense = get_profile("dense_siglip")
    refined = get_profile("temporal_refinement")
    lighthouse = get_profile("lighthouse")
    qwen = get_profile("qwen_verification")
    internvideo = get_profile("internvideo")

    assert baseline.search_plan.text_search == "lexical_and_semantic"
    assert baseline.search_plan.modalities == ("objects", "ocr", "speech")
    assert baseline.search_plan.visual_search == "disabled"
    assert dense.search_plan.visual_search == "dense_siglip"
    assert dense.search_plan.text_search == "lexical_and_semantic"
    assert dense.search_plan.modalities == (
        "objects",
        "ocr",
        "speech",
        "visual",
    )
    assert dense.search_plan.temporal_refinement is False
    assert refined.search_plan.temporal_refinement is True
    assert refined.required_capabilities == (
        "text_vectors",
        "visual_dense",
        "temporal_refinement",
    )
    assert lighthouse.search_plan.lighthouse is True
    assert lighthouse.required_capabilities == (
        "text_vectors",
        "visual_dense",
        "temporal_refinement",
        "lighthouse",
    )
    assert qwen.search_plan.reranker == "qwen"
    assert qwen.search_plan.lighthouse is True
    assert qwen.search_plan.reranker_trigger == "all_candidates"
    assert qwen.search_plan.reranker_candidate_limit == 12
    assert internvideo.search_plan.reranker == "internvideo"
    assert internvideo.search_plan.lighthouse is True
    assert internvideo.search_plan.reranker_trigger == "all_candidates"
    assert internvideo.search_plan.reranker_candidate_limit == 4
    assert all(
        profile.search_plan.reranker_candidate_limit == 0
        for profile in (baseline, dense, refined, lighthouse)
    )
    expected_text_weights = {
        "objects": 0.92,
        "ocr": 0.88,
        "speech": 1.0,
    }
    for profile in (dense, refined, lighthouse, qwen, internvideo):
        weights = {
            item.modality: item.weight
            for item in profile.search_plan.modality_weights
        }
        assert all(weights[name] == value for name, value in expected_text_weights.items())
    with pytest.raises(BenchmarkDataError, match="candidate_limit"):
        replace(qwen.search_plan, reranker_candidate_limit=20)
    with pytest.raises(BenchmarkDataError, match="candidate_limit"):
        replace(internvideo.search_plan, reranker_candidate_limit=12)
    assert len({profile.search_plan.identity for profile in FROZEN_PROFILES.values()}) == 6


def test_search_plan_identity_is_canonical_and_contract_is_deeply_immutable() -> None:
    first = EvaluationSearchPlan(
        schema_version=1,
        modalities=("speech", "ocr", "objects"),
        modality_weights=(
            EvaluationModalityWeight("speech", 1.0),
            EvaluationModalityWeight("ocr", 0.88),
            EvaluationModalityWeight("objects", 0.92),
        ),
        text_search="lexical_and_semantic",
        visual_search="disabled",
        temporal_refinement=False,
        lighthouse=False,
        reranker="none",
        reranker_trigger="disabled",
        reranker_candidate_limit=0,
        result_limit=20,
    )
    second = EvaluationSearchPlan(
        schema_version=1,
        modalities=("objects", "speech", "ocr"),
        modality_weights=(
            EvaluationModalityWeight("objects", 0.92),
            EvaluationModalityWeight("speech", 1.0),
            EvaluationModalityWeight("ocr", 0.88),
        ),
        text_search="lexical_and_semantic",
        visual_search="disabled",
        temporal_refinement=False,
        lighthouse=False,
        reranker="none",
        reranker_trigger="disabled",
        reranker_candidate_limit=0,
        result_limit=20,
    )

    assert first == second
    assert first.identity == second.identity
    with pytest.raises(FrozenInstanceError):
        first.reranker = "qwen"  # type: ignore[misc]


@pytest.mark.parametrize(
    "change",
    [
        {"modalities": ("visual",), "text_search": "lexical_and_semantic"},
        {"visual_search": "disabled", "temporal_refinement": True},
        {"visual_search": "disabled", "lighthouse": True},
        {"reranker": "none", "reranker_trigger": "all_candidates"},
        {"reranker": "qwen", "reranker_trigger": "disabled"},
        {"reranker": "qwen", "reranker_trigger": "all_candidates"},
    ],
)
def test_search_plan_rejects_internally_inconsistent_semantics(change) -> None:  # type: ignore[no-untyped-def]
    values = {
        "schema_version": 1,
        "modalities": ("visual",),
        "modality_weights": (EvaluationModalityWeight("visual", 1.08),),
        "text_search": "disabled",
        "visual_search": "dense_siglip",
        "temporal_refinement": False,
        "lighthouse": False,
        "reranker": "none",
        "reranker_trigger": "disabled",
        "reranker_candidate_limit": 0,
        "result_limit": 20,
    }
    values.update(change)

    with pytest.raises(BenchmarkDataError):
        EvaluationSearchPlan(**values)
