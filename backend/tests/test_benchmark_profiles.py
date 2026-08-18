from dataclasses import FrozenInstanceError

import pytest

from videoscope.benchmark import BenchmarkDataError
from videoscope.benchmark.profiles import (
    BenchmarkProfile,
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
    assert get_profile("dense_siglip").required_capabilities == ("visual_dense",)
    assert get_profile("lighthouse").required_capabilities == (
        "visual_dense",
        "lighthouse",
    )

    with pytest.raises(TypeError):
        FROZEN_PROFILES["new"] = get_profile("dense_siglip")  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        get_profile("dense_siglip").result_limit = 100  # type: ignore[misc]


def test_profile_identity_changes_with_contract_version() -> None:
    first = BenchmarkProfile("profile", 1, ("visual_dense",), 20)
    second = BenchmarkProfile("profile", 2, ("visual_dense",), 20)

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
