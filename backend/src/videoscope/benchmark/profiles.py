from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .schema import BenchmarkDataError, _require_id, _require_positive_int, _require_tuple


@dataclass(frozen=True, slots=True)
class BenchmarkProfile:
    profile_id: str
    schema_version: int
    required_capabilities: tuple[str, ...]
    result_limit: int

    def __post_init__(self) -> None:
        _require_id(self.profile_id, "benchmark profile id")
        _require_positive_int(self.schema_version, "benchmark profile schema_version")
        capabilities = _require_tuple(
            self.required_capabilities,
            "benchmark profile required_capabilities",
        )
        if not capabilities:
            raise BenchmarkDataError(
                "benchmark profile required_capabilities must not be empty"
            )
        for capability in capabilities:
            _require_id(capability, "benchmark profile capability")
        if len(capabilities) != len(set(capabilities)):
            raise BenchmarkDataError(
                "benchmark profile required_capabilities must contain unique values"
            )
        _require_positive_int(self.result_limit, "benchmark profile result_limit")

    @property
    def identity(self) -> str:
        return f"{self.profile_id}@{self.schema_version}"


_PROFILE_VALUES = (
    BenchmarkProfile("lexical_qdrant", 1, ("text_vectors",), 20),
    BenchmarkProfile("dense_siglip", 1, ("visual_dense",), 20),
    BenchmarkProfile("temporal_refinement", 1, ("visual_dense",), 20),
    BenchmarkProfile(
        "lighthouse",
        1,
        ("visual_dense", "lighthouse"),
        20,
    ),
    BenchmarkProfile(
        "qwen_verification",
        1,
        ("visual_dense", "qwen_verification"),
        20,
    ),
    BenchmarkProfile(
        "internvideo",
        1,
        ("visual_dense", "internvideo"),
        20,
    ),
)

FROZEN_PROFILES: Mapping[str, BenchmarkProfile] = MappingProxyType(
    {profile.profile_id: profile for profile in _PROFILE_VALUES}
)


def get_profile(profile_id: str) -> BenchmarkProfile:
    _require_id(profile_id, "benchmark profile id")
    try:
        return FROZEN_PROFILES[profile_id]
    except KeyError:
        raise BenchmarkDataError(
            f"unknown benchmark profile {profile_id!r}"
        ) from None
