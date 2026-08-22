from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Literal, Mapping

from .schema import (
    BenchmarkDataError,
    _require_finite_float,
    _require_id,
    _require_nonnegative_int,
    _require_positive_int,
    _require_tuple,
)


EVALUATION_SEARCH_PLAN_SCHEMA_VERSION = 1
_TEXT_MODALITIES = frozenset({"objects", "ocr", "speech"})
_SEARCH_MODALITIES = _TEXT_MODALITIES | {"visual", "lighthouse"}


@dataclass(frozen=True, slots=True)
class EvaluationModalityWeight:
    modality: str
    weight: float

    def __post_init__(self) -> None:
        _require_id(self.modality, "evaluation modality weight modality")
        if self.modality not in _SEARCH_MODALITIES:
            raise BenchmarkDataError(
                f"unsupported evaluation search modality {self.modality!r}"
            )
        weight = _require_finite_float(
            self.weight,
            "evaluation modality weight",
            strictly_positive=True,
        )
        if weight > 100:
            raise BenchmarkDataError("evaluation modality weight must not exceed 100")
        object.__setattr__(self, "weight", weight)


@dataclass(frozen=True, slots=True)
class EvaluationSearchPlan:
    """Frozen product-search semantics for one benchmark ablation.

    The future concrete adapter must translate this value into the product search
    boundary.  Keeping the plan in the portable benchmark package makes profile
    meaning reviewable without importing a provider or constructing a runtime.
    """

    schema_version: int
    modalities: tuple[str, ...]
    modality_weights: tuple[EvaluationModalityWeight, ...]
    text_search: Literal["disabled", "lexical_and_semantic"]
    visual_search: Literal["disabled", "dense_siglip"]
    temporal_refinement: bool
    lighthouse: bool
    reranker: Literal["none", "qwen", "internvideo"]
    reranker_trigger: Literal["disabled", "all_candidates"]
    reranker_candidate_limit: int
    result_limit: int

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != EVALUATION_SEARCH_PLAN_SCHEMA_VERSION
        ):
            raise BenchmarkDataError(
                "evaluation search plan schema_version must be "
                f"{EVALUATION_SEARCH_PLAN_SCHEMA_VERSION}"
            )
        modalities = _require_tuple(
            self.modalities,
            "evaluation search plan modalities",
        )
        if not modalities:
            raise BenchmarkDataError(
                "evaluation search plan modalities must not be empty"
            )
        for modality in modalities:
            _require_id(modality, "evaluation search plan modality")
            if modality not in _SEARCH_MODALITIES:
                raise BenchmarkDataError(
                    f"unsupported evaluation search modality {modality!r}"
                )
        if len(modalities) != len(set(modalities)):
            raise BenchmarkDataError(
                "evaluation search plan modalities must contain unique values"
            )
        canonical_modalities = tuple(sorted(modalities))
        object.__setattr__(self, "modalities", canonical_modalities)
        weights = _require_tuple(
            self.modality_weights,
            "evaluation search plan modality_weights",
        )
        if any(not isinstance(item, EvaluationModalityWeight) for item in weights):
            raise BenchmarkDataError(
                "evaluation search plan modality_weights must contain only "
                "EvaluationModalityWeight values"
            )
        weight_modalities = tuple(item.modality for item in weights)
        if len(weight_modalities) != len(set(weight_modalities)):
            raise BenchmarkDataError(
                "evaluation search plan modality_weights must be unique by modality"
            )
        if set(weight_modalities) != set(canonical_modalities):
            raise BenchmarkDataError(
                "evaluation search plan modalities and modality_weights disagree"
            )
        object.__setattr__(
            self,
            "modality_weights",
            tuple(sorted(weights, key=lambda item: item.modality)),
        )

        if self.text_search not in {"disabled", "lexical_and_semantic"}:
            raise BenchmarkDataError("evaluation search plan text_search is invalid")
        if self.visual_search not in {"disabled", "dense_siglip"}:
            raise BenchmarkDataError("evaluation search plan visual_search is invalid")
        for field_name in ("temporal_refinement", "lighthouse"):
            if type(getattr(self, field_name)) is not bool:
                raise BenchmarkDataError(
                    f"evaluation search plan {field_name} must be boolean"
                )
        if self.reranker not in {"none", "qwen", "internvideo"}:
            raise BenchmarkDataError("evaluation search plan reranker is invalid")
        if self.reranker_trigger not in {"disabled", "all_candidates"}:
            raise BenchmarkDataError(
                "evaluation search plan reranker_trigger is invalid"
            )
        _require_nonnegative_int(
            self.reranker_candidate_limit,
            "evaluation search plan reranker_candidate_limit",
        )
        _require_positive_int(
            self.result_limit,
            "evaluation search plan result_limit",
        )

        selected = set(canonical_modalities)
        has_text = bool(selected & _TEXT_MODALITIES)
        has_visual = "visual" in selected
        if (self.text_search == "disabled") == has_text:
            raise BenchmarkDataError(
                "evaluation search plan text modalities and text_search disagree"
            )
        if (self.visual_search == "disabled") == has_visual:
            raise BenchmarkDataError(
                "evaluation search plan visual modality and visual_search disagree"
            )
        if self.temporal_refinement and not has_visual:
            raise BenchmarkDataError(
                "temporal refinement requires dense visual search"
            )
        if self.lighthouse != ("lighthouse" in selected):
            raise BenchmarkDataError(
                "evaluation search plan lighthouse modality and flag disagree"
            )
        if self.lighthouse and not has_visual:
            raise BenchmarkDataError("Lighthouse requires dense visual search")
        if self.lighthouse and not self.temporal_refinement:
            raise BenchmarkDataError("Lighthouse requires temporal refinement")
        if self.reranker == "none":
            if (
                self.reranker_trigger != "disabled"
                or self.reranker_candidate_limit != 0
            ):
                raise BenchmarkDataError(
                    "a disabled reranker must use the disabled trigger and zero "
                    "candidate limit"
                )
        elif (
            self.reranker_trigger != "all_candidates"
            or not self.lighthouse
            or not self.temporal_refinement
        ):
            raise BenchmarkDataError(
                "a benchmark reranker requires the Lighthouse base and the "
                "all_candidates trigger"
            )
        expected_candidate_limit = {
            "none": 0,
            "qwen": 12,
            "internvideo": 4,
        }[self.reranker]
        if self.reranker_candidate_limit != expected_candidate_limit:
            raise BenchmarkDataError(
                f"{self.reranker} requires reranker_candidate_limit "
                f"{expected_candidate_limit}"
            )
        if self.reranker_candidate_limit > self.result_limit:
            raise BenchmarkDataError(
                "reranker_candidate_limit must not exceed result_limit"
            )

    @property
    def canonical_json(self) -> str:
        return json.dumps(
            {
                "lighthouse": self.lighthouse,
                "modalities": list(self.modalities),
                "modality_weights": [
                    {"modality": item.modality, "weight": item.weight}
                    for item in self.modality_weights
                ],
                "reranker": self.reranker,
                "reranker_candidate_limit": self.reranker_candidate_limit,
                "reranker_trigger": self.reranker_trigger,
                "result_limit": self.result_limit,
                "schema_version": self.schema_version,
                "temporal_refinement": self.temporal_refinement,
                "text_search": self.text_search,
                "visual_search": self.visual_search,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @property
    def identity(self) -> str:
        digest = sha256(self.canonical_json.encode("utf-8")).hexdigest()
        return f"evaluation-search-plan@{self.schema_version}:{digest}"


@dataclass(frozen=True, slots=True)
class BenchmarkProfile:
    profile_id: str
    schema_version: int
    required_capabilities: tuple[str, ...]
    search_plan: EvaluationSearchPlan

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
        if not isinstance(self.search_plan, EvaluationSearchPlan):
            raise BenchmarkDataError(
                "benchmark profile search_plan must be an EvaluationSearchPlan"
            )

    @property
    def result_limit(self) -> int:
        return self.search_plan.result_limit

    @property
    def canonical_json(self) -> str:
        return json.dumps(
            {
                "profile_id": self.profile_id,
                "required_capabilities": sorted(self.required_capabilities),
                "schema_version": self.schema_version,
                "search_plan_identity": self.search_plan.identity,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @property
    def identity(self) -> str:
        digest = sha256(self.canonical_json.encode("utf-8")).hexdigest()
        return f"{self.profile_id}@{self.schema_version}:{digest}"


_TEXT_PLAN = EvaluationSearchPlan(
    1,
    ("objects", "ocr", "speech"),
    (
        EvaluationModalityWeight("objects", 0.92),
        EvaluationModalityWeight("ocr", 0.88),
        EvaluationModalityWeight("speech", 1.0),
    ),
    "lexical_and_semantic",
    "disabled",
    False,
    False,
    "none",
    "disabled",
    0,
    20,
)
_DENSE_PLAN = EvaluationSearchPlan(
    1,
    ("objects", "ocr", "speech", "visual"),
    (
        *_TEXT_PLAN.modality_weights,
        EvaluationModalityWeight("visual", 1.08),
    ),
    "lexical_and_semantic",
    "dense_siglip",
    False,
    False,
    "none",
    "disabled",
    0,
    20,
)
_REFINED_PLAN = EvaluationSearchPlan(
    1,
    _DENSE_PLAN.modalities,
    _DENSE_PLAN.modality_weights,
    "lexical_and_semantic",
    "dense_siglip",
    True,
    False,
    "none",
    "disabled",
    0,
    20,
)
_LIGHTHOUSE_PLAN = EvaluationSearchPlan(
    1,
    (*_DENSE_PLAN.modalities, "lighthouse"),
    (
        *_DENSE_PLAN.modality_weights,
        EvaluationModalityWeight("lighthouse", 0.78),
    ),
    "lexical_and_semantic",
    "dense_siglip",
    True,
    True,
    "none",
    "disabled",
    0,
    20,
)
_QWEN_PLAN = EvaluationSearchPlan(
    1,
    _LIGHTHOUSE_PLAN.modalities,
    _LIGHTHOUSE_PLAN.modality_weights,
    "lexical_and_semantic",
    "dense_siglip",
    True,
    True,
    "qwen",
    "all_candidates",
    12,
    20,
)
_INTERNVIDEO_PLAN = EvaluationSearchPlan(
    1,
    _LIGHTHOUSE_PLAN.modalities,
    _LIGHTHOUSE_PLAN.modality_weights,
    "lexical_and_semantic",
    "dense_siglip",
    True,
    True,
    "internvideo",
    "all_candidates",
    4,
    20,
)


_PROFILE_VALUES = (
    BenchmarkProfile("lexical_qdrant", 1, ("text_vectors",), _TEXT_PLAN),
    BenchmarkProfile(
        "dense_siglip",
        1,
        ("text_vectors", "visual_dense"),
        _DENSE_PLAN,
    ),
    BenchmarkProfile(
        "temporal_refinement",
        1,
        ("text_vectors", "visual_dense", "temporal_refinement"),
        _REFINED_PLAN,
    ),
    BenchmarkProfile(
        "lighthouse",
        1,
        ("text_vectors", "visual_dense", "temporal_refinement", "lighthouse"),
        _LIGHTHOUSE_PLAN,
    ),
    BenchmarkProfile(
        "qwen_verification",
        1,
        (
            "text_vectors",
            "visual_dense",
            "temporal_refinement",
            "lighthouse",
            "qwen_verification",
        ),
        _QWEN_PLAN,
    ),
    BenchmarkProfile(
        "internvideo",
        1,
        (
            "text_vectors",
            "visual_dense",
            "temporal_refinement",
            "lighthouse",
            "internvideo",
        ),
        _INTERNVIDEO_PLAN,
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
