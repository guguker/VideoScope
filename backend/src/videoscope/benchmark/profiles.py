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
PROFILE_IDENTITY_CONTRACT_SCHEMA_VERSION = 1
PROFILE_IDENTITY_CONTRACT_COMPONENT_ID = "benchmark_profile_identity_contract"
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


@dataclass(frozen=True, slots=True)
class ProfileIdentityExpectation:
    role: Literal["model", "index", "config"]
    component_id: str
    value_contract: Literal[
        "benchmark_environment_v2",
        "evaluation_search_configuration",
        "fastembed_mpnet_v1",
        "internvideo_not_configured",
        "lifecycle",
        "provider_digest_or_not_configured",
        "qwen_verifier_or_not_configured",
        "reviewed_siglip_or_not_configured",
        "sha256",
        "sha256_prefixed",
    ]

    def __post_init__(self) -> None:
        if self.role not in {"model", "index", "config"}:
            raise BenchmarkDataError("profile identity expectation role is invalid")
        _require_id(
            self.component_id,
            "profile identity expectation component_id",
        )
        if self.value_contract not in {
            "benchmark_environment_v2",
            "evaluation_search_configuration",
            "fastembed_mpnet_v1",
            "internvideo_not_configured",
            "lifecycle",
            "provider_digest_or_not_configured",
            "qwen_verifier_or_not_configured",
            "reviewed_siglip_or_not_configured",
            "sha256",
            "sha256_prefixed",
        }:
            raise BenchmarkDataError(
                "profile identity expectation value_contract is invalid"
            )


@dataclass(frozen=True, slots=True)
class ProfileIdentityContract:
    """Exact product identity surface required by one frozen profile.

    The contract intentionally describes roles as well as component names.  A
    component moved from ``model`` to ``index`` is therefore drift, even when
    the union of names is unchanged.
    """

    schema_version: int
    profile_id: str
    profile_identity: str
    search_plan_identity: str
    expectations: tuple[ProfileIdentityExpectation, ...]

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != PROFILE_IDENTITY_CONTRACT_SCHEMA_VERSION
        ):
            raise BenchmarkDataError(
                "profile identity contract schema_version must be "
                f"{PROFILE_IDENTITY_CONTRACT_SCHEMA_VERSION}"
            )
        _require_id(self.profile_id, "profile identity contract profile_id")
        if not isinstance(self.profile_identity, str) or not self.profile_identity:
            raise BenchmarkDataError(
                "profile identity contract profile_identity is invalid"
            )
        if (
            not isinstance(self.search_plan_identity, str)
            or not self.search_plan_identity
        ):
            raise BenchmarkDataError(
                "profile identity contract search_plan_identity is invalid"
            )
        expectations = _require_tuple(
            self.expectations,
            "profile identity contract expectations",
        )
        if not expectations or any(
            not isinstance(item, ProfileIdentityExpectation)
            for item in expectations
        ):
            raise BenchmarkDataError(
                "profile identity contract expectations are invalid"
            )
        keys = tuple((item.role, item.component_id) for item in expectations)
        if len(keys) != len(set(keys)):
            raise BenchmarkDataError(
                "profile identity contract expectations must be unique by role"
            )
        component_ids = tuple(item.component_id for item in expectations)
        if len(component_ids) != len(set(component_ids)):
            raise BenchmarkDataError(
                "profile identity contract component ids must have one exact role"
            )
        object.__setattr__(
            self,
            "expectations",
            tuple(sorted(expectations, key=lambda item: (item.role, item.component_id))),
        )

    def component_ids(
        self,
        role: Literal["model", "index", "config"],
    ) -> tuple[str, ...]:
        return tuple(
            item.component_id for item in self.expectations if item.role == role
        )

    @property
    def canonical_json(self) -> str:
        return json.dumps(
            {
                "expectations": [
                    {
                        "component_id": item.component_id,
                        "role": item.role,
                        "value_contract": item.value_contract,
                    }
                    for item in self.expectations
                ],
                "profile_id": self.profile_id,
                "profile_identity": self.profile_identity,
                "schema_version": self.schema_version,
                "search_plan_identity": self.search_plan_identity,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @property
    def identity(self) -> str:
        digest = sha256(self.canonical_json.encode("utf-8")).hexdigest()
        return f"benchmark-profile-identity-contract@{self.schema_version}:{digest}"


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
    50,
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
    50,
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
    50,
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
    50,
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
    50,
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
    50,
)


_PROFILE_VALUES = (
    BenchmarkProfile("lexical_qdrant", 2, ("text_vectors",), _TEXT_PLAN),
    BenchmarkProfile(
        "dense_siglip",
        2,
        ("text_vectors", "visual_dense"),
        _DENSE_PLAN,
    ),
    BenchmarkProfile(
        "temporal_refinement",
        2,
        ("text_vectors", "visual_dense", "temporal_refinement"),
        _REFINED_PLAN,
    ),
    BenchmarkProfile(
        "lighthouse",
        2,
        ("text_vectors", "visual_dense", "temporal_refinement", "lighthouse"),
        _LIGHTHOUSE_PLAN,
    ),
    BenchmarkProfile(
        "qwen_verification",
        2,
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
        2,
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


def profile_identity_contract(profile: BenchmarkProfile) -> ProfileIdentityContract:
    """Return the canonical execution-identity contract for a frozen profile."""

    if not isinstance(profile, BenchmarkProfile):
        raise BenchmarkDataError("benchmark profile identity contract is invalid")
    frozen = FROZEN_PROFILES.get(profile.profile_id)
    if frozen is None or frozen != profile:
        raise BenchmarkDataError(
            "benchmark profile identity contract requires an exact frozen profile"
        )
    expectations: list[ProfileIdentityExpectation] = []
    plan = profile.search_plan
    if plan.text_search != "disabled":
        expectations.extend(
            (
                ProfileIdentityExpectation(
                    "model",
                    "text_embedding",
                    "fastembed_mpnet_v1",
                ),
                ProfileIdentityExpectation("index", "text_vector_index", "sha256"),
                ProfileIdentityExpectation(
                    "index",
                    "text_vector_generations",
                    "sha256_prefixed",
                ),
            )
        )
    if plan.visual_search != "disabled":
        expectations.extend(
            (
                ProfileIdentityExpectation(
                    "model",
                    "visual_embedding",
                    "reviewed_siglip_or_not_configured",
                ),
                ProfileIdentityExpectation(
                    "index",
                    "visual_generations",
                    "sha256_prefixed",
                ),
            )
        )
    if plan.lighthouse:
        expectations.extend(
            (
                ProfileIdentityExpectation(
                    "model",
                    "lighthouse_model",
                    "provider_digest_or_not_configured",
                ),
                ProfileIdentityExpectation(
                    "index",
                    "lighthouse_generations",
                    "sha256_prefixed",
                ),
            )
        )
    if plan.reranker != "none":
        expectations.append(
            ProfileIdentityExpectation(
                "model",
                f"{plan.reranker}_reranker",
                (
                    "qwen_verifier_or_not_configured"
                    if plan.reranker == "qwen"
                    else "internvideo_not_configured"
                ),
            )
        )
    expectations.extend(
        (
            ProfileIdentityExpectation(
                "config",
                "benchmark_product_environment",
                "benchmark_environment_v2",
            ),
            ProfileIdentityExpectation(
                "config",
                "evaluation_search_configuration",
                "evaluation_search_configuration",
            ),
            ProfileIdentityExpectation(
                "config",
                "product_search_lifecycle",
                "lifecycle",
            ),
            ProfileIdentityExpectation(
                "config",
                "product_search_runtime",
                "sha256_prefixed",
            ),
        )
    )
    return ProfileIdentityContract(
        schema_version=PROFILE_IDENTITY_CONTRACT_SCHEMA_VERSION,
        profile_id=profile.profile_id,
        profile_identity=profile.identity,
        search_plan_identity=profile.search_plan.identity,
        expectations=tuple(expectations),
    )


FROZEN_PROFILE_IDENTITY_CONTRACTS: Mapping[str, ProfileIdentityContract] = (
    MappingProxyType(
        {
            profile_id: profile_identity_contract(profile)
            for profile_id, profile in FROZEN_PROFILES.items()
        }
    )
)


def get_profile(profile_id: str) -> BenchmarkProfile:
    _require_id(profile_id, "benchmark profile id")
    try:
        return FROZEN_PROFILES[profile_id]
    except KeyError:
        raise BenchmarkDataError(
            f"unknown benchmark profile {profile_id!r}"
        ) from None
