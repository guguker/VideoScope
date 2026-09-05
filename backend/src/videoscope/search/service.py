from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import logging
import math
from numbers import Real
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType
from typing import Literal, Protocol
from urllib.parse import quote

from videoscope.artifacts import (
    IndexingSpecifications,
    SegmentGeneration,
    StageKind,
    StageSpecification,
    TextVectorSearchBinding,
    TextVectorSearchHit,
)
from videoscope.repository import (
    ExternalIndexReleaseSnapshot,
    Repository,
    SegmentRecord,
)
from videoscope.search.fusion import EvidenceHit, calibrate_hits, fuse_hits
from videoscope.search.query_router import QueryPlan, QueryRouter
from videoscope.search.text_matching import SearchLexicon, lexical_match, normalize_text
from videoscope.search.text_matching import tokens as text_tokens


logger = logging.getLogger(__name__)

_TRUSTED_ENTITY_MATCHES = {"exact", "stem", "transliteration"}
_EVALUATION_MODALITIES = frozenset(
    {"objects", "ocr", "speech", "visual", "lighthouse"}
)
_EVALUATION_TEXT_MODALITIES = frozenset({"objects", "ocr", "speech"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_BENCHMARK_MAX_TEXT_POINTS = 200_000
_BENCHMARK_MAX_SEGMENTS = 200_000
_BENCHMARK_MAX_TEXT_BYTES = 64 * 1024 * 1024
_BENCHMARK_MAX_METADATA_BYTES = 64 * 1024 * 1024
_BENCHMARK_MAX_ASSETS = 128
_BENCHMARK_MAX_ASSET_BYTES = 16 * 1024**3
_BENCHMARK_MAX_MEDIA_BYTES = 128 * 1024**3
PRODUCT_SEARCH_EXECUTION_TRACE_SCHEMA_VERSION = 1
PRODUCT_SEARCH_EXECUTION_COUNT_MAX = (1 << 63) - 1
_PRODUCT_SEARCH_EXECUTION_COMPONENT_IDS = (
    "text_vectors",
    "lexical_text",
    "visual_dense",
    "temporal_refinement",
    "lighthouse",
    "qwen_verification",
)


def _canonical_digest(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("search identity is not canonical JSON") from error
    return sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class EvaluationSearchConfiguration:
    """Router-free execution contract translated from a frozen benchmark plan."""

    modalities: tuple[str, ...]
    modality_weights: tuple[tuple[str, float], ...]
    text_search: Literal["disabled", "lexical_and_semantic"]
    visual_search: Literal["disabled", "dense_siglip"]
    temporal_refinement: bool
    lighthouse: bool
    reranker: Literal["none", "qwen", "internvideo"]
    reranker_trigger: Literal["disabled", "all_candidates"]
    reranker_candidate_limit: int
    result_limit: int
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported evaluation search configuration schema")
        if (
            type(self.modalities) is not tuple
            or not self.modalities
            or any(
                type(item) is not str or item not in _EVALUATION_MODALITIES
                for item in self.modalities
            )
            or len(set(self.modalities)) != len(self.modalities)
            or self.modalities != tuple(sorted(self.modalities))
        ):
            raise ValueError("evaluation modalities must be a canonical tuple")
        if type(self.modality_weights) is not tuple or any(
            type(item) is not tuple or len(item) != 2
            for item in self.modality_weights
        ):
            raise ValueError("evaluation modality weights must be canonical pairs")
        weight_modalities: list[str] = []
        normalized_weights: list[tuple[str, float]] = []
        for modality, raw_weight in self.modality_weights:
            if (
                type(modality) is not str
                or modality not in _EVALUATION_MODALITIES
                or isinstance(raw_weight, bool)
                or not isinstance(raw_weight, (int, float))
                or not math.isfinite(float(raw_weight))
                or not 0 < float(raw_weight) <= 100
            ):
                raise ValueError("evaluation modality weight is invalid")
            weight_modalities.append(modality)
            normalized_weights.append((modality, float(raw_weight)))
        if (
            tuple(weight_modalities) != self.modalities
            or len(set(weight_modalities)) != len(weight_modalities)
        ):
            raise ValueError("evaluation modalities and weights disagree")
        object.__setattr__(self, "modality_weights", tuple(normalized_weights))
        if self.text_search not in {"disabled", "lexical_and_semantic"}:
            raise ValueError("unsupported evaluation text search policy")
        if self.visual_search not in {"disabled", "dense_siglip"}:
            raise ValueError("unsupported evaluation visual search policy")
        if type(self.temporal_refinement) is not bool or type(self.lighthouse) is not bool:
            raise ValueError("evaluation search flags must be boolean")
        if self.reranker not in {"none", "qwen", "internvideo"}:
            raise ValueError("unsupported evaluation reranker")
        if self.reranker_trigger not in {"disabled", "all_candidates"}:
            raise ValueError("unsupported evaluation reranker trigger")
        if (
            type(self.reranker_candidate_limit) is not int
            or self.reranker_candidate_limit < 0
            or type(self.result_limit) is not int
            or self.result_limit <= 0
        ):
            raise ValueError("evaluation search limits are invalid")

        selected = set(self.modalities)
        has_text = bool(selected & _EVALUATION_TEXT_MODALITIES)
        has_visual = "visual" in selected
        if (self.text_search == "disabled") == has_text:
            raise ValueError("evaluation text policy and modalities disagree")
        if (self.visual_search == "disabled") == has_visual:
            raise ValueError("evaluation visual policy and modalities disagree")
        if self.temporal_refinement and not has_visual:
            raise ValueError("temporal refinement requires visual search")
        if self.lighthouse != ("lighthouse" in selected):
            raise ValueError("Lighthouse flag and modality disagree")
        if self.lighthouse and not self.temporal_refinement:
            raise ValueError("Lighthouse requires temporal refinement")
        expected_limit = {"none": 0, "qwen": 12, "internvideo": 4}[self.reranker]
        if self.reranker_candidate_limit != expected_limit:
            raise ValueError("evaluation reranker candidate limit is not canonical")
        if self.reranker == "none":
            if self.reranker_trigger != "disabled":
                raise ValueError("disabled reranker requires disabled trigger")
        elif (
            self.reranker_trigger != "all_candidates"
            or not self.lighthouse
            or not self.temporal_refinement
        ):
            raise ValueError("evaluation reranker requires the full temporal base")
        if self.reranker_candidate_limit > self.result_limit:
            raise ValueError("evaluation reranker limit exceeds result limit")

    @property
    def canonical_json(self) -> str:
        return json.dumps(
            {
                "lighthouse": self.lighthouse,
                "modalities": list(self.modalities),
                "modality_weights": [
                    {"modality": modality, "weight": weight}
                    for modality, weight in self.modality_weights
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
        return f"evaluation-search-configuration@{self.schema_version}:{sha256(self.canonical_json.encode('utf-8')).hexdigest()}"


@dataclass(frozen=True, slots=True)
class ProductSearchExecutionTrace:
    """Immutable account of strict component calls made by one pinned search."""

    schema_version: int
    configuration_identity: str
    invoked_component_ids: tuple[str, ...]
    component_input_counts: tuple[tuple[str, int], ...]
    component_output_counts: tuple[tuple[str, int], ...]
    component_evidence_counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != PRODUCT_SEARCH_EXECUTION_TRACE_SCHEMA_VERSION
            or type(self.configuration_identity) is not str
            or re.fullmatch(
                r"evaluation-search-configuration@1:[0-9a-f]{64}",
                self.configuration_identity,
            )
            is None
        ):
            raise ValueError("product search execution trace identity is invalid")
        if (
            type(self.invoked_component_ids) is not tuple
            or any(
                type(component_id) is not str
                or component_id not in _PRODUCT_SEARCH_EXECUTION_COMPONENT_IDS
                for component_id in self.invoked_component_ids
            )
            or len(set(self.invoked_component_ids))
            != len(self.invoked_component_ids)
            or self.invoked_component_ids
            != tuple(
                component_id
                for component_id in _PRODUCT_SEARCH_EXECUTION_COMPONENT_IDS
                if component_id in self.invoked_component_ids
            )
        ):
            raise ValueError("product search invoked components are invalid")
        counts_by_kind = tuple(
            self._validate_counts(values, kind)
            for values, kind in (
                (self.component_input_counts, "input"),
                (self.component_output_counts, "output"),
                (self.component_evidence_counts, "evidence"),
            )
        )
        invoked = set(self.invoked_component_ids)
        if any(
            component_id not in invoked
            and any(counts[component_id] != 0 for counts in counts_by_kind)
            for component_id in _PRODUCT_SEARCH_EXECUTION_COMPONENT_IDS
        ):
            raise ValueError("uninvoked product search component has execution counts")

    @staticmethod
    def _validate_counts(
        values: tuple[tuple[str, int], ...],
        kind: str,
    ) -> dict[str, int]:
        if (
            type(values) is not tuple
            or len(values) != len(_PRODUCT_SEARCH_EXECUTION_COMPONENT_IDS)
        ):
            raise ValueError(f"product search component {kind} counts are invalid")
        output: dict[str, int] = {}
        for expected_component_id, item in zip(
            _PRODUCT_SEARCH_EXECUTION_COMPONENT_IDS,
            values,
            strict=True,
        ):
            if (
                type(item) is not tuple
                or len(item) != 2
                or item[0] != expected_component_id
                or type(item[1]) is not int
                or item[1] < 0
                or item[1] > PRODUCT_SEARCH_EXECUTION_COUNT_MAX
            ):
                raise ValueError(
                    f"product search component {kind} counts are invalid"
                )
            output[expected_component_id] = item[1]
        return output

    def component_input_count(self, component_id: str) -> int:
        return dict(self.component_input_counts).get(component_id, 0)

    def component_output_count(self, component_id: str) -> int:
        return dict(self.component_output_counts).get(component_id, 0)

    def component_evidence_count(self, component_id: str) -> int:
        return dict(self.component_evidence_counts).get(component_id, 0)


class _ProductSearchExecutionRecorder:
    """Single-search mutable recorder guarded by pinned-session identity."""

    def __init__(
        self,
        *,
        owner: object,
        configuration_identity: str,
    ) -> None:
        self._owner = owner
        self._configuration_identity = configuration_identity
        self._events: dict[str, tuple[int, int, int]] = {}

    def validate_owner(
        self,
        *,
        owner: object,
        configuration_identity: str,
    ) -> None:
        if (
            owner is not self._owner
            or configuration_identity != self._configuration_identity
        ):
            raise ValueError("pinned search execution recorder ownership is invalid")

    def record(
        self,
        component_id: str,
        *,
        input_count: int,
        output_count: int,
        evidence_count: int,
        owner: object,
    ) -> None:
        self.validate_owner(
            owner=owner,
            configuration_identity=self._configuration_identity,
        )
        if (
            component_id not in _PRODUCT_SEARCH_EXECUTION_COMPONENT_IDS
            or component_id in self._events
            or any(
                type(count) is not int or count < 0
                or count > PRODUCT_SEARCH_EXECUTION_COUNT_MAX
                for count in (input_count, output_count, evidence_count)
            )
        ):
            raise ValueError("product search execution event is invalid")
        self._events[component_id] = (
            input_count,
            output_count,
            evidence_count,
        )

    def finish(self, *, owner: object) -> ProductSearchExecutionTrace:
        self.validate_owner(
            owner=owner,
            configuration_identity=self._configuration_identity,
        )
        return ProductSearchExecutionTrace(
            schema_version=PRODUCT_SEARCH_EXECUTION_TRACE_SCHEMA_VERSION,
            configuration_identity=self._configuration_identity,
            invoked_component_ids=tuple(
                component_id
                for component_id in _PRODUCT_SEARCH_EXECUTION_COMPONENT_IDS
                if component_id in self._events
            ),
            component_input_counts=tuple(
                (component_id, self._events.get(component_id, (0, 0, 0))[0])
                for component_id in _PRODUCT_SEARCH_EXECUTION_COMPONENT_IDS
            ),
            component_output_counts=tuple(
                (component_id, self._events.get(component_id, (0, 0, 0))[1])
                for component_id in _PRODUCT_SEARCH_EXECUTION_COMPONENT_IDS
            ),
            component_evidence_counts=tuple(
                (component_id, self._events.get(component_id, (0, 0, 0))[2])
                for component_id in _PRODUCT_SEARCH_EXECUTION_COMPONENT_IDS
            ),
        )


@dataclass(frozen=True, slots=True)
class SearchAssetBinding:
    external_id: str
    video_id: str
    source_sha256: str
    byte_size: int
    duration_seconds: float

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.external_id, "external search asset ID"),
            (self.video_id, "search video ID"),
        ):
            if (
                type(value) is not str
                or not value
                or len(value) > 128
                or "\x00" in value
            ):
                raise ValueError(f"{field_name} is invalid")
        if type(self.source_sha256) is not str or _SHA256_RE.fullmatch(
            self.source_sha256
        ) is None:
            raise ValueError("search asset SHA-256 is invalid")
        if type(self.byte_size) is not int or self.byte_size <= 0:
            raise ValueError("search asset byte size is invalid")
        if (
            isinstance(self.duration_seconds, bool)
            or not isinstance(self.duration_seconds, (int, float))
            or not math.isfinite(float(self.duration_seconds))
            or float(self.duration_seconds) <= 0
        ):
            raise ValueError("search asset duration is invalid")
        object.__setattr__(self, "duration_seconds", float(self.duration_seconds))


@dataclass(frozen=True, slots=True)
class ProductSearchComponentIdentity:
    component_id: str
    identity: str

    def __post_init__(self) -> None:
        if (
            type(self.component_id) is not str
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", self.component_id)
            is None
            or type(self.identity) is not str
            or not self.identity
            or len(self.identity) > 2_048
        ):
            raise ValueError("product search component identity is invalid")


@dataclass(frozen=True, slots=True)
class ProductSearchIdentities:
    model: tuple[ProductSearchComponentIdentity, ...]
    index: tuple[ProductSearchComponentIdentity, ...]
    config: tuple[ProductSearchComponentIdentity, ...]

    def __post_init__(self) -> None:
        for field_name in ("model", "index", "config"):
            values = getattr(self, field_name)
            if (
                type(values) is not tuple
                or not values
                or any(not isinstance(item, ProductSearchComponentIdentity) for item in values)
                or len({item.component_id for item in values}) != len(values)
            ):
                raise ValueError(f"product search {field_name} identities are invalid")


class SearchDependencyError(RuntimeError):
    """A dependency failed during strict, non-best-effort search."""


def thumbnail_url_for_path(
    video_id: str,
    thumbnail_path: str | None,
    thumbnails_dir: Path | None,
) -> str | None:
    """Convert a validated internal thumbnail path to a storage-free public URL."""
    if thumbnail_path is None or thumbnails_dir is None:
        return None
    if type(thumbnail_path) is not str or not thumbnail_path or "\x00" in thumbnail_path:
        return None
    if "\\" in thumbnail_path:
        return None
    root = Path(thumbnails_dir).resolve(strict=False)
    video_root = (root / video_id).resolve(strict=False)
    raw_path = Path(thumbnail_path)
    candidates = (
        (raw_path,)
        if raw_path.is_absolute()
        else (raw_path, video_root / raw_path)
    )
    relative = None
    for candidate in candidates:
        try:
            relative = candidate.resolve(strict=False).relative_to(video_root)
        except (OSError, RuntimeError, ValueError):
            continue
        break
    if relative is None:
        return None
    parts = relative.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        return None
    encoded_path = "/".join(quote(part, safe="") for part in parts)
    return f"/api/thumbnails/{quote(video_id, safe='')}/{encoded_path}"


SEARCH_MODALITIES = {
    "all": {"speech", "ocr", "objects", "visual", "lighthouse"},
    "speech": {"speech"},
    "visual": {"visual", "lighthouse", "objects"},
    "ocr": {"ocr"},
}

_EVIDENCE_DETAIL_BOOLEAN_KEYS = (
    "matches_query",
    "shot_attempt",
    "ball_through_hoop",
    "shooter_outside_arc",
    "three_point_signal",
    "shooter_jersey_confirmed",
)
_EVIDENCE_DETAIL_TEXT_KEYS = (
    "event_type",
    "possible_shooter_jersey",
    "model_evidence",
)
_EVIDENCE_DETAIL_SCORE_KEYS = (
    "raw_release_score",
    "release_score",
    "outcome_score",
    "followup_score",
    "core_score",
    "plus_score",
    "reset_score",
    "miss_score",
    "transition_score",
    "contrast_score",
)


def _evidence_details(metadata: dict[str, object]) -> dict[str, object]:
    details: dict[str, object] = {}
    for key in _EVIDENCE_DETAIL_TEXT_KEYS:
        if key not in metadata:
            continue
        value = metadata[key]
        if value is None or isinstance(value, str):
            details[key] = value
    for key in _EVIDENCE_DETAIL_BOOLEAN_KEYS:
        if key not in metadata:
            continue
        value = metadata[key]
        if value is None or isinstance(value, bool):
            details[key] = value
    stage_order = metadata.get("stage_order")
    if isinstance(stage_order, (list, tuple)) and all(
        isinstance(stage, str) for stage in stage_order
    ):
        details["stage_order"] = list(stage_order)
    for key in _EVIDENCE_DETAIL_SCORE_KEYS:
        if key not in metadata:
            continue
        value = metadata[key]
        if value is None:
            details[key] = None
        elif isinstance(value, Real) and not isinstance(value, bool):
            score = float(value)
            if math.isfinite(score):
                details[key] = score
    return details


def _looks_like_named_entity(query: str) -> bool:
    tokens = re.findall(r"[^\W\d_]+", query, flags=re.UNICODE)
    return bool(tokens) and len(tokens) <= 3 and all(token[0].isupper() for token in tokens)


def _lexical_affinity(query: str, text: str) -> float:
    return lexical_match(query, text).score


def _hybridize(query: str, hit: EvidenceHit) -> EvidenceHit:
    match = lexical_match(query, hit.text)
    affinity = match.score
    if affinity:
        score = min(1.0, hit.score * 0.72 + affinity * 0.20 + 0.08)
    else:
        score = hit.score * 0.76
    return replace(
        hit,
        score=score,
        metadata={
            **hit.metadata,
            "lexical_affinity": affinity,
            "lexical_strategy": match.strategy,
            "matched_terms": list(match.matched_terms),
        },
    )


def _interval_gap(first: EvidenceHit, second: EvidenceHit) -> float:
    if first.start <= second.end and second.start <= first.end:
        return 0.0
    return min(abs(first.start - second.end), abs(second.start - first.end))


def corroborate_lighthouse_hits(
    hits: list[EvidenceHit],
    support: list[EvidenceHit],
    *,
    tolerance: float = 3.0,
) -> list[EvidenceHit]:
    if not hits:
        return []
    if not support:
        return []

    output: list[EvidenceHit] = []
    for hit in hits:
        related = [
            candidate
            for candidate in support
            if candidate.video_id == hit.video_id and _interval_gap(hit, candidate) <= tolerance
        ]
        if not related:
            continue
        strongest = max(related, key=lambda candidate: candidate.score)
        score = 0.58 * hit.score + 0.42 * strongest.score
        output.append(
            replace(
                hit,
                score=score,
                metadata={
                    **hit.metadata,
                    "source": "lighthouse-corroborated",
                    "corroborated": True,
                    "support_source": strongest.metadata.get("source", strongest.modality),
                },
            )
        )
    return output


def refine_speech_hit(query: str, hit: EvidenceHit, *, context: float = 1.2) -> EvidenceHit:
    if hit.modality != "speech":
        return hit
    raw_words = hit.metadata.get("words")
    if not isinstance(raw_words, list):
        return hit
    words: list[dict[str, object]] = []
    for raw_word in raw_words:
        if not isinstance(raw_word, dict):
            continue
        word = raw_word.get("word")
        raw_start = raw_word.get("start")
        raw_end = raw_word.get("end")
        if (
            not isinstance(word, str)
            or not word.strip()
            or isinstance(raw_start, bool)
            or isinstance(raw_end, bool)
        ):
            continue
        try:
            start = float(raw_start)
            end = float(raw_end)
        except (TypeError, ValueError):
            continue
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end <= start
        ):
            continue
        words.append({"word": word.strip(), "start": start, "end": end})
    if not words:
        return hit

    selected: list[dict[str, object]] = []
    for query_token in (token for token in text_tokens(query) if len(token) >= 3):
        candidate, score = max(
            (
                (word, lexical_match(query_token, str(word["word"])).score)
                for word in words
            ),
            key=lambda item: item[1],
        )
        if score >= 0.72:
            selected.append(candidate)
    if not selected:
        return hit

    start = max(hit.start, min(float(word["start"]) for word in selected) - context)
    end = min(hit.end, max(float(word["end"]) for word in selected) + context)
    if end <= start:
        return hit
    return replace(
        hit,
        start=start,
        end=end,
        metadata={
            **hit.metadata,
            "word_alignment": True,
            "temporal_refinement": True,
            "coarse_start": hit.start,
            "coarse_end": hit.end,
            "aligned_words": [str(word["word"]) for word in selected],
        },
    )


class SearchIndex(Protocol):
    def search(
        self,
        query: str,
        *,
        video_ids: list[str] | None = None,
        modalities: set[str] | None = None,
        limit: int = 50,
    ) -> list[EvidenceHit]: ...

    def search_generations(
        self,
        query: str,
        *,
        bindings: list[TextVectorSearchBinding] | tuple[TextVectorSearchBinding, ...],
        modalities: set[str] | None = None,
        limit: int = 50,
        exhaustive_validation: bool = False,
    ) -> list[TextVectorSearchHit]: ...


class MomentSearch(Protocol):
    def search(self, query: str, video_ids: list[str], *, limit: int = 30) -> list[EvidenceHit]: ...


class VisualSearch(Protocol):
    def search(
        self,
        query: str,
        *,
        video_ids: list[str] | None = None,
        limit: int = 50,
    ) -> list[EvidenceHit]: ...


class TemporalRefinement(Protocol):
    def refine(self, query: str, hits: list[EvidenceHit]) -> list[EvidenceHit]: ...


class CandidateReranker(Protocol):
    def rerank(self, query: str, candidates: list[object]) -> list[object]: ...


@dataclass(frozen=True, slots=True)
class EvidenceView:
    modality: str
    score: float
    text: str
    source: str
    confidence: float | None
    start: float
    end: float
    raw_score: float
    matched_terms: list[str]
    details: dict[str, object]


@dataclass(frozen=True, slots=True)
class SearchResultView:
    id: str
    video_id: str
    video_name: str
    start: float
    end: float
    score: float
    modalities: list[str]
    evidence: list[EvidenceView]
    thumbnail_url: str | None
    intent: str
    explanation: str
    refined: bool


class SearchService:
    def __init__(
        self,
        repository: Repository,
        vector_index: SearchIndex,
        moment_search: MomentSearch | None = None,
        visual_search: VisualSearch | None = None,
        query_router: QueryRouter | None = None,
        lexicon: SearchLexicon | None = None,
        temporal_refiner: TemporalRefinement | None = None,
        candidate_reranker: CandidateReranker | None = None,
        evaluation_rerankers: Mapping[str, CandidateReranker] | None = None,
        semantic_text_min_score: float = 0.42,
        visual_min_score: float = 0.18,
        specification_resolver: Callable[[], IndexingSpecifications] | None = None,
        segment_specifications: Iterable[StageSpecification] | None = None,
        thumbnails_dir: Path | None = None,
        media_root: Path | None = None,
        media_access_root: Path | None = None,
    ) -> None:
        if specification_resolver is not None and segment_specifications is not None:
            raise ValueError("use one indexing specification source")
        self.repository = repository
        self.vector_index = vector_index
        self.moment_search = moment_search
        self.visual_search = visual_search
        self.query_router = query_router or QueryRouter()
        self.lexicon = lexicon
        self.temporal_refiner = temporal_refiner
        self.candidate_reranker = candidate_reranker
        resolved_rerankers = dict(evaluation_rerankers or {})
        unknown_rerankers = set(resolved_rerankers) - {"qwen", "internvideo"}
        if unknown_rerankers:
            raise ValueError("unsupported evaluation reranker mapping")
        if candidate_reranker is not None:
            provider_id = getattr(candidate_reranker, "id", None)
            inferred = {
                "qwen": "qwen",
                "qwen-video": "qwen",
                "internvideo": "internvideo",
            }.get(provider_id)
            if inferred is not None:
                resolved_rerankers.setdefault(inferred, candidate_reranker)
        self.evaluation_rerankers: Mapping[str, CandidateReranker] = MappingProxyType(
            resolved_rerankers
        )
        self.semantic_text_min_score = semantic_text_min_score
        self.visual_min_score = visual_min_score
        self.specification_resolver = specification_resolver
        self._static_segment_specifications = (
            tuple(segment_specifications)
            if segment_specifications is not None
            else None
        )
        if self._static_segment_specifications is not None and any(
            not isinstance(item, StageSpecification)
            for item in self._static_segment_specifications
        ):
            raise ValueError("segment specifications must be validated")
        self.thumbnails_dir = Path(thumbnails_dir) if thumbnails_dir is not None else None
        self.media_root = Path(media_root) if media_root is not None else None
        if media_access_root is not None and media_root is None:
            raise ValueError("retained media access requires a canonical media root")
        self.media_access_root = (
            Path(media_access_root) if media_access_root is not None else None
        )

    def open_pinned_evaluation(
        self,
        configuration: EvaluationSearchConfiguration,
        assets: tuple[SearchAssetBinding, ...],
        *,
        execution_mode: Literal["cold", "warm"] = "warm",
        lifecycle_identity: str | None = None,
    ) -> PinnedProductSearchSession:
        """Pin all addressable product artifacts for one evaluation run."""
        return PinnedProductSearchSession(
            self,
            configuration,
            assets,
            execution_mode=execution_mode,
            lifecycle_identity=lifecycle_identity,
        )

    def _current_segment_specifications(self) -> tuple[StageSpecification, ...]:
        if self.specification_resolver is not None:
            resolved = self.specification_resolver()
            if not isinstance(resolved, IndexingSpecifications):
                raise ValueError("indexing specification resolver returned invalid data")
            return resolved.segment_specifications
        return self._static_segment_specifications or ()

    def _current_indexing_specifications(self) -> IndexingSpecifications | None:
        if self.specification_resolver is None:
            return None
        resolved = self.specification_resolver()
        if not isinstance(resolved, IndexingSpecifications):
            raise ValueError("indexing specification resolver returned invalid data")
        return resolved

    def _require_current_sqlite_artifacts(
        self,
        *,
        modalities: set[str],
        video_ids: list[str],
        specifications: tuple[StageSpecification, ...],
    ) -> None:
        required = {
            "speech": StageKind.SPEECH,
            "ocr": StageKind.OCR,
            "objects": StageKind.OBJECTS,
        }
        by_kind = {specification.kind: specification for specification in specifications}
        for modality, kind in required.items():
            if modality not in modalities:
                continue
            specification = by_kind.get(kind)
            if specification is None:
                raise SearchDependencyError(
                    f"Required {modality} evidence is unavailable"
                )
            try:
                missing = [
                    video_id
                    for video_id in video_ids
                    if not self.repository.is_active_segment_generation_current(
                        video_id,
                        specification,
                    )
                ]
            except Exception as error:
                raise SearchDependencyError(
                    f"Required {modality} evidence failed integrity validation"
                ) from error
            if missing:
                raise SearchDependencyError(
                    f"Required {modality} evidence is unavailable"
                )

    def search(
        self,
        query: str,
        *,
        video_ids: list[str] | None = None,
        limit: int = 20,
        use_lighthouse: bool = True,
        mode: str = "all",
        raise_on_provider_error: bool = False,
        _evaluation_configuration: EvaluationSearchConfiguration | None = None,
        _pinned_session: PinnedProductSearchSession | None = None,
        _execution_recorder: _ProductSearchExecutionRecorder | None = None,
    ) -> list[SearchResultView]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("search query is empty")
        if _evaluation_configuration is None:
            if _execution_recorder is not None:
                raise ValueError("execution recorder requires pinned evaluation")
            if mode not in SEARCH_MODALITIES:
                raise ValueError("unsupported search mode")
            plan = self.query_router.route(
                normalized_query,
                mode=mode,
                requested_lighthouse=use_lighthouse,
            )
        else:
            if (
                not isinstance(_evaluation_configuration, EvaluationSearchConfiguration)
                or _pinned_session is None
                or not raise_on_provider_error
                or _execution_recorder is None
            ):
                raise ValueError("pinned evaluation search contract is invalid")
            _pinned_session._validate_execution_recorder(
                _execution_recorder,
                configuration_identity=_evaluation_configuration.identity,
            )
            plan = QueryPlan(
                query=normalized_query,
                intent="evaluation",
                modalities=frozenset(_evaluation_configuration.modalities),
                modality_weights=dict(_evaluation_configuration.modality_weights),
                use_lighthouse=_evaluation_configuration.lighthouse,
                refine_temporally=_evaluation_configuration.temporal_refinement,
                explanation="frozen evaluation search plan",
            )
        ready_videos = {
            video.id: video
            for video in self.repository.list_videos()
            if video.status == "ready"
        }
        ready_video_ids = (
            list(ready_videos)
            if video_ids is None
            else [video_id for video_id in video_ids if video_id in ready_videos]
        )
        allowed_modalities = set(plan.modalities)
        named_entity_query = plan.intent == "entity"
        semantic_modalities = allowed_modalities & {"speech", "ocr", "objects"}
        external_release = ExternalIndexReleaseSnapshot(frozenset(), {}, {})
        if _pinned_session is None and ready_video_ids:
            try:
                external_release = self.repository.get_external_index_release_snapshot(
                    ready_video_ids
                )
            except Exception as error:
                raise SearchDependencyError(
                    "External index release snapshot is unavailable"
                ) from error
            if not isinstance(external_release, ExternalIndexReleaseSnapshot):
                raise SearchDependencyError(
                    "External index release snapshot is invalid"
                )
        durable_video_ids = [
            video_id
            for video_id in ready_video_ids
            if video_id in external_release.job_backed_video_ids
        ]
        legacy_video_ids = [
            video_id
            for video_id in ready_video_ids
            if video_id not in external_release.job_backed_video_ids
        ]
        if durable_video_ids:
            if "visual" in allowed_modalities and self.visual_search is None:
                raise SearchDependencyError("Visual release binding provider is unavailable")
            if (
                "lighthouse" in allowed_modalities
                and plan.use_lighthouse
                and self.moment_search is None
            ):
                raise SearchDependencyError(
                    "Lighthouse release binding provider is unavailable"
                )
        query_variants = (
            _pinned_session.expand_query(normalized_query)
            if _pinned_session is not None
            else self.lexicon.expand(normalized_query)
            if self.lexicon
            else [normalized_query]
        )

        current_specifications: tuple[StageSpecification, ...] = ()
        selected_specifications: tuple[StageSpecification, ...] = ()
        indexing_specifications: IndexingSpecifications | None = None
        try:
            indexing_specifications = (
                _pinned_session.indexing_specifications
                if _pinned_session is not None
                else self._current_indexing_specifications()
            )
            current_specifications = (
                indexing_specifications.segment_specifications
                if indexing_specifications is not None
                else self._static_segment_specifications or ()
            )
            if raise_on_provider_error and _pinned_session is None:
                self._require_current_sqlite_artifacts(
                    modalities=allowed_modalities,
                    video_ids=ready_video_ids,
                    specifications=current_specifications,
                )
            modality_by_kind = {
                StageKind.SPEECH: "speech",
                StageKind.OCR: "ocr",
                StageKind.OBJECTS: "objects",
            }
            selected_specifications = tuple(
                specification
                for specification in current_specifications
                if modality_by_kind.get(specification.kind) in semantic_modalities
            )
        except SearchDependencyError:
            raise
        except Exception as error:
            logger.exception("Current indexing specifications are unavailable")
            if raise_on_provider_error:
                raise SearchDependencyError(
                    "Current text evidence is unavailable"
                ) from error
        current_segments: list[SegmentRecord] = []
        if _pinned_session is not None:
            current_segments.extend(
                _pinned_session.segments_for(
                    ready_video_ids,
                    {specification.kind for specification in selected_specifications},
                )
            )
        else:
            for specification in selected_specifications:
                try:
                    current_segments.extend(
                        self.repository.list_current_active_segments(
                            [specification],
                            video_ids=ready_video_ids,
                        )
                    )
                except Exception as error:
                    logger.warning(
                        "Current %s segment generation is unavailable",
                        specification.kind.value,
                        exc_info=error,
                    )
                    if raise_on_provider_error:
                        raise SearchDependencyError(
                            f"Required {specification.kind.value} evidence is unavailable"
                        ) from error
        current_scenes: list[SegmentRecord] = []
        scene_specification = next(
            (
                specification
                for specification in current_specifications
                if specification.kind is StageKind.SCENES
            ),
            None,
        )
        if scene_specification is not None and _pinned_session is None:
            try:
                current_scenes = list(
                    self.repository.list_current_active_segments(
                        [scene_specification],
                        video_ids=ready_video_ids,
                    )
                )
            except Exception:
                logger.warning(
                    "Current scene thumbnails are unavailable",
                    exc_info=True,
                )
        current_by_id = {
            (segment.video_id, segment.id): segment for segment in current_segments
        }
        generation_aware = (
            (_pinned_session is not None or indexing_specifications is not None)
            and bool(
                getattr(
                    self.vector_index,
                    "supports_generation_provenance",
                    False,
                )
            )
            and callable(getattr(self.vector_index, "search_generations", None))
        )
        text_vector_bindings: tuple[TextVectorSearchBinding, ...] = ()
        if generation_aware and semantic_modalities and ready_video_ids:
            assert indexing_specifications is not None
            try:
                if _pinned_session is not None:
                    text_vector_bindings = _pinned_session.text_bindings_for(
                        ready_video_ids
                    )
                else:
                    text_vector_bindings = self.repository.list_current_text_vector_bindings(
                        video_ids=ready_video_ids,
                        text_specification=indexing_specifications.text_vectors,
                        semantic_specifications=(
                            indexing_specifications.semantic_segment_specifications
                        ),
                        required_modalities=(
                            semantic_modalities if raise_on_provider_error else None
                        ),
                    )
            except Exception as error:
                logger.warning(
                    "Current text vector generation is unavailable",
                    exc_info=error,
                )
                if raise_on_provider_error:
                    raise SearchDependencyError(
                        "Semantic index failed integrity validation"
                    ) from error
            if raise_on_provider_error and {
                binding.video_id for binding in text_vector_bindings
            } != set(ready_video_ids):
                raise SearchDependencyError(
                    "Semantic index is unavailable for required evidence"
                )

        hits: list[EvidenceHit] = []
        if semantic_modalities and ready_video_ids:
            try:
                if generation_aware:
                    allowed_generation_pairs = {
                        (binding.video_id, binding.generation_id)
                        for binding in text_vector_bindings
                    }
                    generation_hits = self.vector_index.search_generations(
                        normalized_query,
                        bindings=text_vector_bindings,
                        modalities=semantic_modalities,
                        limit=max(limit * 3, 30),
                        exhaustive_validation=(
                            raise_on_provider_error and _pinned_session is None
                        ),
                    )
                    assert indexing_specifications is not None
                    post_search_bindings = (
                        _pinned_session.text_bindings_for(ready_video_ids)
                        if _pinned_session is not None
                        else self.repository.list_current_text_vector_bindings(
                            video_ids=ready_video_ids,
                            text_specification=indexing_specifications.text_vectors,
                            semantic_specifications=(
                                indexing_specifications.semantic_segment_specifications
                            ),
                            required_modalities=(
                                semantic_modalities
                                if raise_on_provider_error
                                else None
                            ),
                        )
                    )
                    if post_search_bindings != text_vector_bindings:
                        raise ValueError(
                            "text vector generation changed during search"
                        )
                    semantic_hits = []
                    for hit in generation_hits:
                        if (
                            not isinstance(hit, TextVectorSearchHit)
                            or (hit.video_id, hit.generation_id)
                            not in allowed_generation_pairs
                        ):
                            continue
                        segment = current_by_id.get((hit.video_id, hit.segment_id))
                        if segment is None or segment.modality != hit.modality:
                            continue
                        semantic_hits.append(
                            EvidenceHit(
                                video_id=segment.video_id,
                                segment_id=segment.id,
                                start=segment.start,
                                end=segment.end,
                                modality=segment.modality,
                                score=hit.score,
                                text=segment.text,
                                metadata={
                                    "source": "semantic-generation",
                                    "text_vector_generation_id": hit.generation_id,
                                },
                            )
                        )
                else:
                    semantic_hits = self.vector_index.search(
                        normalized_query,
                        video_ids=ready_video_ids,
                        modalities=semantic_modalities,
                        limit=max(limit * 3, 30),
                    )
                authoritative_semantic_hits: list[EvidenceHit] = []
                for hit in semantic_hits:
                    segment = current_by_id.get((hit.video_id, hit.segment_id))
                    if (
                        segment is None
                        or segment.modality not in semantic_modalities
                        or hit.modality != segment.modality
                    ):
                        continue
                    authoritative_semantic_hits.append(
                        replace(
                            hit,
                            start=segment.start,
                            end=segment.end,
                            modality=segment.modality,
                            text=segment.text,
                            metadata={
                                **segment.metadata,
                                "thumbnail_path": segment.thumbnail_path,
                                "source": hit.metadata.get("source", "semantic"),
                                "segment_confidence": segment.confidence,
                            },
                        )
                    )
                semantic_hits = authoritative_semantic_hits
                semantic_threshold = max(self.semantic_text_min_score, 0.62) if named_entity_query else self.semantic_text_min_score
                hybrid_hits = [
                    refine_speech_hit(normalized_query, _hybridize(normalized_query, hit))
                    for hit in semantic_hits
                ]
                if named_entity_query:
                    hybrid_hits = [
                        hit
                        for hit in hybrid_hits
                        if hit.metadata.get("lexical_strategy") in _TRUSTED_ENTITY_MATCHES
                    ]
                accepted_semantic_hits = [
                    hit for hit in hybrid_hits if hit.score >= semantic_threshold
                ]
                hits.extend(accepted_semantic_hits)
                if _execution_recorder is not None:
                    if not generation_aware:
                        raise RuntimeError(
                            "pinned semantic execution is not generation-aware"
                        )
                    _execution_recorder.record(
                        "text_vectors",
                        input_count=len(text_vector_bindings),
                        output_count=len(generation_hits),
                        evidence_count=len(accepted_semantic_hits),
                        owner=_pinned_session,
                    )
            except Exception as error:
                logger.exception("Semantic text search failed")
                if raise_on_provider_error:
                    raise SearchDependencyError("Semantic text search failed") from error

        lexical_matches = sorted(
            (
                (
                    segment,
                    max(
                        lexical_match(variant, segment.text).score
                        for variant in query_variants
                    ),
                )
                for segment in current_segments
            ),
            key=lambda item: item[1],
            reverse=True,
        )[: max(limit * 2, 20)]
        lexical_evidence_count = 0
        for segment, score in lexical_matches:
            if score <= 0:
                continue
            if segment.modality not in allowed_modalities:
                continue
            matched_variant, match = max(
                (
                    (variant, lexical_match(variant, segment.text))
                    for variant in query_variants
                ),
                key=lambda value: value[1].score,
            )
            if named_entity_query and match.strategy not in _TRUSTED_ENTITY_MATCHES:
                continue
            lexical_hit = EvidenceHit(
                video_id=segment.video_id,
                segment_id=segment.id,
                start=segment.start,
                end=segment.end,
                modality=segment.modality,
                score=score,
                text=segment.text,
                metadata={
                    **segment.metadata,
                    "thumbnail_path": segment.thumbnail_path,
                    "source": f"lexical-{match.strategy}",
                    "segment_confidence": segment.confidence,
                    "matched_terms": list(match.matched_terms),
                },
            )
            hits.append(refine_speech_hit(matched_variant, lexical_hit))
            lexical_evidence_count += 1
        if (
            _execution_recorder is not None
            and _evaluation_configuration is not None
            and _evaluation_configuration.text_search != "disabled"
        ):
            _execution_recorder.record(
                "lexical_text",
                input_count=len(current_segments),
                output_count=lexical_evidence_count,
                evidence_count=lexical_evidence_count,
                owner=_pinned_session,
            )

        if (
            "visual" in allowed_modalities
            and self.visual_search is not None
            and ready_video_ids
        ):
            try:
                if _pinned_session is not None:
                    search_generations = getattr(
                        self.visual_search,
                        "search_generations",
                        None,
                    )
                    if not callable(search_generations):
                        raise RuntimeError("pinned visual search is unavailable")
                    visual_generation_bindings = (
                        _pinned_session.visual_bindings_for(ready_video_ids)
                    )
                    visual_hits = search_generations(
                        normalized_query,
                        generation_bindings=visual_generation_bindings,
                        limit=max(limit * 2, 20),
                    )
                    raw_visual_output_count = len(visual_hits)
                else:
                    visual_hits = []
                    if durable_video_ids:
                        generation_bindings = external_release.bindings_for(
                            StageKind.VISUAL_DENSE
                        )
                        selected_bindings = {
                            video_id: generation_bindings[video_id]
                            for video_id in durable_video_ids
                            if video_id in generation_bindings
                        }
                        if set(selected_bindings) != set(durable_video_ids):
                            raise SearchDependencyError(
                                "Visual release binding is missing or corrupt"
                            )
                        search_generations = getattr(
                            self.visual_search,
                            "search_generations",
                            None,
                        )
                        if not callable(search_generations):
                            raise SearchDependencyError(
                                "Visual release binding search is unavailable"
                            )
                        exact_hits = search_generations(
                            normalized_query,
                            generation_bindings=selected_bindings,
                            limit=max(limit * 2, 20),
                        )
                        if any(
                            hit.video_id not in selected_bindings
                            for hit in exact_hits
                        ):
                            raise ValueError(
                                "visual generation search returned an unbound video"
                            )
                        visual_hits.extend(exact_hits)
                    if legacy_video_ids:
                        visual_hits.extend(
                            self.visual_search.search(
                                normalized_query,
                                video_ids=legacy_video_ids,
                                limit=max(limit * 2, 20),
                            )
                        )
                visual_hits = [
                    hit
                    for hit in visual_hits
                    if hit.video_id in ready_videos
                    and hit.score >= self.visual_min_score
                ]
                if _execution_recorder is not None:
                    _execution_recorder.record(
                        "visual_dense",
                        input_count=len(visual_generation_bindings),
                        output_count=raw_visual_output_count,
                        evidence_count=len(visual_hits),
                        owner=_pinned_session,
                    )
                if plan.refine_temporally and self.temporal_refiner is not None:
                    if raise_on_provider_error:
                        refine = getattr(
                            self.temporal_refiner,
                            "refine_strict",
                            None,
                        )
                        if not callable(refine):
                            raise RuntimeError(
                                "strict temporal refinement is unavailable"
                            )
                    else:
                        refine = self.temporal_refiner.refine
                    temporal_inputs = visual_hits
                    temporal_input_count = len(temporal_inputs)
                    visual_hits = refine(normalized_query, temporal_inputs)
                    if _execution_recorder is not None:
                        _execution_recorder.record(
                            "temporal_refinement",
                            input_count=temporal_input_count,
                            output_count=len(visual_hits),
                            evidence_count=sum(
                                1
                                for hit in visual_hits
                                if hit.metadata.get("temporal_refinement") is True
                            ),
                            owner=_pinned_session,
                        )
                hits.extend(visual_hits)
            except SearchDependencyError:
                raise
            except Exception as error:
                logger.exception("SigLIP visual search failed")
                if raise_on_provider_error or durable_video_ids:
                    raise SearchDependencyError("Visual search failed") from error

        if (
            "lighthouse" in allowed_modalities
            and plan.use_lighthouse
            and self.moment_search is not None
            and ready_video_ids
        ):
            try:
                if _pinned_session is not None:
                    search_generations = getattr(
                        self.moment_search,
                        "search_generations",
                        None,
                    )
                    if not callable(search_generations):
                        raise RuntimeError("pinned Lighthouse search is unavailable")
                    lighthouse_generation_bindings = (
                        _pinned_session.lighthouse_bindings_for(ready_video_ids)
                    )
                    lighthouse_hits = search_generations(
                        normalized_query,
                        lighthouse_generation_bindings,
                        limit=max(limit * 2, 20),
                    )
                    raw_lighthouse_output_count = len(lighthouse_hits)
                else:
                    lighthouse_hits = []
                    if durable_video_ids:
                        generation_bindings = external_release.bindings_for(
                            StageKind.LIGHTHOUSE
                        )
                        selected_bindings = {
                            video_id: generation_bindings[video_id]
                            for video_id in durable_video_ids
                            if video_id in generation_bindings
                        }
                        if set(selected_bindings) != set(durable_video_ids):
                            raise SearchDependencyError(
                                "Lighthouse release binding is missing or corrupt"
                            )
                        search_generations = getattr(
                            self.moment_search,
                            "search_generations",
                            None,
                        )
                        if not callable(search_generations):
                            raise SearchDependencyError(
                                "Lighthouse release binding search is unavailable"
                            )
                        exact_hits = search_generations(
                            normalized_query,
                            selected_bindings,
                            limit=max(limit * 2, 20),
                        )
                        if any(
                            hit.video_id not in selected_bindings
                            for hit in exact_hits
                        ):
                            raise ValueError(
                                "Lighthouse generation search returned an unbound video"
                            )
                        lighthouse_hits.extend(exact_hits)
                    if legacy_video_ids:
                        lighthouse_hits.extend(
                            self.moment_search.search(
                                normalized_query,
                                legacy_video_ids,
                                limit=max(limit * 2, 20),
                            )
                        )
                viable = [
                    hit
                    for hit in lighthouse_hits
                    if hit.video_id in ready_videos
                    and hit.score >= self.visual_min_score
                ]
                support = [hit for hit in hits if hit.modality in {"visual", "objects"}]
                corroborated_lighthouse_hits = corroborate_lighthouse_hits(
                    viable,
                    support,
                )
                hits.extend(corroborated_lighthouse_hits)
                if _execution_recorder is not None:
                    _execution_recorder.record(
                        "lighthouse",
                        input_count=len(lighthouse_generation_bindings),
                        output_count=raw_lighthouse_output_count,
                        evidence_count=len(corroborated_lighthouse_hits),
                        owner=_pinned_session,
                    )
            except SearchDependencyError:
                raise
            except Exception as error:
                logger.exception("Lighthouse search failed")
                if raise_on_provider_error or durable_video_ids:
                    raise SearchDependencyError("Temporal search failed") from error

        deduplicated: dict[tuple[str, str], EvidenceHit] = {}
        for hit in hits:
            key = hit.video_id, hit.segment_id
            if key not in deduplicated or hit.score > deduplicated[key].score:
                deduplicated[key] = hit

        videos = ready_videos
        scene_segments: dict[str, list[SegmentRecord]] = {}
        for segment in current_scenes:
            if segment.modality == "scene" and segment.thumbnail_path:
                scene_segments.setdefault(segment.video_id, []).append(segment)
        calibrated = calibrate_hits(list(deduplicated.values()))
        fused = fuse_hits(calibrated, limit=limit, modality_weights=plan.modality_weights)
        selected_reranker = (
            _pinned_session.reranker
            if _pinned_session is not None
            else self.candidate_reranker
        )
        if selected_reranker is not None and (
            _pinned_session is not None or plan.intent in {"action", "mixed"}
        ):
            try:
                rerank = getattr(selected_reranker, "rerank", None)
                if raise_on_provider_error:
                    strict_rerank = getattr(
                        selected_reranker,
                        "rerank_strict",
                        None,
                    )
                    if callable(strict_rerank):
                        rerank = strict_rerank
                    elif _pinned_session is not None:
                        rerank = None
                if not callable(rerank):
                    raise RuntimeError("strict candidate reranker is unavailable")
                candidate_limit = (
                    _evaluation_configuration.reranker_candidate_limit
                    if _evaluation_configuration is not None
                    else len(fused)
                )
                selected_candidates = fused[:candidate_limit]
                untouched_tail = fused[candidate_limit:]
                reranked = rerank(  # type: ignore[arg-type]
                    normalized_query,
                    selected_candidates,
                )
                if not isinstance(reranked, list):
                    raise ValueError("candidate reranker returned an invalid collection")
                expected_keys = {
                    (item.video_id, item.start, item.end)
                    for item in selected_candidates
                }
                actual_keys = {
                    (item.video_id, item.start, item.end)
                    for item in reranked
                }
                if (
                    len(reranked) != len(selected_candidates)
                    or len(actual_keys) != len(reranked)
                    or actual_keys != expected_keys
                ):
                    raise ValueError("candidate reranker changed the pinned candidate set")
                if (
                    _execution_recorder is not None
                    and _evaluation_configuration is not None
                    and _evaluation_configuration.reranker == "qwen"
                ):
                    _execution_recorder.record(
                        "qwen_verification",
                        input_count=len(selected_candidates),
                        output_count=len(reranked),
                        evidence_count=sum(
                            1
                            for candidate in reranked
                            for evidence in candidate.evidence
                            if isinstance(evidence, EvidenceHit)
                            and evidence.modality == "qwen_video"
                            and evidence.metadata.get("source")
                            == "qwen-video-verifier"
                        ),
                        owner=_pinned_session,
                    )
                fused = [*reranked, *untouched_tail]
            except Exception as error:
                logger.exception("Candidate video reranking failed")
                if raise_on_provider_error:
                    raise SearchDependencyError("Candidate reranking failed") from error
        output: list[SearchResultView] = []
        for index, result in enumerate(fused):
            video = videos.get(result.video_id)
            if video is None:
                continue
            thumbnail_path = next(
                (
                    str(hit.metadata["thumbnail_path"])
                    for hit in result.evidence
                    if hit.metadata.get("thumbnail_path")
                ),
                None,
            )
            if thumbnail_path is None:
                scenes = scene_segments.get(result.video_id, [])
                nearest = min(
                    scenes,
                    key=lambda scene: abs((scene.start + scene.end) / 2 - (result.start + result.end) / 2),
                    default=None,
                )
                if nearest is not None:
                    thumbnail_path = nearest.thumbnail_path
            thumbnail_url = thumbnail_url_for_path(
                result.video_id,
                thumbnail_path,
                self.thumbnails_dir,
            )
            output.append(
                SearchResultView(
                    id=f"{result.video_id}:{index}:{result.start:.3f}",
                    video_id=result.video_id,
                    video_name=video.name,
                    start=result.start,
                    end=result.end,
                    score=result.score,
                    modalities=result.modalities,
                    evidence=[
                        EvidenceView(
                            modality=hit.modality,
                            score=hit.score,
                            text=hit.text,
                            source=str(hit.metadata.get("source") or "unknown"),
                            confidence=(
                                float(hit.metadata["segment_confidence"])
                                if hit.metadata.get("segment_confidence") is not None
                                else None
                            ),
                            start=hit.start,
                            end=hit.end,
                            raw_score=float(hit.metadata.get("raw_score", hit.score)),
                            matched_terms=[str(value) for value in hit.metadata.get("matched_terms", [])],
                            details=_evidence_details(hit.metadata),
                        )
                        for hit in result.evidence[:5]
                    ],
                    thumbnail_url=thumbnail_url,
                    intent=plan.intent,
                    explanation=plan.explanation,
                    refined=any(hit.metadata.get("temporal_refinement") for hit in result.evidence),
                )
            )
        return output

    def search_for_evaluation(
        self,
        query: str,
        *,
        video_ids: list[str] | None = None,
        limit: int = 20,
        use_lighthouse: bool = True,
        mode: str = "all",
    ) -> list[SearchResultView]:
        """Run search fail-closed so infrastructure outages cannot look like misses."""
        plan = self.query_router.route(
            query.strip(),
            mode=mode,
            requested_lighthouse=use_lighthouse,
        )
        semantic_modalities = set(plan.modalities) & {"speech", "ocr", "objects"}
        if semantic_modalities and not getattr(self.vector_index, "available", True):
            raise SearchDependencyError("Semantic search is unavailable")
        if "visual" in plan.modalities and self.visual_search is None:
            raise SearchDependencyError("Visual search is unavailable")
        if plan.use_lighthouse and "lighthouse" in plan.modalities and self.moment_search is None:
            raise SearchDependencyError("Temporal search is unavailable")

        ready_video_ids = [
            video.id
            for video in self.repository.list_videos()
            if video.status == "ready"
        ]
        ready_video_id_set = set(ready_video_ids)
        selected_ready_ids = (
            ready_video_ids
            if video_ids is None
            else [
                video_id
                for video_id in dict.fromkeys(video_ids)
                if video_id in ready_video_id_set
            ]
        )
        if "visual" in plan.modalities and self.visual_search is not None:
            self._require_current_artifacts(
                self.visual_search,
                "index_is_current",
                selected_ready_ids,
                "Visual index",
            )
        if (
            plan.use_lighthouse
            and "lighthouse" in plan.modalities
            and self.moment_search is not None
        ):
            self._require_current_artifacts(
                self.moment_search,
                "cache_is_current",
                selected_ready_ids,
                "Temporal cache",
            )
        return self.search(
            query,
            video_ids=video_ids,
            limit=limit,
            use_lighthouse=use_lighthouse,
            mode=mode,
            raise_on_provider_error=True,
        )

    @staticmethod
    def _require_current_artifacts(
        provider: object,
        method_name: str,
        video_ids: list[str],
        dependency_name: str,
    ) -> None:
        readiness_check = getattr(provider, method_name, None)
        if not callable(readiness_check) or not video_ids:
            return
        try:
            missing = [
                video_id
                for video_id in video_ids
                if not readiness_check(video_id)
            ]
        except Exception as error:
            raise SearchDependencyError(
                f"{dependency_name} readiness check failed"
            ) from error
        if missing:
            preview = ", ".join(missing[:3])
            suffix = "" if len(missing) <= 3 else f" (+{len(missing) - 3} more)"
            raise SearchDependencyError(
                f"{dependency_name} is missing or stale for selected ready videos: "
                f"{preview}{suffix}"
            )


@dataclass(frozen=True, slots=True)
class _PinnedAssetState:
    binding: SearchAssetBinding
    video_fingerprint: tuple[object, ...]
    external_release_bound: bool | None
    text_state: str
    active_text_generation_id: str | None
    text_binding: TextVectorSearchBinding | None
    segment_snapshots: tuple[
        tuple[StageKind, SegmentGeneration, tuple[SegmentRecord, ...]], ...
    ]
    visual_state: str
    active_visual_generation_id: str | None
    visual_descriptor_json: str | None
    lighthouse_state: str
    active_lighthouse_generation_id: str | None
    lighthouse_descriptor_json: str | None


@dataclass(slots=True)
class _BenchmarkSessionBudget:
    points: int = 0
    segments: int = 0
    text_bytes: int = 0
    metadata_bytes: int = 0

    def consume(
        self,
        *,
        points: int,
        segments: int,
        text_bytes: int,
        metadata_bytes: int,
    ) -> None:
        values = {
            "point": (self.points + points, _BENCHMARK_MAX_TEXT_POINTS),
            "segment": (self.segments + segments, _BENCHMARK_MAX_SEGMENTS),
            "text byte": (self.text_bytes + text_bytes, _BENCHMARK_MAX_TEXT_BYTES),
            "metadata byte": (
                self.metadata_bytes + metadata_bytes,
                _BENCHMARK_MAX_METADATA_BYTES,
            ),
        }
        for label, (total, maximum) in values.items():
            if total > maximum:
                raise SearchDependencyError(
                    f"pinned evaluation exceeds the aggregate {label} limit"
                )
        self.points += points
        self.segments += segments
        self.text_bytes += text_bytes
        self.metadata_bytes += metadata_bytes


class PinnedProductSearchSession:
    """A fail-closed, generation-addressed product search snapshot."""

    _CAPABILITIES = frozenset(
        {
            "text_vectors",
            "visual_dense",
            "temporal_refinement",
            "lighthouse",
            "qwen_verification",
            "internvideo",
        }
    )

    def __init__(
        self,
        service: SearchService,
        configuration: EvaluationSearchConfiguration,
        assets: tuple[SearchAssetBinding, ...],
        *,
        execution_mode: Literal["cold", "warm"],
        lifecycle_identity: str | None,
    ) -> None:
        if not isinstance(configuration, EvaluationSearchConfiguration):
            raise ValueError("evaluation search configuration must be validated")
        if (
            type(assets) is not tuple
            or not assets
            or any(not isinstance(asset, SearchAssetBinding) for asset in assets)
            or len({asset.external_id for asset in assets}) != len(assets)
            or len({asset.video_id for asset in assets}) != len(assets)
        ):
            raise ValueError("pinned search assets must be unique validated bindings")
        if execution_mode not in {"cold", "warm"}:
            raise ValueError("unsupported evaluation execution mode")
        if lifecycle_identity is None:
            if execution_mode == "cold":
                raise SearchDependencyError(
                    "cold evaluation requires an attested cache lifecycle"
                )
            lifecycle_identity = "warm:process-cache-preserved@1"
        if (
            type(lifecycle_identity) is not str
            or not lifecycle_identity.startswith(f"{execution_mode}:")
            or len(lifecycle_identity) > 1_024
            or "\x00" in lifecycle_identity
        ):
            raise ValueError("evaluation lifecycle identity does not match its mode")
        self._validate_asset_scope(assets)

        self._service = service
        self.configuration = configuration
        self.execution_mode = execution_mode
        self._lifecycle_identity = lifecycle_identity
        self._closed = False
        self._active_execution_recorder: _ProductSearchExecutionRecorder | None = None
        self._last_search_execution_trace: ProductSearchExecutionTrace | None = None
        self._assets = tuple(assets)
        self._assets_by_video = {asset.video_id: asset for asset in assets}
        self._lexicon_snapshot = service.lexicon.read() if service.lexicon else {}
        try:
            self.indexing_specifications = service._current_indexing_specifications()
        except Exception:
            self.indexing_specifications = None
        self.reranker = (
            service.evaluation_rerankers.get(configuration.reranker)
            if configuration.reranker != "none"
            else None
        )
        self._provider_identities = self._snapshot_provider_identities()
        self._vector_index_state, self._vector_index_attestation = (
            self._pin_vector_index()
        )
        self._temporal_state, self._temporal_attestation = (
            self._pin_temporal_refiner()
        )
        self._reranker_state, self._reranker_attestation = self._pin_reranker()
        self._external_release = (
            self._read_external_release_snapshot(
                tuple(asset.video_id for asset in self._assets)
            )
            if (
                self.configuration.visual_search != "disabled"
                or self.configuration.lighthouse
            )
            else None
        )
        budget = _BenchmarkSessionBudget()
        pinned_states: list[_PinnedAssetState] = []
        for asset in self._assets:
            state = self._pin_asset(asset)
            self._consume_text_budget(state, budget)
            pinned_states.append(state)
        self._states = self._attest_text_states(tuple(pinned_states))
        try:
            self._state_by_video = {
                state.binding.video_id: state for state in self._states
            }
            self._identities = self._build_identities()
        except BaseException as error:
            try:
                self._release_text_attestations(self._states)
            except Exception as cleanup_error:
                error.add_note(
                    "benchmark text attestation cleanup failed: "
                    f"{cleanup_error!r}"
                )
            raise

    @staticmethod
    def _validate_asset_scope(assets: tuple[SearchAssetBinding, ...]) -> None:
        if len(assets) > _BENCHMARK_MAX_ASSETS:
            raise SearchDependencyError("pinned evaluation exceeds the asset limit")
        aggregate_bytes = 0
        for asset in assets:
            if asset.byte_size > _BENCHMARK_MAX_ASSET_BYTES:
                raise SearchDependencyError(
                    "pinned evaluation exceeds the per-asset media byte limit"
                )
            aggregate_bytes += asset.byte_size
            if aggregate_bytes > _BENCHMARK_MAX_MEDIA_BYTES:
                raise SearchDependencyError(
                    "pinned evaluation exceeds the aggregate media byte limit"
                )

    def _consume_text_budget(
        self,
        state: _PinnedAssetState,
        budget: _BenchmarkSessionBudget,
    ) -> None:
        vector_binding = state.text_binding
        if vector_binding is None:
            return
        specifications = self.indexing_specifications
        if specifications is None:
            raise SearchDependencyError(
                "pinned text artifacts lack indexing specifications"
            )
        metadata_bytes = len(specifications.text_vectors.canonical_json.encode("utf-8"))
        metadata_bytes += len(
            vector_binding.index_specification.canonical_json.encode("utf-8")
        )
        expected_by_kind = {
            item.kind: item for item in specifications.semantic_segment_specifications
        }
        for item in vector_binding.inputs:
            specification = expected_by_kind.get(item.stage_kind)
            if specification is None:
                raise SearchDependencyError(
                    "pinned text input specification is unavailable"
                )
            metadata_bytes += len(specification.canonical_json.encode("utf-8"))
        text_bytes = sum(len(point.text.encode("utf-8")) for point in vector_binding.points)
        segment_count = 0
        for kind, _generation, segments in state.segment_snapshots:
            specification = expected_by_kind.get(kind)
            if specification is None:
                raise SearchDependencyError(
                    "pinned segment specification is unavailable"
                )
            metadata_bytes += len(specification.canonical_json.encode("utf-8"))
            segment_count += len(segments)
            for segment in segments:
                text_bytes += len(segment.text.encode("utf-8"))
                try:
                    metadata_json = json.dumps(
                        segment.metadata,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                except (TypeError, ValueError) as error:
                    raise SearchDependencyError(
                        "pinned segment metadata is not canonical JSON"
                    ) from error
                metadata_bytes += sum(
                    len(value.encode("utf-8"))
                    for value in (
                        segment.id,
                        segment.video_id,
                        segment.modality,
                        segment.thumbnail_path or "",
                        metadata_json,
                    )
                )
        budget.consume(
            points=len(vector_binding.points),
            segments=segment_count,
            text_bytes=text_bytes,
            metadata_bytes=metadata_bytes,
        )

    def _attest_text_states(
        self,
        states: tuple[_PinnedAssetState, ...],
    ) -> tuple[_PinnedAssetState, ...]:
        if self.configuration.text_search == "disabled":
            return states
        validate_generation = getattr(
            self._service.vector_index,
            "validate_generation_for_benchmark_snapshot",
            None,
        )
        release_bindings = getattr(
            self._service.vector_index,
            "release_benchmark_snapshot_bindings",
            None,
        )
        if not callable(validate_generation) or not callable(release_bindings):
            return tuple(
                replace(state, text_state="not_configured")
                if state.text_state == "complete"
                else state
                for state in states
            )
        attested: list[_PinnedAssetState] = []
        for state in states:
            if state.text_state != "complete" or state.text_binding is None:
                attested.append(state)
                continue
            try:
                valid = bool(validate_generation(state.text_binding))
            except Exception:
                valid = False
            attested.append(state if valid else replace(state, text_state="failed"))
        return tuple(attested)

    def _release_text_attestations(
        self,
        states: tuple[_PinnedAssetState, ...],
    ) -> None:
        bindings = tuple(
            state.text_binding
            for state in states
            if state.text_state == "complete" and state.text_binding is not None
        )
        if not bindings:
            return
        release_bindings = getattr(
            self._service.vector_index,
            "release_benchmark_snapshot_bindings",
            None,
        )
        if not callable(release_bindings):
            raise SearchDependencyError(
                "pinned text vector attestation release is unavailable"
            )
        release_bindings(bindings)

    def lifecycle_identity(self) -> str:
        return self._lifecycle_identity

    def identities(self) -> ProductSearchIdentities:
        self._ensure_open()
        return self._identities

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("pinned product search session is closed")

    def _validate_execution_recorder(
        self,
        recorder: _ProductSearchExecutionRecorder,
        *,
        configuration_identity: str,
    ) -> None:
        self._ensure_open()
        if recorder is not self._active_execution_recorder:
            raise ValueError("pinned search execution recorder is not active")
        recorder.validate_owner(
            owner=self,
            configuration_identity=configuration_identity,
        )

    def last_search_execution_trace(self) -> ProductSearchExecutionTrace:
        self._ensure_open()
        if self._last_search_execution_trace is None:
            raise RuntimeError("pinned search execution trace is unavailable")
        return self._last_search_execution_trace

    @staticmethod
    def _stable_provider_identity(provider: object | None, *, visual: bool = False) -> str:
        if provider is None:
            return "not-configured"
        if visual:
            payload: object = {
                "model_identity": getattr(provider, "model_identity", None),
                "specification_identity": getattr(
                    provider,
                    "specification_identity",
                    None,
                ),
                "type": f"{type(provider).__module__}.{type(provider).__qualname__}",
            }
        else:
            payload = getattr(provider, "identity", None)
            if callable(payload):
                payload = payload()
            if isinstance(payload, dict):
                payload = {
                    key: value
                    for key, value in payload.items()
                    if key != "active_generations"
                }
            payload = {
                "identity": payload,
                "type": f"{type(provider).__module__}.{type(provider).__qualname__}",
            }
        return f"sha256:{_canonical_digest(payload)}"

    def _snapshot_provider_identities(self) -> Mapping[str, str]:
        selected: dict[str, str] = {}
        if self.configuration.text_search != "disabled":
            vector_payload = {
                "index_specification": getattr(
                    getattr(
                        self._service.vector_index,
                        "index_specification",
                        None,
                    ),
                    "canonical_json",
                    None,
                ),
                "type": (
                    f"{type(self._service.vector_index).__module__}."
                    f"{type(self._service.vector_index).__qualname__}"
                ),
            }
            selected["vector_index"] = f"sha256:{_canonical_digest(vector_payload)}"
        if self.configuration.visual_search != "disabled":
            selected["visual"] = self._stable_provider_identity(
                self._service.visual_search,
                visual=True,
            )
        if self.configuration.temporal_refinement:
            selected["temporal_refiner"] = self._stable_provider_identity(
                self._service.temporal_refiner
            )
        if self.configuration.lighthouse:
            selected["lighthouse"] = self._stable_provider_identity(
                self._service.moment_search
            )
        return MappingProxyType(selected)

    def _provider_identities_are_current(self) -> bool:
        return dict(self._provider_identities) == dict(
            self._snapshot_provider_identities()
        )

    @staticmethod
    def _provider_is_ready(provider: object | None, *, check_index: bool) -> bool:
        status_method = getattr(provider, "status", None)
        if not callable(status_method):
            return False
        try:
            if check_index:
                status_value = status_method()
            else:
                try:
                    status_value = status_method(check_index=False)
                except TypeError:
                    status_value = status_method()
        except Exception:
            return False
        state = getattr(status_value, "state", None)
        return getattr(state, "value", state) == "ready"

    def _asset_fingerprint(
        self,
        binding: SearchAssetBinding,
        *,
        verify_content: bool,
    ) -> tuple[object, ...]:
        video = self._service.repository.get_video(binding.video_id)
        asset = self._service.repository.get_video_asset(binding.video_id)
        if video is None or asset is None:
            raise SearchDependencyError("pinned asset binding is unavailable")
        duration = video.duration
        if (
            video.status != "ready"
            or video.size_bytes != binding.byte_size
            or asset.sha256 != binding.source_sha256
            or asset.size_bytes != binding.byte_size
            or isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isclose(
                float(duration),
                binding.duration_seconds,
                rel_tol=1e-6,
                abs_tol=0.05,
            )
        ):
            raise SearchDependencyError("pinned asset binding does not match repository")
        if self._service.media_root is None:
            raise SearchDependencyError(
                "pinned evaluation requires a canonical media root"
            )
        source_before = self._managed_source_stat_fingerprint(video.stored_name)
        if verify_content:
            verify_existing = getattr(
                self._service.repository,
                "verify_existing_asset_identity",
                None,
            )
            if not callable(verify_existing):
                raise SearchDependencyError(
                    "read-only media identity verification is unavailable"
                )
            try:
                verification_kwargs: dict[str, Path] = {
                    "media_root": self._service.media_root,
                }
                if self._service.media_access_root is not None:
                    verification_kwargs["media_access_root"] = (
                        self._service.media_access_root
                    )
                verified = verify_existing(binding.video_id, **verification_kwargs)
            except Exception as error:
                raise SearchDependencyError(
                    "pinned media source identity could not be verified"
                ) from error
            if (
                verified.sha256 != binding.source_sha256
                or verified.size_bytes != binding.byte_size
            ):
                raise SearchDependencyError("pinned media source identity changed")
            source_after = self._managed_source_stat_fingerprint(video.stored_name)
            if source_after != source_before:
                raise SearchDependencyError(
                    "pinned media source changed during identity verification"
                )
        else:
            source_after = source_before
        return (
            video.id,
            video.status,
            video.stored_name,
            video.media_path,
            video.size_bytes,
            float(duration),
            asset.asset_id,
            asset.sha256,
            asset.size_bytes,
            source_after,
        )

    def _managed_source_stat_fingerprint(
        self,
        stored_name: str,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        media_root = self._service.media_access_root or self._service.media_root
        if (
            media_root is None
            or type(stored_name) is not str
            or Path(stored_name).name != stored_name
        ):
            raise SearchDependencyError("pinned media path is invalid")
        root_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        file_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            root_descriptor = os.open(media_root, root_flags)
        except OSError as error:
            raise SearchDependencyError("canonical media root is unavailable") from error
        try:
            root_before = os.fstat(root_descriptor)
            file_descriptor = os.open(
                stored_name,
                file_flags,
                dir_fd=root_descriptor,
            )
            try:
                source = os.fstat(file_descriptor)
                path_source = os.stat(
                    stored_name,
                    dir_fd=root_descriptor,
                    follow_symlinks=False,
                )
            finally:
                os.close(file_descriptor)
            root_after = os.fstat(root_descriptor)
        except OSError as error:
            raise SearchDependencyError("pinned media source is unavailable") from error
        finally:
            os.close(root_descriptor)

        def identity(value: os.stat_result) -> tuple[int, ...]:
            return (
                value.st_dev,
                value.st_ino,
                value.st_mode,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
            )

        def directory_identity(value: os.stat_result) -> tuple[int, ...]:
            return value.st_dev, value.st_ino, value.st_mode

        if (
            not stat.S_ISDIR(root_before.st_mode)
            or directory_identity(root_before) != directory_identity(root_after)
            or not stat.S_ISREG(source.st_mode)
            or identity(source) != identity(path_source)
        ):
            raise SearchDependencyError("pinned media source identity changed")
        return directory_identity(root_after), identity(source)

    def _pin_text(
        self,
        binding: SearchAssetBinding,
    ) -> tuple[
        str,
        str | None,
        TextVectorSearchBinding | None,
        tuple[tuple[StageKind, SegmentGeneration, tuple[SegmentRecord, ...]], ...],
    ]:
        if self.configuration.text_search == "disabled":
            return "not_configured", None, None, ()
        if self._vector_index_state != "complete":
            return self._vector_index_state, None, None, ()
        if (
            self.indexing_specifications is None
            or not getattr(
                self._service.vector_index,
                "supports_generation_provenance",
                False,
            )
            or not callable(getattr(self._service.vector_index, "search_generations", None))
            or not callable(
                getattr(
                    self._service.vector_index,
                    "validate_generation_for_benchmark_snapshot",
                    None,
                )
            )
            or not callable(
                getattr(
                    self._service.vector_index,
                    "release_benchmark_snapshot_bindings",
                    None,
                )
            )
            or not callable(
                getattr(
                    self._service.repository,
                    "get_text_vector_search_binding_bounded",
                    None,
                )
            )
            or not callable(
                getattr(
                    self._service.repository,
                    "get_segment_generation_snapshot_bounded",
                    None,
                )
            )
        ):
            return "not_configured", None, None, ()
        try:
            active_id = self._service.repository.get_active_text_vector_generation_id(
                binding.video_id
            )
        except Exception:
            return "failed", None, None, ()
        if active_id is None:
            return "missing", None, None, ()
        try:
            vector_binding = (
                self._service.repository.get_text_vector_search_binding_bounded(
                    active_id,
                    max_points=_BENCHMARK_MAX_TEXT_POINTS,
                    max_segments=_BENCHMARK_MAX_SEGMENTS,
                    max_text_bytes=_BENCHMARK_MAX_TEXT_BYTES,
                    max_metadata_bytes=_BENCHMARK_MAX_METADATA_BYTES,
                )
            )
            if vector_binding is None:
                return "stale", active_id, None, ()
            specifications = self.indexing_specifications
            assert specifications is not None
            if (
                vector_binding.generation.specification_hash
                != specifications.text_vectors.specification_hash
                or vector_binding.generation.source_sha256 != binding.source_sha256
                or vector_binding.index_specification
                != getattr(self._service.vector_index, "index_specification", None)
            ):
                return "stale", active_id, vector_binding, ()
            required_modalities = set(self.configuration.modalities) & set(
                _EVALUATION_TEXT_MODALITIES
            )
            if any(
                not vector_binding.supports_modality(modality)
                for modality in required_modalities
            ):
                return "missing", active_id, vector_binding, ()
            expected_by_kind = {
                item.kind: item
                for item in specifications.semantic_segment_specifications
            }
            snapshots: list[
                tuple[StageKind, SegmentGeneration, tuple[SegmentRecord, ...]]
            ] = []
            for input_binding in vector_binding.inputs:
                if input_binding.modality not in required_modalities:
                    continue
                generation_id = input_binding.segment_generation_id
                if generation_id is None:
                    return "missing", active_id, vector_binding, ()
                snapshot = (
                    self._service.repository.get_segment_generation_snapshot_bounded(
                        generation_id,
                        max_segments=_BENCHMARK_MAX_SEGMENTS,
                        max_text_bytes=_BENCHMARK_MAX_TEXT_BYTES,
                        max_metadata_bytes=_BENCHMARK_MAX_METADATA_BYTES,
                    )
                )
                if snapshot is None:
                    return "stale", active_id, vector_binding, ()
                generation, segments = snapshot
                expected = expected_by_kind[input_binding.stage_kind]
                if (
                    generation.video_id != binding.video_id
                    or generation.source_sha256 != binding.source_sha256
                    or generation.specification_hash != expected.specification_hash
                    or generation.segment_count != len(segments)
                ):
                    return "stale", active_id, vector_binding, ()
                snapshots.append((input_binding.stage_kind, generation, segments))
            return "complete", active_id, vector_binding, tuple(snapshots)
        except Exception:
            return "failed", active_id, None, ()

    @staticmethod
    def _validated_descriptor(
        raw: object,
        binding: SearchAssetBinding,
        *,
        generation_id: str,
        content_field: str,
    ) -> str | None:
        if not isinstance(raw, dict):
            return None
        required = {
            "duration_seconds",
            "generation_id",
            "source_sha256",
            "source_size_bytes",
            "specification_hash",
            "video_id",
            content_field,
        }
        if not required <= set(raw):
            return None
        duration = raw.get("duration_seconds")
        if (
            raw.get("video_id") != binding.video_id
            or raw.get("generation_id") != generation_id
            or raw.get("source_sha256") != binding.source_sha256
            or raw.get("source_size_bytes") != binding.byte_size
            or type(raw.get("generation_id")) is not str
            or type(raw.get("specification_hash")) is not str
            or _SHA256_RE.fullmatch(str(raw["specification_hash"])) is None
            or type(raw.get(content_field)) is not str
            or _SHA256_RE.fullmatch(str(raw[content_field])) is None
            or isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isclose(
                float(duration),
                binding.duration_seconds,
                rel_tol=1e-6,
                abs_tol=0.05,
            )
        ):
            return None
        try:
            return json.dumps(
                raw,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError):
            return None

    def _read_external_release_snapshot(
        self,
        video_ids: tuple[str, ...],
    ) -> ExternalIndexReleaseSnapshot:
        selected = frozenset(video_ids)
        if (
            not video_ids
            or len(selected) != len(video_ids)
            or any(type(video_id) is not str or not video_id for video_id in video_ids)
        ):
            raise SearchDependencyError("external release scope is invalid")
        try:
            raw = self._service.repository.get_external_index_release_snapshot(
                video_ids
            )
        except Exception as error:
            raise SearchDependencyError(
                "external index release snapshot is unavailable"
            ) from error
        if (
            not isinstance(raw, ExternalIndexReleaseSnapshot)
            or type(raw.job_backed_video_ids) is not frozenset
            or not raw.job_backed_video_ids <= selected
            or any(type(video_id) is not str for video_id in raw.job_backed_video_ids)
            or type(raw.visual_dense) is not dict
            or type(raw.lighthouse) is not dict
        ):
            raise SearchDependencyError("external index release snapshot is invalid")
        normalized: dict[StageKind, dict[str, dict[str, object]]] = {}
        for kind, bindings in (
            (StageKind.VISUAL_DENSE, raw.visual_dense),
            (StageKind.LIGHTHOUSE, raw.lighthouse),
        ):
            if any(
                type(video_id) is not str
                or video_id not in raw.job_backed_video_ids
                or type(descriptor) is not dict
                for video_id, descriptor in bindings.items()
            ):
                raise SearchDependencyError(
                    "external index release snapshot is invalid"
                )
            normalized[kind] = {
                video_id: dict(descriptor)
                for video_id, descriptor in bindings.items()
            }
        return ExternalIndexReleaseSnapshot(
            job_backed_video_ids=frozenset(raw.job_backed_video_ids),
            visual_dense=normalized[StageKind.VISUAL_DENSE],
            lighthouse=normalized[StageKind.LIGHTHOUSE],
        )

    def _pin_released_generation_provider(
        self,
        provider: object | None,
        binding: SearchAssetBinding,
        released_descriptor: object,
        *,
        content_field: str,
        check_readiness: bool = True,
    ) -> tuple[str, str | None, str | None]:
        if provider is None:
            return "not_configured", None, None
        if check_readiness and not self._provider_is_ready(
            provider,
            check_index=False,
        ):
            return "not_configured", None, None
        descriptor = getattr(provider, "generation_descriptor", None)
        search_generations = getattr(provider, "search_generations", None)
        if not all(callable(item) for item in (descriptor, search_generations)):
            return "not_configured", None, None
        if released_descriptor is None:
            return "missing", None, None
        generation_id = (
            released_descriptor.get("generation_id")
            if isinstance(released_descriptor, dict)
            else None
        )
        if type(generation_id) is not str:
            return "stale", None, None
        released_json = self._validated_descriptor(
            released_descriptor,
            binding,
            generation_id=generation_id,
            content_field=content_field,
        )
        if released_json is None:
            return "stale", generation_id, None
        try:
            current_json = self._validated_descriptor(
                descriptor(binding.video_id, generation_id),
                binding,
                generation_id=generation_id,
                content_field=content_field,
            )
        except Exception:
            return "failed", generation_id, None
        if current_json != released_json:
            return "stale", generation_id, None
        return "complete", generation_id, released_json

    def _pin_generation_provider(
        self,
        provider: object | None,
        binding: SearchAssetBinding,
        *,
        content_field: str,
        check_readiness: bool = True,
    ) -> tuple[str, str | None, str | None]:
        if provider is None:
            return "not_configured", None, None
        if check_readiness and not self._provider_is_ready(
            provider,
            check_index=False,
        ):
            return "not_configured", None, None
        active_generation_id = getattr(provider, "active_generation_id", None)
        descriptor = getattr(provider, "generation_descriptor", None)
        search_generations = getattr(provider, "search_generations", None)
        if not all(callable(item) for item in (active_generation_id, descriptor, search_generations)):
            return "not_configured", None, None
        try:
            generation_id = active_generation_id(binding.video_id)
        except Exception:
            return "failed", None, None
        if generation_id is None:
            return "missing", None, None
        try:
            descriptor_json = self._validated_descriptor(
                descriptor(binding.video_id, generation_id),
                binding,
                generation_id=generation_id,
                content_field=content_field,
            )
        except Exception:
            return "failed", generation_id, None
        if descriptor_json is None:
            return "stale", generation_id, None
        return "complete", generation_id, descriptor_json

    def _force_generation_descriptor_is_current(
        self,
        provider: object | None,
        state: _PinnedAssetState,
        *,
        generation_id: str | None,
        expected_descriptor_json: str | None,
        content_field: str,
    ) -> bool:
        descriptor = getattr(provider, "generation_descriptor", None)
        if (
            not callable(descriptor)
            or generation_id is None
            or expected_descriptor_json is None
        ):
            return False
        try:
            current_json = self._validated_descriptor(
                descriptor(
                    state.binding.video_id,
                    generation_id,
                    force_content_validation=True,
                ),
                state.binding,
                generation_id=generation_id,
                content_field=content_field,
            )
        except Exception:
            return False
        return current_json == expected_descriptor_json

    def _pin_reranker(self) -> tuple[str, Mapping[str, object] | None]:
        if self.configuration.reranker == "none":
            return "not_configured", None
        provider = self.reranker
        if provider is None:
            return "not_configured", None
        if not self._provider_is_ready(provider, check_index=True):
            return "not_configured", None
        try:
            attestation_source = getattr(provider, "benchmark_attestation", None)
            strict = getattr(provider, "rerank_strict", None)
        except Exception:
            return "failed", None
        if attestation_source is None or not callable(strict):
            return "not_configured", None
        try:
            raw = (
                attestation_source()
                if callable(attestation_source)
                else attestation_source
            )
        except Exception:
            return "failed", None
        if not isinstance(raw, dict):
            return "failed", None
        expected_provider = {
            "qwen": "qwen-video",
            "internvideo": "internvideo",
        }[self.configuration.reranker]
        required_strings = ("model_identity", "protocol_identity", "runtime_identity")
        if (
            raw.get("provider") != expected_provider
            or raw.get("source_bound") is not True
            or raw.get("strict_complete") is not True
            or raw.get("candidate_limit")
            != self.configuration.reranker_candidate_limit
            or any(type(raw.get(name)) is not str or not str(raw[name]) for name in required_strings)
        ):
            return "not_configured", None
        try:
            canonical = json.loads(
                json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False)
            )
        except (TypeError, ValueError):
            return "failed", None
        return "complete", MappingProxyType(canonical)

    @staticmethod
    def _canonical_attestation(raw: object) -> Mapping[str, object] | None:
        if not isinstance(raw, dict):
            return None
        try:
            canonical = json.loads(
                json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False)
            )
        except (TypeError, ValueError):
            return None
        return MappingProxyType(canonical) if isinstance(canonical, dict) else None

    def _pin_vector_index(self) -> tuple[str, Mapping[str, object] | None]:
        if self.configuration.text_search == "disabled":
            return "not_configured", None
        provider = self._service.vector_index
        try:
            attestation_source = getattr(provider, "benchmark_attestation", None)
            index_specification = getattr(provider, "index_specification", None)
        except Exception:
            return "failed", None
        if attestation_source is None or index_specification is None:
            return "not_configured", None
        try:
            raw_attestation = (
                attestation_source()
                if callable(attestation_source)
                else attestation_source
            )
            attestation = self._canonical_attestation(raw_attestation)
        except Exception:
            return "failed", None
        if attestation is None:
            return "failed", None
        embedding = attestation.get("embedding")
        index = attestation.get("index")
        embedding_strings = (
            "algorithm_version",
            "embedding_identity",
            "model_name",
            "model_repository",
            "model_revision",
            "runtime_version",
        )
        if (
            attestation.get("schema_version") != 1
            or attestation.get("provider") != getattr(provider, "id", None)
            or attestation.get("strict_no_fallback") is not True
            or not isinstance(embedding, dict)
            or not isinstance(index, dict)
            or index.get("index_specification_hash")
            != getattr(index_specification, "specification_hash", None)
            or index.get("collection_name")
            != getattr(index_specification, "collection_name", None)
            or embedding.get("embedding_identity")
            != getattr(index_specification, "embedding_identity", None)
            or embedding.get("dimensions")
            != getattr(index_specification, "dimensions", None)
            or _SHA256_RE.fullmatch(
                str(embedding.get("model_content_sha256", ""))
            )
            is None
            or _SHA256_RE.fullmatch(str(index.get("snapshot_sha256", ""))) is None
            or any(
                type(embedding.get(name)) is not str
                or not str(embedding[name])
                for name in embedding_strings
            )
        ):
            return "not_configured", None
        return "complete", attestation

    def _pin_temporal_refiner(self) -> tuple[str, Mapping[str, object] | None]:
        if not self.configuration.temporal_refinement:
            return "not_configured", None
        provider = self._service.temporal_refiner
        try:
            attestation_source = getattr(provider, "benchmark_attestation", None)
            strict = getattr(provider, "refine_strict", None)
        except Exception:
            return "failed", None
        if attestation_source is None or not callable(strict):
            return "not_configured", None
        try:
            raw_attestation = (
                attestation_source()
                if callable(attestation_source)
                else attestation_source
            )
            attestation = self._canonical_attestation(raw_attestation)
        except Exception:
            return "failed", None
        if attestation is None:
            return "failed", None
        required_strings = (
            "ffmpeg_identity",
            "implementation_identity",
            "runtime_identity",
            "scorer_identity",
            "scratch_policy_identity",
        )
        if (
            attestation.get("source_bound") is not True
            or attestation.get("strict_complete") is not True
            or any(
                type(attestation.get(name)) is not str
                or not str(attestation[name])
                for name in required_strings
            )
        ):
            return "not_configured", None
        return "complete", attestation

    def _pin_asset(self, binding: SearchAssetBinding) -> _PinnedAssetState:
        fingerprint = self._asset_fingerprint(
            binding,
            verify_content=True,
        )
        text_state, text_id, text_binding, segments = self._pin_text(binding)
        external_release = self._external_release
        external_release_bound = (
            binding.video_id in external_release.job_backed_video_ids
            if external_release is not None
            else None
        )
        if self.configuration.visual_search == "disabled":
            visual_state, visual_id, visual_descriptor = (
                "not_configured",
                None,
                None,
            )
        elif external_release_bound:
            assert external_release is not None
            visual_state, visual_id, visual_descriptor = (
                self._pin_released_generation_provider(
                    self._service.visual_search,
                    binding,
                    external_release.visual_dense.get(binding.video_id),
                    content_field="content_sha256",
                )
            )
        else:
            visual_state, visual_id, visual_descriptor = self._pin_generation_provider(
                self._service.visual_search,
                binding,
                content_field="content_sha256",
            )
        if not self.configuration.lighthouse:
            lighthouse_state, lighthouse_id, lighthouse_descriptor = (
                "not_configured",
                None,
                None,
            )
        elif external_release_bound:
            assert external_release is not None
            lighthouse_state, lighthouse_id, lighthouse_descriptor = (
                self._pin_released_generation_provider(
                    self._service.moment_search,
                    binding,
                    external_release.lighthouse.get(binding.video_id),
                    content_field="manifest_sha256",
                )
            )
        else:
            lighthouse_state, lighthouse_id, lighthouse_descriptor = (
                self._pin_generation_provider(
                    self._service.moment_search,
                    binding,
                    content_field="manifest_sha256",
                )
            )
        return _PinnedAssetState(
            binding=binding,
            video_fingerprint=fingerprint,
            external_release_bound=external_release_bound,
            text_state=text_state,
            active_text_generation_id=text_id,
            text_binding=text_binding,
            segment_snapshots=segments,
            visual_state=visual_state,
            active_visual_generation_id=visual_id,
            visual_descriptor_json=visual_descriptor,
            lighthouse_state=lighthouse_state,
            active_lighthouse_generation_id=lighthouse_id,
            lighthouse_descriptor_json=lighthouse_descriptor,
        )

    def _capability_state_without_drift(
        self,
        state: _PinnedAssetState,
        capability: str,
    ) -> str:
        if capability == "text_vectors":
            return state.text_state
        if capability == "visual_dense":
            return state.visual_state
        if capability == "temporal_refinement":
            if state.visual_state != "complete":
                return state.visual_state
            return self._temporal_state
        if capability == "lighthouse":
            return state.lighthouse_state
        if capability == "qwen_verification":
            return (
                self._reranker_state
                if self.configuration.reranker == "qwen"
                else "not_configured"
            )
        if capability == "internvideo":
            return (
                self._reranker_state
                if self.configuration.reranker == "internvideo"
                else "not_configured"
            )
        raise ValueError("unsupported product search capability")

    def _required_capabilities(self) -> tuple[str, ...]:
        required: list[str] = []
        if self.configuration.text_search != "disabled":
            required.append("text_vectors")
        if self.configuration.visual_search != "disabled":
            required.append("visual_dense")
        if self.configuration.temporal_refinement:
            required.append("temporal_refinement")
        if self.configuration.lighthouse:
            required.append("lighthouse")
        if self.configuration.reranker == "qwen":
            required.append("qwen_verification")
        elif self.configuration.reranker == "internvideo":
            required.append("internvideo")
        return tuple(required)

    def _state_is_current(
        self,
        state: _PinnedAssetState,
        *,
        external_release: ExternalIndexReleaseSnapshot | None,
        verify_persisted_content: bool = True,
    ) -> bool:
        try:
            if self._asset_fingerprint(
                state.binding,
                verify_content=False,
            ) != state.video_fingerprint:
                return False
            if state.external_release_bound is not None:
                if external_release is None:
                    return False
                if (
                    state.binding.video_id
                    in external_release.job_backed_video_ids
                ) != state.external_release_bound:
                    return False
            if self.configuration.text_search != "disabled":
                if state.text_state != "not_configured":
                    active_id = (
                        self._service.repository.get_active_text_vector_generation_id(
                            state.binding.video_id
                        )
                    )
                    if active_id != state.active_text_generation_id:
                        return False
                if state.text_state == "complete" and verify_persisted_content:
                    assert state.text_binding is not None
                    current_binding = (
                        self._service.repository.get_text_vector_search_binding_bounded(
                            state.text_binding.generation_id,
                            max_points=_BENCHMARK_MAX_TEXT_POINTS,
                            max_segments=_BENCHMARK_MAX_SEGMENTS,
                            max_text_bytes=_BENCHMARK_MAX_TEXT_BYTES,
                            max_metadata_bytes=_BENCHMARK_MAX_METADATA_BYTES,
                        )
                    )
                    if current_binding != state.text_binding:
                        return False
                    for kind, generation, segments in state.segment_snapshots:
                        active_segment = (
                            self._service.repository.get_active_segment_generation(
                                state.binding.video_id,
                                kind,
                            )
                        )
                        if (
                            active_segment is None
                            or active_segment.generation_id != generation.generation_id
                            or self._service.repository.get_segment_generation_snapshot_bounded(
                                generation.generation_id,
                                max_segments=_BENCHMARK_MAX_SEGMENTS,
                                max_text_bytes=_BENCHMARK_MAX_TEXT_BYTES,
                                max_metadata_bytes=_BENCHMARK_MAX_METADATA_BYTES,
                            )
                            != (generation, segments)
                        ):
                            return False
            if self.configuration.visual_search != "disabled":
                current_visual = (
                    self._pin_released_generation_provider(
                        self._service.visual_search,
                        state.binding,
                        (
                            external_release.visual_dense.get(
                                state.binding.video_id
                            )
                            if external_release is not None
                            else None
                        ),
                        content_field="content_sha256",
                        check_readiness=False,
                    )
                    if state.external_release_bound
                    else self._pin_generation_provider(
                        self._service.visual_search,
                        state.binding,
                        content_field="content_sha256",
                        check_readiness=False,
                    )
                )
                if current_visual != (
                    state.visual_state,
                    state.active_visual_generation_id,
                    state.visual_descriptor_json,
                ):
                    return False
            if self.configuration.lighthouse:
                current_lighthouse = (
                    self._pin_released_generation_provider(
                        self._service.moment_search,
                        state.binding,
                        (
                            external_release.lighthouse.get(
                                state.binding.video_id
                            )
                            if external_release is not None
                            else None
                        ),
                        content_field="manifest_sha256",
                        check_readiness=False,
                    )
                    if state.external_release_bound
                    else self._pin_generation_provider(
                        self._service.moment_search,
                        state.binding,
                        content_field="manifest_sha256",
                        check_readiness=False,
                    )
                )
                if current_lighthouse != (
                    state.lighthouse_state,
                    state.active_lighthouse_generation_id,
                    state.lighthouse_descriptor_json,
                ):
                    return False
            if (
                verify_persisted_content
                and self.configuration.text_search != "disabled"
            ):
                if (
                    self.indexing_specifications
                    != self._service._current_indexing_specifications()
                ):
                    return False
                if self._service.lexicon is not None and (
                    self._service.lexicon.read() != self._lexicon_snapshot
                ):
                    return False
            if not self._provider_identities_are_current():
                return False
            if (
                verify_persisted_content
                and self.configuration.text_search != "disabled"
            ):
                current_state, current_attestation = self._pin_vector_index()
                if (
                    current_state != self._vector_index_state
                    or dict(current_attestation or {})
                    != dict(self._vector_index_attestation or {})
                ):
                    return False
            if verify_persisted_content and self.configuration.temporal_refinement:
                current_state, current_attestation = self._pin_temporal_refiner()
                if (
                    current_state != self._temporal_state
                    or dict(current_attestation or {})
                    != dict(self._temporal_attestation or {})
                ):
                    return False
            if verify_persisted_content and self.configuration.reranker != "none":
                current_state, current_attestation = self._pin_reranker()
                if (
                    current_state != self._reranker_state
                    or dict(current_attestation or {})
                    != dict(self._reranker_attestation or {})
                ):
                    return False
        except Exception:
            return False
        return True

    def _current_external_release_for_states(
        self,
        states: Iterable[_PinnedAssetState],
    ) -> ExternalIndexReleaseSnapshot | None:
        video_ids = tuple(
            state.binding.video_id
            for state in states
            if state.external_release_bound is not None
        )
        if not video_ids:
            return None
        return self._read_external_release_snapshot(video_ids)

    def capability_state(self, video_id: str, capability: str) -> str:
        self._ensure_open()
        if type(video_id) is not str or video_id not in self._state_by_video:
            raise ValueError("capability check uses an unpinned video")
        if capability not in self._CAPABILITIES:
            raise ValueError("unsupported product search capability")
        state = self._state_by_video[video_id]
        try:
            external_release = self._current_external_release_for_states((state,))
        except SearchDependencyError:
            return "stale"
        if not self._state_is_current(
            state,
            external_release=external_release,
        ):
            return "stale"
        return self._capability_state_without_drift(state, capability)

    def _assert_current(
        self,
        video_ids: tuple[str, ...],
        *,
        verify_persisted_content: bool = True,
    ) -> None:
        states = tuple(self._state_by_video[video_id] for video_id in video_ids)
        try:
            external_release = self._current_external_release_for_states(states)
        except SearchDependencyError as error:
            raise SearchDependencyError(
                "pinned product search snapshot changed"
            ) from error
        if not self._provider_identities_are_current() or any(
            not self._state_is_current(
                state,
                external_release=external_release,
                verify_persisted_content=verify_persisted_content,
            )
            for state in states
        ):
            raise SearchDependencyError("pinned product search snapshot changed")

    def expand_query(self, query: str) -> list[str]:
        output = [query.strip()]
        normalized_query = normalize_text(query)
        for canonical, aliases in self._lexicon_snapshot.items():
            variants = [canonical, *aliases]
            if any(
                normalized and normalized in normalized_query
                for normalized in (normalize_text(value) for value in variants)
            ):
                output.extend(variants)
        return list(dict.fromkeys(value for value in output if value))

    def segments_for(
        self,
        video_ids: list[str],
        kinds: set[StageKind],
    ) -> tuple[SegmentRecord, ...]:
        return tuple(
            segment
            for video_id in video_ids
            for kind, _generation, segments in self._state_by_video[
                video_id
            ].segment_snapshots
            if kind in kinds
            for segment in segments
        )

    def text_bindings_for(
        self,
        video_ids: list[str],
    ) -> tuple[TextVectorSearchBinding, ...]:
        return tuple(
            state.text_binding
            for video_id in video_ids
            for state in [self._state_by_video[video_id]]
            if state.text_binding is not None and state.text_state == "complete"
        )

    @staticmethod
    def _descriptor_value(value: str | None) -> dict[str, object]:
        if value is None:
            raise SearchDependencyError("pinned generation descriptor is unavailable")
        raw = json.loads(value)
        if not isinstance(raw, dict):
            raise SearchDependencyError("pinned generation descriptor is invalid")
        return raw

    def visual_bindings_for(
        self,
        video_ids: list[str],
    ) -> dict[str, dict[str, object]]:
        return {
            video_id: self._descriptor_value(
                self._state_by_video[video_id].visual_descriptor_json
            )
            for video_id in video_ids
        }

    def lighthouse_bindings_for(
        self,
        video_ids: list[str],
    ) -> dict[str, dict[str, object]]:
        return {
            video_id: self._descriptor_value(
                self._state_by_video[video_id].lighthouse_descriptor_json
            )
            for video_id in video_ids
        }

    def search(
        self,
        query: str,
        video_ids: tuple[str, ...],
        *,
        limit: int,
    ) -> list[SearchResultView]:
        self._ensure_open()
        self._last_search_execution_trace = None
        if self._active_execution_recorder is not None:
            raise RuntimeError("pinned product search is already active")
        if (
            type(video_ids) is not tuple
            or not video_ids
            or len(set(video_ids)) != len(video_ids)
            or any(video_id not in self._state_by_video for video_id in video_ids)
        ):
            raise ValueError("search must use a non-empty subset of pinned videos")
        if limit != self.configuration.result_limit:
            raise ValueError("search limit differs from frozen evaluation plan")
        for video_id in video_ids:
            state = self._state_by_video[video_id]
            for capability in self._required_capabilities():
                if self._capability_state_without_drift(state, capability) != "complete":
                    raise SearchDependencyError(
                        f"required pinned capability {capability} is unavailable"
                    )
        self._assert_current(video_ids, verify_persisted_content=False)
        recorder = _ProductSearchExecutionRecorder(
            owner=self,
            configuration_identity=self.configuration.identity,
        )
        self._active_execution_recorder = recorder
        try:
            results = self._service.search(
                query,
                video_ids=list(video_ids),
                limit=limit,
                use_lighthouse=self.configuration.lighthouse,
                mode="all",
                raise_on_provider_error=True,
                _evaluation_configuration=self.configuration,
                _pinned_session=self,
                _execution_recorder=recorder,
            )
            self._assert_current(video_ids, verify_persisted_content=False)
            if any(result.video_id not in video_ids for result in results):
                raise SearchDependencyError("product search returned an unpinned video")
            trace = recorder.finish(owner=self)
        finally:
            self._active_execution_recorder = None
        self._last_search_execution_trace = trace
        return results

    def _build_identities(self) -> ProductSearchIdentities:
        vector_specification = getattr(
            self._service.vector_index,
            "index_specification",
            None,
        )
        model_identities: list[ProductSearchComponentIdentity] = []
        if self.configuration.text_search != "disabled":
            model_identities.append(
                ProductSearchComponentIdentity(
                    "text_embedding",
                    str(
                        getattr(
                            vector_specification,
                            "embedding_identity",
                            "not-configured",
                        )
                    ),
                )
            )
        if self.configuration.visual_search != "disabled":
            model_identities.append(
                ProductSearchComponentIdentity(
                    "visual_embedding",
                    str(getattr(self._service.visual_search, "model_identity", "not-configured")),
                )
            )
        if self.configuration.lighthouse:
            model_identities.append(
                ProductSearchComponentIdentity(
                    "lighthouse_model",
                    self._provider_identities["lighthouse"],
                )
            )
        if self.configuration.reranker != "none":
            model_identities.append(
                ProductSearchComponentIdentity(
                    f"{self.configuration.reranker}_reranker",
                    (
                        str(self._reranker_attestation.get("model_identity"))
                        if self._reranker_attestation is not None
                        else "not-configured"
                    ),
                )
            )

        text_generations = [
            {
                "asset_id": state.binding.external_id,
                "generation_id": state.active_text_generation_id,
                "state": state.text_state,
                "vector_manifest_sha256": (
                    state.text_binding.generation.vector_manifest_sha256
                    if state.text_binding is not None
                    else None
                ),
            }
            for state in self._states
        ]
        index_identities: list[ProductSearchComponentIdentity] = []
        if self.configuration.text_search != "disabled":
            index_identities.extend(
                [
                    ProductSearchComponentIdentity(
                        "text_vector_index",
                        str(
                            getattr(
                                vector_specification,
                                "specification_hash",
                                "not-configured",
                            )
                        ),
                    ),
                    ProductSearchComponentIdentity(
                        "text_vector_generations",
                        f"sha256:{_canonical_digest(text_generations)}",
                    ),
                ]
            )
        if self.configuration.visual_search != "disabled":
            index_identities.append(
                ProductSearchComponentIdentity(
                    "visual_generations",
                    f"sha256:{_canonical_digest([{'asset_id': state.binding.external_id, 'descriptor': json.loads(state.visual_descriptor_json) if state.visual_descriptor_json else None, 'state': state.visual_state} for state in self._states])}",
                )
            )
        if self.configuration.lighthouse:
            index_identities.append(
                ProductSearchComponentIdentity(
                    "lighthouse_generations",
                    f"sha256:{_canonical_digest([{'asset_id': state.binding.external_id, 'descriptor': json.loads(state.lighthouse_descriptor_json) if state.lighthouse_descriptor_json else None, 'state': state.lighthouse_state} for state in self._states])}",
                )
            )
        runtime_payload = {
            "indexing_specifications": (
                {
                    specification.kind.value: specification.specification_hash
                    for specification in self.indexing_specifications.segment_specifications
                }
                | {
                    "text_vectors": self.indexing_specifications.text_vectors.specification_hash
                }
                if self.configuration.text_search != "disabled"
                and self.indexing_specifications is not None
                else None
            ),
            "lexicon": (
                self._lexicon_snapshot
                if self.configuration.text_search != "disabled"
                else None
            ),
            "provider_identities": dict(self._provider_identities),
            "reranker_attestation": dict(self._reranker_attestation or {}),
            "temporal_attestation": dict(self._temporal_attestation or {}),
            "vector_index_attestation": dict(self._vector_index_attestation or {}),
            "semantic_text_min_score": (
                self._service.semantic_text_min_score
                if self.configuration.text_search != "disabled"
                else None
            ),
            "visual_min_score": (
                self._service.visual_min_score
                if self.configuration.visual_search != "disabled"
                else None
            ),
        }
        config_identities = (
            ProductSearchComponentIdentity(
                "evaluation_search_configuration",
                self.configuration.identity,
            ),
            ProductSearchComponentIdentity(
                "product_search_runtime",
                f"sha256:{_canonical_digest(runtime_payload)}",
            ),
            ProductSearchComponentIdentity(
                "product_search_lifecycle",
                self._lifecycle_identity,
            ),
        )
        return ProductSearchIdentities(
            model=tuple(model_identities),
            index=tuple(index_identities),
            config=config_identities,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._last_search_execution_trace = None
        failures: list[Exception] = []
        try:
            external_release = self._current_external_release_for_states(self._states)
        except Exception as error:
            failures.append(error)
            # Keep checking media and provider content against the immutable
            # opening snapshot; the release-read failure still makes close fail.
            external_release = self._external_release
        for state in self._states:
            try:
                if not self._state_is_current(
                    state,
                    external_release=external_release,
                    verify_persisted_content=True,
                ):
                    raise SearchDependencyError(
                        "pinned product state changed during evaluation"
                    )
                if self._asset_fingerprint(
                    state.binding,
                    verify_content=True,
                ) != state.video_fingerprint:
                    raise SearchDependencyError(
                        "pinned media source changed during evaluation"
                    )
                if (
                    state.visual_state == "complete"
                    and not self._force_generation_descriptor_is_current(
                        self._service.visual_search,
                        state,
                        generation_id=state.active_visual_generation_id,
                        expected_descriptor_json=state.visual_descriptor_json,
                        content_field="content_sha256",
                    )
                ):
                    raise SearchDependencyError(
                        "pinned visual generation failed final validation"
                    )
                if (
                    state.lighthouse_state == "complete"
                    and not self._force_generation_descriptor_is_current(
                        self._service.moment_search,
                        state,
                        generation_id=state.active_lighthouse_generation_id,
                        expected_descriptor_json=state.lighthouse_descriptor_json,
                        content_field="manifest_sha256",
                    )
                ):
                    raise SearchDependencyError(
                        "pinned Lighthouse generation failed final validation"
                    )
            except Exception as error:
                failures.append(error)
        if self.configuration.text_search != "disabled":
            verify_snapshot = getattr(
                self._service.vector_index,
                "verify_benchmark_snapshot_current",
                None,
            )
            try:
                if not callable(verify_snapshot) or not verify_snapshot():
                    raise SearchDependencyError(
                        "pinned text vector storage snapshot changed"
                    )
            except Exception as error:
                failures.append(error)
            finally:
                try:
                    self._release_text_attestations(self._states)
                except Exception as error:
                    failures.append(error)
        if failures:
            raise SearchDependencyError(
                "pinned product snapshot failed final identity verification"
            ) from failures[0]
