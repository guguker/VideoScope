from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
import logging
import math
from numbers import Real
from pathlib import Path
import re
from typing import Protocol
from urllib.parse import quote

from videoscope.artifacts import (
    IndexingSpecifications,
    StageKind,
    StageSpecification,
    TextVectorSearchBinding,
    TextVectorSearchHit,
)
from videoscope.repository import Repository, SegmentRecord
from videoscope.search.fusion import EvidenceHit, calibrate_hits, fuse_hits
from videoscope.search.query_router import QueryRouter
from videoscope.search.text_matching import SearchLexicon, lexical_match
from videoscope.search.text_matching import tokens as text_tokens


logger = logging.getLogger(__name__)

_TRUSTED_ENTITY_MATCHES = {"exact", "stem", "transliteration"}


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
        semantic_text_min_score: float = 0.42,
        visual_min_score: float = 0.18,
        specification_resolver: Callable[[], IndexingSpecifications] | None = None,
        segment_specifications: Iterable[StageSpecification] | None = None,
        thumbnails_dir: Path | None = None,
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
    ) -> list[SearchResultView]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("search query is empty")
        if mode not in SEARCH_MODALITIES:
            raise ValueError("unsupported search mode")

        plan = self.query_router.route(
            normalized_query,
            mode=mode,
            requested_lighthouse=use_lighthouse,
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
        query_variants = self.lexicon.expand(normalized_query) if self.lexicon else [normalized_query]

        current_specifications: tuple[StageSpecification, ...] = ()
        selected_specifications: tuple[StageSpecification, ...] = ()
        indexing_specifications: IndexingSpecifications | None = None
        try:
            indexing_specifications = self._current_indexing_specifications()
            current_specifications = (
                indexing_specifications.segment_specifications
                if indexing_specifications is not None
                else self._static_segment_specifications or ()
            )
            if raise_on_provider_error:
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
        if scene_specification is not None:
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
            indexing_specifications is not None
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
                        exhaustive_validation=raise_on_provider_error,
                    )
                    assert indexing_specifications is not None
                    post_search_bindings = (
                        self.repository.list_current_text_vector_bindings(
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
                hits.extend(hit for hit in hybrid_hits if hit.score >= semantic_threshold)
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

        if (
            "visual" in allowed_modalities
            and self.visual_search is not None
            and ready_video_ids
        ):
            try:
                visual_hits = self.visual_search.search(
                    normalized_query,
                    video_ids=ready_video_ids,
                    limit=max(limit * 2, 20),
                )
                visual_hits = [
                    hit
                    for hit in visual_hits
                    if hit.video_id in ready_videos
                    and hit.score >= self.visual_min_score
                ]
                if plan.refine_temporally and self.temporal_refiner is not None:
                    refine = self.temporal_refiner.refine
                    if raise_on_provider_error:
                        refine_strict = getattr(
                            self.temporal_refiner,
                            "refine_strict",
                            None,
                        )
                        if callable(refine_strict):
                            refine = refine_strict
                    visual_hits = refine(normalized_query, visual_hits)
                hits.extend(visual_hits)
            except Exception as error:
                logger.exception("SigLIP visual search failed")
                if raise_on_provider_error:
                    raise SearchDependencyError("Visual search failed") from error

        if (
            "lighthouse" in allowed_modalities
            and plan.use_lighthouse
            and self.moment_search is not None
            and ready_video_ids
        ):
            try:
                lighthouse_hits = self.moment_search.search(
                    normalized_query,
                    ready_video_ids,
                    limit=max(limit * 2, 20),
                )
                viable = [
                    hit
                    for hit in lighthouse_hits
                    if hit.video_id in ready_videos
                    and hit.score >= self.visual_min_score
                ]
                support = [hit for hit in hits if hit.modality in {"visual", "objects"}]
                hits.extend(corroborate_lighthouse_hits(viable, support))
            except Exception as error:
                logger.exception("Lighthouse search failed")
                if raise_on_provider_error:
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
        if self.candidate_reranker is not None and plan.intent in {"action", "mixed"}:
            try:
                rerank = self.candidate_reranker.rerank
                if raise_on_provider_error:
                    rerank_strict = getattr(
                        self.candidate_reranker,
                        "rerank_strict",
                        None,
                    )
                    if callable(rerank_strict):
                        rerank = rerank_strict
                fused = rerank(normalized_query, fused)  # type: ignore[assignment, arg-type]
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
