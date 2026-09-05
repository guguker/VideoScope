from dataclasses import replace

import pytest

from videoscope.artifacts import (
    IndexingSpecifications,
    StageKind,
    StageSpecification,
    StageState,
)
from videoscope.repository import (
    ExternalIndexReleaseSnapshot,
    Repository,
    SegmentRecord,
)
from videoscope.search.service import (
    EvaluationSearchConfiguration,
    ProductSearchExecutionTrace,
    SearchDependencyError,
    SearchAssetBinding,
    SearchService,
    _hybridize,
    corroborate_lighthouse_hits,
    refine_speech_hit,
)
from videoscope.search.vector_index import EmptyVectorIndex, MemoryVectorIndex
from videoscope.search.fusion import EvidenceHit, FusedResult
from videoscope.search.visual_index import SiglipVisualIndex
from videoscope.providers.base import ProviderState, ProviderStatus


_PINNED_SOURCE_SHA256 = (
    "84d89877f0d4041efb6bf91a16f0248f2fd573e6af05c19f96bedb9f882f7882"
)


def _stage_specification(kind: StageKind) -> StageSpecification:
    return StageSpecification(
        kind=kind,
        schema_version=1,
        implementation_revision=f"tests.search.{kind.value}.v1",
    )


def _attest_memory_index(index: MemoryVectorIndex) -> None:
    index.benchmark_attestation = lambda: {  # type: ignore[attr-defined]
        "embedding": {
            "algorithm_version": "hash-embedding-test-v1",
            "dimensions": index.index_specification.dimensions,
            "embedding_identity": index.index_specification.embedding_identity,
            "model_content_sha256": "8" * 64,
            "model_name": "memory-vector-test-model",
            "model_repository": "local-test-fixture",
            "model_revision": "test-revision@1",
            "runtime_version": "memory-vector-test-runtime@1",
        },
        "index": {
            "collection_name": index.index_specification.collection_name,
            "index_specification_hash": index.index_specification.specification_hash,
            "snapshot_sha256": "7" * 64,
        },
        "provider": index.id,
        "schema_version": 1,
        "strict_no_fallback": True,
    }
    index.validate_generation_for_benchmark_snapshot = (  # type: ignore[attr-defined]
        lambda binding: index.validate_generation(binding, exhaustive=True)
    )
    index.verify_benchmark_snapshot_current = lambda: True  # type: ignore[attr-defined]
    index.release_benchmark_snapshot_bindings = (  # type: ignore[attr-defined]
        lambda _bindings: None
    )


def _publish_segments(
    repository: Repository,
    specification: StageSpecification,
    segments: list[SegmentRecord],
) -> None:
    run = repository.create_stage_run(
        video_id="video-1",
        specification=specification,
    )
    repository.transition_stage_run(run.run_id, StageState.RUNNING)
    repository.commit_segment_generation(run.run_id, segments=segments)


def _indexing_specifications() -> IndexingSpecifications:
    scenes = _stage_specification(StageKind.SCENES)
    speech = _stage_specification(StageKind.SPEECH)
    ocr = _stage_specification(StageKind.OCR)
    objects = _stage_specification(StageKind.OBJECTS)
    text_vectors = StageSpecification(
        kind=StageKind.TEXT_VECTORS,
        schema_version=1,
        implementation_revision="tests.search.text-vectors.v1",
        model_identity="hash-embedding-v1:384",
        dependencies={
            "speech_specification": speech.specification_hash,
            "ocr_specification": ocr.specification_hash,
            "objects_specification": objects.specification_hash,
        },
    )
    return IndexingSpecifications(
        scenes=scenes,
        speech=speech,
        ocr=ocr,
        objects=objects,
        text_vectors=text_vectors,
    )


def _publish_text_vectors(
    repository: Repository,
    index: MemoryVectorIndex,
    specifications: IndexingSpecifications,
) -> str:
    queued = repository.create_stage_run(
        video_id="video-1",
        specification=specifications.text_vectors,
    )
    run = repository.transition_stage_run(queued.run_id, StageState.RUNNING)
    plan = repository.reserve_text_vector_generation(
        run.run_id,
        index_specification=index.index_specification,
        semantic_specifications=specifications.semantic_segment_specifications,
    )
    receipt = index.build_generation(plan)
    return repository.commit_text_vector_generation(
        run.run_id,
        receipt=receipt,
    ).generation_id


def test_search_service_returns_enriched_ranked_moment(tmp_path) -> None:
    repository = Repository(tmp_path / "db.sqlite3")
    repository.initialize()
    repository.create_video_with_asset(
        video_id="video-1",
        original_name="final.mp4",
        stored_name="video-1.mp4",
        media_path=str(tmp_path / "video-1.mp4"),
        size_bytes=100,
        source_sha256="0" * 64,
    )
    repository.update_video("video-1", status="ready")
    thumbnails = tmp_path / "thumbs"
    thumbnail_path = thumbnails / "video-1" / "scene.jpg"
    speech = SegmentRecord(
        id="speech-1",
        video_id="video-1",
        start=10,
        end=15,
        modality="speech",
        text="player makes a three point shot",
        confidence=0.9,
        metadata={},
        thumbnail_path=str(thumbnail_path),
    )
    ocr = SegmentRecord(
        id="ocr-1",
        video_id="video-1",
        start=11,
        end=14,
        modality="ocr",
        text="HOME 87 GUEST 86",
        confidence=0.8,
        metadata={},
        thumbnail_path=str(thumbnail_path),
    )
    speech_specification = _stage_specification(StageKind.SPEECH)
    ocr_specification = _stage_specification(StageKind.OCR)
    _publish_segments(repository, speech_specification, [speech])
    _publish_segments(repository, ocr_specification, [ocr])
    index = MemoryVectorIndex()
    index.replace_video("video-1", [speech, ocr])
    service = SearchService(
        repository,
        index,
        segment_specifications=[speech_specification, ocr_specification],
        thumbnails_dir=thumbnails,
    )

    results = service.search("three point shot", limit=10)

    assert results[0].video_id == "video-1"
    assert results[0].video_name == "final.mp4"
    assert results[0].start == 10
    assert results[0].thumbnail_url == "/api/thumbnails/video-1/scene.jpg"
    assert results[0].evidence[0].text


def test_search_service_rejects_blank_query(tmp_path) -> None:
    repository = Repository(tmp_path / "db.sqlite3")
    repository.initialize()
    service = SearchService(repository, MemoryVectorIndex())

    try:
        service.search("   ")
    except ValueError as error:
        assert "empty" in str(error)
    else:
        raise AssertionError("blank search query should fail")


def test_evaluation_search_fails_when_semantic_capability_is_absent(tmp_path) -> None:
    repository = Repository(tmp_path / "db.sqlite3")
    repository.initialize()
    service = SearchService(repository, EmptyVectorIndex())

    with pytest.raises(SearchDependencyError, match="Semantic"):
        service.search_for_evaluation(
            "spoken phrase",
            mode="speech",
            use_lighthouse=False,
        )


def test_evaluation_search_fails_when_visual_capability_is_absent(tmp_path) -> None:
    repository = Repository(tmp_path / "db.sqlite3")
    repository.initialize()
    service = SearchService(repository, MemoryVectorIndex())

    with pytest.raises(SearchDependencyError, match="Visual"):
        service.search_for_evaluation(
            "player jumps",
            mode="visual",
            use_lighthouse=False,
        )


def test_evaluation_search_fails_when_requested_lighthouse_is_absent(tmp_path) -> None:
    repository = Repository(tmp_path / "db.sqlite3")
    repository.initialize()
    service = SearchService(
        repository,
        MemoryVectorIndex(),
        visual_search=RecordingVisualIndex([]),
    )

    with pytest.raises(SearchDependencyError, match="Temporal"):
        service.search_for_evaluation(
            "player jumps",
            mode="visual",
            use_lighthouse=True,
        )


class RecordingIndex:
    def __init__(self, hits: list[EvidenceHit]) -> None:
        self.hits = hits
        self.modalities: set[str] | None = None
        self.video_ids: list[str] | None = None

    def search(self, _query, *, video_ids=None, modalities=None, limit=50):  # type: ignore[no-untyped-def]
        self.modalities = modalities
        self.video_ids = video_ids
        return self.hits[:limit]


class RecordingVisualIndex:
    def __init__(self, hits: list[EvidenceHit]) -> None:
        self.hits = hits
        self.called = False
        self.video_ids: list[str] | None = None

    def search(self, _query, *, video_ids=None, limit=50):  # type: ignore[no-untyped-def]
        self.called = True
        self.video_ids = video_ids
        return self.hits[:limit]


class RecordingMomentSearch:
    def __init__(self, hits: list[EvidenceHit]) -> None:
        self.hits = hits
        self.called = False
        self.video_ids: list[str] | None = None

    def search(self, _query, video_ids, *, limit=30):  # type: ignore[no-untyped-def]
        self.called = True
        self.video_ids = video_ids
        return self.hits[:limit]


class StaleMomentSearch(RecordingMomentSearch):
    def cache_is_current(self, _video_id: str) -> bool:
        return False


def _repository_with_video(tmp_path) -> Repository:
    repository = Repository(tmp_path / "search.sqlite3")
    repository.initialize()
    (tmp_path / "video-1.mp4").write_bytes(b"0123456789")
    repository.create_video_with_asset(
        video_id="video-1",
        original_name="match.mp4",
        stored_name="video-1.mp4",
        media_path=str(tmp_path / "video-1.mp4"),
        size_bytes=10,
        source_sha256=_PINNED_SOURCE_SHA256,
    )
    repository.update_video("video-1", status="ready")
    return repository


def test_evaluation_requires_current_visual_index_but_interactive_is_best_effort(
    tmp_path,
) -> None:
    repository = _repository_with_video(tmp_path)
    visual_index = SiglipVisualIndex(tmp_path / "missing-index", model_name="test")
    service = SearchService(
        repository,
        MemoryVectorIndex(),
        visual_search=visual_index,
    )

    assert service.search(
        "игрок поднимает руку",
        video_ids=["video-1"],
        mode="visual",
        use_lighthouse=False,
    ) == []
    with pytest.raises(SearchDependencyError, match="Visual index"):
        service.search_for_evaluation(
            "игрок поднимает руку",
            video_ids=["video-1"],
            mode="visual",
            use_lighthouse=False,
        )


def test_evaluation_requires_current_lighthouse_cache_but_interactive_is_best_effort(
    tmp_path,
) -> None:
    repository = _repository_with_video(tmp_path)
    moment_search = StaleMomentSearch([])
    service = SearchService(
        repository,
        MemoryVectorIndex(),
        visual_search=RecordingVisualIndex([]),
        moment_search=moment_search,
    )

    assert service.search(
        "игрок поднимает руку",
        video_ids=["video-1"],
        mode="visual",
        use_lighthouse=True,
    ) == []
    with pytest.raises(SearchDependencyError, match="Temporal cache"):
        service.search_for_evaluation(
            "игрок поднимает руку",
            video_ids=["video-1"],
            mode="visual",
            use_lighthouse=True,
        )


def test_speech_mode_does_not_request_ocr_or_visual_hits(tmp_path) -> None:
    repository = _repository_with_video(tmp_path)
    weak = EvidenceHit("video-1", "weak", 1, 3, "speech", 0.2, "unrelated")
    text_index = RecordingIndex([weak])
    visual_index = RecordingVisualIndex([])
    service = SearchService(repository, text_index, visual_search=visual_index)

    results = service.search("Мозгов", mode="speech")

    assert results == []
    assert text_index.modalities == {"speech"}
    assert visual_index.called is False


def test_visual_mode_uses_siglip_and_exposes_its_evidence(tmp_path) -> None:
    repository = _repository_with_video(tmp_path)
    visual = EvidenceHit(
        "video-1",
        "visual:scene-1",
        12,
        18,
        "visual",
        0.91,
        "поднимает руку",
        {"source": "siglip2"},
    )
    visual_index = RecordingVisualIndex([visual])
    service = SearchService(repository, RecordingIndex([]), visual_search=visual_index)

    results = service.search("поднимает руку", mode="visual")

    assert visual_index.called is True
    assert results[0].modalities == ["visual"]
    assert results[0].evidence[0].source == "siglip2"


def test_search_evidence_exposes_only_safe_structured_details(tmp_path) -> None:
    repository = _repository_with_video(tmp_path)
    metadata = {
        "source": "qwen-video-verifier",
        "event_type": "made_three_point",
        "matches_query": True,
        "shot_attempt": True,
        "ball_through_hoop": True,
        "shooter_outside_arc": True,
        "three_point_signal": False,
        "possible_shooter_jersey": "15",
        "shooter_jersey_confirmed": False,
        "model_evidence": "мяч проходит через кольцо",
        "stage_order": ["release", "outcome", "followup"],
        "raw_release_score": 0.79,
        "release_score": 0.81,
        "outcome_score": 0.92,
        "followup_score": 0.74,
        "core_score": 0.86,
        "plus_score": 0.55,
        "reset_score": 0.44,
        "miss_score": 0.12,
        "transition_score": 0.31,
        "contrast_score": 0.63,
        "thumbnail_path": "/private/video/frame.jpg",
        "media_path": "/private/video/source.mp4",
        "api_key": "secret-value",
        "prompt_version": "sports-v3",
    }
    visual = EvidenceHit(
        "video-1",
        "visual:event-1",
        12,
        18,
        "visual",
        0.91,
        "игрок забивает трёхочковый",
        metadata,
    )
    service = SearchService(
        repository,
        RecordingIndex([]),
        visual_search=RecordingVisualIndex([visual]),
    )

    evidence = service.search("игрок забивает трёхочковый", mode="visual")[0].evidence[0]

    assert evidence.details == {
        "event_type": "made_three_point",
        "matches_query": True,
        "shot_attempt": True,
        "ball_through_hoop": True,
        "shooter_outside_arc": True,
        "three_point_signal": False,
        "possible_shooter_jersey": "15",
        "shooter_jersey_confirmed": False,
        "model_evidence": "мяч проходит через кольцо",
        "stage_order": ["release", "outcome", "followup"],
        "raw_release_score": 0.79,
        "release_score": 0.81,
        "outcome_score": 0.92,
        "followup_score": 0.74,
        "core_score": 0.86,
        "plus_score": 0.55,
        "reset_score": 0.44,
        "miss_score": 0.12,
        "transition_score": 0.31,
        "contrast_score": 0.63,
    }
    assert "thumbnail_path" not in evidence.details
    assert "media_path" not in evidence.details
    assert "api_key" not in evidence.details
    assert "prompt_version" not in evidence.details


def test_named_entity_query_does_not_fall_through_to_random_visual_results(tmp_path) -> None:
    repository = _repository_with_video(tmp_path)
    visual = EvidenceHit("video-1", "visual:1", 4, 8, "visual", 0.99, "Мозгов", {"source": "siglip2"})
    visual_index = RecordingVisualIndex([visual])
    service = SearchService(repository, RecordingIndex([]), visual_search=visual_index)

    results = service.search("Мозгов", mode="all")

    assert results == []
    assert visual_index.called is False


def test_hybrid_rerank_rewards_related_word_roots_and_penalizes_noise() -> None:
    related = EvidenceHit("video-1", "related", 1, 4, "speech", 0.5, "он попал треху")
    unrelated = EvidenceHit("video-1", "noise", 1, 4, "speech", 0.7, "название команды")

    related_score = _hybridize("трехочковый бросок", related).score
    unrelated_score = _hybridize("трехочковый бросок", unrelated).score

    assert related_score > unrelated_score


def test_action_query_uses_plan_and_drops_uncorroborated_lighthouse_hit(tmp_path) -> None:
    repository = _repository_with_video(tmp_path)
    visual = EvidenceHit(
        "video-1",
        "visual:scene-1",
        12,
        18,
        "visual",
        0.86,
        "игрок поднимает руку",
        {"source": "siglip2"},
    )
    lighthouse = RecordingMomentSearch([
        EvidenceHit("video-1", "lh-good", 13, 17, "lighthouse", 0.91, "query"),
        EvidenceHit("video-1", "lh-noise", 50, 65, "lighthouse", 0.99, "query"),
    ])
    service = SearchService(
        repository,
        RecordingIndex([]),
        moment_search=lighthouse,
        visual_search=RecordingVisualIndex([visual]),
    )

    results = service.search("игрок поднимает руку", use_lighthouse=True)

    assert lighthouse.called is True
    assert results[0].intent == "action"
    assert results[0].start == 12
    assert all(item.start < 20 for item in results)
    assert any(evidence.source == "lighthouse-corroborated" for evidence in results[0].evidence)


def test_lighthouse_cannot_create_standalone_results() -> None:
    lighthouse = [EvidenceHit("video-1", "lh", 0, 90, "lighthouse", 0.99, "query")]

    assert corroborate_lighthouse_hits(lighthouse, []) == []


def test_name_search_matches_russian_case_ending(tmp_path) -> None:
    repository = _repository_with_video(tmp_path)
    specification = _stage_specification(StageKind.SPEECH)
    _publish_segments(
        repository,
        specification,
        [
            SegmentRecord(
                id="speech-name",
                video_id="video-1",
                start=100,
                end=106,
                modality="speech",
                text="Сегодня Мозгова заменили",
                confidence=0.88,
                metadata={},
                thumbnail_path=None,
            )
        ],
    )
    service = SearchService(
        repository,
        RecordingIndex([]),
        segment_specifications=[specification],
    )

    results = service.search("Мозгов")

    assert results[0].start == 100
    assert results[0].intent == "entity"
    assert results[0].evidence[0].source == "lexical-stem"


def test_name_search_rejects_unconfirmed_phonetic_collision(tmp_path) -> None:
    repository = _repository_with_video(tmp_path)
    specification = _stage_specification(StageKind.SPEECH)
    _publish_segments(
        repository,
        specification,
        [
            SegmentRecord(
                id="speech-moscow",
                video_id="video-1",
                start=100,
                end=106,
                modality="speech",
                text="команда Москов снова атакует",
                confidence=0.88,
                metadata={},
                thumbnail_path=None,
            )
        ],
    )
    service = SearchService(
        repository,
        RecordingIndex([]),
        segment_specifications=[specification],
    )

    assert service.search("Мозгов") == []


def test_word_alignment_tightens_speech_result() -> None:
    hit = EvidenceHit(
        "video-1",
        "speech-1",
        10,
        20,
        "speech",
        0.9,
        "Сегодня Мозгова заменили",
        {
            "source": "lexical-stem",
            "words": [
                {"word": "Сегодня", "start": 10.2, "end": 11.0},
                {"word": "Мозгова", "start": 13.0, "end": 14.0},
                {"word": "заменили", "start": 18.0, "end": 19.0},
            ],
        },
    )

    refined = refine_speech_hit("Мозгов", hit, context=1.0)

    assert refined.start == 12.0
    assert refined.end == 15.0
    assert refined.metadata["word_alignment"] is True
    assert refined.metadata["temporal_refinement"] is True


def test_word_alignment_drops_malformed_persisted_word_metadata() -> None:
    hit = EvidenceHit(
        "video-1",
        "speech-1",
        10,
        20,
        "speech",
        0.9,
        "Сегодня Мозгова заменили",
        {
            "words": [
                {"word": "сломано", "start": "bad", "end": 12.0},
                {"word": "бесконечно", "start": 12.0, "end": float("inf")},
                {"word": "Мозгова", "start": 13.0, "end": 14.0},
            ]
        },
    )

    refined = refine_speech_hit("Мозгов", hit, context=1.0)

    assert refined.start == 12.0
    assert refined.end == 15.0
    assert refined.metadata["aligned_words"] == ["Мозгова"]


def test_evaluation_uses_strict_temporal_refiner(tmp_path) -> None:
    class StrictAwareRefiner:
        @staticmethod
        def refine(_query: str, hits: list[EvidenceHit]) -> list[EvidenceHit]:
            return hits

        @staticmethod
        def refine_strict(_query: str, _hits: list[EvidenceHit]) -> list[EvidenceHit]:
            raise RuntimeError("dense refinement failed")

    repository = _repository_with_video(tmp_path)
    objects_specification = _stage_specification(StageKind.OBJECTS)
    _publish_segments(repository, objects_specification, [])
    visual = EvidenceHit("video-1", "visual:1", 4, 8, "visual", 0.9, "query")
    service = SearchService(
        repository,
        MemoryVectorIndex(),
        visual_search=RecordingVisualIndex([visual]),
        temporal_refiner=StrictAwareRefiner(),
        segment_specifications=[objects_specification],
    )

    assert service.search("игрок поднимает руку", mode="visual", use_lighthouse=False)
    with pytest.raises(SearchDependencyError, match="Visual"):
        service.search_for_evaluation(
            "игрок поднимает руку",
            mode="visual",
            use_lighthouse=False,
        )


def test_evaluation_uses_strict_candidate_reranker(tmp_path) -> None:
    class StrictAwareReranker:
        @staticmethod
        def rerank(_query: str, candidates):  # type: ignore[no-untyped-def]
            return candidates

        @staticmethod
        def rerank_strict(_query: str, _candidates):  # type: ignore[no-untyped-def]
            raise RuntimeError("candidate verification failed")

    repository = _repository_with_video(tmp_path)
    objects_specification = _stage_specification(StageKind.OBJECTS)
    _publish_segments(repository, objects_specification, [])
    speech_specification = _stage_specification(StageKind.SPEECH)
    ocr_specification = _stage_specification(StageKind.OCR)
    _publish_segments(repository, speech_specification, [])
    _publish_segments(repository, ocr_specification, [])
    visual = EvidenceHit("video-1", "visual:1", 4, 8, "visual", 0.9, "query")
    service = SearchService(
        repository,
        MemoryVectorIndex(),
        visual_search=RecordingVisualIndex([visual]),
        candidate_reranker=StrictAwareReranker(),
        segment_specifications=[
            speech_specification,
            ocr_specification,
            objects_specification,
        ],
    )

    assert service.search("игрок поднимает руку", use_lighthouse=False)
    with pytest.raises(SearchDependencyError, match="Candidate"):
        service.search_for_evaluation(
            "игрок поднимает руку",
            use_lighthouse=False,
        )


def test_generation_aware_search_uses_only_current_verified_vector_binding(tmp_path) -> None:
    repository = _repository_with_video(tmp_path)
    specifications = _indexing_specifications()
    speech = SegmentRecord(
        id="speech-current",
        video_id="video-1",
        start=3,
        end=6,
        modality="speech",
        text="player made winning basket",
        confidence=0.9,
        metadata={},
        thumbnail_path=None,
    )
    _publish_segments(repository, specifications.speech, [speech])
    _publish_segments(repository, specifications.ocr, [])
    _publish_segments(repository, specifications.objects, [])

    class RecordingGenerationIndex(MemoryVectorIndex):
        searched_generations: tuple[str, ...] = ()

        def search_generations(
            self,
            query,
            *,
            bindings,
            modalities=None,
            limit=50,
            exhaustive_validation=False,
        ):  # type: ignore[no-untyped-def]
            self.searched_generations = tuple(item.generation_id for item in bindings)
            return super().search_generations(
                query,
                bindings=bindings,
                modalities=modalities,
                limit=limit,
                exhaustive_validation=exhaustive_validation,
            )

    index = RecordingGenerationIndex()
    generation_id = _publish_text_vectors(repository, index, specifications)
    service = SearchService(
        repository,
        index,
        specification_resolver=lambda: specifications,
    )

    results = service.search("winning basket", mode="speech", use_lighthouse=False)

    assert results and results[0].evidence[0].text == speech.text
    assert index.searched_generations == (generation_id,)


def test_strict_search_requires_verified_text_vector_generation_but_interactive_falls_back(
    tmp_path,
) -> None:
    repository = _repository_with_video(tmp_path)
    specifications = _indexing_specifications()
    speech = SegmentRecord(
        id="speech-current",
        video_id="video-1",
        start=3,
        end=6,
        modality="speech",
        text="player made winning basket",
        confidence=0.9,
        metadata={},
        thumbnail_path=None,
    )
    _publish_segments(repository, specifications.speech, [speech])
    service = SearchService(
        repository,
        MemoryVectorIndex(),
        specification_resolver=lambda: specifications,
    )

    assert service.search("winning basket", mode="speech", use_lighthouse=False)
    with pytest.raises(SearchDependencyError, match="Semantic index"):
        service.search_for_evaluation(
            "winning basket",
            mode="speech",
            use_lighthouse=False,
        )


def test_strict_search_accepts_verified_empty_text_vector_generation(tmp_path) -> None:
    repository = _repository_with_video(tmp_path)
    specifications = _indexing_specifications()
    for specification in specifications.semantic_segment_specifications:
        _publish_segments(repository, specification, [])
    index = MemoryVectorIndex()
    _publish_text_vectors(repository, index, specifications)
    service = SearchService(
        repository,
        index,
        specification_resolver=lambda: specifications,
    )

    assert service.search_for_evaluation(
        "missing phrase",
        mode="speech",
        use_lighthouse=False,
    ) == []


def test_missing_physical_generation_is_lexical_fallback_interactively_and_error_strictly(
    tmp_path,
) -> None:
    repository = _repository_with_video(tmp_path)
    specifications = _indexing_specifications()
    speech = SegmentRecord(
        id="speech-current",
        video_id="video-1",
        start=3,
        end=6,
        modality="speech",
        text="player made winning basket",
        confidence=0.9,
        metadata={},
        thumbnail_path=None,
    )
    _publish_segments(repository, specifications.speech, [speech])
    _publish_segments(repository, specifications.ocr, [])
    _publish_segments(repository, specifications.objects, [])
    index = MemoryVectorIndex()
    generation_id = _publish_text_vectors(repository, index, specifications)
    index.delete_generation(
        generation_id,
        index_specification_hash=index.index_specification.specification_hash,
    )
    service = SearchService(
        repository,
        index,
        specification_resolver=lambda: specifications,
    )

    assert service.search("winning basket", mode="speech", use_lighthouse=False)
    with pytest.raises(SearchDependencyError, match="Semantic text search"):
        service.search_for_evaluation(
            "winning basket",
            mode="speech",
            use_lighthouse=False,
        )


def test_search_scopes_every_provider_and_output_to_ready_videos(tmp_path) -> None:
    repository = _repository_with_video(tmp_path)
    repository.create_video(
        video_id="failed-video",
        original_name="failed.mp4",
        stored_name="failed.mp4",
        media_path=str(tmp_path / "failed.mp4"),
        size_bytes=10,
    )
    repository.update_video("failed-video", status="failed")
    ready_hit = EvidenceHit("video-1", "ready", 4, 8, "visual", 0.7, "ready")
    stale_hit = EvidenceHit("failed-video", "stale", 4, 8, "visual", 0.99, "stale")
    text_index = RecordingIndex([stale_hit])
    visual_index = RecordingVisualIndex([stale_hit, ready_hit])
    moment_search = RecordingMomentSearch([stale_hit])
    service = SearchService(
        repository,
        text_index,
        visual_search=visual_index,
        moment_search=moment_search,
    )

    results = service.search(
        "игрок поднимает руку",
        video_ids=["video-1", "failed-video"],
        use_lighthouse=True,
    )

    assert text_index.video_ids == ["video-1"]
    assert visual_index.video_ids == ["video-1"]
    assert moment_search.video_ids == ["video-1"]
    assert {result.video_id for result in results} == {"video-1"}


def _pinned_asset_binding() -> SearchAssetBinding:
    return SearchAssetBinding(
        external_id="portable-asset",
        video_id="video-1",
        source_sha256=_PINNED_SOURCE_SHA256,
        byte_size=10,
        duration_seconds=12.0,
    )


def _text_evaluation_configuration() -> EvaluationSearchConfiguration:
    return EvaluationSearchConfiguration(
        modalities=("objects", "ocr", "speech"),
        modality_weights=(("objects", 0.92), ("ocr", 0.88), ("speech", 1.0)),
        text_search="lexical_and_semantic",
        visual_search="disabled",
        temporal_refinement=False,
        lighthouse=False,
        reranker="none",
        reranker_trigger="disabled",
        reranker_candidate_limit=0,
        result_limit=20,
    )


class _ForbiddenQueryRouter:
    @staticmethod
    def route(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("the pinned evaluation boundary must not call QueryRouter")


class _ForbiddenVisualProvider:
    def __getattribute__(self, name: str):  # type: ignore[no-untyped-def]
        if name.startswith("__"):
            return object.__getattribute__(self, name)
        raise AssertionError("the text-only plan must not inspect or call visual search")


def _ready_generation_search_service(tmp_path):  # type: ignore[no-untyped-def]
    repository = _repository_with_video(tmp_path)
    repository.update_video("video-1", duration=12.0)
    specifications = _indexing_specifications()
    speech = SegmentRecord(
        id="speech-current",
        video_id="video-1",
        start=3,
        end=6,
        modality="speech",
        text="player made winning basket",
        confidence=0.9,
        metadata={},
        thumbnail_path=None,
    )
    _publish_segments(repository, specifications.speech, [speech])
    _publish_segments(repository, specifications.ocr, [])
    _publish_segments(repository, specifications.objects, [])
    index = MemoryVectorIndex()
    _attest_memory_index(index)
    _publish_text_vectors(repository, index, specifications)
    service = SearchService(
        repository,
        index,
        visual_search=_ForbiddenVisualProvider(),  # type: ignore[arg-type]
        query_router=_ForbiddenQueryRouter(),  # type: ignore[arg-type]
        specification_resolver=lambda: specifications,
        media_root=tmp_path,
    )
    return repository, specifications, index, service


def test_pinned_evaluation_bypasses_router_and_uses_exact_text_generations(
    tmp_path,
) -> None:
    repository, specifications, index, service = _ready_generation_search_service(
        tmp_path
    )
    del repository, specifications
    searched: list[tuple[tuple[str, ...], set[str] | None, bool]] = []
    original_search = index.search_generations

    def recording_search(
        query,
        *,
        bindings,
        modalities=None,
        limit=50,
        exhaustive_validation=False,
    ):  # type: ignore[no-untyped-def]
        searched.append(
            (
                tuple(binding.generation_id for binding in bindings),
                modalities,
                exhaustive_validation,
            )
        )
        return original_search(
            query,
            bindings=bindings,
            modalities=modalities,
            limit=limit,
            exhaustive_validation=exhaustive_validation,
        )

    index.search_generations = recording_search  # type: ignore[method-assign]
    session = service.open_pinned_evaluation(
        _text_evaluation_configuration(),
        (_pinned_asset_binding(),),
    )

    results = session.search("winning basket", ("video-1",), limit=20)

    assert results[0].video_id == "video-1"
    assert len(searched) == 1
    generation_ids, modalities, exhaustive = searched[0]
    assert len(generation_ids) == 1
    assert modalities == {"objects", "ocr", "speech"}
    assert exhaustive is False
    assert session.last_search_execution_trace() == ProductSearchExecutionTrace(
        schema_version=1,
        configuration_identity=_text_evaluation_configuration().identity,
        invoked_component_ids=("text_vectors", "lexical_text"),
        component_input_counts=(
            ("text_vectors", 1),
            ("lexical_text", 1),
            ("visual_dense", 0),
            ("temporal_refinement", 0),
            ("lighthouse", 0),
            ("qwen_verification", 0),
        ),
        component_output_counts=(
            ("text_vectors", 1),
            ("lexical_text", 1),
            ("visual_dense", 0),
            ("temporal_refinement", 0),
            ("lighthouse", 0),
            ("qwen_verification", 0),
        ),
        component_evidence_counts=(
            ("text_vectors", 1),
            ("lexical_text", 1),
            ("visual_dense", 0),
            ("temporal_refinement", 0),
            ("lighthouse", 0),
            ("qwen_verification", 0),
        ),
    )


def test_failed_pinned_search_clears_previous_execution_trace(tmp_path) -> None:
    _repository, _specifications, index, service = (
        _ready_generation_search_service(tmp_path)
    )
    session = service.open_pinned_evaluation(
        _text_evaluation_configuration(),
        (_pinned_asset_binding(),),
    )
    assert session.search("winning basket", ("video-1",), limit=20)
    assert session.last_search_execution_trace().invoked_component_ids

    def fail_search(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("synthetic search failure")

    index.search_generations = fail_search  # type: ignore[method-assign]
    with pytest.raises(SearchDependencyError, match="Semantic text search failed"):
        session.search("winning basket", ("video-1",), limit=20)
    with pytest.raises(RuntimeError, match="execution trace is unavailable"):
        session.last_search_execution_trace()


def test_product_search_execution_trace_rejects_counts_above_signed_64_bit() -> None:
    component_ids = (
        "text_vectors",
        "lexical_text",
        "visual_dense",
        "temporal_refinement",
        "lighthouse",
        "qwen_verification",
    )
    zero_counts = tuple((component_id, 0) for component_id in component_ids)

    with pytest.raises(ValueError, match="input counts are invalid"):
        ProductSearchExecutionTrace(
            schema_version=1,
            configuration_identity=_text_evaluation_configuration().identity,
            invoked_component_ids=("text_vectors",),
            component_input_counts=(
                ("text_vectors", 1 << 63),
                *zero_counts[1:],
            ),
            component_output_counts=zero_counts,
            component_evidence_counts=zero_counts,
        )


def test_pinned_evaluation_invalidates_instead_of_following_new_active_pointer(
    tmp_path,
) -> None:
    repository, specifications, index, service = _ready_generation_search_service(
        tmp_path
    )
    session = service.open_pinned_evaluation(
        _text_evaluation_configuration(),
        (_pinned_asset_binding(),),
    )
    original_generation = repository.get_active_text_vector_generation("video-1")
    assert original_generation is not None
    replacement_generation = _publish_text_vectors(repository, index, specifications)
    assert replacement_generation != original_generation.generation_id

    assert session.capability_state("video-1", "text_vectors") == "stale"
    with pytest.raises(SearchDependencyError, match="changed"):
        session.search("winning basket", ("video-1",), limit=20)


class _PinnedVisualProvider:
    id = "siglip2"
    model_identity = "siglip-model@revision"

    def __init__(self) -> None:
        self.generation_id = "1" * 32
        self.descriptor_generation_id: str | None = None
        self.forced_descriptor_calls = 0
        self.search_calls: list[dict[str, str]] = []

    @property
    def specification_identity(self) -> str:
        return "2" * 64

    def status(self, *, check_index: bool = True) -> ProviderStatus:
        del check_index
        return ProviderStatus(self.id, self.id, ProviderState.READY, "ready")

    def active_generation_id(self, _video_id: str) -> str:
        return self.generation_id

    def generation_descriptor(
        self,
        video_id: str,
        generation_id: str,
        *,
        force_content_validation: bool = False,
    ) -> dict[str, object] | None:
        if force_content_validation:
            self.forced_descriptor_calls += 1
        if generation_id != self.generation_id:
            return None
        return {
            "content_sha256": "6" * 64,
            "duration_seconds": 12.0,
            "generation_id": self.descriptor_generation_id or generation_id,
            "source_sha256": _PINNED_SOURCE_SHA256,
            "source_size_bytes": 10,
            "specification_hash": self.specification_identity,
            "video_id": video_id,
        }

    def search_generations(
        self,
        query: str,
        *,
        generation_bindings: dict[str, dict[str, object]],
        limit: int,
    ) -> list[EvidenceHit]:
        del limit
        self.search_calls.append(
            {
                video_id: str(binding["generation_id"])
                for video_id, binding in generation_bindings.items()
            }
        )
        return [
            EvidenceHit(
                "video-1",
                "visual:pinned",
                4,
                8,
                "visual",
                0.9,
                query,
            )
        ]


def test_visual_only_trace_does_not_claim_text_execution(tmp_path) -> None:
    repository, specifications, index, _service = _ready_generation_search_service(
        tmp_path
    )
    visual = _PinnedVisualProvider()
    service = SearchService(
        repository,
        index,
        visual_search=visual,
        query_router=_ForbiddenQueryRouter(),  # type: ignore[arg-type]
        specification_resolver=lambda: specifications,
        media_root=tmp_path,
    )
    configuration = EvaluationSearchConfiguration(
        modalities=("visual",),
        modality_weights=(("visual", 1.0),),
        text_search="disabled",
        visual_search="dense_siglip",
        temporal_refinement=False,
        lighthouse=False,
        reranker="none",
        reranker_trigger="disabled",
        reranker_candidate_limit=0,
        result_limit=20,
    )
    session = service.open_pinned_evaluation(
        configuration,
        (_pinned_asset_binding(),),
    )

    assert session.search("winning basket", ("video-1",), limit=20)
    trace = session.last_search_execution_trace()

    assert trace.invoked_component_ids == ("visual_dense",)
    for component_id in ("text_vectors", "lexical_text"):
        assert trace.component_input_count(component_id) == 0
        assert trace.component_output_count(component_id) == 0
        assert trace.component_evidence_count(component_id) == 0


class _PinnedLighthouseProvider:
    id = "lighthouse"

    def __init__(self) -> None:
        self.generation_id = "3" * 32
        self.search_calls: list[dict[str, str]] = []
        self.forced_descriptor_calls = 0

    @property
    def identity(self) -> dict[str, object]:
        return {
            "model_identity": "lighthouse-model@revision",
            "runtime_identity": "lighthouse-runtime@revision",
            "specification_hash": "4" * 64,
        }

    def status(self) -> ProviderStatus:
        return ProviderStatus(self.id, self.id, ProviderState.READY, "ready")

    def active_generation_id(self, _video_id: str) -> str:
        return self.generation_id

    def generation_descriptor(
        self,
        video_id: str,
        generation_id: str,
        *,
        force_content_validation: bool = False,
    ) -> dict[str, object] | None:
        if force_content_validation:
            self.forced_descriptor_calls += 1
        if generation_id != self.generation_id:
            return None
        return {
            "duration_seconds": 12.0,
            "generation_id": generation_id,
            "manifest_sha256": "7" * 64,
            "source_sha256": _PINNED_SOURCE_SHA256,
            "source_size_bytes": 10,
            "specification_hash": "4" * 64,
            "video_id": video_id,
        }

    def search_generations(
        self,
        query: str,
        generation_bindings: dict[str, dict[str, object]],
        *,
        limit: int,
    ) -> list[EvidenceHit]:
        del limit
        self.search_calls.append(
            {
                video_id: str(binding["generation_id"])
                for video_id, binding in generation_bindings.items()
            }
        )
        return [
            EvidenceHit(
                "video-1",
                "lighthouse:pinned",
                4,
                8,
                "lighthouse",
                0.85,
                query,
            )
        ]


def _released_external_indexes(
    visual: _PinnedVisualProvider,
    lighthouse: _PinnedLighthouseProvider,
) -> ExternalIndexReleaseSnapshot:
    visual_descriptor = visual.generation_descriptor(
        "video-1",
        visual.generation_id,
    )
    lighthouse_descriptor = lighthouse.generation_descriptor(
        "video-1",
        lighthouse.generation_id,
    )
    assert visual_descriptor is not None
    assert lighthouse_descriptor is not None
    return ExternalIndexReleaseSnapshot(
        job_backed_video_ids=frozenset({"video-1"}),
        visual_dense={"video-1": visual_descriptor},
        lighthouse={"video-1": lighthouse_descriptor},
    )


def _external_only_evaluation_configuration() -> EvaluationSearchConfiguration:
    return EvaluationSearchConfiguration(
        modalities=("lighthouse", "visual"),
        modality_weights=(("lighthouse", 1.0), ("visual", 1.0)),
        text_search="disabled",
        visual_search="dense_siglip",
        temporal_refinement=True,
        lighthouse=True,
        reranker="none",
        reranker_trigger="disabled",
        reranker_candidate_limit=0,
        result_limit=20,
    )


def test_pinned_evaluation_uses_job_release_without_provider_active_pointers(
    tmp_path,
) -> None:
    repository, specifications, index, _service = _ready_generation_search_service(
        tmp_path
    )
    visual = _PinnedVisualProvider()
    lighthouse = _PinnedLighthouseProvider()
    release = _released_external_indexes(visual, lighthouse)
    repository.get_external_index_release_snapshot = (  # type: ignore[method-assign]
        lambda _video_ids: release
    )

    def reject_active_pointer(_video_id: str) -> str:
        raise AssertionError("job-owned releases must not use provider active pointers")

    visual.active_generation_id = reject_active_pointer  # type: ignore[method-assign]
    lighthouse.active_generation_id = reject_active_pointer  # type: ignore[method-assign]
    service = SearchService(
        repository,
        index,
        moment_search=lighthouse,
        visual_search=visual,
        temporal_refiner=_PinnedTemporalRefiner(),
        query_router=_ForbiddenQueryRouter(),  # type: ignore[arg-type]
        specification_resolver=lambda: specifications,
        media_root=tmp_path,
    )
    session = service.open_pinned_evaluation(
        _external_only_evaluation_configuration(),
        (_pinned_asset_binding(),),
    )

    assert session.capability_state("video-1", "visual_dense") == "complete"
    assert session.capability_state("video-1", "lighthouse") == "complete"
    assert session.search("winning basket", ("video-1",), limit=20)
    assert visual.search_calls == [{"video-1": visual.generation_id}]
    assert lighthouse.search_calls == [{"video-1": lighthouse.generation_id}]
    session.close()


def test_pinned_job_release_binding_drift_invalidates_session(tmp_path) -> None:
    repository, specifications, index, _service = _ready_generation_search_service(
        tmp_path
    )
    visual = _PinnedVisualProvider()
    lighthouse = _PinnedLighthouseProvider()
    release = [_released_external_indexes(visual, lighthouse)]
    repository.get_external_index_release_snapshot = (  # type: ignore[method-assign]
        lambda _video_ids: release[0]
    )
    service = SearchService(
        repository,
        index,
        moment_search=lighthouse,
        visual_search=visual,
        temporal_refiner=_PinnedTemporalRefiner(),
        specification_resolver=lambda: specifications,
        media_root=tmp_path,
    )
    session = service.open_pinned_evaluation(
        _external_only_evaluation_configuration(),
        (_pinned_asset_binding(),),
    )
    release[0] = ExternalIndexReleaseSnapshot(
        job_backed_video_ids=frozenset({"video-1"}),
        visual_dense={},
        lighthouse=release[0].lighthouse,
    )

    assert session.capability_state("video-1", "visual_dense") == "stale"
    with pytest.raises(SearchDependencyError, match="changed"):
        session.search("winning basket", ("video-1",), limit=20)
    with pytest.raises(SearchDependencyError, match="final identity"):
        session.close()


def test_pinned_job_release_missing_does_not_fall_back_to_active_pointer(
    tmp_path,
) -> None:
    repository, specifications, index, _service = _ready_generation_search_service(
        tmp_path
    )
    visual = _PinnedVisualProvider()
    lighthouse = _PinnedLighthouseProvider()
    release = _released_external_indexes(visual, lighthouse)
    repository.get_external_index_release_snapshot = (  # type: ignore[method-assign]
        lambda _video_ids: ExternalIndexReleaseSnapshot(
            job_backed_video_ids=release.job_backed_video_ids,
            visual_dense={},
            lighthouse=release.lighthouse,
        )
    )

    def reject_active_pointer(_video_id: str) -> str:
        raise AssertionError("job-owned releases must not use provider active pointers")

    visual.active_generation_id = reject_active_pointer  # type: ignore[method-assign]
    lighthouse.active_generation_id = reject_active_pointer  # type: ignore[method-assign]
    service = SearchService(
        repository,
        index,
        moment_search=lighthouse,
        visual_search=visual,
        temporal_refiner=_PinnedTemporalRefiner(),
        specification_resolver=lambda: specifications,
        media_root=tmp_path,
    )
    session = service.open_pinned_evaluation(
        _external_only_evaluation_configuration(),
        (_pinned_asset_binding(),),
    )

    assert session.capability_state("video-1", "visual_dense") == "missing"
    with pytest.raises(
        SearchDependencyError,
        match="required pinned capability visual_dense is unavailable",
    ):
        session.search("winning basket", ("video-1",), limit=20)
    assert visual.search_calls == []
    assert lighthouse.search_calls == []
    session.close()


def test_pinned_job_release_drift_during_search_discards_result_and_trace(
    tmp_path,
) -> None:
    repository, specifications, index, _service = _ready_generation_search_service(
        tmp_path
    )
    visual = _PinnedVisualProvider()
    lighthouse = _PinnedLighthouseProvider()
    release = [_released_external_indexes(visual, lighthouse)]
    repository.get_external_index_release_snapshot = (  # type: ignore[method-assign]
        lambda _video_ids: release[0]
    )
    original_search = visual.search_generations

    def search_then_change_release(
        query: str,
        *,
        generation_bindings: dict[str, dict[str, object]],
        limit: int,
    ) -> list[EvidenceHit]:
        hits = original_search(
            query,
            generation_bindings=generation_bindings,
            limit=limit,
        )
        changed_visual = dict(release[0].visual_dense["video-1"])
        changed_visual["content_sha256"] = "8" * 64
        release[0] = ExternalIndexReleaseSnapshot(
            job_backed_video_ids=release[0].job_backed_video_ids,
            visual_dense={"video-1": changed_visual},
            lighthouse=release[0].lighthouse,
        )
        return hits

    visual.search_generations = search_then_change_release  # type: ignore[method-assign]
    service = SearchService(
        repository,
        index,
        moment_search=lighthouse,
        visual_search=visual,
        temporal_refiner=_PinnedTemporalRefiner(),
        specification_resolver=lambda: specifications,
        media_root=tmp_path,
    )
    session = service.open_pinned_evaluation(
        _external_only_evaluation_configuration(),
        (_pinned_asset_binding(),),
    )

    with pytest.raises(SearchDependencyError, match="changed"):
        session.search("winning basket", ("video-1",), limit=20)
    with pytest.raises(RuntimeError, match="trace is unavailable"):
        session.last_search_execution_trace()
    with pytest.raises(SearchDependencyError, match="final identity"):
        session.close()


class _PinnedTemporalRefiner:
    identity = {"implementation": "pinned-refiner-v1"}

    def __init__(self, *, mark_outputs: bool = False) -> None:
        self.called = False
        self.mark_outputs = mark_outputs

    @staticmethod
    def benchmark_attestation() -> dict[str, object]:
        return {
            "ffmpeg_identity": "ffmpeg-test-binary@sha256:" + "9" * 64,
            "implementation_identity": "pinned-refiner-v1",
            "runtime_identity": "temporal-test-runtime@1",
            "scorer_identity": "siglip-test-scorer@revision",
            "scratch_policy_identity": "isolated-session-scratch@1",
            "source_bound": True,
            "strict_complete": True,
        }

    def refine_strict(
        self,
        _query: str,
        hits: list[EvidenceHit],
    ) -> list[EvidenceHit]:
        self.called = True
        if not self.mark_outputs:
            return hits
        return [
            replace(
                hit,
                metadata={**hit.metadata, "temporal_refinement": True},
            )
            for hit in hits
        ]


class _PinnedReranker:
    def __init__(
        self,
        provider_id: str,
        top_candidates: int,
        *,
        add_qwen_evidence: bool = False,
    ) -> None:
        self.id = provider_id
        self.top_candidates = top_candidates
        self.add_qwen_evidence = add_qwen_evidence
        self.called = False
        self.last_candidate_count = 0

    @property
    def identity(self) -> dict[str, object]:
        return {"provider": self.id, "model": f"{self.id}-model@revision"}

    def status(self) -> ProviderStatus:
        return ProviderStatus(self.id, self.id, ProviderState.READY, "ready")

    def benchmark_attestation(self) -> dict[str, object]:
        return {
            "candidate_limit": self.top_candidates,
            "model_identity": f"{self.id}-model@revision",
            "protocol_identity": f"{self.id}-protocol@1",
            "provider": self.id,
            "runtime_identity": f"{self.id}-runtime@1",
            "source_bound": True,
            "strict_complete": True,
        }

    def rerank_strict(self, _query: str, candidates):  # type: ignore[no-untyped-def]
        self.called = True
        self.last_candidate_count = len(candidates)
        reranked = list(reversed(candidates))
        if not self.add_qwen_evidence:
            return reranked
        return [
            replace(
                candidate,
                modalities=sorted({*candidate.modalities, "qwen_video"}),
                evidence=[
                    EvidenceHit(
                        video_id=candidate.video_id,
                        segment_id=f"qwen:{index}",
                        start=candidate.start,
                        end=candidate.end,
                        modality="qwen_video",
                        score=0.95,
                        text="verified",
                        metadata={
                            "model_evidence": "observed judgement",
                            "source": "qwen-video-verifier",
                        },
                    ),
                    *candidate.evidence,
                ],
            )
            for index, candidate in enumerate(reranked)
        ]


def _full_evaluation_configuration(
    reranker: str,
) -> EvaluationSearchConfiguration:
    candidate_limit = {"qwen": 12, "internvideo": 4}[reranker]
    return EvaluationSearchConfiguration(
        modalities=("lighthouse", "objects", "ocr", "speech", "visual"),
        modality_weights=(
            ("lighthouse", 0.78),
            ("objects", 0.92),
            ("ocr", 0.88),
            ("speech", 1.0),
            ("visual", 1.08),
        ),
        text_search="lexical_and_semantic",
        visual_search="dense_siglip",
        temporal_refinement=True,
        lighthouse=True,
        reranker=reranker,
        reranker_trigger="all_candidates",
        reranker_candidate_limit=candidate_limit,
        result_limit=20,
    )


def test_pinned_evaluation_uses_exact_weights_and_only_selected_reranker(
    tmp_path,
    monkeypatch,
) -> None:
    repository, specifications, index, _service = _ready_generation_search_service(
        tmp_path
    )
    visual = _PinnedVisualProvider()
    lighthouse = _PinnedLighthouseProvider()
    refiner = _PinnedTemporalRefiner()
    qwen = _PinnedReranker("qwen-video", 12)
    internvideo = _PinnedReranker("internvideo", 4)
    service = SearchService(
        repository,
        index,
        moment_search=lighthouse,
        visual_search=visual,
        query_router=_ForbiddenQueryRouter(),  # type: ignore[arg-type]
        temporal_refiner=refiner,
        evaluation_rerankers={"qwen": qwen, "internvideo": internvideo},
        specification_resolver=lambda: specifications,
        media_root=tmp_path,
    )
    import videoscope.search.service as service_module

    weights: list[dict[str, float] | None] = []
    original_fuse = service_module.fuse_hits

    def recording_fuse(hits, **kwargs):  # type: ignore[no-untyped-def]
        weights.append(kwargs.get("modality_weights"))
        original_fuse(hits, **kwargs)
        return [
            FusedResult(
                video_id="video-1",
                start=float(index * 2),
                end=float(index * 2 + 1),
                score=1.0 - index / 10,
                modalities=["visual"],
                evidence=[
                    EvidenceHit(
                        video_id="video-1",
                        segment_id=f"candidate-{index}",
                        start=float(index * 2),
                        end=float(index * 2 + 1),
                        modality="visual",
                        score=1.0 - index / 10,
                        text="candidate",
                    )
                ],
            )
            for index in range(6)
        ]

    monkeypatch.setattr(service_module, "fuse_hits", recording_fuse)
    configuration = _full_evaluation_configuration("internvideo")
    session = service.open_pinned_evaluation(
        configuration,
        (_pinned_asset_binding(),),
    )

    results = session.search("Мозгов", ("video-1",), limit=20)

    assert results
    assert visual.search_calls == [{"video-1": "1" * 32}]
    assert lighthouse.search_calls == [{"video-1": "3" * 32}]
    assert refiner.called
    assert internvideo.called
    assert internvideo.last_candidate_count == 4
    assert not qwen.called
    assert weights == [dict(configuration.modality_weights)]
    assert [result.start for result in results] == [6.0, 4.0, 2.0, 0.0, 8.0, 10.0]

    session.close()

    assert visual.forced_descriptor_calls == 1
    assert lighthouse.forced_descriptor_calls == 1


def test_pinned_evaluation_traces_all_strict_stage_calls_before_final_view_loss(
    tmp_path,
    monkeypatch,
) -> None:
    repository, specifications, index, _service = _ready_generation_search_service(
        tmp_path
    )
    visual = _PinnedVisualProvider()
    lighthouse = _PinnedLighthouseProvider()
    refiner = _PinnedTemporalRefiner(mark_outputs=True)
    qwen = _PinnedReranker(
        "qwen-video",
        12,
        add_qwen_evidence=True,
    )
    service = SearchService(
        repository,
        index,
        moment_search=lighthouse,
        visual_search=visual,
        query_router=_ForbiddenQueryRouter(),  # type: ignore[arg-type]
        temporal_refiner=refiner,
        evaluation_rerankers={"qwen": qwen},
        specification_resolver=lambda: specifications,
        media_root=tmp_path,
    )
    configuration = _full_evaluation_configuration("qwen")
    session = service.open_pinned_evaluation(
        configuration,
        (_pinned_asset_binding(),),
    )

    assert session.search("winning basket", ("video-1",), limit=20)
    qwen_count = qwen.last_candidate_count
    assert qwen_count > 0
    assert session.last_search_execution_trace() == ProductSearchExecutionTrace(
        schema_version=1,
        configuration_identity=configuration.identity,
        invoked_component_ids=(
            "text_vectors",
            "lexical_text",
            "visual_dense",
            "temporal_refinement",
            "lighthouse",
            "qwen_verification",
        ),
        component_input_counts=(
            ("text_vectors", 1),
            ("lexical_text", 1),
            ("visual_dense", 1),
            ("temporal_refinement", 1),
            ("lighthouse", 1),
            ("qwen_verification", qwen_count),
        ),
        component_output_counts=(
            ("text_vectors", 1),
            ("lexical_text", 1),
            ("visual_dense", 1),
            ("temporal_refinement", 1),
            ("lighthouse", 1),
            ("qwen_verification", qwen_count),
        ),
        component_evidence_counts=(
            ("text_vectors", 1),
            ("lexical_text", 1),
            ("visual_dense", 1),
            ("temporal_refinement", 1),
            ("lighthouse", 1),
            ("qwen_verification", qwen_count),
        ),
    )

    import videoscope.search.service as service_module

    monkeypatch.setattr(service_module, "fuse_hits", lambda *_args, **_kwargs: [])
    assert session.search("winning basket", ("video-1",), limit=20) == []
    empty_reranker_trace = session.last_search_execution_trace()
    assert empty_reranker_trace.invoked_component_ids == (
        "text_vectors",
        "lexical_text",
        "visual_dense",
        "temporal_refinement",
        "lighthouse",
        "qwen_verification",
    )
    assert empty_reranker_trace.component_input_count("qwen_verification") == 0
    assert empty_reranker_trace.component_output_count("qwen_verification") == 0
    assert empty_reranker_trace.component_evidence_count("qwen_verification") == 0


def test_pinned_evaluation_marks_visual_pointer_drift_stale(tmp_path) -> None:
    repository, specifications, index, _service = _ready_generation_search_service(
        tmp_path
    )
    visual = _PinnedVisualProvider()
    service = SearchService(
        repository,
        index,
        visual_search=visual,
        temporal_refiner=_PinnedTemporalRefiner(),
        specification_resolver=lambda: specifications,
        media_root=tmp_path,
    )
    configuration = EvaluationSearchConfiguration(
        modalities=("objects", "ocr", "speech", "visual"),
        modality_weights=(
            ("objects", 0.92),
            ("ocr", 0.88),
            ("speech", 1.0),
            ("visual", 1.08),
        ),
        text_search="lexical_and_semantic",
        visual_search="dense_siglip",
        temporal_refinement=True,
        lighthouse=False,
        reranker="none",
        reranker_trigger="disabled",
        reranker_candidate_limit=0,
        result_limit=20,
    )
    session = service.open_pinned_evaluation(
        configuration,
        (_pinned_asset_binding(),),
    )
    visual.generation_id = "5" * 32

    assert session.capability_state("video-1", "visual_dense") == "stale"
    with pytest.raises(SearchDependencyError, match="changed"):
        session.search("query", ("video-1",), limit=20)


def test_pinned_evaluation_rejects_cross_generation_visual_descriptor(
    tmp_path,
) -> None:
    repository, specifications, index, _service = _ready_generation_search_service(
        tmp_path
    )
    visual = _PinnedVisualProvider()
    visual.descriptor_generation_id = "a" * 32
    service = SearchService(
        repository,
        index,
        visual_search=visual,
        specification_resolver=lambda: specifications,
        media_root=tmp_path,
    )
    configuration = EvaluationSearchConfiguration(
        modalities=("visual",),
        modality_weights=(("visual", 1.08),),
        text_search="disabled",
        visual_search="dense_siglip",
        temporal_refinement=False,
        lighthouse=False,
        reranker="none",
        reranker_trigger="disabled",
        reranker_candidate_limit=0,
        result_limit=20,
    )

    session = service.open_pinned_evaluation(
        configuration,
        (_pinned_asset_binding(),),
    )

    assert session.capability_state("video-1", "visual_dense") == "stale"
    with pytest.raises(SearchDependencyError, match="unavailable"):
        session.search("query", ("video-1",), limit=20)


def test_pinned_text_requires_attested_no_fallback_query_embedding(tmp_path) -> None:
    repository, _specifications, index, service = _ready_generation_search_service(
        tmp_path
    )
    del repository
    del index.benchmark_attestation  # type: ignore[attr-defined]

    session = service.open_pinned_evaluation(
        _text_evaluation_configuration(),
        (_pinned_asset_binding(),),
    )

    assert session.capability_state("video-1", "text_vectors") == "not_configured"
    with pytest.raises(SearchDependencyError, match="unavailable"):
        session.search("query", ("video-1",), limit=20)


def test_pinned_temporal_refinement_requires_runtime_and_scratch_attestation(
    tmp_path,
) -> None:
    repository, specifications, index, _service = _ready_generation_search_service(
        tmp_path
    )
    visual = _PinnedVisualProvider()
    refiner = _PinnedTemporalRefiner()
    refiner.benchmark_attestation = None  # type: ignore[method-assign]
    service = SearchService(
        repository,
        index,
        visual_search=visual,
        temporal_refiner=refiner,
        specification_resolver=lambda: specifications,
        media_root=tmp_path,
    )
    configuration = EvaluationSearchConfiguration(
        modalities=("objects", "ocr", "speech", "visual"),
        modality_weights=(
            ("objects", 0.92),
            ("ocr", 0.88),
            ("speech", 1.0),
            ("visual", 1.08),
        ),
        text_search="lexical_and_semantic",
        visual_search="dense_siglip",
        temporal_refinement=True,
        lighthouse=False,
        reranker="none",
        reranker_trigger="disabled",
        reranker_candidate_limit=0,
        result_limit=20,
    )

    session = service.open_pinned_evaluation(
        configuration,
        (_pinned_asset_binding(),),
    )

    assert (
        session.capability_state("video-1", "temporal_refinement")
        == "not_configured"
    )
    with pytest.raises(SearchDependencyError, match="unavailable"):
        session.search("query", ("video-1",), limit=20)


def test_dense_only_identity_excludes_unselected_text_components(tmp_path) -> None:
    repository, specifications, index, _service = _ready_generation_search_service(
        tmp_path
    )
    visual = _PinnedVisualProvider()
    service = SearchService(
        repository,
        index,
        visual_search=visual,
        specification_resolver=lambda: specifications,
        media_root=tmp_path,
    )
    configuration = EvaluationSearchConfiguration(
        modalities=("visual",),
        modality_weights=(("visual", 1.08),),
        text_search="disabled",
        visual_search="dense_siglip",
        temporal_refinement=False,
        lighthouse=False,
        reranker="none",
        reranker_trigger="disabled",
        reranker_candidate_limit=0,
        result_limit=20,
    )

    session = service.open_pinned_evaluation(
        configuration,
        (_pinned_asset_binding(),),
    )
    identities = session.identities()

    assert {item.component_id for item in identities.model} == {"visual_embedding"}
    assert {item.component_id for item in identities.index} == {
        "visual_generations"
    }
    index.index_specification = replace(  # type: ignore[misc]
        index.index_specification,
        embedding_identity="unrelated-text-model@2",
    )
    assert session.capability_state("video-1", "visual_dense") == "complete"


def test_pinned_evaluation_rehashes_source_for_every_profile(tmp_path) -> None:
    _repository, _specifications, _index, service = _ready_generation_search_service(
        tmp_path
    )
    session = service.open_pinned_evaluation(
        _text_evaluation_configuration(),
        (_pinned_asset_binding(),),
    )
    (tmp_path / "video-1.mp4").write_bytes(b"9876543210")

    assert session.capability_state("video-1", "text_vectors") == "stale"
    with pytest.raises(SearchDependencyError, match="changed"):
        session.search("query", ("video-1",), limit=20)
    with pytest.raises(SearchDependencyError, match="final identity"):
        session.close()


def test_pinned_evaluation_hashes_media_only_at_open_and_close(tmp_path) -> None:
    repository, _specifications, _index, service = _ready_generation_search_service(
        tmp_path
    )
    native_verify = repository.verify_existing_asset_identity
    calls = 0

    def recording_verify(video_id, *, media_root):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return native_verify(video_id, media_root=media_root)

    repository.verify_existing_asset_identity = recording_verify  # type: ignore[method-assign]
    session = service.open_pinned_evaluation(
        _text_evaluation_configuration(),
        (_pinned_asset_binding(),),
    )

    assert calls == 1
    assert session.capability_state("video-1", "text_vectors") == "complete"
    assert session.capability_state("video-1", "text_vectors") == "complete"
    assert session.search("winning basket", ("video-1",), limit=20)
    assert session.search("winning basket", ("video-1",), limit=20)
    assert calls == 1

    session.close()

    assert calls == 2


@pytest.mark.parametrize("final_snapshot_is_current", [True, False])
def test_pinned_evaluation_releases_text_attestation_after_final_verification(
    tmp_path,
    final_snapshot_is_current,
) -> None:  # type: ignore[no-untyped-def]
    _repository, _specifications, index, service = _ready_generation_search_service(
        tmp_path
    )
    released: list[tuple[object, ...]] = []

    def release(bindings):  # type: ignore[no-untyped-def]
        released.append(bindings)

    index.release_benchmark_snapshot_bindings = release  # type: ignore[attr-defined]
    session = service.open_pinned_evaluation(
        _text_evaluation_configuration(),
        (_pinned_asset_binding(),),
    )
    text_binding = session._states[0].text_binding
    assert text_binding is not None
    index.verify_benchmark_snapshot_current = (  # type: ignore[attr-defined]
        lambda: final_snapshot_is_current
    )

    if final_snapshot_is_current:
        session.close()
    else:
        with pytest.raises(SearchDependencyError, match="final identity"):
            session.close()

    assert released == [(text_binding,)]
    session.close()
    assert released == [(text_binding,)]


@pytest.mark.parametrize("release_fails", [False, True])
def test_pinned_evaluation_releases_text_attestation_when_construction_fails(
    tmp_path,
    monkeypatch,
    release_fails,
) -> None:  # type: ignore[no-untyped-def]
    _repository, _specifications, index, service = _ready_generation_search_service(
        tmp_path
    )
    validated: list[object] = []
    released: list[tuple[object, ...]] = []
    native_validate = index.validate_generation_for_benchmark_snapshot

    def validate(binding):  # type: ignore[no-untyped-def]
        validated.append(binding)
        return native_validate(binding)

    def release(bindings):  # type: ignore[no-untyped-def]
        released.append(bindings)
        if release_fails:
            raise RuntimeError("attestation release failed")

    def fail_identity_build(_session):  # type: ignore[no-untyped-def]
        raise RuntimeError("identity build failed")

    index.validate_generation_for_benchmark_snapshot = validate  # type: ignore[attr-defined]
    index.release_benchmark_snapshot_bindings = release  # type: ignore[attr-defined]
    monkeypatch.setattr(
        "videoscope.search.service.PinnedProductSearchSession._build_identities",
        fail_identity_build,
    )

    with pytest.raises(RuntimeError, match="identity build failed") as captured:
        service.open_pinned_evaluation(
            _text_evaluation_configuration(),
            (_pinned_asset_binding(),),
        )

    assert len(validated) == 1
    assert released == [(validated[0],)]
    if release_fails:
        assert any(
            "attestation release failed" in note
            for note in getattr(captured.value, "__notes__", ())
        )


def test_pinned_evaluation_requires_text_attestation_release_capability(tmp_path) -> None:
    _repository, _specifications, index, service = _ready_generation_search_service(
        tmp_path
    )
    del index.release_benchmark_snapshot_bindings  # type: ignore[attr-defined]

    session = service.open_pinned_evaluation(
        _text_evaluation_configuration(),
        (_pinned_asset_binding(),),
    )

    assert session.capability_state("video-1", "text_vectors") == "not_configured"
    session.close()


def test_pinned_search_latency_excludes_full_sqlite_generation_revalidation(
    tmp_path,
) -> None:
    repository, _specifications, _index, service = _ready_generation_search_service(
        tmp_path
    )
    native_text_binding = repository.get_text_vector_search_binding
    native_segment_snapshot = repository.get_segment_generation_snapshot
    calls = {"text_binding": 0, "segment_snapshot": 0}

    def recording_text_binding(generation_id):  # type: ignore[no-untyped-def]
        calls["text_binding"] += 1
        return native_text_binding(generation_id)

    def recording_segment_snapshot(generation_id):  # type: ignore[no-untyped-def]
        calls["segment_snapshot"] += 1
        return native_segment_snapshot(generation_id)

    repository.get_text_vector_search_binding = recording_text_binding  # type: ignore[method-assign]
    repository.get_segment_generation_snapshot = recording_segment_snapshot  # type: ignore[method-assign]
    session = service.open_pinned_evaluation(
        _text_evaluation_configuration(),
        (_pinned_asset_binding(),),
    )
    assert session.capability_state("video-1", "text_vectors") == "complete"
    calls.update(text_binding=0, segment_snapshot=0)

    assert session.search("winning basket", ("video-1",), limit=20)
    assert session.search("winning basket", ("video-1",), limit=20)

    assert calls == {"text_binding": 0, "segment_snapshot": 0}
    session.close()
    assert calls == {"text_binding": 0, "segment_snapshot": 0}


def test_pinned_text_preflight_uses_only_bounded_repository_materializers(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, _specifications, _index, service = _ready_generation_search_service(
        tmp_path
    )
    bounded_binding = repository.get_text_vector_search_binding_bounded
    bounded_segments = repository.get_segment_generation_snapshot_bounded
    calls = {"binding": 0, "segments": 0}

    def recording_binding(generation_id, **limits):  # type: ignore[no-untyped-def]
        calls["binding"] += 1
        return bounded_binding(generation_id, **limits)

    def recording_segments(generation_id, **limits):  # type: ignore[no-untyped-def]
        calls["segments"] += 1
        return bounded_segments(generation_id, **limits)

    monkeypatch.setattr(
        repository,
        "get_text_vector_search_binding_bounded",
        recording_binding,
    )
    monkeypatch.setattr(
        repository,
        "get_segment_generation_snapshot_bounded",
        recording_segments,
    )
    monkeypatch.setattr(
        repository,
        "get_text_vector_search_binding",
        lambda *_args: pytest.fail("benchmark preflight used an unbounded binding read"),
    )
    monkeypatch.setattr(
        repository,
        "get_segment_generation_snapshot",
        lambda *_args: pytest.fail("benchmark preflight used an unbounded segment read"),
    )

    session = service.open_pinned_evaluation(
        _text_evaluation_configuration(),
        (_pinned_asset_binding(),),
    )

    assert session.capability_state("video-1", "text_vectors") == "complete"
    assert calls["binding"] == 2
    assert calls["segments"] >= 2
    session.close()
    assert calls == {"binding": 3, "segments": 9}


def test_pinned_evaluation_requires_canonical_media_root(tmp_path) -> None:
    repository, specifications, index, _service = _ready_generation_search_service(
        tmp_path
    )
    service = SearchService(
        repository,
        index,
        specification_resolver=lambda: specifications,
    )

    with pytest.raises(SearchDependencyError, match="canonical media root"):
        service.open_pinned_evaluation(
            _text_evaluation_configuration(),
            (_pinned_asset_binding(),),
        )


def test_pinned_evaluation_rejects_unbounded_asset_and_media_scope_before_reads(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, _specifications, _index, service = _ready_generation_search_service(
        tmp_path
    )
    monkeypatch.setattr(
        repository,
        "get_video",
        lambda *_args: pytest.fail("scope limits must precede repository reads"),
    )
    too_many = tuple(
        SearchAssetBinding(
            external_id=f"asset-{index}",
            video_id=f"video-{index}",
            source_sha256=f"{index + 1:064x}",
            byte_size=1,
            duration_seconds=1.0,
        )
        for index in range(129)
    )
    with pytest.raises(SearchDependencyError, match="asset limit"):
        service.open_pinned_evaluation(_text_evaluation_configuration(), too_many)

    oversized = SearchAssetBinding(
        external_id="oversized",
        video_id="oversized",
        source_sha256="f" * 64,
        byte_size=16 * 1024**3 + 1,
        duration_seconds=1.0,
    )
    with pytest.raises(SearchDependencyError, match="media byte limit"):
        service.open_pinned_evaluation(
            _text_evaluation_configuration(),
            (oversized,),
        )


def test_pinned_evaluation_enforces_aggregate_text_budget(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _repository, _specifications, _index, service = _ready_generation_search_service(
        tmp_path
    )
    original = service.open_pinned_evaluation(
        _text_evaluation_configuration(),
        (_pinned_asset_binding(),),
    )
    template = original._states[0]
    original._closed = True
    monkeypatch.setattr("videoscope.search.service._BENCHMARK_MAX_TEXT_POINTS", 1)
    monkeypatch.setattr(
        "videoscope.search.service.PinnedProductSearchSession._pin_asset",
        lambda _self, binding: replace(template, binding=binding),
    )
    assets = (
        _pinned_asset_binding(),
        SearchAssetBinding(
            external_id="portable-asset-2",
            video_id="video-2",
            source_sha256="e" * 64,
            byte_size=10,
            duration_seconds=12.0,
        ),
    )

    with pytest.raises(SearchDependencyError, match="aggregate point limit"):
        service.open_pinned_evaluation(_text_evaluation_configuration(), assets)


def test_pinned_evaluation_rejects_database_path_outside_canonical_media_root(
    tmp_path,
) -> None:
    managed = tmp_path / "managed"
    managed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    source = outside / "video-1.mp4"
    source.write_bytes(b"0123456789")
    repository = Repository(tmp_path / "outside.sqlite3")
    repository.initialize()
    repository.create_video_with_asset(
        video_id="video-1",
        original_name="match.mp4",
        stored_name="video-1.mp4",
        media_path=str(source),
        size_bytes=10,
        source_sha256=_PINNED_SOURCE_SHA256,
    )
    repository.update_video("video-1", status="ready", duration=12.0)
    service = SearchService(
        repository,
        MemoryVectorIndex(),
        media_root=managed,
    )

    with pytest.raises(SearchDependencyError, match="media source"):
        service.open_pinned_evaluation(
            _text_evaluation_configuration(),
            (_pinned_asset_binding(),),
        )
