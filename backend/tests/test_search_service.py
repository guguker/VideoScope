import pytest

from videoscope.artifacts import StageKind, StageSpecification, StageState
from videoscope.repository import Repository, SegmentRecord
from videoscope.search.service import (
    SearchDependencyError,
    SearchService,
    _hybridize,
    corroborate_lighthouse_hits,
    refine_speech_hit,
)
from videoscope.search.vector_index import EmptyVectorIndex, MemoryVectorIndex
from videoscope.search.fusion import EvidenceHit
from videoscope.search.visual_index import SiglipVisualIndex


def _stage_specification(kind: StageKind) -> StageSpecification:
    return StageSpecification(
        kind=kind,
        schema_version=1,
        implementation_revision=f"tests.search.{kind.value}.v1",
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
    repository.create_video_with_asset(
        video_id="video-1",
        original_name="match.mp4",
        stored_name="video-1.mp4",
        media_path=str(tmp_path / "video-1.mp4"),
        size_bytes=10,
        source_sha256="0" * 64,
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
