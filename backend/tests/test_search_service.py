from videoscope.repository import Repository
from videoscope.search.service import (
    SearchService,
    _hybridize,
    corroborate_lighthouse_hits,
    refine_speech_hit,
)
from videoscope.search.vector_index import MemoryVectorIndex
from videoscope.search.fusion import EvidenceHit


def test_search_service_returns_enriched_ranked_moment(tmp_path) -> None:
    repository = Repository(tmp_path / "db.sqlite3")
    repository.initialize()
    repository.create_video(
        video_id="video-1",
        original_name="final.mp4",
        stored_name="video-1.mp4",
        media_path=str(tmp_path / "video-1.mp4"),
        size_bytes=100,
    )
    speech = repository.add_segment(
        segment_id="speech-1",
        video_id="video-1",
        start=10,
        end=15,
        modality="speech",
        text="player makes a three point shot",
        confidence=0.9,
        metadata={},
        thumbnail_path="/thumbs/video-1/scene.jpg",
    )
    ocr = repository.add_segment(
        segment_id="ocr-1",
        video_id="video-1",
        start=11,
        end=14,
        modality="ocr",
        text="HOME 87 GUEST 86",
        confidence=0.8,
        metadata={},
        thumbnail_path="/thumbs/video-1/scene.jpg",
    )
    index = MemoryVectorIndex()
    index.replace_video("video-1", [speech, ocr])
    service = SearchService(repository, index)

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


class RecordingIndex:
    def __init__(self, hits: list[EvidenceHit]) -> None:
        self.hits = hits
        self.modalities: set[str] | None = None

    def search(self, _query, *, video_ids=None, modalities=None, limit=50):  # type: ignore[no-untyped-def]
        self.modalities = modalities
        return self.hits[:limit]


class RecordingVisualIndex:
    def __init__(self, hits: list[EvidenceHit]) -> None:
        self.hits = hits
        self.called = False

    def search(self, _query, *, video_ids=None, limit=50):  # type: ignore[no-untyped-def]
        self.called = True
        return self.hits[:limit]


class RecordingMomentSearch:
    def __init__(self, hits: list[EvidenceHit]) -> None:
        self.hits = hits
        self.called = False

    def search(self, _query, _video_ids, *, limit=30):  # type: ignore[no-untyped-def]
        self.called = True
        return self.hits[:limit]


def _repository_with_video(tmp_path) -> Repository:
    repository = Repository(tmp_path / "search.sqlite3")
    repository.initialize()
    repository.create_video(
        video_id="video-1",
        original_name="match.mp4",
        stored_name="video-1.mp4",
        media_path=str(tmp_path / "video-1.mp4"),
        size_bytes=10,
    )
    return repository


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
    repository.add_segment(
        segment_id="speech-name",
        video_id="video-1",
        start=100,
        end=106,
        modality="speech",
        text="Сегодня Мозгова заменили",
        confidence=0.88,
    )
    service = SearchService(repository, RecordingIndex([]))

    results = service.search("Мозгов")

    assert results[0].start == 100
    assert results[0].intent == "entity"
    assert results[0].evidence[0].source == "lexical-stem"


def test_name_search_rejects_unconfirmed_phonetic_collision(tmp_path) -> None:
    repository = _repository_with_video(tmp_path)
    repository.add_segment(
        segment_id="speech-moscow",
        video_id="video-1",
        start=100,
        end=106,
        modality="speech",
        text="команда Москов снова атакует",
        confidence=0.88,
    )
    service = SearchService(repository, RecordingIndex([]))

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
