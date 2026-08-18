from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from videoscope.artifacts import StageKind, StageSpecification, StageState
from videoscope.repository import Repository, SegmentRecord
from videoscope.search.fusion import EvidenceHit
from videoscope.search.service import SearchDependencyError, SearchService


def _specification(kind: StageKind, revision: str = "v1") -> StageSpecification:
    return StageSpecification(
        kind=kind,
        schema_version=1,
        implementation_revision=f"tests.{kind.value}.{revision}",
        parameters={"revision": revision},
    )


def _repository(tmp_path: Path) -> Repository:
    repository = Repository(tmp_path / "search.sqlite3")
    repository.initialize()
    payload = b"video"
    media = tmp_path / "video.mp4"
    media.write_bytes(payload)
    repository.create_video_with_asset(
        video_id="video-1",
        original_name="video.mp4",
        stored_name=media.name,
        media_path=str(media),
        size_bytes=len(payload),
        source_sha256=hashlib.sha256(payload).hexdigest(),
    )
    repository.update_video("video-1", status="ready", duration=30.0)
    return repository


def _publish(
    repository: Repository,
    specification: StageSpecification,
    segment: SegmentRecord | None,
) -> str:
    run = repository.create_stage_run(
        video_id="video-1",
        specification=specification,
    )
    repository.transition_stage_run(run.run_id, StageState.RUNNING)
    return repository.commit_segment_generation(
        run.run_id,
        segments=[] if segment is None else [segment],
    ).generation_id


def _speech(segment_id: str, text: str) -> SegmentRecord:
    return SegmentRecord(
        id=segment_id,
        video_id="video-1",
        start=1.0,
        end=3.0,
        modality="speech",
        text=text,
        confidence=0.9,
        metadata={},
        thumbnail_path=None,
    )


def test_current_active_union_excludes_legacy_old_and_wrong_specification(
    tmp_path,
) -> None:
    repository = _repository(tmp_path)
    specification = _specification(StageKind.SPEECH)
    repository.add_segment(
        segment_id="legacy",
        video_id="video-1",
        start=0,
        end=1,
        modality="speech",
        text="legacy",
        confidence=1,
    )
    _publish(repository, specification, _speech("old", "old generation"))
    _publish(repository, specification, _speech("current", "current generation"))

    current = repository.list_current_active_segments(
        [specification],
        video_ids=["video-1"],
    )

    assert [segment.id for segment in current] == ["current"]
    assert repository.list_current_active_segments(
        [_specification(StageKind.SPEECH, "v2")],
        video_ids=["video-1"],
    ) == []
    matches = repository.search_segments_lexical(
        "generation",
        video_ids=["video-1"],
        specifications=[specification],
    )
    assert [segment.id for segment, _score in matches] == ["current"]


def test_current_active_union_fails_closed_for_corrupt_pointer(tmp_path) -> None:
    repository = _repository(tmp_path)
    specification = _specification(StageKind.SPEECH)
    _publish(repository, specification, _speech("current", "current generation"))
    with sqlite3.connect(repository.database_path) as connection:
        connection.execute(
            """
            UPDATE active_segment_generations
            SET activated_at = ?
            WHERE video_id = ? AND stage_kind = ?
            """,
            ("2000-01-01T00:00:00+00:00", "video-1", "speech"),
        )

    with pytest.raises(ValueError, match="pointer"):
        repository.list_current_active_segments(
            [specification],
            video_ids=["video-1"],
        )


class StaleRecordingIndex:
    available = True

    def search(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        return [
            EvidenceHit(
                "video-1",
                "inactive",
                1,
                3,
                "speech",
                0.99,
                "current generation",
                {"source": "qdrant"},
            ),
            EvidenceHit(
                "video-1",
                "current",
                1,
                3,
                "speech",
                0.9,
                "current generation",
                {"source": "qdrant"},
            ),
            EvidenceHit(
                "video-1",
                "legacy",
                1,
                3,
                "speech",
                1.0,
                "current generation",
                {"source": "qdrant"},
            ),
        ]


def test_search_filters_stale_vector_hits_and_deduplicates_current_lexical_hit(
    tmp_path,
) -> None:
    repository = _repository(tmp_path)
    specification = _specification(StageKind.SPEECH)
    repository.add_segment(
        segment_id="legacy",
        video_id="video-1",
        start=1,
        end=3,
        modality="speech",
        text="current generation",
        confidence=1,
    )
    _publish(repository, specification, _speech("inactive", "current generation"))
    _publish(repository, specification, _speech("current", "current generation"))
    service = SearchService(
        repository,
        StaleRecordingIndex(),
        segment_specifications=[specification],
        thumbnails_dir=tmp_path / "thumbnails",
    )

    results = service.search(
        "current generation",
        mode="speech",
        use_lighthouse=False,
    )

    assert len(results) == 1
    assert {evidence.text for evidence in results[0].evidence} == {"current generation"}
    assert all(
        evidence.source.startswith(("qdrant", "lexical"))
        for evidence in results[0].evidence
    )
    assert len(results[0].evidence) == 1


def test_search_returns_no_legacy_evidence_without_current_specifications(tmp_path) -> None:
    repository = _repository(tmp_path)
    repository.add_segment(
        segment_id="legacy",
        video_id="video-1",
        start=1,
        end=3,
        modality="speech",
        text="legacy searchable phrase",
        confidence=1,
    )
    service = SearchService(repository, StaleRecordingIndex())

    assert service.search(
        "legacy searchable phrase",
        mode="speech",
        use_lighthouse=False,
    ) == []


@pytest.mark.parametrize("failed_run", [False, True])
def test_strict_speech_search_rejects_missing_or_failed_generation(
    tmp_path,
    failed_run: bool,
) -> None:
    repository = _repository(tmp_path)
    specification = _specification(StageKind.SPEECH)
    if failed_run:
        run = repository.create_stage_run(
            video_id="video-1",
            specification=specification,
        )
        repository.transition_stage_run(run.run_id, StageState.RUNNING)
        repository.transition_stage_run(
            run.run_id,
            StageState.FAILED,
            error_code="speech_provider_failed",
        )
    service = SearchService(
        repository,
        StaleRecordingIndex(),
        segment_specifications=[specification],
    )

    with pytest.raises(SearchDependencyError, match="speech evidence"):
        service.search_for_evaluation(
            "spoken phrase",
            mode="speech",
            use_lighthouse=False,
        )


def test_strict_search_accepts_present_empty_complete_generation(tmp_path) -> None:
    repository = _repository(tmp_path)
    specification = _specification(StageKind.SPEECH)
    _publish(repository, specification, None)
    service = SearchService(
        repository,
        StaleRecordingIndex(),
        segment_specifications=[specification],
    )

    assert service.search_for_evaluation(
        "spoken phrase",
        mode="speech",
        use_lighthouse=False,
    ) == []


class CurrentVisualIndex:
    def search(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        return [
            EvidenceHit(
                "video-1",
                "visual:1",
                1,
                3,
                "visual",
                0.9,
                "player jumps",
                {"source": "visual"},
            )
        ]

    def index_is_current(self, _video_id: str) -> bool:
        return True


def test_strict_visual_search_does_not_require_absent_speech(tmp_path) -> None:
    repository = _repository(tmp_path)
    objects = _specification(StageKind.OBJECTS)
    run = repository.create_stage_run(video_id="video-1", specification=objects)
    repository.transition_stage_run(run.run_id, StageState.RUNNING)
    repository.commit_segment_generation(run.run_id, segments=[])
    service = SearchService(
        repository,
        StaleRecordingIndex(),
        visual_search=CurrentVisualIndex(),
        segment_specifications=[objects],
    )

    results = service.search_for_evaluation(
        "player jumps",
        mode="visual",
        use_lighthouse=False,
    )

    assert results and results[0].modalities == ["visual"]


def test_strict_speech_ignores_corrupt_unneeded_scene_pointer(tmp_path) -> None:
    repository = _repository(tmp_path)
    speech = _specification(StageKind.SPEECH)
    scenes = _specification(StageKind.SCENES)
    _publish(repository, speech, None)
    _publish(
        repository,
        scenes,
        SegmentRecord(
            id="scene",
            video_id="video-1",
            start=0,
            end=3,
            modality="scene",
            text="scene",
            confidence=1,
            metadata={},
            thumbnail_path="generations/generation/scene.jpg",
        ),
    )
    with sqlite3.connect(repository.database_path) as connection:
        connection.execute(
            """
            UPDATE active_segment_generations
            SET activated_at = ?
            WHERE video_id = ? AND stage_kind = ?
            """,
            ("2000-01-01T00:00:00+00:00", "video-1", "scenes"),
        )
    service = SearchService(
        repository,
        StaleRecordingIndex(),
        segment_specifications=[speech, scenes],
    )

    assert service.search_for_evaluation(
        "spoken phrase",
        mode="speech",
        use_lighthouse=False,
    ) == []


def test_strict_visual_query_ignores_corrupt_unrouted_speech(tmp_path) -> None:
    repository = _repository(tmp_path)
    speech = _specification(StageKind.SPEECH)
    objects = _specification(StageKind.OBJECTS)
    _publish(repository, speech, None)
    _publish(repository, objects, None)
    with sqlite3.connect(repository.database_path) as connection:
        connection.execute(
            """
            UPDATE active_segment_generations
            SET activated_at = ?
            WHERE video_id = ? AND stage_kind = ?
            """,
            ("2000-01-01T00:00:00+00:00", "video-1", "speech"),
        )
    service = SearchService(
        repository,
        StaleRecordingIndex(),
        visual_search=CurrentVisualIndex(),
        segment_specifications=[speech, objects],
    )

    results = service.search_for_evaluation(
        "player jumps",
        mode="visual",
        use_lighthouse=False,
    )

    assert results and results[0].modalities == ["visual"]


def test_stale_scene_generation_is_not_used_as_thumbnail_fallback(tmp_path) -> None:
    repository = _repository(tmp_path)
    speech = _specification(StageKind.SPEECH)
    scenes_v1 = _specification(StageKind.SCENES, "v1")
    scenes_v2 = _specification(StageKind.SCENES, "v2")
    _publish(repository, speech, _speech("current", "current generation"))
    _publish(
        repository,
        scenes_v1,
        SegmentRecord(
            id="old-scene",
            video_id="video-1",
            start=0,
            end=3,
            modality="scene",
            text="scene",
            confidence=1,
            metadata={},
            thumbnail_path="generations/old/scene.jpg",
        ),
    )
    service = SearchService(
        repository,
        StaleRecordingIndex(),
        segment_specifications=[speech, scenes_v2],
        thumbnails_dir=tmp_path / "thumbnails",
    )

    results = service.search(
        "current generation",
        mode="speech",
        use_lighthouse=False,
    )

    assert results and results[0].thumbnail_url is None
