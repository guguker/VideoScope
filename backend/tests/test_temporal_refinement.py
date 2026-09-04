from pathlib import Path

import pytest

from videoscope.media.ffmpeg import SampledFrame
from videoscope.repository import Repository
from videoscope.search.fusion import EvidenceHit
from videoscope.search.temporal_refinement import TemporalRefiner


class FakeExtractor:
    def extract_frames(self, _source, destination, start, end, *, step, max_width=640):  # type: ignore[no-untyped-def]
        del max_width
        frames = []
        timestamp = start
        while timestamp < end:
            frames.append(SampledFrame(timestamp, Path(destination) / f"{len(frames)}.jpg"))
            timestamp += step
        return frames


class FakeScorer:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores

    def score_images(self, _query: str, _paths: list[Path]) -> list[float]:
        return self.scores[: len(_paths)]


def _repository(tmp_path) -> Repository:  # type: ignore[no-untyped-def]
    repository = Repository(tmp_path / "db.sqlite3")
    repository.initialize()
    repository.create_video(
        video_id="video-1",
        original_name="match.mp4",
        stored_name="video-1.mp4",
        media_path=str(tmp_path / "match.mp4"),
        size_bytes=10,
    )
    repository.update_video("video-1", duration=40)
    return repository


def test_refiner_finds_contiguous_high_scoring_frame_window(tmp_path) -> None:
    hit = EvidenceHit("video-1", "scene-1", 10, 20, "visual", 0.72, "поднимает руку")
    refiner = TemporalRefiner(
        repository=_repository(tmp_path),
        extractor=FakeExtractor(),
        scorer=FakeScorer([0.20, 0.42, 0.91, 0.86, 0.25]),
        temp_dir=tmp_path / "frames",
        sample_step=2,
        min_score=0.45,
    )

    refined = refiner.refine("поднимает руку", [hit])

    assert refined[0].start == pytest.approx(13)
    assert refined[0].end == pytest.approx(17)
    assert refined[0].score == pytest.approx(0.91)
    assert refined[0].metadata["temporal_refinement"] is True
    assert refined[0].metadata["peak_timestamp"] == pytest.approx(14)


def test_refiner_keeps_coarse_hit_when_dense_frames_are_weak(tmp_path) -> None:
    hit = EvidenceHit("video-1", "scene-1", 10, 20, "visual", 0.72, "поднимает руку")
    refiner = TemporalRefiner(
        repository=_repository(tmp_path),
        extractor=FakeExtractor(),
        scorer=FakeScorer([0.10, 0.12, 0.15, 0.14, 0.11]),
        temp_dir=tmp_path / "frames",
        sample_step=2,
        min_score=0.45,
    )

    assert refiner.refine("поднимает руку", [hit]) == [hit]


def test_refiner_only_expands_configured_number_of_candidates(tmp_path) -> None:
    hits = [
        EvidenceHit("video-1", f"scene-{index}", index * 5, index * 5 + 4, "visual", 0.9 - index * 0.1, "query")
        for index in range(3)
    ]
    refiner = TemporalRefiner(
        repository=_repository(tmp_path),
        extractor=FakeExtractor(),
        scorer=FakeScorer([0.9] * 5),
        temp_dir=tmp_path / "frames",
        top_candidates=1,
    )

    refined = refiner.refine("query", hits)

    assert refined[0].metadata["temporal_refinement"] is True
    assert "temporal_refinement" not in refined[1].metadata


def test_refine_strict_surfaces_candidate_failure(tmp_path) -> None:
    class FailingExtractor(FakeExtractor):
        def extract_frames(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("frame extraction failed")

    hit = EvidenceHit("video-1", "scene-1", 10, 20, "visual", 0.72, "query")
    refiner = TemporalRefiner(
        repository=_repository(tmp_path),
        extractor=FailingExtractor(),
        scorer=FakeScorer([0.9]),
        temp_dir=tmp_path / "frames",
    )

    assert refiner.refine("query", [hit]) == [hit]
    with pytest.raises(RuntimeError, match="frame extraction failed"):
        refiner.refine_strict("query", [hit])


def test_benchmark_attestation_requires_explicit_runtime_identities(tmp_path) -> None:
    unpinned = TemporalRefiner(
        repository=_repository(tmp_path),
        extractor=FakeExtractor(),
        scorer=FakeScorer([0.9]),
        temp_dir=tmp_path / "frames",
    )
    pinned = TemporalRefiner(
        repository=unpinned.repository,
        extractor=FakeExtractor(),
        scorer=FakeScorer([0.9]),
        temp_dir=tmp_path / "benchmark-frames",
        top_candidates=3,
        sample_step=1.25,
        ffmpeg_identity="sha256:" + "a" * 64,
        scorer_identity="siglip2@test-revision#sha256:" + "b" * 64,
        runtime_identity="vision-worker:sha256:" + "c" * 64,
    )

    assert unpinned.benchmark_attestation is None
    assert pinned.benchmark_attestation == {
        "ffmpeg_identity": "sha256:" + "a" * 64,
        "implementation_identity": pinned.implementation_identity,
        "runtime_identity": "vision-worker:sha256:" + "c" * 64,
        "scorer_identity": "siglip2@test-revision#sha256:" + "b" * 64,
        "scratch_policy_identity": "private-temporary-directory-delete-on-exit-v1",
        "source_bound": True,
        "strict_complete": True,
        "top_candidates": 3,
    }


def test_benchmark_attestation_rejects_symlinked_scratch_root(tmp_path) -> None:
    target = tmp_path / "real"
    target.mkdir()
    scratch = tmp_path / "scratch"
    scratch.symlink_to(target, target_is_directory=True)
    refiner = TemporalRefiner(
        repository=_repository(tmp_path),
        extractor=FakeExtractor(),
        scorer=FakeScorer([0.9]),
        temp_dir=scratch,
        ffmpeg_identity="sha256:" + "a" * 64,
        scorer_identity="siglip2@test",
        runtime_identity="vision-worker:test",
    )

    assert refiner.benchmark_attestation is None
