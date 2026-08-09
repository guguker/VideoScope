import math

from videoscope.search.fusion import EvidenceHit, calibrate_hits, fuse_hits


def test_fusion_rewards_independent_modalities_for_same_moment() -> None:
    hits = [
        EvidenceHit("v1", "speech-1", 10, 18, "speech", 0.82, "трёхочковый бросок"),
        EvidenceHit("v1", "ocr-1", 11, 17, "ocr", 0.74, "HOME 87 - 86"),
        EvidenceHit("v1", "visual-1", 10, 19, "visual", 0.78, "player shoots"),
        EvidenceHit("v1", "speech-2", 50, 58, "speech", 0.90, "timeout"),
    ]

    results = fuse_hits(hits, limit=10)

    assert results[0].start == 10
    assert results[0].end == 19
    assert results[0].modalities == ["ocr", "speech", "visual"]
    assert results[0].score > results[1].score


def test_fusion_does_not_merge_distant_moments() -> None:
    hits = [
        EvidenceHit("v1", "a", 0, 5, "speech", 0.8, "first"),
        EvidenceHit("v1", "b", 30, 35, "ocr", 0.9, "second"),
    ]

    results = fuse_hits(hits, limit=10)

    assert len(results) == 2


def test_fusion_keeps_different_videos_separate() -> None:
    hits = [
        EvidenceHit("v1", "a", 0, 5, "speech", 0.8, "first"),
        EvidenceHit("v2", "b", 0, 5, "ocr", 0.9, "second"),
    ]

    results = fuse_hits(hits, limit=10)

    assert {result.video_id for result in results} == {"v1", "v2"}


def test_fusion_accepts_query_specific_modality_weights() -> None:
    hits = [
        EvidenceHit("v1", "speech", 0, 3, "speech", 0.72, "Мозгов"),
        EvidenceHit("v1", "visual", 20, 23, "visual", 0.80, "Мозгов"),
    ]

    results = fuse_hits(
        hits,
        modality_weights={"speech": 1.8, "visual": 0.35},
    )

    assert results[0].modalities == ["speech"]


def test_calibration_preserves_raw_score_and_adds_rank_signal() -> None:
    hits = [
        EvidenceHit("v1", "first", 0, 2, "visual", 0.62, "first"),
        EvidenceHit("v1", "second", 4, 6, "visual", 0.58, "second"),
    ]

    calibrated = calibrate_hits(hits)

    assert calibrated[0].score > calibrated[1].score
    assert calibrated[0].metadata["raw_score"] == 0.62
    assert calibrated[0].metadata["calibrated_score"] == calibrated[0].score


def test_action_weight_does_not_saturate_and_replace_rank_with_chronology() -> None:
    hits = [
        EvidenceHit("v1", "earlier", 100, 106, "visual", 0.97, "earlier"),
        EvidenceHit("v1", "stronger-late", 6800, 6806, "visual", 0.99, "later"),
    ]

    results = fuse_hits(
        calibrate_hits(hits),
        limit=1,
        modality_weights={"visual": 1.48},
    )

    assert results[0].start == 6800
    assert results[0].score < 1.0


def test_calibration_drops_non_finite_or_invalid_evidence() -> None:
    valid = EvidenceHit("v1", "valid", 0, 2, "visual", 0.2, "valid")
    invalid = [
        EvidenceHit("v1", "nan-score", 0, 2, "visual", math.nan, "invalid"),
        EvidenceHit("v1", "inf-score", 0, 2, "visual", math.inf, "invalid"),
        EvidenceHit("v1", "nan-start", math.nan, 2, "visual", 0.9, "invalid"),
        EvidenceHit("v1", "inf-end", 0, math.inf, "visual", 0.9, "invalid"),
        EvidenceHit("v1", "reversed", 2, 1, "visual", 0.9, "invalid"),
        EvidenceHit("v1", "empty", 2, 2, "visual", 0.9, "invalid"),
    ]

    calibrated = calibrate_hits([*invalid, valid])

    assert [hit.segment_id for hit in calibrated] == ["valid"]
    assert math.isfinite(calibrated[0].score)


def test_fusion_drops_invalid_evidence_even_without_calibration() -> None:
    hits = [
        EvidenceHit("v1", "nan-score", 0, 2, "visual", math.nan, "invalid"),
        EvidenceHit("v1", "nan-time", math.nan, 2, "speech", 0.99, "invalid"),
        EvidenceHit("v1", "reversed", 3, 1, "ocr", 0.99, "invalid"),
        EvidenceHit("v1", "valid", 10, 12, "visual", 0.1, "valid"),
    ]

    results = fuse_hits(hits)

    assert len(results) == 1
    assert results[0].start == 10
    assert [hit.segment_id for hit in results[0].evidence] == ["valid"]
