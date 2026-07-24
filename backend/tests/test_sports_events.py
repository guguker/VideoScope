import numpy as np
import pytest

from videoscope.search.sports_events import (
    aggregate_stage_percentiles,
    aggregate_top_k_stage_percentiles,
    classify_basketball_event_query,
    event_prompt_stages,
    rank_made_free_throw_windows,
    rank_made_two_point_windows,
    rank_temporal_event_windows,
)


@pytest.mark.parametrize(
    ("query", "event_type"),
    [
        ("игрок забивает трёхочковый", "made_three_point"),
        ("реализованный трехочковый бросок", "made_three_point"),
        ("игрок забивает три очка", "made_three_point"),
        ("player makes a three-pointer", "made_three_point"),
        ("игрок забил двухочковый", "made_two_point"),
        ("игрок забивает два очка", "made_two_point"),
        ("точный бросок из-под кольца", "made_two_point"),
        ("player scores a two-point shot", "made_two_point"),
        ("player scores two points", "made_two_point"),
        ("игрок реализует штрафной бросок", "made_free_throw"),
        ("точное попадание со штрафной линии", "made_free_throw"),
        ("player sinks a free throw", "made_free_throw"),
    ],
)
def test_classifies_made_basketball_event_queries(
    query: str,
    event_type: str,
) -> None:
    specification = classify_basketball_event_query(query)

    assert specification is not None
    assert specification.event_type == event_type
    assert specification.requires_ball_through_hoop is True


@pytest.mark.parametrize(
    "query",
    [
        "игрок бросает штрафной",
        "игрок не забил штрафной",
        "неточный штрафной бросок",
        "промах при двухочковом броске",
        "player misses a three-pointer",
        "player did not make a free throw",
        "not a successful two-point shot",
        "made three-pointer and made free throw",
        "человек открывает дверь",
    ],
)
def test_does_not_label_attempts_misses_or_ambiguous_queries_as_made(
    query: str,
) -> None:
    assert classify_basketball_event_query(query) is None


def test_three_point_prompt_stages_remain_unchanged() -> None:
    specification = event_prompt_stages("made_three_point")

    assert specification.release_cue == "outside_arc"
    assert "basketball player takes a three point shot" in (
        specification.release_prompts
    )
    assert "close view of the basketball and hoop during a made shot" in (
        specification.outcome_prompts
    )
    assert specification.followup_prompts
    assert specification.scoring_strategy == "release_outcome_followup"


def test_two_point_prompt_stages_use_measured_drive_and_contrast_sets() -> None:
    specification = event_prompt_stages("made_two_point")

    assert specification.release_cue == "inside_arc"
    assert specification.release_prompts == (
        "a basketball player driving toward the basket for a layup",
        "a basketball player jumping near the basket for a layup",
        "a basketball player attempts a close range shot near the hoop",
        "a basketball layup during a game",
        "a basketball player drives through defenders toward the hoop",
    )
    assert specification.contrast_prompts == (
        "basketball player shooting from behind the three-point line",
        "basketball player shoots from outside the three-point line",
    )
    assert "basketball falling through the net" in specification.outcome_prompts
    assert "ball above the basketball rim during a successful shot" in (
        specification.outcome_prompts
    )
    assert specification.scoring_strategy == "made_two_fixed_lag"
    assert specification.suppression_seconds == 8.0


def test_free_throw_prompt_stages_use_measured_auxiliary_signals() -> None:
    specification = event_prompt_stages("made_free_throw")

    assert specification.release_cue == "free_throw_line"
    assert specification.release_prompts == (
        "basketball free throw setup with players lined up along the lane",
        "basketball foul shot with players waiting beside the painted lane",
        "a single basketball player shoots while others line the lane",
    )
    assert len(specification.outcome_prompts) == 5
    assert len(specification.plus_prompts) == 4
    assert len(specification.miss_prompts) == 3
    assert specification.transition_prompts == (
        "basketball players running in transition",
    )
    assert specification.scoring_strategy == "made_free_throw_fixed_lag"
    assert specification.suppression_seconds == 8.0


def test_event_prompt_stages_reject_unknown_event_type() -> None:
    with pytest.raises(ValueError, match="unsupported basketball event type"):
        event_prompt_stages("made_four_point")


def test_stage_aggregation_resists_single_prompt_spike() -> None:
    similarities = np.array(
        [
            [0.99, 0.10, 0.10],
            [0.80, 0.80, 0.80],
            [0.20, 0.20, 0.20],
        ],
        dtype=np.float32,
    )

    scores = aggregate_stage_percentiles(similarities)

    assert scores[1] > scores[2] > scores[0]


def test_stage_aggregation_supports_q75_and_top_two_mean() -> None:
    similarities = np.array(
        [
            [4.0, 0.0, 0.0],
            [3.0, 3.0, 0.0],
            [2.0, 2.0, 2.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )

    median = aggregate_stage_percentiles(similarities)
    q75 = aggregate_stage_percentiles(similarities, quantile=0.75)
    top_two = aggregate_top_k_stage_percentiles(similarities, top_k=2)

    assert q75[0] > median[0]
    assert q75 == pytest.approx(top_two)


def test_two_point_scoring_uses_fixed_release_lags_and_three_point_contrast() -> None:
    release = np.zeros(20, dtype=np.float32)
    outcome = np.zeros(20, dtype=np.float32)
    contrast = np.ones(20, dtype=np.float32)
    release[3] = 0.8
    release[4] = 0.6
    release[5] = 1.0
    contrast[4] = 0.25
    outcome[6] = 0.9

    windows = rank_made_two_point_windows(
        release,
        outcome,
        contrast,
        limit=1,
    )

    raw_release = (0.8 + 0.6) / 2
    discriminated_release = (raw_release + 0.2 * (1 - 0.25)) / 1.2
    expected = 0.55 * discriminated_release + 0.45 * 0.9
    assert windows[0].release_index == 3
    assert windows[0].outcome_index == 6
    assert windows[0].raw_release_score == pytest.approx(raw_release)
    assert windows[0].release_score == pytest.approx(discriminated_release)
    assert windows[0].contrast_score == pytest.approx(0.25)
    assert windows[0].score == pytest.approx(expected)


def test_two_point_scoring_suppresses_candidates_within_eight_frames() -> None:
    release = np.full(25, 0.8, dtype=np.float32)
    outcome = np.zeros(25, dtype=np.float32)
    contrast = np.full(25, 0.2, dtype=np.float32)
    outcome[6] = 0.90
    outcome[10] = 1.00
    outcome[19] = 0.95

    windows = rank_made_two_point_windows(
        release,
        outcome,
        contrast,
        limit=2,
    )

    assert [window.outcome_index for window in windows] == [10, 19]


def test_free_throw_scoring_uses_measured_fixed_lag_formula() -> None:
    release = np.zeros(15, dtype=np.float32)
    outcome = np.zeros(15, dtype=np.float32)
    plus = np.zeros(15, dtype=np.float32)
    miss = np.zeros(15, dtype=np.float32)
    transition = np.zeros(15, dtype=np.float32)
    release[2] = 0.8
    release[3] = 1.0
    release[6:10] = [0.6, 0.8, 1.0, 0.4]
    outcome[4] = 0.9
    plus[6] = 0.7
    miss[5] = 0.2
    transition[7] = 0.3

    windows = rank_made_free_throw_windows(
        release,
        outcome,
        plus,
        miss,
        transition,
        limit=1,
    )

    expected = (
        0.50 * 0.8
        + 0.30 * 0.9
        + 0.10 * 0.7
        + 0.10 * 0.7
        - 0.15 * 0.2
        - 0.05 * 0.3
    )
    assert windows[0].release_index == 2
    assert windows[0].outcome_index == 4
    assert windows[0].followup_index == 9
    assert windows[0].release_score == pytest.approx(0.8)
    assert windows[0].outcome_score == pytest.approx(0.9)
    assert windows[0].plus_score == pytest.approx(0.7)
    assert windows[0].reset_score == pytest.approx(0.7)
    assert windows[0].miss_score == pytest.approx(0.2)
    assert windows[0].transition_score == pytest.approx(0.3)
    assert windows[0].score == pytest.approx(expected)


def test_temporal_window_requires_release_before_outcome() -> None:
    release_scores = np.array([0.0, 0.95, 0.0, 0.0, 0.0], dtype=np.float32)
    outcome_scores = np.array([0.99, 0.0, 0.0, 0.90, 0.0], dtype=np.float32)

    windows = rank_temporal_event_windows(
        release_scores,
        outcome_scores,
        max_gap=4,
        suppression_radius=0,
    )

    assert len(windows) == 1
    assert windows[0].release_index == 1
    assert windows[0].outcome_index == 3


def test_temporal_window_rejects_outcome_without_nearby_release() -> None:
    release_scores = np.array([0.0, 0.0, 0.90, 0.0, 0.0, 0.0], dtype=np.float32)
    outcome_scores = np.array([0.95, 0.0, 0.0, 0.0, 0.0, 0.95], dtype=np.float32)

    windows = rank_temporal_event_windows(
        release_scores,
        outcome_scores,
        max_gap=2,
        minimum_stage_score=0.5,
    )

    assert windows == []


def test_temporal_window_collapses_adjacent_outcomes_of_one_shot() -> None:
    release_scores = np.array([0.0, 0.95, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    outcome_scores = np.array([0.0, 0.0, 0.0, 0.85, 0.92, 0.0], dtype=np.float32)

    windows = rank_temporal_event_windows(
        release_scores,
        outcome_scores,
        max_gap=4,
        suppression_radius=2,
    )

    assert len(windows) == 1
    assert windows[0].outcome_index == 4
    assert windows[0].score == pytest.approx(
        2 * 0.95 * 0.92 / (0.95 + 0.92),
        rel=1e-6,
    )


def test_temporal_window_can_require_followup_context() -> None:
    release_scores = np.array(
        [0.0, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0.95, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    outcome_scores = np.array(
        [0.0, 0.0, 0.0, 0.92, 0.0, 0.0, 0.0, 0.0, 0.0, 0.92, 0.0],
        dtype=np.float32,
    )
    followup_scores = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.96, 0.0, 0.0, 0.0, 0.0, 0.20],
        dtype=np.float32,
    )

    windows = rank_temporal_event_windows(
        release_scores,
        outcome_scores,
        followup_scores=followup_scores,
        max_gap=3,
        followup_gap=2,
        minimum_followup_score=0.90,
        suppression_radius=0,
    )

    assert len(windows) == 1
    assert windows[0].release_index == 1
    assert windows[0].outcome_index == 3
    assert windows[0].followup_index == 5
    assert windows[0].followup_score == pytest.approx(0.96)
    core_score = 2 * 0.95 * 0.92 / (0.95 + 0.92)
    assert windows[0].score == pytest.approx(
        core_score * 0.75 + 0.96 * 0.25,
        rel=1e-6,
    )


def test_followup_score_can_promote_a_more_complete_event() -> None:
    release_scores = np.array(
        [0.0, 0.98, 0.0, 0.0, 0.0, 0.0, 0.97, 0.0, 0.0],
        dtype=np.float32,
    )
    outcome_scores = np.array(
        [0.0, 0.0, 0.98, 0.0, 0.0, 0.0, 0.0, 0.97, 0.0],
        dtype=np.float32,
    )
    followup_scores = np.array(
        [0.0, 0.0, 0.94, 0.0, 0.0, 0.0, 0.0, 1.00, 0.0],
        dtype=np.float32,
    )

    windows = rank_temporal_event_windows(
        release_scores,
        outcome_scores,
        followup_scores=followup_scores,
        max_gap=2,
        followup_gap=1,
        minimum_followup_score=0.90,
        suppression_radius=0,
    )

    assert [window.release_index for window in windows] == [6, 1]
    assert windows[0].score == pytest.approx(0.97 * 0.75 + 1.00 * 0.25)


def test_temporal_window_validates_stage_shapes() -> None:
    with pytest.raises(ValueError, match="same one-dimensional shape"):
        rank_temporal_event_windows(
            np.array([0.9, 0.8], dtype=np.float32),
            np.array([[0.9, 0.8]], dtype=np.float32),
        )
    with pytest.raises(ValueError, match="followup scores"):
        rank_temporal_event_windows(
            np.array([0.9, 0.8], dtype=np.float32),
            np.array([0.8, 0.9], dtype=np.float32),
            followup_scores=np.array([0.9], dtype=np.float32),
        )
