from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal

import numpy as np


BasketballEventType = Literal[
    "made_three_point",
    "made_two_point",
    "made_free_throw",
]
EventScoringStrategy = Literal[
    "release_outcome_followup",
    "made_two_fixed_lag",
    "made_free_throw_fixed_lag",
]


@dataclass(frozen=True, slots=True)
class BasketballEventSpecification:
    """Описание визуальных стадий одного типа результативного броска."""

    event_type: BasketballEventType
    release_prompts: tuple[str, ...]
    outcome_prompts: tuple[str, ...]
    followup_prompts: tuple[str, ...]
    release_cue: str
    scoring_strategy: EventScoringStrategy = "release_outcome_followup"
    contrast_prompts: tuple[str, ...] = ()
    plus_prompts: tuple[str, ...] = ()
    miss_prompts: tuple[str, ...] = ()
    transition_prompts: tuple[str, ...] = ()
    outcome_cue: str = "ball_through_hoop"
    requires_ball_through_hoop: bool = True
    max_gap_seconds: float = 6.0
    followup_gap_seconds: float = 5.0
    suppression_seconds: float = 9.0
    minimum_followup_score: float = 0.94


_MADE_EVENT_PATTERN = re.compile(
    r"(?:"
    r"\b(?:make|makes|made|score|scores|scored|sink|sinks|sank|"
    r"hit|hits|convert|converts|converted|successful)\b"
    r"|забива\w*|забил\w*|заброс\w*|попад\w*|попал\w*|"
    r"реализ\w*|точн\w*|успешн\w*"
    r")",
    flags=re.IGNORECASE,
)
_MISSED_EVENT_PATTERN = re.compile(
    r"(?:"
    r"\b(?:miss|misses|missed|unsuccessful|no[\s-]?basket)\b"
    r"|\b(?:does|did)\s+not\s+(?:make|score|hit|convert)\b"
    r"|\bnot\s+(?:an?\s+)?(?:make|made|score|scored|hit|converted|successful)\b"
    r"|не\s+(?:забил\w*|забива\w*|попал\w*|попад\w*|реализ\w*|точн\w*)"
    r"|промах\w*|неудач\w*|неточн\w*|мимо"
    r")",
    flags=re.IGNORECASE,
)
_EVENT_TYPE_PATTERNS: dict[BasketballEventType, re.Pattern[str]] = {
    "made_three_point": re.compile(
        r"(?:"
        r"тр[её]хочк\w*|тр[её]шк\w*|(?:три|3)\s+очк\w*"
        r"|\b(?:three|3)[\s-]*point(?:s|er|ers)?\b"
        r"|(?:outside|behind)\s+the\s+(?:three[\s-]*point\s+)?(?:line|arc)"
        r")",
        flags=re.IGNORECASE,
    ),
    "made_two_point": re.compile(
        r"(?:"
        r"двухочк\w*|(?:два|2)\s+очк\w*|из[\s-]*под\s+кольц\w*|"
        r"лей[\s-]*ап\w*|данк\w*"
        r"|\b(?:two|2)[\s-]*point(?:s|er|ers)?\b"
        r"|\b(?:lay[\s-]*up|dunk|mid[\s-]*range\s+shot)s?\b"
        r"|inside\s+the\s+(?:three[\s-]*point\s+)?(?:line|arc)"
        r")",
        flags=re.IGNORECASE,
    ),
    "made_free_throw": re.compile(
        r"(?:"
        r"штрафн\w*(?:\s+(?:брос\w*|лини\w*))?"
        r"|\bfree[\s-]*throws?\b"
        r"|\bfoul[\s-]*shots?\b"
        r")",
        flags=re.IGNORECASE,
    ),
}

_MADE_BASKET_OUTCOME_PROMPTS = (
    "the basketball goes through the hoop after a shot",
    "a basketball entering the hoop through the net",
    "basketball ball at the rim during a made basket",
    "basketball at the hoop after a made shot",
    "close view of the basketball and hoop during a made shot",
)
_MEASURED_MADE_BASKET_OUTCOME_PROMPTS = (
    "the basketball goes through the hoop after a shot",
    "a basketball entering the hoop through the net",
    "basketball at the hoop after a made shot",
    "basketball falling through the net",
    "ball above the basketball rim during a successful shot",
)

_EVENT_SPECIFICATIONS: dict[BasketballEventType, BasketballEventSpecification] = {
    "made_three_point": BasketballEventSpecification(
        event_type="made_three_point",
        release_prompts=(
            "basketball player shoots a jump shot",
            "basketball player releasing a jump shot",
            "basketball player takes a three point shot",
            "basketball player shooting from outside the three-point line",
            "basketball player takes a corner three pointer",
            "basketball player takes a long distance jump shot",
            "wide view of a basketball jump shot",
        ),
        outcome_prompts=_MADE_BASKET_OUTCOME_PROMPTS,
        followup_prompts=(
            "basketball players celebrating a made three point shot",
            "basketball bench celebrating after a made basket",
            "players raise their arms after a successful three point shot",
            "basketball players running back after scoring a basket",
        ),
        release_cue="outside_arc",
    ),
    "made_two_point": BasketballEventSpecification(
        event_type="made_two_point",
        release_prompts=(
            "a basketball player driving toward the basket for a layup",
            "a basketball player jumping near the basket for a layup",
            "a basketball player attempts a close range shot near the hoop",
            "a basketball layup during a game",
            "a basketball player drives through defenders toward the hoop",
        ),
        outcome_prompts=_MEASURED_MADE_BASKET_OUTCOME_PROMPTS,
        followup_prompts=(),
        release_cue="inside_arc",
        scoring_strategy="made_two_fixed_lag",
        contrast_prompts=(
            "basketball player shooting from behind the three-point line",
            "basketball player shoots from outside the three-point line",
        ),
        suppression_seconds=8.0,
    ),
    "made_free_throw": BasketballEventSpecification(
        event_type="made_free_throw",
        release_prompts=(
            "basketball free throw setup with players lined up along the lane",
            "basketball foul shot with players waiting beside the painted lane",
            "a single basketball player shoots while others line the lane",
        ),
        outcome_prompts=_MEASURED_MADE_BASKET_OUTCOME_PROMPTS,
        followup_prompts=(),
        release_cue="free_throw_line",
        scoring_strategy="made_free_throw_fixed_lag",
        plus_prompts=(
            "basketball broadcast graphic shows plus one point after a free throw",
            "basketball scoreboard overlay increases by one point",
            "television basketball scoreboard shows +1",
            "score counter displays +1 after a free throw",
        ),
        miss_prompts=(
            "basketball players fight for a rebound after a missed free throw",
            "basketball ball bounces off the rim after a missed free throw",
            "players immediately rebound a missed foul shot",
        ),
        transition_prompts=(
            "basketball players running in transition",
        ),
        suppression_seconds=8.0,
    ),
}


def event_prompt_stages(event_type: str) -> BasketballEventSpecification:
    """Возвращает настройку стадий для поддерживаемого типа броска."""
    try:
        return _EVENT_SPECIFICATIONS[event_type]  # type: ignore[index]
    except KeyError as error:
        raise ValueError(
            f"unsupported basketball event type: {event_type}"
        ) from error


def classify_basketball_event_query(
    query: str,
) -> BasketballEventSpecification | None:
    """Распознаёт однозначный запрос о результативном баскетбольном броске."""
    normalized = " ".join(query.split()).strip()
    if not normalized or _MISSED_EVENT_PATTERN.search(normalized):
        return None
    if not _MADE_EVENT_PATTERN.search(normalized):
        return None
    matched_types = [
        event_type
        for event_type, pattern in _EVENT_TYPE_PATTERNS.items()
        if pattern.search(normalized)
    ]
    if len(matched_types) != 1:
        return None
    return _EVENT_SPECIFICATIONS[matched_types[0]]


@dataclass(frozen=True, slots=True)
class TemporalEventWindow:
    """Окно события с хронологически согласованными визуальными стадиями."""

    release_index: int
    outcome_index: int
    score: float
    core_score: float
    release_score: float
    outcome_score: float
    followup_index: int | None = None
    followup_score: float | None = None
    raw_release_score: float | None = None
    contrast_score: float | None = None
    plus_score: float | None = None
    reset_score: float | None = None
    miss_score: float | None = None
    transition_score: float | None = None


def _percentile_ranks(values: np.ndarray) -> np.ndarray:
    """Возвращает средние процентили и одинаково обрабатывает равные значения."""
    if len(values) == 1:
        return np.ones(1, dtype=np.float32)
    _, inverse, counts = np.unique(
        values,
        return_inverse=True,
        return_counts=True,
    )
    preceding = np.cumsum(counts) - counts
    average_ranks = preceding + (counts - 1) / 2
    return (average_ranks[inverse] / (len(values) - 1)).astype(
        np.float32,
        copy=False,
    )


def _prompt_percentiles(similarities: np.ndarray) -> np.ndarray:
    """Калибрует каждый текстовый признак по распределению всех кадров."""
    values = np.asarray(similarities, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("similarities must be a two-dimensional array")
    if values.shape[0] == 0 or values.shape[1] == 0:
        return np.empty(values.shape, dtype=np.float32)

    percentiles = np.empty_like(values)
    for column in range(values.shape[1]):
        percentiles[:, column] = _percentile_ranks(values[:, column])
    return percentiles


def aggregate_stage_percentiles(
    similarities: np.ndarray,
    *,
    quantile: float = 0.5,
) -> np.ndarray:
    """Сводит формулировки стадии заданным квантилем их процентилей."""
    percentiles = _prompt_percentiles(similarities)
    if percentiles.size == 0:
        return np.empty(percentiles.shape[0], dtype=np.float32)
    resolved_quantile = max(0.0, min(1.0, float(quantile)))
    return np.quantile(
        percentiles,
        resolved_quantile,
        axis=1,
    ).astype(np.float32, copy=False)


def aggregate_top_k_stage_percentiles(
    similarities: np.ndarray,
    *,
    top_k: int,
) -> np.ndarray:
    """Усредняет несколько наиболее сильных текстовых признаков стадии."""
    percentiles = _prompt_percentiles(similarities)
    if percentiles.size == 0:
        return np.empty(percentiles.shape[0], dtype=np.float32)
    resolved_top_k = max(1, min(int(top_k), percentiles.shape[1]))
    ordered = np.sort(percentiles, axis=1)
    return np.mean(
        ordered[:, -resolved_top_k:],
        axis=1,
    ).astype(np.float32, copy=False)


def _select_temporal_windows(
    candidates: list[TemporalEventWindow],
    *,
    suppression_radius: int,
    limit: int,
) -> list[TemporalEventWindow]:
    """Оставляет лучший результат для одного близкого временного эпизода."""
    selected: list[TemporalEventWindow] = []
    radius = max(0, int(suppression_radius))
    for candidate in sorted(
        candidates,
        key=lambda item: (item.score, item.outcome_index),
        reverse=True,
    ):
        if any(
            abs(candidate.outcome_index - existing.outcome_index) <= radius
            for existing in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def _validated_stage_scores(*scores: np.ndarray) -> tuple[np.ndarray, ...]:
    """Проверяет одномерность и одинаковую длину оценок стадий."""
    arrays = tuple(np.asarray(score, dtype=np.float32) for score in scores)
    if not arrays:
        return ()
    expected_shape = arrays[0].shape
    if (
        arrays[0].ndim != 1
        or any(array.ndim != 1 or array.shape != expected_shape for array in arrays)
    ):
        raise ValueError("event scores must have the same one-dimensional shape")
    return arrays


def rank_made_two_point_windows(
    release_scores: np.ndarray,
    outcome_scores: np.ndarray,
    contrast_scores: np.ndarray,
    *,
    release_lags: tuple[int, int] = (3, 2),
    contrast_lag: int = 2,
    suppression_radius: int = 8,
    limit: int = 50,
) -> list[TemporalEventWindow]:
    """Ранжирует двухочковые по фиксированным задержкам измеренной формулы."""
    release, outcome, contrast = _validated_stage_scores(
        release_scores,
        outcome_scores,
        contrast_scores,
    )
    if limit <= 0:
        return []
    first_lag, second_lag = (
        max(1, int(release_lags[0])),
        max(1, int(release_lags[1])),
    )
    resolved_contrast_lag = max(1, int(contrast_lag))
    earliest = max(first_lag, second_lag, resolved_contrast_lag)
    candidates: list[TemporalEventWindow] = []
    for outcome_index in range(earliest, len(outcome)):
        first_release_index = outcome_index - first_lag
        second_release_index = outcome_index - second_lag
        raw_release_score = float(
            (
                release[first_release_index]
                + release[second_release_index]
            )
            / 2
        )
        contrast_score = float(
            contrast[outcome_index - resolved_contrast_lag]
        )
        release_score = (
            raw_release_score + 0.2 * (1.0 - contrast_score)
        ) / 1.2
        outcome_score = float(outcome[outcome_index])
        score = 0.55 * release_score + 0.45 * outcome_score
        candidates.append(
            TemporalEventWindow(
                release_index=first_release_index,
                outcome_index=outcome_index,
                score=score,
                core_score=score,
                release_score=release_score,
                outcome_score=outcome_score,
                raw_release_score=raw_release_score,
                contrast_score=contrast_score,
            )
        )
    return _select_temporal_windows(
        candidates,
        suppression_radius=suppression_radius,
        limit=limit,
    )


def rank_made_free_throw_windows(
    release_scores: np.ndarray,
    outcome_scores: np.ndarray,
    plus_scores: np.ndarray,
    miss_scores: np.ndarray,
    transition_scores: np.ndarray,
    *,
    release_lag: int = 2,
    plus_window: int = 2,
    reset_window: tuple[int, int] = (2, 5),
    negative_window: int = 3,
    suppression_radius: int = 8,
    limit: int = 50,
) -> list[TemporalEventWindow]:
    """Ранжирует штрафные по измеренным положительным и отрицательным сигналам."""
    release, outcome, plus, miss, transition = _validated_stage_scores(
        release_scores,
        outcome_scores,
        plus_scores,
        miss_scores,
        transition_scores,
    )
    if limit <= 0:
        return []
    resolved_release_lag = max(1, int(release_lag))
    reset_start, reset_end = (
        max(0, int(reset_window[0])),
        max(0, int(reset_window[1])),
    )
    if reset_end < reset_start:
        raise ValueError("reset window end must not precede its start")
    resolved_plus_window = max(0, int(plus_window))
    resolved_negative_window = max(0, int(negative_window))
    last_future_offset = max(
        resolved_plus_window,
        reset_end,
        resolved_negative_window,
    )
    candidates: list[TemporalEventWindow] = []
    for outcome_index in range(
        resolved_release_lag,
        max(resolved_release_lag, len(outcome) - last_future_offset),
    ):
        release_index = outcome_index - resolved_release_lag
        release_score = float(release[release_index])
        outcome_score = float(outcome[outcome_index])
        plus_score = float(
            np.max(
                plus[
                    outcome_index:
                    outcome_index + resolved_plus_window + 1
                ]
            )
        )
        reset_slice = release[
            outcome_index + reset_start:
            outcome_index + reset_end + 1
        ]
        reset_score = float(np.mean(reset_slice))
        miss_score = float(
            np.max(
                miss[
                    outcome_index:
                    outcome_index + resolved_negative_window + 1
                ]
            )
        )
        transition_score = float(
            np.max(
                transition[
                    outcome_index:
                    outcome_index + resolved_negative_window + 1
                ]
            )
        )
        score = (
            0.50 * release_score
            + 0.30 * outcome_score
            + 0.10 * plus_score
            + 0.10 * reset_score
            - 0.15 * miss_score
            - 0.05 * transition_score
        )
        candidates.append(
            TemporalEventWindow(
                release_index=release_index,
                outcome_index=outcome_index,
                score=score,
                core_score=score,
                release_score=release_score,
                outcome_score=outcome_score,
                followup_index=outcome_index + reset_end,
                followup_score=reset_score,
                plus_score=plus_score,
                reset_score=reset_score,
                miss_score=miss_score,
                transition_score=transition_score,
            )
        )
    return _select_temporal_windows(
        candidates,
        suppression_radius=suppression_radius,
        limit=limit,
    )


def rank_temporal_event_windows(
    release_scores: np.ndarray,
    outcome_scores: np.ndarray,
    *,
    followup_scores: np.ndarray | None = None,
    max_gap: int = 6,
    followup_gap: int = 5,
    suppression_radius: int = 6,
    minimum_stage_score: float = 0.65,
    minimum_followup_score: float = 0.0,
    followup_weight: float = 0.25,
    limit: int = 50,
) -> list[TemporalEventWindow]:
    """Ищет пары «выпуск мяча → результат» и подавляет дубликаты одного броска."""
    release = np.asarray(release_scores, dtype=np.float32)
    outcome = np.asarray(outcome_scores, dtype=np.float32)
    if release.ndim != 1 or outcome.ndim != 1 or release.shape != outcome.shape:
        raise ValueError("stage scores must have the same one-dimensional shape")
    followup = (
        np.asarray(followup_scores, dtype=np.float32)
        if followup_scores is not None
        else None
    )
    if followup is not None and (
        followup.ndim != 1 or followup.shape != release.shape
    ):
        raise ValueError("followup scores must match the event stage shape")
    if len(release) < 2 or limit <= 0:
        return []

    resolved_gap = max(1, int(max_gap))
    resolved_followup_weight = max(0.0, min(1.0, float(followup_weight)))
    candidates: list[TemporalEventWindow] = []
    for outcome_index in range(1, len(outcome)):
        outcome_score = float(outcome[outcome_index])
        if outcome_score < minimum_stage_score:
            continue
        first_release = max(0, outcome_index - resolved_gap)
        release_slice = release[first_release:outcome_index]
        relative_index = int(np.argmax(release_slice))
        release_index = first_release + relative_index
        release_score = float(release[release_index])
        if release_score < minimum_stage_score:
            continue
        followup_index: int | None = None
        followup_score: float | None = None
        if followup is not None:
            last_followup = min(
                len(followup),
                outcome_index + max(0, int(followup_gap)) + 1,
            )
            followup_slice = followup[outcome_index:last_followup]
            relative_followup_index = int(np.argmax(followup_slice))
            followup_index = outcome_index + relative_followup_index
            followup_score = float(followup[followup_index])
            if followup_score < minimum_followup_score:
                continue
        core_score = (
            2 * release_score * outcome_score / (release_score + outcome_score)
        )
        score = core_score
        if followup_score is not None:
            score = (
                core_score * (1.0 - resolved_followup_weight)
                + followup_score * resolved_followup_weight
            )
        candidates.append(
            TemporalEventWindow(
                release_index=release_index,
                outcome_index=outcome_index,
                score=score,
                core_score=core_score,
                release_score=release_score,
                outcome_score=outcome_score,
                followup_index=followup_index,
                followup_score=followup_score,
            )
        )

    selected: list[TemporalEventWindow] = []
    radius = max(0, int(suppression_radius))
    for candidate in sorted(
        candidates,
        key=lambda item: (item.score, item.outcome_index),
        reverse=True,
    ):
        if any(
            abs(candidate.outcome_index - existing.outcome_index) <= radius
            for existing in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected
