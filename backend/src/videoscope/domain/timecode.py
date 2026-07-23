from __future__ import annotations

from collections.abc import Iterable


def format_timecode(seconds: float) -> str:
    total_milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def clamp_interval(start: float, end: float, duration: float) -> tuple[float, float]:
    if duration < 0:
        raise ValueError("duration must be non-negative")
    ordered_start, ordered_end = sorted((float(start), float(end)))
    return (
        max(0.0, min(duration, ordered_start)),
        max(0.0, min(duration, ordered_end)),
    )


def merge_intervals(
    intervals: Iterable[tuple[float, float]],
    *,
    max_gap: float = 0.0,
) -> list[tuple[float, float]]:
    valid = sorted(
        (float(start), float(end))
        for start, end in intervals
        if float(end) > float(start)
    )
    if not valid:
        return []

    merged = [valid[0]]
    for start, end in valid[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end + max_gap:
            merged[-1] = previous_start, max(previous_end, end)
        else:
            merged.append((start, end))
    return merged
