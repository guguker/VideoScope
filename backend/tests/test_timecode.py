from videoscope.domain.timecode import clamp_interval, format_timecode, merge_intervals


def test_format_timecode_supports_hours_and_milliseconds() -> None:
    assert format_timecode(3723.456) == "01:02:03.456"


def test_clamp_interval_orders_and_bounds_values() -> None:
    assert clamp_interval(12.0, -2.0, duration=10.0) == (0.0, 10.0)


def test_clamp_interval_collapses_ranges_after_video_end() -> None:
    assert clamp_interval(20.0, 30.0, duration=10.0) == (10.0, 10.0)


def test_merge_intervals_joins_overlaps_and_small_gaps() -> None:
    intervals = [(0.0, 2.0), (2.3, 4.0), (8.0, 10.0), (9.0, 12.0)]

    assert merge_intervals(intervals, max_gap=0.5) == [(0.0, 4.0), (8.0, 12.0)]


def test_merge_intervals_ignores_empty_ranges() -> None:
    assert merge_intervals([(3.0, 3.0), (5.0, 4.0), (1.0, 2.0)]) == [(1.0, 2.0)]
