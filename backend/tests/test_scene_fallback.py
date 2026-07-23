from videoscope.providers.scenes import normalize_scenes


def test_scene_normalization_inserts_full_video_fallback() -> None:
    assert normalize_scenes([], duration=12.5) == [(0.0, 12.5)]


def test_scene_normalization_splits_very_long_scene() -> None:
    scenes = normalize_scenes([(0.0, 95.0)], duration=95.0, max_scene_seconds=30.0)

    assert scenes == [(0.0, 30.0), (30.0, 60.0), (60.0, 90.0), (90.0, 95.0)]


def test_scene_normalization_drops_micro_scenes_by_merging_forward() -> None:
    scenes = normalize_scenes(
        [(0.0, 0.25), (0.25, 6.0), (6.0, 6.1), (6.1, 10.0)],
        duration=10.0,
        min_scene_seconds=0.5,
    )

    assert scenes == [(0.0, 6.0), (6.0, 10.0)]

