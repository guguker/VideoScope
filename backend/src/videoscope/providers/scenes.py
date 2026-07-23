from __future__ import annotations

from pathlib import Path

from videoscope.providers.base import ProviderState, ProviderStatus


def normalize_scenes(
    scenes: list[tuple[float, float]],
    *,
    duration: float,
    min_scene_seconds: float = 0.5,
    max_scene_seconds: float = 30.0,
) -> list[tuple[float, float]]:
    bounded = sorted(
        (max(0.0, start), min(duration, end))
        for start, end in scenes
        if end > start and start < duration
    )
    if not bounded and duration > 0:
        bounded = [(0.0, duration)]

    without_micro_scenes: list[tuple[float, float]] = []
    pending_start: float | None = None
    for start, end in bounded:
        if end - start < min_scene_seconds:
            pending_start = start if pending_start is None else min(pending_start, start)
            continue
        if pending_start is not None:
            start = pending_start
            pending_start = None
        without_micro_scenes.append((start, end))
    if pending_start is not None:
        if without_micro_scenes:
            start, _ = without_micro_scenes[-1]
            without_micro_scenes[-1] = (start, duration)
        elif duration > pending_start:
            without_micro_scenes.append((pending_start, duration))

    split: list[tuple[float, float]] = []
    for start, end in without_micro_scenes:
        cursor = start
        while end - cursor > max_scene_seconds:
            split.append((cursor, cursor + max_scene_seconds))
            cursor += max_scene_seconds
        if end > cursor:
            split.append((cursor, end))
    return split


class SceneDetector:
    id = "pyscenedetect"

    def __init__(self, threshold: float = 3.0, max_scene_seconds: float = 30.0) -> None:
        self.threshold = threshold
        self.max_scene_seconds = max_scene_seconds

    def status(self) -> ProviderStatus:
        try:
            import scenedetect  # noqa: F401
        except ImportError:
            return ProviderStatus(
                self.id,
                "PySceneDetect",
                ProviderState.UNAVAILABLE,
                "Python package is not installed",
            )
        return ProviderStatus(
            self.id,
            "PySceneDetect",
            ProviderState.READY,
            "Adaptive scene boundary detection",
        )

    def detect(self, source: Path, duration: float) -> list[tuple[float, float]]:
        from scenedetect import AdaptiveDetector, SceneManager, open_video

        video = open_video(str(source))
        manager = SceneManager()
        manager.add_detector(AdaptiveDetector(adaptive_threshold=self.threshold))
        manager.detect_scenes(video, show_progress=False)
        raw = [
            (start.get_seconds(), end.get_seconds())
            for start, end in manager.get_scene_list(start_in_scene=True)
        ]
        return normalize_scenes(
            raw,
            duration=duration,
            max_scene_seconds=self.max_scene_seconds,
        )

