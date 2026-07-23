from __future__ import annotations

from pathlib import Path
from typing import Protocol
from uuid import uuid4

from videoscope.media.ffmpeg import FFmpeg
from videoscope.providers.scenes import SceneDetector, normalize_scenes
from videoscope.providers.types import ObjectTag, TimedText
from videoscope.repository import Repository, SegmentRecord


def merge_timed_text(
    items: list[TimedText],
    *,
    max_duration: float = 14.0,
    max_gap: float = 1.25,
) -> list[TimedText]:
    """Формирует пригодные для поиска фрагменты речи вместо векторизации отдельных слов."""
    merged: list[TimedText] = []
    group: list[TimedText] = []

    def flush() -> None:
        if not group:
            return
        duration = sum(max(0.01, item.end - item.start) for item in group)
        confidence = sum(
            item.confidence * max(0.01, item.end - item.start) for item in group
        ) / duration
        words = [
            word
            for item in group
            for word in item.metadata.get("words", [])
            if isinstance(word, dict)
        ]
        merged.append(
            TimedText(
                start=group[0].start,
                end=group[-1].end,
                text=" ".join(item.text.strip() for item in group if item.text.strip()),
                confidence=confidence,
                metadata={
                    **group[0].metadata,
                    "merged_segments": len(group),
                    **({"words": words} if words else {}),
                },
            )
        )
        group.clear()

    for item in sorted(items, key=lambda value: (value.start, value.end)):
        if not group:
            group.append(item)
            continue
        gap = item.start - group[-1].end
        combined_duration = item.end - group[0].start
        if gap <= max_gap and combined_duration <= max_duration:
            group.append(item)
        else:
            flush()
            group.append(item)
    flush()
    return merged


class SpeechProvider(Protocol):
    def transcribe(self, source: Path) -> list[TimedText]: ...


class OCRProvider(Protocol):
    def read(self, image: Path) -> list[tuple[str, float]]: ...


class ObjectProvider(Protocol):
    def detect(self, image: Path) -> list[ObjectTag]: ...


class VectorIndex(Protocol):
    def replace_video(self, video_id: str, segments: list[SegmentRecord]) -> None: ...


class VisualIndex(Protocol):
    def replace_video(self, video_id: str, segments: list[SegmentRecord]) -> None: ...


class MomentRetriever(Protocol):
    def prepare(self, video_id: str, source: Path, duration: float) -> None: ...


class Indexer:
    def __init__(
        self,
        *,
        repository: Repository,
        thumbnails_dir: Path,
        ffmpeg: FFmpeg,
        scenes: SceneDetector,
        vector_index: VectorIndex,
        visual_index: VisualIndex | None = None,
        speech: SpeechProvider | None = None,
        ocr: OCRProvider | None = None,
        objects: ObjectProvider | None = None,
        moment_retriever: MomentRetriever | None = None,
    ) -> None:
        self.repository = repository
        self.thumbnails_dir = thumbnails_dir
        self.ffmpeg = ffmpeg
        self.scenes = scenes
        self.vector_index = vector_index
        self.visual_index = visual_index
        self.speech = speech
        self.ocr = ocr
        self.objects = objects
        self.moment_retriever = moment_retriever

    def _add_timed_text(self, video_id: str, modality: str, item: TimedText) -> None:
        text = item.text.strip()
        if not text or item.end <= item.start:
            return
        self.repository.add_segment(
            segment_id=uuid4().hex,
            video_id=video_id,
            start=item.start,
            end=item.end,
            modality=modality,
            text=text,
            confidence=item.confidence,
            metadata=item.metadata,
        )

    def process(self, video_id: str) -> None:
        video = self.repository.get_video(video_id)
        if video is None:
            raise KeyError(video_id)
        source = Path(video.media_path)
        warnings: list[str] = []
        self.repository.update_video(video_id, status="processing", progress=0.01, stage="probe", error=None)

        try:
            probe = self.ffmpeg.probe(source)
            if probe.duration <= 0:
                raise ValueError("video duration is zero")
            self.repository.update_video(
                video_id,
                duration=probe.duration,
                width=probe.width,
                height=probe.height,
                fps=probe.fps,
                progress=0.08,
                stage="scenes",
            )
            self.repository.clear_segments(video_id)

            try:
                scene_intervals = self.scenes.detect(source, probe.duration)
            except Exception as error:
                warnings.append(f"scenes: {error}")
                scene_intervals = normalize_scenes([], duration=probe.duration)

            frame_paths: list[tuple[float, float, Path]] = []
            for index, (start, end) in enumerate(scene_intervals):
                frame_path = self.thumbnails_dir / video_id / f"scene-{index:04d}.jpg"
                self.ffmpeg.extract_frame(source, frame_path, start + (end - start) / 2)
                frame_paths.append((start, end, frame_path))
                self.repository.add_segment(
                    segment_id=uuid4().hex,
                    video_id=video_id,
                    start=start,
                    end=end,
                    modality="scene",
                    text=f"scene {index + 1}",
                    confidence=1.0,
                    metadata={"scene_index": index},
                    thumbnail_path=str(frame_path),
                )

            thumbnail_path = frame_paths[0][2] if frame_paths else None
            self.repository.update_video(
                video_id,
                thumbnail_path=str(thumbnail_path) if thumbnail_path else None,
                progress=0.25,
                stage="speech",
            )

            if self.speech is not None:
                try:
                    for item in merge_timed_text(self.speech.transcribe(source)):
                        self._add_timed_text(video_id, "speech", item)
                except Exception as error:
                    warnings.append(f"speech: {error}")

            self.repository.update_video(video_id, progress=0.45, stage="vision")
            total_frames = max(1, len(frame_paths))
            ocr_failed = False
            objects_failed = False
            for index, (start, end, frame_path) in enumerate(frame_paths):
                if self.ocr is not None and not ocr_failed:
                    try:
                        recognized = self.ocr.read(frame_path)
                        unique: dict[str, tuple[str, float]] = {}
                        for text, confidence in recognized:
                            normalized = " ".join(text.split()).strip()
                            key = normalized.casefold()
                            if len(key) < 2:
                                continue
                            current = unique.get(key)
                            if current is None or confidence > current[1]:
                                unique[key] = (normalized, confidence)
                        if unique:
                            ordered = list(unique.values())
                            self.repository.add_segment(
                                segment_id=uuid4().hex,
                                video_id=video_id,
                                start=start,
                                end=end,
                                modality="ocr",
                                text=" · ".join(text for text, _ in ordered),
                                confidence=sum(confidence for _, confidence in ordered) / len(ordered),
                                metadata={"scene_index": index, "line_count": len(ordered)},
                                thumbnail_path=str(frame_path),
                            )
                    except Exception as error:
                        warnings.append(f"ocr: {error}")
                        ocr_failed = True

                if self.objects is not None and not objects_failed:
                    try:
                        tags = self.objects.detect(frame_path)
                        if tags:
                            self.repository.add_segment(
                                segment_id=uuid4().hex,
                                video_id=video_id,
                                start=start,
                                end=end,
                                modality="objects",
                                text=", ".join(sorted({tag.label for tag in tags})),
                                confidence=max(tag.confidence for tag in tags),
                                metadata={
                                    "scene_index": index,
                                    "objects": [
                                        {
                                            "label": tag.label,
                                            "confidence": tag.confidence,
                                            **tag.metadata,
                                        }
                                        for tag in tags
                                    ],
                                },
                                thumbnail_path=str(frame_path),
                            )
                    except Exception as error:
                        warnings.append(f"objects: {error}")
                        objects_failed = True

                self.repository.update_video(
                    video_id,
                    progress=0.45 + 0.4 * ((index + 1) / total_frames),
                    stage="vision",
                )

            self.repository.update_video(video_id, progress=0.9, stage="index")
            segments = self.repository.list_segments(video_id)
            self.vector_index.replace_video(video_id, segments)
            if self.visual_index is not None:
                try:
                    self.repository.update_video(video_id, progress=0.94, stage="visual-index")
                    self.visual_index.replace_video(video_id, segments)
                except Exception as error:
                    warnings.append(f"siglip2: {error}")
            if self.moment_retriever is not None:
                try:
                    self.moment_retriever.prepare(video_id, source, probe.duration)
                except Exception as error:
                    warnings.append(f"lighthouse: {error}")
            self.repository.update_video(
                video_id,
                status="ready",
                progress=1.0,
                stage="ready",
                error="; ".join(warnings) or None,
            )
        except Exception as error:
            self.repository.update_video(
                video_id,
                status="failed",
                stage="failed",
                error=str(error),
            )
            raise
