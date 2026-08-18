from __future__ import annotations

import logging
import inspect
from pathlib import Path
import shutil
from threading import Event, Thread
from typing import Callable, Protocol
from uuid import uuid4

from videoscope.artifacts import (
    IndexingSpecifications,
    StageKind,
    StageRun,
    StageState,
    TextVectorBuildPlan,
    TextVectorBuildReceipt,
    TextVectorIndexSpecification,
    validate_artifact_identifier,
)
from videoscope.media.ffmpeg import FFmpeg
from videoscope.providers.scenes import SceneDetector
from videoscope.providers.types import ObjectTag, TimedText
from videoscope.repository import Repository, SegmentRecord
from videoscope.runtime_lifecycle import TextVectorStorageGate


logger = logging.getLogger(__name__)


class IndexingSpecificationChanged(RuntimeError):
    pass


class TextVectorLeaseLost(RuntimeError):
    pass


class _TextVectorBuildHeartbeat:
    def __init__(
        self,
        repository: Repository,
        generation_id: str,
        *,
        lease_seconds: int,
        interval: float,
    ) -> None:
        self.repository = repository
        self.generation_id = generation_id
        self.lease_seconds = lease_seconds
        self.interval = interval
        self._stop = Event()
        self._error: Exception | None = None
        self._thread = Thread(
            target=self._run,
            name=f"videoscope-vector-lease-{generation_id[:8]}",
            daemon=True,
        )

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.repository.heartbeat_text_vector_build(
                    self.generation_id,
                    lease_seconds=self.lease_seconds,
                )
            except Exception as error:
                self._error = error
                return

    def __enter__(self) -> _TextVectorBuildHeartbeat:
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore[no-untyped-def]
        self._stop.set()
        self._thread.join()
        if exc_type is None and self._error is not None:
            raise TextVectorLeaseLost("text vector build lease heartbeat failed") from self._error


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
    def transcribe(
        self,
        source: Path,
        *,
        prompt_snapshot: object | None = None,
    ) -> list[TimedText]: ...


class OCRProvider(Protocol):
    def read(self, image: Path) -> list[tuple[str, float]]: ...


class ObjectProvider(Protocol):
    def detect(self, image: Path) -> list[ObjectTag]: ...


class VectorIndex(Protocol):
    available: bool
    supports_generation_provenance: bool
    index_specification: TextVectorIndexSpecification

    def replace_video(self, video_id: str, segments: list[SegmentRecord]) -> None: ...

    def build_generation(self, plan: TextVectorBuildPlan) -> TextVectorBuildReceipt: ...


class DenseFrameExtractor(Protocol):
    def extract_frames(
        self,
        source: Path,
        destination: Path,
        start: float,
        end: float,
        *,
        step: float,
        max_width: int = 640,
    ) -> list[object]: ...


class VisualIndex(Protocol):
    def replace_video_source(
        self,
        video_id: str,
        source: Path,
        duration: float,
        extractor: DenseFrameExtractor,
        *,
        frames_dir: Path | None = None,
    ) -> str: ...


class MomentRetriever(Protocol):
    def prepare(self, video_id: str, source: Path, duration: float) -> None: ...


class Indexer:
    def __init__(
        self,
        *,
        repository: Repository,
        media_root: Path,
        thumbnails_dir: Path,
        specification_resolver: Callable[[], object],
        ffmpeg: FFmpeg,
        scenes: SceneDetector,
        vector_index: VectorIndex,
        visual_index: VisualIndex | None = None,
        speech: SpeechProvider | None = None,
        ocr: OCRProvider | None = None,
        objects: ObjectProvider | None = None,
        moment_retriever: MomentRetriever | None = None,
        text_vector_lease_seconds: int = 900,
        text_vector_heartbeat_interval: float = 30.0,
        text_vector_storage_gate: TextVectorStorageGate | None = None,
    ) -> None:
        if (
            isinstance(text_vector_lease_seconds, bool)
            or not isinstance(text_vector_lease_seconds, int)
            or not 1 <= text_vector_lease_seconds <= 86_400
        ):
            raise ValueError("text vector lease must be between 1 and 86400 seconds")
        if (
            isinstance(text_vector_heartbeat_interval, bool)
            or not isinstance(text_vector_heartbeat_interval, (int, float))
            or not 0 < float(text_vector_heartbeat_interval) < text_vector_lease_seconds
        ):
            raise ValueError("text vector heartbeat interval must be shorter than its lease")
        self.repository = repository
        self.media_root = Path(media_root)
        self.thumbnails_dir = thumbnails_dir
        self.specification_resolver = specification_resolver
        self.ffmpeg = ffmpeg
        self.scenes = scenes
        self.vector_index = vector_index
        self.visual_index = visual_index
        self.speech = speech
        self.ocr = ocr
        self.objects = objects
        self.moment_retriever = moment_retriever
        self.text_vector_lease_seconds = text_vector_lease_seconds
        self.text_vector_heartbeat_interval = float(text_vector_heartbeat_interval)
        self.text_vector_storage_gate = (
            text_vector_storage_gate or TextVectorStorageGate()
        )

    @staticmethod
    def _timed_text_segment(
        video_id: str,
        modality: str,
        item: TimedText,
    ) -> SegmentRecord | None:
        text = item.text.strip()
        if not text or item.end <= item.start:
            return None
        return SegmentRecord(
            id=uuid4().hex,
            video_id=video_id,
            start=item.start,
            end=item.end,
            modality=modality,
            text=text,
            confidence=item.confidence,
            metadata=item.metadata,
            thumbnail_path=None,
        )

    def _create_running_stage(
        self,
        video_id: str,
        specifications: IndexingSpecifications,
        kind: StageKind,
    ) -> StageRun:
        queued = self.repository.create_stage_run(
            video_id=video_id,
            specification=specifications.for_kind(kind),
        )
        return self.repository.transition_stage_run(queued.run_id, StageState.RUNNING)

    def _resolve_indexing_plan(self) -> tuple[IndexingSpecifications, object | None]:
        resolved = self.specification_resolver()
        if isinstance(resolved, IndexingSpecifications):
            return resolved, None
        specifications = getattr(resolved, "specifications", None)
        prompt_snapshot = getattr(resolved, "whisper_prompt_snapshot", None)
        if not isinstance(specifications, IndexingSpecifications):
            raise ValueError("indexing specification resolver returned invalid data")
        return specifications, prompt_snapshot

    def _verify_run_specification(self, run: StageRun) -> None:
        try:
            current, _prompt_snapshot = self._resolve_indexing_plan()
        except (TypeError, ValueError) as error:
            raise IndexingSpecificationChanged(
                "indexing specification resolver returned invalid data"
            ) from error
        if current.for_kind(run.stage_kind).specification_hash != run.specification_hash:
            raise IndexingSpecificationChanged("indexing specification changed during stage")

    def _record_not_configured(
        self,
        video_id: str,
        specifications: IndexingSpecifications,
        kind: StageKind,
    ) -> None:
        queued = self.repository.create_stage_run(
            video_id=video_id,
            specification=specifications.for_kind(kind),
        )
        self.repository.transition_stage_run(queued.run_id, StageState.NOT_CONFIGURED)

    def _record_failed(
        self,
        run: StageRun,
        error_code: str,
        *,
        error: Exception | None = None,
    ) -> None:
        if error is not None:
            logger.warning(
                "Indexing stage %s failed for %s",
                run.stage_kind.value,
                run.video_id,
                exc_info=error,
            )
        self.repository.transition_stage_run(
            run.run_id,
            StageState.FAILED,
            error_code=error_code,
        )

    def _validated_generations_root(self, video_id: str) -> Path:
        validate_artifact_identifier(video_id, field_name="thumbnail video id")
        configured_root = self.thumbnails_dir
        if configured_root.is_symlink():
            raise ValueError("thumbnail root must not be a symlink")
        configured_root.mkdir(parents=True, exist_ok=True)
        root = configured_root.resolve(strict=True)
        video_root_path = configured_root / video_id
        if video_root_path.is_symlink():
            raise ValueError("video thumbnail root must not be a symlink")
        video_root_path.mkdir(exist_ok=True)
        video_root = video_root_path.resolve(strict=True)
        if video_root.parent != root:
            raise ValueError("video thumbnail root escaped configured storage")
        generations_path = video_root_path / "generations"
        if generations_path.is_symlink():
            raise ValueError("thumbnail generations root must not be a symlink")
        generations_path.mkdir(exist_ok=True)
        generations_root = generations_path.resolve(strict=True)
        if generations_root.parent != video_root:
            raise ValueError("thumbnail generations root escaped video storage")
        return generations_root

    @staticmethod
    def _remove_generated_directory(path: Path, *, generations_root: Path) -> None:
        if generations_root.is_symlink() or path.is_symlink():
            raise ValueError("refusing to remove a symlinked thumbnail directory")
        try:
            resolved_root = generations_root.resolve(strict=True)
        except OSError as error:
            raise ValueError("thumbnail generations root is unavailable") from error
        if path.parent != generations_root or not path.name:
            raise ValueError("refusing to remove an unexpected thumbnail directory")
        if path.exists():
            try:
                resolved = path.resolve(strict=True)
            except OSError as error:
                raise ValueError("generated thumbnail directory is unavailable") from error
            if resolved.parent != resolved_root or resolved == resolved_root:
                raise ValueError("generated thumbnail directory escaped storage")
            shutil.rmtree(resolved)

    def _index_scenes(
        self,
        video_id: str,
        source: Path,
        duration: float,
        specifications: IndexingSpecifications,
        warnings: list[str],
    ) -> tuple[bool, list[tuple[float, float, Path]]]:
        run = self._create_running_stage(video_id, specifications, StageKind.SCENES)
        try:
            scene_intervals = self.scenes.detect(source, duration)
        except Exception as error:
            self._record_failed(run, "scene_provider_failed", error=error)
            warnings.append("scenes stage failed")
            return False, []

        generation_id = uuid4().hex
        validate_artifact_identifier(
            generation_id,
            field_name="scene generation id",
        )
        generations_root = self.thumbnails_dir / video_id / "generations"
        temporary_dir = generations_root / f".{generation_id}.tmp"
        final_dir = generations_root / generation_id
        published = False
        try:
            generations_root = self._validated_generations_root(video_id)
            temporary_dir = generations_root / f".{generation_id}.tmp"
            final_dir = generations_root / generation_id
            temporary_dir.mkdir(exist_ok=False)
            candidates: list[SegmentRecord] = []
            frame_paths: list[tuple[float, float, Path]] = []
            for index, (start, end) in enumerate(scene_intervals):
                filename = f"scene-{index:04d}.jpg"
                temporary_path = temporary_dir / filename
                final_path = final_dir / filename
                self.ffmpeg.extract_frame(
                    source,
                    temporary_path,
                    start + (end - start) / 2,
                )
                candidates.append(
                    SegmentRecord(
                        id=uuid4().hex,
                        video_id=video_id,
                        start=start,
                        end=end,
                        modality="scene",
                        text=f"scene {index + 1}",
                        confidence=1.0,
                        metadata={"scene_index": index},
                        thumbnail_path=str(final_path),
                    )
                )
                frame_paths.append((start, end, final_path))
            self._verify_run_specification(run)
            temporary_dir.replace(final_dir)
            published = True
            self.repository.commit_segment_generation(
                run.run_id,
                segments=candidates,
                generation_id=generation_id,
                video_thumbnail_path=(
                    str(frame_paths[0][2]) if frame_paths else None
                ),
            )
        except Exception as error:
            cleanup_target = final_dir if published else temporary_dir
            try:
                self._remove_generated_directory(
                    cleanup_target,
                    generations_root=generations_root,
                )
            except (OSError, ValueError):
                logger.error(
                    "Refused unsafe scene generation cleanup for %s",
                    video_id,
                    exc_info=True,
                )
            error_code = (
                "scene_specification_changed"
                if isinstance(error, IndexingSpecificationChanged)
                else "scene_build_failed"
            )
            self._record_failed(run, error_code, error=error)
            warnings.append("scenes stage failed")
            return False, []

        return True, frame_paths

    def _index_speech(
        self,
        video_id: str,
        source: Path,
        specifications: IndexingSpecifications,
        prompt_snapshot: object | None,
        warnings: list[str],
    ) -> None:
        if self.speech is None:
            self._record_not_configured(video_id, specifications, StageKind.SPEECH)
            return
        run = self._create_running_stage(video_id, specifications, StageKind.SPEECH)
        try:
            transcribe = self.speech.transcribe
            accepts_prompt_snapshot = "prompt_snapshot" in inspect.signature(
                transcribe
            ).parameters
            transcribed = (
                transcribe(source, prompt_snapshot=prompt_snapshot)
                if accepts_prompt_snapshot
                else transcribe(source)
            )
            candidates = [
                segment
                for item in merge_timed_text(transcribed)
                if (segment := self._timed_text_segment(video_id, "speech", item))
                is not None
            ]
            self._verify_run_specification(run)
            self.repository.commit_segment_generation(
                run.run_id,
                segments=candidates,
            )
        except Exception as error:
            error_code = (
                "speech_specification_changed"
                if isinstance(error, IndexingSpecificationChanged)
                else "speech_provider_failed"
            )
            self._record_failed(run, error_code, error=error)
            warnings.append("speech stage failed")

    def _index_ocr(
        self,
        video_id: str,
        frame_paths: list[tuple[float, float, Path]],
        *,
        scenes_available: bool,
        specifications: IndexingSpecifications,
        warnings: list[str],
    ) -> None:
        if self.ocr is None:
            self._record_not_configured(video_id, specifications, StageKind.OCR)
            return
        run = self._create_running_stage(video_id, specifications, StageKind.OCR)
        if not scenes_available:
            self._record_failed(run, "scene_dependency_unavailable")
            warnings.append("ocr stage failed")
            return
        try:
            candidates: list[SegmentRecord] = []
            for index, (start, end, frame_path) in enumerate(frame_paths):
                unique: dict[str, tuple[str, float]] = {}
                for text, confidence in self.ocr.read(frame_path):
                    normalized = " ".join(text.split()).strip()
                    key = normalized.casefold()
                    if len(key) < 2:
                        continue
                    current = unique.get(key)
                    if current is None or confidence > current[1]:
                        unique[key] = (normalized, confidence)
                if not unique:
                    continue
                ordered = list(unique.values())
                candidates.append(
                    SegmentRecord(
                        id=uuid4().hex,
                        video_id=video_id,
                        start=start,
                        end=end,
                        modality="ocr",
                        text=" · ".join(text for text, _ in ordered),
                        confidence=sum(confidence for _, confidence in ordered)
                        / len(ordered),
                        metadata={"scene_index": index, "line_count": len(ordered)},
                        thumbnail_path=str(frame_path),
                    )
                )
            self._verify_run_specification(run)
            self.repository.commit_segment_generation(run.run_id, segments=candidates)
        except Exception as error:
            error_code = (
                "ocr_specification_changed"
                if isinstance(error, IndexingSpecificationChanged)
                else "ocr_provider_failed"
            )
            self._record_failed(run, error_code, error=error)
            warnings.append("ocr stage failed")

    def _index_objects(
        self,
        video_id: str,
        frame_paths: list[tuple[float, float, Path]],
        *,
        scenes_available: bool,
        specifications: IndexingSpecifications,
        warnings: list[str],
    ) -> None:
        if self.objects is None:
            self._record_not_configured(video_id, specifications, StageKind.OBJECTS)
            return
        run = self._create_running_stage(video_id, specifications, StageKind.OBJECTS)
        if not scenes_available:
            self._record_failed(run, "scene_dependency_unavailable")
            warnings.append("objects stage failed")
            return
        try:
            candidates: list[SegmentRecord] = []
            for index, (start, end, frame_path) in enumerate(frame_paths):
                tags = self.objects.detect(frame_path)
                if not tags:
                    continue
                candidates.append(
                    SegmentRecord(
                        id=uuid4().hex,
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
                )
            self._verify_run_specification(run)
            self.repository.commit_segment_generation(run.run_id, segments=candidates)
        except Exception as error:
            error_code = (
                "objects_specification_changed"
                if isinstance(error, IndexingSpecificationChanged)
                else "objects_provider_failed"
            )
            self._record_failed(run, error_code, error=error)
            warnings.append("objects stage failed")

    def _replace_text_vectors(
        self,
        video_id: str,
        specifications: IndexingSpecifications,
        warnings: list[str],
    ) -> None:
        try:
            current, _prompt_snapshot = self._resolve_indexing_plan()
        except Exception as error:
            logger.warning(
                "Text vector specification resolution failed for %s",
                video_id,
                exc_info=error,
            )
            warnings.append("text vector stage failed")
            return
        if not bool(getattr(self.vector_index, "available", True)):
            self._record_not_configured(
                video_id,
                current,
                StageKind.TEXT_VECTORS,
            )
            warnings.append("text vector stage not configured")
            return

        generation_capable = bool(
            getattr(self.vector_index, "supports_generation_provenance", False)
        ) and all(
            callable(getattr(self.vector_index, name, None))
            for name in ("build_generation", "validate_generation")
        )
        if generation_capable:
            run = self._create_running_stage(
                video_id,
                current,
                StageKind.TEXT_VECTORS,
            )
            with self.text_vector_storage_gate.build_activity():
                self._build_text_vector_generation(
                    video_id,
                    current,
                    run,
                    warnings,
                )
            return

        # Compatibility seam for old test/in-process indexes. The write remains
        # deliberately untrusted and the run is terminal FAILED, never COMPLETE.
        run = self._create_running_stage(
            video_id,
            current,
            StageKind.TEXT_VECTORS,
        )
        try:
            segments = self.repository.list_current_active_segments(
                current.semantic_segment_specifications,
                video_ids=[video_id],
            )
            self.vector_index.replace_video(video_id, segments)
            self._record_failed(run, "text_vector_generation_unverified")
            warnings.append("text vector generation unverified")
        except Exception as error:
            self._record_failed(run, "text_vector_index_failed", error=error)
            warnings.append("text vector stage failed")

    def _build_text_vector_generation(
        self,
        video_id: str,
        specifications: IndexingSpecifications,
        run: StageRun,
        warnings: list[str],
    ) -> None:
        plan: TextVectorBuildPlan | None = None
        try:
            index_specification = getattr(
                self.vector_index,
                "index_specification",
                None,
            )
            if not isinstance(index_specification, TextVectorIndexSpecification):
                raise ValueError("vector writer has no validated physical identity")
            plan = self.repository.reserve_text_vector_generation(
                run.run_id,
                index_specification=index_specification,
                semantic_specifications=specifications.semantic_segment_specifications,
                lease_seconds=self.text_vector_lease_seconds,
            )
            with _TextVectorBuildHeartbeat(
                self.repository,
                plan.generation_id,
                lease_seconds=self.text_vector_lease_seconds,
                interval=self.text_vector_heartbeat_interval,
            ):
                receipt = self.vector_index.build_generation(plan)
            if not isinstance(receipt, TextVectorBuildReceipt):
                raise ValueError("vector writer returned an invalid generation receipt")
            self._verify_run_specification(run)
            self.repository.commit_text_vector_generation(
                run.run_id,
                receipt=receipt,
            )
        except Exception as error:
            message = str(error)
            if isinstance(error, IndexingSpecificationChanged):
                error_code = "text_vector_specification_changed"
            elif isinstance(error, TextVectorLeaseLost):
                error_code = "text_vector_lease_lost"
            elif (
                "inputs changed" in message
                or "active text vector generation changed" in message
            ):
                error_code = "text_vector_inputs_changed"
            elif isinstance(error, ValueError) and (
                "receipt" in message or "manifest" in message
            ):
                error_code = "text_vector_generation_invalid"
            else:
                error_code = "text_vector_index_failed"
            logger.warning(
                "Immutable text vector build failed for %s",
                video_id,
                exc_info=error,
            )
            if plan is not None:
                try:
                    self.repository.fail_text_vector_build(
                        run.run_id,
                        error_code=error_code,
                    )
                except Exception as cleanup_error:
                    logger.warning(
                        "Text vector build cleanup was deferred for %s",
                        video_id,
                        exc_info=cleanup_error,
                    )
                    persisted = self.repository.get_stage_run(run.run_id)
                    if persisted is not None and persisted.state is StageState.RUNNING:
                        self._record_failed(run, error_code)
            else:
                self._record_failed(run, error_code)
            warnings.append("text vector stage failed")

    def process(self, video_id: str) -> None:
        video = self.repository.get_video(video_id)
        if video is None:
            raise KeyError(video_id)
        source = Path(video.media_path)
        warnings: list[str] = []
        self.repository.update_video(video_id, status="processing", progress=0.01, stage="probe", error=None)

        try:
            self.repository.verify_asset_identity(video_id, media_root=self.media_root)
            specifications, whisper_prompt_snapshot = self._resolve_indexing_plan()
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
            scenes_available, frame_paths = self._index_scenes(
                video_id,
                source,
                probe.duration,
                specifications,
                warnings,
            )
            self.repository.update_video(
                video_id,
                progress=0.25,
                stage="speech",
            )

            self._index_speech(
                video_id,
                source,
                specifications,
                whisper_prompt_snapshot,
                warnings,
            )

            self.repository.update_video(video_id, progress=0.45, stage="vision")
            self._index_ocr(
                video_id,
                frame_paths,
                scenes_available=scenes_available,
                specifications=specifications,
                warnings=warnings,
            )
            self._index_objects(
                video_id,
                frame_paths,
                scenes_available=scenes_available,
                specifications=specifications,
                warnings=warnings,
            )
            self.repository.update_video(video_id, progress=0.85, stage="vision")

            self.repository.update_video(video_id, progress=0.9, stage="index")
            self._replace_text_vectors(video_id, specifications, warnings)
            if self.visual_index is not None:
                try:
                    self.repository.update_video(video_id, progress=0.94, stage="visual-index")
                    self.visual_index.replace_video_source(
                        video_id,
                        source,
                        probe.duration,
                        self.ffmpeg,
                        frames_dir=self.thumbnails_dir / video_id,
                    )
                except Exception as error:
                    logger.warning("Dense visual indexing failed for %s", video_id, exc_info=error)
                    warnings.append("dense visual stage failed")
            if self.moment_retriever is not None:
                try:
                    self.moment_retriever.prepare(video_id, source, probe.duration)
                except Exception as error:
                    logger.warning("Lighthouse indexing failed for %s", video_id, exc_info=error)
                    warnings.append("lighthouse stage failed")
            if warnings:
                logger.warning("Optional indexing stages failed for %s: %s", video_id, "; ".join(warnings))
            self.repository.update_video(
                video_id,
                status="ready",
                progress=1.0,
                stage="ready",
                error="; ".join(warnings) or None,
            )
        except Exception as error:
            logger.exception("Video processing failed for %s", video_id)
            self.repository.update_video(
                video_id,
                status="failed",
                stage="failed",
                error=str(error),
            )
            raise
