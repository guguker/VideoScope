from __future__ import annotations

import hashlib
import logging
import shutil
from dataclasses import dataclass, field
from threading import Event, Lock, Thread
from time import sleep

from videoscope.artifacts import IndexingSpecifications, StageKind, StageSpecification
from videoscope.clips import ClipService
from videoscope.config import AppSettings
from videoscope.media.ffmpeg import FFmpeg
from videoscope.model_manifest import (
    FASTEMBED_ALGORITHM_VERSION,
    QWEN_VIDEO_MODEL,
    fastembed_snapshot,
    model_identity,
    model_revision,
)
from videoscope.processing.indexer import Indexer
from videoscope.processing.queue import ThreadedProcessingQueue
from videoscope.providers.base import ProviderRegistry, ProviderState, StaticProvider
from videoscope.providers.lighthouse import (
    DisabledLighthouseRetriever,
    LighthouseRetriever,
)
from videoscope.providers.lighthouse_worker import LighthouseWorkerClient
from videoscope.providers.internvideo import InternVideoReranker
from videoscope.providers.paddle_ocr import PaddleOCRReader
from videoscope.providers.qwen_video import QwenVideoReranker
from videoscope.providers.qwen_worker import QwenWorkerClient
from videoscope.providers.roboflow import RoboflowDetector
from videoscope.providers.scenes import SceneDetector
from videoscope.providers.whisper import WhisperTranscriber, resolve_whisper_prompt
from videoscope.repository import Repository
from videoscope.runtime_lifecycle import (
    ArtifactGarbageCollector,
    ExclusiveRuntimeLock,
    TextVectorStorageGate,
)
from videoscope.search.service import SearchService
from videoscope.search.text_matching import SearchLexicon
from videoscope.search.temporal_refinement import TemporalRefiner
from videoscope.search.embeddings import create_semantic_embedding
from videoscope.search.vector_index import EmptyVectorIndex, QdrantVectorIndex
from videoscope.search.visual_index import SiglipVisualIndex


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Runtime:
    queue: ThreadedProcessingQueue
    search: SearchService
    clips: ClipService
    providers: ProviderRegistry
    runtime_lock: ExclusiveRuntimeLock
    artifact_gc: ArtifactGarbageCollector
    generation_store: QdrantVectorIndex
    resume_video_ids: tuple[str, ...] = ()
    _started: bool = field(default=False, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _shutdown_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _shutdown_complete: Event = field(default_factory=Event, init=False, repr=False)
    _shutdown_reaper: Thread | None = field(default=None, init=False, repr=False)
    _generation_store_closed: bool = field(default=False, init=False, repr=False)

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("runtime is closed")
        if self._started:
            raise RuntimeError("runtime is already started")
        try:
            if not getattr(self.runtime_lock, "is_acquired", False):
                self.runtime_lock.acquire()
            self.artifact_gc.recover_startup()
            self.artifact_gc.start()
            self.queue.start()
            for video_id in self.resume_video_ids:
                self.queue.submit(video_id)
        except Exception:
            self.close()
            raise
        self._started = True

    @staticmethod
    def _stop_worker(worker: object, *, timeout: float | None) -> bool:
        try:
            return bool(worker.close(timeout=timeout))  # type: ignore[attr-defined]
        except Exception:
            logger.exception("Runtime worker shutdown failed")
            return False

    def _finalize_ownership(self) -> bool:
        if self._shutdown_complete.is_set():
            return True
        if not self._generation_store_closed:
            try:
                self.generation_store.close()
            except Exception:
                logger.exception(
                    "Runtime generation store close failed; ownership retained"
                )
                return False
            self._generation_store_closed = True
        try:
            self.runtime_lock.close()
        except Exception:
            logger.exception("Runtime storage lock release failed")
            return False
        self._closed = True
        self._shutdown_complete.set()
        return True

    def _reap_shutdown(self) -> None:
        while not self._stop_worker(self.queue, timeout=None):
            sleep(0.1)
        while not self._stop_worker(self.artifact_gc, timeout=None):
            sleep(0.1)
        while True:
            with self._shutdown_lock:
                if self._finalize_ownership():
                    return
            sleep(0.1)

    def _start_shutdown_reaper_locked(self) -> bool:
        try:
            reaper = Thread(
                target=self._reap_shutdown,
                name="videoscope-runtime-reaper",
                daemon=True,
            )
            self._shutdown_reaper = reaper
            reaper.start()
        except Exception:
            self._shutdown_reaper = None
            logger.exception("Runtime ownership reaper could not start")
            return False
        return True

    def close(self) -> bool:
        reap_synchronously = False
        with self._shutdown_lock:
            if self._shutdown_complete.is_set():
                return True
            if self._shutdown_reaper is not None:
                return False
            queue_stopped = self._stop_worker(self.queue, timeout=5)
            gc_stopped = self._stop_worker(self.artifact_gc, timeout=5)
            if queue_stopped and gc_stopped:
                if self._finalize_ownership():
                    return True
                logger.warning(
                    "Runtime ownership finalization will continue in the reaper"
                )
            else:
                logger.warning(
                    "Runtime workers are still stopping; ownership reaper retained storage"
                )
            reap_synchronously = not self._start_shutdown_reaper_locked()
        if reap_synchronously:
            self._reap_shutdown()
            return True
        return False


def create_indexing_specifications(settings: AppSettings) -> IndexingSpecifications:
    """Snapshot every configuration input that changes trusted SQLite/text output."""
    effective_prompt, glossary_state = resolve_whisper_prompt(
        settings.whisper_initial_prompt,
        settings.glossary_path,
    )
    prompt_digest = hashlib.sha256(
        (effective_prompt or "").encode("utf-8")
    ).hexdigest()
    embedding_repository, embedding_revision = fastembed_snapshot(
        settings.text_embedding_model
    )
    embedding_identity = model_identity(
        embedding_repository or settings.text_embedding_model,
        embedding_revision,
    )
    segment_schema = "sqlite-segment-v1"
    scenes = StageSpecification(
        kind=StageKind.SCENES,
        schema_version=1,
        implementation_revision="videoscope.scene-segments.v1",
        parameters={
            "detector": "adaptive",
            "max_scene_seconds": settings.max_scene_seconds,
            "scene_threshold": settings.scene_threshold,
            "segment_schema": segment_schema,
            "thumbnail_format": "jpeg",
            "thumbnail_position": "midpoint",
        },
        dependencies={"provider": "pyscenedetect-adaptive"},
    )
    speech = StageSpecification(
        kind=StageKind.SPEECH,
        schema_version=1,
        implementation_revision="videoscope.speech-segments.v1",
        parameters={
            "effective_prompt_sha256": prompt_digest,
            "glossary_state": glossary_state,
            "language": settings.whisper_language,
            "merge_max_duration_seconds": 14.0,
            "merge_max_gap_seconds": 1.25,
            "segment_schema": segment_schema,
        },
        model_identity=model_identity(
            settings.whisper_model,
            model_revision(settings.whisper_model),
        ),
        dependencies={"provider": "mlx-whisper"},
    )
    ocr = StageSpecification(
        kind=StageKind.OCR,
        schema_version=1,
        implementation_revision="videoscope.ocr-segments.v1",
        parameters={
            "deduplication": "casefold-highest-confidence-v1",
            "minimum_confidence": 0.55,
            "segment_schema": segment_schema,
            "worker_protocol": "json-lines-v1",
        },
        model_identity="paddleocr/default-transformers",
        dependencies={
            "provider": "paddleocr",
            "scenes_specification": scenes.specification_hash,
        },
    )
    objects = StageSpecification(
        kind=StageKind.OBJECTS,
        schema_version=1,
        implementation_revision="videoscope.object-segments.v1",
        parameters={
            "credential_configured": bool(settings.roboflow_api_key),
            "minimum_confidence": 0.25,
            "segment_schema": segment_schema,
        },
        model_identity=f"roboflow/{settings.roboflow_model_id or 'unconfigured'}",
        dependencies={
            "provider": "roboflow",
            "scenes_specification": scenes.specification_hash,
        },
    )
    text_vectors = StageSpecification(
        kind=StageKind.TEXT_VECTORS,
        schema_version=1,
        implementation_revision="videoscope.text-vectors.v2",
        parameters={
            "dimensions": settings.text_embedding_dimensions,
            "embedding_algorithm": FASTEMBED_ALGORITHM_VERSION,
            "segment_schema": segment_schema,
            "semantic_modalities": ["speech", "ocr", "objects"],
        },
        model_identity=embedding_identity,
        dependencies={
            "objects_specification": objects.specification_hash,
            "ocr_specification": ocr.specification_hash,
            "provider": "qdrant",
            "speech_specification": speech.specification_hash,
        },
    )
    return IndexingSpecifications(
        scenes=scenes,
        speech=speech,
        ocr=ocr,
        objects=objects,
        text_vectors=text_vectors,
    )


def create_visual_index(settings: AppSettings) -> SiglipVisualIndex:
    """Build the one canonical dense visual writer used by runtime and backfill."""
    return SiglipVisualIndex(
        settings.visual_index_dir,
        model_name=settings.siglip_model,
        model_revision=model_revision(settings.siglip_model),
        batch_size=settings.siglip_batch_size,
        sample_step=settings.visual_index_step,
        max_width=settings.visual_index_max_width,
    )


def create_qwen_reranker(
    settings: AppSettings,
    repository: Repository,
    extractor: FFmpeg,
) -> QwenVideoReranker:
    """Keep worker wiring explicit and free of MLX imports in the backend."""
    qwen_inference = (
        QwenWorkerClient(
            endpoint=settings.qwen_video_endpoint,
            api_key=settings.qwen_video_api_key or "",
            input_root=settings.temp_dir,
            expected_model_identity=model_identity(
                settings.qwen_video_model or QWEN_VIDEO_MODEL,
                model_revision(settings.qwen_video_model or QWEN_VIDEO_MODEL),
            ),
            timeout=settings.qwen_video_timeout,
        )
        if settings.qwen_video_endpoint
        else None
    )
    return QwenVideoReranker(
        model_name=settings.qwen_video_model,
        model_revision=model_revision(settings.qwen_video_model),
        repository=repository,
        extractor=extractor,
        temp_dir=settings.temp_dir,
        cache_dir=settings.cache_dir / "qwen-video",
        top_candidates=settings.qwen_video_top_candidates,
        context_seconds=settings.qwen_video_context_seconds,
        min_clip_seconds=settings.qwen_video_min_clip_seconds,
        max_clip_seconds=settings.qwen_video_max_clip_seconds,
        frame_count=settings.qwen_video_frame_count,
        video_fps=settings.qwen_video_fps,
        inference_client=qwen_inference,
        allow_in_process=settings.qwen_video_allow_in_process,
    )


def create_lighthouse_retriever(
    settings: AppSettings,
    ffmpeg: FFmpeg,
) -> LighthouseWorkerClient | LighthouseRetriever | DisabledLighthouseRetriever:
    """Keep Lighthouse ML imports outside the default backend process."""
    if settings.lighthouse_endpoint:
        return LighthouseWorkerClient(
            endpoint=settings.lighthouse_endpoint,
            api_key=settings.lighthouse_api_key or "",
            input_root=settings.media_dir,
            cache_dir=settings.cache_dir,
            timeout=settings.lighthouse_timeout,
        )
    if settings.lighthouse_allow_in_process:
        return LighthouseRetriever(
            checkpoint=settings.lighthouse_checkpoint,
            cache_dir=settings.cache_dir,
            ffmpeg=ffmpeg,
            source_root=settings.lighthouse_root,
        )
    return DisabledLighthouseRetriever()


def build_runtime(settings: AppSettings, repository: Repository) -> Runtime:
    ffmpeg = FFmpeg()
    scenes = SceneDetector(
        threshold=settings.scene_threshold,
        max_scene_seconds=settings.max_scene_seconds,
    )
    whisper = WhisperTranscriber(
        settings.whisper_model,
        settings.whisper_language,
        settings.whisper_initial_prompt,
        settings.glossary_path,
        model_revision=model_revision(settings.whisper_model),
    )
    ocr = PaddleOCRReader(
        worker_python=settings.ocr_worker_python,
        worker_script=settings.ocr_worker_script,
    )
    roboflow = RoboflowDetector(
        api_key=settings.roboflow_api_key,
        model_id=settings.roboflow_model_id,
        cache_dir=settings.models_dir / "rfdetr",
    )
    lighthouse = create_lighthouse_retriever(settings, ffmpeg)
    text_embedding = create_semantic_embedding(
        model_name=settings.text_embedding_model,
        dimensions=settings.text_embedding_dimensions,
        cache_dir=settings.models_dir / "fastembed",
    )
    qdrant = QdrantVectorIndex(settings.qdrant_dir / "text", embedding=text_embedding)
    siglip = create_visual_index(settings)
    internvideo = InternVideoReranker(
        endpoint=settings.internvideo_endpoint,
        api_key=settings.internvideo_api_key,
        repository=repository,
        extractor=ffmpeg,
        temp_dir=settings.temp_dir,
        top_candidates=settings.internvideo_top_candidates,
        timeout=settings.internvideo_timeout,
    )
    qwen_video = create_qwen_reranker(settings, repository, ffmpeg)

    qdrant_ready = qdrant.status().state is ProviderState.READY
    vector_index = qdrant if qdrant_ready else EmptyVectorIndex()
    speech_provider = whisper if whisper.status().state is ProviderState.READY else None
    ocr_provider = ocr if ocr.status().state is ProviderState.READY else None
    object_provider = roboflow if roboflow.status().state is ProviderState.READY else None
    moment_provider = lighthouse if lighthouse.status().state is ProviderState.READY else None
    # Stale/legacy generations must remain readable only as unavailable evidence,
    # but they must not disable the writer needed by ingest/API reindex to replace
    # them with the canonical dense specification.
    visual_provider = (
        siglip
        if siglip.status(check_index=False).state is ProviderState.READY
        else None
    )
    candidate_reranker = (
        qwen_video
        if qwen_video.status().state is ProviderState.READY
        else internvideo
        if internvideo.status().state is ProviderState.READY
        else None
    )
    temporal_refiner = (
        TemporalRefiner(
            repository=repository,
            extractor=ffmpeg,
            scorer=siglip,
            temp_dir=settings.temp_dir,
            top_candidates=settings.temporal_refinement_candidates,
            sample_step=settings.temporal_refinement_step,
            min_score=settings.visual_min_score,
        )
        if visual_provider is not None
        else None
    )

    text_vector_storage_gate = TextVectorStorageGate()
    indexer = Indexer(
        repository=repository,
        media_root=settings.media_dir,
        thumbnails_dir=settings.thumbnails_dir,
        specification_resolver=lambda: create_indexing_specifications(settings),
        ffmpeg=ffmpeg,
        scenes=scenes,
        speech=speech_provider,
        ocr=ocr_provider,
        objects=object_provider,
        vector_index=vector_index,
        visual_index=visual_provider,
        moment_retriever=moment_provider,
        text_vector_storage_gate=text_vector_storage_gate,
    )
    queue = ThreadedProcessingQueue(indexer, start_immediately=False)
    resume_video_ids = tuple(
        video.id
        for video in repository.list_videos()
        if video.status in {"queued", "processing"}
    )

    ffmpeg_binary = shutil.which("ffmpeg")
    ffmpeg_status = StaticProvider(
        "ffmpeg",
        "FFmpeg",
        ProviderState.READY if ffmpeg_binary else ProviderState.UNAVAILABLE,
        "ffmpeg executable is available" if ffmpeg_binary else "ffmpeg executable is missing",
    )
    providers = ProviderRegistry(
        [
            ffmpeg_status,
            scenes,
            whisper,
            ocr,
            roboflow,
            lighthouse,
            qdrant,
            siglip,
            qwen_video,
            internvideo,
        ]
    )
    return Runtime(
        queue=queue,
        search=SearchService(
            repository,
            vector_index,
            moment_provider,
            visual_provider,
            lexicon=SearchLexicon(settings.glossary_path),
            temporal_refiner=temporal_refiner,
            candidate_reranker=candidate_reranker,
            semantic_text_min_score=settings.semantic_text_min_score,
            visual_min_score=settings.visual_min_score,
            specification_resolver=lambda: create_indexing_specifications(settings),
            thumbnails_dir=settings.thumbnails_dir,
        ),
        clips=ClipService(repository, ffmpeg, settings.clips_dir, settings.temp_dir),
        providers=providers,
        runtime_lock=ExclusiveRuntimeLock(settings.data_dir),
        artifact_gc=ArtifactGarbageCollector(
            repository,
            qdrant,
            storage_gate=text_vector_storage_gate,
        ),
        generation_store=qdrant,
        resume_video_ids=resume_video_ids,
    )
