from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import logging
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread
from time import sleep
from typing import Callable

from videoscope.artifacts import (
    IndexingSpecifications,
    StageKind,
    StageSpecification,
    TextVectorIndexSpecification,
)
from videoscope.clips import ClipService
from videoscope.config import AppSettings
from videoscope.jobs import VideoIndexPlanSnapshot
from videoscope.indexing_attestation import (
    AttestedIndexingToolchain,
    attest_indexing_toolchain,
)
from videoscope.media.ffmpeg import FFmpeg
from videoscope.model_manifest import (
    FASTEMBED_ALGORITHM_VERSION,
    QWEN_VIDEO_MODEL,
    fastembed_snapshot,
    model_identity,
    model_revision,
)
from videoscope.processing.indexer import Indexer
from videoscope.processing.dispatcher import DurableVideoIndexDispatcher
from videoscope.providers.base import ProviderRegistry, ProviderState, StaticProvider
from videoscope.providers.lighthouse import (
    DisabledLighthouseRetriever,
    LighthouseRetriever,
)
from videoscope.providers.lighthouse_worker import (
    DEFAULT_LIGHTHOUSE_SPECIFICATION,
    LIGHTHOUSE_MODEL_IDENTITY,
    LIGHTHOUSE_WORKER_RUNTIME_IDENTITY,
    LIGHTHOUSE_WORKER_SCHEMA_VERSION,
    LighthouseSpecification,
    LighthouseWorkerClient,
)
from videoscope.providers.internvideo import InternVideoReranker
from videoscope.providers.paddle_ocr import (
    OCR_MODEL_ARTIFACT_IDENTITY,
    OCR_WORKER_DEPENDENCY_IDENTITY,
    OCR_WORKER_PROTOCOL,
    OCR_WORKER_RUNTIME_IDENTITY,
    PaddleOCRReader,
)
from videoscope.providers.qwen_video import QwenVideoReranker
from videoscope.providers.qwen_worker import QwenWorkerClient
from videoscope.providers.roboflow import RoboflowDetector
from videoscope.providers.scenes import SceneDetector
from videoscope.providers.whisper import (
    WhisperPromptSnapshot,
    WhisperTranscriber,
    snapshot_whisper_prompt,
)
from videoscope.providers.whisper_worker import (
    WHISPER_DEPENDENCY_IDENTITY,
    WHISPER_INFERENCE_RUNTIME_IDENTITY,
    WHISPER_WORKER_SCHEMA_VERSION,
    WhisperWorkerClient,
)
from videoscope.repository import Repository
from videoscope.runtime_lifecycle import (
    ArtifactGarbageCollector,
    ExclusiveRuntimeLock,
    TextVectorStorageGate,
)
from videoscope.search.service import SearchService
from videoscope.search.text_matching import SearchLexicon
from videoscope.search.temporal_refinement import TemporalRefiner
from videoscope.search.embeddings import (
    ReviewedSemanticEmbeddingContract,
    ReviewedSemanticEmbeddingError,
    ReviewedSemanticEmbeddingRuntime,
    SemanticEmbedding,
    UnavailableReviewedSemanticEmbedding,
    create_reviewed_semantic_embedding_runtime,
    create_semantic_embedding,
    load_reviewed_semantic_embedding_contract,
)
from videoscope.search.vector_index import EmptyVectorIndex, QdrantVectorIndex
from videoscope.search.visual_index import (
    VISUAL_INDEX_EXTRACTOR_IDENTITY,
    SiglipVisualIndex,
    VisualIndexSpecification,
)
from videoscope.providers.vision_worker_client import VisionWorkerClient
from videoscope.providers.vision_worker_contract import (
    REVIEWED_SIGLIP_PROFILES,
    VISION_WORKER_SCHEMA_VERSION,
    VisionWorkerSpecification,
)


logger = logging.getLogger(__name__)


VIDEO_INDEX_PLAN_SCHEMA_VERSION = 2
VIDEO_INDEX_EXECUTOR_SCHEMA_VERSION = 1
VIDEO_INDEXER_IMPLEMENTATION_REVISION = "videoscope.indexer.v1"
FFMPEG_INDEXING_CONTRACT_REVISION = "videoscope.ffmpeg-indexing.v1"
TEXT_VECTOR_WRITER_IMPLEMENTATION_REVISION = (
    "videoscope.qdrant-generation-writer.v1"
)
OCR_WORKER_ADAPTER_REVISION = "videoscope.paddleocr-jsonl-adapter.v2"
_OCR_DEPENDENCY_IDENTITY_RE = re.compile(
    r"^paddleocr-deps-v1:sha256:[0-9a-f]{64}$"
)
_OCR_RUNTIME_IDENTITY_RE = re.compile(
    r"^videoscope-paddleocr-worker-v1\|python==3\.12\.13\|"
    r"platform==aarch64-apple-darwin-macos14plus\|"
    r"lock-sha256:[0-9a-f]{64}$"
)
_OCR_MODEL_IDENTITY_RE = re.compile(
    r"^paddleocr-models-v1:sha256:[0-9a-f]{64}$"
)
_MAX_OCR_WORKER_SCRIPT_BYTES = 1024 * 1024
_INDEXING_TOOLCHAIN_IDENTITY_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _open_ocr_script_parent(path: Path) -> tuple[Path, int]:
    """Open a script parent without following any component symlink."""
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if not nofollow or not directory:
        raise ValueError("video index plan requires a reviewed OCR worker script")
    absolute = Path(os.path.abspath(path))
    if not absolute.name:
        raise ValueError("video index plan requires a reviewed OCR worker script")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | directory | nofollow
    try:
        descriptor = os.open(absolute.anchor, flags)
        try:
            for part in absolute.parts[1:-1]:
                child = os.open(part, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
        except BaseException:
            os.close(descriptor)
            raise
    except OSError as error:
        raise ValueError(
            "video index plan requires a reviewed OCR worker script"
        ) from error
    return absolute, descriptor


def _ocr_script_fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _reviewed_ocr_script_sha256(path: Path) -> str:
    """Hash one bounded stable regular script through a no-follow chain."""
    absolute, parent_descriptor = _open_ocr_script_parent(path)
    file_descriptor = -1
    try:
        try:
            lexical = os.stat(
                absolute.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(lexical.st_mode)
                or lexical.st_nlink != 1
                or not 1 <= lexical.st_size <= _MAX_OCR_WORKER_SCRIPT_BYTES
            ):
                raise ValueError(
                    "video index plan requires a reviewed OCR worker script"
                )
            file_descriptor = os.open(
                absolute.name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_descriptor,
            )
            before = os.fstat(file_descriptor)
            if _ocr_script_fingerprint(before) != _ocr_script_fingerprint(lexical):
                raise ValueError(
                    "video index plan requires a reviewed OCR worker script"
                )
            digest = hashlib.sha256()
            total = 0
            while True:
                chunk = os.read(file_descriptor, 64 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MAX_OCR_WORKER_SCRIPT_BYTES:
                    raise ValueError(
                        "video index plan requires a reviewed OCR worker script"
                    )
                digest.update(chunk)
            after = os.fstat(file_descriptor)
            current = os.stat(
                absolute.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except ValueError:
            raise
        except OSError as error:
            raise ValueError(
                "video index plan requires a reviewed OCR worker script"
            ) from error
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        os.close(parent_descriptor)

    expected = _ocr_script_fingerprint(before)
    if (
        total != before.st_size
        or _ocr_script_fingerprint(after) != expected
        or _ocr_script_fingerprint(current) != expected
    ):
        raise ValueError("video index plan requires a reviewed OCR worker script")

    # Re-open the lexical chain once more so an ancestor replacement cannot be
    # mistaken for the script whose bytes were hashed above.
    verification_absolute, verification_parent = _open_ocr_script_parent(absolute)
    try:
        verification = os.stat(
            verification_absolute.name,
            dir_fd=verification_parent,
            follow_symlinks=False,
        )
    except OSError as error:
        raise ValueError(
            "video index plan requires a reviewed OCR worker script"
        ) from error
    finally:
        os.close(verification_parent)
    if _ocr_script_fingerprint(verification) != expected:
        raise ValueError("video index plan requires a reviewed OCR worker script")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class IndexingRunPlan:
    """Ephemeral inputs shared by stage identities and provider requests."""

    specifications: IndexingSpecifications
    whisper_prompt_snapshot: WhisperPromptSnapshot


@dataclass(slots=True)
class Runtime:
    queue: DurableVideoIndexDispatcher
    search: SearchService
    clips: ClipService
    providers: ProviderRegistry
    runtime_lock: ExclusiveRuntimeLock
    artifact_gc: ArtifactGarbageCollector
    generation_store: QdrantVectorIndex
    video_index_plan_factory: Callable[[], VideoIndexPlanSnapshot] | None = None
    legacy_job_adopter: Callable[[], object] | None = None
    text_embedding_runtime: ReviewedSemanticEmbeddingRuntime | None = None
    _started: bool = field(default=False, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _shutdown_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _shutdown_complete: Event = field(default_factory=Event, init=False, repr=False)
    _shutdown_reaper: Thread | None = field(default=None, init=False, repr=False)
    _generation_store_closed: bool = field(default=False, init=False, repr=False)
    _text_embedding_runtime_closed: bool = field(
        default=False,
        init=False,
        repr=False,
    )

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("runtime is closed")
        if self._started:
            raise RuntimeError("runtime is already started")
        try:
            if not getattr(self.runtime_lock, "is_acquired", False):
                self.runtime_lock.acquire()
            self.artifact_gc.recover_startup()
            if self.legacy_job_adopter is not None:
                self.legacy_job_adopter()
            self.queue.recover_startup()
            self.artifact_gc.start()
            self.queue.start()
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
        if not self._text_embedding_runtime_closed:
            try:
                if self.text_embedding_runtime is not None:
                    self.text_embedding_runtime.close()
            except Exception:
                logger.exception(
                    "Runtime reviewed text embedding cleanup failed; ownership retained"
                )
                return False
            self._text_embedding_runtime_closed = True
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


def create_vision_worker_specification(
    settings: AppSettings,
) -> VisionWorkerSpecification:
    revision = model_revision(settings.siglip_model)
    if revision is None:
        raise ValueError("SigLIP worker model must have a pinned revision")
    profile = REVIEWED_SIGLIP_PROFILES.get((settings.siglip_model, revision))
    if profile is None:
        raise ValueError("SigLIP worker model profile has not been reviewed")
    dimensions, logit_scale, logit_bias = profile
    return VisionWorkerSpecification(
        siglip_model=settings.siglip_model,
        siglip_revision=revision,
        embedding_dimensions=dimensions,
        detector_model_id=settings.vision_detector_model_id,
        detector_checkpoint_sha256=settings.vision_detector_checkpoint_sha256,
        siglip_logit_scale=logit_scale,
        siglip_logit_bias=logit_bias,
        minimum_confidence=settings.vision_worker_minimum_confidence,
    )


def _reviewed_text_embedding_contract(
    settings: AppSettings,
) -> ReviewedSemanticEmbeddingContract:
    try:
        contract = load_reviewed_semantic_embedding_contract(
            model_name=settings.text_embedding_model,
            dimensions=settings.text_embedding_dimensions,
        )
    except ReviewedSemanticEmbeddingError as error:
        raise ValueError("video index plan requires a reviewed text embedding") from error
    expected_repository, expected_revision = fastembed_snapshot(
        settings.text_embedding_model
    )
    if (
        type(contract) is not ReviewedSemanticEmbeddingContract
        or not expected_repository
        or not expected_revision
        or contract.model_repository != expected_repository
        or contract.model_revision != expected_revision
        or contract.algorithm_version != FASTEMBED_ALGORITHM_VERSION
        or contract.dimensions != settings.text_embedding_dimensions
    ):
        raise ValueError("video index plan requires a reviewed text embedding")
    return contract


def _create_indexing_specifications(
    settings: AppSettings,
    *,
    whisper_prompt_snapshot: WhisperPromptSnapshot,
) -> IndexingSpecifications:
    """Snapshot every configuration input that changes trusted SQLite/text output."""
    embedding_contract = _reviewed_text_embedding_contract(settings)
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
        schema_version=2,
        implementation_revision="videoscope.speech-segments.v2",
        parameters={
            "effective_prompt_sha256": whisper_prompt_snapshot.effective_prompt_sha256,
            "glossary_state": whisper_prompt_snapshot.glossary_state,
            "language": settings.whisper_language,
            "merge_max_duration_seconds": 14.0,
            "merge_max_gap_seconds": 1.25,
            "segment_schema": segment_schema,
        },
        model_identity=model_identity(
            settings.whisper_model,
            model_revision(settings.whisper_model),
        ),
        dependencies={
            "boundary": (
                "isolated-worker" if settings.whisper_worker_endpoint else "disabled"
            ),
            "dependency_identity": WHISPER_DEPENDENCY_IDENTITY,
            "provider": WHISPER_WORKER_SCHEMA_VERSION,
            "runtime_identity": WHISPER_INFERENCE_RUNTIME_IDENTITY,
        },
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
    if settings.vision_worker_endpoint:
        vision_specification = create_vision_worker_specification(settings)
        object_model_identity = vision_specification.detector_model_identity
        object_dependencies = {
            "detector_specification": vision_specification.detector_identity,
            "provider": VISION_WORKER_SCHEMA_VERSION,
            "scenes_specification": scenes.specification_hash,
        }
        object_boundary = "isolated-worker"
    elif settings.roboflow_api_key and settings.roboflow_model_id:
        object_model_identity = f"roboflow/{settings.roboflow_model_id}"
        object_dependencies = {
            "provider": "roboflow-hosted-v1",
            "scenes_specification": scenes.specification_hash,
        }
        object_boundary = "hosted"
    else:
        object_model_identity = "objects/not-configured"
        object_dependencies = {
            "provider": "disabled",
            "scenes_specification": scenes.specification_hash,
        }
        object_boundary = "disabled"
    objects = StageSpecification(
        kind=StageKind.OBJECTS,
        schema_version=2,
        implementation_revision="videoscope.object-segments.v2",
        parameters={
            "boundary": object_boundary,
            "minimum_confidence": settings.vision_worker_minimum_confidence,
            "segment_schema": segment_schema,
        },
        model_identity=object_model_identity,
        dependencies=object_dependencies,
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
            "model_content_sha256": embedding_contract.model_content_sha256,
            "objects_specification": objects.specification_hash,
            "ocr_specification": ocr.specification_hash,
            "provider": "qdrant",
            "semantic_embedding_type": embedding_contract.semantic_embedding_type,
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


def create_indexing_run_plan(settings: AppSettings) -> IndexingRunPlan:
    prompt_snapshot = snapshot_whisper_prompt(
        settings.whisper_initial_prompt,
        settings.glossary_path,
    )
    return IndexingRunPlan(
        specifications=_create_indexing_specifications(
            settings,
            whisper_prompt_snapshot=prompt_snapshot,
        ),
        whisper_prompt_snapshot=prompt_snapshot,
    )


def create_indexing_specifications_from_prompt_snapshot(
    settings: AppSettings,
    prompt_snapshot: WhisperPromptSnapshot,
) -> IndexingSpecifications:
    """Build stage identities from one caller-owned immutable prompt snapshot."""
    if not isinstance(settings, AppSettings):
        raise TypeError("settings must be validated")
    if not isinstance(prompt_snapshot, WhisperPromptSnapshot):
        raise TypeError("Whisper prompt snapshot must be validated")
    return _create_indexing_specifications(
        settings,
        whisper_prompt_snapshot=prompt_snapshot,
    )


def create_indexing_specifications(settings: AppSettings) -> IndexingSpecifications:
    """Compatibility projection for consumers that need only persisted identities."""
    return create_indexing_run_plan(settings).specifications


def create_vision_worker_client(
    settings: AppSettings,
    *,
    input_root: Path | None = None,
) -> VisionWorkerClient | None:
    if not settings.vision_worker_endpoint:
        return None
    resolved_input_root = settings.data_dir
    if input_root is not None:
        try:
            raw_input_root = os.fspath(input_root)
            if (
                not isinstance(input_root, Path)
                or not input_root.is_absolute()
                or not raw_input_root
                or "\x00" in raw_input_root
            ):
                raise OSError
            lexical_input_root = Path(os.path.abspath(raw_input_root))
            candidate = lexical_input_root.resolve(strict=True)
            lexical_product_data = Path(os.path.abspath(settings.data_dir))
            product_data = lexical_product_data.resolve(strict=True)
            expected_input_root = product_data.parent
            filesystem_root = Path(candidate.anchor).resolve(strict=True)
            user_home = Path.home().resolve(strict=True)
            if (
                lexical_input_root != candidate
                or lexical_product_data != product_data
                or not candidate.is_dir()
                or candidate != expected_input_root
                or candidate in {filesystem_root, user_home}
            ):
                raise OSError
        except (TypeError, ValueError, OSError, RuntimeError) as error:
            raise ValueError(
                "Vision worker input root must be an existing no-symlink absolute "
                "directory equal to the exact parent of product data"
            ) from error
        resolved_input_root = candidate
    return VisionWorkerClient(
        endpoint=settings.vision_worker_endpoint,
        api_key=settings.vision_worker_api_key or "",
        input_root=resolved_input_root,
        specification=create_vision_worker_specification(settings),
        timeout=settings.vision_worker_timeout,
    )


def create_visual_index(
    settings: AppSettings,
    inference_client: VisionWorkerClient | None = None,
) -> SiglipVisualIndex:
    """Build the one canonical dense visual writer used by runtime and backfill."""
    return SiglipVisualIndex(
        settings.visual_index_dir,
        model_name=settings.siglip_model,
        model_revision=model_revision(settings.siglip_model),
        batch_size=settings.siglip_batch_size,
        sample_step=settings.visual_index_step,
        max_width=settings.visual_index_max_width,
        inference_client=inference_client,
    )


def _temporal_refiner_scorer_identities(
    scorer: object,
    vision_client: object | None,
) -> tuple[str | None, str | None]:
    """Project only immutable, path-free scorer/runtime identities."""
    model = getattr(scorer, "model_identity", None)
    specification = getattr(scorer, "specification_identity", None)
    if (
        type(model) is not str
        or not model
        or type(specification) is not str
        or not specification
        or vision_client is None
    ):
        return None, None
    try:
        worker_identity = getattr(vision_client, "identity")
    except Exception:
        return None, None
    if not isinstance(worker_identity, dict):
        return None, None
    runtime_identity = worker_identity.get("siglip_specification_hash")
    if type(runtime_identity) is not str or not runtime_identity:
        return None, None
    return (
        f"{model}#{specification}",
        f"vision-worker:{runtime_identity}",
    )


def create_object_detector(
    settings: AppSettings,
    *,
    vision_client: VisionWorkerClient | None = None,
) -> VisionWorkerClient | RoboflowDetector | None:
    if vision_client is not None:
        return vision_client
    if settings.roboflow_api_key and settings.roboflow_model_id:
        return RoboflowDetector(
            api_key=settings.roboflow_api_key,
            model_id=settings.roboflow_model_id,
            minimum_confidence=settings.vision_worker_minimum_confidence,
        )
    return None


def create_whisper_transcriber(settings: AppSettings) -> WhisperTranscriber | None:
    if not settings.whisper_worker_endpoint:
        return None
    inference_client = WhisperWorkerClient(
        endpoint=settings.whisper_worker_endpoint,
        api_key=settings.whisper_worker_api_key or "",
        input_root=settings.media_dir,
        expected_model_identity=model_identity(
            settings.whisper_model,
            model_revision(settings.whisper_model),
        ),
        timeout=settings.whisper_worker_timeout,
    )
    return WhisperTranscriber(
        settings.whisper_model,
        settings.whisper_language,
        settings.whisper_initial_prompt,
        settings.glossary_path,
        model_revision=model_revision(settings.whisper_model),
        inference_client=inference_client,
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


def _visual_executor_projection(settings: AppSettings) -> dict[str, object]:
    if not settings.vision_worker_endpoint:
        return {
            "boundary": "disabled",
            "provider": SiglipVisualIndex.id,
        }

    client = create_vision_worker_client(settings)
    if not isinstance(client, VisionWorkerClient):
        raise ValueError("video index plan requires a reviewed visual provider")
    visual = create_visual_index(settings, inference_client=client)
    specification = getattr(visual, "specification", None)
    worker_specification = getattr(client, "specification", None)
    if (
        not isinstance(visual, SiglipVisualIndex)
        or not isinstance(specification, VisualIndexSpecification)
        or not isinstance(worker_specification, VisionWorkerSpecification)
        or specification.inference_identity != worker_specification.siglip_identity
        or specification.model_name != worker_specification.siglip_model
        or specification.model_revision != worker_specification.siglip_revision
        or visual.batch_size != settings.siglip_batch_size
        or isinstance(client.timeout, bool)
        or not isinstance(client.timeout, (int, float))
        or float(client.timeout) != float(settings.vision_worker_timeout)
    ):
        raise ValueError("video index plan requires a reviewed visual provider")
    return {
        "batch_size": visual.batch_size,
        "boundary": "isolated-worker",
        "provider": visual.id,
        "timeout_seconds": float(settings.vision_worker_timeout),
        "visual_specification_hash": specification.identity,
        "worker_contract": VISION_WORKER_SCHEMA_VERSION,
        "worker_specification_hash": worker_specification.identity,
    }


def _lighthouse_executor_projection(
    settings: AppSettings,
    *,
    ffmpeg: FFmpeg,
) -> dict[str, object]:
    if settings.lighthouse_allow_in_process:
        raise ValueError(
            "video index plan requires a reviewed isolated Lighthouse provider"
        )
    provider = create_lighthouse_retriever(settings, ffmpeg)
    if isinstance(provider, DisabledLighthouseRetriever):
        return {
            "boundary": "disabled",
            "provider": provider.id,
        }
    if not isinstance(provider, LighthouseWorkerClient):
        raise ValueError(
            "video index plan requires a reviewed isolated Lighthouse provider"
        )
    specification = getattr(provider, "specification", None)
    if (
        not isinstance(specification, LighthouseSpecification)
        or specification != DEFAULT_LIGHTHOUSE_SPECIFICATION
        or specification.runtime_identity != LIGHTHOUSE_WORKER_RUNTIME_IDENTITY
        or isinstance(provider.timeout, bool)
        or not isinstance(provider.timeout, (int, float))
        or float(provider.timeout) != float(settings.lighthouse_timeout)
    ):
        raise ValueError(
            "video index plan requires a reviewed isolated Lighthouse provider"
        )
    return {
        "boundary": "isolated-worker",
        "model_identity": LIGHTHOUSE_MODEL_IDENTITY,
        "provider": provider.id,
        "specification_hash": specification.identity,
        "timeout_seconds": float(settings.lighthouse_timeout),
        "worker_contract": LIGHTHOUSE_WORKER_SCHEMA_VERSION,
        "worker_runtime_identity": LIGHTHOUSE_WORKER_RUNTIME_IDENTITY,
    }


def _ocr_executor_projection(settings: AppSettings) -> dict[str, object]:
    dependency_identity = OCR_WORKER_DEPENDENCY_IDENTITY
    runtime_identity = OCR_WORKER_RUNTIME_IDENTITY
    model_identity = OCR_MODEL_ARTIFACT_IDENTITY
    if (
        type(dependency_identity) is not str
        or _OCR_DEPENDENCY_IDENTITY_RE.fullmatch(dependency_identity) is None
        or type(runtime_identity) is not str
        or _OCR_RUNTIME_IDENTITY_RE.fullmatch(runtime_identity) is None
        or type(model_identity) is not str
        or _OCR_MODEL_IDENTITY_RE.fullmatch(model_identity) is None
    ):
        raise ValueError(
            "video index plan requires reviewed OCR dependency/runtime identities"
        )
    return {
        "adapter_revision": OCR_WORKER_ADAPTER_REVISION,
        "boundary": "isolated-subprocess",
        "dependency_identity": dependency_identity,
        "model_identity": model_identity,
        "protocol": OCR_WORKER_PROTOCOL,
        "runtime_identity": runtime_identity,
        "script_sha256": _reviewed_ocr_script_sha256(settings.ocr_worker_script),
    }


def _speech_executor_projection(settings: AppSettings) -> dict[str, object]:
    if not settings.whisper_worker_endpoint:
        return {
            "boundary": "disabled",
            "provider": WhisperTranscriber.id,
        }
    revision = model_revision(settings.whisper_model)
    model = model_identity(settings.whisper_model, revision)
    if revision is None or "@" not in model:
        raise ValueError("video index plan requires a reviewed Whisper worker")
    return {
        "boundary": "isolated-worker",
        "dependency_identity": WHISPER_DEPENDENCY_IDENTITY,
        "language": settings.whisper_language,
        "model_identity": model,
        "provider": WhisperTranscriber.id,
        "runtime_identity": WHISPER_INFERENCE_RUNTIME_IDENTITY,
        "timeout_seconds": float(settings.whisper_worker_timeout),
        "worker_contract": WHISPER_WORKER_SCHEMA_VERSION,
    }


def _text_vector_executor_projection(
    settings: AppSettings,
    *,
    contract: ReviewedSemanticEmbeddingContract | None = None,
    embedding: SemanticEmbedding | None = None,
    require_strict: bool = False,
) -> dict[str, object]:
    resolved_contract = contract or _reviewed_text_embedding_contract(settings)
    if type(resolved_contract) is not ReviewedSemanticEmbeddingContract:
        raise ValueError("video index plan requires a reviewed text embedding")
    resolved_embedding = embedding or create_semantic_embedding(
        model_name=settings.text_embedding_model,
        dimensions=settings.text_embedding_dimensions,
        cache_dir=settings.models_dir / "fastembed",
    )
    if type(resolved_embedding) is not SemanticEmbedding:
        raise ValueError(
            "video index plan requires a concrete semantic embedding"
        )
    embedding_identity = resolved_embedding.identity
    expected_attestation = resolved_contract.embedding_attestation_dict
    if (
        resolved_embedding.model_name != resolved_contract.model_name
        or resolved_embedding.model_repository != resolved_contract.model_repository
        or resolved_embedding.model_revision != resolved_contract.model_revision
        or resolved_embedding.expected_runtime_version
        != resolved_contract.runtime_version
        or resolved_embedding.algorithm_version
        != resolved_contract.algorithm_version
        or resolved_embedding.dimensions != resolved_contract.dimensions
        or type(embedding_identity) is not str
        or embedding_identity != resolved_contract.embedding_identity
    ):
        raise ValueError("video index plan requires a reviewed text embedding")
    if require_strict:
        try:
            actual_attestation = resolved_embedding.attestation_identity
        except Exception as error:
            raise ValueError(
                "video index runtime requires an attested text embedding"
            ) from error
        if (
            resolved_embedding.strict_no_fallback is not True
            or actual_attestation != expected_attestation
        ):
            raise ValueError(
                "video index runtime requires an attested text embedding"
            )
    physical_specification = TextVectorIndexSpecification(
        embedding_identity=embedding_identity,
        dimensions=settings.text_embedding_dimensions,
    )
    return {
        "embedding": resolved_contract.canonical_dict,
        "physical_specification_hash": physical_specification.specification_hash,
        "provider": QdrantVectorIndex.id,
        "strict_no_fallback": True,
        "writer_revision": TEXT_VECTOR_WRITER_IMPLEMENTATION_REVISION,
    }


def _verified_indexing_toolchain(
    indexing_toolchain: AttestedIndexingToolchain | None,
) -> tuple[AttestedIndexingToolchain, str]:
    selected = indexing_toolchain or attest_indexing_toolchain()
    verify_current = getattr(selected, "verify_current", None)
    if not callable(verify_current):
        raise TypeError("indexing toolchain must support current-state verification")
    identity = verify_current()
    if (
        type(identity) is not str
        or not _INDEXING_TOOLCHAIN_IDENTITY_RE.fullmatch(identity)
    ):
        raise ValueError("video index plan requires an attested indexing toolchain")
    return selected, identity


def _attested_ffmpeg(indexing_toolchain: AttestedIndexingToolchain) -> FFmpeg:
    factory = getattr(indexing_toolchain, "create_ffmpeg", None)
    if not callable(factory):
        raise TypeError("indexing toolchain must create an attested FFmpeg adapter")
    ffmpeg = factory()
    if not isinstance(ffmpeg, FFmpeg):
        raise ValueError("indexing toolchain returned an invalid FFmpeg adapter")
    return ffmpeg


def _video_index_executor_payload(
    settings: AppSettings,
    *,
    indexing_toolchain: AttestedIndexingToolchain | None = None,
) -> dict[str, object]:
    if not isinstance(settings, AppSettings):
        raise TypeError("settings must be validated")
    if (
        not settings.vision_worker_endpoint
        and settings.roboflow_api_key
        and settings.roboflow_model_id
    ):
        raise ValueError(
            "video index plan requires an attested isolated object provider"
        )
    resolved_toolchain, toolchain_identity = _verified_indexing_toolchain(
        indexing_toolchain
    )
    ffmpeg = _attested_ffmpeg(resolved_toolchain)
    return {
        "executor_schema_version": VIDEO_INDEX_EXECUTOR_SCHEMA_VERSION,
        "ffmpeg": {
            "adapter_revision": FFMPEG_INDEXING_CONTRACT_REVISION,
            "dense_frame_extractor": VISUAL_INDEX_EXTRACTOR_IDENTITY,
            "probe_contract": "ffprobe-json-primary-video-stream-v1",
            "scene_frame_contract": "jpeg-midpoint-scale-min-960-q2-v1",
            "toolchain_identity": toolchain_identity,
        },
        "indexer_revision": VIDEO_INDEXER_IMPLEMENTATION_REVISION,
        "lighthouse": _lighthouse_executor_projection(settings, ffmpeg=ffmpeg),
        "ocr": _ocr_executor_projection(settings),
        "speech": _speech_executor_projection(settings),
        "text_vector_writer": _text_vector_executor_projection(settings),
        "visual_dense": _visual_executor_projection(settings),
    }


def _executor_identity_from_payload(payload: dict[str, object]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def create_video_index_executor_identity(
    settings: AppSettings,
    *,
    indexing_toolchain: AttestedIndexingToolchain | None = None,
) -> str:
    """Return the pathless identity of every production indexing executor seam."""
    return _executor_identity_from_payload(
        _video_index_executor_payload(
            settings,
            indexing_toolchain=indexing_toolchain,
        )
    )


def _external_indexing_specifications(
    specifications: IndexingSpecifications,
    executor_payload: dict[str, object],
) -> tuple[StageSpecification, StageSpecification]:
    visual_projection = executor_payload.get("visual_dense")
    lighthouse_projection = executor_payload.get("lighthouse")
    if not isinstance(visual_projection, dict) or not isinstance(
        lighthouse_projection,
        dict,
    ):
        raise ValueError("video index executor projections are incomplete")
    visual_model = visual_projection.get("worker_specification_hash")
    lighthouse_model = lighthouse_projection.get("model_identity")
    visual = StageSpecification(
        kind=StageKind.VISUAL_DENSE,
        schema_version=1,
        implementation_revision="videoscope.visual-dense-generations.v1",
        parameters=visual_projection,
        model_identity=(
            str(visual_model)
            if type(visual_model) is str
            else "visual-dense/not-configured"
        ),
        dependencies={
            "provider": str(visual_projection.get("provider")),
            "scenes_specification": specifications.scenes.specification_hash,
        },
    )
    lighthouse = StageSpecification(
        kind=StageKind.LIGHTHOUSE,
        schema_version=1,
        implementation_revision="videoscope.lighthouse-generations.v1",
        parameters=lighthouse_projection,
        model_identity=(
            str(lighthouse_model)
            if type(lighthouse_model) is str
            else "lighthouse/not-configured"
        ),
        dependencies={
            "provider": str(lighthouse_projection.get("provider")),
            "scenes_specification": specifications.scenes.specification_hash,
        },
    )
    return visual, lighthouse


def create_video_index_plan_snapshot(
    settings: AppSettings,
    *,
    indexing_toolchain: AttestedIndexingToolchain | None = None,
) -> VideoIndexPlanSnapshot:
    """Freeze one durable, retry-safe production indexing plan without I/O writes."""
    if not isinstance(settings, AppSettings):
        raise TypeError("settings must be validated")
    run_plan = create_indexing_run_plan(settings)
    executor_payload = _video_index_executor_payload(
        settings,
        indexing_toolchain=indexing_toolchain,
    )
    text_executor = executor_payload.get("text_vector_writer")
    text_embedding = (
        text_executor.get("embedding") if isinstance(text_executor, dict) else None
    )
    text_stage = run_plan.specifications.text_vectors
    if (
        not isinstance(text_embedding, dict)
        or text_stage.dependencies.get("model_content_sha256")
        != text_embedding.get("model_content_sha256")
        or text_stage.dependencies.get("semantic_embedding_type")
        != text_embedding.get("semantic_embedding_type")
        or text_stage.model_identity
        != model_identity(
            str(text_embedding.get("model_repository", "")),
            str(text_embedding.get("model_revision", "")),
        )
        or text_stage.parameters.get("dimensions")
        != text_embedding.get("dimensions")
    ):
        raise ValueError(
            "reviewed text embedding changed while the video index plan was snapshotted"
        )
    visual, lighthouse = _external_indexing_specifications(
        run_plan.specifications,
        executor_payload,
    )
    return VideoIndexPlanSnapshot(
        schema_version=VIDEO_INDEX_PLAN_SCHEMA_VERSION,
        specifications=run_plan.specifications,
        visual_dense_specification=visual,
        lighthouse_specification=lighthouse,
        whisper_prompt_snapshot=run_plan.whisper_prompt_snapshot,
        executor_identity=_executor_identity_from_payload(executor_payload),
    )


def adopt_legacy_video_index_jobs(
    repository: Repository,
    *,
    plan_factory: Callable[[], VideoIndexPlanSnapshot],
    batch_size: int = 100,
) -> tuple[str, ...]:
    """Adopt bounded pre-v9 queued videos without fabricating provenance."""
    if not isinstance(repository, Repository):
        raise TypeError("legacy job adoption repository must be validated")
    if not callable(plan_factory):
        raise TypeError("legacy job adoption plan factory must be callable")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or not 1 <= batch_size <= 1000
    ):
        raise ValueError("legacy job adoption batch size must be between 1 and 1000")
    adopted: list[str] = []
    for _batch in range(1024):
        candidates = repository.list_legacy_video_index_candidates(limit=batch_size)
        if not candidates:
            return tuple(adopted)
        plan = plan_factory()
        if not isinstance(plan, VideoIndexPlanSnapshot):
            raise ValueError("legacy job adoption requires a validated plan")
        for video in candidates:
            job = repository.adopt_legacy_video_index_job(video.id, plan=plan)
            adopted.append(job.job_id)
        if len(candidates) < batch_size:
            return tuple(adopted)
    raise RuntimeError("legacy video index adoption exceeded its bounded work")


def build_runtime(
    settings: AppSettings,
    repository: Repository,
    *,
    indexing_toolchain: AttestedIndexingToolchain | None = None,
    ocr_worker_environment: Mapping[str, str] | None = None,
    vision_worker_input_root: Path | None = None,
) -> Runtime:
    resolved_toolchain, toolchain_identity = _verified_indexing_toolchain(
        indexing_toolchain
    )
    ffmpeg = _attested_ffmpeg(resolved_toolchain)
    scenes = SceneDetector(
        threshold=settings.scene_threshold,
        max_scene_seconds=settings.max_scene_seconds,
    )
    vision_client = (
        create_vision_worker_client(settings)
        if vision_worker_input_root is None
        else create_vision_worker_client(
            settings,
            input_root=vision_worker_input_root,
        )
    )
    whisper = create_whisper_transcriber(settings)
    ocr = PaddleOCRReader(
        worker_python=settings.ocr_worker_python,
        worker_script=settings.ocr_worker_script,
        worker_model_root=getattr(settings, "ocr_model_root", None),
        expected_dependency_identity=OCR_WORKER_DEPENDENCY_IDENTITY,
        expected_runtime_identity=OCR_WORKER_RUNTIME_IDENTITY,
        expected_model_identity=OCR_MODEL_ARTIFACT_IDENTITY,
        expected_script_sha256=_reviewed_ocr_script_sha256(
            settings.ocr_worker_script
        ),
        worker_environment=ocr_worker_environment,
    )
    object_detector = create_object_detector(
        settings,
        vision_client=vision_client,
    )
    lighthouse = create_lighthouse_retriever(settings, ffmpeg)
    text_embedding_contract = _reviewed_text_embedding_contract(settings)
    text_embedding_runtime: ReviewedSemanticEmbeddingRuntime | None = None
    try:
        text_embedding_runtime = create_reviewed_semantic_embedding_runtime(
            model_name=settings.text_embedding_model,
            dimensions=settings.text_embedding_dimensions,
            cache_dir=settings.models_dir / "fastembed",
            scratch_parent=settings.temp_dir,
            contract=text_embedding_contract,
        )
    except ReviewedSemanticEmbeddingError as error:
        logger.warning(
            "Reviewed FastEmbed bytes are unavailable; text vectors are disabled: %s",
            error,
        )
        text_embedding: object = UnavailableReviewedSemanticEmbedding(
            text_embedding_contract,
        )
    else:
        text_embedding = getattr(text_embedding_runtime, "embedding", None)
        try:
            if type(text_embedding) is not SemanticEmbedding:
                raise ValueError(
                    "video index runtime requires a concrete semantic embedding"
                )
            _text_vector_executor_projection(
                settings,
                contract=text_embedding_contract,
                embedding=text_embedding,
                require_strict=True,
            )
        except BaseException:
            text_embedding_runtime.close()
            raise
    try:
        qdrant = QdrantVectorIndex(
            settings.qdrant_dir / "text",
            embedding=text_embedding,
        )
    except BaseException:
        if text_embedding_runtime is not None:
            text_embedding_runtime.close()
        raise
    siglip = create_visual_index(settings, inference_client=vision_client)
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
    temporal_scorer_identity, temporal_runtime_identity = (
        _temporal_refiner_scorer_identities(siglip, vision_client)
    )

    qdrant_ready = qdrant.status().state is ProviderState.READY
    vector_index = qdrant if qdrant_ready else EmptyVectorIndex()
    speech_provider = whisper
    ocr_provider = ocr if ocr.status().state is ProviderState.READY else None
    object_provider = object_detector
    moment_provider = lighthouse if lighthouse.status().state is ProviderState.READY else None
    # Stale/legacy generations must remain readable only as unavailable evidence,
    # but they must not disable the writer needed by ingest/API reindex to replace
    # them with the canonical dense specification.
    visual_provider = siglip if vision_client is not None else None
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
            ffmpeg_identity=toolchain_identity,
            scorer_identity=temporal_scorer_identity,
            runtime_identity=temporal_runtime_identity,
        )
        if visual_provider is not None
        else None
    )

    text_vector_storage_gate = TextVectorStorageGate()
    indexer = Indexer(
        repository=repository,
        media_root=settings.media_dir,
        thumbnails_dir=settings.thumbnails_dir,
        specification_resolver=lambda: create_indexing_run_plan(settings),
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
    queue = DurableVideoIndexDispatcher(
        repository,
        indexer,
        executor_identity_resolver=lambda: create_video_index_executor_identity(
            settings,
            indexing_toolchain=resolved_toolchain,
        ),
        start_immediately=False,
    )

    ffmpeg_status = StaticProvider(
        "ffmpeg",
        "FFmpeg",
        ProviderState.READY,
        "reviewed ffmpeg and ffprobe executables are attested",
    )
    whisper_status = whisper or StaticProvider(
        "whisper",
        "Whisper MLX",
        ProviderState.NEEDS_CONFIGURATION,
        "Настройте изолированный Whisper worker",
        optional=True,
    )
    object_status = object_detector or StaticProvider(
        "roboflow",
        "Object detection",
        ProviderState.NEEDS_CONFIGURATION,
        "Настройте vision worker или hosted Roboflow",
        optional=True,
    )
    providers = ProviderRegistry(
        [
            ffmpeg_status,
            scenes,
            whisper_status,
            ocr,
            object_status,
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
            media_root=settings.media_dir,
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
        video_index_plan_factory=lambda: create_video_index_plan_snapshot(
            settings,
            indexing_toolchain=resolved_toolchain,
        ),
        text_embedding_runtime=text_embedding_runtime,
        legacy_job_adopter=lambda: adopt_legacy_video_index_jobs(
            repository,
            plan_factory=lambda: create_video_index_plan_snapshot(
                settings,
                indexing_toolchain=resolved_toolchain,
            ),
        ),
    )
