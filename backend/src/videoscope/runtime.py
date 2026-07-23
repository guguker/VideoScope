from __future__ import annotations

import shutil
from dataclasses import dataclass

from videoscope.clips import ClipService
from videoscope.config import AppSettings
from videoscope.media.ffmpeg import FFmpeg
from videoscope.processing.indexer import Indexer
from videoscope.processing.queue import ThreadedProcessingQueue
from videoscope.providers.base import ProviderRegistry, ProviderState, StaticProvider
from videoscope.providers.lighthouse import LighthouseRetriever
from videoscope.providers.internvideo import InternVideoReranker
from videoscope.providers.paddle_ocr import PaddleOCRReader
from videoscope.providers.roboflow import RoboflowDetector
from videoscope.providers.scenes import SceneDetector
from videoscope.providers.whisper import WhisperTranscriber
from videoscope.repository import Repository
from videoscope.search.service import SearchService
from videoscope.search.text_matching import SearchLexicon
from videoscope.search.temporal_refinement import TemporalRefiner
from videoscope.search.embeddings import SemanticEmbedding
from videoscope.search.vector_index import EmptyVectorIndex, QdrantVectorIndex
from videoscope.search.visual_index import SiglipVisualIndex


@dataclass(slots=True)
class Runtime:
    queue: ThreadedProcessingQueue
    search: SearchService
    clips: ClipService
    providers: ProviderRegistry


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
    )
    ocr = PaddleOCRReader()
    roboflow = RoboflowDetector(
        api_key=settings.roboflow_api_key,
        model_id=settings.roboflow_model_id,
        cache_dir=settings.models_dir / "rfdetr",
    )
    lighthouse = LighthouseRetriever(
        checkpoint=settings.lighthouse_checkpoint,
        cache_dir=settings.cache_dir,
        ffmpeg=ffmpeg,
        source_root=settings.lighthouse_root,
    )
    text_embedding = SemanticEmbedding(
        model_name=settings.text_embedding_model,
        dimensions=settings.text_embedding_dimensions,
        cache_dir=settings.models_dir / "fastembed",
    )
    qdrant = QdrantVectorIndex(settings.qdrant_dir / "text", embedding=text_embedding)
    siglip = SiglipVisualIndex(
        settings.visual_index_dir,
        model_name=settings.siglip_model,
        batch_size=settings.siglip_batch_size,
    )
    internvideo = InternVideoReranker(
        endpoint=settings.internvideo_endpoint,
        api_key=settings.internvideo_api_key,
        repository=repository,
        extractor=ffmpeg,
        temp_dir=settings.temp_dir,
        top_candidates=settings.internvideo_top_candidates,
        timeout=settings.internvideo_timeout,
    )

    qdrant_ready = qdrant.status().state is ProviderState.READY
    vector_index = qdrant if qdrant_ready else EmptyVectorIndex()
    speech_provider = whisper if whisper.status().state is ProviderState.READY else None
    ocr_provider = ocr if ocr.status().state is ProviderState.READY else None
    object_provider = roboflow if roboflow.status().state is ProviderState.READY else None
    moment_provider = lighthouse if lighthouse.status().state is ProviderState.READY else None
    visual_provider = siglip if siglip.status().state is ProviderState.READY else None
    candidate_reranker = internvideo if internvideo.status().state is ProviderState.READY else None
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

    indexer = Indexer(
        repository=repository,
        thumbnails_dir=settings.thumbnails_dir,
        ffmpeg=ffmpeg,
        scenes=scenes,
        speech=speech_provider,
        ocr=ocr_provider,
        objects=object_provider,
        vector_index=vector_index,
        visual_index=visual_provider,
        moment_retriever=moment_provider,
    )
    queue = ThreadedProcessingQueue(indexer)
    for video in repository.list_videos():
        if video.status in {"queued", "processing"}:
            queue.submit(video.id)

    ffmpeg_binary = shutil.which("ffmpeg")
    ffmpeg_status = StaticProvider(
        "ffmpeg",
        "FFmpeg",
        ProviderState.READY if ffmpeg_binary else ProviderState.UNAVAILABLE,
        ffmpeg_binary or "ffmpeg executable is missing",
    )
    providers = ProviderRegistry(
        [ffmpeg_status, scenes, whisper, ocr, roboflow, lighthouse, qdrant, siglip, internvideo]
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
        ),
        clips=ClipService(repository, ffmpeg, settings.clips_dir, settings.temp_dir),
        providers=providers,
    )
