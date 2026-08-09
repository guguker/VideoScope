from __future__ import annotations

from uuid import uuid4

from videoscope.config import AppSettings
from videoscope.model_manifest import model_revision
from videoscope.processing.indexer import merge_timed_text
from videoscope.providers.base import ProviderState
from videoscope.providers.whisper import WhisperTranscriber
from videoscope.repository import Repository
from videoscope.search.embeddings import create_semantic_embedding
from videoscope.search.vector_index import QdrantVectorIndex


def main() -> None:
    settings = AppSettings()
    settings.ensure_directories()
    repository = Repository(settings.database_path)
    repository.initialize()
    whisper = WhisperTranscriber(
        settings.whisper_model,
        settings.whisper_language,
        settings.whisper_initial_prompt,
        settings.glossary_path,
        model_revision=model_revision(settings.whisper_model),
    )
    if whisper.status().state is not ProviderState.READY:
        raise SystemExit(whisper.status().detail)

    embedding = create_semantic_embedding(
        model_name=settings.text_embedding_model,
        dimensions=settings.text_embedding_dimensions,
        cache_dir=settings.models_dir / "fastembed",
    )
    vector_index = QdrantVectorIndex(settings.qdrant_dir / "text", embedding=embedding)
    vector_index.invalidate()
    videos = [video for video in repository.list_videos() if video.status == "ready"]
    for position, video in enumerate(videos, start=1):
        print(f"[{position}/{len(videos)}] {video.name}: transcribing")
        items = merge_timed_text(whisper.transcribe(settings.media_dir / video.stored_name))
        repository.clear_segments(video.id, modality="speech")
        for item in items:
            if not item.text.strip() or item.end <= item.start:
                continue
            repository.add_segment(
                segment_id=uuid4().hex,
                video_id=video.id,
                start=item.start,
                end=item.end,
                modality="speech",
                text=item.text.strip(),
                confidence=item.confidence,
                metadata=item.metadata,
            )
        print(f"[{position}/{len(videos)}] {video.name}: {len(items)} speech windows")
    vector_index.rebuild_repository(repository)
    print("Speech index and word timestamps are ready.")


if __name__ == "__main__":
    main()
