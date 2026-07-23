from __future__ import annotations

from uuid import uuid4

from videoscope.config import AppSettings
from videoscope.processing.indexer import merge_timed_text
from videoscope.providers.base import ProviderState
from videoscope.providers.whisper import WhisperTranscriber
from videoscope.repository import Repository
from videoscope.search.embeddings import SemanticEmbedding
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
    )
    if whisper.status().state is not ProviderState.READY:
        raise SystemExit(whisper.status().detail)

    embedding = SemanticEmbedding(
        model_name=settings.text_embedding_model,
        dimensions=settings.text_embedding_dimensions,
        cache_dir=settings.models_dir / "fastembed",
    )
    vector_index = QdrantVectorIndex(settings.qdrant_dir / "text", embedding=embedding)
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
        vector_index.replace_video(video.id, repository.list_segments(video.id))
        print(f"[{position}/{len(videos)}] {video.name}: {len(items)} speech windows")
    print("Speech index and word timestamps are ready.")


if __name__ == "__main__":
    main()
