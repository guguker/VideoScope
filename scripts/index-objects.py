from __future__ import annotations

import argparse
from pathlib import Path
from uuid import uuid4

from videoscope.config import AppSettings
from videoscope.providers.roboflow import RoboflowDetector
from videoscope.repository import Repository
from videoscope.search.embeddings import SemanticEmbedding
from videoscope.search.vector_index import QdrantVectorIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild VideoScope object segments")
    parser.add_argument("--video-id", action="append", dest="video_ids")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = AppSettings()
    repository = Repository(settings.database_path)
    detector = RoboflowDetector(
        api_key=settings.roboflow_api_key,
        model_id=settings.roboflow_model_id,
        cache_dir=settings.models_dir / "rfdetr",
    )
    if detector.status().state.value != "ready":
        raise RuntimeError(detector.status().detail)

    videos = repository.list_videos()
    if args.video_ids:
        selected = set(args.video_ids)
        videos = [video for video in videos if video.id in selected]

    for video in videos:
        scenes = [
            segment
            for segment in repository.list_segments(video.id)
            if segment.modality == "scene"
            and segment.thumbnail_path
            and Path(segment.thumbnail_path).is_file()
        ]
        detected: list[tuple[object, list[object]]] = []
        print(f"{video.name}: detecting objects in {len(scenes)} scenes", flush=True)
        for index, scene in enumerate(scenes, start=1):
            tags = detector.detect(Path(scene.thumbnail_path or ""))
            detected.append((scene, tags))
            if index == len(scenes) or index % 20 == 0:
                print(f"  {index}/{len(scenes)}", flush=True)

        repository.clear_segments(video.id, modality="objects")
        for scene, tags in detected:
            if not tags:
                continue
            repository.add_segment(
                segment_id=uuid4().hex,
                video_id=video.id,
                start=scene.start,
                end=scene.end,
                modality="objects",
                text=", ".join(sorted({tag.label for tag in tags})),
                confidence=max(tag.confidence for tag in tags),
                metadata={
                    "scene_index": scene.metadata.get("scene_index"),
                    "objects": [
                        {"label": tag.label, "confidence": tag.confidence, **tag.metadata}
                        for tag in tags
                    ],
                    "provider": settings.roboflow_model_id,
                },
                thumbnail_path=scene.thumbnail_path,
            )

    embedding = SemanticEmbedding(
        model_name=settings.text_embedding_model,
        dimensions=settings.text_embedding_dimensions,
        cache_dir=settings.models_dir / "fastembed",
    )
    vector_index = QdrantVectorIndex(settings.qdrant_dir / "text", embedding=embedding)
    for video in videos:
        vector_index.replace_video(video.id, repository.list_segments(video.id))
    print("Object index is ready.", flush=True)


if __name__ == "__main__":
    main()
