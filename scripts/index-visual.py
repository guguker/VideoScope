from __future__ import annotations

from videoscope.config import AppSettings
from videoscope.providers.base import ProviderState
from videoscope.repository import Repository
from videoscope.search.visual_index import SiglipVisualIndex


def main() -> None:
    settings = AppSettings()
    settings.ensure_directories()
    repository = Repository(settings.database_path)
    repository.initialize()
    index = SiglipVisualIndex(
        settings.visual_index_dir,
        model_name=settings.siglip_model,
        batch_size=settings.siglip_batch_size,
    )
    status = index.status()
    if status.state is not ProviderState.READY:
        raise SystemExit(status.detail)

    videos = [video for video in repository.list_videos() if video.status == "ready"]
    for position, video in enumerate(videos, start=1):
        scenes = [
            segment
            for segment in repository.list_segments(video.id)
            if segment.modality == "scene"
        ]
        print(f"[{position}/{len(videos)}] {video.name}: {len(scenes)} scenes")
        index.replace_video(video.id, scenes)
    print(f"Visual index is ready: {settings.siglip_model}")


if __name__ == "__main__":
    main()
