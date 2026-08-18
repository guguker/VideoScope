from __future__ import annotations

from videoscope.config import AppSettings
from videoscope.media.ffmpeg import FFmpeg
from videoscope.providers.base import ProviderState
from videoscope.repository import Repository
from videoscope.runtime import create_visual_index


def main() -> None:
    settings = AppSettings()
    settings.ensure_directories()
    repository = Repository(settings.database_path)
    repository.initialize()
    ffmpeg = FFmpeg()
    index = create_visual_index(settings)
    status = index.status(check_index=False)
    if status.state is not ProviderState.READY:
        raise SystemExit(status.detail)

    videos = [video for video in repository.list_videos() if video.status == "ready"]
    for position, video in enumerate(videos, start=1):
        duration = float(video.duration or ffmpeg.probe(video.media_path).duration)
        expected_frames = max(1, int(duration / settings.visual_index_step))
        print(
            f"[{position}/{len(videos)}] {video.name}: "
            f"около {expected_frames} кадров с шагом {settings.visual_index_step:.2f} с"
        )
        index.replace_video_source(
            video.id,
            video.media_path,
            duration,
            ffmpeg,
            frames_dir=settings.thumbnails_dir / video.id,
        )
    print(f"Visual index is ready: {settings.siglip_model}")


if __name__ == "__main__":
    main()
