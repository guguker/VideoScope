from __future__ import annotations

from videoscope.config import AppSettings
from videoscope.media.ffmpeg import FFmpeg
from videoscope.providers.base import ProviderState
from videoscope.repository import Repository
from videoscope.runtime import create_vision_worker_client, create_visual_index
from videoscope.runtime_lifecycle import ExclusiveRuntimeLock


def _close_if_supported(resource: object | None) -> None:
    close = getattr(resource, "close", None)
    if callable(close):
        close()


def main() -> None:
    settings = AppSettings()
    runtime_lock = ExclusiveRuntimeLock(settings.data_dir)
    runtime_lock.acquire()
    index = None
    try:
        settings.ensure_directories()
        repository = Repository(settings.database_path)
        repository.initialize()
        ffmpeg = FFmpeg()
        vision_client = create_vision_worker_client(settings)
        index = create_visual_index(settings, inference_client=vision_client)
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
    finally:
        try:
            _close_if_supported(index)
        finally:
            runtime_lock.close()


if __name__ == "__main__":
    main()
