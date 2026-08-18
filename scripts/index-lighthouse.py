from __future__ import annotations

from pathlib import Path

from videoscope.config import AppSettings
from videoscope.media.ffmpeg import FFmpeg
from videoscope.providers.base import ProviderState
from videoscope.providers.lighthouse import DisabledLighthouseRetriever, LighthouseRetriever
from videoscope.providers.lighthouse_worker import LighthouseWorkerClient
from videoscope.repository import Repository
from videoscope.runtime_lifecycle import ExclusiveRuntimeLock


def _close_if_supported(resource: object | None) -> None:
    close = getattr(resource, "close", None)
    if callable(close):
        close()


def main() -> None:
    settings = AppSettings()
    runtime_lock = ExclusiveRuntimeLock(settings.data_dir)
    runtime_lock.acquire()
    retriever = None
    try:
        settings.ensure_directories()
        repository = Repository(settings.database_path)
        repository.initialize()
        ffmpeg = FFmpeg()
        if settings.lighthouse_endpoint:
            retriever = LighthouseWorkerClient(
                endpoint=settings.lighthouse_endpoint,
                api_key=settings.lighthouse_api_key or "",
                input_root=settings.media_dir,
                cache_dir=settings.cache_dir,
                timeout=settings.lighthouse_timeout,
            )
        elif settings.lighthouse_allow_in_process:
            retriever = LighthouseRetriever(
                checkpoint=settings.lighthouse_checkpoint,
                cache_dir=settings.cache_dir,
                ffmpeg=ffmpeg,
                source_root=settings.lighthouse_root,
            )
        else:
            retriever = DisabledLighthouseRetriever()
        status = retriever.status()
        if status.state is not ProviderState.READY:
            raise SystemExit(status.detail)

        videos = [video for video in repository.list_videos() if video.status == "ready"]
        for position, video in enumerate(videos, start=1):
            source = Path(video.media_path)
            duration = float(video.duration or ffmpeg.probe(source).duration)
            print(f"[{position}/{len(videos)}] {video.name}: Lighthouse cache")
            retriever.prepare(video.id, source, duration)
        print("Lighthouse cache is ready.")
    finally:
        try:
            _close_if_supported(retriever)
        finally:
            runtime_lock.close()


if __name__ == "__main__":
    main()
