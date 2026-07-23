from threading import Event

from videoscope.processing.queue import ThreadedProcessingQueue


def test_processing_queue_executes_submitted_video() -> None:
    processed = Event()
    video_ids: list[str] = []

    class FakeIndexer:
        def process(self, video_id: str) -> None:
            video_ids.append(video_id)
            processed.set()

    queue = ThreadedProcessingQueue(FakeIndexer())  # type: ignore[arg-type]
    queue.submit("video-1")

    assert processed.wait(timeout=2)
    queue.close()
    queue.close()
    assert video_ids == ["video-1"]

