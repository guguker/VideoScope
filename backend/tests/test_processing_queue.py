from threading import Event

import pytest

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


def test_processing_queue_can_be_constructed_without_starting_work() -> None:
    processed = Event()

    class FakeIndexer:
        def process(self, _video_id: str) -> None:
            processed.set()

    queue = ThreadedProcessingQueue(  # type: ignore[arg-type]
        FakeIndexer(),
        start_immediately=False,
    )

    with pytest.raises(RuntimeError, match="not started"):
        queue.submit("video-1")
    assert processed.is_set() is False

    queue.start()
    queue.submit("video-1")
    assert processed.wait(timeout=2)
    assert queue.close() is True


def test_processing_queue_close_can_finish_after_an_initial_timeout() -> None:
    entered = Event()
    release = Event()

    class BlockingIndexer:
        def process(self, _video_id: str) -> None:
            entered.set()
            release.wait(timeout=2)

    queue = ThreadedProcessingQueue(BlockingIndexer())  # type: ignore[arg-type]
    queue.submit("video-1")
    assert entered.wait(timeout=2)

    assert queue.close(timeout=0.01) is False
    release.set()
    assert queue.close(timeout=2) is True


def test_processing_queue_close_discards_backlog_after_current_work() -> None:
    entered = Event()
    release = Event()
    processed: list[str] = []

    class BlockingIndexer:
        def process(self, video_id: str) -> None:
            processed.append(video_id)
            if video_id == "video-1":
                entered.set()
                release.wait(timeout=2)

    queue = ThreadedProcessingQueue(BlockingIndexer())  # type: ignore[arg-type]
    queue.submit("video-1")
    assert entered.wait(timeout=2)
    queue.submit("video-2")
    queue.submit("video-3")

    assert queue.close(timeout=0.01) is False
    release.set()
    assert queue.close(timeout=2) is True
    assert processed == ["video-1"]


def test_processing_queue_thread_start_failure_is_safe_to_close(monkeypatch) -> None:
    class FailingThread:
        def __init__(self, **_kwargs) -> None:  # type: ignore[no-untyped-def]
            return None

        def start(self) -> None:
            raise RuntimeError("thread unavailable")

    class FakeIndexer:
        def process(self, _video_id: str) -> None:
            raise AssertionError("must not run")

    monkeypatch.setattr("videoscope.processing.queue.Thread", FailingThread)
    queue = ThreadedProcessingQueue(  # type: ignore[arg-type]
        FakeIndexer(),
        start_immediately=False,
    )

    with pytest.raises(RuntimeError, match="thread unavailable"):
        queue.start()
    with pytest.raises(RuntimeError, match="not started"):
        queue.submit("video-1")
    assert queue.close() is True
