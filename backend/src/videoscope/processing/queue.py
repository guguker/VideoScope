from __future__ import annotations

import logging
from queue import Queue
from threading import Thread

from videoscope.processing.indexer import Indexer


logger = logging.getLogger(__name__)


class ThreadedProcessingQueue:
    """Serial worker queue that keeps memory-heavy ML models off request threads."""

    def __init__(self, indexer: Indexer) -> None:
        self.indexer = indexer
        self._queue: Queue[str | None] = Queue()
        self._closed = False
        self._worker = Thread(target=self._run, name="videoscope-indexer", daemon=True)
        self._worker.start()

    def submit(self, video_id: str) -> None:
        if self._closed:
            raise RuntimeError("processing queue is closed")
        self._queue.put(video_id)

    def _run(self) -> None:
        while True:
            video_id = self._queue.get()
            try:
                if video_id is None:
                    return
                self.indexer.process(video_id)
            except Exception:
                logger.exception("Video indexing failed", extra={"video_id": video_id})
            finally:
                self._queue.task_done()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._worker.join(timeout=5)

