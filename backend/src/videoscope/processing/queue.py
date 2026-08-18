from __future__ import annotations

import logging
from queue import Empty, Queue
from threading import Event, Lock, Thread, current_thread

from videoscope.processing.indexer import Indexer


logger = logging.getLogger(__name__)


class ThreadedProcessingQueue:
    """Последовательная очередь, не запускающая ресурсоёмкие модели в потоках обработки запросов."""

    def __init__(
        self,
        indexer: Indexer,
        *,
        start_immediately: bool = True,
    ) -> None:
        self.indexer = indexer
        self._queue: Queue[str | None] = Queue()
        self._state_lock = Lock()
        self._stop = Event()
        self._started = False
        self._closed = False
        self._worker: Thread | None = None
        if start_immediately:
            self.start()

    def start(self) -> None:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("processing queue is closed")
            if self._started:
                raise RuntimeError("processing queue is already started")
            worker = Thread(
                target=self._run,
                name="videoscope-indexer",
                daemon=True,
            )
            try:
                worker.start()
            except Exception:
                self._worker = None
                self._started = False
                raise
            self._worker = worker
            self._started = True

    def submit(self, video_id: str) -> None:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("processing queue is closed")
            if not self._started:
                raise RuntimeError("processing queue is not started")
            self._queue.put(video_id)

    def _run(self) -> None:
        while True:
            video_id = self._queue.get()
            try:
                if video_id is None:
                    return
                if self._stop.is_set():
                    return
                self.indexer.process(video_id)
            except Exception:
                logger.exception("Video indexing failed", extra={"video_id": video_id})
            finally:
                self._queue.task_done()

    def close(self, *, timeout: float | None = 5) -> bool:
        with self._state_lock:
            if not self._closed:
                self._closed = True
                self._stop.set()
                if self._started:
                    while True:
                        try:
                            self._queue.get_nowait()
                        except Empty:
                            break
                        else:
                            self._queue.task_done()
                    self._queue.put(None)
            worker = self._worker
        if worker is None:
            return True
        if worker is current_thread():
            return False
        worker.join(timeout=timeout)
        return not worker.is_alive()
