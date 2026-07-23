from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock
from time import monotonic


@dataclass(frozen=True, slots=True)
class RateLimit:
    requests: int
    seconds: float


class SlidingWindowLimiter:
    def __init__(self) -> None:
        self._events: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, client: str, bucket: str, limit: RateLimit) -> bool:
        now = monotonic()
        boundary = now - limit.seconds
        key = client, bucket
        with self._lock:
            events = self._events[key]
            while events and events[0] <= boundary:
                events.popleft()
            if len(events) >= limit.requests:
                return False
            events.append(now)
            return True

