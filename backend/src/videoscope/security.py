from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock
from time import monotonic
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class _BodyLimitExceeded(Exception):
    pass


class RequestBodyLimitMiddleware:
    """Reject oversized upload bodies before Starlette's multipart parser spools them."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_body_bytes: int,
        path: str = "/api/videos",
    ) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.path = path

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") != self.path
        ):
            await self.app(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        raw_length = headers.get(b"content-length")
        if raw_length is not None:
            try:
                declared_length = int(raw_length)
            except ValueError:
                await JSONResponse(
                    {"detail": "Invalid Content-Length"},
                    status_code=400,
                )(scope, receive, send)
                return
            if declared_length < 0 or declared_length > self.max_body_bytes:
                await JSONResponse(
                    {"detail": "Video upload body is too large"},
                    status_code=413,
                )(scope, receive, send)
                return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    raise _BodyLimitExceeded
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _BodyLimitExceeded:
            await JSONResponse(
                {"detail": "Video upload body is too large"},
                status_code=413,
            )(scope, receive, send)


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
