"""Loopback review server for immutable, privately prepared annotation batches.

Run ``python -m videoscope.annotation_review.server --batch DIR --port 8766``.
This app deliberately has no connection to the production Runtime or database.
"""

import argparse
import fcntl
import hashlib
import json
import os
import stat
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError
from starlette.datastructures import Headers

from .schema import (
    ANNOTATION_FIELDS, AnnotationRecord, AnnotationRequest, BatchManifest,
    annotation_complete, batch_revision, canonical_json,
)


MAX_BODY_BYTES = 16 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024
WEB_ROOT = Path(__file__).parent / "web"
SECURITY_HEADERS = {
    "content-security-policy": "default-src 'self'; script-src 'self'; style-src 'self'; "
                               "img-src 'self'; media-src 'self'; connect-src 'self'; "
                               "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; "
                               "form-action 'self'",
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer",
    "cache-control": "no-store",
    "cross-origin-resource-policy": "same-origin",
}


class ReviewConflict(ValueError):
    """Stored inputs/history drifted or a revision is no longer current."""


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _contained(root: Path, relative: str, *, directory=False) -> Path:
    path = root / relative
    if Path(relative).is_absolute() or any(part in {"..", "."} for part in Path(relative).parts):
        raise ReviewConflict("unsafe contained path")
    current = root
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ReviewConflict("symlink is not an attested artifact")
    try:
        path.resolve(strict=True).relative_to(root)
        mode = path.stat().st_mode
    except (OSError, ValueError) as error:
        raise ReviewConflict("contained artifact unavailable") from error
    if not (stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)):
        raise ReviewConflict("artifact has wrong file type")
    return path


def _read_json(path: Path, limit: int) -> dict:
    try:
        with path.open("rb") as handle:
            raw = handle.read(limit + 1)
        if len(raw) > limit:
            raise ValueError("JSON exceeds bound")
        data = json.loads(raw, object_pairs_hook=_json_object,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
        if not isinstance(data, dict):
            raise ValueError("JSON must be an object")
        return data
    except (OSError, ValueError) as error:
        raise ReviewConflict("stored JSON is invalid") from error


def _file_identity(path):
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class ReviewStore:
    def __init__(self, batch_dir: Path):
        selected = Path(batch_dir).absolute()
        if selected.is_symlink() or not selected.is_dir():
            raise ReviewConflict("batch must be a real directory")
        self.root = selected.resolve(strict=True)
        raw = _read_json(_contained(self.root, "batch.json"), MAX_MANIFEST_BYTES)
        self.batch = BatchManifest.model_validate(raw)
        self.revision = batch_revision(raw)
        self.examples = {item.example_id: item for item in self.batch.examples}
        self.sources = {item.source_id: item for item in self.batch.sources}
        self.media_identities = {}
        self.lock = threading.RLock()
        for item in self.batch.examples:
            path = _contained(self.root, item.clip_path)
            before = _file_identity(path)
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            after = _file_identity(path)
            if before != after or after[2] != item.prepared_input_byte_size or digest != item.prepared_input_sha256:
                raise ReviewConflict("prepared input content attestation failed")
            self.media_identities[item.clip_path] = after
            poster = _contained(self.root, item.poster_path)
            if not 0 < poster.stat().st_size <= 10 * 1024 * 1024:
                raise ReviewConflict("poster exceeds bound")
            self.media_identities[item.poster_path] = _file_identity(poster)
        self._history()

    def check_batch(self):
        raw = _read_json(_contained(self.root, "batch.json"), MAX_MANIFEST_BYTES)
        if batch_revision(raw) != self.revision:
            raise ReviewConflict("batch changed; restart with a new immutable batch")

    def media(self, example_id, *, poster=False):
        self.check_batch()
        item = self.examples.get(example_id)
        if item is None:
            raise HTTPException(404, "unknown example")
        relative = item.poster_path if poster else item.clip_path
        path = _contained(self.root, relative)
        if _file_identity(path) != self.media_identities[relative]:
            raise ReviewConflict("prepared artifact changed")
        return path

    def _annotation_directory(self, *, create=False):
        path = self.root / "annotations"
        if not path.exists() and not path.is_symlink():
            if not create:
                return None
            path.mkdir(mode=0o700, exist_ok=True)
            _fsync_directory(self.root)
        return _contained(self.root, "annotations", directory=True)

    @contextmanager
    def _write_lock(self):
        with self.lock:
            directory = self._annotation_directory(create=True)
            descriptor = os.open(directory / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise ReviewConflict("invalid annotation lock")
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                yield
            finally:
                os.close(descriptor)

    def _history(self):
        directory = self._annotation_directory()
        if directory is None:
            return []
        records = []
        for item_dir in sorted(directory.iterdir()):
            if item_dir.name == ".lock":
                _contained(self.root, "annotations/.lock")
                continue
            if item_dir.name not in self.examples:
                raise ReviewConflict("unknown annotation example")
            _contained(self.root, f"annotations/{item_dir.name}", directory=True)
            revision = 0
            for path in sorted(item_dir.iterdir()):
                # A crash before publication may leave an inert private temporary file.
                if path.name.startswith(".pending-"):
                    _contained(self.root, f"annotations/{item_dir.name}/{path.name}")
                    continue
                revision += 1
                if len(records) >= 10000 or path.name != f"{revision:06d}.json":
                    raise ReviewConflict("annotation history has gaps or exceeds bound")
                _contained(self.root, f"annotations/{item_dir.name}/{path.name}")
                try:
                    record = AnnotationRecord.model_validate(_read_json(path, MAX_BODY_BYTES))
                except ValidationError as error:
                    raise ReviewConflict("annotation history violates schema") from error
                item = self.examples[item_dir.name]
                source = self.sources[item.source_id]
                if (record.batch_id != self.batch.batch_id or record.batch_revision != self.revision
                        or record.example_id != item.example_id or record.revision != revision
                        or record.source_id != source.source_id or record.source_sha256 != source.sha256
                        or record.source_start_seconds != item.source_start_seconds
                        or record.source_end_seconds != item.source_end_seconds
                        or record.prepared_input_sha256 != item.prepared_input_sha256
                        or (record.end_seconds is not None and record.end_seconds > item.clip_duration_seconds)):
                    raise ReviewConflict("annotation history provenance mismatch")
                records.append(record.model_dump())
        return records

    @staticmethod
    def _latest(records):
        return {record["example_id"]: {"revision": record["revision"],
                 **{field: record[field] for field in ANNOTATION_FIELDS}} for record in records}

    def review(self):
        self.check_batch()
        with self.lock:
            annotations = self._latest(self._history())
        return {
            "batch_id": self.batch.batch_id, "batch_revision": self.revision,
            "title": self.batch.title,
            "examples": [{"example_id": item.example_id, "source_alias": item.source_id,
                "source_start_seconds": item.source_start_seconds,
                "source_end_seconds": item.source_end_seconds,
                "clip_duration_seconds": item.clip_duration_seconds,
                "video_url": f"/media/{item.example_id}",
                "poster_url": f"/posters/{item.example_id}"} for item in self.batch.examples],
            "annotations": annotations,
        }

    def save(self, request: AnnotationRequest):
        self.check_batch()
        if request.batch_revision != self.revision:
            raise ReviewConflict("different batch revision")
        item = self.examples.get(request.example_id)
        if item is None:
            raise HTTPException(404, "unknown example")
        if request.end_seconds is not None and request.end_seconds > item.clip_duration_seconds:
            raise HTTPException(422, "annotation boundaries exceed prepared clip")
        self.media(item.example_id)
        with self._write_lock():
            self.check_batch()
            history = self._history()
            latest = self._latest(history)
            revision = latest.get(item.example_id, {}).get("revision", 0)
            if revision != request.expected_revision:
                raise ReviewConflict("annotation changed; reload before saving")
            if revision >= 10000 or len(history) >= 10000:
                raise ReviewConflict("annotation history limit reached")
            source = self.sources[item.source_id]
            fields = {field: getattr(request, field) for field in ANNOTATION_FIELDS}
            record = AnnotationRecord(
                **fields, schema_version=1, batch_id=self.batch.batch_id,
                batch_revision=self.revision, example_id=item.example_id, revision=revision + 1,
                created_at=datetime.now(timezone.utc).isoformat(), reviewer="local_owner",
                label_status="human_reviewed" if annotation_complete(fields) else "draft",
                destination="annotation_inbox", gold=False,
                training_allowed=False, promotion_allowed=False, source_id=source.source_id,
                source_sha256=source.sha256, source_start_seconds=item.source_start_seconds,
                source_end_seconds=item.source_end_seconds, prepared_input_sha256=item.prepared_input_sha256,
            ).model_dump()
            directory = self.root / "annotations" / item.example_id
            directory.mkdir(mode=0o700, exist_ok=True)
            _contained(self.root, f"annotations/{item.example_id}", directory=True)
            _fsync_directory(directory.parent)
            descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=directory)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(canonical_json(record) + b"\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                # link is atomic and fails if the destination already exists.
                os.link(temporary, directory / f"{revision + 1:06d}.json", follow_symlinks=False)
                _fsync_directory(directory)
            finally:
                os.unlink(temporary)
            _fsync_directory(directory)
            latest[item.example_id] = {"revision": revision + 1, **fields}
            return {"annotation": latest[item.example_id],
                    "reviewed_count": sum(annotation_complete(item) for item in latest.values()),
                    "total_count": len(self.examples)}

    def export(self):
        self.check_batch()
        with self.lock:
            records = self._history()
        return {"schema_version": 1, "batch_id": self.batch.batch_id,
                "batch_revision": self.revision, "purpose": "annotation_pilot",
                "training_allowed": False, "promotion_allowed": False,
                "sources": [source.model_dump() for source in self.batch.sources],
                "examples": [item.model_dump(exclude={"clip_path", "poster_path"}) for item in self.batch.examples],
                "records": records}


class LocalBoundary:
    """Validate Host/Origin before reading a bounded JSON request body."""

    def __init__(self, app, port):
        self.app = app
        self.host = f"127.0.0.1:{port}"
        self.origin = f"http://{self.host}"

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def secured_send(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                names = {key.lower() for key, _ in headers}
                headers.extend((key.encode(), value.encode()) for key, value in SECURITY_HEADERS.items()
                               if key.encode() not in names)
                message = {**message, "headers": headers}
            await send(message)

        async def reject(status, message):
            await JSONResponse({"detail": message}, status_code=status)(scope, receive, secured_send)

        headers = Headers(scope=scope)
        if headers.getlist("host") != [self.host]:
            return await reject(400, "untrusted host")
        origin = headers.getlist("origin")
        if origin and origin != [self.origin]:
            return await reject(403, "untrusted origin")
        if scope["method"] not in {"GET", "HEAD", "OPTIONS"}:
            if origin != [self.origin]:
                return await reject(403, "same-origin browser request required")
            if headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
                return await reject(415, "application/json required")
            body = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                body.extend(message.get("body", b""))
                if len(body) > MAX_BODY_BYTES:
                    return await reject(413, "annotation request too large")
                if not message.get("more_body", False):
                    break
            try:
                json.loads(body, object_pairs_hook=_json_object,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
            except (ValueError, UnicodeDecodeError):
                return await reject(422, "invalid JSON")

            async def replay():
                return {"type": "http.request", "body": bytes(body), "more_body": False}

            receive = replay
        await self.app(scope, receive, secured_send)


def create_app(batch_dir: Path, *, port: int = 8766) -> FastAPI:
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError("review port must be between 1024 and 65535")
    store = ReviewStore(batch_dir)
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(LocalBoundary, port=port)

    @app.exception_handler(ReviewConflict)
    async def conflict_handler(_request, _error):
        return JSONResponse({"detail": "review artifacts changed or revision is stale; reload the batch"}, status_code=409)

    @app.exception_handler(OSError)
    async def storage_handler(_request, _error):
        return JSONResponse({"detail": "local review storage unavailable; previous revisions are retained"}, status_code=409)

    @app.get("/api/health")
    def health():
        store.check_batch()
        return {"status": "ready", "batch_id": store.batch.batch_id}

    @app.get("/api/review")
    def review():
        return store.review()

    @app.post("/api/annotations")
    def annotate(request: AnnotationRequest):
        return store.save(request)

    @app.get("/api/export")
    def export():
        return JSONResponse(store.export(), headers={
            "Content-Disposition": f'attachment; filename="{store.batch.batch_id}-reviews.json"'})

    @app.get("/media/{example_id}")
    def media(example_id: str):
        return FileResponse(store.media(example_id), media_type="video/mp4")

    @app.get("/posters/{example_id}")
    def poster(example_id: str):
        return FileResponse(store.media(example_id, poster=True), media_type="image/jpeg")

    @app.get("/")
    def index():
        return static("index.html")

    @app.get("/{filename}")
    def static(filename: str):
        if filename not in {"index.html", "app.js", "style.css"}:
            raise HTTPException(404, "unknown resource")
        if not (WEB_ROOT / filename).is_file():
            raise HTTPException(404, "review interface unavailable")
        path = _contained(WEB_ROOT.resolve(), filename)
        types = {"index.html": "text/html", "app.js": "text/javascript", "style.css": "text/css"}
        return FileResponse(path, media_type=types[filename])

    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description="Review a private VideoScope annotation pilot locally")
    parser.add_argument("--batch", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args(argv)
    import uvicorn

    uvicorn.run(create_app(args.batch, port=args.port), host="127.0.0.1", port=args.port,
                access_log=False, server_header=False)


if __name__ == "__main__":
    main()
