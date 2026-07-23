from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict
import mimetypes
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from fastapi import FastAPI, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from videoscope.clips import ClipSelection, ClipService
from videoscope.config import AppSettings
from videoscope.evaluation import EvaluationCase, EvaluationService, EvaluationStore
from videoscope.media.ffmpeg import FFmpeg
from videoscope.media.uploads import UploadRejected, validate_upload
from videoscope.providers.base import ProviderRegistry, ProviderState, StaticProvider
from videoscope.repository import Repository
from videoscope.search.service import SearchService
from videoscope.search.text_matching import SearchLexicon
from videoscope.search.vector_index import MemoryVectorIndex
from videoscope.security import RateLimit, SlidingWindowLimiter


UPLOAD_CHUNK_SIZE = 1024 * 1024


class ProcessingQueue(Protocol):
    def submit(self, video_id: str) -> None: ...

    def close(self) -> None: ...


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    video_ids: list[str] | None = Field(default=None, max_length=100)
    limit: int = Field(default=20, ge=1, le=50)
    use_lighthouse: bool = True
    mode: str = Field(default="all", pattern="^(all|speech|visual|ocr)$")


class RenameVideoRequest(BaseModel):
    name: str = Field(min_length=1, max_length=160)


class GlossaryRequest(BaseModel):
    entries: dict[str, list[str]]


class EvaluationCaseRequest(BaseModel):
    id: str = Field(min_length=1, max_length=100)
    query: str = Field(min_length=1, max_length=500)
    video_id: str = Field(min_length=1, max_length=64)
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    mode: str = Field(default="all", pattern="^(all|speech|visual|ocr)$")
    label_source: str = Field(default="gold", pattern="^(gold|silver)$")
    notes: str = Field(default="", max_length=500)


class EvaluationCasesRequest(BaseModel):
    cases: list[EvaluationCaseRequest] = Field(max_length=200)


class EvaluationRunRequest(BaseModel):
    variants: list[str] = Field(default=["auto"], min_length=1, max_length=6)


class ClipSelectionRequest(BaseModel):
    video_id: str = Field(min_length=1, max_length=64)
    start: float = Field(ge=0)
    end: float = Field(gt=0)


class ExportRequest(BaseModel):
    name: str = Field(default="videoscope-export", min_length=1, max_length=120)
    selections: list[ClipSelectionRequest] = Field(min_length=1, max_length=30)


def create_app(
    *,
    settings: AppSettings | None = None,
    repository: Repository | None = None,
    processing_queue: ProcessingQueue | None = None,
    search_service: SearchService | None = None,
    clip_service: ClipService | None = None,
    provider_registry: ProviderRegistry | None = None,
) -> FastAPI:
    resolved_settings = settings or AppSettings()
    resolved_settings.ensure_directories()
    resolved_repository = repository or Repository(resolved_settings.database_path)
    resolved_repository.initialize()
    if processing_queue is None:
        from videoscope.runtime import build_runtime

        runtime = build_runtime(resolved_settings, resolved_repository)
        resolved_queue = runtime.queue
        resolved_search = search_service or runtime.search
        resolved_clips = clip_service or runtime.clips
        resolved_providers = provider_registry or runtime.providers
    else:
        resolved_queue = processing_queue
        resolved_search = search_service or SearchService(
            resolved_repository,
            MemoryVectorIndex(),
            lexicon=SearchLexicon(resolved_settings.glossary_path),
        )
        resolved_clips = clip_service or ClipService(
            resolved_repository,
            FFmpeg(),
            resolved_settings.clips_dir,
            resolved_settings.temp_dir,
        )
        resolved_providers = provider_registry or ProviderRegistry(
            [StaticProvider("ffmpeg", "FFmpeg", ProviderState.READY, "test runtime")]
        )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        resolved_queue.close()

    app = FastAPI(
        title="VideoScope API",
        version="0.1.0",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings
    app.state.repository = resolved_repository
    app.state.processing_queue = resolved_queue
    app.state.search_service = resolved_search
    resolved_lexicon = resolved_search.lexicon or SearchLexicon(resolved_settings.glossary_path)
    resolved_search.lexicon = resolved_lexicon
    app.state.search_lexicon = resolved_lexicon
    evaluation_store = EvaluationStore(
        resolved_settings.evaluation_cases_path,
        resolved_settings.evaluation_report_path,
    )
    evaluation_service = EvaluationService(resolved_search, evaluation_store)
    app.state.evaluation_service = evaluation_service
    app.state.clip_service = resolved_clips
    app.state.provider_registry = resolved_providers
    limiter = SlidingWindowLimiter()
    trusted_origins = {
        f"http://{resolved_settings.host}:{resolved_settings.port}",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    }

    def video_payload(record: object) -> dict[str, object]:
        payload = asdict(record)  # type: ignore[arg-type]
        payload.pop("media_path", None)
        video_id = str(payload["id"])
        payload["media_url"] = f"/api/videos/{video_id}/media"
        payload["thumbnail_url"] = (
            f"/api/thumbnails/{video_id}/{Path(str(payload['thumbnail_path'])).name}"
            if payload.get("thumbnail_path")
            else None
        )
        payload.pop("thumbnail_path", None)
        return payload

    @app.middleware("http")
    async def security_headers(request, call_next):  # type: ignore[no-untyped-def]
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            origin = request.headers.get("origin")
            if origin and origin not in trusted_origins:
                return JSONResponse({"detail": "Untrusted origin"}, status_code=403)

        path = request.url.path
        client = request.client.host if request.client else "local"
        if path.startswith("/api/search"):
            bucket, rate = "search", RateLimit(120, 60)
        elif path.startswith("/api/videos") and request.method == "POST":
            bucket, rate = "upload", RateLimit(20, 3600)
        elif path.startswith("/api/exports") and request.method == "POST":
            bucket, rate = "export", RateLimit(30, 3600)
        else:
            bucket, rate = "api", RateLimit(600, 60)
        if not limiter.allow(client, bucket, rate):
            return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429)

        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "videoscope"}

    @app.get("/api/videos")
    def list_videos() -> list[dict[str, object]]:
        return [video_payload(record) for record in resolved_repository.list_videos()]

    @app.get("/api/videos/{video_id}")
    def get_video(video_id: str) -> dict[str, object]:
        record = resolved_repository.get_video(video_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Video not found")
        return video_payload(record)

    @app.patch("/api/videos/{video_id}")
    def rename_video(video_id: str, request: RenameVideoRequest) -> dict[str, object]:
        record = resolved_repository.get_video(video_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Video not found")
        name = " ".join(request.name.split()).strip(" .")
        if not name:
            raise HTTPException(status_code=422, detail="Video name is empty")
        updated = resolved_repository.update_video(video_id, display_name=name)
        return video_payload(updated)

    @app.post("/api/videos", status_code=status.HTTP_202_ACCEPTED)
    async def upload_video(file: UploadFile) -> dict[str, object]:
        filename = file.filename or ""
        try:
            preliminary = validate_upload(filename, 1, max_bytes=resolved_settings.max_upload_bytes)
        except UploadRejected as error:
            raise HTTPException(status_code=415, detail=str(error)) from error

        video_id = uuid4().hex
        stored_name = f"{video_id}{preliminary.extension}"
        destination = resolved_settings.media_dir / stored_name
        temporary = resolved_settings.temp_dir / f"{video_id}.upload"
        size_bytes = 0
        try:
            with temporary.open("wb") as output:
                while chunk := await file.read(UPLOAD_CHUNK_SIZE):
                    size_bytes += len(chunk)
                    if size_bytes > resolved_settings.max_upload_bytes:
                        raise HTTPException(status_code=413, detail="Video is too large")
                    output.write(chunk)
            validate_upload(filename, size_bytes, max_bytes=resolved_settings.max_upload_bytes)
            temporary.replace(destination)
        except HTTPException:
            temporary.unlink(missing_ok=True)
            destination.unlink(missing_ok=True)
            raise
        except UploadRejected as error:
            temporary.unlink(missing_ok=True)
            destination.unlink(missing_ok=True)
            raise HTTPException(status_code=415, detail=str(error)) from error
        finally:
            await file.close()

        record = resolved_repository.create_video(
            video_id=video_id,
            original_name=filename,
            stored_name=stored_name,
            media_path=str(destination),
            size_bytes=size_bytes,
        )
        resolved_queue.submit(video_id)
        return video_payload(record)

    @app.get("/api/videos/{video_id}/media")
    def video_media(video_id: str) -> FileResponse:
        record = resolved_repository.get_video(video_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Video not found")
        path = Path(record.media_path)
        if not path.is_file() or path.parent.resolve() != resolved_settings.media_dir.resolve():
            raise HTTPException(status_code=404, detail="Media not found")
        media_type = mimetypes.guess_type(record.original_name)[0] or "application/octet-stream"
        return FileResponse(path, media_type=media_type)

    @app.get("/api/thumbnails/{video_id}/{filename}")
    def thumbnail(video_id: str, filename: str) -> FileResponse:
        if Path(filename).name != filename:
            raise HTTPException(status_code=404, detail="Thumbnail not found")
        path = resolved_settings.thumbnails_dir / video_id / filename
        expected_parent = (resolved_settings.thumbnails_dir / video_id).resolve()
        if not path.is_file() or path.parent.resolve() != expected_parent:
            raise HTTPException(status_code=404, detail="Thumbnail not found")
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/api/videos/{video_id}/segments")
    def video_segments(video_id: str) -> list[dict[str, object]]:
        if resolved_repository.get_video(video_id) is None:
            raise HTTPException(status_code=404, detail="Video not found")
        return [asdict(segment) for segment in resolved_repository.list_segments(video_id)]

    @app.post("/api/videos/{video_id}/reindex", status_code=status.HTTP_202_ACCEPTED)
    def reindex_video(video_id: str) -> dict[str, str]:
        if resolved_repository.get_video(video_id) is None:
            raise HTTPException(status_code=404, detail="Video not found")
        resolved_repository.update_video(video_id, status="queued", progress=0.0, stage="queued", error=None)
        resolved_queue.submit(video_id)
        return {"status": "queued", "video_id": video_id}

    @app.post("/api/search")
    def search(request: SearchRequest) -> list[dict[str, object]]:
        try:
            results = resolved_search.search(
                request.query,
                video_ids=request.video_ids,
                limit=request.limit,
                use_lighthouse=request.use_lighthouse,
                mode=request.mode,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return [asdict(result) for result in results]

    @app.get("/api/search/glossary")
    def get_search_glossary() -> dict[str, object]:
        return {"entries": resolved_lexicon.read()}

    @app.put("/api/search/glossary")
    def update_search_glossary(request: GlossaryRequest) -> dict[str, object]:
        if len(request.entries) > 500:
            raise HTTPException(status_code=422, detail="Glossary is too large")
        if any(
            len(term) > 120
            or len(aliases) > 20
            or any(len(alias) > 120 for alias in aliases)
            for term, aliases in request.entries.items()
        ):
            raise HTTPException(status_code=422, detail="Glossary entry is too large")
        resolved_lexicon.replace(request.entries)
        return {"entries": resolved_lexicon.read()}

    @app.get("/api/evaluation")
    def get_evaluation() -> dict[str, object]:
        return {
            "cases": [asdict(case) for case in evaluation_store.read_cases()],
            "report": evaluation_store.read_report(),
        }

    @app.put("/api/evaluation/cases")
    def update_evaluation_cases(request: EvaluationCasesRequest) -> dict[str, object]:
        cases = [EvaluationCase(**item.model_dump()) for item in request.cases]
        try:
            evaluation_store.replace_cases(cases)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {"cases": [asdict(case) for case in evaluation_store.read_cases()]}

    @app.post("/api/evaluation/run")
    def run_evaluation(request: EvaluationRunRequest) -> dict[str, object]:
        try:
            return asdict(evaluation_service.run(request.variants))
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/api/providers")
    def providers() -> list[dict[str, object]]:
        return [asdict(provider) for provider in resolved_providers.statuses()]

    @app.post("/api/exports", status_code=status.HTTP_201_CREATED)
    def export_clips(request: ExportRequest) -> dict[str, object]:
        try:
            exported = resolved_clips.export(
                request.name,
                [
                    ClipSelection(
                        video_id=selection.video_id,
                        start=selection.start,
                        end=selection.end,
                    )
                    for selection in request.selections
                ],
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {
            "name": exported.name,
            "duration": exported.duration,
            "created_at": exported.created_at,
            "url": f"/api/exports/{exported.name}",
        }

    @app.get("/api/exports/{filename}")
    def exported_clip(filename: str) -> FileResponse:
        if Path(filename).name != filename or Path(filename).suffix.lower() != ".mp4":
            raise HTTPException(status_code=404, detail="Export not found")
        path = resolved_settings.clips_dir / filename
        if not path.is_file() or path.parent.resolve() != resolved_settings.clips_dir.resolve():
            raise HTTPException(status_code=404, detail="Export not found")
        return FileResponse(path, media_type="video/mp4", filename=filename)

    return app
