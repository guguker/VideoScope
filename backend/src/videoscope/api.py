from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict
import hashlib
import logging
import mimetypes
from pathlib import Path
import re
import sqlite3
import stat
from typing import Callable, Protocol
from urllib.parse import quote
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import FileResponse, JSONResponse
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware

from videoscope.api_models import (
    EvaluationCasesRequest,
    EvaluationCasesResponse,
    EvaluationPayloadResponse,
    EvaluationReportResponse,
    EvaluationRunRequest,
    ErrorResponse,
    ExportRequest,
    ExportResponse,
    GlossaryRequest,
    GlossaryResponse,
    HealthResponse,
    JobResponse,
    ProviderResponse,
    RenameVideoRequest,
    SearchRequest,
    SearchResultResponse,
    VIDEO_ID_PATTERN,
    VideoResponse,
)
from videoscope.clips import ClipSelection, ClipService
from videoscope.config import AppSettings
from videoscope.evaluation import (
    EvaluationCase,
    EvaluationDataError,
    EvaluationService,
    EvaluationStore,
    evaluation_provider_snapshot,
    evaluation_revision,
    evaluation_runtime_revision,
    validate_evaluation_video_references,
)
from videoscope.media.ffmpeg import FFmpeg
from videoscope.media.uploads import UploadRejected, validate_upload
from videoscope.jobs import (
    JobState,
    JobTransitionError,
    VideoIndexJob,
    VideoIndexPlanSnapshot,
)
from videoscope.processing.coordinator import (
    VideoIndexCoordinator,
    VideoIndexPlanUnavailable,
)
from videoscope.providers.base import ProviderRegistry, ProviderState, StaticProvider
from videoscope.repository import AssetIdentityError, Repository, VideoRecord
from videoscope.runtime_lifecycle import ExclusiveRuntimeLock
from videoscope.search.service import SearchService
from videoscope.search.service import thumbnail_url_for_path
from videoscope.search.text_matching import SearchLexicon
from videoscope.search.vector_index import MemoryVectorIndex
from videoscope.security import RequestBodyLimitMiddleware, RateLimit, SlidingWindowLimiter


UPLOAD_CHUNK_SIZE = 1024 * 1024
UPLOAD_BODY_OVERHEAD_BYTES = 64 * 1024
LATEST_JOB_BATCH_SIZE = 1000
VIDEO_ID_RE = re.compile(VIDEO_ID_PATTERN)
JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
BINARY_FILE_SCHEMA = {"schema": {"type": "string", "format": "binary"}}
logger = logging.getLogger(__name__)


class ProcessingQueue(Protocol):
    def close(self) -> None: ...


class _ProcessingWakeup:
    def __init__(self, worker: ProcessingQueue) -> None:
        self.worker = worker

    def wake(self, video_id: str) -> None:
        wake = getattr(self.worker, "wake", None)
        if callable(wake):
            wake()
            return
        submit = getattr(self.worker, "submit", None)
        if not callable(submit):
            raise RuntimeError("video index dispatcher cannot be notified")
        submit(video_id)


def create_app(
    *,
    settings: AppSettings | None = None,
    repository: Repository | None = None,
    processing_queue: ProcessingQueue | None = None,
    search_service: SearchService | None = None,
    clip_service: ClipService | None = None,
    provider_registry: ProviderRegistry | None = None,
    video_index_plan_factory: Callable[[], VideoIndexPlanSnapshot] | None = None,
) -> FastAPI:
    resolved_settings = settings or AppSettings()
    owns_runtime = processing_queue is None
    owned_runtime = None
    ownership_lock: ExclusiveRuntimeLock | None = None
    resolved_repository: Repository | None = None
    resolved_queue: ProcessingQueue | None = None
    resolved_search: SearchService | None = None
    resolved_clips: ClipService | None = None
    resolved_providers: ProviderRegistry | None = None
    resolved_coordinator: VideoIndexCoordinator | None = None
    evaluation_service: EvaluationService | None = None
    evaluation_store = EvaluationStore(
        resolved_settings.evaluation_cases_path,
        resolved_settings.evaluation_report_path,
    )

    if not owns_runtime:
        from videoscope.runtime import create_indexing_specifications

        resolved_settings.ensure_directories()
        resolved_repository = repository or Repository(resolved_settings.database_path)
        resolved_repository.initialize()
        resolved_queue = processing_queue
        resolved_search = search_service or SearchService(
            resolved_repository,
            MemoryVectorIndex(),
            lexicon=SearchLexicon(resolved_settings.glossary_path),
            specification_resolver=lambda: create_indexing_specifications(
                resolved_settings
            ),
            thumbnails_dir=resolved_settings.thumbnails_dir,
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

    def bind_components(
        target_app: FastAPI,
        *,
        active_repository: Repository,
        active_queue: ProcessingQueue,
        active_search: SearchService,
        active_clips: ClipService,
        active_providers: ProviderRegistry,
        active_video_index_plan_factory: (
            Callable[[], VideoIndexPlanSnapshot] | None
        ) = None,
    ) -> None:
        nonlocal resolved_repository, resolved_queue, resolved_search
        nonlocal resolved_clips, resolved_providers, resolved_coordinator
        nonlocal evaluation_service
        resolved_repository = active_repository
        resolved_queue = active_queue
        resolved_search = active_search
        resolved_clips = active_clips
        resolved_providers = active_providers
        if video_index_plan_factory is not None:
            plan_factory = video_index_plan_factory
        elif active_video_index_plan_factory is not None:
            plan_factory = active_video_index_plan_factory
        else:
            from videoscope.runtime import create_video_index_plan_snapshot

            plan_factory = lambda: create_video_index_plan_snapshot(
                resolved_settings
            )
        resolved_coordinator = VideoIndexCoordinator(
            active_repository,
            plan_factory=plan_factory,
            wakeup=_ProcessingWakeup(active_queue),
        )
        resolved_lexicon = active_search.lexicon or SearchLexicon(
            resolved_settings.glossary_path
        )
        active_search.lexicon = resolved_lexicon
        evaluation_providers = evaluation_provider_snapshot(active_providers)
        evaluation_service = EvaluationService(
            active_search,
            evaluation_store,
            repository=active_repository,
            runtime_revision=lambda: evaluation_runtime_revision(
                active_search,
                evaluation_providers,
                resolved_settings,
            ),
        )
        target_app.state.repository = active_repository
        target_app.state.processing_queue = active_queue
        target_app.state.search_service = active_search
        target_app.state.search_lexicon = resolved_lexicon
        target_app.state.evaluation_service = evaluation_service
        target_app.state.clip_service = active_clips
        target_app.state.provider_registry = active_providers
        target_app.state.video_index_coordinator = resolved_coordinator

    def active_repository() -> Repository:
        if resolved_repository is None:
            raise RuntimeError("VideoScope runtime is not started")
        return resolved_repository

    def active_coordinator() -> VideoIndexCoordinator:
        if resolved_coordinator is None:
            raise RuntimeError("VideoScope runtime is not started")
        return resolved_coordinator

    def active_search() -> SearchService:
        if resolved_search is None:
            raise RuntimeError("VideoScope runtime is not started")
        return resolved_search

    def active_lexicon() -> SearchLexicon:
        lexicon = active_search().lexicon
        if lexicon is None:
            raise RuntimeError("VideoScope search lexicon is not initialized")
        return lexicon

    def active_clips() -> ClipService:
        if resolved_clips is None:
            raise RuntimeError("VideoScope runtime is not started")
        return resolved_clips

    def active_providers() -> ProviderRegistry:
        if resolved_providers is None:
            raise RuntimeError("VideoScope runtime is not started")
        return resolved_providers

    def active_evaluation_service() -> EvaluationService:
        if evaluation_service is None:
            raise RuntimeError("VideoScope runtime is not started")
        return evaluation_service

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        nonlocal owned_runtime, ownership_lock
        if owns_runtime:
            ownership_lock = ExclusiveRuntimeLock(resolved_settings.data_dir)
            ownership_lock.acquire()
            try:
                resolved_settings.ensure_directories()
                active_repository = repository or Repository(
                    resolved_settings.database_path
                )
                active_repository.initialize()
                from videoscope.runtime import build_runtime

                owned_runtime = build_runtime(resolved_settings, active_repository)
                owned_runtime.runtime_lock = ownership_lock
                bind_components(
                    _app,
                    active_repository=active_repository,
                    active_queue=owned_runtime.queue,
                    active_search=search_service or owned_runtime.search,
                    active_clips=clip_service or owned_runtime.clips,
                    active_providers=provider_registry or owned_runtime.providers,
                    active_video_index_plan_factory=getattr(
                        owned_runtime,
                        "video_index_plan_factory",
                        None,
                    ),
                )
                owned_runtime.start()
            except Exception:
                if owned_runtime is None:
                    ownership_lock.close()
                else:
                    try:
                        stopped = owned_runtime.close()
                    except Exception:
                        logger.exception("VideoScope runtime startup cleanup failed")
                        stopped = False
                    if stopped:
                        ownership_lock.close()
                raise
        try:
            yield
        finally:
            if owned_runtime is not None:
                try:
                    stopped = owned_runtime.close()
                except Exception:
                    logger.exception("VideoScope runtime shutdown failed")
                    stopped = False
                if stopped:
                    if ownership_lock is not None:
                        try:
                            ownership_lock.close()
                        except Exception:
                            logger.exception("VideoScope ownership lock release failed")
                else:
                    logger.warning(
                        "VideoScope runtime is finishing current work in its ownership reaper"
                    )
            else:
                assert resolved_queue is not None
                resolved_queue.close()

    app = FastAPI(
        title="VideoScope API",
        version="0.1.0",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
        responses={
            403: {"model": ErrorResponse, "description": "Request origin or host is not trusted"},
            429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        },
    )
    allowed_hosts = {"127.0.0.1", "localhost", resolved_settings.host}
    if processing_queue is not None:
        allowed_hosts.add("testserver")
    if ":" in resolved_settings.host:
        # Starlette currently parses a bracketed IPv6 Host header to this token.
        allowed_hosts.add("[")
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=sorted(allowed_hosts),
        www_redirect=False,
    )
    app.add_middleware(
        RequestBodyLimitMiddleware,
        max_body_bytes=(
            resolved_settings.max_upload_bytes + UPLOAD_BODY_OVERHEAD_BYTES
        ),
    )
    app.state.settings = resolved_settings
    if not owns_runtime:
        assert resolved_repository is not None
        assert resolved_queue is not None
        assert resolved_search is not None
        assert resolved_clips is not None
        assert resolved_providers is not None
        bind_components(
            app,
            active_repository=resolved_repository,
            active_queue=resolved_queue,
            active_search=resolved_search,
            active_clips=resolved_clips,
            active_providers=resolved_providers,
        )
    limiter = SlidingWindowLimiter()
    trusted_origins = {
        f"http://{resolved_settings.host}:{resolved_settings.port}",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    }

    def public_video_error(record: VideoRecord) -> str | None:
        if not record.error:
            return None
        if record.status == "failed":
            return "Не удалось обработать видео. Подробности записаны в журнале сервера."
        return "Некоторые необязательные этапы индексации завершились с предупреждением."

    def job_summary_payload(job: VideoIndexJob) -> dict[str, object]:
        return {
            "job_id": job.job_id,
            "intent": job.intent.value,
            "state": job.state.value,
            "progress": job.progress,
            "stage": job.stage,
            "attempt": job.attempt,
            "cancel_requested_at": job.cancel_requested_at,
            "error_code": job.error_code,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "updated_at": job.updated_at,
        }

    def job_payload(job: VideoIndexJob) -> dict[str, object]:
        return {
            **job_summary_payload(job),
            "video_id": job.video_id,
            "retry_of_job_id": job.retry_of_job_id,
        }

    def video_payload(
        record: VideoRecord,
        *,
        latest_job: VideoIndexJob | None = None,
    ) -> dict[str, object]:
        return {
            "id": record.id,
            "original_name": record.original_name,
            "display_name": record.display_name,
            "size_bytes": record.size_bytes,
            "status": record.status,
            "progress": record.progress,
            "stage": record.stage,
            "duration": record.duration,
            "width": record.width,
            "height": record.height,
            "fps": record.fps,
            "error": public_video_error(record),
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "media_url": f"/api/videos/{record.id}/media",
            "thumbnail_url": thumbnail_url_for_path(
                record.id,
                record.thumbnail_path,
                resolved_settings.thumbnails_dir,
            ),
            "latest_job": (
                job_summary_payload(latest_job)
                if latest_job is not None
                else None
            ),
        }

    def latest_jobs_for_records(
        repository: Repository,
        records: list[VideoRecord],
    ) -> dict[str, VideoIndexJob]:
        latest: dict[str, VideoIndexJob] = {}
        video_ids = tuple(record.id for record in records)
        for offset in range(0, len(video_ids), LATEST_JOB_BATCH_SIZE):
            latest.update(
                repository.get_latest_video_index_jobs(
                    video_ids[offset : offset + LATEST_JOB_BATCH_SIZE]
                )
            )
        return latest

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

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "videoscope"}

    @app.get("/api/videos", response_model=list[VideoResponse])
    def list_videos() -> list[dict[str, object]]:
        repository = active_repository()
        records = repository.list_videos()
        latest = latest_jobs_for_records(repository, records)
        return [
            video_payload(record, latest_job=latest.get(record.id))
            for record in records
        ]

    @app.get("/api/videos/{video_id}", response_model=VideoResponse)
    def get_video(video_id: str) -> dict[str, object]:
        record = active_repository().get_video(video_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Video not found")
        latest = active_repository().get_latest_video_index_jobs((record.id,))
        return video_payload(record, latest_job=latest.get(record.id))

    @app.patch("/api/videos/{video_id}", response_model=VideoResponse)
    def rename_video(video_id: str, request: RenameVideoRequest) -> dict[str, object]:
        repository = active_repository()
        record = repository.get_video(video_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Video not found")
        name = " ".join(request.name.split()).strip(" .")
        if not name:
            raise HTTPException(status_code=422, detail="Video name is empty")
        updated = repository.update_video(video_id, display_name=name)
        latest = repository.get_latest_video_index_jobs((updated.id,))
        return video_payload(updated, latest_job=latest.get(updated.id))

    @app.post(
        "/api/videos",
        status_code=status.HTTP_202_ACCEPTED,
        response_model=VideoResponse,
        responses={
            413: {"model": ErrorResponse, "description": "Upload body or file is too large"},
            415: {"model": ErrorResponse, "description": "Unsupported video extension"},
            422: {"model": ErrorResponse, "description": "Multipart upload must contain one file"},
        },
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "required": ["file"],
                            "properties": {
                                "file": {"type": "string", "format": "binary"}
                            },
                        }
                    }
                },
            }
        },
    )
    async def upload_video(request: Request, response: Response) -> dict[str, object]:
        try:
            form = await request.form(max_files=1, max_fields=0)
        except StarletteHTTPException as error:
            raise HTTPException(
                status_code=422,
                detail="Upload must contain exactly one file field",
            ) from error
        items = form.multi_items()
        if (
            len(items) != 1
            or items[0][0] != "file"
            or not isinstance(items[0][1], StarletteUploadFile)
        ):
            await form.close()
            raise HTTPException(
                status_code=422,
                detail="Upload must contain exactly one file field",
            )
        file = items[0][1]
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
        source_digest = hashlib.sha256()
        try:
            with temporary.open("wb") as output:
                while chunk := await file.read(UPLOAD_CHUNK_SIZE):
                    size_bytes += len(chunk)
                    if size_bytes > resolved_settings.max_upload_bytes:
                        raise HTTPException(status_code=413, detail="Video is too large")
                    source_digest.update(chunk)
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
        except Exception:
            temporary.unlink(missing_ok=True)
            destination.unlink(missing_ok=True)
            raise
        finally:
            await form.close()

        try:
            record, job = active_coordinator().create_ingest(
                video_id=video_id,
                original_name=filename,
                stored_name=stored_name,
                media_path=str(destination),
                size_bytes=size_bytes,
                source_sha256=source_digest.hexdigest(),
            )
        except VideoIndexPlanUnavailable as error:
            destination.unlink(missing_ok=True)
            logger.warning("Video index plan is unavailable", exc_info=error)
            raise HTTPException(
                status_code=503,
                detail="Indexing runtime is unavailable",
            ) from error
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        response.headers["Location"] = f"/api/jobs/{job.job_id}"
        return video_payload(record, latest_job=job)

    @app.get(
        "/api/videos/{video_id}/media",
        response_class=FileResponse,
        responses={
            200: {
                "description": "Original video bytes",
                "content": {"application/octet-stream": BINARY_FILE_SCHEMA},
            },
            404: {"description": "Video or media file not found"},
        },
    )
    def video_media(video_id: str) -> FileResponse:
        record = active_repository().get_video(video_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Video not found")
        media_root = resolved_settings.media_dir.resolve()
        path = Path(record.media_path).resolve()
        if not path.is_file() or path.parent != media_root:
            raise HTTPException(status_code=404, detail="Media not found")
        media_type = mimetypes.guess_type(record.original_name)[0] or "application/octet-stream"
        return FileResponse(path, media_type=media_type)

    @app.get(
        "/api/thumbnails/{video_id}/{relative_path:path}",
        response_class=FileResponse,
        responses={
            200: {
                "description": "JPEG thumbnail bytes",
                "content": {"image/jpeg": BINARY_FILE_SCHEMA},
            },
            404: {"description": "Thumbnail not found"},
        },
    )
    def thumbnail(video_id: str, relative_path: str) -> FileResponse:
        if (
            not VIDEO_ID_RE.fullmatch(video_id)
            or not relative_path
            or "\x00" in relative_path
            or "\\" in relative_path
            or relative_path.startswith("/")
        ):
            raise HTTPException(status_code=404, detail="Thumbnail not found")
        parts = relative_path.split("/")
        if any(part in {"", ".", ".."} for part in parts):
            raise HTTPException(status_code=404, detail="Thumbnail not found")
        try:
            configured_root = resolved_settings.thumbnails_dir
            if configured_root.is_symlink():
                raise ValueError("thumbnail root must not be a symlink")
            thumbnails_root = configured_root.resolve(strict=True)
            video_root_path = configured_root / video_id
            if video_root_path.is_symlink():
                raise ValueError("video thumbnail root must not be a symlink")
            video_root = video_root_path.resolve(strict=True)
            if video_root.parent != thumbnails_root:
                raise ValueError("video thumbnail root escaped its storage root")
            candidate_path = video_root_path
            for part in parts:
                candidate_path = candidate_path / part
                if candidate_path.is_symlink():
                    raise ValueError("thumbnail path contains a symlink")
            candidate = candidate_path.resolve(strict=True)
            relative = candidate.relative_to(video_root)
            if not relative.parts or candidate == video_root:
                raise ValueError("thumbnail must be inside the video root")
            if not stat.S_ISREG(candidate.lstat().st_mode):
                raise ValueError("thumbnail is not a regular file")
        except (OSError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=404, detail="Thumbnail not found") from error
        return FileResponse(candidate, media_type="image/jpeg")

    @app.post(
        "/api/videos/{video_id}/reindex",
        status_code=status.HTTP_202_ACCEPTED,
        response_model=VideoResponse,
    )
    def reindex_video(video_id: str, response: Response) -> dict[str, object]:
        repository = active_repository()
        if repository.get_video(video_id) is None:
            raise HTTPException(status_code=404, detail="Video not found")
        try:
            job = active_coordinator().enqueue_reindex(video_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Video not found") from error
        except (AssetIdentityError, JobTransitionError, sqlite3.IntegrityError) as error:
            raise HTTPException(
                status_code=409,
                detail="Video already has incompatible active indexing work",
            ) from error
        except VideoIndexPlanUnavailable as error:
            logger.warning("Video reindex plan is unavailable", exc_info=error)
            raise HTTPException(
                status_code=503,
                detail="Indexing runtime is unavailable",
            ) from error
        record = repository.get_video(video_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Video not found")
        response.headers["Location"] = f"/api/jobs/{job.job_id}"
        return video_payload(record, latest_job=job)

    def public_job(job_id: str) -> VideoIndexJob:
        if JOB_ID_RE.fullmatch(job_id) is None:
            raise HTTPException(status_code=404, detail="Job not found")
        try:
            job = active_coordinator().get_job(job_id)
        except ValueError as error:
            logger.exception("Persisted video index job is invalid")
            raise HTTPException(
                status_code=503,
                detail="Job state is unavailable",
            ) from error
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @app.get("/api/jobs/{job_id}", response_model=JobResponse)
    def get_job(job_id: str) -> dict[str, object]:
        return job_payload(public_job(job_id))

    @app.post(
        "/api/jobs/{job_id}/cancel",
        response_model=JobResponse,
        responses={409: {"model": ErrorResponse, "description": "Terminal job"}},
    )
    def cancel_job(job_id: str, response: Response) -> dict[str, object]:
        current = public_job(job_id)
        if current.state is JobState.CANCELLED:
            response.status_code = status.HTTP_200_OK
            return job_payload(current)
        if current.state in {JobState.COMPLETE, JobState.FAILED}:
            raise HTTPException(
                status_code=409,
                detail="Job is already terminal",
            )
        try:
            job = active_coordinator().request_cancellation(job_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Job not found") from error
        except JobTransitionError as error:
            raise HTTPException(
                status_code=409,
                detail="Job is already terminal",
            ) from error
        response.status_code = (
            status.HTTP_202_ACCEPTED
            if job.state is JobState.RUNNING
            else status.HTTP_200_OK
        )
        return job_payload(job)

    @app.post(
        "/api/jobs/{job_id}/retry",
        status_code=status.HTTP_202_ACCEPTED,
        response_model=JobResponse,
        responses={409: {"model": ErrorResponse, "description": "Job state conflict"}},
    )
    def retry_job_endpoint(job_id: str, response: Response) -> dict[str, object]:
        public_job(job_id)
        try:
            child = active_coordinator().retry(job_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Job not found") from error
        except (JobTransitionError, sqlite3.IntegrityError) as error:
            raise HTTPException(
                status_code=409,
                detail="Job cannot be retried",
            ) from error
        except ValueError as error:
            logger.exception("Persisted video index retry is invalid")
            raise HTTPException(
                status_code=503,
                detail="Job state is unavailable",
            ) from error
        response.headers["Location"] = f"/api/jobs/{child.job_id}"
        return job_payload(child)

    @app.post("/api/search", response_model=list[SearchResultResponse])
    def search(request: SearchRequest) -> list[dict[str, object]]:
        try:
            results = active_search().search(
                request.query,
                video_ids=request.video_ids,
                limit=request.limit,
                use_lighthouse=request.use_lighthouse,
                mode=request.mode,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return [asdict(result) for result in results]

    @app.get("/api/search/glossary", response_model=GlossaryResponse)
    def get_search_glossary() -> dict[str, object]:
        return {"entries": active_lexicon().read()}

    @app.put("/api/search/glossary", response_model=GlossaryResponse)
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
        lexicon = active_lexicon()
        lexicon.replace(request.entries)
        return {"entries": lexicon.read()}

    @app.get("/api/evaluation", response_model=EvaluationPayloadResponse)
    def get_evaluation() -> dict[str, object]:
        try:
            cases = evaluation_store.read_cases()
            runtime_revision = active_evaluation_service().current_runtime_revision()
            return {
                "cases": [asdict(case) for case in cases],
                "runtime_revision": runtime_revision,
                "evaluation_revision": evaluation_revision(cases, runtime_revision),
                "report": evaluation_store.read_report(),
            }
        except EvaluationDataError as error:
            logger.exception("Persisted evaluation cases are invalid")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Evaluation data is invalid",
            ) from error

    @app.put("/api/evaluation/cases", response_model=EvaluationCasesResponse)
    def update_evaluation_cases(request: EvaluationCasesRequest) -> dict[str, object]:
        cases = [EvaluationCase(**item.model_dump()) for item in request.cases]
        try:
            validate_evaluation_video_references(cases, active_repository())
            evaluation_store.replace_cases(cases)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        persisted_cases = evaluation_store.read_cases()
        runtime_revision = active_evaluation_service().current_runtime_revision()
        return {
            "cases": [asdict(case) for case in persisted_cases],
            "runtime_revision": runtime_revision,
            "evaluation_revision": evaluation_revision(
                persisted_cases,
                runtime_revision,
            ),
        }

    @app.post("/api/evaluation/run", response_model=EvaluationReportResponse)
    def run_evaluation(request: EvaluationRunRequest) -> dict[str, object]:
        try:
            return asdict(active_evaluation_service().run(list(request.variants)))
        except EvaluationDataError as error:
            logger.exception("Persisted evaluation cases are invalid")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Evaluation data is invalid",
            ) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/api/providers", response_model=list[ProviderResponse])
    def providers() -> list[dict[str, object]]:
        return [asdict(provider) for provider in active_providers().statuses()]

    @app.post(
        "/api/exports",
        status_code=status.HTTP_201_CREATED,
        response_model=ExportResponse,
    )
    def export_clips(request: ExportRequest, response: Response) -> dict[str, object]:
        try:
            exported = active_clips().export(
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
        location = f"/api/exports/{quote(exported.name, safe='')}"
        response.headers["Location"] = location
        return {
            "name": exported.name,
            "duration": exported.duration,
            "created_at": exported.created_at,
            "url": location,
        }

    @app.get(
        "/api/exports/{filename}",
        response_class=FileResponse,
        responses={
            200: {
                "description": "Exported MP4 bytes",
                "content": {"video/mp4": BINARY_FILE_SCHEMA},
            },
            404: {"description": "Export not found"},
        },
    )
    def exported_clip(filename: str) -> FileResponse:
        if Path(filename).name != filename or Path(filename).suffix.lower() != ".mp4":
            raise HTTPException(status_code=404, detail="Export not found")
        clips_root = resolved_settings.clips_dir.resolve()
        path = (clips_root / filename).resolve()
        if not path.is_file() or path.parent != clips_root:
            raise HTTPException(status_code=404, detail="Export not found")
        return FileResponse(path, media_type="video/mp4", filename=filename)

    return app
