import hashlib
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

from fastapi.testclient import TestClient
import pytest

from videoscope.artifacts import StageState
from videoscope.api import UPLOAD_BODY_OVERHEAD_BYTES, create_app
from videoscope.config import AppSettings
from videoscope.providers.base import ProviderRegistry
from videoscope.repository import Repository
from videoscope.runtime import create_indexing_specifications
from videoscope.search.service import SearchService
from videoscope.search.vector_index import MemoryVectorIndex


class RecordingQueue:
    def __init__(self) -> None:
        self.video_ids: list[str] = []

    def submit(self, video_id: str) -> None:
        self.video_ids.append(video_id)

    def close(self) -> None:
        return None


def settings_for(tmp_path: Path) -> AppSettings:
    return AppSettings(data_dir=tmp_path / "data", max_upload_bytes=1_024)


def test_owned_runtime_starts_and_closes_with_the_app_lifespan(
    tmp_path,
    monkeypatch,
) -> None:
    settings = settings_for(tmp_path)
    repository = Repository(settings.database_path)
    repository.initialize()
    events: list[str] = []
    runtime = SimpleNamespace(
        queue=RecordingQueue(),
        search=SearchService(repository, MemoryVectorIndex()),
        clips=object(),
        providers=object(),
        start=lambda: events.append("start"),
        close=lambda: events.append("close") or True,
    )
    monkeypatch.setattr(
        "videoscope.runtime.build_runtime",
        lambda _settings, _repository: runtime,
    )

    app = create_app(settings=settings, repository=repository)
    assert events == []

    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert events == ["start"]
        assert client.get("/api/videos").status_code == 200

    assert events == ["start", "close"]


def test_owned_runtime_acquires_data_lock_before_initialization(
    tmp_path,
    monkeypatch,
) -> None:
    settings = settings_for(tmp_path)
    events: list[str] = []
    original_ensure_directories = AppSettings.ensure_directories
    original_initialize = Repository.initialize

    class RecordingLock:
        is_acquired = False

        def __init__(self, data_dir: Path) -> None:
            assert data_dir == settings.data_dir

        def acquire(self) -> None:
            self.is_acquired = True
            events.append("lock")

        def close(self) -> None:
            self.is_acquired = False
            events.append("unlock")

    def ensure_directories(candidate: AppSettings) -> None:
        events.append("directories")
        original_ensure_directories(candidate)

    def initialize(candidate: Repository) -> None:
        events.append("repository")
        original_initialize(candidate)

    def build_runtime(_settings, repository):  # type: ignore[no-untyped-def]
        events.append("build")
        return SimpleNamespace(
            queue=RecordingQueue(),
            search=SearchService(repository, MemoryVectorIndex()),
            clips=object(),
            providers=ProviderRegistry([]),
            runtime_lock=None,
            start=lambda: events.append("start"),
            close=lambda: events.append("close") or True,
        )

    monkeypatch.setattr(AppSettings, "ensure_directories", ensure_directories)
    monkeypatch.setattr(Repository, "initialize", initialize)
    monkeypatch.setattr("videoscope.api.ExclusiveRuntimeLock", RecordingLock)
    monkeypatch.setattr("videoscope.runtime.build_runtime", build_runtime)

    app = create_app(settings=settings)
    assert events == []

    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert events[:5] == ["lock", "directories", "repository", "build", "start"]
        assert client.get("/api/videos").status_code == 200

    assert events[-2:] == ["close", "unlock"]


def test_owned_runtime_lock_refusal_prevents_data_initialization(
    tmp_path,
    monkeypatch,
) -> None:
    settings = settings_for(tmp_path)
    events: list[str] = []

    class RefusingLock:
        def __init__(self, _data_dir: Path) -> None:
            return None

        def acquire(self) -> None:
            events.append("lock")
            raise RuntimeError("already owned")

        def close(self) -> None:
            events.append("unlock")

    monkeypatch.setattr("videoscope.api.ExclusiveRuntimeLock", RefusingLock)
    monkeypatch.setattr(
        AppSettings,
        "ensure_directories",
        lambda _settings: events.append("directories"),
    )
    monkeypatch.setattr(
        Repository,
        "initialize",
        lambda _repository: events.append("repository"),
    )

    app = create_app(settings=settings)
    with pytest.raises(RuntimeError, match="already owned"):
        with TestClient(app, base_url="http://127.0.0.1"):
            pass

    assert events == ["lock"]


def ready_repository(settings: AppSettings, *, duration: float = 30.0) -> Repository:
    repository = Repository(settings.database_path)
    repository.initialize()
    repository.create_video(
        video_id="video-1",
        original_name="match.mp4",
        stored_name="match.mp4",
        media_path=str(settings.media_dir / "match.mp4"),
        size_bytes=1,
    )
    repository.update_video(
        "video-1",
        status="ready",
        stage="ready",
        duration=duration,
    )
    return repository


def test_upload_creates_library_item_and_queues_processing(tmp_path) -> None:
    queue = RecordingQueue()
    app = create_app(settings=settings_for(tmp_path), processing_queue=queue)

    with TestClient(app) as client:
        response = client.post(
            "/api/videos",
            files={"file": ("match.mp4", b"synthetic video bytes", "video/mp4")},
        )
        listing = client.get("/api/videos")

    assert response.status_code == 202
    video = response.json()
    assert video["original_name"] == "match.mp4"
    assert video["status"] == "queued"
    assert queue.video_ids == [video["id"]]
    assert listing.json()[0]["id"] == video["id"]
    stored_files = list((tmp_path / "data" / "media").iterdir())
    assert len(stored_files) == 1
    assert stored_files[0].suffix == ".mp4"


def test_upload_hashes_stream_while_writing_and_persists_asset_identity(
    tmp_path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    settings = settings_for(tmp_path)
    repository = Repository(settings.database_path)
    repository.initialize()
    payload = b"content whose digest identifies the asset"
    opened_media_modes: list[str] = []
    original_open = Path.open

    def tracking_open(path, mode="r", *args, **kwargs):  # type: ignore[no-untyped-def]
        try:
            path.resolve().relative_to(settings.data_dir.resolve())
        except (FileNotFoundError, ValueError):
            pass
        else:
            opened_media_modes.append(mode)
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracking_open)
    app = create_app(
        settings=settings,
        repository=repository,
        processing_queue=RecordingQueue(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/videos",
            files={"file": ("match.mp4", payload, "video/mp4")},
        )

    assert response.status_code == 202
    video_id = response.json()["id"]
    asset = repository.get_video_asset(video_id)
    assert asset is not None
    assert asset.sha256 == hashlib.sha256(payload).hexdigest()
    assert asset.size_bytes == len(payload)
    assert not any("r" in mode for mode in opened_media_modes)


def test_upload_removes_moved_media_when_asset_transaction_is_rejected(tmp_path) -> None:
    class RejectingAssetRepository(Repository):
        def create_video_with_asset(self, **kwargs):  # type: ignore[no-untyped-def]
            kwargs["source_sha256"] = "invalid-digest"
            return super().create_video_with_asset(**kwargs)

    settings = settings_for(tmp_path)
    repository = RejectingAssetRepository(settings.database_path)
    repository.initialize()
    app = create_app(
        settings=settings,
        repository=repository,
        processing_queue=RecordingQueue(),
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/videos",
            files={"file": ("match.mp4", b"moved before database failure", "video/mp4")},
        )

    assert response.status_code == 500
    assert list(settings.media_dir.iterdir()) == []
    assert list(settings.temp_dir.iterdir()) == []
    assert repository.list_videos() == []


def test_upload_removes_temporary_media_when_atomic_move_fails(
    tmp_path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    settings = settings_for(tmp_path)
    original_replace = Path.replace

    def reject_upload_move(path, target):  # type: ignore[no-untyped-def]
        if path.suffix == ".upload":
            raise OSError("simulated atomic move failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", reject_upload_move)
    app = create_app(settings=settings, processing_queue=RecordingQueue())

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/videos",
            files={"file": ("match.mp4", b"temporary upload", "video/mp4")},
        )

    assert response.status_code == 500
    assert list(settings.media_dir.iterdir()) == []
    assert list(settings.temp_dir.iterdir()) == []


def test_video_api_contract_does_not_expose_storage_implementation(tmp_path) -> None:
    app = create_app(settings=settings_for(tmp_path), processing_queue=RecordingQueue())

    with TestClient(app) as client:
        response = client.post(
            "/api/videos",
            files={"file": ("match.mp4", b"synthetic video bytes", "video/mp4")},
        )

    assert response.status_code == 202
    payload = response.json()
    assert "stored_name" not in payload
    assert "media_path" not in payload
    assert "thumbnail_path" not in payload
    assert "asset_id" not in payload
    assert payload["media_url"].startswith("/api/videos/")


def test_video_api_does_not_expose_internal_processing_error(tmp_path) -> None:
    settings = settings_for(tmp_path)
    repository = Repository(settings.database_path)
    repository.initialize()
    repository.create_video(
        video_id="video-1",
        original_name="match.mp4",
        stored_name="stored.mp4",
        media_path="/Users/example/private/stored.mp4",
        size_bytes=5,
    )
    repository.update_video(
        "video-1",
        status="failed",
        stage="failed",
        error="ffprobe failed for /Users/example/private/stored.mp4",
    )
    app = create_app(
        settings=settings,
        repository=repository,
        processing_queue=RecordingQueue(),
    )

    with TestClient(app) as client:
        payload = client.get("/api/videos/video-1").json()

    assert payload["error"] == "Не удалось обработать видео. Подробности записаны в журнале сервера."
    assert "/Users/" not in payload["error"]


def test_request_models_reject_unknown_fields(tmp_path) -> None:
    app = create_app(settings=settings_for(tmp_path), processing_queue=RecordingQueue())

    with TestClient(app) as client:
        uploaded = client.post(
            "/api/videos",
            files={"file": ("match.mp4", b"video", "video/mp4")},
        ).json()
        response = client.patch(
            f"/api/videos/{uploaded['id']}",
            json={"name": "Матч", "media_path": "/tmp/override.mp4"},
        )

    assert response.status_code == 422


def test_openapi_describes_structured_response_contracts(tmp_path) -> None:
    app = create_app(settings=settings_for(tmp_path), processing_queue=RecordingQueue())

    schema = app.openapi()

    videos = schema["paths"]["/api/videos"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
    search = schema["paths"]["/api/search"]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
    providers = schema["paths"]["/api/providers"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert videos["items"]["$ref"].endswith("/VideoResponse")
    assert search["items"]["$ref"].endswith("/SearchResultResponse")
    assert providers["items"]["$ref"].endswith("/ProviderResponse")
    assert "/api/videos/{video_id}/segments" not in schema["paths"]
    assert schema["paths"]["/api/search"]["post"]["responses"]["403"]["content"][
        "application/json"
    ]["schema"]["$ref"].endswith("/ErrorResponse")
    assert "429" in schema["paths"]["/api/search"]["post"]["responses"]
    for path in (
        "/api/videos/{video_id}/media",
        "/api/thumbnails/{video_id}/{relative_path}",
        "/api/exports/{filename}",
    ):
        content = schema["paths"][path]["get"]["responses"]["200"]["content"]
        assert "application/json" not in content
        assert any(
            media["schema"] == {"type": "string", "format": "binary"}
            for media in content.values()
        )


def test_upload_rejects_invalid_container_without_leaving_a_file(tmp_path) -> None:
    app = create_app(settings=settings_for(tmp_path), processing_queue=RecordingQueue())

    with TestClient(app) as client:
        response = client.post(
            "/api/videos",
            files={"file": ("payload.exe", b"not a video", "application/octet-stream")},
        )

    assert response.status_code == 415
    assert list((tmp_path / "data" / "media").iterdir()) == []


def test_upload_enforces_streamed_size_limit(tmp_path) -> None:
    app = create_app(settings=settings_for(tmp_path), processing_queue=RecordingQueue())

    with TestClient(app) as client:
        response = client.post(
            "/api/videos",
            files={"file": ("large.mp4", b"x" * 1_025, "video/mp4")},
        )

    assert response.status_code == 413
    assert list((tmp_path / "data" / "media").iterdir()) == []


def test_upload_body_limit_ignores_a_false_small_content_length(tmp_path) -> None:
    app = create_app(settings=settings_for(tmp_path), processing_queue=RecordingQueue())
    oversized = b"x" * (1_024 + UPLOAD_BODY_OVERHEAD_BYTES + 1)

    with TestClient(app) as client:
        response = client.post(
            "/api/videos",
            headers={"Content-Length": "1"},
            files={"file": ("large.mp4", oversized, "video/mp4")},
        )

    assert response.status_code == 413
    assert list((tmp_path / "data" / "media").iterdir()) == []


def test_upload_rejects_extra_multipart_files(tmp_path) -> None:
    app = create_app(settings=settings_for(tmp_path), processing_queue=RecordingQueue())

    with TestClient(app) as client:
        response = client.post(
            "/api/videos",
            files=[
                ("file", ("first.mp4", b"video", "video/mp4")),
                ("extra", ("second.mp4", b"video", "video/mp4")),
            ],
        )

    assert response.status_code == 422


def test_video_media_route_cannot_escape_media_directory(tmp_path) -> None:
    app = create_app(settings=settings_for(tmp_path), processing_queue=RecordingQueue())

    with TestClient(app) as client:
        response = client.get("/api/videos/../../etc/passwd/media")

    assert response.status_code in {404, 422}


def test_video_media_route_rejects_symlink_outside_media_directory(tmp_path) -> None:
    settings = settings_for(tmp_path)
    settings.ensure_directories()
    secret = settings.data_dir / "secret.mp4"
    secret.write_bytes(b"not public")
    link = settings.media_dir / "linked.mp4"
    link.symlink_to(secret)
    repository = Repository(settings.database_path)
    repository.initialize()
    repository.create_video(
        video_id="video-1",
        original_name="linked.mp4",
        stored_name=link.name,
        media_path=str(link),
        size_bytes=secret.stat().st_size,
    )
    app = create_app(
        settings=settings,
        repository=repository,
        processing_queue=RecordingQueue(),
    )

    with TestClient(app) as client:
        response = client.get("/api/videos/video-1/media")

    assert response.status_code == 404
    assert response.content != b"not public"


def test_thumbnail_route_cannot_escape_with_encoded_video_id(tmp_path) -> None:
    settings = settings_for(tmp_path)
    settings.ensure_directories()
    secret = settings.data_dir / "secret.jpg"
    secret.write_bytes(b"not public")
    app = create_app(settings=settings, processing_queue=RecordingQueue())

    with TestClient(app) as client:
        response = client.get("/api/thumbnails/%2e%2e/secret.jpg")

    assert response.status_code == 404
    assert response.content != b"not public"


def test_thumbnail_route_serves_nested_generation_and_legacy_flat_paths(tmp_path) -> None:
    settings = settings_for(tmp_path)
    settings.ensure_directories()
    repository = ready_repository(settings)
    generation = settings.thumbnails_dir / "video-1" / "generations" / "generation-1"
    generation.mkdir(parents=True)
    nested = generation / "scene name.jpg"
    nested.write_bytes(b"nested jpeg")
    flat = settings.thumbnails_dir / "video-1" / "legacy.jpg"
    flat.write_bytes(b"legacy jpeg")
    repository.update_video("video-1", thumbnail_path=str(nested))
    app = create_app(
        settings=settings,
        repository=repository,
        processing_queue=RecordingQueue(),
    )

    with TestClient(app) as client:
        nested_response = client.get(
            "/api/thumbnails/video-1/generations/generation-1/scene%20name.jpg"
        )
        flat_response = client.get("/api/thumbnails/video-1/legacy.jpg")
        video = client.get("/api/videos/video-1")

    assert nested_response.status_code == 200
    assert nested_response.content == b"nested jpeg"
    assert flat_response.status_code == 200
    assert flat_response.content == b"legacy jpeg"
    assert video.json()["thumbnail_url"] == (
        "/api/thumbnails/video-1/generations/generation-1/scene%20name.jpg"
    )


@pytest.mark.parametrize(
    "relative_path",
    [
        "%2e%2e/secret.jpg",
        "generations/%2e%2e/secret.jpg",
        "%2Fetc%2Fpasswd",
        "generations%5Csecret.jpg",
        "generations//secret.jpg",
        ".",
    ],
)
def test_thumbnail_route_rejects_unsafe_nested_relative_paths(
    tmp_path,
    relative_path: str,
) -> None:
    settings = settings_for(tmp_path)
    settings.ensure_directories()
    secret = settings.data_dir / "secret.jpg"
    secret.write_bytes(b"not public")
    app = create_app(settings=settings, processing_queue=RecordingQueue())

    with TestClient(app) as client:
        response = client.get(f"/api/thumbnails/video-1/{relative_path}")

    assert response.status_code == 404
    assert response.content != b"not public"


def test_thumbnail_route_rejects_symlink_in_any_nested_component(tmp_path) -> None:
    settings = settings_for(tmp_path)
    settings.ensure_directories()
    secret_dir = settings.data_dir / "private"
    secret_dir.mkdir()
    (secret_dir / "secret.jpg").write_bytes(b"not public")
    video_root = settings.thumbnails_dir / "video-1"
    video_root.mkdir()
    (video_root / "linked-dir").symlink_to(secret_dir, target_is_directory=True)
    safe_dir = video_root / "generations" / "generation-1"
    safe_dir.mkdir(parents=True)
    (safe_dir / "linked-file.jpg").symlink_to(secret_dir / "secret.jpg")
    app = create_app(settings=settings, processing_queue=RecordingQueue())

    with TestClient(app) as client:
        directory_link = client.get(
            "/api/thumbnails/video-1/linked-dir/secret.jpg"
        )
        file_link = client.get(
            "/api/thumbnails/video-1/generations/generation-1/linked-file.jpg"
        )

    assert directory_link.status_code == 404
    assert file_link.status_code == 404
    assert directory_link.content != b"not public"
    assert file_link.content != b"not public"


def test_export_route_rejects_symlink_outside_clips_directory(tmp_path) -> None:
    settings = settings_for(tmp_path)
    settings.ensure_directories()
    secret = settings.data_dir / "secret.mp4"
    secret.write_bytes(b"not public")
    (settings.clips_dir / "linked.mp4").symlink_to(secret)
    app = create_app(settings=settings, processing_queue=RecordingQueue())

    with TestClient(app) as client:
        response = client.get("/api/exports/linked.mp4")

    assert response.status_code == 404
    assert response.content != b"not public"


def test_created_export_includes_location_header(tmp_path) -> None:
    class FakeClipService:
        def export(self, name, selections):  # type: ignore[no-untyped-def]
            del name, selections
            return SimpleNamespace(
                name="highlights.mp4",
                duration=3.0,
                created_at="2026-08-09T00:00:00+00:00",
            )

    app = create_app(
        settings=settings_for(tmp_path),
        processing_queue=RecordingQueue(),
        clip_service=FakeClipService(),  # type: ignore[arg-type]
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/exports",
            json={
                "name": "highlights",
                "selections": [{"video_id": "video-1", "start": 1.0, "end": 4.0}],
            },
        )

    assert response.status_code == 201
    assert response.headers["location"] == "/api/exports/highlights.mp4"


def test_created_export_percent_encodes_unicode_location(tmp_path) -> None:
    export_name = "Лучшие_моменты.mp4"

    class FakeClipService:
        def export(self, name, selections):  # type: ignore[no-untyped-def]
            del name, selections
            return SimpleNamespace(
                name=export_name,
                duration=3.0,
                created_at="2026-08-09T00:00:00+00:00",
            )

    app = create_app(
        settings=settings_for(tmp_path),
        processing_queue=RecordingQueue(),
        clip_service=FakeClipService(),  # type: ignore[arg-type]
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/exports",
            json={
                "name": "Лучшие моменты",
                "selections": [{"video_id": "video-1", "start": 1.0, "end": 4.0}],
            },
        )

    expected = f"/api/exports/{quote(export_name, safe='')}"
    assert response.status_code == 201
    assert response.headers["location"] == expected
    assert response.json()["url"] == expected


def test_state_changing_request_rejects_untrusted_browser_origin(tmp_path) -> None:
    app = create_app(settings=settings_for(tmp_path), processing_queue=RecordingQueue())

    with TestClient(app) as client:
        response = client.post(
            "/api/videos",
            headers={"Origin": "https://malicious.example"},
            files={"file": ("match.mp4", b"video", "video/mp4")},
        )

    assert response.status_code == 403


def test_api_rejects_untrusted_host_header(tmp_path) -> None:
    app = create_app(settings=settings_for(tmp_path), processing_queue=RecordingQueue())

    with TestClient(app) as client:
        response = client.get(
            "/api/videos",
            headers={"Host": "attacker.example"},
        )

    assert response.status_code == 400


def test_video_can_be_renamed_without_changing_uploaded_filename(tmp_path) -> None:
    queue = RecordingQueue()
    app = create_app(settings=settings_for(tmp_path), processing_queue=queue)

    with TestClient(app) as client:
        uploaded = client.post(
            "/api/videos",
            files={"file": ("IMG_9956.MP4", b"video", "video/mp4")},
        ).json()
        response = client.patch(
            f"/api/videos/{uploaded['id']}",
            json={"name": "Разбор матча"},
        )

    assert response.status_code == 200
    assert response.json()["display_name"] == "Разбор матча"
    assert response.json()["original_name"] == "IMG_9956.MP4"


def test_search_glossary_can_be_updated_and_read(tmp_path) -> None:
    app = create_app(settings=settings_for(tmp_path), processing_queue=RecordingQueue())

    with TestClient(app) as client:
        updated = client.put(
            "/api/search/glossary",
            json={"entries": {"Мозгов": ["Mozgov", "Москов"]}},
        )
        listing = client.get("/api/search/glossary")

    assert updated.status_code == 200
    assert listing.json()["entries"]["Мозгов"] == ["Mozgov", "Москов"]


def test_evaluation_cases_and_report_are_available_via_api(tmp_path) -> None:
    settings = settings_for(tmp_path)
    settings.ensure_directories()
    repository = ready_repository(settings)
    media = settings.media_dir / "match.mp4"
    media.write_bytes(b"x")
    repository.ensure_asset_identity("video-1", media_root=settings.media_dir)
    specifications = create_indexing_specifications(settings)
    speech_run = repository.create_stage_run(
        video_id="video-1",
        specification=specifications.speech,
    )
    repository.transition_stage_run(speech_run.run_id, StageState.RUNNING)
    repository.commit_segment_generation(speech_run.run_id, segments=[])
    vector_index = MemoryVectorIndex()
    text_run = repository.create_stage_run(
        video_id="video-1",
        specification=specifications.text_vectors,
    )
    repository.transition_stage_run(text_run.run_id, StageState.RUNNING)
    plan = repository.reserve_text_vector_generation(
        text_run.run_id,
        index_specification=vector_index.index_specification,
        semantic_specifications=specifications.semantic_segment_specifications,
    )
    repository.commit_text_vector_generation(
        text_run.run_id,
        receipt=vector_index.build_generation(plan),
    )
    search = SearchService(
        repository,
        vector_index,
        specification_resolver=lambda: create_indexing_specifications(settings),
        thumbnails_dir=settings.thumbnails_dir,
    )
    app = create_app(
        settings=settings,
        repository=repository,
        processing_queue=RecordingQueue(),
        search_service=search,
    )

    with TestClient(app) as client:
        cases = client.put(
            "/api/evaluation/cases",
            json={
                "cases": [
                    {
                        "id": "speech-1",
                        "query": "Мозгов",
                        "video_id": "video-1",
                        "start": 10,
                        "end": 14,
                        "mode": "speech",
                        "label_source": "gold",
                    }
                ]
            },
        )
        report = client.post("/api/evaluation/run", json={"variants": ["speech"]})
        current = client.get("/api/evaluation")
        replacement = client.put(
            "/api/evaluation/cases",
            json={
                "cases": [
                    {
                        "id": "speech-1",
                        "query": "другой запрос",
                        "video_id": "video-1",
                        "start": 10,
                        "end": 14,
                        "mode": "speech",
                        "label_source": "gold",
                    }
                ]
            },
        )
        stale = client.get("/api/evaluation")

    assert cases.status_code == 200
    assert cases.json()["cases"][0]["query"] == "Мозгов"
    assert len(cases.json()["cases_revision"]) == 64
    assert len(cases.json()["runtime_revision"]) == 64
    assert report.status_code == 200
    assert report.json()["variants"][0]["name"] == "speech"
    assert report.json()["variants"][0]["status"] == "complete"
    assert report.json()["variants"][0]["total_case_count"] == 1
    assert report.json()["variants"][0]["successful_case_count"] == 1
    assert report.json()["variants"][0]["error_count"] == 0
    assert report.json()["schema_version"] == 3
    assert report.json()["methodology_version"] == 3
    assert report.json()["cases_revision"] == cases.json()["cases_revision"]
    assert report.json()["runtime_revision"] == cases.json()["runtime_revision"]
    assert report.json()["evaluation_revision"] == cases.json()["evaluation_revision"]
    assert current.json()["cases_revision"] == report.json()["cases_revision"]
    assert current.json()["evaluation_revision"] == report.json()["evaluation_revision"]
    assert current.json()["runtime_revision"] == report.json()["runtime_revision"]
    assert replacement.status_code == 200
    assert len(stale.json()["cases"]) == len(current.json()["cases"])
    assert stale.json()["cases_revision"] != stale.json()["report"]["cases_revision"]
    assert stale.json()["evaluation_revision"] != stale.json()["report"]["evaluation_revision"]


def test_evaluation_cases_must_reference_ready_video_within_duration(tmp_path) -> None:
    settings = settings_for(tmp_path)
    repository = Repository(settings.database_path)
    repository.initialize()
    repository.create_video(
        video_id="video-1",
        original_name="match.mp4",
        stored_name="match.mp4",
        media_path=str(settings.media_dir / "match.mp4"),
        size_bytes=1,
    )
    app = create_app(
        settings=settings,
        repository=repository,
        processing_queue=RecordingQueue(),
    )
    case = {
        "id": "case",
        "query": "query",
        "video_id": "video-1",
        "start": 0,
        "end": 2,
    }

    with TestClient(app) as client:
        queued = client.put("/api/evaluation/cases", json={"cases": [case]})
        repository.update_video("video-1", status="ready", stage="ready", duration=1.0)
        outside = client.put("/api/evaluation/cases", json={"cases": [case]})
        missing = client.put(
            "/api/evaluation/cases",
            json={"cases": [{**case, "video_id": "missing-video"}]},
        )

    assert queued.status_code == 422
    assert queued.json() == {"detail": "Evaluation video is not ready"}
    assert outside.status_code == 422
    assert outside.json() == {"detail": "Evaluation interval exceeds video duration"}
    assert missing.status_code == 422
    assert missing.json() == {"detail": "Evaluation video does not exist"}


def test_run_revalidates_persisted_video_reference(tmp_path) -> None:
    settings = settings_for(tmp_path)
    repository = ready_repository(settings, duration=1.0)
    settings.evaluation_cases_path.parent.mkdir(parents=True, exist_ok=True)
    settings.evaluation_cases_path.write_text(
        '{"version":1,"cases":[{"id":"stale","query":"query","video_id":"video-1","start":0,"end":2}]}',
        encoding="utf-8",
    )
    app = create_app(
        settings=settings,
        repository=repository,
        processing_queue=RecordingQueue(),
    )

    with TestClient(app) as client:
        response = client.post("/api/evaluation/run", json={"variants": ["auto"]})

    assert response.status_code == 422
    assert response.json() == {"detail": "Evaluation interval exceeds video duration"}


def test_evaluation_freshness_changes_with_runtime_config(tmp_path) -> None:
    first_settings = AppSettings(
        data_dir=tmp_path / "first",
        semantic_text_min_score=0.42,
    )
    second_settings = AppSettings(
        data_dir=tmp_path / "second",
        semantic_text_min_score=0.58,
    )
    first_app = create_app(
        settings=first_settings,
        processing_queue=RecordingQueue(),
    )
    second_app = create_app(
        settings=second_settings,
        processing_queue=RecordingQueue(),
    )

    with TestClient(first_app) as first_client, TestClient(second_app) as second_client:
        first = first_client.get("/api/evaluation").json()
        second = second_client.get("/api/evaluation").json()

    assert first["cases_revision"] == second["cases_revision"]
    assert first["runtime_revision"] != second["runtime_revision"]
    assert first["evaluation_revision"] != second["evaluation_revision"]


def test_corrupt_derived_evaluation_report_does_not_break_payload(tmp_path) -> None:
    settings = settings_for(tmp_path)
    settings.evaluation_report_path.parent.mkdir(parents=True, exist_ok=True)
    settings.evaluation_report_path.write_text(
        '{"generated_at":"2026-08-01T00:00:00Z","variants":[{"name":"auto","case_count":1,"status":"failed","cases":[]}]}',
        encoding="utf-8",
    )
    app = create_app(settings=settings, processing_queue=RecordingQueue())

    with TestClient(app) as client:
        response = client.get("/api/evaluation")

    assert response.status_code == 200
    assert response.json()["report"] is None


def test_evaluation_cases_api_rejects_unsafe_video_id(tmp_path) -> None:
    app = create_app(settings=settings_for(tmp_path), processing_queue=RecordingQueue())

    with TestClient(app) as client:
        response = client.put(
            "/api/evaluation/cases",
            json={
                "cases": [{
                    "id": "unsafe",
                    "query": "query",
                    "video_id": "../secret",
                    "start": 0,
                    "end": 1,
                }],
            },
        )

    assert response.status_code == 422


def test_corrupt_evaluation_store_returns_safe_server_error(tmp_path) -> None:
    settings = settings_for(tmp_path)
    settings.ensure_directories()
    settings.evaluation_cases_path.parent.mkdir(parents=True, exist_ok=True)
    settings.evaluation_cases_path.write_text(
        '{"version":1,"cases":[{"id":"bad","query":"query","video_id":"../secret","start":0,"end":1}]}',
        encoding="utf-8",
    )
    app = create_app(settings=settings, processing_queue=RecordingQueue())

    with TestClient(app, raise_server_exceptions=False) as client:
        listing = client.get("/api/evaluation")
        run = client.post("/api/evaluation/run", json={"variants": ["auto"]})

    assert listing.status_code == 500
    assert listing.json() == {"detail": "Evaluation data is invalid"}
    assert run.status_code == 500
    assert run.json() == {"detail": "Evaluation data is invalid"}
    assert str(settings.evaluation_cases_path) not in listing.text
