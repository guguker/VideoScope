from pathlib import Path

from fastapi.testclient import TestClient

from videoscope.api import create_app
from videoscope.config import AppSettings


class RecordingQueue:
    def __init__(self) -> None:
        self.video_ids: list[str] = []

    def submit(self, video_id: str) -> None:
        self.video_ids.append(video_id)

    def close(self) -> None:
        return None


def settings_for(tmp_path: Path) -> AppSettings:
    return AppSettings(data_dir=tmp_path / "data", max_upload_bytes=1_024)


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
    assert (tmp_path / "data" / "media" / video["stored_name"]).exists()


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


def test_video_media_route_cannot_escape_media_directory(tmp_path) -> None:
    app = create_app(settings=settings_for(tmp_path), processing_queue=RecordingQueue())

    with TestClient(app) as client:
        response = client.get("/api/videos/../../etc/passwd/media")

    assert response.status_code in {404, 422}


def test_state_changing_request_rejects_untrusted_browser_origin(tmp_path) -> None:
    app = create_app(settings=settings_for(tmp_path), processing_queue=RecordingQueue())

    with TestClient(app) as client:
        response = client.post(
            "/api/videos",
            headers={"Origin": "https://malicious.example"},
            files={"file": ("match.mp4", b"video", "video/mp4")},
        )

    assert response.status_code == 403


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
    app = create_app(settings=settings_for(tmp_path), processing_queue=RecordingQueue())

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

    assert cases.status_code == 200
    assert cases.json()["cases"][0]["query"] == "Мозгов"
    assert report.status_code == 200
    assert report.json()["variants"][0]["name"] == "speech"
