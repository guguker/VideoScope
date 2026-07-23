from pathlib import Path

from videoscope.media.ffmpeg import SampledFrame
from videoscope.providers.base import ProviderState
from videoscope.providers.internvideo import InternVideoReranker
from videoscope.repository import Repository
from videoscope.search.fusion import EvidenceHit, FusedResult


class FakeExtractor:
    def extract_frames(self, _source, destination, start, end, *, step, max_width=640):  # type: ignore[no-untyped-def]
        del end, step, max_width
        destination.mkdir(parents=True, exist_ok=True)
        output = []
        for index in range(2):
            path = destination / f"frame-{index}.jpg"
            path.write_bytes(f"frame-{index}".encode())
            output.append(SampledFrame(start + index, path))
        return output


class FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self):  # type: ignore[no-untyped-def]
        return {
            "scores": [
                {"id": "video-1:20.000:25.000", "score": 0.96, "reason": "action matches"},
                {"id": "video-1:0.000:5.000", "score": 0.10, "reason": "no action"},
            ]
        }


class FakeClient:
    def __init__(self) -> None:
        self.payload = None

    def post(self, _url, *, json, headers, timeout):  # type: ignore[no-untyped-def]
        assert headers["Authorization"] == "Bearer token"
        assert timeout == 30
        self.payload = json
        return FakeResponse()


def _repository(tmp_path) -> Repository:  # type: ignore[no-untyped-def]
    repository = Repository(tmp_path / "db.sqlite3")
    repository.initialize()
    media = tmp_path / "video.mp4"
    media.write_bytes(b"video")
    repository.create_video(
        video_id="video-1",
        original_name="video.mp4",
        stored_name="video.mp4",
        media_path=str(media),
        size_bytes=5,
    )
    repository.update_video("video-1", duration=30)
    return repository


def test_internvideo_requires_gpu_endpoint(tmp_path) -> None:
    provider = InternVideoReranker(
        endpoint=None,
        api_key=None,
        repository=_repository(tmp_path),
        extractor=FakeExtractor(),
        temp_dir=tmp_path / "frames",
    )

    assert provider.status().state is ProviderState.NEEDS_CONFIGURATION


def test_internvideo_reranks_top_candidates_and_adds_evidence(tmp_path) -> None:
    candidates = [
        FusedResult("video-1", 0, 5, 0.9, ["visual"], [EvidenceHit("video-1", "a", 0, 5, "visual", 0.9, "query")]),
        FusedResult("video-1", 20, 25, 0.7, ["visual"], [EvidenceHit("video-1", "b", 20, 25, "visual", 0.7, "query")]),
    ]
    client = FakeClient()
    provider = InternVideoReranker(
        endpoint="https://gpu.example",
        api_key="token",
        repository=_repository(tmp_path),
        extractor=FakeExtractor(),
        temp_dir=tmp_path / "frames",
        timeout=30,
        client=client,
    )

    reranked = provider.rerank("поднимает руку", candidates)

    assert provider.status().state is ProviderState.READY
    assert reranked[0].start == 20
    assert reranked[0].evidence[0].modality == "internvideo"
    assert reranked[0].evidence[0].text == "action matches"
    assert client.payload["model"] == "OpenGVLab/InternVideo2_5_Chat_8B"  # type: ignore[index]
    assert len(client.payload["candidates"][0]["frames"]) == 2  # type: ignore[index]
