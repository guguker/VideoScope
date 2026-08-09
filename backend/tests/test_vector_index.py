from types import SimpleNamespace
import shutil

import numpy as np
import pytest

from videoscope.repository import Repository, SegmentRecord
from videoscope.search.embeddings import HashEmbedding
from videoscope.search.vector_index import QdrantVectorIndex


def segment(segment_id: str, text: str, *, video_id: str = "video-1") -> SegmentRecord:
    return SegmentRecord(
        id=segment_id,
        video_id=video_id,
        start=1.0,
        end=4.0,
        modality="speech",
        text=text,
        confidence=0.9,
        metadata={},
        thumbnail_path=None,
    )


def test_qdrant_index_returns_semantically_closest_local_segment(tmp_path) -> None:
    index = QdrantVectorIndex(tmp_path / "qdrant", embedding=HashEmbedding(384))
    index.replace_video(
        "video-1",
        [
            segment("shot", "player makes a three point shot"),
            segment("scoreboard", "scoreboard during timeout"),
        ],
    )

    hits = index.search("three point shot", limit=2)

    assert hits[0].segment_id == "shot"
    assert hits[0].video_id == "video-1"


def test_replace_video_removes_stale_points_without_touching_other_videos(tmp_path) -> None:
    index = QdrantVectorIndex(tmp_path / "qdrant", embedding=HashEmbedding(384))
    index.replace_video("video-1", [segment("stale", "old phrase")])
    index.replace_video("video-2", [segment("other", "other phrase", video_id="video-2")])

    index.replace_video("video-1", [segment("fresh", "new phrase")])

    hits = index.search("phrase", limit=10)
    assert {hit.segment_id for hit in hits} == {"fresh", "other"}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("score", float("nan")),
        ("score", float("inf")),
        ("confidence", float("nan")),
        ("start", float("nan")),
        ("end", float("inf")),
    ],
)
def test_qdrant_drops_non_finite_provider_values(tmp_path, field: str, value: float) -> None:
    payload = {
        "video_id": "video-1",
        "segment_id": "bad",
        "start": 1.0,
        "end": 4.0,
        "modality": "speech",
        "text": "bad score",
        "confidence": 0.9,
    }
    score = 0.8
    if field == "score":
        score = value
    else:
        payload[field] = value

    class FakeEmbedding:
        dimensions = 2

        @staticmethod
        def embed_query(_query: str):  # type: ignore[no-untyped-def]
            return np.array([1.0, 0.0])

    class FakeClient:
        @staticmethod
        def query_points(**_kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(points=[SimpleNamespace(payload=payload, score=score)])

    index = QdrantVectorIndex(tmp_path / "qdrant", embedding=FakeEmbedding())
    index._client = FakeClient()

    assert index.search("query") == []


def test_qdrant_collection_is_scoped_to_embedding_identity(tmp_path) -> None:
    first = QdrantVectorIndex(tmp_path / "qdrant", embedding=HashEmbedding(384))
    second = QdrantVectorIndex(tmp_path / "qdrant", embedding=HashEmbedding(768))

    assert first.collection_name != second.collection_name


def test_qdrant_rebuild_marks_the_model_identity_current(tmp_path) -> None:
    path = tmp_path / "qdrant" / "text"
    index = QdrantVectorIndex(path, embedding=HashEmbedding(384))

    assert index.needs_rebuild() is True
    index.rebuild_library([("video-1", [segment("shot", "three point shot")])])

    assert index.needs_rebuild() is False
    assert index.search("three point", limit=1)[0].segment_id == "shot"
    index._client.close()
    index._client = None
    same_identity = QdrantVectorIndex(path, embedding=HashEmbedding(384))
    different_identity = QdrantVectorIndex(path, embedding=HashEmbedding(768))
    assert same_identity.needs_rebuild() is False
    assert different_identity.needs_rebuild() is True

    same_identity._client.close()
    same_identity._client = None
    shutil.rmtree(path)
    deleted_storage = QdrantVectorIndex(path, embedding=HashEmbedding(384))
    assert deleted_storage.needs_rebuild() is True


def test_qdrant_status_reports_rebuild_failure(tmp_path, monkeypatch) -> None:
    index = QdrantVectorIndex(tmp_path / "qdrant", embedding=HashEmbedding(384))
    monkeypatch.setattr(
        index,
        "_get_client",
        lambda: (_ for _ in ()).throw(RuntimeError("storage corrupt")),
    )

    with pytest.raises(RuntimeError, match="storage corrupt"):
        index.rebuild_library([])

    status = index.status()
    assert status.state.value == "unavailable"
    assert "storage corrupt" not in status.detail


def test_repository_rebuild_always_keeps_every_ready_video(tmp_path) -> None:
    repository = Repository(tmp_path / "db.sqlite3")
    repository.initialize()
    for video_id in ("video-1", "video-2"):
        media = tmp_path / f"{video_id}.mp4"
        media.write_bytes(b"video")
        repository.create_video(
            video_id=video_id,
            original_name=media.name,
            stored_name=media.name,
            media_path=str(media),
            size_bytes=5,
        )
        repository.update_video(video_id, status="ready", duration=10.0)
        item = segment(f"segment-{video_id}", f"phrase for {video_id}", video_id=video_id)
        repository.add_segment(
            segment_id=item.id,
            video_id=item.video_id,
            start=item.start,
            end=item.end,
            modality=item.modality,
            text=item.text,
            confidence=item.confidence,
            metadata=item.metadata,
        )

    index = QdrantVectorIndex(tmp_path / "qdrant", embedding=HashEmbedding(384))
    index.rebuild_repository(repository)

    hits = index.search("phrase", limit=10)
    assert {hit.video_id for hit in hits} == {"video-1", "video-2"}


class ControlledEmbedding:
    dimensions = 2
    identity = "controlled:2"

    def __init__(self, vectors: list[np.ndarray] | Exception | None = None) -> None:
        self.vectors = vectors

    @staticmethod
    def ensure_ready() -> bool:
        return True

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        if isinstance(self.vectors, Exception):
            raise self.vectors
        if self.vectors is not None:
            return self.vectors
        return [np.array([1.0, 0.0], dtype=np.float32) for _ in texts]

    @staticmethod
    def embed_query(_query: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)


class RecordingMutationClient:
    def __init__(self, *, fail_at: str | None = None, index=None) -> None:  # type: ignore[no-untyped-def]
        self.fail_at = fail_at
        self.index = index
        self.mutations: list[str] = []
        self.marker_states: list[bool] = []

    def _record(self, operation: str) -> None:
        self.mutations.append(operation)
        if self.index is not None:
            self.marker_states.append(self.index.marker_path.exists())
        if self.fail_at == operation:
            raise RuntimeError(f"{operation} failed")

    @staticmethod
    def collection_exists(_collection_name: str) -> bool:
        return True

    def delete_collection(self, _collection_name: str) -> None:
        self._record("delete_collection")

    def create_collection(self, **_kwargs) -> None:  # type: ignore[no-untyped-def]
        self._record("create_collection")

    def delete(self, **_kwargs) -> None:  # type: ignore[no-untyped-def]
        self._record("delete")

    def upsert(self, **_kwargs) -> None:  # type: ignore[no-untyped-def]
        self._record("upsert")


def _seed_marker(index: QdrantVectorIndex) -> None:
    index._write_marker()


@pytest.mark.parametrize(
    "vectors",
    [
        RuntimeError("embedding failed"),
        [],
        [np.array([float("nan"), 0.0], dtype=np.float32)],
        [np.array([1.0, 0.0, 0.0], dtype=np.float32)],
    ],
)
def test_replace_video_validates_vectors_before_mutating_qdrant(tmp_path, vectors) -> None:  # type: ignore[no-untyped-def]
    index = QdrantVectorIndex(
        tmp_path / "qdrant",
        embedding=ControlledEmbedding(vectors),
    )
    client = RecordingMutationClient()
    index._client = client
    _seed_marker(index)

    with pytest.raises((RuntimeError, ValueError)):
        index.replace_video("video-1", [segment("bad", "bad vector")])

    assert client.mutations == []
    assert index.marker_path.exists()


@pytest.mark.parametrize("fail_at", ["delete", "upsert"])
def test_replace_video_failure_leaves_index_marker_stale(tmp_path, fail_at: str) -> None:
    index = QdrantVectorIndex(
        tmp_path / "qdrant",
        embedding=ControlledEmbedding(),
    )
    client = RecordingMutationClient(fail_at=fail_at)
    index._client = client
    _seed_marker(index)

    with pytest.raises(RuntimeError, match=f"{fail_at} failed"):
        index.replace_video("video-1", [segment("fresh", "valid vector")])

    assert not index.marker_path.exists()


def test_successful_replace_video_restores_index_marker(tmp_path) -> None:
    index = QdrantVectorIndex(
        tmp_path / "qdrant",
        embedding=ControlledEmbedding(),
    )
    client = RecordingMutationClient()
    index._client = client

    index.replace_video("video-1", [segment("fresh", "valid vector")])

    assert client.mutations == ["delete", "upsert"]
    assert index.marker_path.exists()


def test_rebuild_writes_marker_only_after_every_video_is_updated(tmp_path) -> None:
    index = QdrantVectorIndex(
        tmp_path / "qdrant",
        embedding=ControlledEmbedding(),
    )
    client = RecordingMutationClient(index=index)
    index._client = client
    _seed_marker(index)

    index.rebuild_library(
        [
            ("video-1", [segment("first", "first phrase")]),
            ("video-2", [segment("second", "second phrase", video_id="video-2")]),
        ]
    )

    assert client.marker_states
    assert all(state is False for state in client.marker_states)
    assert index.marker_path.exists()


@pytest.mark.parametrize(
    "bad_payload",
    [
        "not-a-mapping",
        {"start": 1.0, "end": 2.0, "modality": "speech", "text": "missing ids"},
        {
            "video_id": "video-1",
            "segment_id": "bad",
            "start": 1.0,
            "end": 2.0,
            "modality": "speech",
            "text": "bad metadata",
            "metadata": "not-a-mapping",
        },
        {
            "video_id": "",
            "segment_id": "bad",
            "start": 1.0,
            "end": 2.0,
            "modality": "speech",
            "text": "empty id",
        },
    ],
)
def test_qdrant_search_drops_malformed_payload_without_losing_valid_hits(
    tmp_path,
    bad_payload,
) -> None:  # type: ignore[no-untyped-def]
    valid_payload = {
        "video_id": "video-1",
        "segment_id": "valid",
        "start": 1.0,
        "end": 2.0,
        "modality": "speech",
        "text": "valid phrase",
        "confidence": 0.9,
        "metadata": {},
    }

    class FakeClient:
        @staticmethod
        def query_points(**_kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(
                points=[
                    SimpleNamespace(payload=bad_payload, score=0.99),
                    SimpleNamespace(payload=valid_payload, score=0.5),
                ]
            )

    index = QdrantVectorIndex(
        tmp_path / "qdrant",
        embedding=ControlledEmbedding(),
    )
    index._client = FakeClient()

    hits = index.search("query")

    assert [hit.segment_id for hit in hits] == ["valid"]
