from videoscope.repository import SegmentRecord
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

