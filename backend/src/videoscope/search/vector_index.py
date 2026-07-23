from __future__ import annotations

from pathlib import Path
import re
from uuid import NAMESPACE_URL, uuid5

from videoscope.providers.base import ProviderState, ProviderStatus
from videoscope.repository import SegmentRecord
from videoscope.search.embeddings import HashEmbedding, SemanticEmbedding
from videoscope.search.fusion import EvidenceHit


class QdrantVectorIndex:
    id = "qdrant"
    collection_name = "videoscope_segments_v1"

    def __init__(self, path: Path, *, embedding=None) -> None:  # type: ignore[no-untyped-def]
        self.path = Path(path)
        self.embedding = embedding or SemanticEmbedding()
        self._client = None

    @property
    def dimensions(self) -> int:
        return int(getattr(self.embedding, "dimensions", 384))

    def status(self) -> ProviderStatus:
        try:
            import qdrant_client  # noqa: F401
        except ImportError:
            return ProviderStatus(
                self.id,
                "Qdrant",
                ProviderState.UNAVAILABLE,
                "qdrant-client is not installed",
            )
        ensure_ready = getattr(self.embedding, "ensure_ready", None)
        if ensure_ready is not None and not ensure_ready():
            detail = str(getattr(self.embedding, "last_error", None) or "text encoder failed to load")
            return ProviderStatus(
                self.id,
                "Qdrant",
                ProviderState.UNAVAILABLE,
                f"Semantic encoder is unavailable: {detail[:160]}",
            )
        backend = getattr(self.embedding, "backend", "local")
        last_error = getattr(self.embedding, "last_error", None)
        detail = f"Embedded local index, encoder: {backend}"
        if last_error:
            detail += f" ({str(last_error)[:120]})"
        return ProviderStatus(
            self.id,
            "Qdrant",
            ProviderState.READY,
            detail,
        )

    def _get_client(self):  # type: ignore[no-untyped-def]
        if self._client is not None:
            return self._client
        from qdrant_client import QdrantClient, models

        self.path.mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=str(self.path))
        if not client.collection_exists(self.collection_name):
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.dimensions,
                    distance=models.Distance.COSINE,
                ),
            )
        self._client = client
        return client

    def replace_video(self, video_id: str, segments: list[SegmentRecord]) -> None:
        from qdrant_client import models

        client = self._get_client()
        video_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="video_id",
                    match=models.MatchValue(value=video_id),
                )
            ]
        )
        client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(filter=video_filter),
            wait=True,
        )
        searchable = [
            segment
            for segment in segments
            if segment.text.strip() and segment.modality in {"speech", "ocr", "objects"}
            and (
                segment.modality != "speech"
                or len(re.findall(r"[\w]+", segment.text, flags=re.UNICODE)) >= 2
            )
        ]
        if not searchable:
            return
        vectors = self.embedding.embed([segment.text for segment in searchable])
        points = [
            models.PointStruct(
                id=str(uuid5(NAMESPACE_URL, segment.id)),
                vector=vector.tolist(),
                payload={
                    "segment_id": segment.id,
                    "video_id": segment.video_id,
                    "start": segment.start,
                    "end": segment.end,
                    "modality": segment.modality,
                    "text": segment.text,
                    "confidence": segment.confidence,
                    "metadata": segment.metadata,
                    "thumbnail_path": segment.thumbnail_path,
                },
            )
            for segment, vector in zip(searchable, vectors, strict=True)
        ]
        client.upsert(collection_name=self.collection_name, points=points, wait=True)

    def search(
        self,
        query: str,
        *,
        video_ids: list[str] | None = None,
        modalities: set[str] | None = None,
        limit: int = 50,
    ) -> list[EvidenceHit]:
        from qdrant_client import models

        if not query.strip() or limit <= 0:
            return []
        query_filter = None
        must = []
        if video_ids:
            must.append(
                models.FieldCondition(
                    key="video_id",
                    match=models.MatchAny(any=video_ids),
                )
            )
        if modalities:
            must.append(
                models.FieldCondition(
                    key="modality",
                    match=models.MatchAny(any=sorted(modalities)),
                )
            )
        if must:
            query_filter = models.Filter(must=must)
        embed_query = getattr(self.embedding, "embed_query", None)
        vector = embed_query(query) if embed_query else self.embedding.embed([query])[0]
        response = self._get_client().query_points(
            collection_name=self.collection_name,
            query=vector.tolist(),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        hits: list[EvidenceHit] = []
        for point in response.points:
            payload = point.payload or {}
            confidence = float(payload.get("confidence") or 1.0)
            similarity = max(0.0, min(1.0, float(point.score)))
            hits.append(
                EvidenceHit(
                    video_id=str(payload["video_id"]),
                    segment_id=str(payload["segment_id"]),
                    start=float(payload["start"]),
                    end=float(payload["end"]),
                    modality=str(payload["modality"]),
                    score=similarity * (0.8 + 0.2 * confidence),
                    text=str(payload["text"]),
                    metadata={
                        **dict(payload.get("metadata") or {}),
                        "thumbnail_path": payload.get("thumbnail_path"),
                        "source": "qdrant",
                        "segment_confidence": confidence,
                    },
                )
            )
        return hits


class MemoryVectorIndex:
    """Small deterministic fallback used only when qdrant-client is unavailable."""

    id = "memory-index"

    def __init__(self, embedding: HashEmbedding | None = None) -> None:
        self.embedding = embedding or HashEmbedding()
        self._segments: dict[str, SegmentRecord] = {}

    def replace_video(self, video_id: str, segments: list[SegmentRecord]) -> None:
        self._segments = {
            key: value for key, value in self._segments.items() if value.video_id != video_id
        }
        self._segments.update({segment.id: segment for segment in segments if segment.text.strip()})

    def search(
        self,
        query: str,
        *,
        video_ids: list[str] | None = None,
        modalities: set[str] | None = None,
        limit: int = 50,
    ) -> list[EvidenceHit]:
        import numpy as np

        candidates = [
            segment
            for segment in self._segments.values()
            if not video_ids or segment.video_id in video_ids
            if not modalities or segment.modality in modalities
        ]
        if not candidates or not query.strip():
            return []
        query_vector = self.embedding.embed([query])[0]
        vectors = self.embedding.embed([segment.text for segment in candidates])
        ranked = sorted(
            zip(candidates, vectors, strict=True),
            key=lambda pair: float(np.dot(query_vector, pair[1])),
            reverse=True,
        )[:limit]
        return [
            EvidenceHit(
                video_id=segment.video_id,
                segment_id=segment.id,
                start=segment.start,
                end=segment.end,
                modality=segment.modality,
                score=max(0.0, float(np.dot(query_vector, vector))),
                text=segment.text,
                metadata={
                    **segment.metadata,
                    "thumbnail_path": segment.thumbnail_path,
                    "source": "memory",
                    "segment_confidence": segment.confidence,
                },
            )
            for segment, vector in ranked
        ]


class EmptyVectorIndex:
    """Fail-closed fallback: lexical search stays available without fake semantic scores."""

    def replace_video(self, _video_id: str, _segments: list[SegmentRecord]) -> None:
        return None

    def search(
        self,
        _query: str,
        *,
        video_ids: list[str] | None = None,
        modalities: set[str] | None = None,
        limit: int = 50,
    ) -> list[EvidenceHit]:
        return []
