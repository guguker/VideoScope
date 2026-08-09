from __future__ import annotations

from collections.abc import Mapping
import hashlib
import math
from pathlib import Path
import re
from uuid import NAMESPACE_URL, uuid5

import numpy as np

from videoscope.providers.base import ProviderState, ProviderStatus
from videoscope.repository import Repository, SegmentRecord
from videoscope.search.embeddings import HashEmbedding, SemanticEmbedding
from videoscope.search.fusion import EvidenceHit
from videoscope.storage import atomic_write_json


class QdrantVectorIndex:
    id = "qdrant"
    available = True

    def __init__(self, path: Path, *, embedding=None) -> None:  # type: ignore[no-untyped-def]
        self.path = Path(path)
        self.embedding = embedding or SemanticEmbedding()
        self.embedding_identity = str(
            getattr(
                self.embedding,
                "identity",
                f"{type(self.embedding).__module__}.{type(self.embedding).__qualname__}:{self.dimensions}",
            )
        )
        identity_hash = hashlib.sha256(self.embedding_identity.encode("utf-8")).hexdigest()[:12]
        self.collection_name = f"videoscope_segments_v2_{identity_hash}"
        self.marker_path = self.path.parent / f".{self.path.name}-{self.collection_name}.json"
        self._client = None
        self._rebuild_failed = False

    @property
    def dimensions(self) -> int:
        return int(getattr(self.embedding, "dimensions", 384))

    def status(self, *, check_index: bool = True) -> ProviderStatus:
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
            return ProviderStatus(
                self.id,
                "Qdrant",
                ProviderState.UNAVAILABLE,
                "Семантический кодировщик недоступен; подробности записаны в журнале",
            )
        if self._rebuild_failed:
            return ProviderStatus(
                self.id,
                "Qdrant",
                ProviderState.UNAVAILABLE,
                "Семантический индекс недоступен; подробности записаны в журнале",
            )
        if check_index and self.needs_rebuild():
            return ProviderStatus(
                self.id,
                "Qdrant",
                ProviderState.NEEDS_CONFIGURATION,
                "Семантический индекс требует безопасной перестройки",
            )
        backend = getattr(self.embedding, "backend", "local")
        detail = f"Встроенный локальный индекс, кодировщик: {backend}"
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

    def needs_rebuild(self) -> bool:
        if not self.marker_path.is_file():
            return True
        try:
            import json

            payload = json.loads(self.marker_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return True
        if payload != self._marker_payload():
            return True
        if not self.path.is_dir():
            return True
        try:
            if self._client is None:
                from qdrant_client import QdrantClient

                self._client = QdrantClient(path=str(self.path))
            return not self._client.collection_exists(self.collection_name)
        except Exception:
            return True

    def _marker_payload(self) -> dict[str, object]:
        return {
            "collection": self.collection_name,
            "dimensions": self.dimensions,
            "embedding_identity": self.embedding_identity,
        }

    def _write_marker(self) -> None:
        atomic_write_json(
            self.marker_path,
            self._marker_payload(),
            sort_keys=True,
        )

    def invalidate(self) -> None:
        self.marker_path.unlink(missing_ok=True)

    def rebuild_library(
        self,
        videos: list[tuple[str, list[SegmentRecord]]],
    ) -> None:
        from qdrant_client import models

        self.invalidate()
        try:
            client = self._get_client()
            if client.collection_exists(self.collection_name):
                client.delete_collection(self.collection_name)
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.dimensions,
                    distance=models.Distance.COSINE,
                ),
            )
            for video_id, segments in videos:
                self._replace_video(video_id, segments, restore_marker=False)
            self._write_marker()
        except Exception:
            self._rebuild_failed = True
            raise
        self._rebuild_failed = False

    def rebuild_repository(self, repository: Repository) -> None:
        """Rebuild every ready video so a filtered backfill cannot erase other videos."""
        self.rebuild_library([
            (video.id, repository.list_segments(video.id))
            for video in repository.list_videos()
            if video.status == "ready"
        ])

    def replace_video(self, video_id: str, segments: list[SegmentRecord]) -> None:
        self._replace_video(video_id, segments, restore_marker=True)

    def _replace_video(
        self,
        video_id: str,
        segments: list[SegmentRecord],
        *,
        restore_marker: bool,
    ) -> None:
        from qdrant_client import models

        ensure_ready = getattr(self.embedding, "ensure_ready", None)
        if ensure_ready is not None and not ensure_ready():
            raise RuntimeError("semantic embedding is not ready")

        searchable = [
            segment
            for segment in segments
            if segment.text.strip() and segment.modality in {"speech", "ocr", "objects"}
            and (
                segment.modality != "speech"
                or len(re.findall(r"[\w]+", segment.text, flags=re.UNICODE)) >= 2
            )
        ]
        if any(segment.video_id != video_id for segment in searchable):
            raise ValueError("segment video_id does not match replacement scope")
        vectors = self.embedding.embed([segment.text for segment in searchable])
        if len(vectors) != len(searchable):
            raise ValueError("embedding model returned an unexpected vector count")
        validated_vectors: list[np.ndarray] = []
        for vector in vectors:
            try:
                resolved = np.asarray(vector, dtype=np.float32)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("embedding model returned an invalid vector") from error
            if (
                resolved.ndim != 1
                or resolved.shape[0] != self.dimensions
                or not bool(np.isfinite(resolved).all())
            ):
                raise ValueError("embedding model returned an invalid vector")
            validated_vectors.append(resolved)
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
            for segment, vector in zip(searchable, validated_vectors, strict=True)
        ]

        video_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="video_id",
                    match=models.MatchValue(value=video_id),
                )
            ]
        )
        self.invalidate()
        client = self._get_client()
        client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(filter=video_filter),
            wait=True,
        )
        if points:
            client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
        if restore_marker:
            self._write_marker()

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
            payload = getattr(point, "payload", None)
            if not isinstance(payload, Mapping):
                continue
            required_text = {
                key: payload.get(key)
                for key in ("video_id", "segment_id", "modality", "text")
            }
            if not all(
                isinstance(value, str) and bool(value.strip())
                for value in required_text.values()
            ):
                continue
            if required_text["modality"] not in {"speech", "ocr", "objects"}:
                continue
            raw_metadata = payload.get("metadata")
            if raw_metadata is None:
                metadata: dict[str, object] = {}
            elif isinstance(raw_metadata, Mapping) and all(
                isinstance(key, str) for key in raw_metadata
            ):
                metadata = dict(raw_metadata)
            else:
                continue
            thumbnail_path = payload.get("thumbnail_path")
            if thumbnail_path is not None and not isinstance(thumbnail_path, str):
                continue
            try:
                raw_confidence = payload.get("confidence")
                if isinstance(raw_confidence, bool):
                    continue
                confidence = 1.0 if raw_confidence is None else float(raw_confidence)
                similarity = float(point.score)
                start = float(payload["start"])
                end = float(payload["end"])
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
            if not all(
                math.isfinite(value)
                for value in (confidence, similarity, start, end)
            ) or start < 0 or end <= start:
                continue
            confidence = max(0.0, min(1.0, confidence))
            similarity = max(0.0, min(1.0, similarity))
            hits.append(
                EvidenceHit(
                    video_id=required_text["video_id"],
                    segment_id=required_text["segment_id"],
                    start=start,
                    end=end,
                    modality=required_text["modality"],
                    score=similarity * (0.8 + 0.2 * confidence),
                    text=required_text["text"],
                    metadata={
                        **metadata,
                        "thumbnail_path": thumbnail_path,
                        "source": "qdrant",
                        "segment_confidence": confidence,
                    },
                )
            )
        return hits


class MemoryVectorIndex:
    """Небольшой детерминированный резервный индекс при недоступности qdrant-client."""

    id = "memory-index"
    available = True

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
    """Безопасный резервный вариант: лексический поиск без фиктивных семантических оценок."""

    available = False

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
