from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import re
import stat
from threading import RLock
from uuid import NAMESPACE_URL, uuid5

import numpy as np

from videoscope.providers.base import ProviderState, ProviderStatus
from videoscope.artifacts import (
    TextVectorBuildPlan,
    TextVectorBuildReceipt,
    TextVectorIndexSpecification,
    TextVectorPointSource,
    TextVectorSearchBinding,
    TextVectorSearchHit,
    validate_artifact_identifier,
)
from videoscope.repository import Repository, SegmentRecord
from videoscope.search.embeddings import HashEmbedding, SemanticEmbedding
from videoscope.search.fusion import EvidenceHit
from videoscope.storage import atomic_write_json


_TEXT_VECTOR_COLLECTION_PREFIX = "videoscope_text_v1_"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_QDRANT_OPEN_CLEANUP_ATTEMPTS = 3
_MAX_QDRANT_BENCHMARK_BINDINGS = 128
_QDRANT_COSINE_SCORE_TOLERANCE = 1e-6


class QdrantStorageContractError(ValueError):
    """A persisted GC scope is invalid and must not be retried."""


class QdrantStorageUnavailableError(RuntimeError):
    """Embedded Qdrant storage failed transiently and may be retried."""


class QdrantSnapshotCleanupError(QdrantStorageUnavailableError):
    """A failed private-snapshot open still owns a retryable client handle."""

    def __init__(
        self,
        primary_error: Exception,
        instance: QdrantVectorIndex,
    ) -> None:
        self.primary_error = primary_error
        self._instance = instance
        super().__init__(
            f"{primary_error}; Qdrant benchmark snapshot cleanup remains pending"
        )

    @property
    def cleanup_pending(self) -> bool:
        return self._instance._client is not None

    def retry_cleanup(self) -> None:
        try:
            _close_failed_qdrant_snapshot_open(self._instance)
        except Exception as error:
            raise QdrantStorageUnavailableError(
                "Qdrant benchmark snapshot cleanup remains pending"
            ) from error


def _close_failed_qdrant_snapshot_open(instance: QdrantVectorIndex) -> None:
    last_error: Exception | None = None
    for _attempt in range(_QDRANT_OPEN_CLEANUP_ATTEMPTS):
        try:
            instance.close()
        except Exception as error:
            last_error = error
            continue
        return
    assert last_error is not None
    raise QdrantStorageUnavailableError(
        "Qdrant benchmark snapshot client cleanup remains pending"
    ) from last_error


def _validate_generation_storage_scope(
    *,
    index_specification_hash: str,
    collection_name: str,
) -> None:
    if (
        type(index_specification_hash) is not str
        or _SHA256_PATTERN.fullmatch(index_specification_hash) is None
    ):
        raise QdrantStorageContractError(
            "text vector GC index specification must be lowercase SHA-256"
        )
    expected_collection_name = (
        f"{_TEXT_VECTOR_COLLECTION_PREFIX}{index_specification_hash[:32]}"
    )
    if type(collection_name) is not str or collection_name != expected_collection_name:
        raise QdrantStorageContractError(
            "text vector GC collection does not match index specification"
        )


class QdrantVectorIndex:
    id = "qdrant"
    available = True
    supports_generation_provenance = True

    def __init__(
        self,
        path: Path,
        *,
        embedding=None,  # type: ignore[no-untyped-def]
        _existing_snapshot_digest: str | None = None,
        _existing_snapshot: object | None = None,
    ) -> None:
        self.path = Path(path)
        self.embedding = embedding or SemanticEmbedding()
        self.embedding_identity = str(
            getattr(
                self.embedding,
                "identity",
                f"{type(self.embedding).__module__}.{type(self.embedding).__qualname__}:{self.dimensions}",
            )
        )
        self.index_specification = TextVectorIndexSpecification(
            embedding_identity=self.embedding_identity,
            dimensions=self.dimensions,
        )
        self.collection_name = self.index_specification.collection_name
        if (
            _existing_snapshot_digest is not None
            and (
                type(_existing_snapshot_digest) is not str
                or _SHA256_PATTERN.fullmatch(_existing_snapshot_digest) is None
            )
        ):
            raise ValueError("Qdrant snapshot digest must be lowercase SHA-256")
        self._existing_snapshot_digest = _existing_snapshot_digest
        self._existing_snapshot = _existing_snapshot
        self.marker_path = self.path.parent / f".{self.path.name}-{self.collection_name}.json"
        self._client = None
        self._client_lock = RLock()
        self._current_collection_ready = False
        self._rebuild_failed = False
        self._validated_generation_cache: OrderedDict[
            tuple[str, str, str, str, int], None
        ] = OrderedDict()
        self._validated_benchmark_binding_cache: OrderedDict[
            int,
            tuple[TextVectorSearchBinding, Mapping[str, TextVectorPointSource]],
        ] = OrderedDict()

    @classmethod
    def open_existing_snapshot(
        cls,
        snapshot: object,
        *,
        embedding: object,
    ) -> QdrantVectorIndex:
        """Open one verified private copy without creating storage or a collection."""
        from videoscope.benchmark.snapshots import (
            QdrantStorageSnapshot,
            verify_qdrant_storage_snapshot,
        )

        if not isinstance(snapshot, QdrantStorageSnapshot):
            raise ValueError("Qdrant storage snapshot must be validated")
        verify_qdrant_storage_snapshot(snapshot)
        instance = cls(
            snapshot.path,
            embedding=embedding,
            _existing_snapshot_digest=snapshot.snapshot_sha256,
            _existing_snapshot=snapshot,
        )
        instance._validate_benchmark_embedding()
        try:
            instance._get_client()
        except Exception as open_error:
            try:
                _close_failed_qdrant_snapshot_open(instance)
            except Exception as cleanup_error:
                raise QdrantSnapshotCleanupError(
                    open_error,
                    instance,
                ) from ExceptionGroup(
                    "Qdrant snapshot open and cleanup both failed",
                    [open_error, cleanup_error],
                )
            raise
        return instance

    def _validate_benchmark_embedding(self) -> dict[str, object]:
        if getattr(self.embedding, "strict_no_fallback", False) is not True:
            raise RuntimeError("Qdrant benchmark snapshot requires a strict embedding")
        identity = getattr(self.embedding, "benchmark_identity", None)
        if callable(identity):
            identity = identity()
        if not isinstance(identity, Mapping):
            raise RuntimeError("Qdrant benchmark embedding identity is unavailable")
        required = {
            "embedding_identity",
            "model_name",
            "model_repository",
            "model_revision",
            "runtime_version",
            "algorithm_version",
            "dimensions",
            "model_content_sha256",
        }
        if set(identity) != required:
            raise RuntimeError("Qdrant benchmark embedding identity is invalid")
        resolved = dict(identity)
        if (
            resolved["embedding_identity"] != self.index_specification.embedding_identity
            or resolved["dimensions"] != self.dimensions
            or any(
                type(resolved[name]) is not str or not str(resolved[name]).strip()
                for name in required - {"dimensions", "model_content_sha256"}
            )
            or type(resolved["model_content_sha256"]) is not str
            or _SHA256_PATTERN.fullmatch(str(resolved["model_content_sha256"])) is None
        ):
            raise RuntimeError("Qdrant benchmark embedding identity is invalid")
        return resolved

    @property
    def benchmark_attestation(self) -> dict[str, object]:
        if self._existing_snapshot_digest is None or self._existing_snapshot is None:
            raise RuntimeError("Qdrant benchmark attestation requires an existing snapshot")
        embedding = self._validate_benchmark_embedding()
        return {
            "schema_version": 1,
            "provider": self.id,
            "strict_no_fallback": True,
            "embedding": embedding,
            "index": {
                "index_specification_hash": self.index_specification.specification_hash,
                "collection_name": self.collection_name,
                "snapshot_sha256": self._existing_snapshot_digest,
            },
        }

    def verify_benchmark_snapshot_current(self) -> bool:
        """Re-hash private model/storage state outside timed search."""
        if self._existing_snapshot is None:
            return False
        from videoscope.benchmark.snapshots import verify_qdrant_storage_snapshot

        try:
            verify_model = getattr(
                self.embedding,
                "verify_benchmark_model_current",
                None,
            )
            if not callable(verify_model) or verify_model() is not True:
                self._validated_benchmark_binding_cache.clear()
                return False
            verify_qdrant_storage_snapshot(self._existing_snapshot)
        except Exception:
            self._validated_benchmark_binding_cache.clear()
            return False
        return True

    def _assert_query_only_snapshot_is_not_mutated(self) -> None:
        if self._existing_snapshot_digest is not None:
            raise RuntimeError("Qdrant benchmark snapshot is query-only")

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
        # Legacy marker state is intentionally not readiness for immutable
        # generations. A fresh collection is writable and becomes searchable
        # only after SQLite activates a verified per-video generation.
        backend = getattr(self.embedding, "backend", "local")
        detail = f"Встроенный локальный индекс, кодировщик: {backend}"
        return ProviderStatus(
            self.id,
            "Qdrant",
            ProviderState.READY,
            detail,
        )

    def _get_storage_client(self, *, create_path: bool):  # type: ignore[no-untyped-def]
        with self._client_lock:
            if self._client is not None:
                return self._client
            if self._existing_snapshot_digest is not None:
                create_path = False
                self._reject_symlinked_storage_path()
            if not self.path.exists():
                if not create_path:
                    return None
                self.path.mkdir(parents=True, exist_ok=True)
            elif not self.path.is_dir():
                raise RuntimeError("Qdrant storage path is not a directory")
            from qdrant_client import QdrantClient

            try:
                client = QdrantClient(path=str(self.path))
            except ValueError as error:
                raise QdrantStorageUnavailableError(
                    "Qdrant storage client could not be opened"
                ) from error
            self._client = client
            return client

    def _reject_symlinked_storage_path(self) -> None:
        """Fail closed before destructive access through any existing symlink."""
        candidate = self.path.absolute()
        current = Path(candidate.anchor)
        for component in candidate.parts[1:]:
            current /= component
            try:
                metadata = current.lstat()
            except FileNotFoundError:
                break
            except OSError as error:
                raise RuntimeError("Qdrant storage path could not be validated") from error
            if stat.S_ISLNK(metadata.st_mode):
                raise RuntimeError("Qdrant storage path must not contain a symlink")

    def _get_client(self):  # type: ignore[no-untyped-def]
        with self._client_lock:
            if self._client is not None and self._current_collection_ready:
                return self._client
            if self._existing_snapshot_digest is not None:
                client = self._get_storage_client(create_path=False)
                if client is None:
                    raise QdrantStorageUnavailableError(
                        "Qdrant benchmark snapshot storage is missing"
                    )
                collection_exists = getattr(client, "collection_exists", None)
                if collection_exists is None:
                    raise QdrantStorageUnavailableError(
                        "Qdrant benchmark snapshot cannot verify its collection"
                    )
                try:
                    exists = bool(collection_exists(self.collection_name))
                except (OSError, RuntimeError, ValueError) as error:
                    raise QdrantStorageUnavailableError(
                        "Qdrant benchmark snapshot collection is corrupt"
                    ) from error
                if not exists:
                    raise QdrantStorageUnavailableError(
                        "Qdrant benchmark snapshot collection is missing"
                    )
                self._current_collection_ready = True
                return client
            from qdrant_client import models

            client = self._get_storage_client(create_path=True)
            if client is None:  # pragma: no cover - create_path=True guarantees a client
                raise RuntimeError("Qdrant storage client was not created")
            collection_exists = getattr(client, "collection_exists", None)
            if collection_exists is None:
                # A narrow compatibility seam for injected test clients. Real storage
                # clients always expose collection_exists.
                self._current_collection_ready = True
                return client
            try:
                if not collection_exists(self.collection_name):
                    client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=models.VectorParams(
                            size=self.dimensions,
                            distance=models.Distance.COSINE,
                        ),
                    )
            except ValueError as error:
                raise QdrantStorageUnavailableError(
                    "Qdrant collection could not be prepared"
                ) from error
            self._current_collection_ready = True
            return client

    def close(self) -> None:
        with self._client_lock:
            client = self._client
            if client is None:
                return
            close = getattr(client, "close", None)
            if close is not None:
                try:
                    close()
                except ValueError as error:
                    raise QdrantStorageUnavailableError(
                        "Qdrant storage client could not be closed"
                    ) from error
            self._client = None
            self._current_collection_ready = False
            self._validated_benchmark_binding_cache.clear()

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
            client = self._get_storage_client(create_path=False)
            return client is None or not client.collection_exists(self.collection_name)
        except Exception:
            return True

    def _marker_payload(self) -> dict[str, object]:
        return {
            "collection": self.collection_name,
            "dimensions": self.dimensions,
            "embedding_identity": self.embedding_identity,
        }

    def _write_marker(self) -> None:
        self._assert_query_only_snapshot_is_not_mutated()
        atomic_write_json(
            self.marker_path,
            self._marker_payload(),
            sort_keys=True,
        )

    def invalidate(self) -> None:
        self._assert_query_only_snapshot_is_not_mutated()
        self.marker_path.unlink(missing_ok=True)

    def rebuild_library(
        self,
        videos: list[tuple[str, list[SegmentRecord]]],
    ) -> None:
        from qdrant_client import models

        self._assert_query_only_snapshot_is_not_mutated()
        self.invalidate()
        try:
            client = self._get_client()
            legacy_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="record_type",
                        match=models.MatchValue(value="legacy"),
                    )
                ]
            )
            client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=legacy_filter),
                wait=True,
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

    @staticmethod
    def point_id(generation_id: str, segment_id: str) -> str:
        validate_artifact_identifier(generation_id, field_name="text vector generation id")
        validate_artifact_identifier(segment_id, field_name="text vector segment id")
        return str(uuid5(NAMESPACE_URL, f"videoscope:text:{generation_id}:{segment_id}"))

    @staticmethod
    def manifest_point_id(generation_id: str) -> str:
        validate_artifact_identifier(generation_id, field_name="text vector generation id")
        return str(uuid5(NAMESPACE_URL, f"videoscope:text:{generation_id}:manifest"))

    @staticmethod
    def _canonical_digest(payload: object) -> str:
        return hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

    def _validate_build_plan(self, plan: TextVectorBuildPlan) -> None:
        if not isinstance(plan, TextVectorBuildPlan):
            raise ValueError("text vector build plan must be validated")
        if plan.index_specification != self.index_specification:
            raise ValueError("text vector build index specification does not match writer")

    def _validated_plan_vectors(
        self,
        plan: TextVectorBuildPlan,
    ) -> tuple[list[np.ndarray], str]:
        ensure_ready = getattr(self.embedding, "ensure_ready", None)
        if ensure_ready is not None and not ensure_ready():
            raise RuntimeError("semantic embedding is not ready")
        vectors = self.embedding.embed([point.text for point in plan.points])
        if len(vectors) != len(plan.points):
            raise ValueError("embedding model returned an unexpected vector count")
        validated: list[np.ndarray] = []
        manifest: list[dict[str, str]] = []
        for point, vector in zip(plan.points, vectors, strict=True):
            try:
                resolved = np.asarray(vector, dtype="<f4")
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("embedding model returned an invalid vector") from error
            if (
                resolved.ndim != 1
                or resolved.shape[0] != self.dimensions
                or not bool(np.isfinite(resolved).all())
            ):
                raise ValueError("embedding model returned an invalid vector")
            contiguous = np.ascontiguousarray(resolved, dtype="<f4")
            validated.append(contiguous)
            manifest.append(
                {
                    "point_id": self.point_id(plan.generation_id, point.segment_id),
                    "vector_sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
                }
            )
        return validated, self._canonical_digest(manifest)

    def _manifest_payload(
        self,
        plan: TextVectorBuildPlan,
        *,
        vector_manifest_sha256: str,
    ) -> dict[str, object]:
        return {
            "collection_name": plan.index_specification.collection_name,
            "generation_id": plan.generation_id,
            "index_specification_hash": plan.index_specification.specification_hash,
            "input_manifest_sha256": plan.input_manifest_sha256,
            "payload_schema_version": plan.index_specification.payload_schema_version,
            "point_count": len(plan.points),
            "point_manifest_sha256": plan.point_manifest_sha256,
            "record_type": "manifest",
            "source_sha256": plan.source_sha256,
            "stage_specification_hash": plan.stage_specification_hash,
            "vector_manifest_sha256": vector_manifest_sha256,
            "video_id": plan.video_id,
        }

    def build_generation(self, plan: TextVectorBuildPlan) -> TextVectorBuildReceipt:
        """Build an invisible immutable generation; SQLite activation is separate."""
        from qdrant_client import models

        self._assert_query_only_snapshot_is_not_mutated()
        self._validate_build_plan(plan)
        vectors, _candidate_vector_manifest = self._validated_plan_vectors(plan)
        segment_points = [
            models.PointStruct(
                id=self.point_id(plan.generation_id, point.segment_id),
                vector=vector.tolist(),
                payload={
                    "generation_id": plan.generation_id,
                    "modality": point.modality,
                    "record_type": "segment",
                    "segment_generation_id": point.segment_generation_id,
                    "segment_id": point.segment_id,
                    "text_sha256": point.text_sha256,
                    "video_id": point.video_id,
                },
            )
            for point, vector in zip(plan.points, vectors, strict=True)
        ]
        manifest_id = self.manifest_point_id(plan.generation_id)
        point_ids = [point.id for point in segment_points] + [manifest_id]
        client = self._get_client()
        existing = client.retrieve(
            collection_name=self.collection_name,
            ids=point_ids,
            with_payload=False,
            with_vectors=False,
        )
        if existing:
            raise ValueError("text vector generation already exists")
        if segment_points:
            client.upsert(
                collection_name=self.collection_name,
                points=segment_points,
                wait=True,
                update_mode=models.UpdateMode.INSERT_ONLY,
            )
        durable_records = client.retrieve(
            collection_name=self.collection_name,
            ids=[point.id for point in segment_points],
            with_payload=True,
            with_vectors=True,
        )
        durable_by_id = {str(record.id): record for record in durable_records}
        durable_vector_manifest: list[dict[str, str]] = []
        for source, point in zip(plan.points, segment_points, strict=True):
            record = durable_by_id.get(str(point.id))
            if record is None or getattr(record, "payload", None) != point.payload:
                raise RuntimeError("text vector generation point was not durably written")
            vector = np.asarray(getattr(record, "vector", None), dtype="<f4")
            if (
                vector.ndim != 1
                or vector.shape[0] != self.dimensions
                or not bool(np.isfinite(vector).all())
            ):
                raise RuntimeError("text vector generation vector was not durably written")
            durable_vector_manifest.append(
                {
                    "point_id": self.point_id(plan.generation_id, source.segment_id),
                    "vector_sha256": hashlib.sha256(
                        np.ascontiguousarray(vector, dtype="<f4").tobytes()
                    ).hexdigest(),
                }
            )
        vector_manifest_sha256 = self._canonical_digest(durable_vector_manifest)
        sentinel = models.PointStruct(
            id=manifest_id,
            vector=[1.0, *([0.0] * (self.dimensions - 1))],
            payload=self._manifest_payload(
                plan,
                vector_manifest_sha256=vector_manifest_sha256,
            ),
        )
        client.upsert(
            collection_name=self.collection_name,
            points=[sentinel],
            wait=True,
            update_mode=models.UpdateMode.INSERT_ONLY,
        )
        receipt = TextVectorBuildReceipt(
            generation_id=plan.generation_id,
            index_specification_hash=plan.index_specification.specification_hash,
            point_count=len(plan.points),
            point_manifest_sha256=plan.point_manifest_sha256,
            vector_manifest_sha256=vector_manifest_sha256,
        )
        probe_binding_payload = self._manifest_payload(
            plan,
            vector_manifest_sha256=receipt.vector_manifest_sha256,
        )
        records = client.retrieve(
            collection_name=self.collection_name,
            ids=[manifest_id],
            with_payload=True,
            with_vectors=False,
        )
        if (
            len(records) != 1
            or getattr(records[0], "payload", None) != probe_binding_payload
        ):
            raise RuntimeError("text vector generation manifest was not durably written")
        count_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="generation_id",
                    match=models.MatchValue(value=plan.generation_id),
                ),
                models.FieldCondition(
                    key="record_type",
                    match=models.MatchValue(value="segment"),
                ),
            ]
        )
        count = client.count(
            collection_name=self.collection_name,
            count_filter=count_filter,
            exact=True,
        ).count
        if count != len(plan.points):
            raise RuntimeError("text vector generation point count is incomplete")
        return receipt

    def _expected_binding_manifest(
        self,
        binding: TextVectorSearchBinding,
    ) -> dict[str, object]:
        generation = binding.generation
        return {
            "collection_name": generation.collection_name,
            "generation_id": generation.generation_id,
            "index_specification_hash": generation.index_specification_hash,
            "input_manifest_sha256": generation.input_manifest_sha256,
            "payload_schema_version": binding.index_specification.payload_schema_version,
            "point_count": generation.point_count,
            "point_manifest_sha256": generation.point_manifest_sha256,
            "record_type": "manifest",
            "source_sha256": generation.source_sha256,
            "stage_specification_hash": generation.specification_hash,
            "vector_manifest_sha256": generation.vector_manifest_sha256,
            "video_id": generation.video_id,
        }

    @staticmethod
    def _generation_validation_cache_key(
        binding: TextVectorSearchBinding,
    ) -> tuple[str, str, str, str, int]:
        generation = binding.generation
        return (
            generation.generation_id,
            generation.index_specification_hash,
            generation.point_manifest_sha256,
            generation.vector_manifest_sha256,
            generation.point_count,
        )

    def validate_generation(
        self,
        binding: TextVectorSearchBinding,
        *,
        exhaustive: bool = False,
    ) -> bool:
        from qdrant_client import models

        if not isinstance(binding, TextVectorSearchBinding):
            raise ValueError("text vector search binding must be validated")
        if binding.index_specification != self.index_specification:
            return False
        try:
            manifest_id = self.manifest_point_id(binding.generation_id)
            manifest_records = self._get_client().retrieve(
                collection_name=self.collection_name,
                ids=[manifest_id],
                with_payload=True,
                with_vectors=False,
            )
            if (
                len(manifest_records) != 1
                or getattr(manifest_records[0], "payload", None)
                != self._expected_binding_manifest(binding)
            ):
                return False
            count_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="generation_id",
                        match=models.MatchValue(value=binding.generation_id),
                    ),
                    models.FieldCondition(
                        key="record_type",
                        match=models.MatchValue(value="segment"),
                    ),
                ]
            )
            count = self._get_client().count(
                collection_name=self.collection_name,
                count_filter=count_filter,
                exact=True,
            ).count
            if count != binding.generation.point_count:
                return False
            if not exhaustive:
                return True
            cache_key = self._generation_validation_cache_key(binding)
            if cache_key in self._validated_generation_cache:
                self._validated_generation_cache.move_to_end(cache_key)
                return True
            segment_ids = [
                self.point_id(binding.generation_id, point.segment_id)
                for point in binding.points
            ]
            records = self._get_client().retrieve(
                collection_name=self.collection_name,
                ids=segment_ids,
                with_payload=True,
                with_vectors=True,
            )
            by_id = {str(record.id): record for record in records}
            if len(by_id) != len(segment_ids):
                return False
            vector_manifest: list[dict[str, str]] = []
            for point, point_id in zip(binding.points, segment_ids, strict=True):
                record = by_id.get(point_id)
                if record is None or getattr(record, "payload", None) != {
                    "generation_id": binding.generation_id,
                    "modality": point.modality,
                    "record_type": "segment",
                    "segment_generation_id": point.segment_generation_id,
                    "segment_id": point.segment_id,
                    "text_sha256": point.text_sha256,
                    "video_id": point.video_id,
                }:
                    return False
                vector = np.asarray(getattr(record, "vector", None), dtype="<f4")
                if (
                    vector.ndim != 1
                    or vector.shape[0] != self.dimensions
                    or not bool(np.isfinite(vector).all())
                ):
                    return False
                vector_manifest.append(
                    {
                        "point_id": point_id,
                        "vector_sha256": hashlib.sha256(
                            np.ascontiguousarray(vector, dtype="<f4").tobytes()
                        ).hexdigest(),
                    }
                )
            if self._canonical_digest(vector_manifest) != binding.generation.vector_manifest_sha256:
                return False
            self._validated_generation_cache[cache_key] = None
            self._validated_generation_cache.move_to_end(cache_key)
            while len(self._validated_generation_cache) > 128:
                self._validated_generation_cache.popitem(last=False)
        except Exception:
            return False
        return True

    def validate_generation_for_benchmark_snapshot(
        self,
        binding: TextVectorSearchBinding,
    ) -> bool:
        """Fully validate persisted points under an attested immutable tree.

        Qdrant cosine collections renormalize float32 values when local storage is
        reopened, so legacy byte-exact vector manifests are not restart-stable.
        This benchmark-only validator skips only that byte digest: it still proves
        the exact manifest, point set, payloads, dimensions, finiteness, and norms,
        while ``snapshot_sha256`` attests every persisted storage byte.
        """
        from qdrant_client import models

        if self._existing_snapshot_digest is None:
            return False
        if not isinstance(binding, TextVectorSearchBinding):
            raise ValueError("text vector search binding must be validated")
        if binding.index_specification != self.index_specification:
            return False
        cache_key = id(binding)
        if (
            cache_key not in self._validated_benchmark_binding_cache
            and len(self._validated_benchmark_binding_cache)
            >= _MAX_QDRANT_BENCHMARK_BINDINGS
        ):
            return False
        self._validated_benchmark_binding_cache.pop(cache_key, None)
        try:
            # Also proves strict encoder identity and the full storage snapshot SHA.
            self.benchmark_attestation
            manifest_records = self._get_client().retrieve(
                collection_name=self.collection_name,
                ids=[self.manifest_point_id(binding.generation_id)],
                with_payload=True,
                with_vectors=False,
            )
            if (
                len(manifest_records) != 1
                or getattr(manifest_records[0], "payload", None)
                != self._expected_binding_manifest(binding)
            ):
                return False
            generation_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="generation_id",
                        match=models.MatchValue(value=binding.generation_id),
                    )
                ]
            )
            segment_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="generation_id",
                        match=models.MatchValue(value=binding.generation_id),
                    ),
                    models.FieldCondition(
                        key="record_type",
                        match=models.MatchValue(value="segment"),
                    ),
                ]
            )
            client = self._get_client()
            total_count = client.count(
                collection_name=self.collection_name,
                count_filter=generation_filter,
                exact=True,
            ).count
            segment_count = client.count(
                collection_name=self.collection_name,
                count_filter=segment_filter,
                exact=True,
            ).count
            if (
                total_count != binding.generation.point_count + 1
                or segment_count != binding.generation.point_count
            ):
                return False
            expected_ids = [
                self.point_id(binding.generation_id, point.segment_id)
                for point in binding.points
            ]
            records = client.retrieve(
                collection_name=self.collection_name,
                ids=expected_ids,
                with_payload=True,
                with_vectors=True,
            )
            if len(records) != len(expected_ids):
                return False
            by_id = {str(getattr(record, "id", "")): record for record in records}
            if set(by_id) != set(expected_ids) or len(by_id) != len(records):
                return False
            for point, point_id in zip(binding.points, expected_ids, strict=True):
                record = by_id[point_id]
                if getattr(record, "payload", None) != {
                    "generation_id": binding.generation_id,
                    "modality": point.modality,
                    "record_type": "segment",
                    "segment_generation_id": point.segment_generation_id,
                    "segment_id": point.segment_id,
                    "text_sha256": point.text_sha256,
                    "video_id": point.video_id,
                }:
                    return False
                vector = np.asarray(getattr(record, "vector", None), dtype=np.float64)
                if (
                    vector.ndim != 1
                    or vector.shape[0] != self.dimensions
                    or not bool(np.isfinite(vector).all())
                ):
                    return False
                norm = float(np.linalg.norm(vector))
                if not math.isfinite(norm) or abs(norm - 1.0) > 1e-4:
                    return False
        except Exception:
            return False
        point_lookup = {point.segment_id: point for point in binding.points}
        if len(point_lookup) != binding.generation.point_count:
            return False
        self._validated_benchmark_binding_cache[cache_key] = (
            binding,
            point_lookup,
        )
        self._validated_benchmark_binding_cache.move_to_end(cache_key)
        return True

    def _benchmark_binding_attestation(
        self,
        binding: TextVectorSearchBinding,
    ) -> Mapping[str, TextVectorPointSource] | None:
        if self._existing_snapshot_digest is None:
            return None
        cache_key = id(binding)
        cached = self._validated_benchmark_binding_cache.get(cache_key)
        if cached is None or cached[0] is not binding:
            return None
        self._validated_benchmark_binding_cache.move_to_end(cache_key)
        return cached[1]

    def release_benchmark_snapshot_bindings(
        self,
        bindings: tuple[TextVectorSearchBinding, ...],
    ) -> None:
        """Release exact binding attestations owned by one closed session."""
        if (
            type(bindings) is not tuple
            or len(bindings) > _MAX_QDRANT_BENCHMARK_BINDINGS
            or any(not isinstance(binding, TextVectorSearchBinding) for binding in bindings)
        ):
            raise ValueError("benchmark binding release must be a bounded validated tuple")
        for binding in bindings:
            cache_key = id(binding)
            cached = self._validated_benchmark_binding_cache.get(cache_key)
            if cached is not None and cached[0] is binding:
                self._validated_benchmark_binding_cache.pop(cache_key, None)

    def search_generations(
        self,
        query: str,
        *,
        bindings: list[TextVectorSearchBinding] | tuple[TextVectorSearchBinding, ...],
        modalities: set[str] | None = None,
        limit: int = 50,
        exhaustive_validation: bool = False,
    ) -> list[TextVectorSearchHit]:
        from qdrant_client import models

        if not query.strip() or limit <= 0 or not bindings:
            return []
        resolved_bindings = tuple(bindings)
        if any(not isinstance(item, TextVectorSearchBinding) for item in resolved_bindings):
            raise ValueError("text vector search bindings must be validated")
        allowed_pairs: set[tuple[str, str]] = set()
        expected_points: dict[
            tuple[str, str], Mapping[str, TextVectorPointSource]
        ] = {}
        for item in resolved_bindings:
            pair = (item.video_id, item.generation_id)
            if pair in allowed_pairs:
                raise ValueError("text vector search bindings must be unique")
            attested_points = (
                self._benchmark_binding_attestation(item)
                if not exhaustive_validation
                else None
            )
            if attested_points is None:
                if not self.validate_generation(
                    item,
                    exhaustive=exhaustive_validation,
                ):
                    raise ValueError(
                        "text vector generation manifest is missing or invalid"
                    )
                attested_points = {
                    point.segment_id: point for point in item.points
                }
            allowed_pairs.add(pair)
            expected_points[pair] = attested_points
        allowed_modalities = {"speech", "ocr", "objects"} if modalities is None else set(modalities)
        if not allowed_modalities <= {"speech", "ocr", "objects"}:
            raise ValueError("unsupported text vector search modality")
        pair_filters = [
            models.Filter(
                must=[
                    models.FieldCondition(
                        key="video_id",
                        match=models.MatchValue(value=video_id),
                    ),
                    models.FieldCondition(
                        key="generation_id",
                        match=models.MatchValue(value=generation_id),
                    ),
                ]
            )
            for video_id, generation_id in sorted(allowed_pairs)
        ]
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="record_type",
                    match=models.MatchValue(value="segment"),
                ),
                models.FieldCondition(
                    key="modality",
                    match=models.MatchAny(any=sorted(allowed_modalities)),
                ),
            ],
            should=pair_filters,
            min_should=models.MinShould(conditions=pair_filters, min_count=1),
        )
        embed_query = getattr(self.embedding, "embed_query", None)
        raw_vector = embed_query(query) if embed_query else self.embedding.embed([query])[0]
        try:
            vector = np.asarray(raw_vector, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("embedding model returned an invalid query vector") from error
        if (
            vector.ndim != 1
            or vector.shape[0] != self.dimensions
            or not bool(np.isfinite(vector).all())
        ):
            raise ValueError("embedding model returned an invalid query vector")
        response = self._get_client().query_points(
            collection_name=self.collection_name,
            query=vector.tolist(),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        response_points = getattr(response, "points", None)
        if type(response_points) not in (list, tuple) or len(response_points) > limit:
            raise RuntimeError(
                "Qdrant returned an invalid generation-scoped search response"
            )
        hits: list[TextVectorSearchHit] = []
        returned_points: set[tuple[str, str, str]] = set()
        for point in response_points:
            payload = getattr(point, "payload", None)
            if not isinstance(payload, Mapping):
                raise RuntimeError(
                    "Qdrant returned an invalid generation-scoped search result"
                )
            video_id = payload.get("video_id")
            generation_id = payload.get("generation_id")
            segment_id = payload.get("segment_id")
            modality = payload.get("modality")
            if (
                type(video_id) is not str
                or type(generation_id) is not str
                or (video_id, generation_id) not in allowed_pairs
                or type(segment_id) is not str
                or type(modality) is not str
                or modality not in allowed_modalities
            ):
                raise RuntimeError(
                    "Qdrant returned an invalid generation-scoped search result"
                )
            expected_point = expected_points[(video_id, generation_id)].get(segment_id)
            point_key = (video_id, generation_id, segment_id)
            if (
                expected_point is None
                or point_key in returned_points
                or dict(payload)
                != {
                    "generation_id": generation_id,
                    "modality": expected_point.modality,
                    "record_type": "segment",
                    "segment_generation_id": expected_point.segment_generation_id,
                    "segment_id": expected_point.segment_id,
                    "text_sha256": expected_point.text_sha256,
                    "video_id": expected_point.video_id,
                }
            ):
                raise RuntimeError(
                    "Qdrant returned an invalid generation-scoped search result"
                )
            try:
                if isinstance(point.score, bool):
                    raise ValueError("Qdrant score must be numeric")
                score = float(point.score)
                if (
                    not math.isfinite(score)
                    or score < -1.0 - _QDRANT_COSINE_SCORE_TOLERANCE
                    or score > 1.0 + _QDRANT_COSINE_SCORE_TOLERANCE
                ):
                    raise ValueError("Qdrant score is outside the cosine range")
                hit = TextVectorSearchHit(
                    video_id=video_id,
                    generation_id=generation_id,
                    segment_id=segment_id,
                    modality=modality,
                    score=max(0.0, min(1.0, score)),
                )
            except (AttributeError, TypeError, ValueError, OverflowError) as error:
                raise RuntimeError(
                    "Qdrant returned an invalid generation-scoped search result"
                ) from error
            returned_points.add(point_key)
            hits.append(hit)
        return hits

    def delete_generation(
        self,
        generation_id: str,
        *,
        index_specification_hash: str,
    ) -> None:
        validate_artifact_identifier(generation_id, field_name="text vector generation id")
        if index_specification_hash != self.index_specification.specification_hash:
            raise ValueError("text vector GC index specification does not match writer")
        self.delete_generation_from_storage(
            generation_id,
            index_specification_hash=index_specification_hash,
            collection_name=self.collection_name,
        )

    def delete_generation_from_storage(
        self,
        generation_id: str,
        *,
        index_specification_hash: str,
        collection_name: str,
    ) -> None:
        """Delete one persisted generation without consulting the current embedder."""
        self._assert_query_only_snapshot_is_not_mutated()
        validate_artifact_identifier(generation_id, field_name="text vector generation id")
        _validate_generation_storage_scope(
            index_specification_hash=index_specification_hash,
            collection_name=collection_name,
        )
        self._reject_symlinked_storage_path()
        from qdrant_client import models

        generation_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="generation_id",
                    match=models.MatchValue(value=generation_id),
                )
            ]
        )
        try:
            client = self._get_storage_client(create_path=False)
            if client is None or not client.collection_exists(collection_name):
                self._evict_generation_validation_cache(generation_id)
                return
            client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(filter=generation_filter),
                wait=True,
            )
            remaining = client.count(
                collection_name=collection_name,
                count_filter=generation_filter,
                exact=True,
            ).count
        except QdrantStorageUnavailableError:
            raise
        except ValueError as error:
            raise QdrantStorageUnavailableError(
                "Qdrant generation storage is unavailable"
            ) from error
        if remaining != 0:
            raise RuntimeError("text vector generation still contains points after GC")
        self._evict_generation_validation_cache(generation_id)

    def _evict_generation_validation_cache(self, generation_id: str) -> None:
        for cache_key in tuple(self._validated_generation_cache):
            if cache_key[0] == generation_id:
                self._validated_generation_cache.pop(cache_key, None)

    def replace_video(self, video_id: str, segments: list[SegmentRecord]) -> None:
        self._assert_query_only_snapshot_is_not_mutated()
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
                    "record_type": "legacy",
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
                ),
                models.FieldCondition(
                    key="record_type",
                    match=models.MatchValue(value="legacy"),
                ),
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
        must = [
            models.FieldCondition(
                key="record_type",
                match=models.MatchValue(value="legacy"),
            )
        ]
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
    supports_generation_provenance = True

    def __init__(self, embedding: HashEmbedding | None = None) -> None:
        self.embedding = embedding or HashEmbedding()
        self.embedding_identity = str(
            getattr(
                self.embedding,
                "identity",
                f"{type(self.embedding).__module__}.{type(self.embedding).__qualname__}:{self.embedding.dimensions}",
            )
        )
        self.index_specification = TextVectorIndexSpecification(
            embedding_identity=self.embedding_identity,
            dimensions=int(self.embedding.dimensions),
        )
        self._segments: dict[str, SegmentRecord] = {}
        self._generation_records: dict[
            str,
            tuple[TextVectorBuildPlan, TextVectorBuildReceipt, tuple[np.ndarray, ...]],
        ] = {}

    def build_generation(self, plan: TextVectorBuildPlan) -> TextVectorBuildReceipt:
        if not isinstance(plan, TextVectorBuildPlan):
            raise ValueError("text vector build plan must be validated")
        if plan.index_specification != self.index_specification:
            raise ValueError("text vector build index specification does not match writer")
        if plan.generation_id in self._generation_records:
            raise ValueError("text vector generation already exists")
        vectors = self.embedding.embed([point.text for point in plan.points])
        if len(vectors) != len(plan.points):
            raise ValueError("embedding model returned an unexpected vector count")
        validated: list[np.ndarray] = []
        manifest: list[dict[str, str]] = []
        for point, vector in zip(plan.points, vectors, strict=True):
            try:
                resolved = np.asarray(vector, dtype="<f4")
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("embedding model returned an invalid vector") from error
            if (
                resolved.ndim != 1
                or resolved.shape[0] != self.index_specification.dimensions
                or not bool(np.isfinite(resolved).all())
            ):
                raise ValueError("embedding model returned an invalid vector")
            contiguous = np.ascontiguousarray(resolved, dtype="<f4")
            validated.append(contiguous)
            manifest.append(
                {
                    "point_id": QdrantVectorIndex.point_id(
                        plan.generation_id,
                        point.segment_id,
                    ),
                    "vector_sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
                }
            )
        receipt = TextVectorBuildReceipt(
            generation_id=plan.generation_id,
            index_specification_hash=plan.index_specification.specification_hash,
            point_count=len(plan.points),
            point_manifest_sha256=plan.point_manifest_sha256,
            vector_manifest_sha256=QdrantVectorIndex._canonical_digest(manifest),
        )
        self._generation_records[plan.generation_id] = (
            plan,
            receipt,
            tuple(validated),
        )
        return receipt

    def validate_generation(
        self,
        binding: TextVectorSearchBinding,
        *,
        exhaustive: bool = False,
    ) -> bool:
        del exhaustive
        if not isinstance(binding, TextVectorSearchBinding):
            raise ValueError("text vector search binding must be validated")
        record = self._generation_records.get(binding.generation_id)
        if record is None:
            return False
        plan, receipt, vectors = record
        generation = binding.generation
        return (
            binding.index_specification == self.index_specification
            and plan.video_id == generation.video_id
            and plan.stage_specification_hash == generation.specification_hash
            and plan.source_sha256 == generation.source_sha256
            and plan.index_specification.specification_hash
            == generation.index_specification_hash
            and plan.input_manifest_sha256 == generation.input_manifest_sha256
            and plan.point_manifest_sha256 == generation.point_manifest_sha256
            and plan.points == binding.points
            and receipt.vector_manifest_sha256 == generation.vector_manifest_sha256
            and receipt.point_count == generation.point_count == len(vectors)
        )

    def search_generations(
        self,
        query: str,
        *,
        bindings: list[TextVectorSearchBinding] | tuple[TextVectorSearchBinding, ...],
        modalities: set[str] | None = None,
        limit: int = 50,
        exhaustive_validation: bool = False,
    ) -> list[TextVectorSearchHit]:
        del exhaustive_validation
        if not query.strip() or limit <= 0 or not bindings:
            return []
        allowed_modalities = {"speech", "ocr", "objects"} if modalities is None else set(modalities)
        if not allowed_modalities <= {"speech", "ocr", "objects"}:
            raise ValueError("unsupported text vector search modality")
        query_vector = np.asarray(self.embedding.embed([query])[0], dtype=np.float32)
        if (
            query_vector.ndim != 1
            or query_vector.shape[0] != self.index_specification.dimensions
            or not bool(np.isfinite(query_vector).all())
        ):
            raise ValueError("embedding model returned an invalid query vector")
        ranked: list[TextVectorSearchHit] = []
        for binding in bindings:
            if not self.validate_generation(binding):
                raise ValueError("text vector generation manifest is missing or invalid")
            plan, _receipt, vectors = self._generation_records[binding.generation_id]
            for point, vector in zip(plan.points, vectors, strict=True):
                if point.modality not in allowed_modalities:
                    continue
                score = float(np.dot(query_vector, vector))
                if not math.isfinite(score):
                    continue
                ranked.append(
                    TextVectorSearchHit(
                        video_id=point.video_id,
                        generation_id=plan.generation_id,
                        segment_id=point.segment_id,
                        modality=point.modality,
                        score=max(0.0, min(1.0, score)),
                    )
                )
        return sorted(ranked, key=lambda item: item.score, reverse=True)[:limit]

    def delete_generation(
        self,
        generation_id: str,
        *,
        index_specification_hash: str,
    ) -> None:
        validate_artifact_identifier(generation_id, field_name="text vector generation id")
        if index_specification_hash != self.index_specification.specification_hash:
            raise ValueError("text vector GC index specification does not match writer")
        self.delete_generation_from_storage(
            generation_id,
            index_specification_hash=index_specification_hash,
            collection_name=self.index_specification.collection_name,
        )

    def delete_generation_from_storage(
        self,
        generation_id: str,
        *,
        index_specification_hash: str,
        collection_name: str,
    ) -> None:
        validate_artifact_identifier(generation_id, field_name="text vector generation id")
        _validate_generation_storage_scope(
            index_specification_hash=index_specification_hash,
            collection_name=collection_name,
        )
        record = self._generation_records.get(generation_id)
        if record is None:
            return
        plan, _receipt, _vectors = record
        if (
            plan.index_specification.specification_hash != index_specification_hash
            or plan.index_specification.collection_name != collection_name
        ):
            raise ValueError("stored text vector generation does not match GC scope")
        self._generation_records.pop(generation_id)
        if generation_id in self._generation_records:  # pragma: no cover - dict invariant
            raise RuntimeError("text vector generation still exists after GC")

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
