from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
import hashlib
from threading import Barrier, BrokenBarrierError, Lock
from types import SimpleNamespace

import numpy as np
import pytest

import videoscope.search.vector_index as vector_index_module
from videoscope.artifacts import (
    StageKind,
    TextVectorBuildPlan,
    TextVectorGeneration,
    TextVectorGenerationInput,
    TextVectorIndexSpecification,
    TextVectorPointSource,
    TextVectorSearchBinding,
)
from videoscope.search.embeddings import HashEmbedding
from videoscope.search.vector_index import (
    MemoryVectorIndex,
    QdrantStorageContractError,
    QdrantStorageUnavailableError,
    QdrantVectorIndex,
)


def _index_specification(dimensions: int = 16) -> TextVectorIndexSpecification:
    return TextVectorIndexSpecification(
        embedding_identity=f"hash-embedding-v1:{dimensions}",
        dimensions=dimensions,
    )


def _inputs(*, present: bool = True) -> tuple[TextVectorGenerationInput, ...]:
    output = []
    for kind, marker in zip(
        (StageKind.SPEECH, StageKind.OCR, StageKind.OBJECTS),
        ("b", "c", "d"),
        strict=True,
    ):
        is_present = present and kind is StageKind.SPEECH
        output.append(
            TextVectorGenerationInput(
                stage_kind=kind,
                specification_hash=marker * 64,
                segment_generation_id=f"{kind.value}-generation-1" if is_present else None,
                segment_run_id=f"{kind.value}-run-1" if is_present else None,
                source_sha256="a" * 64 if is_present else None,
                segment_count=1 if is_present else 0,
                content_manifest_sha256=marker * 64,
            )
        )
    return tuple(output)


def _plan(
    generation_id: str,
    *,
    text: str = "made basket",
    empty: bool = False,
    dimensions: int = 16,
) -> TextVectorBuildPlan:
    now = datetime.now(UTC)
    point = TextVectorPointSource(
        video_id="video-1",
        segment_id=f"segment-{generation_id}",
        modality="speech",
        text=text,
        segment_generation_id="speech-generation-1",
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )
    return TextVectorBuildPlan(
        generation_id=generation_id,
        run_id=f"run-{generation_id}",
        video_id="video-1",
        stage_specification_hash="e" * 64,
        source_sha256="a" * 64,
        index_specification=_index_specification(dimensions),
        expected_previous_generation_id=None,
        inputs=_inputs(present=not empty),
        points=() if empty else (point,),
        input_manifest_sha256="1" * 64,
        point_manifest_sha256="2" * 64,
        reserved_at=now.isoformat(),
        lease_expires_at=(now + timedelta(minutes=5)).isoformat(),
    )


def _binding(plan: TextVectorBuildPlan, receipt) -> TextVectorSearchBinding:  # type: ignore[no-untyped-def]
    generation = TextVectorGeneration(
        generation_id=plan.generation_id,
        video_id=plan.video_id,
        specification_hash=plan.stage_specification_hash,
        source_sha256=plan.source_sha256,
        run_id=plan.run_id,
        index_specification_hash=plan.index_specification.specification_hash,
        collection_name=plan.index_specification.collection_name,
        input_manifest_sha256=plan.input_manifest_sha256,
        point_manifest_sha256=plan.point_manifest_sha256,
        vector_manifest_sha256=receipt.vector_manifest_sha256,
        point_count=len(plan.points),
        completed_at=datetime.now(UTC).isoformat(),
    )
    return TextVectorSearchBinding(
        generation=generation,
        index_specification=plan.index_specification,
        inputs=plan.inputs,
        points=plan.points,
    )


@pytest.mark.parametrize("index_type", [QdrantVectorIndex, MemoryVectorIndex])
def test_generation_build_is_immutable_and_search_is_scoped_to_active_binding(
    tmp_path,
    index_type,
) -> None:  # type: ignore[no-untyped-def]
    if index_type is QdrantVectorIndex:
        index = index_type(tmp_path / "qdrant", embedding=HashEmbedding(16))
    else:
        index = index_type(HashEmbedding(16))
    first = _plan("11111111111111111111111111111111", text="old phrase")
    second = _plan("22222222222222222222222222222222", text="new basket phrase")

    first_receipt = index.build_generation(first)
    second_receipt = index.build_generation(second)
    hits = index.search_generations(
        "new basket phrase",
        bindings=[_binding(second, second_receipt)],
        modalities={"speech"},
        limit=10,
    )

    assert [(hit.generation_id, hit.segment_id) for hit in hits] == [
        (second.generation_id, second.points[0].segment_id)
    ]
    assert all(hit.generation_id != first.generation_id for hit in hits)
    with pytest.raises(ValueError, match="already exists"):
        index.build_generation(first)
    assert index.validate_generation(_binding(first, first_receipt)) is True


@pytest.mark.parametrize("index_type", [QdrantVectorIndex, MemoryVectorIndex])
def test_empty_generation_has_verified_manifest_sentinel(tmp_path, index_type) -> None:  # type: ignore[no-untyped-def]
    if index_type is QdrantVectorIndex:
        index = index_type(tmp_path / "qdrant", embedding=HashEmbedding(16))
    else:
        index = index_type(HashEmbedding(16))
    plan = _plan("33333333333333333333333333333333", empty=True)

    receipt = index.build_generation(plan)
    binding = _binding(plan, receipt)

    assert receipt.point_count == 0
    assert index.validate_generation(binding) is True
    assert index.search_generations(
        "anything",
        bindings=[binding],
        modalities={"speech"},
        limit=10,
    ) == []


def test_qdrant_rejects_missing_or_tampered_manifest_before_returning_hits(tmp_path) -> None:
    index = QdrantVectorIndex(tmp_path / "qdrant", embedding=HashEmbedding(16))
    plan = _plan("44444444444444444444444444444444")
    receipt = index.build_generation(plan)
    binding = _binding(plan, receipt)
    client = index._get_client()
    client.delete(
        collection_name=index.collection_name,
        points_selector=[index.manifest_point_id(plan.generation_id)],
        wait=True,
    )

    assert index.validate_generation(binding) is False
    with pytest.raises(ValueError, match="manifest"):
        index.search_generations(
            "basket",
            bindings=[binding],
            modalities={"speech"},
            limit=10,
        )


def test_failed_new_qdrant_build_does_not_mutate_previous_generation(tmp_path) -> None:
    class MutableEmbedding(HashEmbedding):
        broken = False

        def embed(self, texts):  # type: ignore[no-untyped-def]
            if self.broken:
                return [np.array([float("nan")] * self.dimensions)] * len(texts)
            return super().embed(texts)

    embedding = MutableEmbedding(16)
    index = QdrantVectorIndex(tmp_path / "qdrant", embedding=embedding)
    first = _plan("55555555555555555555555555555555")
    first_receipt = index.build_generation(first)
    embedding.broken = True

    with pytest.raises(ValueError, match="invalid vector"):
        index.build_generation(_plan("66666666666666666666666666666666"))

    assert index.validate_generation(_binding(first, first_receipt)) is True


def test_generation_gc_deletes_only_exact_generation(tmp_path) -> None:
    index = QdrantVectorIndex(tmp_path / "qdrant", embedding=HashEmbedding(16))
    first = _plan("77777777777777777777777777777777")
    second = _plan("88888888888888888888888888888888")
    first_receipt = index.build_generation(first)
    second_receipt = index.build_generation(second)

    index.delete_generation(
        first.generation_id,
        index_specification_hash=first.index_specification.specification_hash,
    )

    assert index.validate_generation(_binding(first, first_receipt)) is False
    assert index.validate_generation(_binding(second, second_receipt)) is True


def test_qdrant_storage_gc_uses_persisted_scope_not_current_embedding(tmp_path) -> None:
    path = tmp_path / "qdrant"
    writer = QdrantVectorIndex(path, embedding=HashEmbedding(16))
    plan = _plan("79797979797979797979797979797979")
    writer.build_generation(plan)
    persisted_hash = plan.index_specification.specification_hash
    persisted_collection = plan.index_specification.collection_name
    writer.close()

    gc_index = QdrantVectorIndex(path, embedding=HashEmbedding(32))
    gc_index.delete_generation_from_storage(
        plan.generation_id,
        index_specification_hash=persisted_hash,
        collection_name=persisted_collection,
    )

    client = gc_index._client
    assert client is not None
    assert client.collection_exists(persisted_collection) is True
    assert client.collection_exists(gc_index.collection_name) is False
    assert gc_index._get_client() is client
    assert client.collection_exists(gc_index.collection_name) is True


def test_qdrant_storage_gc_does_not_create_absent_storage_or_collection(tmp_path) -> None:
    absent_path = tmp_path / "absent-qdrant"
    index = QdrantVectorIndex(absent_path, embedding=HashEmbedding(16))
    specification = _index_specification(16)

    index.delete_generation_from_storage(
        "7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a",
        index_specification_hash=specification.specification_hash,
        collection_name=specification.collection_name,
    )

    assert absent_path.exists() is False
    assert index._client is None

    active_client = index._get_client()
    other_specification = _index_specification(32)
    index.delete_generation_from_storage(
        "7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b",
        index_specification_hash=other_specification.specification_hash,
        collection_name=other_specification.collection_name,
    )

    assert active_client.collection_exists(index.collection_name) is True
    assert active_client.collection_exists(other_specification.collection_name) is False
    assert index._client is active_client


def test_one_qdrant_client_is_reused_by_concurrent_gc_and_indexer(
    tmp_path,
    monkeypatch,
) -> None:
    import qdrant_client

    path = tmp_path / "qdrant"
    path.mkdir()
    constructor_barrier = Barrier(2)
    start_barrier = Barrier(3)
    counter_lock = Lock()
    constructed: list[object] = []

    class ConcurrentClient:
        def __init__(self, *, path: str) -> None:
            del path
            with counter_lock:
                constructed.append(self)
            try:
                constructor_barrier.wait(timeout=0.5)
            except BrokenBarrierError:
                pass
            self.collections: set[str] = set()
            self.closed = False

        def collection_exists(self, collection_name: str) -> bool:
            return collection_name in self.collections

        def create_collection(self, *, collection_name: str, **_kwargs) -> None:  # type: ignore[no-untyped-def]
            self.collections.add(collection_name)

        @staticmethod
        def delete(**_kwargs) -> None:  # type: ignore[no-untyped-def]
            pass

        @staticmethod
        def count(**_kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(count=0)

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(qdrant_client, "QdrantClient", ConcurrentClient)
    index = QdrantVectorIndex(path, embedding=HashEmbedding(16))

    def run_gc() -> None:
        start_barrier.wait()
        index.delete_generation_from_storage(
            "7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a7a",
            index_specification_hash=index.index_specification.specification_hash,
            collection_name=index.collection_name,
        )

    def run_indexer():  # type: ignore[no-untyped-def]
        start_barrier.wait()
        return index._get_client()

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            gc_future = pool.submit(run_gc)
            indexer_future = pool.submit(run_indexer)
            start_barrier.wait()
            gc_future.result(timeout=2)
            client = indexer_future.result(timeout=2)

        assert len(constructed) == 1
        assert client is constructed[0]
        assert index._client is client
        assert client.collection_exists(index.collection_name) is True
    finally:
        index.close()


def test_qdrant_gc_separates_contract_errors_from_transient_storage_value_errors(
    tmp_path,
) -> None:
    class StorageFailureClient:
        @staticmethod
        def collection_exists(_collection_name: str) -> bool:
            return True

        @staticmethod
        def delete(**_kwargs) -> None:  # type: ignore[no-untyped-def]
            raise ValueError("embedded storage is already borrowed")

    index = QdrantVectorIndex(tmp_path / "qdrant", embedding=HashEmbedding(16))
    index.path.mkdir()
    index._client = StorageFailureClient()

    with pytest.raises(QdrantStorageContractError):
        index.delete_generation_from_storage(
            "7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b",
            index_specification_hash="f" * 64,
            collection_name="unsafe",
        )

    with pytest.raises(QdrantStorageUnavailableError) as failure:
        index.delete_generation_from_storage(
            "7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b7b",
            index_specification_hash=index.index_specification.specification_hash,
            collection_name=index.collection_name,
        )
    assert not isinstance(failure.value, ValueError)


def test_qdrant_client_open_value_error_is_a_transient_storage_failure(
    tmp_path,
    monkeypatch,
) -> None:
    import qdrant_client

    class FailingClient:
        def __init__(self, *, path: str) -> None:
            del path
            raise ValueError("storage folder is already accessed")

    monkeypatch.setattr(qdrant_client, "QdrantClient", FailingClient)
    index = QdrantVectorIndex(tmp_path / "qdrant", embedding=HashEmbedding(16))

    with pytest.raises(QdrantStorageUnavailableError) as failure:
        index._get_storage_client(create_path=True)

    assert not isinstance(failure.value, ValueError)
    assert index._client is None


def test_qdrant_close_retains_client_until_underlying_close_succeeds(tmp_path) -> None:
    class RetryableCloseClient:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise ValueError("embedded storage is still closing")

    index = QdrantVectorIndex(tmp_path / "qdrant", embedding=HashEmbedding(16))
    client = RetryableCloseClient()
    index._client = client
    index._current_collection_ready = True

    with pytest.raises(QdrantStorageUnavailableError):
        index.close()

    assert index._client is client
    assert index._current_collection_ready is True

    index.close()

    assert client.close_calls == 2
    assert index._client is None
    assert index._current_collection_ready is False


@pytest.mark.parametrize("symlink_final_path", [True, False])
def test_qdrant_storage_gc_rejects_symlinked_storage_path_or_ancestor(
    tmp_path,
    symlink_final_path,
) -> None:
    real_root = tmp_path / "real-root"
    real_path = real_root / "text"
    writer = QdrantVectorIndex(real_path, embedding=HashEmbedding(16))
    plan = _plan("7a8a7a8a7a8a7a8a7a8a7a8a7a8a7a8a")
    receipt = writer.build_generation(plan)
    binding = _binding(plan, receipt)
    writer.close()

    if symlink_final_path:
        unsafe_path = tmp_path / "linked-text"
        unsafe_path.symlink_to(real_path, target_is_directory=True)
    else:
        linked_root = tmp_path / "linked-root"
        linked_root.symlink_to(real_root, target_is_directory=True)
        unsafe_path = linked_root / "text"
    gc_index = QdrantVectorIndex(unsafe_path, embedding=HashEmbedding(32))

    try:
        with pytest.raises(RuntimeError, match="symlink"):
            gc_index.delete_generation_from_storage(
                plan.generation_id,
                index_specification_hash=plan.index_specification.specification_hash,
                collection_name=plan.index_specification.collection_name,
            )
    finally:
        gc_index.close()

    verifier = QdrantVectorIndex(real_path, embedding=HashEmbedding(16))
    try:
        assert verifier.validate_generation(binding) is True
    finally:
        verifier.close()


def test_qdrant_storage_gc_waits_and_verifies_exact_generation_count(
    tmp_path,
    monkeypatch,
) -> None:
    index = QdrantVectorIndex(tmp_path / "qdrant", embedding=HashEmbedding(16))
    plan = _plan("7c7c7c7c7c7c7c7c7c7c7c7c7c7c7c7c")
    index.build_generation(plan)
    client = index._get_client()
    original_delete = client.delete
    original_count = client.count
    observed: dict[str, object] = {}

    def recording_delete(**kwargs):  # type: ignore[no-untyped-def]
        observed["wait"] = kwargs.get("wait")
        return original_delete(**kwargs)

    def recording_count(**kwargs):  # type: ignore[no-untyped-def]
        observed["exact"] = kwargs.get("exact")
        return original_count(**kwargs)

    monkeypatch.setattr(client, "delete", recording_delete)
    monkeypatch.setattr(client, "count", recording_count)

    index.delete_generation_from_storage(
        plan.generation_id,
        index_specification_hash=plan.index_specification.specification_hash,
        collection_name=plan.index_specification.collection_name,
    )

    assert observed == {"wait": True, "exact": True}


def test_qdrant_storage_gc_fails_when_points_remain_after_delete(
    tmp_path,
    monkeypatch,
) -> None:
    index = QdrantVectorIndex(tmp_path / "qdrant", embedding=HashEmbedding(16))
    plan = _plan("7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d7d")
    index.build_generation(plan)
    monkeypatch.setattr(index._get_client(), "delete", lambda **_kwargs: None)

    with pytest.raises(RuntimeError, match="still contains points"):
        index.delete_generation_from_storage(
            plan.generation_id,
            index_specification_hash=plan.index_specification.specification_hash,
            collection_name=plan.index_specification.collection_name,
        )


@pytest.mark.parametrize("index_type", [QdrantVectorIndex, MemoryVectorIndex])
@pytest.mark.parametrize(
    ("specification_hash", "collection_name"),
    [
        ("not-a-sha256", "videoscope_text_v1_unsafe"),
        ("f" * 64, "../unsafe"),
        ("f" * 64, "videoscope_text_v1_" + "e" * 32),
    ],
)
def test_storage_gc_rejects_unsafe_or_mismatched_scope(
    tmp_path,
    index_type,
    specification_hash,
    collection_name,
) -> None:  # type: ignore[no-untyped-def]
    index = (
        index_type(tmp_path / "qdrant", embedding=HashEmbedding(16))
        if index_type is QdrantVectorIndex
        else index_type(HashEmbedding(16))
    )

    with pytest.raises(ValueError, match="specification|collection"):
        index.delete_generation_from_storage(
            "7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e",
            index_specification_hash=specification_hash,
            collection_name=collection_name,
        )


def test_memory_storage_gc_uses_recorded_generation_scope() -> None:
    index = MemoryVectorIndex(HashEmbedding(16))
    plan = _plan("7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f")
    index.build_generation(plan)
    index.index_specification = _index_specification(32)

    index.delete_generation_from_storage(
        plan.generation_id,
        index_specification_hash=plan.index_specification.specification_hash,
        collection_name=plan.index_specification.collection_name,
    )

    assert plan.generation_id not in index._generation_records


def test_exhaustive_validation_is_cached_but_sentinel_and_count_are_always_rechecked(
    tmp_path,
    monkeypatch,
) -> None:
    index = QdrantVectorIndex(tmp_path / "qdrant", embedding=HashEmbedding(16))
    plan = _plan("99999999999999999999999999999999")
    receipt = index.build_generation(plan)
    binding = _binding(plan, receipt)
    client = index._get_client()
    original_retrieve = client.retrieve
    vector_reads = 0

    def recording_retrieve(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal vector_reads
        if kwargs.get("with_vectors"):
            vector_reads += 1
        return original_retrieve(*args, **kwargs)

    monkeypatch.setattr(client, "retrieve", recording_retrieve)

    assert index.validate_generation(binding, exhaustive=True) is True
    assert index.validate_generation(binding, exhaustive=True) is True
    index.search_generations(
        "basket",
        bindings=[binding],
        modalities={"speech"},
        exhaustive_validation=True,
    )
    assert vector_reads == 1

    client.delete(
        collection_name=index.collection_name,
        points_selector=[index.manifest_point_id(plan.generation_id)],
        wait=True,
    )
    assert index.validate_generation(binding, exhaustive=True) is False


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_generation_search_rejects_non_finite_qdrant_scores(tmp_path, monkeypatch, score) -> None:  # type: ignore[no-untyped-def]
    index = QdrantVectorIndex(tmp_path / "qdrant", embedding=HashEmbedding(16))
    plan = _plan("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    receipt = index.build_generation(plan)
    binding = _binding(plan, receipt)
    point = plan.points[0]
    monkeypatch.setattr(
        index._get_client(),
        "query_points",
        lambda **_kwargs: SimpleNamespace(
            points=[
                SimpleNamespace(
                    score=score,
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
            ]
        ),
    )

    with pytest.raises(RuntimeError, match="invalid generation-scoped search result"):
        index.search_generations(
            "basket",
            bindings=[binding],
            modalities={"speech"},
        )


@pytest.mark.parametrize(
    "corruption",
    [
        "non_mapping_payload",
        "unbound_generation",
        "unknown_segment",
        "wrong_record_type",
    ],
)
def test_generation_search_rejects_malformed_or_out_of_scope_qdrant_payloads(
    tmp_path,
    monkeypatch,
    corruption,
) -> None:  # type: ignore[no-untyped-def]
    index = QdrantVectorIndex(tmp_path / "qdrant", embedding=HashEmbedding(16))
    plan = _plan("abababababababababababababababab")
    receipt = index.build_generation(plan)
    binding = _binding(plan, receipt)
    point = plan.points[0]
    payload: object = {
        "generation_id": plan.generation_id,
        "modality": point.modality,
        "record_type": "segment",
        "segment_generation_id": point.segment_generation_id,
        "segment_id": point.segment_id,
        "text_sha256": point.text_sha256,
        "video_id": point.video_id,
    }
    if corruption == "non_mapping_payload":
        payload = ["not", "a", "mapping"]
    elif corruption == "unbound_generation":
        assert isinstance(payload, dict)
        payload["generation_id"] = "cd" * 16
    elif corruption == "unknown_segment":
        assert isinstance(payload, dict)
        payload["segment_id"] = "unknown-segment"
    else:
        assert isinstance(payload, dict)
        payload["record_type"] = "manifest"
    monkeypatch.setattr(
        index._get_client(),
        "query_points",
        lambda **_kwargs: SimpleNamespace(
            points=[SimpleNamespace(score=0.75, payload=payload)]
        ),
    )

    with pytest.raises(RuntimeError, match="invalid generation-scoped search result"):
        index.search_generations(
            "basket",
            bindings=[binding],
            modalities={"speech"},
        )


def test_generation_search_surfaces_hit_construction_failure(
    tmp_path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    index = QdrantVectorIndex(tmp_path / "qdrant", embedding=HashEmbedding(16))
    plan = _plan("acacacacacacacacacacacacacacacac")
    receipt = index.build_generation(plan)
    binding = _binding(plan, receipt)
    point = plan.points[0]
    monkeypatch.setattr(
        index._get_client(),
        "query_points",
        lambda **_kwargs: SimpleNamespace(
            points=[
                SimpleNamespace(
                    score=0.75,
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
            ]
        ),
    )
    monkeypatch.setattr(
        vector_index_module,
        "TextVectorSearchHit",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("fixture construction failure")),
    )

    with pytest.raises(RuntimeError, match="invalid generation-scoped search result"):
        index.search_generations(
            "basket",
            bindings=[binding],
            modalities={"speech"},
        )


@pytest.mark.parametrize("score", [1.01, -1.01, 1e300, -1e300, True])
def test_generation_search_rejects_out_of_range_or_boolean_qdrant_scores(
    tmp_path,
    monkeypatch,
    score,
) -> None:  # type: ignore[no-untyped-def]
    index = QdrantVectorIndex(tmp_path / "qdrant", embedding=HashEmbedding(16))
    plan = _plan("adadadadadadadadadadadadadadadad")
    receipt = index.build_generation(plan)
    binding = _binding(plan, receipt)
    point = plan.points[0]
    monkeypatch.setattr(
        index._get_client(),
        "query_points",
        lambda **_kwargs: SimpleNamespace(
            points=[
                SimpleNamespace(
                    score=score,
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
            ]
        ),
    )

    with pytest.raises(RuntimeError, match="invalid generation-scoped search result"):
        index.search_generations(
            "basket",
            bindings=[binding],
            modalities={"speech"},
        )


@pytest.mark.parametrize(
    ("score", "expected"),
    [(1.0 + 5e-7, 1.0), (-1.0 - 5e-7, 0.0)],
)
def test_generation_search_allows_tiny_cosine_rounding_tolerance(
    tmp_path,
    monkeypatch,
    score,
    expected,
) -> None:  # type: ignore[no-untyped-def]
    index = QdrantVectorIndex(tmp_path / "qdrant", embedding=HashEmbedding(16))
    plan = _plan("b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0")
    receipt = index.build_generation(plan)
    binding = _binding(plan, receipt)
    point = plan.points[0]
    monkeypatch.setattr(
        index._get_client(),
        "query_points",
        lambda **_kwargs: SimpleNamespace(
            points=[
                SimpleNamespace(
                    score=score,
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
            ]
        ),
    )

    hits = index.search_generations(
        "basket",
        bindings=[binding],
        modalities={"speech"},
    )

    assert [hit.score for hit in hits] == [expected]


def test_generation_search_rejects_over_limit_response_before_point_iteration(
    tmp_path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    index = QdrantVectorIndex(tmp_path / "qdrant", embedding=HashEmbedding(16))
    plan = _plan("aeaeaeaeaeaeaeaeaeaeaeaeaeaeaeae")
    receipt = index.build_generation(plan)
    binding = _binding(plan, receipt)

    class ExplodingPointList(list):
        def __iter__(self):  # type: ignore[no-untyped-def]
            raise AssertionError("over-limit response points were consumed")

    monkeypatch.setattr(
        index._get_client(),
        "query_points",
        lambda **_kwargs: SimpleNamespace(
            points=ExplodingPointList([object(), object()])
        ),
    )

    with pytest.raises(RuntimeError, match="invalid generation-scoped search response"):
        index.search_generations(
            "basket",
            bindings=[binding],
            modalities={"speech"},
            limit=1,
        )


def test_generation_search_rejects_lazy_response_points_without_iteration(
    tmp_path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    index = QdrantVectorIndex(tmp_path / "qdrant", embedding=HashEmbedding(16))
    plan = _plan("afafafafafafafafafafafafafafafaf")
    receipt = index.build_generation(plan)
    binding = _binding(plan, receipt)

    def exploding_points():  # type: ignore[no-untyped-def]
        raise AssertionError("lazy response points were consumed")
        yield object()

    monkeypatch.setattr(
        index._get_client(),
        "query_points",
        lambda **_kwargs: SimpleNamespace(points=exploding_points()),
    )

    with pytest.raises(RuntimeError, match="invalid generation-scoped search response"):
        index.search_generations(
            "basket",
            bindings=[binding],
            modalities={"speech"},
            limit=1,
        )
