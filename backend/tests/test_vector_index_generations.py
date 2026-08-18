from __future__ import annotations

from datetime import UTC, datetime, timedelta
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

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
from videoscope.search.vector_index import MemoryVectorIndex, QdrantVectorIndex


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
def test_generation_search_drops_non_finite_qdrant_scores(tmp_path, monkeypatch, score) -> None:  # type: ignore[no-untyped-def]
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

    assert index.search_generations(
        "basket",
        bindings=[binding],
        modalities={"speech"},
    ) == []
