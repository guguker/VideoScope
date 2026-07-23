import numpy as np
import pytest
import sys
import types

from videoscope.search.embeddings import HashEmbedding, SemanticEmbedding


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


def test_hash_embedding_is_stable_and_normalized() -> None:
    embedder = HashEmbedding(dimensions=128)

    first = embedder.embed(["трёхочковый бросок"])[0]
    second = embedder.embed(["трёхочковый бросок"])[0]

    assert np.array_equal(first, second)
    assert np.linalg.norm(first) == pytest.approx(1.0)


def test_hash_embedding_preserves_more_overlap_for_related_text() -> None:
    embedder = HashEmbedding(dimensions=256)
    query, related, unrelated = embedder.embed(
        ["player makes a three point shot", "three point shot by player", "red scoreboard timeout"]
    )

    assert cosine(query, related) > cosine(query, unrelated)


def test_hash_embedding_handles_empty_text() -> None:
    vector = HashEmbedding(dimensions=32).embed([""])[0]

    assert vector.shape == (32,)
    assert np.count_nonzero(vector) == 0


def test_hash_embedding_rejects_tiny_vector_size() -> None:
    with pytest.raises(ValueError):
        HashEmbedding(dimensions=8)


def test_semantic_embedding_uses_fastembed_for_documents_and_queries(monkeypatch) -> None:
    class FakeTextEmbedding:
        def __init__(self, *, model_name: str) -> None:
            assert model_name == "test-model"

        def embed(self, texts):  # type: ignore[no-untyped-def]
            return iter(np.eye(1, 16, 0, dtype=np.float32)[0] for _ in texts)

        def query_embed(self, _query: str):
            return iter([np.eye(1, 16, 1, dtype=np.float32)[0]])

    module = types.ModuleType("fastembed")
    module.TextEmbedding = FakeTextEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", module)
    embedding = SemanticEmbedding(model_name="test-model", dimensions=16)

    documents = embedding.embed(["one", "two"])
    query = embedding.embed_query("query")

    assert embedding.backend == "fastembed"
    assert len(documents) == 2
    assert np.array_equal(query, np.eye(1, 16, 1, dtype=np.float32)[0])


def test_semantic_embedding_falls_back_after_one_failed_load(monkeypatch) -> None:
    attempts = 0

    class BrokenTextEmbedding:
        def __init__(self, **_kwargs) -> None:  # type: ignore[no-untyped-def]
            nonlocal attempts
            attempts += 1
            raise RuntimeError("offline")

    module = types.ModuleType("fastembed")
    module.TextEmbedding = BrokenTextEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", module)
    embedding = SemanticEmbedding(model_name="missing", dimensions=32)

    first = embedding.embed(["fallback"])[0]
    second = embedding.embed_query("fallback")

    assert attempts == 1
    assert embedding.backend == "hash-fallback"
    assert embedding.last_error == "offline"
    assert np.array_equal(first, second)
