import numpy as np
import pytest
import sys
import types
from pathlib import Path

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


def test_semantic_embedding_loads_exact_offline_snapshot(monkeypatch, tmp_path: Path) -> None:
    snapshot = tmp_path / "cache" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    calls: list[dict[str, object]] = []

    def snapshot_download(repository: str, **kwargs: object) -> str:
        assert repository == "xenova/model"
        calls.append(kwargs)
        return str(snapshot)

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    class FakeTextEmbedding:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["model_name"] == "sentence-transformers/model"
            assert kwargs["specific_model_path"] == str(snapshot)

        def embed(self, texts):  # type: ignore[no-untyped-def]
            return iter(np.eye(1, 16, 0, dtype=np.float32)[0] for _ in texts)

        def query_embed(self, _query: str):
            return iter([np.eye(1, 16, 1, dtype=np.float32)[0]])

    fastembed = types.ModuleType("fastembed")
    fastembed.TextEmbedding = FakeTextEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", fastembed)

    embedding = SemanticEmbedding(
        model_name="sentence-transformers/model",
        model_repository="xenova/model",
        model_revision="a" * 40,
        dimensions=16,
        cache_dir=tmp_path / "cache",
    )

    assert embedding.ensure_ready() is True
    assert calls == [
        {
            "revision": "a" * 40,
            "local_files_only": True,
            "cache_dir": str(tmp_path / "cache"),
        }
    ]
    assert embedding.identity == (
        "fastembed@unmanaged:unmanaged:sentence-transformers/model:xenova/model@"
        + "a" * 40
        + ":16"
    )


def test_pinned_embedding_identity_includes_runtime_and_pooling(tmp_path: Path) -> None:
    from videoscope.model_manifest import TEXT_EMBEDDING_MODEL
    from videoscope.search.embeddings import create_semantic_embedding

    embedding = create_semantic_embedding(
        model_name=TEXT_EMBEDDING_MODEL,
        dimensions=768,
        cache_dir=tmp_path,
    )

    assert "fastembed@0.8.0:mean-pooling-v1" in embedding.identity


def test_semantic_embedding_fails_readiness_on_dimension_mismatch(monkeypatch) -> None:
    class WrongDimensionEmbedding:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def embed(self, _texts):  # type: ignore[no-untyped-def]
            return iter([np.ones(8, dtype=np.float32)])

    module = types.ModuleType("fastembed")
    module.TextEmbedding = WrongDimensionEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", module)
    embedding = SemanticEmbedding(model_name="wrong", dimensions=16)

    assert embedding.ensure_ready() is False
    assert embedding.last_error == "embedding model returned an unexpected vector size"


def test_loaded_semantic_embedding_does_not_mix_hash_vectors_after_runtime_failure(
    monkeypatch,
) -> None:
    class BrokenAfterProbeEmbedding:
        calls = 0

        def __init__(self, **_kwargs: object) -> None:
            pass

        def embed(self, _texts):  # type: ignore[no-untyped-def]
            self.calls += 1
            if self.calls == 1:
                return iter([np.ones(16, dtype=np.float32)])
            raise RuntimeError("onnx inference failed")

    module = types.ModuleType("fastembed")
    module.TextEmbedding = BrokenAfterProbeEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", module)
    embedding = SemanticEmbedding(model_name="broken", dimensions=16)

    assert embedding.ensure_ready() is True
    with pytest.raises(RuntimeError, match="semantic embedding failed"):
        embedding.embed(["must not become a hash vector"])
