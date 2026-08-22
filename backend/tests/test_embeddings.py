import numpy as np
import os
import pytest
import sys
import types
from pathlib import Path

from videoscope.search.embeddings import HashEmbedding, SemanticEmbedding


def _strict_embedding_options(model_path: Path) -> dict[str, object]:
    return {
        "model_name": "sentence-transformers/model",
        "model_repository": "xenova/model",
        "model_revision": "a" * 40,
        "expected_runtime_version": "0.8.0",
        "algorithm_version": "mean-pooling-v1",
        "dimensions": 16,
        "strict": True,
        "specific_model_path": model_path,
        "model_content_sha256": "b" * 64,
    }


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


def test_strict_semantic_embedding_uses_only_verified_specific_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "verified-model"
    model_path.mkdir()
    calls: list[dict[str, object]] = []

    class FakeTextEmbedding:
        def __init__(self, **kwargs: object) -> None:
            calls.append(kwargs)

        def embed(self, texts):  # type: ignore[no-untyped-def]
            return iter(np.array([3.0, 4.0, *([0.0] * 14)]) for _ in texts)

        def query_embed(self, _query: str):
            return iter([np.array([0.0, 6.0, 8.0, *([0.0] * 13)])])

    fastembed = types.ModuleType("fastembed")
    fastembed.TextEmbedding = FakeTextEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", fastembed)
    monkeypatch.setattr(
        "videoscope.search.embeddings.metadata.version",
        lambda package: "0.8.0" if package == "fastembed" else pytest.fail(package),
    )
    embedding = SemanticEmbedding(
        model_name="sentence-transformers/model",
        model_repository="xenova/model",
        model_revision="a" * 40,
        expected_runtime_version="0.8.0",
        algorithm_version="mean-pooling-v1",
        dimensions=16,
        cache_dir=tmp_path / "must-not-be-created",
        strict=True,
        specific_model_path=model_path,
        model_content_sha256="b" * 64,
    )

    query = embedding.embed_query("query")

    assert calls == [
        {
            "model_name": "sentence-transformers/model",
            "specific_model_path": str(model_path),
        }
    ]
    assert np.linalg.norm(query) == pytest.approx(1.0)
    assert np.all(np.isfinite(query))
    assert (tmp_path / "must-not-be-created").exists() is False
    assert embedding.strict_no_fallback is True
    assert embedding.model_content_sha256 == "b" * 64


def test_strict_embedding_reverifies_model_snapshot_after_loading(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "verified-model"
    model_path.mkdir()
    verifier_calls = 0
    model_current = True

    def verify_model() -> None:
        nonlocal verifier_calls
        verifier_calls += 1
        if not model_current:
            raise RuntimeError("model snapshot drifted")

    class FakeTextEmbedding:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def embed(self, texts):  # type: ignore[no-untyped-def]
            return iter(np.ones(16, dtype=np.float32) for _ in texts)

    fastembed = types.ModuleType("fastembed")
    fastembed.TextEmbedding = FakeTextEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", fastembed)
    monkeypatch.setattr(
        "videoscope.search.embeddings.metadata.version",
        lambda _package: "0.8.0",
    )
    embedding = SemanticEmbedding(
        **_strict_embedding_options(model_path),
        model_verifier=verify_model,
    )

    assert embedding.ensure_ready() is True
    # Pre-load, post-load, and post-probe verification are distinct boundaries.
    assert verifier_calls == 3
    assert embedding.verify_benchmark_model_current() is True
    assert verifier_calls == 4
    model_current = False
    assert embedding.verify_benchmark_model_current() is False
    assert verifier_calls == 5


def test_strict_snapshot_keeps_the_production_storage_embedding_identity(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "verified-model"
    model_path.mkdir()
    common = {
        "model_name": "sentence-transformers/model",
        "model_repository": "xenova/model",
        "model_revision": "a" * 40,
        "expected_runtime_version": "0.8.0",
        "algorithm_version": "mean-pooling-v1",
        "dimensions": 16,
    }
    production = SemanticEmbedding(**common)
    benchmark = SemanticEmbedding(
        **common,
        strict=True,
        specific_model_path=model_path,
        model_content_sha256="b" * 64,
        model_verifier=lambda: None,
    )

    assert benchmark.identity == production.identity
    assert benchmark.benchmark_identity == {
        "algorithm_version": "mean-pooling-v1",
        "dimensions": 16,
        "embedding_identity": production.identity,
        "model_content_sha256": "b" * 64,
        "model_name": "sentence-transformers/model",
        "model_repository": "xenova/model",
        "model_revision": "a" * 40,
        "runtime_version": "0.8.0",
    }
    from videoscope.search.vector_index import QdrantVectorIndex

    writer = QdrantVectorIndex(tmp_path / "writer", embedding=production)
    reader = QdrantVectorIndex(tmp_path / "reader", embedding=benchmark)
    assert reader.index_specification == writer.index_specification
    assert reader.collection_name == writer.collection_name


def test_strict_semantic_embedding_never_falls_back_or_creates_cache(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "verified-model"
    model_path.mkdir()

    class BrokenTextEmbedding:
        def __init__(self, **_kwargs: object) -> None:
            raise RuntimeError("cannot load verified model")

    fastembed = types.ModuleType("fastembed")
    fastembed.TextEmbedding = BrokenTextEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", fastembed)
    monkeypatch.setattr(
        "videoscope.search.embeddings.metadata.version",
        lambda _package: "0.8.0",
    )
    monkeypatch.setattr(
        HashEmbedding,
        "embed",
        lambda *_args, **_kwargs: pytest.fail("strict mode reached hash fallback"),
    )
    cache = tmp_path / "must-not-exist"
    embedding = SemanticEmbedding(
        model_name="sentence-transformers/model",
        model_repository="xenova/model",
        model_revision="a" * 40,
        expected_runtime_version="0.8.0",
        algorithm_version="mean-pooling-v1",
        dimensions=16,
        cache_dir=cache,
        strict=True,
        specific_model_path=model_path,
        model_content_sha256="b" * 64,
    )

    with pytest.raises(RuntimeError, match="strict semantic embedding is unavailable"):
        embedding.embed(["query"])

    assert embedding.backend == "unavailable"
    assert cache.exists() is False


def test_strict_semantic_embedding_fails_closed_on_runtime_version_drift(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "verified-model"
    model_path.mkdir()
    fastembed = types.ModuleType("fastembed")
    fastembed.TextEmbedding = lambda **_kwargs: pytest.fail(  # type: ignore[attr-defined]
        "model must not load under the wrong runtime"
    )
    monkeypatch.setitem(sys.modules, "fastembed", fastembed)
    monkeypatch.setattr(
        "videoscope.search.embeddings.metadata.version",
        lambda _package: "0.8.1",
    )
    embedding = SemanticEmbedding(
        model_name="sentence-transformers/model",
        model_repository="xenova/model",
        model_revision="a" * 40,
        expected_runtime_version="0.8.0",
        algorithm_version="mean-pooling-v1",
        dimensions=16,
        strict=True,
        specific_model_path=model_path,
        model_content_sha256="b" * 64,
    )

    with pytest.raises(RuntimeError, match="strict semantic embedding is unavailable"):
        embedding.embed_query("query")

    assert embedding.last_error == "fastembed 0.8.0 is required"


@pytest.mark.parametrize(
    "overrides",
    [
        {"specific_model_path": None},
        {"model_content_sha256": None},
        {"expected_runtime_version": None},
        {"model_repository": None},
        {"model_revision": None},
    ],
)
def test_strict_semantic_embedding_requires_complete_verified_identity(
    tmp_path: Path,
    overrides: dict[str, object],
) -> None:
    model_path = tmp_path / "verified-model"
    model_path.mkdir()
    options: dict[str, object] = {
        "model_name": "sentence-transformers/model",
        "model_repository": "xenova/model",
        "model_revision": "a" * 40,
        "expected_runtime_version": "0.8.0",
        "algorithm_version": "mean-pooling-v1",
        "dimensions": 16,
        "strict": True,
        "specific_model_path": model_path,
        "model_content_sha256": "b" * 64,
    }
    options.update(overrides)

    with pytest.raises(ValueError, match="strict semantic embedding"):
        SemanticEmbedding(**options)  # type: ignore[arg-type]


def test_strict_semantic_embedding_rejects_symlinked_model_ancestor(
    tmp_path: Path,
) -> None:
    real_root = tmp_path / "real"
    model_path = real_root / "model"
    model_path.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(ValueError, match="verified model path"):
        SemanticEmbedding(
            model_name="sentence-transformers/model",
            model_repository="xenova/model",
            model_revision="a" * 40,
            expected_runtime_version="0.8.0",
            algorithm_version="mean-pooling-v1",
            dimensions=16,
            strict=True,
            specific_model_path=alias / "model",
            model_content_sha256="b" * 64,
        )


def test_strict_semantic_embedding_accepts_canonical_macos_file_id_path_after_rename(
    tmp_path: Path,
    monkeypatch,
) -> None:
    logical_parent = tmp_path / "logical-parent"
    anchor = logical_parent / "anchor"
    model = anchor / "model"
    model.mkdir(parents=True)
    anchor_metadata = anchor.stat()
    stable_root = Path(
        f"/.vol/{anchor_metadata.st_dev}/{anchor_metadata.st_ino}"
    )
    stable_model = stable_root / "model"
    logical_parent.rename(tmp_path / "moved-parent")
    replacement = logical_parent / "anchor" / "model"
    replacement.mkdir(parents=True)
    real_open = os.open
    direct_root_opens: list[int] = []
    child_opens: list[int] = []

    def tracked_open(path, flags, *args, **kwargs):  # type: ignore[no-untyped-def]
        if os.fspath(path) == os.fspath(stable_root):
            direct_root_opens.append(flags)
        if path == "model" and isinstance(kwargs.get("dir_fd"), int):
            child_opens.append(kwargs["dir_fd"])
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", tracked_open)

    embedding = SemanticEmbedding(**_strict_embedding_options(stable_model))

    assert embedding.specific_model_path == stable_model
    assert direct_root_opens
    assert all(flags & os.O_NOFOLLOW for flags in direct_root_opens)
    assert child_opens
    assert list(replacement.iterdir()) == []


@pytest.mark.parametrize(
    "mutation",
    ["device-leading-zero", "inode-leading-zero", "parent", "wrong-magic"],
)
def test_strict_semantic_embedding_rejects_noncanonical_macos_file_id_paths(
    tmp_path: Path,
    mutation: str,
) -> None:
    anchor = tmp_path / "anchor"
    model = anchor / "model"
    model.mkdir(parents=True)
    metadata = anchor.stat()
    if mutation == "device-leading-zero":
        candidate = Path(f"/.vol/0{metadata.st_dev}/{metadata.st_ino}/model")
    elif mutation == "inode-leading-zero":
        candidate = Path(f"/.vol/{metadata.st_dev}/0{metadata.st_ino}/model")
    elif mutation == "parent":
        candidate = Path(f"/.vol/{metadata.st_dev}/{metadata.st_ino}/../model")
    else:
        candidate = Path(f"/.volume/{metadata.st_dev}/{metadata.st_ino}/model")

    with pytest.raises(ValueError, match="verified model path"):
        SemanticEmbedding(**_strict_embedding_options(candidate))


def test_strict_semantic_embedding_rejects_macos_file_id_symlink_child(
    tmp_path: Path,
) -> None:
    anchor = tmp_path / "anchor"
    real_model = anchor / "real-model"
    real_model.mkdir(parents=True)
    (anchor / "alias").symlink_to(real_model, target_is_directory=True)
    metadata = anchor.stat()
    candidate = Path(f"/.vol/{metadata.st_dev}/{metadata.st_ino}/alias")

    with pytest.raises(ValueError, match="verified model path"):
        SemanticEmbedding(**_strict_embedding_options(candidate))


def test_strict_semantic_embedding_rejects_macos_file_id_identity_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    anchor = tmp_path / "anchor"
    model = anchor / "model"
    model.mkdir(parents=True)
    metadata = anchor.stat()
    stable_root = Path(f"/.vol/{metadata.st_dev}/{metadata.st_ino}")
    candidate = stable_root / "model"
    real_stat = os.stat
    stable_stats = 0

    def drifting_stat(path, *args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal stable_stats
        result = real_stat(path, *args, **kwargs)
        if os.fspath(path) == os.fspath(stable_root):
            stable_stats += 1
            if stable_stats == 2:
                values = list(result)
                values[1] += 1
                return os.stat_result(values)
        return result

    monkeypatch.setattr(os, "stat", drifting_stat)

    with pytest.raises(ValueError, match="verified model path"):
        SemanticEmbedding(**_strict_embedding_options(candidate))


@pytest.mark.parametrize(
    "query_vector",
    [
        np.array([float("nan"), *([1.0] * 15)]),
        np.zeros(16),
        np.ones(8),
    ],
)
def test_strict_semantic_embedding_rejects_invalid_query_vectors(
    monkeypatch,
    tmp_path: Path,
    query_vector: np.ndarray,
) -> None:
    model_path = tmp_path / "verified-model"
    model_path.mkdir()

    class FakeTextEmbedding:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def query_embed(self, _query: str):
            return iter([query_vector])

    fastembed = types.ModuleType("fastembed")
    fastembed.TextEmbedding = FakeTextEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", fastembed)
    monkeypatch.setattr(
        "videoscope.search.embeddings.metadata.version",
        lambda _package: "0.8.0",
    )
    embedding = SemanticEmbedding(
        model_name="sentence-transformers/model",
        model_repository="xenova/model",
        model_revision="a" * 40,
        expected_runtime_version="0.8.0",
        algorithm_version="mean-pooling-v1",
        dimensions=16,
        strict=True,
        specific_model_path=model_path,
        model_content_sha256="b" * 64,
    )

    with pytest.raises(RuntimeError, match="semantic query embedding failed"):
        embedding.embed_query("query")


def test_strict_semantic_embedding_discards_vectors_when_snapshot_changes_during_inference(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "verified-model"
    model_path.mkdir()
    current = True
    verifier_calls = 0

    def verify_model() -> None:
        nonlocal verifier_calls
        verifier_calls += 1
        if not current:
            raise RuntimeError("reviewed model bytes changed")

    class MutatingTextEmbedding:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def embed(self, texts):  # type: ignore[no-untyped-def]
            nonlocal current
            vectors = [np.ones(16, dtype=np.float32) for _ in texts]
            current = False
            return iter(vectors)

    fastembed = types.ModuleType("fastembed")
    fastembed.TextEmbedding = MutatingTextEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", fastembed)
    monkeypatch.setattr(
        "videoscope.search.embeddings.metadata.version",
        lambda _package: "0.8.0",
    )
    embedding = SemanticEmbedding(
        **_strict_embedding_options(model_path),
        model_verifier=verify_model,
    )

    with pytest.raises(RuntimeError, match="semantic embedding failed"):
        embedding.embed(["must be discarded"])

    assert verifier_calls >= 3
    assert embedding.backend == "unavailable"
