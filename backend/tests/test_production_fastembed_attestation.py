from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from videoscope.benchmark.snapshots import (
    FastEmbedFileManifestEntry,
    FastEmbedSnapshot,
    FastEmbedSnapshotManifest,
)
from videoscope.model_manifest import (
    FASTEMBED_ALGORITHM_VERSION,
    FASTEMBED_REPOSITORY,
    FASTEMBED_RUNTIME_VERSION,
    MODEL_REVISIONS,
    TEXT_EMBEDDING_DIMENSIONS,
    TEXT_EMBEDDING_MODEL,
    fastembed_cache_snapshot_path,
)
from videoscope.search.embeddings import (
    SEMANTIC_EMBEDDING_TYPE_IDENTITY,
    ReviewedSemanticEmbeddingContract,
    ReviewedSemanticEmbeddingError,
    SemanticEmbedding,
    create_reviewed_semantic_embedding_runtime,
    create_semantic_embedding,
    load_reviewed_semantic_embedding_contract,
)
import videoscope.search.embeddings as embeddings_module
import videoscope.benchmark.snapshots as snapshots_module


_MODEL_FILES = {
    "config.json": b"{}",
    "onnx/model.onnx": b"reviewed-onnx",
    "special_tokens_map.json": b"{}",
    "tokenizer.json": b"{}",
    "tokenizer_config.json": b"{}",
}


def _manifest(
    files: dict[str, bytes] | None = None,
) -> FastEmbedSnapshotManifest:
    resolved = _MODEL_FILES if files is None else files
    return FastEmbedSnapshotManifest(
        model_name=TEXT_EMBEDDING_MODEL,
        model_repository=FASTEMBED_REPOSITORY,
        model_revision=MODEL_REVISIONS[FASTEMBED_REPOSITORY],
        runtime_version=FASTEMBED_RUNTIME_VERSION,
        algorithm_version=FASTEMBED_ALGORITHM_VERSION,
        dimensions=TEXT_EMBEDDING_DIMENSIONS,
        files=tuple(
            FastEmbedFileManifestEntry(
                relative_path=relative_path,
                size_bytes=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
            )
            for relative_path, content in sorted(resolved.items())
        ),
    )


def _write_reviewed_cache(cache_dir: Path, manifest: FastEmbedSnapshotManifest) -> Path:
    source = fastembed_cache_snapshot_path(cache_dir, TEXT_EMBEDDING_MODEL)
    assert source is not None
    source.mkdir(parents=True)
    for relative_path, content in _MODEL_FILES.items():
        target = source / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return source


def _install_fake_fastembed(
    monkeypatch: pytest.MonkeyPatch,
    implementation: type[object],
) -> None:
    fastembed = types.ModuleType("fastembed")
    fastembed.TextEmbedding = implementation  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", fastembed)
    monkeypatch.setattr(
        embeddings_module.metadata,
        "version",
        lambda package: (
            FASTEMBED_RUNTIME_VERSION
            if package == "fastembed"
            else pytest.fail(f"unexpected package lookup: {package}")
        ),
    )


def _contract(
    monkeypatch: pytest.MonkeyPatch,
    manifest: FastEmbedSnapshotManifest,
) -> ReviewedSemanticEmbeddingContract:
    monkeypatch.setattr(
        snapshots_module,
        "load_reviewed_fastembed_snapshot_manifest",
        lambda: manifest,
    )
    return load_reviewed_semantic_embedding_contract(
        model_name=TEXT_EMBEDDING_MODEL,
        dimensions=TEXT_EMBEDDING_DIMENSIONS,
    )


def test_reviewed_contract_is_pathless_and_keeps_the_production_storage_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    contract = _contract(monkeypatch, manifest)
    production = create_semantic_embedding(
        model_name=TEXT_EMBEDDING_MODEL,
        dimensions=TEXT_EMBEDDING_DIMENSIONS,
        cache_dir=Path("must-not-be-resolved"),
    )

    assert type(contract) is ReviewedSemanticEmbeddingContract
    assert contract.embedding_identity == production.identity
    assert contract.model_content_sha256 == manifest.model_content_sha256
    assert contract.semantic_embedding_type == SEMANTIC_EMBEDDING_TYPE_IDENTITY
    assert contract.semantic_embedding_type == (
        "videoscope.search.embeddings.SemanticEmbedding"
    )
    assert not any(
        str(value).startswith("/") for value in contract.canonical_dict.values()
    )


def test_runtime_materializes_and_warms_only_the_reviewed_offline_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    contract = _contract(monkeypatch, manifest)
    cache_dir = tmp_path / "cache"
    source = _write_reviewed_cache(cache_dir, manifest)
    scratch_parent = tmp_path / "scratch"
    scratch_parent.mkdir()
    calls: list[dict[str, object]] = []

    class FakeTextEmbedding:
        def __init__(self, **kwargs: object) -> None:
            calls.append(kwargs)

        def embed(self, texts):  # type: ignore[no-untyped-def]
            return iter(
                np.ones(TEXT_EMBEDDING_DIMENSIONS, dtype=np.float32)
                for _ in texts
            )

        def query_embed(self, _query: str):
            return iter([np.ones(TEXT_EMBEDDING_DIMENSIONS, dtype=np.float32)])

    _install_fake_fastembed(monkeypatch, FakeTextEmbedding)
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda *_args, **_kwargs: pytest.fail(  # type: ignore[attr-defined]
        "reviewed production runtime attempted a hub lookup"
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    runtime = create_reviewed_semantic_embedding_runtime(
        model_name=TEXT_EMBEDDING_MODEL,
        dimensions=TEXT_EMBEDDING_DIMENSIONS,
        cache_dir=cache_dir,
        scratch_parent=scratch_parent,
        contract=contract,
    )

    assert type(runtime.embedding) is SemanticEmbedding
    assert runtime.embedding.strict_no_fallback is True
    assert runtime.embedding.identity == contract.embedding_identity
    assert runtime.embedding.attestation_identity == contract.embedding_attestation_dict
    assert runtime.verify_current() is True
    assert len(calls) == 1
    assert set(calls[0]) == {"model_name", "specific_model_path"}
    private_snapshot = Path(str(calls[0]["specific_model_path"]))
    assert private_snapshot != source
    assert private_snapshot.is_dir()
    assert runtime.close() is True
    assert runtime.close() is False
    assert private_snapshot.exists() is False
    assert list(scratch_parent.iterdir()) == []


@pytest.mark.parametrize("failure", ["missing", "corrupt"])
def test_runtime_fails_closed_for_missing_or_corrupt_reviewed_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: str,
) -> None:
    manifest = _manifest()
    contract = _contract(monkeypatch, manifest)
    cache_dir = tmp_path / "cache"
    if failure == "corrupt":
        source = _write_reviewed_cache(cache_dir, manifest)
        (source / "config.json").write_bytes(b"tampered")
    scratch_parent = tmp_path / "scratch"
    scratch_parent.mkdir()

    class MustNotLoad:
        def __init__(self, **_kwargs: object) -> None:
            pytest.fail("FastEmbed received missing or corrupt model bytes")

    _install_fake_fastembed(monkeypatch, MustNotLoad)

    with pytest.raises(
        ReviewedSemanticEmbeddingError,
        match="reviewed semantic embedding runtime is unavailable",
    ):
        create_reviewed_semantic_embedding_runtime(
            model_name=TEXT_EMBEDDING_MODEL,
            dimensions=TEXT_EMBEDDING_DIMENSIONS,
            cache_dir=cache_dir,
            scratch_parent=scratch_parent,
            contract=contract,
        )

    assert list(scratch_parent.iterdir()) == []


def test_runtime_rejects_a_duck_typed_embedding_identity_and_cleans_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    contract = _contract(monkeypatch, manifest)
    cache_dir = tmp_path / "cache"
    _write_reviewed_cache(cache_dir, manifest)
    scratch_parent = tmp_path / "scratch"
    scratch_parent.mkdir()
    forged = SimpleNamespace(
        identity=contract.embedding_identity,
        dimensions=contract.dimensions,
        strict_no_fallback=True,
        attestation_identity=contract.embedding_attestation_dict,
        ensure_ready=lambda: True,
        verify_benchmark_model_current=lambda: True,
    )
    monkeypatch.setattr(
        FastEmbedSnapshot,
        "create_embedding",
        lambda _snapshot: forged,
    )

    with pytest.raises(ReviewedSemanticEmbeddingError):
        create_reviewed_semantic_embedding_runtime(
            model_name=TEXT_EMBEDDING_MODEL,
            dimensions=TEXT_EMBEDDING_DIMENSIONS,
            cache_dir=cache_dir,
            scratch_parent=scratch_parent,
            contract=contract,
        )

    assert list(scratch_parent.iterdir()) == []


def test_runtime_final_verification_failure_discards_the_private_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    contract = _contract(monkeypatch, manifest)
    cache_dir = tmp_path / "cache"
    _write_reviewed_cache(cache_dir, manifest)
    scratch_parent = tmp_path / "scratch"
    scratch_parent.mkdir()

    class FakeTextEmbedding:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def embed(self, texts):  # type: ignore[no-untyped-def]
            return iter(
                np.ones(TEXT_EMBEDDING_DIMENSIONS, dtype=np.float32)
                for _ in texts
            )

    _install_fake_fastembed(monkeypatch, FakeTextEmbedding)
    monkeypatch.setattr(
        SemanticEmbedding,
        "verify_benchmark_model_current",
        lambda _embedding: False,
    )

    with pytest.raises(ReviewedSemanticEmbeddingError):
        create_reviewed_semantic_embedding_runtime(
            model_name=TEXT_EMBEDDING_MODEL,
            dimensions=TEXT_EMBEDDING_DIMENSIONS,
            cache_dir=cache_dir,
            scratch_parent=scratch_parent,
            contract=contract,
        )

    assert list(scratch_parent.iterdir()) == []


def test_runtime_discards_inference_if_private_snapshot_is_tampered_mid_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    contract = _contract(monkeypatch, manifest)
    cache_dir = tmp_path / "cache"
    _write_reviewed_cache(cache_dir, manifest)
    scratch_parent = tmp_path / "scratch"
    scratch_parent.mkdir()
    calls = 0
    private_path: Path | None = None

    class MutatingTextEmbedding:
        def __init__(self, **kwargs: object) -> None:
            nonlocal private_path
            private_path = Path(str(kwargs["specific_model_path"]))

        def embed(self, texts):  # type: ignore[no-untyped-def]
            nonlocal calls
            calls += 1
            vectors = [
                np.ones(TEXT_EMBEDDING_DIMENSIONS, dtype=np.float32)
                for _ in texts
            ]
            if calls == 2:
                assert private_path is not None
                (private_path / "config.json").write_bytes(b"tampered-mid-inference")
            return iter(vectors)

    _install_fake_fastembed(monkeypatch, MutatingTextEmbedding)
    runtime = create_reviewed_semantic_embedding_runtime(
        model_name=TEXT_EMBEDDING_MODEL,
        dimensions=TEXT_EMBEDDING_DIMENSIONS,
        cache_dir=cache_dir,
        scratch_parent=scratch_parent,
        contract=contract,
    )

    with pytest.raises(RuntimeError, match="semantic embedding failed"):
        runtime.embedding.embed(["discard this vector"])

    assert runtime.embedding.backend == "unavailable"
    assert runtime.verify_current() is False
    assert runtime.close() is True
