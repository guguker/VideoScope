from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace
import types

import numpy as np
import pytest

from videoscope.artifacts import (
    StageKind,
    TextVectorBuildPlan,
    TextVectorGeneration,
    TextVectorGenerationInput,
    TextVectorPointSource,
    TextVectorSearchBinding,
)
from videoscope.benchmark import snapshots as snapshots_module
from videoscope.benchmark.snapshots import (
    FastEmbedFileManifestEntry,
    FastEmbedSnapshotManifest,
    QdrantSnapshotLimits,
    QdrantStorageSnapshot,
    SnapshotError,
    load_fastembed_snapshot_manifest,
    load_reviewed_fastembed_snapshot_manifest,
    materialize_fastembed_snapshot,
    snapshot_qdrant_storage,
    verify_qdrant_storage_snapshot,
)
from videoscope.model_manifest import (
    FASTEMBED_ALGORITHM_VERSION,
    FASTEMBED_REPOSITORY,
    FASTEMBED_RUNTIME_VERSION,
    MODEL_REVISIONS,
    TEXT_EMBEDDING_DIMENSIONS,
    TEXT_EMBEDDING_MODEL,
)
from videoscope.search.embeddings import HashEmbedding
from videoscope.search.vector_index import (
    QdrantSnapshotCleanupError,
    QdrantVectorIndex,
)


_MODEL_FILES = {
    "config.json": b'{"model":"fixture"}',
    "tokenizer.json": b'{"tokenizer":"fixture"}',
    "tokenizer_config.json": b"{}",
    "special_tokens_map.json": b"{}",
    "onnx/model.onnx": b"fixture-onnx-model",
}


def _write_files(root: Path, files: dict[str, bytes]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for relative, content in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)


def _tree_identity(root: Path) -> tuple[tuple[object, ...], ...]:
    entries: list[tuple[object, ...]] = []
    for path in (root, *sorted(root.rglob("*"))):
        metadata = path.lstat()
        relative = "." if path == root else path.relative_to(root).as_posix()
        digest = (
            hashlib.sha256(path.read_bytes()).hexdigest()
            if stat.S_ISREG(metadata.st_mode)
            else None
        )
        entries.append(
            (
                relative,
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_mode,
                metadata.st_nlink,
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
                digest,
            )
        )
    return tuple(entries)


class _CountingScandir:
    def __init__(self, iterator, counter: list[int]) -> None:  # type: ignore[no-untyped-def]
        self._iterator = iterator
        self._counter = counter

    def __enter__(self) -> _CountingScandir:
        self._iterator.__enter__()
        return self

    def __exit__(self, *args: object) -> object:
        return self._iterator.__exit__(*args)

    def __iter__(self) -> _CountingScandir:
        return self

    def __next__(self):  # type: ignore[no-untyped-def]
        entry = next(self._iterator)
        self._counter[0] += 1
        return entry

    def close(self) -> None:
        self._iterator.close()


def _track_scandir_consumption(
    monkeypatch,
    target: Path,
) -> list[list[int]]:  # type: ignore[no-untyped-def]
    target_metadata = target.stat()
    target_identity = (target_metadata.st_dev, target_metadata.st_ino)
    real_scandir = os.scandir
    consumptions: list[list[int]] = []

    def tracked(path):  # type: ignore[no-untyped-def]
        iterator = real_scandir(path)
        if isinstance(path, int):
            metadata = os.fstat(path)
            if (metadata.st_dev, metadata.st_ino) == target_identity:
                counter = [0]
                consumptions.append(counter)
                return _CountingScandir(iterator, counter)
        return iterator

    monkeypatch.setattr(os, "scandir", tracked)
    return consumptions


def _invoke_snapshot_builder(kind: str, source: Path, scratch: Path) -> object:
    if kind == "fastembed":
        return materialize_fastembed_snapshot(source, scratch, _fastembed_manifest())
    return snapshot_qdrant_storage(source, scratch)


def _snapshot_builder_source(tmp_path: Path, kind: str) -> Path:
    source = tmp_path / "source"
    if kind == "fastembed":
        _write_files(source, _MODEL_FILES)
    else:
        source.mkdir()
        (source / "state").write_bytes(b"qdrant fixture")
    return source


def _retain_scratch_then_replace_its_logical_parent(
    tmp_path: Path,
):  # type: ignore[no-untyped-def]
    logical_parent = tmp_path / "logical-parent"
    logical_parent.mkdir(mode=0o700)
    scratch = logical_parent / "scratch"
    scratch.mkdir(mode=0o700)
    scratch_fd = os.open(scratch, os.O_RDONLY | os.O_DIRECTORY)
    try:
        capability = snapshots_module.RetainedDirectory.retain(
            scratch,
            scratch_fd,
        )
    finally:
        os.close(scratch_fd)
    moved_parent = tmp_path / "moved-parent"
    logical_parent.rename(moved_parent)
    logical_parent.mkdir(mode=0o700)
    replacement = logical_parent / "scratch"
    replacement.mkdir(mode=0o700)
    return capability, moved_parent / "scratch", replacement


def _retain_data_root_then_replace_its_logical_parent(
    tmp_path: Path,
    kind: str,
):  # type: ignore[no-untyped-def]
    logical_parent = tmp_path / "logical-data-parent"
    data_root = logical_parent / "data"
    source = data_root / "source"
    if kind == "fastembed":
        _write_files(source, _MODEL_FILES)
    else:
        source.mkdir(parents=True)
        (source / "state").write_bytes(b"original qdrant")
    (data_root / "glossary.json").write_bytes(b'{"source":"original"}')
    data_fd = os.open(data_root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        capability = snapshots_module.RetainedDirectory.retain(data_root, data_fd)
    finally:
        os.close(data_fd)
    source_view = capability.child("source")
    moved_parent = tmp_path / "moved-data-parent"
    logical_parent.rename(moved_parent)
    replacement_root = logical_parent / "data"
    replacement_source = replacement_root / "source"
    if kind == "fastembed":
        _write_files(
            replacement_source,
            {name: b"replacement" for name in _MODEL_FILES},
        )
    else:
        replacement_source.mkdir(parents=True)
        (replacement_source / "state").write_bytes(b"replacement qdrant")
    (replacement_root / "glossary.json").write_bytes(b'{"source":"replacement"}')
    return capability, source_view, moved_parent / "data", replacement_root


def _fastembed_manifest(
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
                relative_path=relative,
                size_bytes=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
            )
            for relative, content in sorted(resolved.items())
        ),
    )


def _write_huggingface_fastembed_snapshot(
    root: Path,
    *,
    files: dict[str, bytes] | None = None,
) -> tuple[Path, Path, dict[str, Path]]:
    resolved = _MODEL_FILES if files is None else files
    repository = (
        root
        / "models--xenova--paraphrase-multilingual-mpnet-base-v2"
    )
    snapshot = (
        repository
        / "snapshots"
        / MODEL_REVISIONS[FASTEMBED_REPOSITORY]
    )
    blobs = repository / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    blob_paths: dict[str, Path] = {}
    for relative, content in sorted(resolved.items()):
        blob_name = hashlib.sha256(relative.encode("utf-8") + b"\0" + content).hexdigest()
        blob = blobs / blob_name
        blob.write_bytes(content)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(os.path.relpath(blob, start=target.parent))
        blob_paths[relative] = blob
    return snapshot, repository, blob_paths


def test_fastembed_snapshot_copies_only_exact_manifest_into_private_atomic_root(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_files(source, {**_MODEL_FILES, "README.md": b"must not be copied"})
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    manifest = _fastembed_manifest()

    snapshot = materialize_fastembed_snapshot(source, scratch, manifest)

    assert snapshot.identity.model_content_sha256 == manifest.model_content_sha256
    assert snapshot.identity.canonical_dict == {
        "algorithm_version": FASTEMBED_ALGORITHM_VERSION,
        "dimensions": TEXT_EMBEDDING_DIMENSIONS,
        "model_content_sha256": manifest.model_content_sha256,
        "model_name": TEXT_EMBEDDING_MODEL,
        "model_repository": FASTEMBED_REPOSITORY,
        "model_revision": MODEL_REVISIONS[FASTEMBED_REPOSITORY],
        "runtime_version": FASTEMBED_RUNTIME_VERSION,
    }
    assert stat.S_IMODE(snapshot.path.stat().st_mode) == 0o700
    assert sorted(
        str(item.relative_to(snapshot.path))
        for item in snapshot.path.rglob("*")
        if item.is_file()
    ) == sorted(_MODEL_FILES)
    assert all(
        stat.S_IMODE((snapshot.path / relative).stat().st_mode) == 0o600
        for relative in _MODEL_FILES
    )
    assert (snapshot.path / "README.md").exists() is False
    assert not any("partial" in item.name for item in scratch.iterdir())


def test_reviewed_fastembed_manifest_is_packaged_and_exactly_pinned() -> None:
    manifest = load_reviewed_fastembed_snapshot_manifest()

    assert manifest.canonical_dict == {
        "algorithm_version": FASTEMBED_ALGORITHM_VERSION,
        "dimensions": TEXT_EMBEDDING_DIMENSIONS,
        "files": [
            {
                "relative_path": "config.json",
                "sha256": "14533f158fb2476510b3a83256386b983daa8cfa799ea35217c89947be63835e",
                "size_bytes": 751,
            },
            {
                "relative_path": "onnx/model.onnx",
                "sha256": "92f682c55d39e728187c6bffd488f17462d4e05e3c5fe86ef9ba6f38d66122ed",
                "size_bytes": 1_110_059_084,
            },
            {
                "relative_path": "special_tokens_map.json",
                "sha256": "06e405a36dfe4b9604f484f6a1e619af1a7f7d09e34a8555eb0b77b66318067f",
                "size_bytes": 280,
            },
            {
                "relative_path": "tokenizer.json",
                "sha256": "b60b6b43406a48bf3638526314f3d232d97058bc93472ff2de930d43686fa441",
                "size_bytes": 17_082_913,
            },
            {
                "relative_path": "tokenizer_config.json",
                "sha256": "efb5c0d09722e5fe59a462cd2a9976ee216d55b037597d997cd3fe833216da15",
                "size_bytes": 418,
            },
        ],
        "model_name": TEXT_EMBEDDING_MODEL,
        "model_repository": FASTEMBED_REPOSITORY,
        "model_revision": MODEL_REVISIONS[FASTEMBED_REPOSITORY],
        "runtime_version": FASTEMBED_RUNTIME_VERSION,
    }


def test_fastembed_snapshot_materializes_exact_pinned_huggingface_blob_links(
    tmp_path: Path,
) -> None:
    source, _repository, _blobs = _write_huggingface_fastembed_snapshot(tmp_path)
    (source / "README.md").write_bytes(b"ignored extra")
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)

    snapshot = materialize_fastembed_snapshot(source, scratch, _fastembed_manifest())

    assert sorted(
        str(item.relative_to(snapshot.path))
        for item in snapshot.path.rglob("*")
        if item.is_file()
    ) == sorted(_MODEL_FILES)
    assert all(not item.is_symlink() for item in snapshot.path.rglob("*"))
    assert (snapshot.path / "README.md").exists() is False


def test_fastembed_huggingface_snapshot_materializes_from_retained_capability(
    tmp_path: Path,
) -> None:
    source, _repository, _blobs = _write_huggingface_fastembed_snapshot(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    source_fd = os.open(source, os.O_RDONLY | os.O_DIRECTORY)
    try:
        capability = snapshots_module.RetainedDirectory.retain(source, source_fd)
    finally:
        os.close(source_fd)

    try:
        snapshot = materialize_fastembed_snapshot(
            capability,
            scratch,
            _fastembed_manifest(),
        )

        assert sorted(
            str(item.relative_to(snapshot.path))
            for item in snapshot.path.rglob("*")
            if item.is_file()
        ) == sorted(_MODEL_FILES)
        assert all(not item.is_symlink() for item in snapshot.path.rglob("*"))
    finally:
        capability.close()


def test_fastembed_huggingface_snapshot_materializes_from_retained_child(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    source, _repository, _blobs = _write_huggingface_fastembed_snapshot(cache_root)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    cache_fd = os.open(cache_root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        capability = snapshots_module.RetainedDirectory.retain(cache_root, cache_fd)
    finally:
        os.close(cache_fd)
    source_view = capability.child(source.relative_to(cache_root).as_posix())
    cache_root.rename(tmp_path / "moved-cache")
    _write_huggingface_fastembed_snapshot(cache_root)

    try:
        snapshot = materialize_fastembed_snapshot(
            source_view,
            scratch,
            _fastembed_manifest(),
        )

        assert (snapshot.path / "config.json").read_bytes() == _MODEL_FILES[
            "config.json"
        ]
        assert all(not item.is_symlink() for item in snapshot.path.rglob("*"))
    finally:
        capability.close()


def test_fastembed_retained_capability_rejects_missing_logical_repository(
    tmp_path: Path,
) -> None:
    source, repository, _blobs = _write_huggingface_fastembed_snapshot(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    source_fd = os.open(source, os.O_RDONLY | os.O_DIRECTORY)
    try:
        capability = snapshots_module.RetainedDirectory.retain(source, source_fd)
    finally:
        os.close(source_fd)
    repository.rename(tmp_path / "moved-repository")

    try:
        with pytest.raises(SnapshotError):
            materialize_fastembed_snapshot(
                capability,
                scratch,
                _fastembed_manifest(),
            )

        assert list(scratch.iterdir()) == []
    finally:
        capability.close()


def test_fastembed_retained_capability_rejects_replaced_repository_binding(
    tmp_path: Path,
) -> None:
    source, repository, _blobs = _write_huggingface_fastembed_snapshot(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    source_fd = os.open(source, os.O_RDONLY | os.O_DIRECTORY)
    try:
        capability = snapshots_module.RetainedDirectory.retain(source, source_fd)
    finally:
        os.close(source_fd)
    repository.rename(tmp_path / "moved-repository")
    _write_huggingface_fastembed_snapshot(tmp_path)

    try:
        with pytest.raises(
            SnapshotError,
            match="FastEmbed snapshot is not the pinned repository revision",
        ):
            materialize_fastembed_snapshot(
                capability,
                scratch,
                _fastembed_manifest(),
            )

        assert list(scratch.iterdir()) == []
    finally:
        capability.close()


def test_fastembed_huggingface_snapshot_accepts_a_canonical_sha1_blob_name(
    tmp_path: Path,
) -> None:
    source, repository, blobs = _write_huggingface_fastembed_snapshot(tmp_path)
    target = source / "config.json"
    sha1_blob = repository / "blobs" / ("a" * 40)
    blobs["config.json"].rename(sha1_blob)
    target.unlink()
    target.symlink_to(os.path.relpath(sha1_blob, start=target.parent))
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)

    snapshot = materialize_fastembed_snapshot(source, scratch, _fastembed_manifest())

    assert (snapshot.path / "config.json").read_bytes() == _MODEL_FILES["config.json"]


@pytest.mark.parametrize(
    "unsafe_target",
    ["absolute", "escape", "traversal_alias", "non_digest"],
)
def test_fastembed_huggingface_snapshot_rejects_unsafe_blob_link_targets_and_rolls_back(
    tmp_path: Path,
    unsafe_target: str,
) -> None:
    source, repository, blobs = _write_huggingface_fastembed_snapshot(tmp_path)
    target = source / "config.json"
    blob = blobs["config.json"]
    target.unlink()
    if unsafe_target == "absolute":
        target.symlink_to(blob)
    elif unsafe_target == "escape":
        outside = tmp_path / "outside" / ("a" * 64)
        outside.parent.mkdir()
        outside.write_bytes(_MODEL_FILES["config.json"])
        target.symlink_to(os.path.relpath(outside, start=target.parent))
    elif unsafe_target == "traversal_alias":
        canonical = os.path.relpath(blob, start=target.parent)
        prefix, name = canonical.rsplit("/", 1)
        target.symlink_to(f"{prefix}/unused/../{name}")
    else:
        invalid_blob = repository / "blobs" / "not-a-content-address"
        invalid_blob.write_bytes(_MODEL_FILES["config.json"])
        target.symlink_to(os.path.relpath(invalid_blob, start=target.parent))
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)

    with pytest.raises(SnapshotError):
        materialize_fastembed_snapshot(source, scratch, _fastembed_manifest())

    assert list(scratch.iterdir()) == []


@pytest.mark.parametrize("unsafe_blob", ["hardlink", "fifo", "symlink", "missing", "hash"])
def test_fastembed_huggingface_snapshot_rejects_unsafe_or_wrong_blobs_and_rolls_back(
    tmp_path: Path,
    unsafe_blob: str,
) -> None:
    source, _repository, blobs = _write_huggingface_fastembed_snapshot(tmp_path)
    blob = blobs["config.json"]
    if unsafe_blob == "hardlink":
        os.link(blob, tmp_path / "second-link")
    elif unsafe_blob == "fifo":
        blob.unlink()
        os.mkfifo(blob)
    elif unsafe_blob == "symlink":
        content = blob.read_bytes()
        blob.unlink()
        outside = tmp_path / "outside-blob"
        outside.write_bytes(content)
        blob.symlink_to(outside)
    elif unsafe_blob == "missing":
        blob.unlink()
    else:
        content = blob.read_bytes()
        blob.write_bytes(bytes([content[0] ^ 1]) + content[1:])
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)

    with pytest.raises(SnapshotError):
        materialize_fastembed_snapshot(source, scratch, _fastembed_manifest())

    assert list(scratch.iterdir()) == []


def test_fastembed_huggingface_snapshot_detects_changed_link_target_and_rolls_back(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source, repository, _blobs = _write_huggingface_fastembed_snapshot(tmp_path)
    target = source / "config.json"
    replacement = repository / "blobs" / ("f" * 64)
    replacement.write_bytes(_MODEL_FILES["config.json"])
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    original_scan = snapshots_module._scan_allowlisted_files
    calls = 0

    def change_link_between_scans(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 2:
            target.unlink()
            target.symlink_to(os.path.relpath(replacement, start=target.parent))
        return original_scan(*args, **kwargs)

    monkeypatch.setattr(
        snapshots_module,
        "_scan_allowlisted_files",
        change_link_between_scans,
    )

    with pytest.raises(SnapshotError, match="changed"):
        materialize_fastembed_snapshot(source, scratch, _fastembed_manifest())

    assert list(scratch.iterdir()) == []


def test_fastembed_huggingface_snapshot_detects_link_replacement_during_blob_read(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source, repository, _blobs = _write_huggingface_fastembed_snapshot(tmp_path)
    target = source / "config.json"
    replacement = repository / "blobs" / ("e" * 64)
    replacement.write_bytes(_MODEL_FILES["config.json"])
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    original_readlink = os.readlink
    config_reads = 0

    def replace_before_second_readlink(path, *args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal config_reads
        if path == "config.json":
            config_reads += 1
            if config_reads == 2:
                target.unlink()
                target.symlink_to(os.path.relpath(replacement, start=target.parent))
        return original_readlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "readlink", replace_before_second_readlink)

    with pytest.raises(SnapshotError, match="changed"):
        materialize_fastembed_snapshot(source, scratch, _fastembed_manifest())

    assert list(scratch.iterdir()) == []


def test_fastembed_blob_links_require_the_exact_pinned_repository_layout(
    tmp_path: Path,
) -> None:
    source, repository, _blobs = _write_huggingface_fastembed_snapshot(tmp_path)
    wrong_repository = tmp_path / "models--attacker--lookalike"
    repository.rename(wrong_repository)
    source = (
        wrong_repository
        / "snapshots"
        / MODEL_REVISIONS[FASTEMBED_REPOSITORY]
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)

    with pytest.raises(SnapshotError):
        materialize_fastembed_snapshot(source, scratch, _fastembed_manifest())

    assert list(scratch.iterdir()) == []


@pytest.mark.parametrize("relationship", ["same", "scratch_inside", "source_inside"])
def test_fastembed_snapshot_rejects_overlapping_roots_before_any_mutation(
    tmp_path: Path,
    relationship: str,
) -> None:
    if relationship == "source_inside":
        scratch = tmp_path / "scratch"
        scratch.mkdir(mode=0o700)
        source = scratch / "source"
        _write_files(source, _MODEL_FILES)
    else:
        source = tmp_path / "source"
        _write_files(source, _MODEL_FILES)
        source.chmod(0o700)
        scratch = source if relationship == "same" else source / "scratch"
        if relationship == "scratch_inside":
            scratch.mkdir(mode=0o700)
    before = _tree_identity(tmp_path)

    with pytest.raises(SnapshotError, match="overlap"):
        materialize_fastembed_snapshot(source, scratch, _fastembed_manifest())

    assert _tree_identity(tmp_path) == before


@pytest.mark.parametrize("snapshot_kind", ["fastembed", "qdrant"])
def test_invalid_scratch_validation_never_leaks_source_descriptors(
    tmp_path: Path,
    snapshot_kind: str,
) -> None:
    source = tmp_path / "source"
    if snapshot_kind == "fastembed":
        _write_files(source, _MODEL_FILES)
    else:
        source.mkdir()
        (source / "state").write_bytes(b"qdrant")
    invalid_scratch = tmp_path / "invalid-scratch"
    invalid_scratch.mkdir(mode=0o755)
    before = len(os.listdir("/dev/fd"))

    for _ in range(32):
        with pytest.raises(SnapshotError, match="0700"):
            if snapshot_kind == "fastembed":
                materialize_fastembed_snapshot(
                    source,
                    invalid_scratch,
                    _fastembed_manifest(),
                )
            else:
                snapshot_qdrant_storage(source, invalid_scratch)

    assert len(os.listdir("/dev/fd")) == before


def test_published_fastembed_snapshot_is_reverified_before_embedding_creation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_files(source, _MODEL_FILES)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    snapshot = materialize_fastembed_snapshot(source, scratch, _fastembed_manifest())
    (snapshot.path / "config.json").write_bytes(b"tampered-after-publication")

    with pytest.raises(SnapshotError):
        snapshot.create_embedding()


def test_fastembed_snapshot_is_reverified_immediately_before_model_load(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    _write_files(source, _MODEL_FILES)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    snapshot = materialize_fastembed_snapshot(source, scratch, _fastembed_manifest())
    embedding = snapshot.create_embedding()
    (snapshot.path / "config.json").write_bytes(b"tampered-before-load")
    fastembed = types.ModuleType("fastembed")
    fastembed.TextEmbedding = lambda **_kwargs: pytest.fail(  # type: ignore[attr-defined]
        "FastEmbed must not receive a tampered snapshot"
    )
    monkeypatch.setitem(sys.modules, "fastembed", fastembed)
    monkeypatch.setattr(
        "videoscope.search.embeddings.metadata.version",
        lambda _package: FASTEMBED_RUNTIME_VERSION,
    )

    with pytest.raises(RuntimeError, match="strict semantic embedding is unavailable"):
        embedding.embed_query("query")

    assert "manifest" in str(embedding.last_error).lower()


def test_fastembed_snapshot_is_reverified_immediately_after_model_load(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    _write_files(source, _MODEL_FILES)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    snapshot = materialize_fastembed_snapshot(source, scratch, _fastembed_manifest())
    embedding = snapshot.create_embedding()

    class MutatingTextEmbedding:
        def __init__(self, **_kwargs: object) -> None:
            (snapshot.path / "config.json").write_bytes(b"mutated-during-model-load")

        def query_embed(self, _query: str):
            return iter([np.ones(TEXT_EMBEDDING_DIMENSIONS)])

    fastembed = types.ModuleType("fastembed")
    fastembed.TextEmbedding = MutatingTextEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", fastembed)
    monkeypatch.setattr(
        "videoscope.search.embeddings.metadata.version",
        lambda _package: FASTEMBED_RUNTIME_VERSION,
    )

    with pytest.raises(RuntimeError, match="strict semantic embedding is unavailable"):
        embedding.embed_query("query")

    assert "manifest" in str(embedding.last_error).lower()


@pytest.mark.parametrize("failure", ["missing", "size", "hash"])
def test_fastembed_snapshot_fails_closed_on_manifest_mismatch(
    tmp_path: Path,
    failure: str,
) -> None:
    source = tmp_path / "source"
    files = dict(_MODEL_FILES)
    _write_files(source, files)
    target = source / "config.json"
    if failure == "missing":
        target.unlink()
    elif failure == "size":
        target.write_bytes(b"different-size")
    else:
        original = target.read_bytes()
        target.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)

    with pytest.raises(SnapshotError):
        materialize_fastembed_snapshot(source, scratch, _fastembed_manifest())

    assert list(scratch.iterdir()) == []


@pytest.mark.parametrize("unsafe_kind", ["symlink", "hardlink", "fifo"])
def test_fastembed_snapshot_rejects_non_regular_or_multiply_linked_content(
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    source = tmp_path / "source"
    _write_files(source, _MODEL_FILES)
    target = source / "config.json"
    content = target.read_bytes()
    target.unlink()
    if unsafe_kind == "symlink":
        outside = tmp_path / "outside"
        outside.write_bytes(content)
        target.symlink_to(outside)
    elif unsafe_kind == "hardlink":
        outside = tmp_path / "outside"
        outside.write_bytes(content)
        os.link(outside, target)
    else:
        os.mkfifo(target)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)

    with pytest.raises(SnapshotError):
        materialize_fastembed_snapshot(source, scratch, _fastembed_manifest())

    assert list(scratch.iterdir()) == []


def test_fastembed_snapshot_detects_same_size_source_drift_and_rolls_back(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    _write_files(source, _MODEL_FILES)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    original_scan = snapshots_module._scan_allowlisted_files
    calls = 0

    def mutate_between_scans(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 2:
            target = source / "config.json"
            content = target.read_bytes()
            target.write_bytes(bytes([content[0] ^ 1]) + content[1:])
        return original_scan(*args, **kwargs)

    monkeypatch.setattr(
        snapshots_module,
        "_scan_allowlisted_files",
        mutate_between_scans,
    )

    with pytest.raises(SnapshotError, match="changed"):
        materialize_fastembed_snapshot(source, scratch, _fastembed_manifest())

    assert list(scratch.iterdir()) == []


def test_fastembed_manifest_is_exactly_pinned_and_requires_runtime_files() -> None:
    with pytest.raises(ValueError, match="pinned"):
        FastEmbedSnapshotManifest(
            model_name=TEXT_EMBEDDING_MODEL,
            model_repository=FASTEMBED_REPOSITORY,
            model_revision="f" * 40,
            runtime_version=FASTEMBED_RUNTIME_VERSION,
            algorithm_version=FASTEMBED_ALGORITHM_VERSION,
            dimensions=TEXT_EMBEDDING_DIMENSIONS,
            files=_fastembed_manifest().files,
        )
    with pytest.raises(ValueError, match="required FastEmbed file"):
        _fastembed_manifest(
            {
                key: value
                for key, value in _MODEL_FILES.items()
                if key != "onnx/model.onnx"
            }
        )


def test_fastembed_external_manifest_load_is_exact_and_never_follows_symlinks(
    tmp_path: Path,
) -> None:
    manifest = _fastembed_manifest()
    manifest_path = tmp_path / "fastembed-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.canonical_dict, sort_keys=True),
        encoding="utf-8",
    )

    assert load_fastembed_snapshot_manifest(manifest_path) == manifest

    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(manifest.canonical_dict), encoding="utf-8")
    manifest_path.unlink()
    manifest_path.symlink_to(outside)
    with pytest.raises(SnapshotError):
        load_fastembed_snapshot_manifest(manifest_path)


def test_qdrant_snapshot_rejects_special_entries_and_bounds(tmp_path: Path) -> None:
    source = tmp_path / "qdrant-source"
    source.mkdir()
    (source / "state").write_bytes(b"1234")
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)

    with pytest.raises(SnapshotError, match="byte limit"):
        snapshot_qdrant_storage(
            source,
            scratch,
            limits=QdrantSnapshotLimits(
                max_entries=10,
                max_depth=4,
                max_total_bytes=3,
                max_file_bytes=3,
            ),
        )

    os.mkfifo(source / "unsafe")
    with pytest.raises(SnapshotError, match="regular"):
        snapshot_qdrant_storage(source, scratch)
    assert list(scratch.iterdir()) == []


def test_qdrant_snapshot_detects_source_drift_and_removes_partial_copy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "qdrant-source"
    source.mkdir()
    state = source / "state"
    state.write_bytes(b"same-size")
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    original_scan = snapshots_module._scan_storage_tree
    calls = 0

    def mutate_between_scans(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 2:
            state.write_bytes(b"changed!!")
        return original_scan(*args, **kwargs)

    monkeypatch.setattr(snapshots_module, "_scan_storage_tree", mutate_between_scans)

    with pytest.raises(SnapshotError, match="changed"):
        snapshot_qdrant_storage(source, scratch)

    assert list(scratch.iterdir()) == []


def test_qdrant_snapshot_bounds_directory_consumption_before_sorting(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "qdrant-source"
    source.mkdir()
    for index in range(8):
        (source / f"state-{index}").write_bytes(b"x")
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    consumptions = _track_scandir_consumption(monkeypatch, source)

    with pytest.raises(SnapshotError, match="entry limit"):
        snapshot_qdrant_storage(
            source,
            scratch,
            limits=QdrantSnapshotLimits(
                max_entries=1,
                max_depth=4,
                max_total_bytes=16,
                max_file_bytes=16,
            ),
        )

    assert consumptions == [[2]]
    assert list(scratch.iterdir()) == []


def test_owned_staging_cleanup_bounds_directory_consumption_before_mutation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    staging = scratch / ".owned.partial"
    staging.mkdir(mode=0o700)
    for index in range(8):
        (staging / f"entry-{index}").write_bytes(b"x")
    scratch_fd = os.open(scratch, os.O_RDONLY | os.O_DIRECTORY)
    staging_fd = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
    state = snapshots_module._OwnedStaging(
        name=staging.name,
        max_entries=1,
    )
    try:
        state.bind(staging_fd)
    finally:
        os.close(staging_fd)
    consumptions = _track_scandir_consumption(monkeypatch, staging)

    try:
        with pytest.raises(SnapshotError, match="cleanup entry limit"):
            snapshots_module._remove_owned_staging(scratch_fd, state)
    finally:
        os.close(scratch_fd)

    assert consumptions == [[2]]
    assert sorted(item.name for item in staging.iterdir()) == [
        f"entry-{index}" for index in range(8)
    ]


@pytest.mark.parametrize("snapshot_kind", ["fastembed", "qdrant"])
def test_snapshot_publish_fsync_failure_removes_the_already_renamed_final(
    tmp_path: Path,
    monkeypatch,
    snapshot_kind: str,
) -> None:
    source = _snapshot_builder_source(tmp_path, snapshot_kind)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    scratch_metadata = scratch.stat()
    scratch_identity = (scratch_metadata.st_dev, scratch_metadata.st_ino)
    real_rename = os.rename
    real_fsync = os.fsync
    renamed = False
    failed = False

    def tracking_rename(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal renamed
        result = real_rename(*args, **kwargs)
        renamed = True
        return result

    def fail_parent_fsync_once(file_descriptor: int) -> None:
        nonlocal failed
        metadata = os.fstat(file_descriptor)
        if (
            renamed
            and not failed
            and (metadata.st_dev, metadata.st_ino) == scratch_identity
        ):
            failed = True
            raise OSError("injected parent fsync failure")
        real_fsync(file_descriptor)

    monkeypatch.setattr(os, "rename", tracking_rename)
    monkeypatch.setattr(os, "fsync", fail_parent_fsync_once)

    with pytest.raises(SnapshotError, match="atomically published"):
        _invoke_snapshot_builder(snapshot_kind, source, scratch)

    assert renamed is True
    assert failed is True
    assert list(scratch.iterdir()) == []


@pytest.mark.parametrize("snapshot_kind", ["fastembed", "qdrant"])
def test_snapshot_failure_retries_cleanup_and_preserves_the_primary_error(
    tmp_path: Path,
    monkeypatch,
    snapshot_kind: str,
) -> None:
    source = _snapshot_builder_source(tmp_path, snapshot_kind)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    primary = SnapshotError("primary snapshot failure")
    real_remove = snapshots_module._remove_owned_staging
    cleanup_calls = 0

    def fail_publish(*_args, **_kwargs) -> None:
        raise primary

    def fail_cleanup_once(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal cleanup_calls
        cleanup_calls += 1
        if cleanup_calls == 1:
            raise SnapshotError("injected cleanup failure")
        return real_remove(*args, **kwargs)

    monkeypatch.setattr(snapshots_module, "_publish_staging", fail_publish)
    monkeypatch.setattr(snapshots_module, "_remove_owned_staging", fail_cleanup_once)

    with pytest.raises(SnapshotError) as captured:
        _invoke_snapshot_builder(snapshot_kind, source, scratch)

    assert captured.value is primary
    assert cleanup_calls == 2
    assert list(scratch.iterdir()) == []


@pytest.mark.parametrize("snapshot_kind", ["fastembed", "qdrant"])
def test_snapshot_cleanup_retries_only_parent_sync_after_root_removal(
    tmp_path: Path,
    monkeypatch,
    snapshot_kind: str,
) -> None:
    source = _snapshot_builder_source(tmp_path, snapshot_kind)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    scratch_metadata = scratch.stat()
    scratch_identity = (scratch_metadata.st_dev, scratch_metadata.st_ino)
    primary = SnapshotError("primary snapshot failure")
    real_rmdir = os.rmdir
    real_fsync = os.fsync
    root_removals = 0
    root_removed = False
    failed = False

    def fail_publish(*_args, **_kwargs) -> None:
        raise primary

    def track_root_removal(path, *args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal root_removals, root_removed
        result = real_rmdir(path, *args, **kwargs)
        parent_fd = kwargs.get("dir_fd")
        if isinstance(parent_fd, int):
            metadata = os.fstat(parent_fd)
            if (metadata.st_dev, metadata.st_ino) == scratch_identity:
                root_removals += 1
                root_removed = True
        return result

    def fail_cleanup_parent_sync_once(file_descriptor: int) -> None:
        nonlocal failed
        metadata = os.fstat(file_descriptor)
        if (
            root_removed
            and not failed
            and (metadata.st_dev, metadata.st_ino) == scratch_identity
        ):
            failed = True
            raise OSError("injected cleanup parent fsync failure")
        real_fsync(file_descriptor)

    monkeypatch.setattr(snapshots_module, "_publish_staging", fail_publish)
    monkeypatch.setattr(os, "rmdir", track_root_removal)
    monkeypatch.setattr(os, "fsync", fail_cleanup_parent_sync_once)

    with pytest.raises(SnapshotError) as captured:
        _invoke_snapshot_builder(snapshot_kind, source, scratch)

    assert captured.value is primary
    assert root_removals == 1
    assert failed is True
    assert list(scratch.iterdir()) == []


@pytest.mark.parametrize("snapshot_kind", ["fastembed", "qdrant"])
def test_persistent_snapshot_cleanup_failure_is_observable_and_recoverable(
    tmp_path: Path,
    monkeypatch,
    snapshot_kind: str,
) -> None:
    source = _snapshot_builder_source(tmp_path, snapshot_kind)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    primary = SnapshotError("primary snapshot failure")
    real_remove = snapshots_module._remove_owned_staging
    cleanup_failures: list[SnapshotError] = []
    cleanup_calls = 0

    def fail_publish(*_args, **_kwargs) -> None:
        raise primary

    def fail_cleanup_until_explicit_recovery(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal cleanup_calls
        cleanup_calls += 1
        if cleanup_calls <= 3:
            error = SnapshotError(f"injected cleanup failure {cleanup_calls}")
            cleanup_failures.append(error)
            raise error
        return real_remove(*args, **kwargs)

    monkeypatch.setattr(snapshots_module, "_publish_staging", fail_publish)
    monkeypatch.setattr(
        snapshots_module,
        "_remove_owned_staging",
        fail_cleanup_until_explicit_recovery,
    )

    with pytest.raises(SnapshotError) as captured:
        _invoke_snapshot_builder(snapshot_kind, source, scratch)

    error = captured.value
    assert isinstance(error, snapshots_module.SnapshotCleanupError)
    assert error.primary_error is primary
    assert error.cleanup_error is cleanup_failures[-1]
    assert error.cleanup_pending is True
    assert os.fspath(tmp_path) not in str(error)
    assert "primary snapshot failure" not in str(error)
    assert "injected cleanup failure" not in str(error)
    assert any("partial" in item.name for item in scratch.iterdir())

    assert error.retry_cleanup() is True
    assert error.cleanup_pending is False
    assert error.retry_cleanup() is False
    assert list(scratch.iterdir()) == []


@pytest.mark.parametrize("snapshot_kind", ["fastembed", "qdrant"])
def test_retained_scratch_capability_survives_parent_replacement_end_to_end(
    tmp_path: Path,
    snapshot_kind: str,
) -> None:
    source = _snapshot_builder_source(tmp_path, snapshot_kind)
    capability, owned_scratch, replacement = (
        _retain_scratch_then_replace_its_logical_parent(tmp_path)
    )

    try:
        snapshot = _invoke_snapshot_builder(snapshot_kind, source, capability)

        assert snapshot.path.parent == capability.stable_path
        assert (owned_scratch / snapshot.path.name).is_dir()
        assert list(replacement.iterdir()) == []
        if snapshot_kind == "fastembed":
            embedding = snapshot.create_embedding()
            assert embedding.specific_model_path == snapshot.path
        else:
            verify_qdrant_storage_snapshot(snapshot)
        assert capability.close() is True
        assert capability.close() is False
    finally:
        capability.close()


@pytest.mark.parametrize("snapshot_kind", ["fastembed", "qdrant"])
def test_retained_source_child_uses_original_data_root_after_parent_replacement(
    tmp_path: Path,
    snapshot_kind: str,
) -> None:
    capability, source, owned_data, replacement_data = (
        _retain_data_root_then_replace_its_logical_parent(tmp_path, snapshot_kind)
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)

    try:
        source_fd = source.duplicate_descriptor()
        try:
            source_metadata = os.fstat(source_fd)
            owned_metadata = (owned_data / "source").stat()
            assert (source_metadata.st_dev, source_metadata.st_ino) == (
                owned_metadata.st_dev,
                owned_metadata.st_ino,
            )
        finally:
            os.close(source_fd)
        assert capability.read_regular_file(
            "glossary.json",
            max_bytes=1024,
        ) == b'{"source":"original"}'

        snapshot = _invoke_snapshot_builder(snapshot_kind, source, scratch)

        if snapshot_kind == "fastembed":
            assert (snapshot.path / "config.json").read_bytes() == _MODEL_FILES[
                "config.json"
            ]
        else:
            assert (snapshot.path / "state").read_bytes() == b"original qdrant"
        assert (replacement_data / "glossary.json").read_bytes() == (
            b'{"source":"replacement"}'
        )
    finally:
        capability.close()


def test_retained_directory_optional_read_distinguishes_missing_from_unsafe(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir(mode=0o700)
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        capability = snapshots_module.RetainedDirectory.retain(root, descriptor)
    finally:
        os.close(descriptor)
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"outside")
    (root / "unsafe.json").symlink_to(outside)

    try:
        assert capability.read_optional_regular_file(
            "missing.json",
            max_bytes=1024,
        ) is None
        with pytest.raises(SnapshotError):
            capability.read_optional_regular_file(
                "unsafe.json",
                max_bytes=1024,
            )
    finally:
        capability.close()


@pytest.mark.parametrize("relationship", ["same", "scratch_inside", "source_inside"])
def test_qdrant_snapshot_rejects_overlapping_roots_before_any_mutation(
    tmp_path: Path,
    relationship: str,
) -> None:
    if relationship == "source_inside":
        scratch = tmp_path / "scratch"
        scratch.mkdir(mode=0o700)
        source = scratch / "source"
        source.mkdir()
    else:
        source = tmp_path / "source"
        source.mkdir(mode=0o700)
        scratch = source if relationship == "same" else source / "scratch"
        if relationship == "scratch_inside":
            scratch.mkdir(mode=0o700)
    (source / "state").write_bytes(b"qdrant")
    before = _tree_identity(tmp_path)

    with pytest.raises(SnapshotError, match="overlap"):
        snapshot_qdrant_storage(source, scratch)

    assert _tree_identity(tmp_path) == before


class _StrictFixtureEmbedding(HashEmbedding):
    strict_no_fallback = True
    model_name = TEXT_EMBEDDING_MODEL
    model_repository = FASTEMBED_REPOSITORY
    model_revision = MODEL_REVISIONS[FASTEMBED_REPOSITORY]
    expected_runtime_version = FASTEMBED_RUNTIME_VERSION
    algorithm_version = FASTEMBED_ALGORITHM_VERSION
    model_content_sha256 = "c" * 64

    def __init__(self, dimensions: int) -> None:
        super().__init__(dimensions)
        self.model_current = True

    def verify_benchmark_model_current(self) -> bool:
        return self.model_current

    @property
    def identity(self) -> str:
        return (
            f"strict-fastembed:{self.model_name}:{self.model_repository}@"
            f"{self.model_revision}:{self.model_content_sha256}:{self.dimensions}"
        )

    @property
    def benchmark_identity(self) -> dict[str, object]:
        return {
            "embedding_identity": self.identity,
            "model_name": self.model_name,
            "model_repository": self.model_repository,
            "model_revision": self.model_revision,
            "runtime_version": self.expected_runtime_version,
            "algorithm_version": self.algorithm_version,
            "dimensions": self.dimensions,
            "model_content_sha256": self.model_content_sha256,
        }


def _generation_plan(index: QdrantVectorIndex) -> TextVectorBuildPlan:
    now = datetime.now(UTC)
    text = "made basket"
    point = TextVectorPointSource(
        video_id="video-1",
        segment_id="segment-1",
        modality="speech",
        text=text,
        segment_generation_id="speech-generation-1",
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )
    inputs = tuple(
        TextVectorGenerationInput(
            stage_kind=kind,
            specification_hash=marker * 64,
            segment_generation_id="speech-generation-1" if kind is StageKind.SPEECH else None,
            segment_run_id="speech-run-1" if kind is StageKind.SPEECH else None,
            source_sha256="a" * 64 if kind is StageKind.SPEECH else None,
            segment_count=1 if kind is StageKind.SPEECH else 0,
            content_manifest_sha256=marker * 64,
        )
        for kind, marker in zip(
            (StageKind.SPEECH, StageKind.OCR, StageKind.OBJECTS),
            ("b", "c", "d"),
            strict=True,
        )
    )
    return TextVectorBuildPlan(
        generation_id="1" * 32,
        run_id="run-1",
        video_id="video-1",
        stage_specification_hash="e" * 64,
        source_sha256="a" * 64,
        index_specification=index.index_specification,
        expected_previous_generation_id=None,
        inputs=inputs,
        points=(point,),
        input_manifest_sha256="f" * 64,
        point_manifest_sha256="9" * 64,
        reserved_at=now.isoformat(),
        lease_expires_at=(now + timedelta(minutes=5)).isoformat(),
    )


def _binding(plan: TextVectorBuildPlan, receipt) -> TextVectorSearchBinding:  # type: ignore[no-untyped-def]
    return TextVectorSearchBinding(
        generation=TextVectorGeneration(
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
        ),
        index_specification=plan.index_specification,
        inputs=plan.inputs,
        points=plan.points,
    )


def test_existing_qdrant_snapshot_is_queryable_and_attested_without_fallback(
    tmp_path: Path,
) -> None:
    embedding = _StrictFixtureEmbedding(16)
    source = tmp_path / "qdrant-source"
    writer = QdrantVectorIndex(source, embedding=embedding)
    plan = _generation_plan(writer)
    receipt = writer.build_generation(plan)
    binding = _binding(plan, receipt)
    writer.close()
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    snapshot = snapshot_qdrant_storage(source, scratch)

    index = QdrantVectorIndex.open_existing_snapshot(snapshot, embedding=embedding)
    try:
        hits = index.search_generations(
            "made basket",
            bindings=[binding],
            modalities={"speech"},
            limit=5,
        )
        assert [hit.segment_id for hit in hits] == ["segment-1"]
        # Qdrant's local cosine storage slightly renormalizes persisted float32
        # vectors on reopen, so the legacy byte-exact digest is not restart-stable.
        assert index.validate_generation(binding, exhaustive=True) is False
        assert index.validate_generation_for_benchmark_snapshot(binding) is True
        assert index.benchmark_attestation == {
            "schema_version": 1,
            "provider": "qdrant",
            "strict_no_fallback": True,
            "embedding": {
                **embedding.benchmark_identity,
                "embedding_identity": index.index_specification.embedding_identity,
            },
            "index": {
                "index_specification_hash": index.index_specification.specification_hash,
                "collection_name": index.collection_name,
                "snapshot_sha256": snapshot.snapshot_sha256,
            },
        }
        assert index.verify_benchmark_snapshot_current() is True
        with pytest.raises(RuntimeError, match="query-only"):
            index.build_generation(plan)
    finally:
        index.close()
    assert index.verify_benchmark_snapshot_current() is True


def test_attested_qdrant_binding_search_avoids_revalidating_manifest_and_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedding = _StrictFixtureEmbedding(16)
    source = tmp_path / "qdrant-source"
    writer = QdrantVectorIndex(source, embedding=embedding)
    plan = _generation_plan(writer)
    receipt = writer.build_generation(plan)
    binding = _binding(plan, receipt)
    writer.close()
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    snapshot = snapshot_qdrant_storage(source, scratch)
    index = QdrantVectorIndex.open_existing_snapshot(snapshot, embedding=embedding)
    try:
        assert index.validate_generation_for_benchmark_snapshot(binding) is True
        client = index._get_client()
        native_retrieve = client.retrieve
        native_count = client.count
        calls = {"retrieve": 0, "count": 0}

        def recording_retrieve(*args, **kwargs):  # type: ignore[no-untyped-def]
            calls["retrieve"] += 1
            return native_retrieve(*args, **kwargs)

        def recording_count(*args, **kwargs):  # type: ignore[no-untyped-def]
            calls["count"] += 1
            return native_count(*args, **kwargs)

        monkeypatch.setattr(client, "retrieve", recording_retrieve)
        monkeypatch.setattr(client, "count", recording_count)

        class ExplodingPointTuple(tuple):
            def __iter__(self):  # type: ignore[no-untyped-def]
                raise AssertionError("timed search consumed all binding points")

        object.__setattr__(binding, "points", ExplodingPointTuple(binding.points))

        assert index.search_generations(
            "made basket",
            bindings=[binding],
            modalities={"speech"},
            limit=5,
        )
        assert calls == {"retrieve": 0, "count": 0}
        assert index.verify_benchmark_snapshot_current() is True
    finally:
        index.close()


def test_129th_benchmark_binding_is_rejected_without_evicting_timed_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedding = _StrictFixtureEmbedding(16)
    source = tmp_path / "qdrant-source"
    writer = QdrantVectorIndex(source, embedding=embedding)
    plan = _generation_plan(writer)
    receipt = writer.build_generation(plan)
    binding = _binding(plan, receipt)
    writer.close()
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    snapshot = snapshot_qdrant_storage(source, scratch)
    index = QdrantVectorIndex.open_existing_snapshot(snapshot, embedding=embedding)
    try:
        assert index.validate_generation_for_benchmark_snapshot(binding) is True
        first_key = id(binding)
        first_entry = index._validated_benchmark_binding_cache[first_key]
        for ordinal in range(1, 128):
            synthetic_key = -ordinal
            index._validated_benchmark_binding_cache[synthetic_key] = first_entry
        assert len(index._validated_benchmark_binding_cache) == 128
        overflow_binding = replace(
            binding,
            generation=replace(binding.generation, generation_id="f" * 32),
        )
        client = index._get_client()
        native_retrieve = client.retrieve
        native_count = client.count
        calls = {"retrieve": 0, "count": 0}

        def recording_retrieve(*args, **kwargs):  # type: ignore[no-untyped-def]
            calls["retrieve"] += 1
            return native_retrieve(*args, **kwargs)

        def recording_count(*args, **kwargs):  # type: ignore[no-untyped-def]
            calls["count"] += 1
            return native_count(*args, **kwargs)

        monkeypatch.setattr(client, "retrieve", recording_retrieve)
        monkeypatch.setattr(client, "count", recording_count)

        assert index.validate_generation_for_benchmark_snapshot(overflow_binding) is False
        assert first_key in index._validated_benchmark_binding_cache

        class ExplodingPointTuple(tuple):
            def __iter__(self):  # type: ignore[no-untyped-def]
                raise AssertionError("timed search consumed all binding points")

        object.__setattr__(binding, "points", ExplodingPointTuple(binding.points))
        assert index.search_generations(
            "made basket",
            bindings=[binding],
            modalities={"speech"},
            limit=5,
        )
        assert calls == {"retrieve": 0, "count": 0}
    finally:
        index.close()


def test_closed_benchmark_binding_leases_do_not_exhaust_later_sessions(
    tmp_path: Path,
) -> None:
    embedding = _StrictFixtureEmbedding(16)
    source = tmp_path / "qdrant-source"
    writer = QdrantVectorIndex(source, embedding=embedding)
    plan = _generation_plan(writer)
    receipt = writer.build_generation(plan)
    binding = _binding(plan, receipt)
    writer.close()
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    snapshot = snapshot_qdrant_storage(source, scratch)
    index = QdrantVectorIndex.open_existing_snapshot(snapshot, embedding=embedding)
    retained_bindings = []
    try:
        for _ordinal in range(129):
            session_binding = replace(binding)
            retained_bindings.append(session_binding)
            assert index.validate_generation_for_benchmark_snapshot(session_binding) is True
            assert len(index._validated_benchmark_binding_cache) == 1

            index.release_benchmark_snapshot_bindings((session_binding,))
            index.release_benchmark_snapshot_bindings((session_binding,))

            assert not index._validated_benchmark_binding_cache
    finally:
        index.close()


def test_retained_qdrant_snapshot_opens_after_scratch_parent_replacement(
    tmp_path: Path,
) -> None:
    embedding = _StrictFixtureEmbedding(16)
    source = tmp_path / "qdrant-source"
    writer = QdrantVectorIndex(source, embedding=embedding)
    plan = _generation_plan(writer)
    receipt = writer.build_generation(plan)
    binding = _binding(plan, receipt)
    writer.close()
    capability, _owned_scratch, replacement = (
        _retain_scratch_then_replace_its_logical_parent(tmp_path)
    )
    index = None
    try:
        snapshot = snapshot_qdrant_storage(source, capability)
        index = QdrantVectorIndex.open_existing_snapshot(
            snapshot,
            embedding=embedding,
        )

        assert [
            hit.segment_id
            for hit in index.search_generations(
                "made basket",
                bindings=[binding],
                modalities={"speech"},
                limit=5,
            )
        ] == ["segment-1"]
        assert list(replacement.iterdir()) == []
    finally:
        if index is not None:
            index.close()
        capability.close()


@pytest.mark.parametrize("corruption", ["payload", "id", "nan", "dimension"])
def test_benchmark_snapshot_generation_validation_rejects_structural_corruption(
    tmp_path: Path,
    corruption: str,
) -> None:
    embedding = _StrictFixtureEmbedding(16)
    source = tmp_path / "qdrant-source"
    writer = QdrantVectorIndex(source, embedding=embedding)
    plan = _generation_plan(writer)
    receipt = writer.build_generation(plan)
    binding = _binding(plan, receipt)
    writer.close()
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    snapshot = snapshot_qdrant_storage(source, scratch)
    index = QdrantVectorIndex.open_existing_snapshot(snapshot, embedding=embedding)
    real_client = index._client
    assert real_client is not None
    point_id = index.point_id(binding.generation_id, binding.points[0].segment_id)
    real_record = real_client.retrieve(
        collection_name=index.collection_name,
        ids=[point_id],
        with_payload=True,
        with_vectors=True,
    )[0]
    payload = dict(real_record.payload)
    record_id = point_id
    vector = list(real_record.vector)
    if corruption == "payload":
        payload["text_sha256"] = "0" * 64
    elif corruption == "id":
        record_id = "00000000-0000-0000-0000-000000000000"
    elif corruption == "nan":
        vector[0] = float("nan")
    else:
        vector = vector[:-1]
    corrupted_record = SimpleNamespace(
        id=record_id,
        payload=payload,
        vector=vector,
    )

    class CorruptingClient:
        def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
            return getattr(real_client, name)

        def retrieve(self, **kwargs):  # type: ignore[no-untyped-def]
            if kwargs.get("ids") == [point_id] and kwargs.get("with_vectors") is True:
                return [corrupted_record]
            return real_client.retrieve(**kwargs)

    index._client = CorruptingClient()
    try:
        assert index.validate_generation_for_benchmark_snapshot(binding) is False
    finally:
        index._client = real_client
        index.close()


def test_existing_qdrant_snapshot_open_never_creates_absent_path(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    snapshot = QdrantStorageSnapshot(
        path=missing,
        snapshot_sha256="a" * 64,
        entry_count=0,
        total_bytes=0,
        source_device=1,
        source_inode=1,
    )

    with pytest.raises(SnapshotError):
        QdrantVectorIndex.open_existing_snapshot(
            snapshot,
            embedding=_StrictFixtureEmbedding(16),
        )

    assert missing.exists() is False


def test_existing_qdrant_snapshot_rejects_missing_collection_without_creating_it(
    tmp_path: Path,
) -> None:
    source = tmp_path / "empty-source"
    source.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    snapshot = snapshot_qdrant_storage(source, scratch)

    with pytest.raises(RuntimeError, match="collection.*missing"):
        QdrantVectorIndex.open_existing_snapshot(
            snapshot,
            embedding=_StrictFixtureEmbedding(16),
        )

    from qdrant_client import QdrantClient

    client = QdrantClient(path=str(snapshot.path))
    try:
        assert client.collection_exists(
            QdrantVectorIndex(
                snapshot.path,
                embedding=_StrictFixtureEmbedding(16),
            ).collection_name
        ) is False
    finally:
        client.close()


def test_existing_qdrant_snapshot_retries_client_close_and_preserves_open_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "state").write_bytes(b"fixture")
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    snapshot = snapshot_qdrant_storage(source, scratch)

    class FailingClient:
        def __init__(self) -> None:
            self.close_calls = 0

        def collection_exists(self, _name: str) -> bool:
            raise RuntimeError("collection probe failed")

        def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise ValueError("transient close failure")

    client = FailingClient()

    def open_client(self, *, create_path: bool):  # type: ignore[no-untyped-def]
        assert create_path is False
        self._client = client
        return client

    monkeypatch.setattr(QdrantVectorIndex, "_get_storage_client", open_client)

    with pytest.raises(RuntimeError, match="collection is corrupt"):
        QdrantVectorIndex.open_existing_snapshot(
            snapshot,
            embedding=_StrictFixtureEmbedding(16),
        )

    assert client.close_calls == 2


def test_existing_qdrant_snapshot_persistent_close_failure_retains_retry_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "state").write_bytes(b"fixture")
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    snapshot = snapshot_qdrant_storage(source, scratch)
    blocked = True

    class FailingClient:
        def __init__(self) -> None:
            self.close_calls = 0

        def collection_exists(self, _name: str) -> bool:
            raise RuntimeError("collection probe failed")

        def close(self) -> None:
            self.close_calls += 1
            if blocked:
                raise ValueError("persistent close failure")

    client = FailingClient()

    def open_client(self, *, create_path: bool):  # type: ignore[no-untyped-def]
        assert create_path is False
        self._client = client
        return client

    monkeypatch.setattr(QdrantVectorIndex, "_get_storage_client", open_client)

    with pytest.raises(QdrantSnapshotCleanupError) as captured:
        QdrantVectorIndex.open_existing_snapshot(
            snapshot,
            embedding=_StrictFixtureEmbedding(16),
        )

    error = captured.value
    assert error.cleanup_pending is True
    assert str(tmp_path) not in str(error)
    assert client.close_calls == 3

    blocked = False
    error.retry_cleanup()
    assert error.cleanup_pending is False
    assert client.close_calls == 4


def test_existing_qdrant_snapshot_rejects_corrupt_collection_storage(
    tmp_path: Path,
) -> None:
    embedding = _StrictFixtureEmbedding(16)
    source = tmp_path / "source"
    writer = QdrantVectorIndex(source, embedding=embedding)
    writer._get_client()
    writer.close()
    (source / "meta.json").write_bytes(b"not-json")
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    snapshot = snapshot_qdrant_storage(source, scratch)

    with pytest.raises(RuntimeError):
        QdrantVectorIndex.open_existing_snapshot(snapshot, embedding=embedding)


def test_qdrant_snapshot_verification_detects_post_publication_tampering(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "state").write_bytes(b"original")
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    snapshot = snapshot_qdrant_storage(source, scratch)
    (snapshot.path / "state").write_bytes(b"tampered")

    with pytest.raises(SnapshotError, match="digest"):
        verify_qdrant_storage_snapshot(snapshot)


def test_qdrant_snapshot_verification_rejects_relaxed_private_permissions(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "state").write_bytes(b"original")
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    snapshot = snapshot_qdrant_storage(source, scratch)
    (snapshot.path / "state").chmod(0o644)

    with pytest.raises(SnapshotError, match="permissions"):
        verify_qdrant_storage_snapshot(snapshot)


def test_qdrant_index_can_reverify_private_snapshot_after_close(
    tmp_path: Path,
) -> None:
    embedding = _StrictFixtureEmbedding(16)
    source = tmp_path / "source"
    writer = QdrantVectorIndex(source, embedding=embedding)
    writer._get_client()
    writer.close()
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    snapshot = snapshot_qdrant_storage(source, scratch)
    index = QdrantVectorIndex.open_existing_snapshot(snapshot, embedding=embedding)

    index.close()
    assert index.verify_benchmark_snapshot_current() is True
    (snapshot.path / "unexpected").write_bytes(b"tampered")
    assert index.verify_benchmark_snapshot_current() is False


def test_qdrant_final_snapshot_verification_requires_current_model_bytes(
    tmp_path: Path,
) -> None:
    embedding = _StrictFixtureEmbedding(16)
    source = tmp_path / "source"
    writer = QdrantVectorIndex(source, embedding=embedding)
    writer._get_client()
    writer.close()
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    snapshot = snapshot_qdrant_storage(source, scratch)
    index = QdrantVectorIndex.open_existing_snapshot(snapshot, embedding=embedding)

    assert index.verify_benchmark_snapshot_current() is True
    embedding.model_current = False
    assert index.verify_benchmark_snapshot_current() is False


def test_final_qdrant_digest_detects_tree_mutation_during_query(
    tmp_path: Path,
) -> None:
    embedding = _StrictFixtureEmbedding(16)
    source = tmp_path / "source"
    writer = QdrantVectorIndex(source, embedding=embedding)
    plan = _generation_plan(writer)
    receipt = writer.build_generation(plan)
    binding = _binding(plan, receipt)
    writer.close()
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    snapshot = snapshot_qdrant_storage(source, scratch)
    index = QdrantVectorIndex.open_existing_snapshot(snapshot, embedding=embedding)
    original_embed = embedding.embed

    def mutate_tree_during_embedding(texts: list[str]) -> list[np.ndarray]:
        (snapshot.path / "injected-during-query").write_bytes(b"drift")
        return original_embed(texts)

    embedding.embed = mutate_tree_during_embedding  # type: ignore[method-assign]
    try:
        assert index.search_generations(
            "basket",
            bindings=[binding],
            modalities={"speech"},
            limit=1,
        )
    finally:
        index.close()
    assert index.verify_benchmark_snapshot_current() is False
