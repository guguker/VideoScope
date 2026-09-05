from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import Field

from videoscope.benchmark import environment as environment_module
from videoscope.benchmark.environment import (
    BenchmarkEnvironmentCleanupError,
    BenchmarkEnvironmentError,
    open_product_benchmark_environment,
)
from videoscope.benchmark.product_runtime import ProductSnapshotCleanupError
from videoscope.benchmark.profiles import FROZEN_PROFILES
from videoscope.config import AppSettings
from videoscope.media.ffmpeg import FFmpeg
from videoscope.providers.whisper import snapshot_whisper_prompt
from videoscope.repository import Repository
from videoscope.runtime import build_runtime


_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_FASTEMBED_REPOSITORY_DIRECTORY = (
    "models--xenova--paraphrase-multilingual-mpnet-base-v2"
)
_FASTEMBED_REVISION = "e5d116277351513fd260955ece953ecddde7046e"


def _fastembed_snapshot_root(cache_root: Path) -> Path:
    return (
        cache_root
        / _FASTEMBED_REPOSITORY_DIRECTORY
        / "snapshots"
        / _FASTEMBED_REVISION
    )


class _ExternalModelsSettings(AppSettings):
    immutable_models_dir: Path = Field(exclude=True)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):  # type: ignore[no-untyped-def]
        del cls, settings_cls, env_settings, dotenv_settings, file_secret_settings
        return (init_settings,)

    @property
    def models_dir(self) -> Path:
        return self.immutable_models_dir


_ExternalModelsSettings.model_rebuild(_types_namespace={"Path": Path})


def _private_directory(path: Path) -> Path:
    path.mkdir(mode=0o700)
    path.chmod(0o700)
    return path


def _tree(root: Path) -> dict[str, tuple[int, bytes | None]]:
    return {
        path.relative_to(root).as_posix() or ".": (
            path.stat(follow_symlinks=False).st_mode,
            path.read_bytes() if path.is_file() else None,
        )
        for path in sorted((root, *root.rglob("*")))
    }


class _Repository:
    is_read_only = True

    def find_assets_by_sha256(self, _digest: str) -> tuple[object, ...]:
        return ()


class _ProductSnapshot:
    def __init__(self, data_root: Path, media_root: Path, events: list[str]) -> None:
        self.repository = _Repository()
        self.data_root = data_root
        self.media_root = media_root
        self.identity = SimpleNamespace(snapshot_sha256=_SHA_A)
        self._events = events
        self.close_calls = 0
        self.close_failures = 0
        self.is_closed = False
        self.session_root: Path | None = None
        self._data_descriptor: int | None = None
        self._media_descriptor: int | None = None

    def bind_roots(self) -> None:
        if self._data_descriptor is None:
            self._data_descriptor = os.open(self.data_root, os.O_RDONLY)
        if self._media_descriptor is None:
            self._media_descriptor = os.open(self.media_root, os.O_RDONLY)

    def duplicate_data_root_descriptor(self) -> int:
        assert self._data_descriptor is not None
        return os.dup(self._data_descriptor)

    def duplicate_media_root_descriptor(self) -> int:
        assert self._media_descriptor is not None
        return os.dup(self._media_descriptor)

    def close(self) -> None:
        self.close_calls += 1
        self._events.append("product.close")
        if self.close_calls <= self.close_failures:
            raise RuntimeError("product close failed")
        if self.session_root is not None:
            assert not self.session_root.exists()
        if self._media_descriptor is not None:
            os.close(self._media_descriptor)
            self._media_descriptor = None
        if self._data_descriptor is not None:
            os.close(self._data_descriptor)
            self._data_descriptor = None
        self.is_closed = True


class _FastEmbedding:
    strict_no_fallback = True
    identity = "fastembed@fixture"
    dimensions = 768

    def __init__(self, events: list[str]) -> None:
        self._events = events

    def ensure_ready(self) -> bool:
        self._events.append("fastembed.ensure_ready")
        return True

    @property
    def benchmark_identity(self) -> dict[str, object]:
        return {
            "embedding_identity": self.identity,
            "model_name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            "model_repository": "xenova/paraphrase-multilingual-mpnet-base-v2",
            "model_revision": "e5d116277351513fd260955ece953ecddde7046e",
            "runtime_version": "fixture-runtime",
            "algorithm_version": "fixture-algorithm",
            "dimensions": self.dimensions,
            "model_content_sha256": _SHA_B,
        }


class _FastSnapshot:
    def __init__(self, path: Path, events: list[str]) -> None:
        self.path = path
        self.identity = SimpleNamespace(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            model_repository="xenova/paraphrase-multilingual-mpnet-base-v2",
            model_revision="e5d116277351513fd260955ece953ecddde7046e",
            runtime_version="fixture-runtime",
            algorithm_version="fixture-algorithm",
            dimensions=768,
            model_content_sha256=_SHA_B,
        )
        self._events = events

    def create_embedding(self) -> _FastEmbedding:
        self._events.append("fastembed.create_embedding")
        return _FastEmbedding(self._events)


@dataclass(frozen=True)
class _QdrantSnapshot:
    path: Path
    snapshot_sha256: str = _SHA_C


class _Index:
    def __init__(
        self,
        snapshot: _QdrantSnapshot,
        embedding: _FastEmbedding,
        events: list[str],
    ) -> None:
        self.snapshot = snapshot
        self.embedding = embedding
        self._events = events
        self.close_failures = 0
        self.close_calls = 0
        self.index_specification = SimpleNamespace(
            canonical_json='{"fixture":true}',
            specification_hash="d" * 64,
        )

    @property
    def benchmark_attestation(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "provider": "qdrant",
            "strict_no_fallback": True,
            "embedding": self.embedding.benchmark_identity,
            "index": {
                "index_specification_hash": "d" * 64,
                "collection_name": "fixture",
                "snapshot_sha256": self.snapshot.snapshot_sha256,
            },
        }

    def close(self) -> None:
        self.close_calls += 1
        self._events.append("qdrant.close")
        if self.close_calls <= self.close_failures:
            raise RuntimeError("qdrant close failed")


class _Adapter:
    def __init__(
        self,
        search: object,
        events: list[str],
        environment_identity: object,
    ) -> None:
        self.search = search
        self.environment_identity = environment_identity
        self._events = events
        self.close_failures = 0
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        self._events.append("adapter.close")
        if self.close_calls <= self.close_failures:
            raise RuntimeError("adapter close failed")


class _SearchService:
    def __init__(self, repository: object, vector_index: object, **kwargs: object) -> None:
        self.repository = repository
        self.vector_index = vector_index
        self.kwargs = kwargs

    def open_pinned_evaluation(self) -> None:
        return None


def _specifications() -> object:
    stages = {
        name: SimpleNamespace(specification_hash=character * 64)
        for name, character in zip(
            ("scenes", "speech", "ocr", "objects", "text_vectors"),
            "ef012",
            strict=True,
        )
    }
    return SimpleNamespace(**stages)


@pytest.fixture
def environment_fakes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> SimpleNamespace:
    events: list[str] = []
    data_dir = tmp_path / "product" / "data"
    media_root = data_dir / "media"
    fastembed_cache_root = data_dir / "models" / "fastembed"
    fastembed_root = _fastembed_snapshot_root(fastembed_cache_root)
    qdrant_root = data_dir / "qdrant" / "text"
    media_root.mkdir(parents=True)
    fastembed_root.mkdir(parents=True)
    qdrant_root.mkdir(parents=True)
    (data_dir / "videoscope.sqlite3").write_bytes(b"product database sentinel")
    (fastembed_root / "model.sentinel").write_bytes(b"model")
    (qdrant_root / "index.sentinel").write_bytes(b"index")
    (data_dir / "search-glossary.json").write_text(
        json.dumps({"VideoScope": ["video scope"]}),
        encoding="utf-8",
    )
    scratch_parent = _private_directory(tmp_path / "scratch")
    product = _ProductSnapshot(data_dir.absolute(), media_root.absolute(), events)
    state = SimpleNamespace(
        events=events,
        product=product,
        index=None,
        adapter=None,
        search=None,
        settings=None,
        scratch_parent=scratch_parent,
        fastembed_source=None,
        qdrant_source=None,
    )

    def open_product(**kwargs: object) -> _ProductSnapshot:
        events.append("product.open")
        assert kwargs == {
            "data_dir": data_dir.absolute(),
            "database_path": (data_dir / "videoscope.sqlite3").absolute(),
            "media_root": media_root.absolute(),
            "scratch_parent": scratch_parent.absolute(),
        }
        product.bind_roots()
        return product

    def materialize(source: object, scratch: object, manifest: object) -> _FastSnapshot:
        del manifest
        events.append("fastembed.snapshot")
        state.fastembed_source = source
        assert getattr(source, "parts") == (
            "models",
            "fastembed",
            _FASTEMBED_REPOSITORY_DIRECTORY,
            "snapshots",
            _FASTEMBED_REVISION,
        )
        assert getattr(source, "root").logical_path == data_dir.absolute()
        assert (getattr(source, "stable_path") / "model.sentinel").read_bytes() == b"model"
        assert getattr(scratch, "logical_path").parent == scratch_parent.absolute()
        assert getattr(scratch, "logical_path") != getattr(product, "scratch_root", None)
        output = getattr(scratch, "stable_path") / "fastembed-copy"
        output.mkdir(mode=0o700)
        (output / "model").write_bytes(b"private model")
        (output / "model").chmod(0o600)
        return _FastSnapshot(output, events)

    def snapshot_qdrant(source: object, scratch: object) -> _QdrantSnapshot:
        events.append("qdrant.snapshot")
        state.qdrant_source = source
        assert getattr(source, "parts") == ("qdrant", "text")
        assert getattr(source, "root").logical_path == data_dir.absolute()
        assert (getattr(source, "stable_path") / "index.sentinel").read_bytes() == b"index"
        output = getattr(scratch, "stable_path") / "qdrant-copy"
        output.mkdir(mode=0o700)
        (output / "index").write_bytes(b"private index")
        (output / "index").chmod(0o600)
        return _QdrantSnapshot(output)

    def open_index(snapshot: _QdrantSnapshot, *, embedding: _FastEmbedding) -> _Index:
        events.append("qdrant.open")
        index = _Index(snapshot, embedding, events)
        state.index = index
        return index

    def make_search(repository: object, vector_index: object, **kwargs: object) -> _SearchService:
        events.append("search.create")
        search = _SearchService(repository, vector_index, **kwargs)
        state.search = search
        return search

    def make_adapter(
        search: object,
        *,
        environment_identity: object,
    ) -> _Adapter:
        events.append("adapter.create")
        adapter = _Adapter(search, events, environment_identity)
        state.adapter = adapter
        return adapter

    monkeypatch.setattr(environment_module, "open_product_runtime_snapshot", open_product)
    monkeypatch.setattr(
        environment_module,
        "load_reviewed_fastembed_snapshot_manifest",
        lambda: SimpleNamespace(
            model_content_sha256=_SHA_B,
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            model_revision=_FASTEMBED_REVISION,
            dimensions=768,
        ),
    )
    monkeypatch.setattr(environment_module, "materialize_fastembed_snapshot", materialize)
    monkeypatch.setattr(environment_module, "snapshot_qdrant_storage", snapshot_qdrant)
    monkeypatch.setattr(
        environment_module,
        "verify_fastembed_snapshot",
        lambda _snapshot: events.append("fastembed.verify"),
    )
    monkeypatch.setattr(
        environment_module.QdrantVectorIndex,
        "open_existing_snapshot",
        staticmethod(open_index),
    )
    monkeypatch.setattr(
        environment_module,
        "create_indexing_specifications_from_prompt_snapshot",
        lambda _settings, _prompt: _specifications(),
    )
    monkeypatch.setattr(environment_module, "SearchService", make_search)
    monkeypatch.setattr(environment_module, "ProductBenchmarkSearchAdapter", make_adapter)
    monkeypatch.setattr(
        environment_module,
        "build_runtime",
        lambda *_args, **_kwargs: pytest.fail("build_runtime must never be called"),
        raising=False,
    )
    monkeypatch.setattr(
        Repository,
        "initialize",
        lambda *_args, **_kwargs: pytest.fail("Repository.initialize must never be called"),
    )
    # Keep the imported production symbol live so a false-positive unused import
    # cannot make this guard look stronger than it is.
    assert callable(build_runtime)
    monkeypatch.chdir(tmp_path)
    state.settings = AppSettings(data_dir=Path("product/data"), _env_file=None)
    return state


def test_opens_lexical_environment_from_verified_private_snapshots_only(
    environment_fakes: SimpleNamespace,
) -> None:
    state = environment_fakes
    source_before = _tree(Path("product/data"))

    environment = open_product_benchmark_environment(
        state.settings,
        state.scratch_parent,
        profile_id="lexical_qdrant",
        execution_mode="warm",
    )
    state.product.session_root = environment.scratch_root

    assert state.events == [
        "product.open",
        "fastembed.snapshot",
        "fastembed.create_embedding",
        "fastembed.ensure_ready",
        "qdrant.snapshot",
        "qdrant.open",
        "search.create",
        "adapter.create",
    ]
    assert environment.repository is state.product.repository
    assert environment.asset_resolver is not None
    assert environment.search_adapter is state.adapter
    assert state.adapter.environment_identity.component_id == (
        "benchmark_product_environment"
    )
    assert state.adapter.environment_identity.identity == environment.identity.identity
    assert environment.identity.profile_id == "lexical_qdrant"
    assert environment.identity.execution_mode == "warm"
    assert environment.identity.product_snapshot_sha256 == _SHA_A
    assert environment.identity.fastembed_model_content_sha256 == _SHA_B
    assert environment.identity.qdrant_snapshot_sha256 == _SHA_C
    assert environment.identity.identity.startswith("benchmark-product-environment@2:")
    assert state.search.kwargs["media_root"] == Path("product/data/media").absolute()
    media_access_root = state.search.kwargs["media_access_root"]
    assert str(media_access_root).startswith("/.vol/")
    assert os.stat(media_access_root) == os.stat(Path("product/data/media"))
    assert state.search.kwargs["moment_search"] is None
    assert state.search.kwargs["visual_search"] is None
    assert state.search.kwargs["temporal_refiner"] is None
    assert state.search.kwargs["candidate_reranker"] is None
    assert state.search.kwargs["evaluation_rerankers"] == {}
    assert state.search.kwargs["semantic_text_min_score"] == state.settings.semantic_text_min_score
    assert state.search.kwargs["visual_min_score"] == state.settings.visual_min_score
    assert state.search.kwargs["specification_resolver"]() is state.search.kwargs[
        "specification_resolver"
    ]()
    assert state.search.kwargs["lexicon"].read() == {"VideoScope": ["video scope"]}
    assert _tree(Path("product/data")) == source_before

    environment.close()
    environment.close()

    assert state.events[-4:] == [
        "adapter.close",
        "fastembed.verify",
        "qdrant.close",
        "product.close",
    ]
    assert state.adapter.close_calls == 1
    assert state.index.close_calls == 1
    assert state.product.close_calls == 1
    assert not environment.scratch_root.exists()
    assert list(state.scratch_parent.iterdir()) == []
    assert _tree(Path("product/data")) == source_before


def test_external_reviewed_fastembed_source_is_retained_exactly_and_released_after_copy(
    environment_fakes: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = environment_fakes
    immutable_models = tmp_path / "reviewed-models"
    external_fastembed = _fastembed_snapshot_root(immutable_models / "fastembed")
    external_fastembed.mkdir(parents=True)
    (external_fastembed / "model.sentinel").write_bytes(b"reviewed external model")
    settings = _ExternalModelsSettings(
        _env_file=None,
        data_dir=state.settings.data_dir,
        immutable_models_dir=immutable_models,
    )
    retained_sources: list[object] = []
    opened_source_descriptors: list[int] = []
    original_open_directory = environment_module._open_directory_path

    def track_open_directory(path: Path, *, label: str) -> tuple[Path, int]:
        opened = original_open_directory(path, label=label)
        if label == "FastEmbed model source":
            opened_source_descriptors.append(opened[1])
        return opened

    def materialize(source: object, scratch: object, _manifest: object) -> _FastSnapshot:
        state.events.append("fastembed.snapshot")
        retained_sources.append(source)
        assert isinstance(source, environment_module.RetainedDirectory)
        assert source.logical_path == external_fastembed.absolute()
        assert source.closed is False
        assert (source.stable_path / "model.sentinel").read_bytes() == (
            b"reviewed external model"
        )
        output = getattr(scratch, "stable_path") / "fastembed-copy"
        output.mkdir(mode=0o700)
        (output / "model").write_bytes(b"private model")
        (output / "model").chmod(0o600)
        return _FastSnapshot(output, state.events)

    monkeypatch.setattr(
        environment_module,
        "materialize_fastembed_snapshot",
        materialize,
    )
    monkeypatch.setattr(
        environment_module,
        "_open_directory_path",
        track_open_directory,
    )

    environment = open_product_benchmark_environment(
        settings,
        state.scratch_parent,
    )

    assert len(retained_sources) == 1
    assert getattr(retained_sources[0], "closed") is True
    assert len(opened_source_descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(opened_source_descriptors[0])
    assert (
        _fastembed_snapshot_root(state.settings.models_dir / "fastembed")
        / "model.sentinel"
    ).read_bytes() == b"model"

    environment.close()
    assert state.product.close_calls == 1
    assert list(state.scratch_parent.iterdir()) == []


def test_external_fastembed_source_capability_is_released_when_snapshot_fails(
    environment_fakes: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = environment_fakes
    immutable_models = tmp_path / "reviewed-models"
    external_fastembed = _fastembed_snapshot_root(immutable_models / "fastembed")
    external_fastembed.mkdir(parents=True)
    (external_fastembed / "model.sentinel").write_bytes(b"reviewed external model")
    settings = _ExternalModelsSettings(
        _env_file=None,
        data_dir=state.settings.data_dir,
        immutable_models_dir=immutable_models,
    )
    retained_sources: list[object] = []

    def fail_snapshot(source: object, _scratch: object, _manifest: object) -> None:
        state.events.append("fastembed.snapshot")
        retained_sources.append(source)
        assert isinstance(source, environment_module.RetainedDirectory)
        assert source.closed is False
        raise RuntimeError("snapshot failed")

    monkeypatch.setattr(
        environment_module,
        "materialize_fastembed_snapshot",
        fail_snapshot,
    )

    with pytest.raises(RuntimeError, match="snapshot failed"):
        open_product_benchmark_environment(settings, state.scratch_parent)

    assert len(retained_sources) == 1
    assert getattr(retained_sources[0], "closed") is True
    assert state.product.close_calls == 1
    assert list(state.scratch_parent.iterdir()) == []


def test_external_fastembed_retention_failure_closes_the_opened_source_descriptor(
    environment_fakes: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = environment_fakes
    immutable_models = tmp_path / "reviewed-models"
    external_fastembed = _fastembed_snapshot_root(immutable_models / "fastembed")
    external_fastembed.mkdir(parents=True)
    settings = _ExternalModelsSettings(
        _env_file=None,
        data_dir=state.settings.data_dir,
        immutable_models_dir=immutable_models,
    )
    original_retain = environment_module.RetainedDirectory.retain
    source_descriptors: list[int] = []

    def fail_external_retain(
        _cls: type[object],
        path: Path,
        descriptor: int,
    ) -> object:
        if path == external_fastembed.absolute():
            source_descriptors.append(descriptor)
            raise RuntimeError("retention failed")
        return original_retain(path, descriptor)

    monkeypatch.setattr(
        environment_module.RetainedDirectory,
        "retain",
        classmethod(fail_external_retain),
    )

    with pytest.raises(BenchmarkEnvironmentError, match="could not be retained"):
        open_product_benchmark_environment(settings, state.scratch_parent)

    assert len(source_descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(source_descriptors[0])
    assert state.product.close_calls == 1
    assert list(state.scratch_parent.iterdir()) == []


def test_external_fastembed_source_symlink_is_rejected_without_following_it(
    environment_fakes: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = environment_fakes
    immutable_models = tmp_path / "reviewed-models"
    immutable_models.mkdir()
    outside = tmp_path / "outside-fastembed"
    outside.mkdir()
    (outside / "model.sentinel").write_bytes(b"outside")
    (immutable_models / "fastembed").symlink_to(outside, target_is_directory=True)
    settings = _ExternalModelsSettings(
        _env_file=None,
        data_dir=state.settings.data_dir,
        immutable_models_dir=immutable_models,
    )
    monkeypatch.setattr(
        environment_module,
        "materialize_fastembed_snapshot",
        lambda *_args, **_kwargs: pytest.fail("unsafe source must not be copied"),
    )

    with pytest.raises(BenchmarkEnvironmentError, match="FastEmbed model source"):
        open_product_benchmark_environment(settings, state.scratch_parent)

    assert (outside / "model.sentinel").read_bytes() == b"outside"
    assert state.product.close_calls == 1
    assert list(state.scratch_parent.iterdir()) == []


def test_glossary_is_frozen_and_environment_identity_is_pathless(
    environment_fakes: SimpleNamespace,
) -> None:
    state = environment_fakes
    environment = open_product_benchmark_environment(
        state.settings,
        state.scratch_parent,
    )
    original = state.search.kwargs["lexicon"].read()
    state.settings.glossary_path.write_text(
        json.dumps({"changed": ["after-open"]}),
        encoding="utf-8",
    )

    assert state.search.kwargs["lexicon"].read() == original
    assert str(Path.cwd()) not in json.dumps(environment.identity.canonical_dict)
    environment.close()


def test_frozen_glossary_expands_queries_and_refuses_mutation(
    environment_fakes: SimpleNamespace,
) -> None:
    state = environment_fakes
    environment = open_product_benchmark_environment(state.settings, state.scratch_parent)
    lexicon = state.search.kwargs["lexicon"]

    assert lexicon.expand("find VideoScope demo") == [
        "find VideoScope demo",
        "VideoScope",
        "video scope",
    ]
    assert lexicon.expand("unrelated") == ["unrelated"]
    with pytest.raises(RuntimeError, match="read-only"):
        lexicon.replace({"new": ["entry"]})

    environment.close()


def test_missing_glossary_is_a_stable_empty_snapshot(
    environment_fakes: SimpleNamespace,
) -> None:
    state = environment_fakes
    state.settings.glossary_path.unlink()

    with open_product_benchmark_environment(
        state.settings,
        state.scratch_parent,
    ) as environment:
        assert state.search.kwargs["lexicon"].read() == {}

    assert environment.is_closed is True
    with pytest.raises(BenchmarkEnvironmentError, match="already closed"):
        environment.__enter__()


def test_indexing_specs_use_the_exact_retained_glossary_prompt_snapshot(
    environment_fakes: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = environment_fakes
    captured: list[object] = []

    def make_specifications(_settings: AppSettings, prompt: object) -> object:
        captured.append(prompt)
        return _specifications()

    expected = snapshot_whisper_prompt(
        state.settings.whisper_initial_prompt,
        state.settings.glossary_path,
    )
    monkeypatch.setattr(
        environment_module,
        "create_indexing_specifications_from_prompt_snapshot",
        make_specifications,
    )

    environment = open_product_benchmark_environment(
        state.settings,
        state.scratch_parent,
    )

    assert captured == [expected]
    environment.close()


def test_environment_identity_rejects_tampering(
    environment_fakes: SimpleNamespace,
) -> None:
    state = environment_fakes
    environment = open_product_benchmark_environment(state.settings, state.scratch_parent)

    with pytest.raises(ValueError, match="product_snapshot_sha256"):
        replace(environment.identity, product_snapshot_sha256="not-a-digest")
    with pytest.raises(ValueError, match="score threshold"):
        replace(environment.identity, semantic_text_min_score=True)
    with pytest.raises(ValueError, match="indexing identities"):
        replace(environment.identity, indexing_specification_hashes=())

    environment.close()


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("semantic_text_min_score", -0.01),
        ("semantic_text_min_score", 1.01),
        ("semantic_text_min_score", float("nan")),
        ("visual_min_score", -0.01),
        ("visual_min_score", 1.01),
        ("visual_min_score", float("inf")),
    ),
)
def test_environment_identity_rejects_non_probability_thresholds(
    environment_fakes: SimpleNamespace,
    field_name: str,
    invalid_value: float,
) -> None:
    state = environment_fakes
    environment = open_product_benchmark_environment(state.settings, state.scratch_parent)

    with pytest.raises(ValueError, match="score threshold"):
        replace(environment.identity, **{field_name: invalid_value})

    environment.close()


def test_requires_validated_settings_before_any_product_open(
    environment_fakes: SimpleNamespace,
) -> None:
    state = environment_fakes

    with pytest.raises(ValueError, match="validated AppSettings"):
        open_product_benchmark_environment(  # type: ignore[arg-type]
            object(),
            state.scratch_parent,
        )

    assert state.events == []


def test_product_open_cleanup_owner_is_propagated_unchanged(
    environment_fakes: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = environment_fakes

    class PendingCleanup(ProductSnapshotCleanupError):
        def __init__(self) -> None:
            RuntimeError.__init__(self, "pending product cleanup")

    pending = PendingCleanup()

    def fail_open(**_kwargs: object) -> object:
        raise pending

    monkeypatch.setattr(environment_module, "open_product_runtime_snapshot", fail_open)

    with pytest.raises(ProductSnapshotCleanupError) as captured:
        open_product_benchmark_environment(state.settings, state.scratch_parent)

    assert captured.value is pending
    assert list(state.scratch_parent.iterdir()) == []


@pytest.mark.parametrize("profile_id", tuple(FROZEN_PROFILES))
def test_opens_every_frozen_warm_profile(
    environment_fakes: SimpleNamespace,
    profile_id: str,
) -> None:
    state = environment_fakes

    environment = open_product_benchmark_environment(
        state.settings,
        state.scratch_parent,
        profile_id=profile_id,
        execution_mode="warm",
    )

    assert environment.identity.profile_id == profile_id
    assert environment.identity.profile_identity == FROZEN_PROFILES[profile_id].identity
    environment.close()


@pytest.mark.parametrize(
    ("profile_id", "execution_mode"),
    [("unknown_profile", "warm"), ("lexical_qdrant", "cold")],
)
def test_rejects_unknown_profiles_and_cold_mode_before_opening_product(
    environment_fakes: SimpleNamespace,
    profile_id: str,
    execution_mode: str,
) -> None:
    state = environment_fakes

    with pytest.raises(BenchmarkEnvironmentError, match="supports only"):
        open_product_benchmark_environment(
            state.settings,
            state.scratch_parent,
            profile_id=profile_id,
            execution_mode=execution_mode,  # type: ignore[arg-type]
        )

    assert state.events == []
    assert list(state.scratch_parent.iterdir()) == []


def test_qwen_profile_wires_only_read_only_selected_providers(
    environment_fakes: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = environment_fakes
    source_before = _tree(Path("product/data"))
    probes: list[tuple[str, Path, Path]] = []

    def record_probe(role: str, client: object, source: Path) -> bool:
        assert source.is_file()
        assert source.read_bytes() == b"videoscope-worker-source-probe-v1\n"
        input_root = Path(getattr(client, "input_root"))
        assert source.is_relative_to(input_root)
        probes.append((role, input_root, source))
        return True

    monkeypatch.setattr(
        environment_module.VisionWorkerClient,
        "probe_source",
        lambda client, source: record_probe("vision", client, source),
    )
    monkeypatch.setattr(
        environment_module.QwenWorkerClient,
        "probe_source",
        lambda client, source: record_probe("qwen", client, source),
    )

    class _Toolchain:
        def verify_current(self) -> str:
            return "sha256:" + "9" * 64

        def create_ffmpeg(self) -> FFmpeg:
            return FFmpeg.from_attested_paths(
                Path("/usr/bin/false"),
                Path("/usr/bin/false"),
            )

    monkeypatch.setattr(
        environment_module,
        "attest_indexing_toolchain",
        lambda: _Toolchain(),
        raising=False,
    )
    settings = state.settings.model_copy(
        update={
            "vision_worker_endpoint": "http://127.0.0.1:9101",
            "vision_worker_api_key": "v" * 32,
            "lighthouse_endpoint": "http://127.0.0.1:9102",
            "lighthouse_api_key": "l" * 32,
            "qwen_video_endpoint": "http://127.0.0.1:9103",
            "qwen_video_api_key": "q" * 32,
            "qwen_video_model": "mlx-community/Qwen3.5-9B-MLX-4bit",
        }
    )

    environment = open_product_benchmark_environment(
        settings,
        state.scratch_parent,
        profile_id="qwen_verification",
        execution_mode="warm",
    )

    assert getattr(state.search.kwargs["visual_search"], "id", None) == "siglip2"
    temporal_refiner = state.search.kwargs["temporal_refiner"]
    assert temporal_refiner is not None
    assert temporal_refiner.benchmark_attestation is not None
    assert not temporal_refiner.temp_dir.exists()
    assert getattr(state.search.kwargs["moment_search"], "id", None) == "lighthouse"
    assert state.search.kwargs["candidate_reranker"] is None
    assert tuple(state.search.kwargs["evaluation_rerankers"]) == ("qwen",)
    qwen = state.search.kwargs["evaluation_rerankers"]["qwen"]
    assert getattr(qwen, "id", None) == "qwen-video"
    assert qwen.benchmark_attestation["candidate_limit"] == 12
    assert not qwen.temp_dir.exists()
    assert not qwen.cache_dir.exists()
    assert environment.probe_worker_sources() == ("qwen", "vision")
    assert [role for role, _root, _source in probes] == ["qwen", "vision"]
    assert all(root == state.scratch_parent for _role, root, _source in probes)
    assert all(not source.exists() for _role, _root, source in probes)
    visual_client = state.search.kwargs["visual_search"].inference_client
    assert environment.identity.worker_input_root_identities == (
        ("qwen", "sha256:" + qwen.inference_client.input_root_sha256),
        ("vision", visual_client.input_root_identity),
    )
    assert _tree(Path("product/data")) == source_before

    environment.close()
    assert _tree(Path("product/data")) == source_before


def test_internvideo_profile_is_explicitly_not_wired(
    environment_fakes: SimpleNamespace,
) -> None:
    state = environment_fakes
    settings = state.settings.model_copy(
        update={
            "internvideo_endpoint": "https://example.invalid",
            "internvideo_api_key": "i" * 32,
        }
    )

    environment = open_product_benchmark_environment(
        settings,
        state.scratch_parent,
        profile_id="internvideo",
        execution_mode="warm",
    )

    assert state.search.kwargs["candidate_reranker"] is None
    assert "internvideo" not in state.search.kwargs["evaluation_rerankers"]
    environment.close()


def test_worker_source_probe_fails_closed_and_removes_marker(
    environment_fakes: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = environment_fakes
    settings = state.settings.model_copy(
        update={
            "vision_worker_endpoint": "http://127.0.0.1:9101",
            "vision_worker_api_key": "v" * 32,
        }
    )
    monkeypatch.setattr(
        environment_module.VisionWorkerClient,
        "probe_source",
        lambda _client, _source: (_ for _ in ()).throw(RuntimeError("stale root")),
    )
    environment = open_product_benchmark_environment(
        settings,
        state.scratch_parent,
        profile_id="dense_siglip",
    )

    with pytest.raises(BenchmarkEnvironmentError, match="source boundary"):
        environment.probe_worker_sources()

    assert not list(environment.scratch_root.rglob("worker-source-probe.bin"))
    environment.close()


def test_snapshot_setup_failure_rolls_back_private_scratch_then_product_lock(
    environment_fakes: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = environment_fakes

    def fail_snapshot(_source: object, scratch: object, _manifest: object) -> None:
        state.events.append("fastembed.snapshot")
        partial = getattr(scratch, "stable_path") / "partial"
        partial.mkdir(mode=0o700)
        (partial / "file").write_bytes(b"partial")
        (partial / "file").chmod(0o600)
        raise RuntimeError("snapshot failed")

    monkeypatch.setattr(environment_module, "materialize_fastembed_snapshot", fail_snapshot)

    with pytest.raises(RuntimeError, match="snapshot failed"):
        open_product_benchmark_environment(state.settings, state.scratch_parent)

    assert state.events == ["product.open", "fastembed.snapshot", "product.close"]
    assert state.product.close_calls == 1
    assert list(state.scratch_parent.iterdir()) == []


def test_warm_profile_rejects_an_embedding_that_did_not_finish_warming(
    environment_fakes: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = environment_fakes

    class UnreadyEmbedding(_FastEmbedding):
        def ensure_ready(self) -> bool:
            self._events.append("fastembed.ensure_ready")
            return False

    def create_unready(snapshot: _FastSnapshot) -> _FastEmbedding:
        snapshot._events.append("fastembed.create_embedding")
        return UnreadyEmbedding(snapshot._events)

    monkeypatch.setattr(_FastSnapshot, "create_embedding", create_unready)

    with pytest.raises(BenchmarkEnvironmentError, match="warmed"):
        open_product_benchmark_environment(state.settings, state.scratch_parent)

    assert "qdrant.snapshot" not in state.events
    assert state.events[:4] == [
        "product.open",
        "fastembed.snapshot",
        "fastembed.create_embedding",
        "fastembed.ensure_ready",
    ]
    assert state.events[-2:] == ["fastembed.verify", "product.close"]
    assert state.product.close_calls == 1
    assert list(state.scratch_parent.iterdir()) == []


def test_scratch_setup_and_product_close_failure_exposes_retry_owner(
    environment_fakes: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = environment_fakes
    state.product.close_failures = 1
    monkeypatch.setattr(
        environment_module._PrivateScratch,
        "create",
        classmethod(lambda _cls, _parent: (_ for _ in ()).throw(RuntimeError("scratch failed"))),
    )

    with pytest.raises(BenchmarkEnvironmentCleanupError) as captured:
        open_product_benchmark_environment(state.settings, state.scratch_parent)

    error = captured.value
    assert "scratch failed" in str(error.__cause__)
    assert state.product.close_calls == 1
    assert error.environment.is_closed is False
    with pytest.raises(BenchmarkEnvironmentError, match="scratch is unavailable"):
        _ = error.environment.scratch_root

    error.environment.close()
    assert state.product.close_calls == 2
    assert error.environment.is_closed is True


def test_scratch_cleanup_enforces_entry_bound_before_product_unlock(
    environment_fakes: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = environment_fakes
    environment = open_product_benchmark_environment(state.settings, state.scratch_parent)
    for index in range(3):
        path = environment.scratch_root / f"extra-{index}"
        path.write_bytes(b"x")
        path.chmod(0o600)
    monkeypatch.setattr(environment_module, "_MAX_SCRATCH_ENTRIES", 2)

    with pytest.raises(BenchmarkEnvironmentError, match="entry limit"):
        environment.close()

    assert state.product.close_calls == 0
    assert environment.scratch_root.is_dir()

    monkeypatch.setattr(environment_module, "_MAX_SCRATCH_ENTRIES", 200_000)
    environment.close()
    assert state.product.close_calls == 1


def test_scratch_cleanup_removes_nested_private_directories(
    environment_fakes: SimpleNamespace,
) -> None:
    state = environment_fakes
    environment = open_product_benchmark_environment(state.settings, state.scratch_parent)
    nested = environment.scratch_root / "one" / "two" / "three"
    nested.mkdir(parents=True, mode=0o700)
    payload = nested / "payload"
    payload.write_bytes(b"private")
    payload.chmod(0o600)

    environment.close()

    assert state.product.close_calls == 1
    assert list(state.scratch_parent.iterdir()) == []


def test_provider_close_failure_retains_lock_and_scratch_until_retry(
    environment_fakes: SimpleNamespace,
) -> None:
    state = environment_fakes
    environment = open_product_benchmark_environment(state.settings, state.scratch_parent)
    state.index.close_failures = 1

    with pytest.raises(BenchmarkEnvironmentError, match="Qdrant"):
        environment.close()

    assert state.adapter.close_calls == 1
    assert state.index.close_calls == 1
    assert state.product.close_calls == 0
    assert environment.scratch_root.is_dir()

    environment.close()

    assert state.adapter.close_calls == 1
    assert state.index.close_calls == 2
    assert state.product.close_calls == 1
    assert not environment.scratch_root.exists()


def test_retained_scratch_capability_close_failure_blocks_owner_release_until_retry(
    environment_fakes: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = environment_fakes
    environment = open_product_benchmark_environment(state.settings, state.scratch_parent)
    scratch_capability = environment._scratch_root
    original_close = environment_module.RetainedDirectory.close
    attempts = 0

    def fail_once(capability: object) -> bool:
        nonlocal attempts
        if capability is scratch_capability:
            attempts += 1
            if attempts == 1:
                raise RuntimeError("capability close failed")
        return original_close(capability)  # type: ignore[arg-type]

    monkeypatch.setattr(environment_module.RetainedDirectory, "close", fail_once)

    with pytest.raises(BenchmarkEnvironmentError, match="capability"):
        environment.close()

    assert state.index.close_calls == 1
    assert state.product.close_calls == 0
    assert environment.scratch_root.is_dir()

    environment.close()
    assert attempts == 2
    assert state.index.close_calls == 1
    assert state.product.close_calls == 1


def test_adapter_cleanup_failure_blocks_all_later_close_phases_until_retry(
    environment_fakes: SimpleNamespace,
) -> None:
    state = environment_fakes
    environment = open_product_benchmark_environment(state.settings, state.scratch_parent)
    state.adapter.close_failures = 1

    with pytest.raises(BenchmarkEnvironmentError, match="adapter"):
        environment.close()

    assert state.index.close_calls == 0
    assert state.product.close_calls == 0
    assert environment.scratch_root.is_dir()

    environment.close()
    assert state.adapter.close_calls == 2
    assert state.index.close_calls == 1
    assert state.product.close_calls == 1


def test_final_model_attestation_failure_reports_drift_but_still_cleans_owners(
    environment_fakes: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = environment_fakes
    environment = open_product_benchmark_environment(state.settings, state.scratch_parent)

    def fail_verification(_snapshot: object) -> None:
        state.events.append("fastembed.verify.failed")
        raise RuntimeError("model bytes drifted")

    monkeypatch.setattr(
        environment_module,
        "verify_fastembed_snapshot",
        fail_verification,
    )

    with pytest.raises(BenchmarkEnvironmentError, match="final verification"):
        environment.close()

    assert environment.is_closed is True
    assert state.index.close_calls == 1
    assert state.product.close_calls == 1
    assert list(state.scratch_parent.iterdir()) == []
    environment.close()


def test_invalid_qdrant_attestation_is_rejected_and_rolled_back(
    environment_fakes: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = environment_fakes

    class BadIndex(_Index):
        @property
        def benchmark_attestation(self) -> dict[str, object]:
            payload = super().benchmark_attestation
            payload["strict_no_fallback"] = False
            return payload

    def open_bad(snapshot: _QdrantSnapshot, *, embedding: _FastEmbedding) -> BadIndex:
        state.events.append("qdrant.open")
        index = BadIndex(snapshot, embedding, state.events)
        state.index = index
        return index

    monkeypatch.setattr(
        environment_module.QdrantVectorIndex,
        "open_existing_snapshot",
        staticmethod(open_bad),
    )

    with pytest.raises(BenchmarkEnvironmentError, match="attestation"):
        open_product_benchmark_environment(state.settings, state.scratch_parent)

    assert state.index.close_calls == 1
    assert state.product.close_calls == 1
    assert list(state.scratch_parent.iterdir()) == []


def test_setup_and_cleanup_failure_exposes_retryable_retained_environment(
    environment_fakes: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = environment_fakes

    def fail_search(*_args: object, **_kwargs: object) -> object:
        state.events.append("search.create")
        assert state.index is not None
        state.index.close_failures = 1
        raise RuntimeError("search construction failed")

    monkeypatch.setattr(environment_module, "SearchService", fail_search)

    with pytest.raises(BenchmarkEnvironmentCleanupError) as captured:
        open_product_benchmark_environment(state.settings, state.scratch_parent)

    error = captured.value
    assert "search construction failed" in str(error.__cause__)
    assert state.product.close_calls == 0
    assert error.environment.scratch_root.is_dir()

    error.environment.close()
    assert state.index.close_calls == 2
    assert state.product.close_calls == 1
    assert list(state.scratch_parent.iterdir()) == []


def test_glossary_symlink_is_rejected_without_following_it(
    environment_fakes: SimpleNamespace,
) -> None:
    state = environment_fakes
    outside = Path("outside-glossary.json")
    outside.write_text(json.dumps({"secret": ["outside"]}), encoding="utf-8")
    state.settings.glossary_path.unlink()
    state.settings.glossary_path.symlink_to(outside.absolute())

    with pytest.raises(BenchmarkEnvironmentError, match="glossary"):
        open_product_benchmark_environment(state.settings, state.scratch_parent)

    assert outside.read_text(encoding="utf-8") == json.dumps({"secret": ["outside"]})
    assert state.product.close_calls == 1
    assert list(state.scratch_parent.iterdir()) == []


def test_scratch_parent_replacement_cannot_redirect_private_snapshot_bytes(
    environment_fakes: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = environment_fakes
    original_materialize = environment_module.materialize_fastembed_snapshot
    moved_parent = tmp_path / "scratch-moved"
    replacement_root: Path | None = None

    def swap_parent(source: object, scratch: object, manifest: object) -> object:
        nonlocal replacement_root
        logical_root = getattr(scratch, "logical_path")
        state.scratch_parent.rename(moved_parent)
        _private_directory(state.scratch_parent)
        replacement_root = _private_directory(
            state.scratch_parent / logical_root.name
        )
        snapshot = original_materialize(source, scratch, manifest)
        assert list(replacement_root.iterdir()) == []
        return snapshot

    monkeypatch.setattr(
        environment_module,
        "materialize_fastembed_snapshot",
        swap_parent,
    )

    environment = open_product_benchmark_environment(
        state.settings,
        state.scratch_parent,
    )
    environment.close()

    assert replacement_root is not None
    assert list(replacement_root.iterdir()) == []
    assert list(moved_parent.iterdir()) == []
    assert state.product.close_calls == 1
    replacement_root.rmdir()
    state.scratch_parent.rmdir()
    moved_parent.rmdir()


def test_product_root_replacement_after_retention_cannot_mix_component_bytes(
    environment_fakes: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = environment_fakes
    original_materialize = environment_module.materialize_fastembed_snapshot
    lexical_data = Path("product/data").absolute()
    moved_data = tmp_path / "product-data-moved"

    def swap_product_root(
        source: object,
        scratch: object,
        manifest: object,
    ) -> object:
        lexical_data.rename(moved_data)
        replacement_model = _fastembed_snapshot_root(
            lexical_data / "models" / "fastembed"
        )
        replacement_qdrant = lexical_data / "qdrant" / "text"
        replacement_media = lexical_data / "media"
        replacement_model.mkdir(parents=True)
        replacement_qdrant.mkdir(parents=True)
        replacement_media.mkdir(parents=True)
        (replacement_model / "model.sentinel").write_bytes(b"replacement model")
        (replacement_qdrant / "index.sentinel").write_bytes(b"replacement index")
        (lexical_data / "search-glossary.json").write_text(
            json.dumps({"replacement": ["unsafe"]}),
            encoding="utf-8",
        )
        return original_materialize(source, scratch, manifest)

    monkeypatch.setattr(
        environment_module,
        "materialize_fastembed_snapshot",
        swap_product_root,
    )

    environment = open_product_benchmark_environment(
        state.settings,
        state.scratch_parent,
    )

    assert state.search.kwargs["lexicon"].read() == {
        "VideoScope": ["video scope"]
    }
    assert (
        Path(getattr(state.fastembed_source, "stable_path")) / "model.sentinel"
    ).read_bytes() == b"model"
    assert (
        Path(getattr(state.qdrant_source, "stable_path")) / "index.sentinel"
    ).read_bytes() == b"index"
    assert (
        (
            _fastembed_snapshot_root(Path("product/data/models/fastembed"))
            / "model.sentinel"
        ).read_bytes()
        == b"replacement model"
    )

    environment.close()
    assert state.product.close_calls == 1
    assert list(state.scratch_parent.iterdir()) == []
