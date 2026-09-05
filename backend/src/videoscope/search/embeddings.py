from __future__ import annotations

import hashlib
import gc
from dataclasses import dataclass, field
from importlib import metadata
import os
from pathlib import Path
import re
import stat
from tempfile import TemporaryDirectory
from threading import Event, Lock
import traceback
import unicodedata
from collections.abc import Callable, Iterable

import numpy as np

from videoscope.model_manifest import (
    FASTEMBED_ALGORITHM_VERSION,
    FASTEMBED_RUNTIME_VERSION,
    fastembed_cache_snapshot_path,
    fastembed_snapshot,
    model_identity,
)


_EMBEDDING_CLOSE_TIMEOUT_SECONDS = 1.0


def _directory_identity(metadata_value: os.stat_result) -> tuple[int, int, int, int]:
    return (
        metadata_value.st_dev,
        metadata_value.st_ino,
        metadata_value.st_mode,
        metadata_value.st_uid,
    )


def _validated_macos_file_id_directory(candidate: Path) -> Path:
    parts = candidate.parts
    if (
        len(parts) < 4
        or parts[1] != ".vol"
        or not parts[2].isdigit()
        or not parts[3].isdigit()
        or str(int(parts[2])) != parts[2]
        or str(int(parts[3])) != parts[3]
        or any(part in {"", ".", ".."} for part in parts[4:])
    ):
        raise ValueError("strict semantic embedding requires a verified model path")
    expected = (int(parts[2]), int(parts[3]))
    if expected[0] <= 0 or expected[1] <= 0:
        raise ValueError("strict semantic embedding requires a verified model path")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if not nofollow or not directory:
        raise ValueError("strict semantic embedding requires a verified model path")
    flags = os.O_RDONLY | nofollow | directory | getattr(os, "O_CLOEXEC", 0)
    stable_root = Path(f"/.vol/{expected[0]}/{expected[1]}")
    current_fd: int | None = None
    try:
        root_before = os.stat(stable_root, follow_symlinks=False)
        current_fd = os.open(stable_root, flags)
        root_opened = os.fstat(current_fd)
        root_after = os.stat(stable_root, follow_symlinks=False)
        if (
            not stat.S_ISDIR(root_before.st_mode)
            or _directory_identity(root_before) != _directory_identity(root_opened)
            or _directory_identity(root_opened) != _directory_identity(root_after)
            or (root_opened.st_dev, root_opened.st_ino) != expected
        ):
            raise ValueError(
                "strict semantic embedding requires a verified model path"
            )
        for component in parts[4:]:
            observed = os.stat(
                component,
                dir_fd=current_fd,
                follow_symlinks=False,
            )
            if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
                raise ValueError(
                    "strict semantic embedding requires a verified model path"
                )
            child_fd = os.open(component, flags, dir_fd=current_fd)
            try:
                opened = os.fstat(child_fd)
                current = os.stat(
                    component,
                    dir_fd=current_fd,
                    follow_symlinks=False,
                )
                if (
                    _directory_identity(observed) != _directory_identity(opened)
                    or _directory_identity(opened) != _directory_identity(current)
                ):
                    raise ValueError(
                        "strict semantic embedding requires a verified model path"
                    )
            except Exception:
                os.close(child_fd)
                raise
            os.close(current_fd)
            current_fd = child_fd
    except OSError as error:
        raise ValueError(
            "strict semantic embedding requires an existing verified model path"
        ) from error
    finally:
        if current_fd is not None:
            os.close(current_fd)
    return candidate


def _validated_nofollow_directory(path: Path) -> Path:
    candidate = Path(path).absolute()
    if len(candidate.parts) >= 2 and candidate.parts[1] == ".vol":
        return _validated_macos_file_id_directory(candidate)
    current = Path(candidate.anchor)
    try:
        for component in candidate.parts[1:]:
            current /= component
            metadata_value = current.lstat()
            if stat.S_ISLNK(metadata_value.st_mode) or not stat.S_ISDIR(
                metadata_value.st_mode
            ):
                raise ValueError(
                    "strict semantic embedding requires a verified model path"
                )
    except OSError as error:
        raise ValueError(
            "strict semantic embedding requires an existing verified model path"
        ) from error
    return candidate


class HashEmbedding:
    """Детерминированный локальный резервный вариант на основе признаков слов и символов."""

    def __init__(self, dimensions: int = 384) -> None:
        if dimensions < 16:
            raise ValueError("embedding dimensions must be at least 16")
        self.dimensions = dimensions

    @property
    def identity(self) -> str:
        return f"hash-embedding-v1:{self.dimensions}"

    def _features(self, text: str) -> Iterable[str]:
        normalized = unicodedata.normalize("NFKC", text).casefold()
        words = re.findall(r"[\w]+", normalized, flags=re.UNICODE)
        for word in words:
            yield f"w:{word}"
            padded = f"^{word}$"
            for width in (3, 4, 5):
                for index in range(max(0, len(padded) - width + 1)):
                    yield f"c:{padded[index:index + width]}"

    def _one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for feature in self._features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            index = value % self.dimensions
            vector[index] += 1.0 if value & 1 else -1.0
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        return [self._one(text) for text in texts]


class SemanticEmbedding:
    """FastEmbed с детерминированным автономным резервным вариантом."""

    def __init__(
        self,
        *,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_repository: str | None = None,
        model_revision: str | None = None,
        expected_runtime_version: str | None = None,
        algorithm_version: str = "unmanaged",
        dimensions: int = 384,
        cache_dir: Path | None = None,
        strict: bool = False,
        specific_model_path: Path | None = None,
        model_content_sha256: str | None = None,
        model_verifier: Callable[[], None] | None = None,
    ) -> None:
        self.model_name = model_name
        self.model_repository = model_repository
        self.model_revision = model_revision
        self.expected_runtime_version = expected_runtime_version
        self.algorithm_version = algorithm_version
        self.dimensions = dimensions
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.strict_no_fallback = bool(strict)
        self.specific_model_path = None
        self.model_content_sha256 = model_content_sha256
        self._model_verifier = model_verifier
        if model_verifier is not None and not callable(model_verifier):
            raise ValueError("semantic embedding model verifier must be callable")
        if self.strict_no_fallback:
            if (
                not self.model_repository
                or not self.model_revision
                or not self.expected_runtime_version
                or self.algorithm_version == "unmanaged"
                or specific_model_path is None
                or type(self.model_content_sha256) is not str
                or re.fullmatch(r"[0-9a-f]{64}", self.model_content_sha256) is None
            ):
                raise ValueError(
                    "strict semantic embedding requires a complete verified identity"
                )
            self.specific_model_path = _validated_nofollow_directory(
                Path(specific_model_path)
            )
        self._model = None
        self._operation_lock = Lock()
        self._closing = Event()
        self._closed = False
        self._attempted = False
        self._validated = False
        self._attestation_failed = False
        self._fallback = HashEmbedding(dimensions)
        self.last_error: str | None = None

    @property
    def backend(self) -> str:
        if self._closing.is_set():
            return "unavailable"
        if self._model is not None:
            return "fastembed"
        if self.strict_no_fallback and self._attempted:
            return "unavailable"
        return "hash-fallback" if self._attempted else "not-loaded"

    @property
    def identity(self) -> str:
        source = model_identity(
            self.model_repository or self.model_name,
            self.model_revision,
        )
        runtime = self.expected_runtime_version or "unmanaged"
        identity = (
            f"fastembed@{runtime}:{self.algorithm_version}:"
            f"{self.model_name}:{source}:{self.dimensions}"
        )
        # This identity is persisted in the physical Qdrant collection contract.
        # Strict benchmark readers must address the exact production collection;
        # reviewed model bytes are attested separately in ``benchmark_identity``.
        return identity

    @property
    def benchmark_identity(self) -> dict[str, object]:
        if not self.strict_no_fallback:
            raise RuntimeError("benchmark embedding identity requires strict mode")
        return self.attestation_identity

    @property
    def attestation_identity(self) -> dict[str, object]:
        """Pathless identity of the reviewed bytes used by this exact runtime."""
        if not self.strict_no_fallback or self._model_verifier is None:
            raise RuntimeError("reviewed embedding identity requires attested strict mode")
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

    def ensure_ready(self) -> bool:
        if self._closing.is_set():
            return False
        with self._operation_lock:
            if self._closing.is_set():
                return False
            return self._ensure_ready()

    def _ensure_ready(self) -> bool:
        if self._validated:
            if self.strict_no_fallback and not self._verify_benchmark_model_current():
                return False
            return True
        model = self._load()
        if model is None:
            return False
        try:
            probe = next(iter(model.embed(["VideoScope embedding dimension probe"])))
            self._normalize_vector(probe)
            if self.strict_no_fallback:
                self._verify_strict_model()
        except Exception as error:
            self.last_error = str(error)
            self._model = None
            return False
        self._validated = True
        return True

    def verify_benchmark_model_current(self) -> bool:
        """Revalidate reviewed model bytes outside the measured query boundary."""
        if self._closing.is_set():
            return False
        with self._operation_lock:
            if self._closing.is_set():
                return False
            return self._verify_benchmark_model_current()

    def _verify_benchmark_model_current(self) -> bool:
        if (
            not self.strict_no_fallback
            or not self._validated
            or self._model is None
            or self._model_verifier is None
            or self._attestation_failed
        ):
            return False
        try:
            self._verify_strict_model()
        except Exception as error:
            self.last_error = str(error)
            return False
        return True

    def _verify_strict_model(self) -> None:
        if not self.strict_no_fallback or self._model_verifier is None:
            return
        if self._attestation_failed:
            raise RuntimeError("reviewed semantic embedding attestation failed")
        try:
            self._model_verifier()
        except Exception:
            # Once any verification boundary observes drift, the already-loaded
            # ONNX session can no longer be trusted even if files are restored.
            self._attestation_failed = True
            self._validated = False
            self._model = None
            raise

    def _load(self):  # type: ignore[no-untyped-def]
        if self._attestation_failed or self._closed:
            return None
        if self._model is not None:
            return self._model
        if self._attempted:
            return None
        self._attempted = True
        try:
            self._verify_strict_model()
            if (
                self.expected_runtime_version is not None
                and metadata.version("fastembed") != self.expected_runtime_version
            ):
                raise RuntimeError(
                    f"fastembed {self.expected_runtime_version} is required"
                )
            from fastembed import TextEmbedding

            options: dict[str, object] = {"model_name": self.model_name}
            if self.strict_no_fallback:
                # The benchmark materializer has already copied and verified every
                # allowed model file. Passing the private directory directly keeps
                # FastEmbed away from caches, downloads, and mutable hub pointers.
                options["specific_model_path"] = str(self.specific_model_path)
            elif self.cache_dir is not None:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                options["cache_dir"] = str(self.cache_dir)
            if (
                not self.strict_no_fallback
                and self.model_repository
                and self.model_revision
            ):
                from huggingface_hub import snapshot_download

                snapshot = snapshot_download(
                    self.model_repository,
                    revision=self.model_revision,
                    local_files_only=True,
                    cache_dir=str(self.cache_dir) if self.cache_dir is not None else None,
                )
                options["specific_model_path"] = str(snapshot)
            candidate = TextEmbedding(**options)
            self._verify_strict_model()
            self._model = candidate
            return self._model
        except Exception as error:
            self.last_error = str(error)
            return None

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        self._require_open()
        with self._operation_lock:
            self._require_open()
            return self._embed(texts)

    def _embed(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            if self.strict_no_fallback:
                try:
                    self._verify_strict_model()
                except Exception as error:
                    self.last_error = str(error)
                    raise RuntimeError("semantic embedding failed") from error
            return []
        model = self._load()
        if model is None:
            if self.strict_no_fallback:
                raise RuntimeError("strict semantic embedding is unavailable")
            return self._fallback.embed(texts)
        try:
            self._verify_strict_model()
            vectors = [self._normalize_vector(vector) for vector in model.embed(texts)]
            if len(vectors) != len(texts):
                raise ValueError("embedding model returned an unexpected vector count")
            self._verify_strict_model()
            return vectors
        except Exception as error:
            self.last_error = str(error)
            raise RuntimeError("semantic embedding failed") from error

    def embed_query(self, query: str) -> np.ndarray:
        self._require_open()
        with self._operation_lock:
            self._require_open()
            return self._embed_query(query)

    def _embed_query(self, query: str) -> np.ndarray:
        model = self._load()
        if model is None:
            if self.strict_no_fallback:
                raise RuntimeError("strict semantic embedding is unavailable")
            return self._fallback.embed([query])[0]
        try:
            self._verify_strict_model()
            vector = self._normalize_vector(next(iter(model.query_embed(query))))
            self._verify_strict_model()
            return vector
        except Exception as error:
            self.last_error = str(error)
            raise RuntimeError("semantic query embedding failed") from error

    def _require_open(self) -> None:
        if self._closing.is_set():
            raise RuntimeError("semantic embedding is closed")

    def close(self) -> bool:
        """Stop accepting calls and release the model once inference is quiescent.

        Busy owners must retain their model snapshot and retry this operation;
        closing never cancels an encode while it is reading that snapshot.
        """
        self._closing.set()
        if not self._operation_lock.acquire(timeout=_EMBEDDING_CLOSE_TIMEOUT_SECONDS):
            raise RuntimeError("semantic embedding is busy; close requires retry")
        try:
            if self._closed:
                return False
            self._model = None
            self._validated = False
            gc.collect()
            self._closed = True
            return True
        finally:
            self._operation_lock.release()

    def _normalize_vector(self, value: object) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float32)
        if vector.shape != (self.dimensions,):
            raise ValueError("embedding model returned an unexpected vector size")
        if not np.all(np.isfinite(vector)):
            raise ValueError("embedding model returned non-finite values")
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError("embedding model returned a zero vector")
        return vector / norm


SEMANTIC_EMBEDDING_TYPE_IDENTITY = (
    f"{SemanticEmbedding.__module__}.{SemanticEmbedding.__qualname__}"
)


class ReviewedSemanticEmbeddingError(RuntimeError):
    """The checked-in FastEmbed contract or its local bytes are unavailable."""


@dataclass(frozen=True, slots=True)
class ReviewedSemanticEmbeddingContract:
    """Pathless serving contract shared by plans and the production writer."""

    model_name: str
    model_repository: str
    model_revision: str
    runtime_version: str
    algorithm_version: str
    dimensions: int
    embedding_identity: str
    model_content_sha256: str
    semantic_embedding_type: str
    manifest: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        from videoscope.benchmark.snapshots import FastEmbedSnapshotManifest

        if type(self.manifest) is not FastEmbedSnapshotManifest:
            raise ValueError("reviewed semantic embedding manifest is invalid")
        manifest_identity = (
            self.manifest.model_name,
            self.manifest.model_repository,
            self.manifest.model_revision,
            self.manifest.runtime_version,
            self.manifest.algorithm_version,
            self.manifest.dimensions,
            self.manifest.model_content_sha256,
        )
        contract_identity = (
            self.model_name,
            self.model_repository,
            self.model_revision,
            self.runtime_version,
            self.algorithm_version,
            self.dimensions,
            self.model_content_sha256,
        )
        if contract_identity != manifest_identity:
            raise ValueError("reviewed semantic embedding contract differs from manifest")
        expected_embedding = SemanticEmbedding(
            model_name=self.model_name,
            model_repository=self.model_repository,
            model_revision=self.model_revision,
            expected_runtime_version=self.runtime_version,
            algorithm_version=self.algorithm_version,
            dimensions=self.dimensions,
        )
        if (
            type(expected_embedding) is not SemanticEmbedding
            or self.embedding_identity != expected_embedding.identity
            or self.semantic_embedding_type != SEMANTIC_EMBEDDING_TYPE_IDENTITY
        ):
            raise ValueError("reviewed semantic embedding contract is not concrete")

    @property
    def embedding_attestation_dict(self) -> dict[str, object]:
        return {
            "algorithm_version": self.algorithm_version,
            "dimensions": self.dimensions,
            "embedding_identity": self.embedding_identity,
            "model_content_sha256": self.model_content_sha256,
            "model_name": self.model_name,
            "model_repository": self.model_repository,
            "model_revision": self.model_revision,
            "runtime_version": self.runtime_version,
        }

    @property
    def canonical_dict(self) -> dict[str, object]:
        return {
            **self.embedding_attestation_dict,
            "semantic_embedding_type": self.semantic_embedding_type,
        }


@dataclass(frozen=True, slots=True)
class UnavailableReviewedSemanticEmbedding:
    """Explicit no-writer state for a reviewed model whose bytes are absent.

    The logical identity remains the production identity so existing Qdrant
    generations can still be addressed for lifecycle operations.  This object
    cannot encode text, cannot become ready, and contains no hash fallback.
    """

    contract: ReviewedSemanticEmbeddingContract
    last_error: str = "reviewed semantic embedding bytes are unavailable"
    strict_no_fallback: bool = field(default=True, init=False)
    backend: str = field(default="unavailable", init=False)

    def __post_init__(self) -> None:
        if type(self.contract) is not ReviewedSemanticEmbeddingContract:
            raise ValueError("unavailable semantic embedding contract is invalid")
        if type(self.last_error) is not str or not self.last_error.strip():
            raise ValueError("unavailable semantic embedding reason is invalid")

    @property
    def identity(self) -> str:
        return self.contract.embedding_identity

    @property
    def dimensions(self) -> int:
        return self.contract.dimensions

    @property
    def required_attestation_identity(self) -> dict[str, object]:
        """The exact attestation a future enabled writer must satisfy."""
        return self.contract.canonical_dict

    def ensure_ready(self) -> bool:
        return False

    def verify_benchmark_model_current(self) -> bool:
        return False

    def embed(self, _texts: list[str]) -> list[np.ndarray]:
        raise ReviewedSemanticEmbeddingError(
            "reviewed semantic embedding is unavailable"
        )

    def embed_query(self, _query: str) -> np.ndarray:
        raise ReviewedSemanticEmbeddingError(
            "reviewed semantic embedding is unavailable"
        )


def load_reviewed_semantic_embedding_contract(
    *,
    model_name: str,
    dimensions: int,
) -> ReviewedSemanticEmbeddingContract:
    """Load the checked-in byte allowlist without touching a model cache."""
    from videoscope.benchmark.snapshots import (
        FastEmbedSnapshotManifest,
        load_reviewed_fastembed_snapshot_manifest,
    )

    try:
        manifest = load_reviewed_fastembed_snapshot_manifest()
    except Exception as error:
        raise ReviewedSemanticEmbeddingError(
            "reviewed semantic embedding manifest is unavailable"
        ) from error
    if type(manifest) is not FastEmbedSnapshotManifest:
        raise ReviewedSemanticEmbeddingError(
            "reviewed semantic embedding manifest is invalid"
        )
    if manifest.model_name != model_name or manifest.dimensions != dimensions:
        raise ReviewedSemanticEmbeddingError(
            "reviewed semantic embedding differs from validated settings"
        )
    embedding = SemanticEmbedding(
        model_name=manifest.model_name,
        model_repository=manifest.model_repository,
        model_revision=manifest.model_revision,
        expected_runtime_version=manifest.runtime_version,
        algorithm_version=manifest.algorithm_version,
        dimensions=manifest.dimensions,
    )
    if type(embedding) is not SemanticEmbedding:
        raise ReviewedSemanticEmbeddingError(
            "reviewed semantic embedding concrete type is unavailable"
        )
    return ReviewedSemanticEmbeddingContract(
        model_name=manifest.model_name,
        model_repository=manifest.model_repository,
        model_revision=manifest.model_revision,
        runtime_version=manifest.runtime_version,
        algorithm_version=manifest.algorithm_version,
        dimensions=manifest.dimensions,
        embedding_identity=embedding.identity,
        model_content_sha256=manifest.model_content_sha256,
        semantic_embedding_type=SEMANTIC_EMBEDDING_TYPE_IDENTITY,
        manifest=manifest,
    )


@dataclass(slots=True)
class ReviewedSemanticEmbeddingRuntime:
    """Own one private verified snapshot for the lifetime of a writer."""

    embedding: SemanticEmbedding
    snapshot: object
    contract: ReviewedSemanticEmbeddingContract
    _scratch: TemporaryDirectory[str] = field(repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _close_lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        from videoscope.benchmark.snapshots import FastEmbedSnapshot

        if (
            type(self.embedding) is not SemanticEmbedding
            or type(self.snapshot) is not FastEmbedSnapshot
            or type(self.contract) is not ReviewedSemanticEmbeddingContract
            or self.embedding.attestation_identity
            != self.contract.embedding_attestation_dict
        ):
            raise ValueError("reviewed semantic embedding runtime is invalid")

    @property
    def closed(self) -> bool:
        return self._closed

    def verify_current(self) -> bool:
        if self._closed:
            return False
        return self.embedding.verify_benchmark_model_current()

    def close(self) -> bool:
        with self._close_lock:
            if self._closed:
                return False
            self.embedding.close()
            self._scratch.cleanup()
            self._closed = True
            return True


def create_reviewed_semantic_embedding_runtime(
    *,
    model_name: str,
    dimensions: int,
    cache_dir: Path,
    scratch_parent: Path,
    contract: ReviewedSemanticEmbeddingContract | None = None,
) -> ReviewedSemanticEmbeddingRuntime:
    """Materialize, warm, and finally attest one offline production snapshot."""
    from videoscope.benchmark.snapshots import (
        FastEmbedSnapshot,
        materialize_fastembed_snapshot,
    )

    resolved_contract = contract or load_reviewed_semantic_embedding_contract(
        model_name=model_name,
        dimensions=dimensions,
    )
    if (
        type(resolved_contract) is not ReviewedSemanticEmbeddingContract
        or resolved_contract.model_name != model_name
        or resolved_contract.dimensions != dimensions
    ):
        raise ReviewedSemanticEmbeddingError(
            "reviewed semantic embedding contract is invalid"
        )
    source = fastembed_cache_snapshot_path(cache_dir, model_name)
    if source is None:
        raise ReviewedSemanticEmbeddingError(
            "reviewed semantic embedding cache is unavailable"
        )
    scratch: TemporaryDirectory[str] | None = None
    embedding: object | None = None
    try:
        scratch = TemporaryDirectory(
            prefix=".videoscope-fastembed-",
            dir=Path(scratch_parent).absolute(),
        )
        os.chmod(scratch.name, 0o700)
        snapshot = materialize_fastembed_snapshot(
            source,
            Path(scratch.name),
            resolved_contract.manifest,
        )
        if type(snapshot) is not FastEmbedSnapshot:
            raise ValueError("reviewed FastEmbed materializer returned an invalid snapshot")
        embedding = snapshot.create_embedding()
        if (
            type(embedding) is not SemanticEmbedding
            or embedding.attestation_identity
            != resolved_contract.embedding_attestation_dict
            or embedding.ensure_ready() is not True
            or embedding.verify_benchmark_model_current() is not True
        ):
            raise ValueError("reviewed FastEmbed runtime attestation is invalid")
        return ReviewedSemanticEmbeddingRuntime(
            embedding=embedding,
            snapshot=snapshot,
            contract=resolved_contract,
            _scratch=scratch,
        )
    except BaseException as error:
        traceback.clear_frames(error.__traceback__)
        if type(embedding) is SemanticEmbedding:
            embedding.close()
        if scratch is not None:
            scratch.cleanup()
        if isinstance(error, Exception):
            raise ReviewedSemanticEmbeddingError(
                "reviewed semantic embedding runtime is unavailable"
            ) from error
        raise


def create_semantic_embedding(
    *,
    model_name: str,
    dimensions: int,
    cache_dir: Path,
) -> SemanticEmbedding:
    repository, revision = fastembed_snapshot(model_name)
    return SemanticEmbedding(
        model_name=model_name,
        model_repository=repository,
        model_revision=revision,
        expected_runtime_version=FASTEMBED_RUNTIME_VERSION,
        algorithm_version=FASTEMBED_ALGORITHM_VERSION,
        dimensions=dimensions,
        cache_dir=cache_dir,
    )
