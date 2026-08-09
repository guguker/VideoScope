from __future__ import annotations

import hashlib
from importlib import metadata
from pathlib import Path
import re
import unicodedata
from collections.abc import Iterable

import numpy as np

from videoscope.model_manifest import (
    FASTEMBED_ALGORITHM_VERSION,
    FASTEMBED_RUNTIME_VERSION,
    fastembed_snapshot,
    model_identity,
)


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
    ) -> None:
        self.model_name = model_name
        self.model_repository = model_repository
        self.model_revision = model_revision
        self.expected_runtime_version = expected_runtime_version
        self.algorithm_version = algorithm_version
        self.dimensions = dimensions
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._model = None
        self._attempted = False
        self._validated = False
        self._fallback = HashEmbedding(dimensions)
        self.last_error: str | None = None

    @property
    def backend(self) -> str:
        if self._model is not None:
            return "fastembed"
        return "hash-fallback" if self._attempted else "not-loaded"

    @property
    def identity(self) -> str:
        source = model_identity(
            self.model_repository or self.model_name,
            self.model_revision,
        )
        runtime = self.expected_runtime_version or "unmanaged"
        return (
            f"fastembed@{runtime}:{self.algorithm_version}:"
            f"{self.model_name}:{source}:{self.dimensions}"
        )

    def ensure_ready(self) -> bool:
        if self._validated:
            return True
        model = self._load()
        if model is None:
            return False
        try:
            probe = next(iter(model.embed(["VideoScope embedding dimension probe"])))
            self._normalize_vector(probe)
        except Exception as error:
            self.last_error = str(error)
            self._model = None
            return False
        self._validated = True
        return True

    def _load(self):  # type: ignore[no-untyped-def]
        if self._model is not None:
            return self._model
        if self._attempted:
            return None
        self._attempted = True
        try:
            if (
                self.expected_runtime_version is not None
                and metadata.version("fastembed") != self.expected_runtime_version
            ):
                raise RuntimeError(
                    f"fastembed {self.expected_runtime_version} is required"
                )
            from fastembed import TextEmbedding

            options: dict[str, object] = {"model_name": self.model_name}
            if self.cache_dir is not None:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                options["cache_dir"] = str(self.cache_dir)
            if self.model_repository and self.model_revision:
                from huggingface_hub import snapshot_download

                snapshot = snapshot_download(
                    self.model_repository,
                    revision=self.model_revision,
                    local_files_only=True,
                    cache_dir=str(self.cache_dir) if self.cache_dir is not None else None,
                )
                options["specific_model_path"] = str(snapshot)
            self._model = TextEmbedding(**options)
            return self._model
        except Exception as error:
            self.last_error = str(error)
            return None

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        model = self._load()
        if model is None:
            return self._fallback.embed(texts)
        try:
            vectors = [self._normalize_vector(vector) for vector in model.embed(texts)]
            if len(vectors) != len(texts):
                raise ValueError("embedding model returned an unexpected vector count")
            return vectors
        except Exception as error:
            self.last_error = str(error)
            raise RuntimeError("semantic embedding failed") from error

    def embed_query(self, query: str) -> np.ndarray:
        model = self._load()
        if model is None:
            return self._fallback.embed([query])[0]
        try:
            return self._normalize_vector(next(iter(model.query_embed(query))))
        except Exception as error:
            self.last_error = str(error)
            raise RuntimeError("semantic query embedding failed") from error

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
