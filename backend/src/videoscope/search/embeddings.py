from __future__ import annotations

import hashlib
from pathlib import Path
import re
import unicodedata
from collections.abc import Iterable

import numpy as np


class HashEmbedding:
    """Deterministic local fallback based on word and character features."""

    def __init__(self, dimensions: int = 384) -> None:
        if dimensions < 16:
            raise ValueError("embedding dimensions must be at least 16")
        self.dimensions = dimensions

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
    """FastEmbed with a deterministic offline fallback."""

    def __init__(
        self,
        *,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        dimensions: int = 384,
        cache_dir: Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.dimensions = dimensions
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._model = None
        self._attempted = False
        self._fallback = HashEmbedding(dimensions)
        self.last_error: str | None = None

    @property
    def backend(self) -> str:
        if self._model is not None:
            return "fastembed"
        return "hash-fallback" if self._attempted else "not-loaded"

    def ensure_ready(self) -> bool:
        return self._load() is not None

    def _load(self):  # type: ignore[no-untyped-def]
        if self._model is not None:
            return self._model
        if self._attempted:
            return None
        self._attempted = True
        try:
            from fastembed import TextEmbedding

            options: dict[str, object] = {"model_name": self.model_name}
            if self.cache_dir is not None:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                options["cache_dir"] = str(self.cache_dir)
            self._model = TextEmbedding(**options)
            return self._model
        except Exception as error:
            self.last_error = str(error)
            return None

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        model = self._load()
        if model is None:
            return self._fallback.embed(texts)
        try:
            vectors = [np.asarray(vector, dtype=np.float32) for vector in model.embed(texts)]
            if any(vector.shape != (self.dimensions,) for vector in vectors):
                raise ValueError("embedding model returned an unexpected vector size")
            return [vector / max(float(np.linalg.norm(vector)), 1e-12) for vector in vectors]
        except Exception as error:
            self.last_error = str(error)
            return self._fallback.embed(texts)

    def embed_query(self, query: str) -> np.ndarray:
        model = self._load()
        if model is None:
            return self._fallback.embed([query])[0]
        try:
            vector = np.asarray(next(iter(model.query_embed(query))), dtype=np.float32)
            if vector.shape != (self.dimensions,):
                raise ValueError("embedding model returned an unexpected vector size")
            return vector / max(float(np.linalg.norm(vector)), 1e-12)
        except Exception as error:
            self.last_error = str(error)
            return self._fallback.embed([query])[0]
