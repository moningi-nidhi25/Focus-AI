"""Embedding backends.

* ``sentence-transformers`` (default when installed): real semantic embeddings.
* ``lexical``: dependency-free hashed bag-of-stems vectors. Works offline, used as
  automatic fallback and in tests. It is weaker at synonyms/paraphrase.

Select with env ``EMBEDDING_BACKEND`` = auto | sentence-transformers | lexical and
``EMBEDDING_MODEL`` (default ``sentence-transformers/all-MiniLM-L6-v2``).
"""
import logging
import os
import zlib
from functools import lru_cache
from math import log
from typing import List, Tuple

import numpy as np

from app.ai.text_utils import analyze_tokens, content_bigrams, content_stems

logger = logging.getLogger("focusai.embeddings")


def _normalize_rows(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class Embedder:
    name: str = "base"
    dense: bool = False
    # cosine similarity range mapped to 0..1 relevance (below low -> 0, above high -> 1)
    calibration: Tuple[float, float] = (0.0, 1.0)

    def embed(self, texts: List[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def calibrate(self, cosine: float) -> float:
        low, high = self.calibration
        return float(min(1.0, max(0.0, (cosine - low) / (high - low))))


class LexicalEmbedder(Embedder):
    name = "lexical-hash-v1"
    dense = False
    calibration = (0.03, 0.40)

    def __init__(self, dims: int = 8192):
        self.dims = dims

    def _vector(self, text: str) -> np.ndarray:
        toks = analyze_tokens(text)
        counts: dict = {}
        for s in content_stems(toks):
            counts[s] = counts.get(s, 0) + 1
        for a, b in content_bigrams(toks):
            key = f"{a}_{b}"
            counts[key] = counts.get(key, 0) + 1
        v = np.zeros(self.dims, dtype=np.float32)
        for feat, c in counts.items():
            v[zlib.crc32(feat.encode("utf-8")) % self.dims] += 1.0 + log(c)
        return v

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dims), dtype=np.float32)
        return _normalize_rows(np.vstack([self._vector(t) for t in texts]))


class SentenceTransformerEmbedder(Embedder):
    dense = True
    calibration = (0.12, 0.55)  # tuned for MiniLM-style models; re-tune with scripts/eval_relevance.py

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # heavy import, keep lazy

        self.name = model_name
        self._model = SentenceTransformer(model_name)

    @property
    def st_model(self):
        """The underlying SentenceTransformer (read-only access for the token-attribution layer)."""
        return self._model

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    backend = os.getenv("EMBEDDING_BACKEND", "auto").lower()
    model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    if backend in ("auto", "sentence-transformers"):
        try:
            return SentenceTransformerEmbedder(model)
        except Exception as exc:  # not installed, offline, bad model name...
            if backend == "sentence-transformers":
                raise
            logger.warning("sentence-transformers unavailable (%s); using lexical fallback.", exc)
    return LexicalEmbedder()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
