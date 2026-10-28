"""Embeddings densos sin dependencias pesadas.

`HashingEmbedder` es un embedder lexico de hashing firmado (unigramas +
bigramas, tf sublineal, L2 normalizado). No necesita descargas ni GPU y es
determinista entre procesos, asi que el indice se puede reconstruir igual.

`SentenceTransformerEmbedder` es opcional: si `sentence-transformers` esta
instalado, se puede usar un modelo denso real (`--embedder st:modelo`).
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


@runtime_checkable
class Embedder(Protocol):
    """Interfaz minima de un embedder."""

    spec: str
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        ...


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


class HashingEmbedder:
    """Embedder lexico determinista basado en hashing firmado."""

    def __init__(self, dim: int = 512, ngrams: tuple[int, ...] = (1, 2)) -> None:
        if dim < 8:
            raise ValueError("dim debe ser >= 8")
        if not ngrams:
            raise ValueError("ngrams no puede estar vacio")
        self.dim = int(dim)
        self.ngrams = tuple(int(n) for n in ngrams)
        self.spec = f"hashing:{self.dim}"
        self._weights = {1: 1.0, 2: 0.5, 3: 0.25}

    def _hash(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        return value % self.dim, 1.0 if (value >> 63) & 1 else -1.0

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = _WORD_RE.findall(text.lower())
            if not tokens:
                continue
            counts: Counter[str] = Counter()
            for n in self.ngrams:
                weight = self._weights.get(n, 0.1)
                for k in range(len(tokens) - n + 1):
                    counts[" ".join(tokens[k : k + n])] += weight
            for gram, tf in counts.items():
                index, sign = self._hash(gram)
                vectors[row, index] += sign * (1.0 + math.log(tf))
        return _normalize_rows(vectors)


class SentenceTransformerEmbedder:
    """Embedder denso opcional basado en sentence-transformers."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depende del entorno
            raise ImportError(
                "sentence-transformers no esta instalado. "
                "Instala con `pip install sentence-transformers` o usa "
                "`--embedder hashing`."
            ) from exc

        self._model = SentenceTransformer(model_name, device=device)
        self.spec = f"st:{model_name}"
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def get_embedder(spec: str | None = None, *, dim: int = 512) -> Embedder:
    """Crea un embedder a partir de una especificacion.

    Formatos: `hashing`, `hashing:1024`, `st:modelo`, `sentence-transformers:modelo`.
    """
    spec = (spec or "hashing").strip()
    if spec in ("hashing", ""):
        return HashingEmbedder(dim)
    if spec.startswith("hashing:"):
        return HashingEmbedder(int(spec.split(":", 1)[1]))
    if spec.startswith("st:"):
        return SentenceTransformerEmbedder(spec.split(":", 1)[1])
    if spec.startswith("sentence-transformers:"):
        return SentenceTransformerEmbedder(spec.split(":", 1)[1])
    raise ValueError(
        f"embedder desconocido: {spec!r}. Usa `hashing[:dim]` o `st:modelo`."
    )
