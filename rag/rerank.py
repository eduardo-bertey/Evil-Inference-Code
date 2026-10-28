"""Reranking de candidatos recuperados.

`LexicalReranker` funciona sin dependencias: combina cobertura IDF de los
terminos de la consulta con solape de bigramas (frases). Para calidad tipo
paper, `CrossEncoderReranker` usa sentence-transformers si esta instalado.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol, Sequence, runtime_checkable

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _terms(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


@runtime_checkable
class Reranker(Protocol):
    spec: str

    def rerank(self, query: str, texts: Sequence[str]) -> list[float]:
        ...


class LexicalReranker:
    """Reranker lexico determinista (IDF + frases)."""

    spec = "lexical"

    def __init__(self, coverage_weight: float = 0.8) -> None:
        self.coverage_weight = float(coverage_weight)

    def rerank(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []

        query_terms = _terms(query)
        if not query_terms:
            return [0.0] * len(texts)

        docs = [_terms(text) for text in texts]
        doc_freq: Counter[str] = Counter()
        for terms in docs:
            doc_freq.update(set(terms))

        total = len(docs)
        default_idf = math.log(1.0 + (total + 0.5) / 0.5)
        idf = {
            term: math.log(1.0 + (total - count + 0.5) / (count + 0.5))
            for term, count in doc_freq.items()
        }

        query_set = set(query_terms)
        query_bigrams = set(zip(query_terms, query_terms[1:]))
        query_weight = sum(idf.get(term, default_idf) for term in query_set) or 1.0

        scores: list[float] = []
        for terms in docs:
            doc_set = set(terms)
            coverage = sum(idf.get(term, 0.0) for term in query_set & doc_set) / query_weight
            if query_bigrams:
                doc_bigrams = set(zip(terms, terms[1:]))
                phrase = len(query_bigrams & doc_bigrams) / len(query_bigrams)
            else:
                phrase = 0.0
            scores.append(
                self.coverage_weight * coverage + (1.0 - self.coverage_weight) * phrase
            )
        return scores


class CrossEncoderReranker:
    """Reranker cross-encoder opcional (sentence-transformers)."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - depende del entorno
            raise ImportError(
                "sentence-transformers no esta instalado. "
                "Instala con `pip install sentence-transformers` o usa "
                "`--reranker lexical`."
            ) from exc

        self._model = CrossEncoder(model_name, device=device)
        self.spec = f"cross:{model_name}"

    def rerank(self, query: str, texts: Sequence[str]) -> list[float]:
        pairs = [(query, text) for text in texts]
        raw = self._model.predict(pairs)
        # Normaliza a [0, 1] con sigmoide; funciona con logits y con scores.
        return [1.0 / (1.0 + math.exp(-float(value))) for value in raw]


def get_reranker(spec: str | None) -> Reranker | None:
    """Crea un reranker: `lexical`, `none`, `cross:modelo`."""
    spec = (spec or "lexical").strip()
    if spec in ("none", "off", "false", "0"):
        return None
    if spec == "lexical":
        return LexicalReranker()
    if spec.startswith("cross:"):
        return CrossEncoderReranker(spec.split(":", 1)[1])
    raise ValueError(
        f"reranker desconocido: {spec!r}. Usa `lexical`, `none` o `cross:modelo`."
    )
