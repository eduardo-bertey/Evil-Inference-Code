"""Motor RAG: usa el pipeline real de RAW-RAG sobre este repositorio.

Que se reutiliza de https://github.com/Litmeb/RAW-RAG (MIT):

- `RAGAdapter` de `methods/EverCore/evaluation/src/adapters/rag_adapter.py`
- chunking por tokens con `tiktoken` (gpt-4), `chunk_size=256`, `overlap=32`
- recuperacion por similitud coseno sobre embeddings (`embedding_top_k=128`)
- reranking de candidatos y seleccion final (`rerank_top_k=15`)
- formato de contexto y prompt de lectura

Que se reemplaza para poder correr local, sin claves ni datasets:

- embeddings de OpenAI -> `rag.embeddings.HashingEmbedder` (o `st:modelo`)
- rerank HTTP (Cohere/Jina) -> `rag.rerank.LexicalReranker` (o `cross:modelo`)

El repo se modela como una unica `Conversation` de RAW-RAG donde cada archivo
es un `Message`; asi el chunker y el buscador del paper funcionan sin cambios.
"""

from __future__ import annotations

import asyncio
import io
import json
import math
import re
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .embeddings import Embedder, get_embedder
from .loaders import Document
from .rerank import Reranker, get_reranker

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)

# Stopwords ES/EN: sin esto, "donde/activa/se" dominan el ranking.
STOPWORDS = {
    "que", "como", "donde", "cuando", "porque", "para", "por", "con", "sin",
    "los", "las", "del", "al", "una", "uno", "unos", "unas", "este", "esta",
    "esto", "estos", "estas", "ese", "esa", "esos", "esas", "hay", "muy",
    "mas", "pero", "sobre", "entre", "desde", "hasta", "todo", "toda",
    "the", "and", "for", "with", "from", "into", "that", "this", "these",
    "those", "are", "was", "were", "have", "has", "had", "does", "did",
    "how", "what", "where", "when", "why", "which", "who", "can", "could",
    "should", "would", "you", "your", "its", "not", "all", "any",
}

RAW_RAG_ROOT = Path(__file__).resolve().parent.parent / "vendor" / "RAW-RAG"
EVALUATION_ROOT = RAW_RAG_ROOT / "methods" / "EverCore"
CONVERSATION_ID = "repo"
INDEX_FILENAME = "rag_index.json"
EMBEDDINGS_FILENAME = "embeddings.npy"
MAP_FILENAME = "repo_map.json"

# Valores del paper: config/systems/rag.yaml
DEFAULT_CHUNK_SIZE = 256
DEFAULT_CHUNK_OVERLAP = 32
DEFAULT_EMBEDDING_TOP_K = 128
DEFAULT_RERANK_TOP_K = 15

if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))


@dataclass(frozen=True)
class Hit:
    """Fragmento recuperado por RAW-RAG, con su archivo de origen."""

    content: str
    score: float
    path: str | None = None
    chunk_index: int | None = None
    start_token: int | None = None
    end_token: int | None = None
    start_turn: int | None = None
    end_turn: int | None = None

    def to_json(self, max_chars: int | None = None) -> dict[str, Any]:
        content = self.content
        if max_chars is not None and len(content) > max_chars:
            content = content[:max_chars].rstrip() + "..."
        return {
            "path": self.path,
            "score": round(float(self.score), 4),
            "chunk_index": self.chunk_index,
            "start_token": self.start_token,
            "end_token": self.end_token,
            "content": content,
        }


def _adapter_config(
    *,
    chunk_size: int,
    chunk_overlap: int,
    embedding_top_k: int,
    rerank_top_k: int,
    rerank_enabled: bool,
) -> dict[str, Any]:
    """Config con la misma forma que config/systems/rag.yaml."""
    return {
        "chunking": {
            "strategy": "token",
            "expand_turns_on_display": False,
            "dedup_turns_on_display": False,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "tokenizer_model": "gpt-4",
        },
        "embedding": {
            "model": "local-hashing",
            "api_key": "local",
            "base_url": "http://127.0.0.1/v1",
            "batch_size": 128,
            "max_retries": 1,
        },
        "storage": {"backend": "local", "save_every_conversations": 1},
        "rerank": {
            "enabled": rerank_enabled,
            "model": "local-lexical",
            "api_key": "local",
            "base_url": "http://127.0.0.1",
            "endpoint": "/rerank",
            "timeout_seconds": 1,
            "max_retries": 1,
        },
        "search": {
            "embedding_top_k": embedding_top_k,
            "rerank_top_k": rerank_top_k,
            "sort_selected_chunks_by_time": False,
            "dynamic_top_k": {"enabled": False},
        },
        "llm": {
            "model": "local",
            "api_key": "local",
            "base_url": "http://127.0.0.1/v1",
            "temperature": 0,
            "max_tokens": 256,
        },
    }


_ADAPTER_CLASS: dict[bool, Any] = {}


def _local_adapter_class(stub_openai: bool = True):
    """Construye (una vez) la subclase local de `RAGAdapter`."""
    cached = _ADAPTER_CLASS.get(stub_openai)
    if cached is not None:
        return cached

    if stub_openai:
        from ._openai_stub import install as _install_openai_stub

        _install_openai_stub()

    from evaluation.src.adapters.rag_adapter import RAGAdapter

    class LocalRAGAdapter(RAGAdapter):
        """RAGAdapter con embeddings y rerank locales (sin HTTP)."""

        def __init__(
            self,
            config: dict,
            output_dir: Path,
            embedder: Embedder,
            reranker: Reranker | None,
            quiet: bool = False,
        ) -> None:
            self._local_embedder = embedder
            self._local_reranker = reranker
            if quiet:
                import contextlib

                with contextlib.redirect_stdout(io.StringIO()):
                    super().__init__(config, output_dir)
                from rich.console import Console

                self.console = Console(file=io.StringIO())
            else:
                super().__init__(config, output_dir)

        async def _embed_texts(
            self,
            texts: list[str],
            stage: str,
            conversation_id: str | None = None,
            question_id: str | None = None,
            metadata: dict[str, Any] | None = None,
        ) -> list[list[float]]:
            return self._local_embedder.encode(list(texts)).tolist()

        def _has_complete_embeddings(self, conv_index: dict[str, Any] | None) -> bool:
            if not conv_index:
                return False
            chunks = conv_index.get("chunks")
            embeddings = conv_index.get("embeddings")
            return bool(chunks) and embeddings is not None and len(embeddings) == len(chunks)

        def _get_local_embedding_candidates(
            self,
            chunks: list[dict[str, Any]],
            embeddings: Any,
            query_embedding: list[float],
        ) -> list[dict[str, Any]]:
            """Version vectorizada del coseno (la original recorre en Python)."""
            if not chunks:
                return []
            matrix = (
                embeddings
                if isinstance(embeddings, np.ndarray)
                else np.asarray(embeddings, dtype=np.float32)
            )
            if matrix.size == 0:
                return []

            query = np.asarray(query_embedding, dtype=np.float32)
            query_norm = float(np.linalg.norm(query)) or 1.0
            norms = np.linalg.norm(matrix, axis=1)
            norms[norms == 0] = 1.0
            similarities = (matrix @ (query / query_norm)) / norms

            depth = min(self.embedding_top_k, len(similarities))
            top = np.argpartition(-similarities, depth - 1)[:depth]
            top = top[np.argsort(-similarities[top], kind="stable")]

            return [
                {
                    "index": int(index),
                    "content": chunks[int(index)]["content"],
                    "chunk": chunks[int(index)],
                    "embedding_rank": rank,
                    "similarity_score": float(similarities[int(index)]),
                }
                for rank, index in enumerate(top, start=1)
            ]

        async def _rerank_candidates(
            self,
            query: str,
            candidates: list[dict[str, Any]],
            top_n: int,
            conversation_id: str | None = None,
            question_id: str | None = None,
        ) -> list[dict[str, Any]]:
            if not candidates:
                return []
            if self._local_reranker is None:
                return candidates[:top_n]

            scores = self._local_reranker.rerank(
                query, [candidate["content"] for candidate in candidates]
            )
            enriched = [
                {**candidate, "rerank_score": float(score)}
                for candidate, score in zip(candidates, scores)
            ]
            enriched.sort(key=lambda candidate: candidate["rerank_score"], reverse=True)
            return enriched[:top_n]

    _ADAPTER_CLASS[stub_openai] = LocalRAGAdapter
    return _ADAPTER_CLASS[stub_openai]


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _make_adapter(
    embedder: Embedder,
    reranker: Reranker | None,
    output_dir: Path,
    *,
    chunk_size: int,
    chunk_overlap: int,
    embedding_top_k: int,
    rerank_top_k: int,
    quiet: bool,
):
    adapter_class = _local_adapter_class()
    config = _adapter_config(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_top_k=embedding_top_k,
        rerank_top_k=rerank_top_k,
        rerank_enabled=reranker is not None,
    )
    return adapter_class(config, output_dir, embedder, reranker, quiet=quiet)


class _EmbeddingMatrix:
    """Lista perezosa respaldada por numpy (compatible con `if not embeddings`)."""

    __slots__ = ("matrix",)

    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix

    def __len__(self) -> int:
        return len(self.matrix)

    def __bool__(self) -> bool:
        return len(self.matrix) > 0

    def __getitem__(self, item: Any) -> Any:
        return self.matrix[item]

    def __array__(self, dtype: Any = None) -> np.ndarray:
        return np.asarray(self.matrix, dtype=dtype)


class RawRagIndex:
    """Indice de RAW-RAG aplicado a los archivos de un repositorio."""

    def __init__(
        self,
        adapter: Any,
        index: dict[str, Any],
        paths: Sequence[dict[str, Any]],
        meta: dict[str, Any],
    ) -> None:
        self.adapter = adapter
        self.index = index
        self.paths = list(paths)
        self.meta = dict(meta)
        self.reranker: Reranker | None = getattr(adapter, "_local_reranker", None)
        self._chunks: list[dict[str, Any]] | None = None
        self._lower: list[str] | None = None
        self._lengths: np.ndarray | None = None
        self._avgdl: float = 1.0
        self._weights_cache: dict[str, tuple[list[str], dict[str, float]]] = {}

    def __len__(self) -> int:
        conversation = self.index.get("conversations", {}).get(CONVERSATION_ID, {})
        return len(conversation.get("chunks", []))

    @property
    def documents(self) -> int:
        return len(self.paths)

    @classmethod
    def build(
        cls,
        documents: Iterable[Document],
        out_dir: str | Path,
        *,
        embedder: str = "hashing",
        dim: int = 512,
        reranker: str | None = "lexical",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        embedding_top_k: int = DEFAULT_EMBEDDING_TOP_K,
        rerank_top_k: int = DEFAULT_RERANK_TOP_K,
        quiet: bool = False,
    ) -> "RawRagIndex":
        from evaluation.src.core.data_models import Conversation, Message

        documents = list(documents)
        if not documents:
            raise ValueError("no hay documentos para indexar")

        messages = [
            Message(
                sender_id=document.doc_id,
                sender_name=document.doc_id,
                content=document.text,
                timestamp=_parse_timestamp(document.timestamp),
                metadata={"path": document.doc_id},
            )
            for document in documents
        ]
        conversation = Conversation(
            conversation_id=CONVERSATION_ID,
            messages=messages,
            metadata={"speaker_a": "repo", "speaker_b": "repo"},
        )

        embedder_obj = get_embedder(embedder, dim=dim)
        reranker_obj = get_reranker(reranker)
        out_path = Path(out_dir)
        adapter = _make_adapter(
            embedder_obj,
            reranker_obj,
            out_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_top_k=embedding_top_k,
            rerank_top_k=rerank_top_k,
            quiet=quiet,
        )

        index = asyncio.run(adapter.add([conversation]))

        # RAW-RAG guarda los embeddings como listas dentro del JSON. Para un
        # repo grande eso son cientos de MB: los pasamos a float32 en .npy.
        conv_index = index.get("conversations", {}).get(CONVERSATION_ID, {})
        raw_embeddings = conv_index.pop("embeddings", None)
        if raw_embeddings is not None and len(raw_embeddings) > 0:
            matrix = np.asarray(raw_embeddings, dtype=np.float32)
            np.save(out_path / EMBEDDINGS_FILENAME, matrix)
            conv_index["embeddings_file"] = EMBEDDINGS_FILENAME
            (out_path / INDEX_FILENAME).write_text(
                json.dumps(index, ensure_ascii=False), encoding="utf-8"
            )
            conv_index["embeddings"] = _EmbeddingMatrix(matrix)

        meta = {
            "engine": "RAW-RAG",
            "conversation_id": CONVERSATION_ID,
            "embedder": embedder_obj.spec,
            "reranker": reranker_obj.spec if reranker_obj else "none",
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding_top_k": embedding_top_k,
            "rerank_top_k": rerank_top_k,
            "built_at": datetime.now().astimezone().isoformat(),
        }
        paths = [
            {
                "path": document.doc_id,
                "timestamp": document.timestamp,
                "metadata": document.metadata,
            }
            for document in documents
        ]
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / MAP_FILENAME).write_text(
            json.dumps({"meta": meta, "paths": paths}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return cls(adapter, index, paths, meta)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        embedder: str | None = None,
        reranker: str | None = None,
        quiet: bool = True,
    ) -> "RawRagIndex":
        path = Path(path)
        index_path = path / INDEX_FILENAME
        map_path = path / MAP_FILENAME
        if not index_path.exists() or not map_path.exists():
            raise FileNotFoundError(
                f"no hay indice RAG en {path}. Crealo con "
                f"`python -m rag index . --out {path}`."
            )

        index = json.loads(index_path.read_text(encoding="utf-8"))
        mapping = json.loads(map_path.read_text(encoding="utf-8"))
        meta = mapping.get("meta", {})

        conv_index = index.get("conversations", {}).get(CONVERSATION_ID, {})
        embeddings_file = conv_index.get("embeddings_file")
        if embeddings_file:
            embeddings_path = path / str(embeddings_file)
            if embeddings_path.exists():
                conv_index["embeddings"] = _EmbeddingMatrix(np.load(embeddings_path))

        embedder_obj = get_embedder(embedder or meta.get("embedder"), dim=512)
        reranker_spec = reranker if reranker is not None else meta.get("reranker")

        adapter = _make_adapter(
            embedder_obj,
            get_reranker(reranker_spec),
            path,
            chunk_size=int(meta.get("chunk_size", DEFAULT_CHUNK_SIZE)),
            chunk_overlap=int(meta.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)),
            embedding_top_k=int(meta.get("embedding_top_k", DEFAULT_EMBEDDING_TOP_K)),
            rerank_top_k=int(meta.get("rerank_top_k", DEFAULT_RERANK_TOP_K)),
            quiet=quiet,
        )
        return cls(adapter, index, mapping.get("paths", []), meta)

    def search(self, query: str, *, top_k: int = 8) -> list[Hit]:
        """Híbrido: recuperación densa de RAW-RAG + candidatos léxicos exactos."""
        if not query.strip():
            return []
        top_k = max(1, int(top_k))
        self.adapter.rerank_top_k = max(top_k, 8)
        dense = asyncio.run(self.adapter.search(query, CONVERSATION_ID, self.index))

        candidates: dict[Any, Hit] = {}
        for item in dense.results:
            hit = self._hit_from_item(item)
            key = hit.chunk_index if hit.chunk_index is not None else hit.content[:64]
            candidates.setdefault(key, hit)

        terms, weights = self._query_weights(query)
        for hit in self._lexical_candidates(query, limit=max(top_k * 4, 24)):
            key = hit.chunk_index if hit.chunk_index is not None else hit.content[:64]
            candidates.setdefault(key, hit)

        ordered = list(candidates.values())
        if not ordered:
            return []

        bm25 = [
            self._bm25(hit.content.lower(), weights, average_length=self._avgdl)
            for hit in ordered
        ]
        bm25_max = max(bm25) if bm25 else 0.0
        if self.reranker is not None:
            rerank_scores = self.reranker.rerank(query, [hit.content for hit in ordered])
        else:
            rerank_scores = [0.0] * len(ordered)

        # Mezcla: BM25 global (capta identificadores raros) + rerank + bonus de ruta.
        rescored = [
            replace(
                hit,
                score=(
                    0.55 * (score_bm / bm25_max if bm25_max else 0.0)
                    + 0.30 * score_rerank
                    + 0.15 * self._path_bonus(terms, hit.path)
                ),
            )
            for hit, score_bm, score_rerank in zip(ordered, bm25, rerank_scores)
        ]
        rescored.sort(key=lambda hit: hit.score, reverse=True)
        return rescored[:top_k]

    def _query_weights(self, query: str) -> tuple[list[str], dict[str, float]]:
        cached = self._weights_cache.get(query)
        if cached is not None:
            return cached

        terms = [
            term
            for term in dict.fromkeys(_WORD_RE.findall(query.lower()))
            if len(term) >= 3 and term not in STOPWORDS
        ]
        total = len(self._chunk_list()) or 1
        if self._lower is None:
            self._lower = [
                str(chunk.get("content", "")).lower() for chunk in self._chunk_list()
            ]
            self._lengths = np.asarray(
                [max(1, len(text.split())) for text in self._lower], dtype=np.float32
            )
            self._avgdl = float(self._lengths.mean()) if len(self._lengths) else 1.0

        weights: dict[str, float] = {}
        for term in terms:
            df = sum(1 for text in self._lower if term in text)
            if df:
                weight = math.log(1.0 + total / df)
                # Identificadores raros (AdamW8bit, parámetros, símbolos) pesan más.
                if df <= 5:
                    weight *= 3.0
                elif df <= 25:
                    weight *= 1.6
                weights[term] = weight

        self._weights_cache[query] = (terms, weights)
        return terms, weights

    @staticmethod
    def _bm25(
        text: str,
        weights: dict[str, float],
        *,
        length: float | None = None,
        average_length: float = 1.0,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> float:
        """BM25 con normalización por longitud (evita que ganen chunks largos)."""
        if not weights:
            return 0.0
        dl = float(length if length is not None else max(1, len(text.split())))
        denominator_base = k1 * (1.0 - b + b * dl / (average_length or 1.0))

        score = 0.0
        for term, weight in weights.items():
            tf = text.count(term)
            if tf:
                score += weight * (tf * (k1 + 1.0)) / (tf + denominator_base)
        return score

    @staticmethod
    def _path_bonus(terms: list[str], path: str | None) -> float:
        if not path or not terms:
            return 0.0
        normalized = re.sub(r"[^a-z0-9]+", " ", path.lower())
        hits = sum(1 for term in terms if term in normalized)
        return hits / len(terms)

    def _hit_from_item(self, item: dict[str, Any]) -> Hit:
        metadata = item.get("metadata") or {}
        content = str(item.get("content", ""))
        start_turn = metadata.get("start_turn")
        end_turn = metadata.get("end_turn")
        return Hit(
            content=content,
            score=float(item.get("score", 0.0)),
            path=self._path_for(content, start_turn, end_turn),
            chunk_index=metadata.get("chunk_index"),
            start_token=metadata.get("start_token"),
            end_token=metadata.get("end_token"),
            start_turn=start_turn,
            end_turn=end_turn,
        )

    def _chunk_list(self) -> list[dict[str, Any]]:
        if self._chunks is None:
            conversation = self.index.get("conversations", {}).get(CONVERSATION_ID, {})
            self._chunks = list(conversation.get("chunks", []))
        return self._chunks

    def _lexical_candidates(self, query: str, *, limit: int) -> list[Hit]:
        """BM25 simplificado sobre todos los chunks (capta identificadores)."""
        chunks = self._chunk_list()
        if not chunks:
            return []

        terms, weights = self._query_weights(query)
        if not weights:
            return []

        scores = np.zeros(len(chunks), dtype=np.float32)
        for index, text in enumerate(self._lower or []):
            score = self._bm25(
                text,
                weights,
                length=(
                    float(self._lengths[index]) if self._lengths is not None else None
                ),
                average_length=self._avgdl,
            )
            if score:
                scores[index] = score

        if not scores.any():
            return []

        depth = min(max(1, limit), len(chunks))
        top = np.argpartition(-scores, depth - 1)[:depth]
        top = top[np.argsort(-scores[top], kind="stable")]

        hits: list[Hit] = []
        for index in top:
            if scores[index] <= 0:
                continue
            chunk = chunks[int(index)]
            content = str(chunk.get("content", ""))
            start_turn = chunk.get("start_turn")
            end_turn = chunk.get("end_turn")
            hits.append(
                Hit(
                    content=content,
                    score=float(scores[index]),
                    path=self._path_for(content, start_turn, end_turn),
                    chunk_index=chunk.get("chunk_index"),
                    start_token=chunk.get("start_token"),
                    end_token=chunk.get("end_token"),
                    start_turn=start_turn,
                    end_turn=end_turn,
                )
            )
        return hits

    def _path_for(
        self,
        content: str,
        start_turn: int | None,
        end_turn: int | None,
    ) -> str | None:
        """Atribuye el chunk al archivo dominante cuando cruza varios turnos."""
        if not isinstance(start_turn, int) or start_turn < 0:
            return None

        last = len(self.paths) - 1
        if last < 0:
            return None
        end = end_turn if isinstance(end_turn, int) else start_turn
        candidates = [
            self.paths[index].get("path")
            for index in range(start_turn, min(end, last) + 1)
        ]
        candidates = [path for path in candidates if path]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        # RAW-RAG prefija cada turno con "[fecha] archivo: "; el archivo que
        # aparece mas veces dentro del chunk es el que aporta mas contenido.
        counts = {path: content.count(f"{path}: ") for path in set(candidates)}
        best = max(counts, key=lambda path: (counts[path], -candidates.index(path)))
        return best if counts[best] > 0 else candidates[0]
