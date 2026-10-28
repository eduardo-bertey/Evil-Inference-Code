"""RAG local sobre este repo usando el motor de RAW-RAG.

    python -m rag index . --out .rag_index
    python -m rag search "como funciona el kv cache"
    python -m rag prompt "que hace moe-plus" -k 6

El chunking, la recuperacion por coseno y el reranking son los de
https://github.com/Litmeb/RAW-RAG; solo se reemplazan los backends que en el
paper llaman a OpenAI/Cohere por embeddings y rerank locales.
"""

from .embeddings import Embedder, HashingEmbedder, SentenceTransformerEmbedder, get_embedder
from .loaders import Document, load_documents, serialize_turns
from .prompt import DEFAULT_INSTRUCTIONS, build_prompt, estimate_tokens, format_evidence
from .rawrag import Hit, RawRagIndex
from .rerank import CrossEncoderReranker, LexicalReranker, Reranker, get_reranker

__version__ = "0.2.0"

__all__ = [
    "CrossEncoderReranker",
    "DEFAULT_INSTRUCTIONS",
    "Document",
    "Embedder",
    "HashingEmbedder",
    "Hit",
    "LexicalReranker",
    "RawRagIndex",
    "Reranker",
    "SentenceTransformerEmbedder",
    "build_prompt",
    "estimate_tokens",
    "format_evidence",
    "get_embedder",
    "get_reranker",
    "load_documents",
    "serialize_turns",
    "__version__",
]
