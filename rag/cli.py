"""CLI del RAG (motor RAW-RAG): index / search / prompt / info."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .loaders import TEXT_EXTENSIONS, load_documents
from .prompt import build_prompt
from .rawrag import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_TOP_K,
    DEFAULT_RERANK_TOP_K,
    MAP_FILENAME,
    RawRagIndex,
)


def _normalize_extensions(values: Sequence[str] | None) -> set[str] | None:
    if not values:
        return None
    return {value if value.startswith(".") else f".{value}" for value in values}


def _cmd_index(args: argparse.Namespace) -> int:
    documents = load_documents(
        args.inputs,
        extensions=_normalize_extensions(args.ext),
        max_file_bytes=args.max_file_bytes,
    )
    if not documents:
        print("No se encontraron documentos de texto en las rutas dadas.", file=sys.stderr)
        return 2

    index = RawRagIndex.build(
        documents,
        args.out,
        embedder=args.embedder,
        dim=args.dim,
        reranker=args.reranker,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_top_k=args.embedding_top_k,
        rerank_top_k=args.rerank_top_k,
        quiet=args.quiet,
    )

    if not args.quiet:
        print(
            f"RAW-RAG index: {index.documents} documentos -> {len(index)} chunks "
            f"(chunk={args.chunk_size}/{args.chunk_overlap}, "
            f"embedder={index.meta.get('embedder')}, "
            f"reranker={index.meta.get('reranker')})"
        )
        print(f"Indice guardado en {Path(args.out).resolve()}")
    return 0


def _load(args: argparse.Namespace) -> RawRagIndex:
    return RawRagIndex.load(
        args.index,
        reranker=args.reranker,
        quiet=True,
    )


def _cmd_search(args: argparse.Namespace) -> int:
    index = _load(args)
    hits = index.search(args.query, top_k=args.top_k)

    if args.json:
        payload = [hit.to_json(max_chars=args.max_chars) for hit in hits]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if not hits:
        print("Sin resultados.")
        return 0

    for number, hit in enumerate(hits, start=1):
        print(f"[{number}] score={hit.score:.4f}  {hit.path}  (chunk {hit.chunk_index})")
        preview = hit.content.strip().splitlines()
        for line in preview[:6]:
            print(f"    {line[:160]}")
        if len(preview) > 6:
            print("    ...")
    return 0


def _cmd_prompt(args: argparse.Namespace) -> int:
    index = _load(args)
    hits = index.search(args.query, top_k=args.top_k)
    print(
        build_prompt(
            args.query,
            hits,
            max_chars=args.max_chars,
            per_item_chars=args.per_item_chars,
        )
    )
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    map_path = Path(args.index) / MAP_FILENAME
    if not map_path.exists():
        print(f"No hay indice en {args.index}", file=sys.stderr)
        return 2
    mapping = json.loads(map_path.read_text(encoding="utf-8"))
    print(json.dumps(mapping.get("meta", {}), ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rag",
        description="RAG local con el pipeline de RAW-RAG: indexar, recuperar y armar prompts.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    index = sub.add_parser("index", help="construir el indice desde archivos o carpetas")
    index.add_argument("inputs", nargs="+", help="archivos o carpetas a indexar")
    index.add_argument("--out", default=".rag_index", help="directorio del indice")
    index.add_argument("--embedder", default="hashing", help="hashing[:dim] | st:modelo")
    index.add_argument("--dim", type=int, default=512, help="dimension para hashing")
    index.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    index.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    index.add_argument("--embedding-top-k", type=int, default=DEFAULT_EMBEDDING_TOP_K)
    index.add_argument("--rerank-top-k", type=int, default=DEFAULT_RERANK_TOP_K)
    index.add_argument("--reranker", default="lexical", help="lexical | none | cross:modelo")
    index.add_argument(
        "--ext",
        action="append",
        help=f"extension a incluir (repetible). Por defecto: {sorted(TEXT_EXTENSIONS)}",
    )
    index.add_argument("--max-file-bytes", type=int, default=2_000_000)
    index.add_argument("--quiet", action="store_true")
    index.set_defaults(func=_cmd_index)

    search = sub.add_parser("search", help="recuperar fragmentos para una consulta")
    search.add_argument("query")
    search.add_argument("--index", default=".rag_index")
    search.add_argument("-k", "--top-k", type=int, default=8)
    search.add_argument("--json", action="store_true")
    search.add_argument("--max-chars", type=int, default=1200)
    search.add_argument("--reranker", default=None, help="override: lexical | none")
    search.set_defaults(func=_cmd_search)

    prompt = sub.add_parser("prompt", help="imprimir el prompt aumentado con evidencia")
    prompt.add_argument("query")
    prompt.add_argument("--index", default=".rag_index")
    prompt.add_argument("-k", "--top-k", type=int, default=6)
    prompt.add_argument("--max-chars", type=int, default=6000)
    prompt.add_argument("--per-item-chars", type=int, default=1200)
    prompt.add_argument("--reranker", default=None, help="override: lexical | none")
    prompt.set_defaults(func=_cmd_prompt)

    info = sub.add_parser("info", help="mostrar metadatos del indice")
    info.add_argument("--index", default=".rag_index")
    info.set_defaults(func=_cmd_info)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (ImportError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
