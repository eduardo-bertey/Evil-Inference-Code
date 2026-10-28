"""Tests del RAG local montado sobre RAW-RAG."""

from __future__ import annotations

import json

import numpy as np
import pytest

from rag import (
    Document,
    HashingEmbedder,
    Hit,
    RawRagIndex,
    build_prompt,
    format_evidence,
    get_embedder,
    get_reranker,
    load_documents,
)
from rag.loaders import serialize_turns
from rag.prompt import estimate_tokens


def test_hashing_embedder_es_determinista_y_normalizado() -> None:
    embedder = HashingEmbedder(dim=128)
    first = embedder.encode(["kv cache attention", "python training loop"])
    second = embedder.encode(["kv cache attention", "python training loop"])

    assert first.shape == (2, 128)
    assert np.allclose(first, second)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0)


def test_hashing_embedder_recupera_lo_relevante() -> None:
    texts = [
        "el kv cache guarda claves y valores para reutilizar atencion",
        "el entrenamiento usa adamw con precision mixta",
        "script de inferencia interactiva con tokenizer",
    ]
    embedder = HashingEmbedder(dim=256)
    vectors = embedder.encode(texts)
    query = embedder.encode(["kv cache atencion"])[0]

    assert int(np.argmax(vectors @ query)) == 0


def test_rawrag_indexa_y_recupera(tmp_path) -> None:
    documents = [
        Document(
            text="kv cache " * 30 + "reutiliza claves y valores en atencion " * 5,
            doc_id="laurelia-llm/model.py",
        ),
        Document(
            text="optimizador adamw precision mixta gradientes " * 10,
            doc_id="laurelia-llm/train.py",
        ),
        Document(
            text="tokenizer vocabulario bytes entrenamiento de texto " * 10,
            doc_id="laurelia-llm/dataset.py",
        ),
    ]
    index = RawRagIndex.build(
        documents, tmp_path / "idx", chunk_size=32, chunk_overlap=4, quiet=True
    )
    assert len(index) > 3
    assert index.documents == 3

    hits = index.search("kv cache atencion", top_k=3)
    assert hits
    assert any(hit.path == "laurelia-llm/model.py" for hit in hits)
    assert hits[0].content.strip()


def test_rawrag_persistencia(tmp_path) -> None:
    index = RawRagIndex.build(
        [
            Document(text="kv cache " * 40, doc_id="a.py"),
            Document(text="adamw optimizer " * 40, doc_id="b.py"),
        ],
        tmp_path / "idx",
        chunk_size=32,
        chunk_overlap=4,
        quiet=True,
    )
    loaded = RawRagIndex.load(tmp_path / "idx")
    assert len(loaded) == len(index)

    original = index.search("adamw", top_k=2)
    restored = loaded.search("adamw", top_k=2)
    assert [h.path for h in original] == [h.path for h in restored]


def test_rawrag_indice_inexistente(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        RawRagIndex.load(tmp_path / "nope")


def test_reranker_none_usa_coseno(tmp_path) -> None:
    index = RawRagIndex.build(
        [
            Document(text="kv cache atencion " * 20, doc_id="a.py"),
            Document(text="adamw optimizador " * 20 + "zetaunico " * 20, doc_id="b.py"),
        ],
        tmp_path / "idx",
        chunk_size=24,
        chunk_overlap=4,
        reranker="none",
        quiet=True,
    )
    hits = index.search("zetaunico", top_k=2)
    assert hits[0].path == "b.py"
    assert "zetaunico" in hits[0].content


def test_prompt_respeta_presupuesto() -> None:
    hits = [
        Hit(content="x" * 500, score=1.0, path=f"d{i}.py", chunk_index=i)
        for i in range(10)
    ]
    evidence = format_evidence(hits, max_chars=1200)
    assert len(evidence) <= 1200 + len("\n\n")

    prompt = build_prompt("que es esto?", hits, max_chars=800)
    assert "### Evidencia" in prompt
    assert "### Pregunta" in prompt
    assert "que es esto?" in prompt
    assert estimate_tokens(prompt) > 0


def test_loaders_jsonl_serializa_turnos(tmp_path) -> None:
    path = tmp_path / "chat.jsonl"
    record = {
        "turns": [
            {"speaker": "ana", "timestamp": "2026-01-02", "text": "hola"},
            {"speaker": "leo", "text": "que tal"},
        ]
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    documents = load_documents([tmp_path])
    assert len(documents) == 1
    assert "[2026-01-02] ana: hola" in documents[0].text
    assert "leo: que tal" in documents[0].text


def test_loaders_salta_directorios_excluidos(tmp_path) -> None:
    (tmp_path / "ok.txt").write_text("contenido", encoding="utf-8")
    ignored = tmp_path / "node_modules"
    ignored.mkdir()
    (ignored / "dep.txt").write_text("no deberia aparecer", encoding="utf-8")

    documents = load_documents([tmp_path])
    ids = {document.doc_id.replace("\\", "/") for document in documents}
    assert any(doc_id.endswith("ok.txt") for doc_id in ids)
    assert not any("node_modules" in doc_id for doc_id in ids)


def test_get_embedder_y_reranker() -> None:
    assert get_embedder("hashing:64").dim == 64
    assert get_reranker("none") is None
    with pytest.raises(ValueError):
        get_embedder("desconocido")
    with pytest.raises(ValueError):
        get_reranker("desconocido")


def test_serialize_turns() -> None:
    assert serialize_turns(["uno", {"role": "user", "content": "dos"}]) == "uno\nuser: dos"
