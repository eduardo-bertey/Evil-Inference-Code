"""Carga de documentos desde archivos, carpetas y conversaciones JSONL/JSON."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst",
    ".py", ".rs", ".ts", ".tsx", ".js", ".jsx",
    ".toml", ".cfg", ".ini", ".yaml", ".yml", ".jsonl", ".json",
    ".c", ".h", ".cc", ".cpp", ".cu", ".cuh", ".sh", ".ps1", ".sql",
}

DEFAULT_EXCLUDES = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    "env", ".rag_index", ".mypy_cache", ".pytest_cache", ".ipynb_checkpoints",
    "dist", "build", "target", ".idea", ".vscode", ".tox", "vendor",
}

TEXT_KEYS = (
    "text", "content", "message", "utterance", "observation",
    "raw", "body", "answer", "summary", "value",
)
SPEAKER_KEYS = ("speaker", "role", "author", "name", "user")
TIMESTAMP_KEYS = ("timestamp", "ts", "time", "date", "created_at", "dia_id", "session_id")


@dataclass(frozen=True)
class Document:
    """Unidad de indexacion: texto crudo + identificador estable."""

    text: str
    doc_id: str
    timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _first(mapping: dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def serialize_turns(turns: Iterable[Any]) -> str:
    """Serializa turnos de conversacion preservando hablante y timestamp."""
    lines: list[str] = []
    for turn in turns:
        if isinstance(turn, str):
            lines.append(turn.strip())
            continue
        if not isinstance(turn, dict):
            lines.append(str(turn))
            continue

        text = _first(turn, TEXT_KEYS)
        if text is None:
            # Algunos formatos anidan el contenido en listas de partes.
            parts = turn.get("parts") or turn.get("segments")
            if isinstance(parts, list):
                text = " ".join(
                    str(part.get("text", part)) if isinstance(part, dict) else str(part)
                    for part in parts
                )
        if text is None:
            continue

        speaker = _first(turn, SPEAKER_KEYS)
        timestamp = _first(turn, TIMESTAMP_KEYS)
        prefix = ""
        if timestamp:
            prefix += f"[{timestamp}] "
        if speaker:
            prefix += f"{speaker}: "
        lines.append(f"{prefix}{text}")
    return "\n".join(line for line in lines if line.strip())


def _record_to_document(record: Any, doc_id: str) -> Document | None:
    if isinstance(record, str):
        return Document(text=record, doc_id=doc_id) if record.strip() else None
    if not isinstance(record, dict):
        return None

    for container in ("turns", "messages", "conversation", "dialogue", "history"):
        value = record.get(container)
        if isinstance(value, list):
            text = serialize_turns(value)
            if not text.strip():
                return None
            timestamp = _first(record, TIMESTAMP_KEYS)
            return Document(
                text=text,
                doc_id=doc_id,
                timestamp=str(timestamp) if timestamp is not None else None,
                metadata={"kind": container},
            )

    text = _first(record, TEXT_KEYS)
    if text is None:
        text = json.dumps(record, ensure_ascii=False)
    if not str(text).strip():
        return None

    timestamp = _first(record, TIMESTAMP_KEYS)
    metadata = {
        key: value
        for key, value in record.items()
        if isinstance(value, (str, int, float, bool)) and key not in TEXT_KEYS
    }
    return Document(
        text=str(text),
        doc_id=doc_id,
        timestamp=str(timestamp) if timestamp is not None else None,
        metadata=metadata,
    )


def _load_jsonl(path: Path, doc_id: str) -> list[Document]:
    documents: list[Document] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            document = _record_to_document(record, f"{doc_id}#{number:05d}")
            if document is not None:
                documents.append(document)
    return documents


def _load_json(path: Path, doc_id: str) -> list[Document]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        for key in ("documents", "data", "records", "items", "questions"):
            if isinstance(data.get(key), list):
                items = data[key]
                break
        else:
            items = [data]
    else:
        return []

    documents: list[Document] = []
    for number, record in enumerate(items):
        document = _record_to_document(record, f"{doc_id}#{number:05d}")
        if document is not None:
            documents.append(document)
    return documents


def iter_files(
    paths: Iterable[str | Path],
    *,
    extensions: set[str] | None = None,
    excludes: set[str] = DEFAULT_EXCLUDES,
    max_file_bytes: int = 2_000_000,
) -> Iterable[Path]:
    """Itera archivos de texto de las rutas dadas, saltando ruido y binarios."""
    allowed = {ext.lower() for ext in (extensions or TEXT_EXTENSIONS)}

    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file():
            yield path
            continue
        if not path.is_dir():
            continue

        for candidate in sorted(path.rglob("*")):
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() not in allowed:
                continue
            relative_parts = candidate.relative_to(path).parts
            if any(part in excludes for part in relative_parts):
                continue
            try:
                if candidate.stat().st_size > max_file_bytes:
                    continue
            except OSError:
                continue
            yield candidate


def load_documents(
    paths: Iterable[str | Path],
    *,
    extensions: set[str] | None = None,
    excludes: set[str] = DEFAULT_EXCLUDES,
    max_file_bytes: int = 2_000_000,
) -> list[Document]:
    """Carga todos los documentos de texto de `paths` (archivos o carpetas)."""
    documents: list[Document] = []
    for path in iter_files(
        paths, extensions=extensions, excludes=excludes, max_file_bytes=max_file_bytes
    ):
        doc_id = Path(os.path.relpath(path)).as_posix()
        try:
            if path.suffix.lower() == ".jsonl":
                documents.extend(_load_jsonl(path, doc_id))
            elif path.suffix.lower() == ".json":
                documents.extend(_load_json(path, doc_id))
            else:
                text = path.read_text(encoding="utf-8")
                if text.strip():
                    documents.append(Document(text=text, doc_id=doc_id))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
    return documents
