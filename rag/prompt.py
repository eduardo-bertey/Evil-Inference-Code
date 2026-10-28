"""Construccion del prompt de lectura a partir de la evidencia de RAW-RAG."""

from __future__ import annotations

from typing import Any, Sequence

DEFAULT_INSTRUCTIONS = (
    "Responde la pregunta usando unicamente la evidencia de abajo. "
    "Si la evidencia no alcanza para responder, dilo explicitamente. "
    "Cita cada dato con su numero de evidencia, por ejemplo [2]."
)


def _header(number: int, hit: Any) -> str:
    path = getattr(hit, "path", None) or "fragmento"
    header = f"[{number}] {path}"
    if getattr(hit, "chunk_index", None) is not None:
        header += f" (chunk {hit.chunk_index})"
    return header


def _content(hit: Any) -> str:
    content = getattr(hit, "content", None)
    if content is None:
        content = getattr(hit, "text", "")
    return str(content).strip()


def format_evidence(
    hits: Sequence[Any],
    *,
    max_chars: int = 6000,
    per_item_chars: int | None = None,
) -> str:
    """Formatea la evidencia respetando un presupuesto de caracteres."""
    blocks: list[str] = []
    used = 0

    for number, hit in enumerate(hits, start=1):
        body = _content(hit)
        if not body:
            continue
        if per_item_chars is not None and len(body) > per_item_chars:
            body = body[:per_item_chars].rstrip() + "..."

        block = f"{_header(number, hit)}\n{body}"
        remaining = max_chars - used
        if remaining <= 0:
            break
        if len(block) > remaining:
            if remaining < 120:
                break
            block = block[:remaining].rstrip() + "..."
        blocks.append(block)
        used += len(block) + 2

    text = "\n\n".join(blocks)
    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text


def build_prompt(
    question: str,
    hits: Sequence[Any],
    *,
    max_chars: int = 6000,
    per_item_chars: int | None = None,
    instructions: str = DEFAULT_INSTRUCTIONS,
) -> str:
    """Prompt completo (instrucciones + evidencia + pregunta) para el lector."""
    evidence = format_evidence(hits, max_chars=max_chars, per_item_chars=per_item_chars)
    if not evidence:
        evidence = "(sin evidencia recuperada)"
    return (
        f"{instructions}\n\n"
        f"### Evidencia\n{evidence}\n\n"
        f"### Pregunta\n{question.strip()}\n\n"
        f"### Respuesta\n"
    )


def estimate_tokens(text: str) -> int:
    """Aproximacion rapida de tokens (~4 caracteres por token)."""
    return max(1, len(text) // 4)
