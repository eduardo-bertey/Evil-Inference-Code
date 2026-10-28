"""Stub de `openai` para el camino 100% local.

El `RAGAdapter` vendorizado hace `from openai import AsyncOpenAI` a nivel de
modulo y construye dos clientes en `__init__` (embeddings y answer). Con
`LocalRAGAdapter` esos clientes nunca se usan: `_embed_texts` y
`_rerank_candidates` estan sobrescritos y no hay ninguna llamada HTTP.

Importar `openai` cuesta ~4 s en esta maquina (pydantic + providers), asi que
se reemplaza por un modulo minimo cuando embedder y reranker son locales.
Si algun dia se usa un backend que hable con la API real, `install()` no hace
nada (el modulo verdadero ya esta en `sys.modules` o se importa normalmente).
"""

from __future__ import annotations

import sys
import types
from typing import Any


class _BaseUrl:
    """Imita `openai._base_client.URL`: `str()` siempre termina en `/`."""

    def __init__(self, value: str) -> None:
        self._value = value

    def __str__(self) -> str:
        return self._value if self._value.endswith("/") else self._value + "/"

    def __repr__(self) -> str:
        return f"URL({str(self)!r})"


class _DisabledProxy:
    """Permite encadenar atributos (`client.chat.completions.create`) y falla al llamar."""

    def __init__(self, path: str) -> None:
        self._path = path

    def __getattr__(self, name: str) -> "_DisabledProxy":
        if name.startswith("__"):
            raise AttributeError(name)
        return _DisabledProxy(f"{self._path}.{name}")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            "cliente OpenAI deshabilitado: rag/ usa embeddings y rerank locales "
            f"(se intento usar '{self._path}')"
        )


class _DisabledClient:
    """Reemplazo de `AsyncOpenAI`: guarda la config y falla si se usa."""

    def __init__(
        self,
        *args: Any,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.api_key = api_key
        self.base_url = _BaseUrl(str(base_url or "http://127.0.0.1/v1"))
        self._kwargs = kwargs

    def __getattr__(self, name: str) -> _DisabledProxy:
        if name.startswith("__"):
            raise AttributeError(name)
        return _DisabledProxy(name)

    def close(self) -> None:
        return None


def install() -> None:
    """Registra un modulo `openai` minimo si el real todavia no fue importado."""
    if "openai" in sys.modules:
        return

    module = types.ModuleType("openai")
    module.AsyncOpenAI = _DisabledClient
    module.OpenAI = _DisabledClient
    module.__version__ = "0.0.0-local-stub"
    module.__doc__ = "stub de rag/_openai_stub.py (backend local, sin API)"
    sys.modules["openai"] = module
