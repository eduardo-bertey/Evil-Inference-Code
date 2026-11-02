import ctypes
import json
import os
import sys

from .agent.tools import Field, build_schema, pydantic_schema, tool, _is_pydantic_model

__version__ = "2.0.0"
__all__ = ["Needle", "tool", "Field", "extract", "__version__"]


def _library_path():
    from .agent import fetch

    here = os.path.dirname(os.path.abspath(__file__))
    local = os.path.join(here, fetch._lib_name())
    if os.path.exists(local):
        return local
    cache = os.path.join(os.path.expanduser("~"), ".cache", "cactus-needle", fetch.ENGINE_VERSION)
    cached = os.path.join(cache, fetch._lib_name())
    if os.path.exists(cached):
        return cached
    os.makedirs(cache, exist_ok=True)
    return fetch.fetch_library(fetch.ENGINE_VERSION, cache)


_lib_handle = None


def _lib():
    global _lib_handle
    if _lib_handle is None:
        lib = ctypes.CDLL(_library_path())
        lib.needle_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        lib.needle_init.restype = ctypes.c_int
        lib.needle_complete.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        lib.needle_complete.restype = ctypes.c_int
        lib.needle_reset.argtypes = []
        lib.needle_reset.restype = None
        lib.needle_load.argtypes = [ctypes.c_char_p, ctypes.c_uint64]
        lib.needle_load.restype = ctypes.c_int
        _lib_handle = lib
    return _lib_handle


class Needle:
    def __init__(self, tools=None, system=None, weights=None, tool_index_path=None, buffer_size=65536):
        self._functions = {}
        if weights:
            with open(weights, "rb") as handle:
                blob = handle.read()
            if _lib().needle_load(blob, len(blob)) != 0:
                raise RuntimeError(f"failed to load weights from {weights}")
        tools_json = tools if isinstance(tools, str) else json.dumps(self._resolve(tools))
        if _lib().needle_init((system or "").encode("utf-8"), tools_json.encode("utf-8"),
                              tool_index_path.encode("utf-8") if tool_index_path else None) < 0:
            raise RuntimeError("needle_init failed")
        self._buffer = ctypes.create_string_buffer(buffer_size)

    def _resolve(self, tools):
        schemas = []
        for entry in tools or []:
            if _is_pydantic_model(entry):
                schema = pydantic_schema(entry)
                self._functions[schema["name"]] = entry
                schemas.append(schema)
            elif callable(entry):
                schema = getattr(entry, "_needle_tool", None) or build_schema(entry)
                self._functions[schema["name"]] = entry
                schemas.append(schema)
            elif isinstance(entry, dict):
                schemas.append(entry)
        return schemas

    def complete(self, text, max_new_tokens=256):
        _lib().needle_complete(text.encode("utf-8"), int(max_new_tokens),
                               self._buffer, len(self._buffer))
        return json.loads(self._buffer.value.decode("utf-8"))

    def run(self, query, max_steps=8, max_new_tokens=256):
        response = self.complete(query, max_new_tokens)
        executed = []
        for _ in range(max_steps):
            calls = response.get("function_calls") or []
            if response.get("type") != "call" or not calls:
                break
            results = []
            for call in calls:
                fn = self._functions.get(call.get("name"))
                if fn is None:
                    results.append({"error": "unknown tool: " + str(call.get("name"))})
                    continue
                try:
                    results.append(fn(**(call.get("arguments") or {})))
                except Exception as exc:
                    results.append({"error": str(exc)})
            executed.extend(results)
            response = self.complete(json.dumps(results, default=_jsonable), max_new_tokens)
        response["results"] = executed
        return response

    def extract(self, text, schema, max_new_tokens=256):
        return extract(text, schema, max_new_tokens=max_new_tokens)

    def reset(self):
        _lib().needle_reset()


def _jsonable(value):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict") and _is_pydantic_model(type(value)):
        return value.dict()
    return str(value)


def extract(text, schema, system=None, max_new_tokens=256):
    """One-shot structured extraction: declare `schema` as the only tool and return
    the parsed object (a Pydantic instance if `schema` is a model, else a dict).
    Re-initializes the shared engine with this single schema."""
    agent = Needle(tools=[schema], system=system)
    response = agent.complete(text, max_new_tokens)
    calls = response.get("function_calls") or []
    if not calls:
        return None
    arguments = calls[0].get("arguments") or {}
    return schema(**arguments) if _is_pydantic_model(schema) else arguments
