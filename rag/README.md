# RAG sobre este repo (motor RAW-RAG)

Este paquete usa el pipeline real de [RAW-RAG](https://github.com/Litmeb/RAW-RAG)
(MIT) para indexar y recuperar evidencia de este repositorio.

## Qué usa de RAW-RAG

- `methods/EverCore/evaluation/src/adapters/rag_adapter.py` → `RAGAdapter`
- chunking por tokens con `tiktoken` (`gpt-4`), `chunk_size=256`, `overlap=32`
- recuperación por similitud coseno, `embedding_top_k=128`
- reranking y selección final, `rerank_top_k=15`
- formato de contexto/prompt de lectura

Valores tomados de `config/systems/rag.yaml` del repo original.

## Qué se reemplaza (para correr sin claves)

| En el paper | Acá |
| --- | --- |
| embeddings OpenAI (`text-embedding-3-small`) | `HashingEmbedder` (unigramas + bigramas, L2) |
| rerank HTTP (Cohere/Jina) | `LexicalReranker` (IDF + bigramas) |
| lector GPT/DeepSeek | no se usa: se genera el prompt y lo responde tu modelo |

Opcional, si instalás `sentence-transformers`:

```bash
python -m rag index . --embedder st:sentence-transformers/all-MiniLM-L6-v2 \
  --reranker cross:cross-encoder/ms-marco-MiniLM-L-6-v2
```

## Latencia

Medido en este repo (19.036 chunks), por invocación del CLI:

| Etapa | Tiempo |
| --- | --- |
| arranque de Python + imports | ~1,0 s |
| carga del índice (20 MB JSON + `embeddings.npy`) | ~1,9 s |
| búsqueda (densa + BM25 sobre 19k chunks) | ~0,3-0,7 s |
| **total** | **~3,4 s** |

El import de `openai` (~4 s en esta máquina) se evita con `rag/_openai_stub.py`:
el `RAGAdapter` lo importa a nivel de módulo, pero con backends locales nunca
usa el cliente. Si algún día se agrega un backend que llame a la API real, hay
que desactivar ese stub.

## Uso

```bash
# instalar dependencias del motor
pip install -r requirements-rag.txt

# construir el índice del repo (genera .rag_index/)
python -m rag index . --out .rag_index

# buscar
python -m rag search "como funciona el kv cache" -k 8
python -m rag search "configuracion del optimizador" --json

# prompt listo para el lector (tu modelo)
python -m rag prompt "que hace moe-plus" -k 6

# metadatos del índice
python -m rag info
```

El índice se guarda en `.rag_index/`:

- `rag_index.json` → formato nativo de RAW-RAG (chunks + embeddings)
- `repo_map.json` → metadatos y mapa `turno → archivo`

## Uso como herramienta de OpenCode

`.opencode/plugins/rag/index.ts` registra la herramienta `rag_search`, que
ejecuta `python -m rag search ... --json` en el directorio del proyecto. El
agente la usa para localizar evidencia antes de leer archivos completos.

Si el índice no existe, reconstruilo con el comando de arriba.

## Tests

```bash
python -m pytest tests/test_rag.py -q --noconftest
```

`--noconftest` es necesario porque `tests/conftest.py` salta todo sin GPU.

## Notas

- El repo se modela como una sola `Conversation` de RAW-RAG: cada archivo es un
  `Message`, así el chunker y el buscador del paper funcionan sin cambios.
- Un chunk puede cruzar el límite entre dos archivos; en ese caso se atribuye
  al archivo que aporta más contenido.
- `chunk_size`/`overlap` se pueden cambiar por CLI, pero los defaults son los
  del paper para que los resultados sean comparables.
- Para búsquedas conceptuales en español el ranking es más débil que un modelo
  denso; para nombres de funciones, flags y archivos gana el híbrido actual
  (BM25 con boost a términos raros + bonus por ruta).
