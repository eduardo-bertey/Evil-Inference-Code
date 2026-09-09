---
license: other
language:
- es
task_categories:
- text-generation
size_categories:
- 10K<n<100K
---

# data-fine-es

Bloques de pre-entrenamiento en español (~64MB c/u): conocimiento general + chat pelado.

*Spanish pre-training blocks (~64MB each): general knowledge + bare chat. [English below](#english).*

## Composición (por `data.{N}.txt`)

| Parte | Tamaño | Fuente |
|---|---|---|
| FineWeb2-HQ `spa_Latn` | ~54.9 MB (85.8%) | `epfml/FineWeb2-HQ` — conocimiento general |
| Tuits en español, limpios | ~9.1 MB (14.2%) | `pysentimiento/spanish-tweets` — chat |

Proporción según peso de cantera (500GB fine / 82.65GB tuits): ambas fuentes
se agotan juntas (~9,300 bloques). Sin wrap ni repetición.

Formato: cuerpo de corpus directo (FineWeb2-HQ + tuits).
Bloques de menos de 32MB se descartan (control de calidad, no se suben).

## Limpieza de tuits (chat pelado)

Cada tuit queda en texto de conversación, sin nada más:

- URLs eliminadas
- `@menciones` eliminadas
- `#hashtags` eliminados
- espacios extra colapsados

Queda chat en español pelado (jerga, typos y todo): sin metadatos ni enlaces.

## FineWeb2-HQ

Texto web español de calidad para conocimiento general. El grueso del bloque.

## Stats

- Tope: 9,000 bloques (agote ~9,322; sin wrap)
- ~64MB brutos por bloque (~63.9MB netos tras limpieza)
- Orden determinístico, índice base-0 (`data.1.txt` = primera ventana)

## Uso

```python
from datasets import load_dataset
ds = load_dataset("ScortexIA/data-fine-es", split="train", streaming=True)
print(next(iter(ds))["text"][:500])
```

## Fuentes y licencia

- FineWeb2-HQ: `epfml/FineWeb2-HQ` (ODC-By)
- Tuits: `pysentimiento/spanish-tweets`
- Armado con scripts de streaming (`part_data/`)

Revisar la licencia de cada fuente antes de uso comercial.

---

<a id="english"></a>
## English

Spanish pre-training blocks (~64MB each).

### Composition (per `data.{N}.txt`)

| Part | Size | Source |
|---|---|---|
| FineWeb2-HQ `spa_Latn` | ~54.9 MB (85.8%) | `epfml/FineWeb2-HQ` — general knowledge |
| Spanish tweets, cleaned | ~9.1 MB (14.2%) | `pysentimiento/spanish-tweets` — chat |

Proportions follow quarry weight (500GB fine / 82.65GB tweets): both sources
exhaust together (~9,300 blocks). No wrap, no repetition.

Layout: plain corpus body (FineWeb2-HQ + tweets).
Blocks under 32MB are discarded (quality gate, never uploaded).

### Tweet cleaning (bare chat)

Each tweet is stripped to raw conversation text: URLs, `@mentions` and
`#hashtags` removed, whitespace collapsed. Plain Spanish chat (slang, typos
and all) — no metadata, no links.

#### FineWeb2-HQ

High-quality Spanish web text for general knowledge. The bulk of every block.

### Stats

- Target: up to 9,000 blocks (exhaustion ~9,322; no wrap)
- ~64MB gross per block (~63.9MB net after cleaning)
- Deterministic order, base-0 indexing (`data.1.txt` = first window)

### Usage

```python
from datasets import load_dataset
ds = load_dataset("ScortexIA/data-fine-es", split="train", streaming=True)
print(next(iter(ds))["text"][:500])
```

### Sources & license

- FineWeb2-HQ: `epfml/FineWeb2-HQ` (ODC-By)
- Spanish tweets: `pysentimiento/spanish-tweets`
- Built with streaming scripts (`part_data/`)

Check each source's license before commercial use.
