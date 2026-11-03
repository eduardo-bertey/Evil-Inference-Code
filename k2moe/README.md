# k2moe — fine-tune top-down por bloques de capas (FineWeb, para Colab)

Ejecuta **solo** esto:

```bash
python k2moe/train.py
```

y responde a lo que pregunta. Nada más.

## Qué pregunta

1. `Repo HF destino` [ScortexIA/laurelia] y `Rama` [k2moe]
2. **Token HF** (lo usa del entorno `HF_TOKEN` si está; si no, te lo pide por consola)
3. Precisión `n=f32` / `b=bf16` [b]
4. `Capas por bloque (desde arriba)` [2] — cuántas capas se entrenan a la vez, empezando por las de arriba
5. **Avance de bloque** — elige uno:
   - `l` → **loss por tiempo**: pide `Minutos por bloque antes de saltar` [10] y `Caida de avg_loss para saltar antes` [0.05] (avanza cuando el loss cae esa cantidad o se acaban esos minutos)
   - `s` → **steps elegidos**: pide `Steps por bloque` [200]
6. `Seq len` [1024], `Batch size` [1], `Grad accum` [8], `lr` [2e-5], `Steps` [1000]
7. `Block de datos` [guardado] — bloque de FineWeb donde empezar/continuar

Empieza entrenando el bloque SUPERIOR de capas (+ embeddings/norms/lm_head).
Al avanzar, descongela el siguiente bloque hacia abajo (acumulativo) hasta
cerrar todo el modelo.

## Continuar desde un checkpoint

Si ya existe `k2moe_checkpoint.pt` (local o en `ScortexIA/laurelia@k2moe`),
te pregunta `¿Continuar desde ahí? [s/n]` y retoma step/block/capas (y el
optimizer) desde ahí.

## Archivos

| Archivo | Qué hace |
|---|---|
| `train.py` | entry: prompts + loop de entrenamiento |
| `layers.py` | freeze top-down por bloques (de arriba hacia abajo) |
| `fineweb.py` | FineWeb2-HQ (`spa_Latn`) en bloques de ~4MB |
| `hf.py` | token/repo HF, uploads/downloads |
| `plot.py` | `train_history.json` + `plot_train_step_N.png` (loss, líneas rojas = saltos) |
| `checkpoint.py` | `k2moe_checkpoint.pt` + `k2moe_state.json`, push a HF |
| `d2dmoe.py` | núcleo D2DMoE (MIT) para convertir capas a MoE (opcional) |

## Checkpoints en HF

Cada 20 min, en cada salto de bloque y al final se sube a `<repo>@<branch>`:
`k2moe_checkpoint.pt` (pesos+optimizer+estado), `k2moe_state.json`,
`train_history.json`, `plot_train_step_N.png`. Al final además el
`save_pretrained` completo con los `.py` del custom_code.

## Datos

FineWeb2-HQ español en streaming, bloques de ~4MB en `k2moe/k2moe_block_N.txt`
(ignorados por git).
