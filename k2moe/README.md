# k2moe — fine-tune top-down por bloques de capas (FineWeb, para Colab)

Ejecuta **solo** esto:

```bash
python k2moe/train.py
```

y responde a lo que pregunta. Nada más.

## Qué pregunta

1. `Repo HF destino` [ScortexIA/laurelia] y `Rama` [k2moe]
2. **Token HF** (usa `HF_TOKEN` del entorno si está; si no, lo pide por consola)
3. Precisión `n=f32` / `b=bf16` [b]
4. **`Cuantas capas entrenar por turno (desde arriba)` [2]** — y **solo esas capas entrenan**: todo lo demás queda congelado siempre (otras capas densas o MoE ya convertidas, embeddings, norms, lm_head)
5. **Avance de bloque** — elige uno:
   - `l` → **loss por tiempo**: `Minutos por bloque antes de saltar` [10] y `Caida de avg_loss para saltar antes` [0.05]
   - `s` → **steps elegidos**: `Steps por bloque` [200]
6. **¿Aplicar d2dmoe progresivo?** [n] — si `s`, te pregunta: expertos por capa [32], swap SiLU/GELU→ReLU [s], y **router training por bloque** (steps [500], lr [1e-3], ancho [512])
7. `Seq len` [1024], `Batch size` [1], `Grad accum` [8], `lr` [2e-5], `Steps` [1000]
8. `Block de datos` [guardado]

## ¿Qué entrena?

- **Turno actual**: solo las N capas del bloque activo (de arriba hacia abajo) + nada más. Las demás: congeladas, sean densas o MoE ya convertidas.
- **d2dmoe progresivo**: el bloque que entra se convierte dense→MoE *en ese momento* (no todo junto), luego entrena sus routers (resto del modelo congelado, MSE contra ‖E_i(x)‖, como el paper) y después sus pesos completos en el turno.
- **Al avanzar**: el bloque anterior se congela del todo y entra el siguiente.
- **Resume**: la receta MoE (`arch`: expertos, swap_relu, bloques convertidos) viaja en el checkpoint → se reaplica sola y sigue donde quedaste.

Files:
| Archivo | Qué hace |
|---|---|
| `train.py` | entry: prompts + loop |
| `layers.py` | freeze: solo el bloque actual entrena |
| `fineweb.py` | FineWeb2-HQ (`spa_Latn`) en bloques de ~4MB |
| `hf.py` | token/repo HF, uploads/downloads |
| `plot.py` | `train_history.json` + plot (loss, líneas rojas = saltos) |
| `checkpoint.py` | checkpoint + state, push a HF |
| `d2dmoe.py` | dense→MoE (k-means de gate_proj) + routers + router training |

## Continuar desde un checkpoint

Si existe `k2moe_checkpoint.pt` (local o en HF), pregunta `¿Continuar desde ahí? [s/n]` y retoma step/block/capas + optimizer.

## Datos

FineWeb2-HQ español en streaming, bloques de ~4MB en `k2moe/k2moe_block_N.txt` (ignorados por git).
