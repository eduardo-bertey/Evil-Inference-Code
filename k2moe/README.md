# k2moe — fine-tune top-down por bloques de capas (FineWeb, para Colab)

Ejecuta **solo** esto:

```bash
python k2moe/train.py
```

y responde a lo que pregunta. Nada más.

## Qué pregunta

1. `Token HF` (usa `HF_TOKEN` del entorno si está; si no, `getpass` enmascarado). Repo y rama son fijos: `ScortexIA/laurelia@k2moe`
2. Precisión `n=f32` / `b=bf16` [b]
3. **`Cuantas capas entrenar por turno (desde arriba)` [2]** — y **solo esas capas entrenan**: todo lo demás queda congelado siempre (otras capas densas o MoE ya convertidas, embeddings, norms, lm_head)
4. **Avance de bloque** — elige uno:
   - `l` → **loss por tiempo**: `Minutos por bloque antes de saltar` [10] y `Caida de avg_loss para saltar antes` [0.05]
   - `s` → **steps elegidos**: `Steps por bloque (salto de capa)` [200]  
   (el `Steps totales del run` [1000] del paso 6 es el tope total del entrenamiento, no el avance entre capas)
5. **¿Aplicar d2dmoe progresivo?** [n] — si `s`, te pregunta: expertos por capa [32], swap SiLU/GELU→ReLU [s], y **router training por bloque** (steps [500], lr [1e-3], ancho [512])
6. `Seq len` [1024], `Batch size` [1], `Grad accum` [8], `lr` [2e-5], `Steps totales del run` [1000]
7. `Block de datos` [guardado]

## ¿Qué entrena?

- **Turno actual**: solo las N capas del bloque activo (de arriba hacia abajo) + nada más. Las demás: congeladas, sean densas o MoE ya convertidas.
- **d2dmoe progresivo**: el bloque que entra se convierte dense→MoE *en ese momento* (no todo junto), luego entrena sus routers (resto del modelo congelado, MSE contra ‖E_i(x)‖, como el paper) y después sus pesos completos en el turno.
- **Al avanzar**: el bloque anterior se congela del todo y entra el siguiente.
- **Resume**: la receta MoE (`arch`: expertos, swap_relu, bloques convertidos) viaja en el checkpoint → se reaplica sola y sigue donde quedaste.

## ¿El denso queda o se va?

**Se va.** La capa convertida queda como `MoeficationMoE` en el modelo (el módulo denso se sobrescribe, no hay copia). Entrenamiento = modo `all` (denso-equivalente, gradientes estables); al guardar el modelo final, las capas MoE pasan a **`dynk`** (inferencia con router, Eq. 4 del paper, τ en la pregunta `Tau dynk`). Si capa 3 quedó MoE, infiere como MoE para siempre (el `forward_mode` va en el forward del módulo, y el checkpoint solo contiene pesos MoE).

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

## Requisitos (Colab)

**Ninguno.** `train.py` instala solo lo que falte (`transformers`, `datasets`, `huggingface_hub`, `k-means-constrained`) la primera vez. No toca torch/CUDA. Si quieres preinstalarlo a mano: `!bash k2moe/setup.sh`.

## Datos

FineWeb2-HQ español en streaming, bloques de ~4MB en `k2moe/k2moe_block_N.txt` (ignorados por git).
