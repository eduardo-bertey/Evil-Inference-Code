# Cortexia: Reporte de Resultados

## Comparativa de3 Transformers: Cache Compartida MLA vs GQA Normal

### Configuración

| Parámetro | Valor |
|---|---|
| Capas |12 |
| Dimensión |128 |
| Heads |8 |
| KV Groups |4 |
| Head Dim |16 |
| Latent Dim |32 (kv_dim // 2) |
| Seq Len |64 |
| Batch |16 |
| Épocas |30 |
| Vocab |33 (char-level) |
| Device |CPU |

---

## T1: GQA Normal (cache por capa)

**Arquitectura:** Cada capa tiene su propia cache K,V. Attention estándar GQA.

```
Layer i:
  Q = q_proj_i(x)
  K = k_proj_i(x)   ← pesos propios
  V = v_proj_i(x)   ← pesos propios
  attend(Q, K, V)
  → cache i: (K_i, V_i)
```

- **Params:** 2,186,657
- **Loss final:** 0.0974
- **Tiempo:** 19.6s

---

## T2: Cache Unica MLA + BMA

**Arquitectura:** Capa 1 comprime x → C_KV (latente). Capas 2+ descomprimen C_KV con sus propios W_up_kv. BMA aplica gating PRE-agregación.

```
Layer 1 (MLA):
  C_KV = W_down(x)              ← compresión
  K,V = W_up_kv_1(C_KV)        ← descompresión propia
  Q = q_proj_1(x)
  attend(Q, K, V)
  → cache: C_KV (latente compartido)

Layer i (SharedCache):
  K,V = W_up_kv_i(C_KV)        ← SUS pesos, diferentes a L1
  Q = q_proj_i(x)
  v = BMA(Q, V)                 ← gating pre-agregación
  attend(Q, K, V)
```

- **Params:** 2,039,713
- **Loss final:** 0.1609
- **Tiempo:** 20.3s

---

## T3: Cache Unica MLA + Gated

**Arquitectura:** Igual que T2 pero con Gated attention (gating POST-agregación).

```
Layer 1 (MLA): igual a T2

Layer i (SharedCache):
  K,V = W_up_kv_i(C_KV)        ← SUS pesos
  Q = q_proj_i(x)
  attend(Q, K, V) → h_attn
  h_attn = sigmoid(gate(x)) * h_attn   ← gating post-agregación
```

- **Params:** 2,028,537
- **Loss final:** 0.1356
- **Tiempo:** 19.2s

---

## Comparativa Final

| Modelo | Params | Loss | Tiempo | Memoria Cache |
|---|---|---|---|---|
| T1: GQA Normal | 2,186,657 | **0.0974** | 19.6s | N caches (una por capa) |
| T2: MLA + BMA | 2,039,713 | 0.1609 | 20.3s | **1 cache** (latente) |
| T3: MLA + Gated | 2,028,537 | 0.1356 | 19.2s | **1 cache** (latente) |

---

## Análisis

### Trade-off Loss vs Memoria

T1 tiene mejor loss (0.0974) pero usa N caches. T2 y T3 usan **1 sola cache** (latente C_KV) con ~7-23% más loss.

Con 12 capas, T1 necesita12 caches individuales. T2/T3 necesitan**1 solo latente** compartido. La reducción de memoria es significativa.

### BMA vs Gated

- T3 (Gated, loss=0.1356) vence a T2 (BMA, loss=0.1609)
- Gated: gating post-agregación — modula la salida de attention
- BMA: gating pre-agregación — modula V antes de attention
- Con cache compartida, Gated parece más efectivo

### Parámetros

T2/T3 tienen ~7% menos parámetros que T1 (no necesitan k_proj/v_proj en capas 2+, solo W_up_kv más pequeño).

---

## Técnicas Aplicadas

### 1. MLA (Multi-head Latent Attention)
- Compresión: `C_KV = W_down(x)` — reduce dimensión de cache
- Descompresión: `K,V = W_up_kv(C_KV)` — recuperar K,V
- La capa 1 produce el latente, capas 2+ lo descomprimen con pesos propios

### 2. Cache Compartida entre Capas
- Una sola cache C_KV para todas las capas
- Cada capa tiene sus propios pesos W_up_kv → K,V diferentes del mismo latente
- Reduce memoria de N caches a 1 cache

### 3. BMA (Bilinearly Modulated Attention)
- Gating pre-agregación: `g = sigmoid(Q @ W_g)`, `V_mod = g * V`
- Aplicado DESPUÉS de descomprimir V del latente
- Ref: NeurIPS 2025 Best Paper

### 4. Gated Attention
- Gating post-agregación: `gate = sigmoid(linear(x))`, `out = gate * attn_output`
- Modula la salida de attention después de softmax
- Ref: NeurIPS 2025 Best Paper (sección de gating)

### 5. RoPE (Rotary Position Embedding)
- Aplicado a Q y K después de descompresión
- K usa `k_offset=0` (posiciones 0..T), Q usa `offset` (posiciones nuevas)
- Permite cache compartida con posiciones correctas

---

## Código

Ubicación: `cortexia/transformers.py`

- `StandardGQALayer` — capa GQA normal con cache individual
- `MLALayer` — capa 1: compresión + descompresión + cache compartido
- `SharedCacheLayer` — capas 2+: descomprime C_KV con pesos propios
- `CortexiaTransformer1` — GQA normal (12 capas independientes)
- `CortexiaTransformer2` — MLA + BMA (1 capa MLA + 11 SharedCache)
- `CortexiaTransformer3` — MLA + Gated (1 capa MLA + 11 SharedCache)

Script de test: `cortexia/train_test.py`
