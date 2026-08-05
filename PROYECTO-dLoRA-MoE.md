# PROYECTO dLoRA-MoE

Solo diseño. MoE sobre base `laurelia-llm` con expertos dLoRA (DoRA), ruteador estilo `moe-mla`, atención Q–K=V + Gated Attention G1, capas pares comparten K, y Spectral Modulation (LISA) en vez de AttnRes.

## 1. Visión

- Base: `laurelia-llm` (checkpoint en HF `ScortexIA/laurelia@laurelia-llm`).
- Atención: Q–K=V (BrainChip) + XSA (Apple) + G1-headwise (Gated Attention).
- Residual: aditivo estándar. Se elimina AttnRes (block mode) de `filosofia/model.py` (`block_attn_res`, softmax sobre bloques).
- LISA-SM: escalado espectral de Q/K por zona de profundidad (reemplaza el papel del residual de bloque).
- MoE: ruteador de `rust/moe-mla/moe.py` (solo router); expertos como dLoRA (DoRA).
- Capas pares comparten K de la capa anterior (referencia conceptual `rust/LLM_D3/LLM_2.py:136`, "shared RoPE Key").

## 2. Bloque de capa

```
x ──┬── ln1 ── Attention (Q–K=V + LISA-SM + G1) ──┐
    └────────────────────────────────────────────── + ── h
h ──┬── ln2 ── Router → top-k expertos DoRA ──────┐
    └────────────────────────────────────────────── + ── x'
```

Residual aditivo estándar (sin softmax sobre bloques).

### 2.1 Atención Q–K=V

- Un solo `kv_proj` (grupos), sin `v_proj`; `v = k` (rotado).
- Cache KV de un solo tensor (50% menos) — patrón de `filosofia/model.py`.
- XSA: `Z = Y - (Y·Vn).sum(-1,keepdim)·Vn`, `Vn = normalize(v,-1)`.

### 2.2 LISA Spectral Modulation (SM)

Reemplaza AttnRes. Escalado por capa de Q y K antes de SDPA:

```
alpha, beta, gamma = params_zona(layer_idx)
norm_q = 1 + gamma / log(trace_q + eps)
norm_k = 1 + gamma / log(trace_k + eps)
q_scaled = q * norm_q * alpha
k_scaled = k * norm_k * beta
```

- `trace_q = (q*q).sum(-1)` por head/token (Frobenius² por token). Corrección del código de LISA, que usa `trace_W_Q_2_query_states`/`trace_W_Q_2_key_states` indefinidos (`lisa/.../modeling_qwen2_5_vl.py:977-978`).
- Zonas adaptadas a 16 capas (LISA usa umbrales para ~40 capas):

| Zona | Capas | (α, β, γ) |
|------|-------|-----------|
| Preservation (shallow) | `< 6` | (0.9, 0.9, 0.3) |
| Interaction (mid) | `6–11` | (1.0, 1.1, 0.6) |
| Suppression (deep) | `≥ 12` | (1.2, 1.3, 1.0) |

### 2.3 Gated Attention G1 (headwise)

- Post-SDPA: `g = sigmoid(proj(Z))`, 1 scalar por head; `out = Z * g`.
- Referencia `cortexia/gated_attention.py`, G1-headwise en `report/attention_mechanisms_wiki.md:88-94`.

### 2.4 Capas pares comparten K

- `k_even = k_prev_odd` (misma capa con K compartido, sin `kv_proj` en capa par).
- Cache KV de 1 tensor, menos parámetros.

## 3. Ruteador MoE

Reuso de `rust/moe-mla/moe.py` (solo router):

- Router Linear + bias trick.
- z-loss opcional.
- top-k con capacity (C).
- shared experts opcional.

Expertos NO son MLP de moe-mla: son dLoRA (sección 4).

## 4. dLoRA = DoRA

DoRA (Weight-Decomposed Low-Rank):

```
W_eff = m * (W0 + ΔW) / ||W0 + ΔW||_F
ΔW = B·A / r
```

- Init: `A ~ N(0, σ)`, `B = 0`, `m = ||W0||_F`.
- Magnitud `m` aprendible; dirección normalizada.
- No existe LoRA/DoRA en el repo: se diseña desde cero.

## 5. Config propuesta

```
dim = 768
heads = 12
kv_groups = 4
layers = 16
ffn_dim = 3072
block_size = 1024
emb_num = 32000
rotary_pct = 0.25

# SM (LISA)
sm_eps = 1e-6
sm_zones = [(0.9,0.9,0.3), (1.0,1.1,0.6), (1.2,1.3,1.0)]  # shallow/mid/deep

# G1
g1_headwise = True

# MoE
num_experts = 8
top_k = 2
capacity = 1.25
z_loss = 0.001
shared_experts = 1

# DoRA
lora_rank = 64
lora_alpha = 64
```

## 6. Notas de ahorro

- K=V: −50% cache KV, sin `v_proj` (−3.1M vs `laurelia-llm`).
- Capas pares comparten K: menos parámetros y cache.
- DoRA: low-rank ΔW + magnitud full-rank (`m`), dirección normalizada.
- SM: sin costo de params; solo escalado por token (barato, diferenciable).
- AttnRes eliminado: se libera el stack de bloques previos de `filosofia` (~1.5GB bf16 en training).
