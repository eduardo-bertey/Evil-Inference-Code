# Wiki de Tecnologias: Mecanismos de Atencion

> **Fecha:** 2026-07-25  
> **Autor:** Evil Inference Code Team  
> **Repositorios analizados:** gated_attention, Bilinearly-Modulated-Attention

---

## Indice

1. [Resumen Ejecutivo](#resumen-ejecutivo)
2. [Gated Attention (NeurIPS 2025 Best Paper)](#1-gated-attention)
3. [Bilinearly Modulated Attention (BMA)](#2-bilinearly-modulated-attention)
4. [Comparativa Tecnica](#3-comparativa-tecnica)
5. [Diagramas de Arquitectura](#4-diagramas-de-arquitectura)
6. [Codigo Relevante](#5-codigo-relevante)
7. [Aplicacion en Evil Inference Code](#6-aplicacion-en-evil-inference-code)
8. [Referencias](#7-referencias)

---

## Resumen Ejecutivo

Este reporte analiza dos mecanismos de atencion innovadores que buscan mejorar la atencion estandar del Transformer:

| Mecanismo | Siguiente donde | Paper | Estado |
|-----------|-----------------|-------|--------|
| **Gated Attention** | Despues de SDPA (post-aggregation) | arXiv 2505.06708 | NeurIPS 2025 Best Paper |
| **BMA** | Antes de la agregacion (pre-aggregation) | arXiv 2025 | Implementacion en PyTorch/JAX |

Ambos buscan resolver el problema del **"attention sink"** y mejorar la **estabilidad de entrenamiento** y **generalizacion a largo contexto**, pero lo hacen desde direcciones opuestas.

---

## 1. Gated Attention

### 1.1 Paper
- **Titulo:** "Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free"
- **Autores:** Zihan Qiu, Zekun Wang, Bo Zheng, et al. (Alibaba/Qwen)
- **Award:** NeurIPS 2025 Best Paper (4 de 5,290 papers aceptados)

### 1.2 Problema que resuelve

En la atencion estandar, el fenomeno de **"attention sink"** hace que el primer token reciba una atencion desproporcionada en multiples capas. Esto limita la capacidad del modelo para distribuir atencion de forma significativa.

```
Baseline Attention (attention sink):
Layer 1:  [0.45, 0.12, 0.08, 0.05, ...]  <- Primer token domina
Layer 7:  [0.38, 0.15, 0.10, 0.07, ...]
Layer 21: [0.42, 0.11, 0.09, 0.06, ...]
```

### 1.3 Metodologia

Gated Attention introduce un **gate dependiente de la query** despues de la salida de Scaled Dot-Product Attention (SDPA). Este gate:

1. **Introduce no-linearidad** en la transformacion de bajo rango formada por las proyecciones V y O
2. **Habilita esparsidad dependiente del input**, previniendo el attention sink
3. **Mejora la estabilidad de entrenamiento**, permitiendo learning rates mas grandes
4. **Mejora la extrapolacion a largo contexto** (hasta 1M tokens)

### 1.4 Arquitectura - Flujo de Datos

```
Input x
    |
    v
[Q, K, V] = x * W_q, x * W_k, x * W_v
    |
    v
[Q, Gate_Score] = split(Q)  <-- La query se proyecta extra para el gate
    |
    v
A = softmax(Q @ K^T / sqrt(d_k))
    |
    v
O = A @ V
    |
    v
O = O * sigmoid(Gate_Score)  <-- APLICACION DEL GATE
    |
    v
Output = O * W_o
```

### 1.5 Variantes de Gating

#### 1.5.1 Headwise Gating (G1-headwise)
- Cada head de atencion tiene **un solo escalar gate**
- Parametros extra por capa: `num_heads` escalares
- El gate se obtiene de la query: `gate_score = Q[:, :, :, num_heads]`
- Aplicacion: `output = output * sigmoid(gate_score)`

#### 1.5.2 Elementwise Gating (G1-elementwise)
- Cada elemento del output de atencion se modula **independientemente**
- Parametros extra por capa: `num_heads * head_dim` valores
- El gate tiene la misma dimension que el output: `gate_score = Q[:, :, :, head_dim]`
- Aplicacion: `output = output * sigmoid(gate_score)`

### 1.6 Implementacion Clave (modeling_qwen3.py)

```python
# Lineas 263-281: Configuracion del gate
self.headwise_attn_output_gate = config.headwise_attn_output_gate
self.elementwise_attn_output_gate = config.elementwise_attn_output_gate

if self.headwise_attn_output_gate:
    # Proyecta Q + un escalar extra por head
    self.q_proj = nn.Linear(hidden_size, num_heads * head_dim + num_heads, bias=qkv_bias)
elif self.elementwise_attn_output_gate:
    # Proyecta Q + Q extra (duplicado)
    self.q_proj = nn.Linear(hidden_size, num_heads * head_dim * 2, bias=qkv_bias)

# Lineas 309-318: Extraccion del gate score
if self.headwise_attn_output_gate:
    query_states, gate_score = torch.split(query_states, 
        [head_dim * num_kv_groups, num_kv_groups], dim=-1)
    gate_score = gate_score.reshape(bsz, q_len, -1, 1)
elif self.elementwise_attn_output_gate:
    query_states, gate_score = torch.split(query_states, 
        [head_dim * num_kv_groups, head_dim * num_kv_groups], dim=-1)
    gate_score = gate_score.reshape(bsz, q_len, -1, head_dim)

# Lineas 361-362: Aplicacion del gate
if self.headwise_attn_output_gate or self.elementwise_attn_output_gate:
    attn_output = attn_output * torch.sigmoid(gate_score)
```

### 1.7 Resultados

- Integrado en **Qwen3-Next** (80B-A3B-Instruct)
- Soporte para contexto ultra-largo (1M tokens)
- Mejora significativa en benchmarks RULER
- Training mas estable con learning rates mas altos

---

## 2. Bilinearly Modulated Attention (BMA)

### 2.1 Paper
- **Titulo:** "Bilinearly Modulated Attention: Query-Conditioned Value Gating"
- **Autor:** Iheb Gafsi
- **Implementacion:** PyTorch + JAX

### 2.2 Problema que resuelve

BMA busca una alternativa teoricamente motivada al post-SDPA gating (como Gated Attention). En lugar de hacer gate **despues** de la agregacion, BMA lo hace **antes**, preservando la geometria del softmax.

### 2.3 Metodologia

BMA aplica **gating dependiente de la query sobre los valores** antes de la agregacion de atencion. Esto:

1. **Preserva la geometria del softmax** (los logits no se modifican)
2. **Habilita filtrado dependiente de la query** (cada token filtra valores segun necesidad contextual)
3. **Genera interacciones mas ricas** - mapas bilineales generan d_h^2 combinaciones de features vs escalares en gating simpler
4. **Flujo de gradientes mejorado** - los gates reciben senales tanto de Q como del loss final
5. **Parametros extra minimos** - solo H * d_h^2 por capa

### 2.4 Arquitectura - Flujo de Datos

```
Input x
    |
    v
[Q, K, V] = x * W_qkv  (proyeccion conjunta)
    |
    v
A = softmax(Q @ K^T / sqrt(d_h))
    |
    v
G = sigmoid(Q @ W_g)  <-- Gate dependiente de la QUERY
    |
    v
V_modulated = G * V  <-- Valores modulados ANTES de la agregacion
    |
    v
O = A @ V_modulated
    |
    v
Output = O * W_o
```

### 2.5 Diferencia Clave con Gated Attention

```
Gated Attention:        O = (A @ V) * sigmoid(gate)  <-- POST-aggregation
BMA:                    O = A @ (sigmoid(Q @ W_g) * V)  <-- PRE-aggregation
```

### 2.6 Implementacion Clave (attention.py)

```python
class BilinearlyModulatedAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0, bias=True, causal=True):
        super().__init__()
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out = nn.Linear(d_model, d_model, bias=bias)
        
        # Matriz de gate por-head: d_head x d_head
        self.W_g = nn.Parameter(
            torch.randn(n_heads, self.d_head, self.d_head) * 0.02
        )
    
    def forward(self, x, mask=None):
        B, T, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape multi-head
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        # Attention scores (sin modificar)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = torch.softmax(scores, dim=-1)
        
        # Query-conditioned value gating
        g = torch.sigmoid(
            torch.einsum("bhtd,hde->bhte", q, self.W_g)
        )
        
        # Modulate values ANTES de la agregacion
        v = g * v
        
        # Aggregation
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out(out)
        return out
```

### 2.7 Eficiencia de Parametros

| Mecanismo | Parametros extra/capa (H=8, d_h=64) |
|-----------|--------------------------------------|
| BMA | 8 * 64 * 64 = 32,768 |
| Gated Attention (headwise) | 8 escalares |
| Gated Attention (elementwise) | 8 * 64 = 512 |
| Post-SDPA Gating (lineal) | 512 * 512 = 262,144 |

BMA es **4x mas eficiente** en parametros que el post-SDPA gating lineal.

### 2.8 Resultados Preliminares

- Modelo de 35M parametros entrenado en 400B tokens
- **BMA:** Perplejidad 216.2
- **Standard Attention:** Perplejidad 221.1
- **Post-SDPA Gating:** Perplejidad 217.49
- Sin spikes de perdida durante entrenamiento
- Solo 1M parametros extra vs 4.1M del post-SDPA gating

---

## 3. Comparativa Tecnica

### 3.1 Donde se aplica el Gate

```
┌─────────────────────────────────────────────────────┐
│              Standard Attention                     │
│                                                     │
│   Q, K, V = x * W_q, x * W_k, x * W_v            │
│   A = softmax(Q @ K^T / sqrt(d_k))                │
│   O = A @ V                                         │
│   output = O * W_o                                  │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│              Gated Attention                        │
│                                                     │
│   Q, K, V = x * W_q, x * W_k, x * W_v            │
│   [Q, gate] = split(Q)                             │
│   A = softmax(Q @ K^T / sqrt(d_k))                │
│   O = A @ V                                         │
│   O = O * sigmoid(gate)  ← POST-AGGREGATION        │
│   output = O * W_o                                  │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│              BMA                                    │
│                                                     │
│   [Q, K, V] = x * W_qkv                            │
│   A = softmax(Q @ K^T / sqrt(d_h))                │
│   G = sigmoid(Q @ W_g)  ← PRE-AGGREGATION          │
│   V_mod = G * V                                     │
│   O = A @ V_mod                                     │
│   output = O * W_o                                  │
└─────────────────────────────────────────────────────┘
```

### 3.2 Tabla Comparativa

| Caracteristica | Gated Attention | BMA |
|---------------|-----------------|-----|
| **Posicion del gate** | Post-aggregation | Pre-aggregation |
| **Que modula** | Output de atencion | Valores (V) |
| **Dependencia** | Query-dependent | Query-dependent |
| **No-linearidad** | sigmoid en gate | sigmoid en gate |
| **Esparsidad** | Input-dependent | Input-dependent |
| **Afecta softmax?** | No | No |
| **Parametros extra** | Muy pocos (escalar/d_h por head) | d_h^2 por head |
| **Complejidad** | O(1) por elemento | O(d_h^2) por head |
| **Backbone** | Qwen3 (Transformer) | Transformer generico |
| **Implementacion** | PyTorch (HF) | PyTorch + JAX |
| **Paper** | NeurIPS 2025 Best Paper | Preprint 2025 |

### 3.3 Ventajas y Desventajas

#### Gated Attention
**Ventajas:**
- Extremadamente ligero en parametros
- Easy de integrar (solo modifica Q projection)
- Probado en modelos de 80B parametros
- Soporte para contexto ultra-largo (1M tokens)

**Desventajas:**
- Gate post-aggregation no modula los valores directamente
- Menor expresividad que pre-aggregation gating
- Dependiente de la arquitectura Qwen3

#### BMA
**Ventajas:**
- Mayor expresividad (d_h^2 combinaciones de features)
- Gate modula valores antes de la mezcla de atencion
- Flujo de gradientes mas limpio
- Framework generico (PyTorch + JAX)

**Desventajas:**
- Mas parametros que gated attention headwise
- No probado en modelos de gran escala
- Implementacion mas compleja

---

## 4. Diagramas de Arquitectura

### 4.1 Gated Attention - Bloque de Atencion

```
                    ┌─────────────┐
                    │  Input (x)  │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              v            v            v
         ┌────────┐   ┌────────┐   ┌────────┐
         │  W_q   │   │  W_k   │   │  W_v   │
         └───┬────┘   └───┬────┘   └───┬────┘
             │            │            │
             v            │            │
    ┌────────────────┐    │            │
    │ Split: Q, Gate │    │            │
    └───────┬────────┘    │            │
            │             │            │
            v             v            v
    ┌───────────────────────────────────────┐
    │         Scaled Dot-Product            │
    │    A = softmax(Q @ K^T / sqrt(d))    │
    └───────────────────┬───────────────────┘
                        │
                        v
    ┌───────────────────────────────────────┐
    │           O = A @ V                   │
    └───────────────────┬───────────────────┘
                        │
                        v
    ┌───────────────────────────────────────┐
    │    O = O * sigmoid(Gate_Score)        │
    └───────────────────┬───────────────────┘
                        │
                        v
                 ┌────────────┐
                 │    W_o     │
                 └─────┬──────┘
                       │
                       v
                 ┌────────────┐
                 │  Output    │
                 └────────────┘
```

### 4.2 BMA - Bloque de Atencion

```
                    ┌─────────────┐
                    │  Input (x)  │
                    └──────┬──────┘
                           │
                    ┌──────┴──────┐
                    v             v
             ┌──────────┐   ┌──────────┐
             │  W_qkv   │   │   W_g    │
             └─────┬────┘   └────┬─────┘
                   │             │
           ┌───────┼───────┐     │
           v       v       v     │
        ┌─────┐ ┌─────┐ ┌─────┐ │
        │  Q  │ │  K  │ │  V  │ │
        └──┬──┘ └──┬──┘ └──┬──┘ │
           │       │       │     │
           v       v       │     │
    ┌────────────────────┐  │     │
    │  A = softmax(      │  │     │
    │    Q @ K^T / sqrt) │  │     │
    └─────────┬──────────┘  │     │
              │             │     │
              │             v     v
              │    ┌────────────────┐
              │    │ G = sigmoid(   │
              │    │   Q @ W_g)     │
              │    └───────┬────────┘
              │            │
              v            v
    ┌───────────────────────────────────────┐
    │         V_mod = G * V                 │
    └───────────────────┬───────────────────┘
                        │
                        v
    ┌───────────────────────────────────────┐
    │         O = A @ V_mod                 │
    └───────────────────┬───────────────────┘
                        │
                        v
                 ┌────────────┐
                 │    W_o     │
                 └─────┬──────┘
                       │
                       v
                 ┌────────────┐
                 │  Output    │
                 └────────────┘
```

---

## 5. Codigo Relevante

### 5.1 Gated Attention - Qwen3Config (configuracion_qwen3.py)

```python
# Parametros de gating en la configuracion
elementwise_attn_output_gate: bool = False  # Gate por elemento
headwise_attn_output_gate: bool = False     # Gate por head
```

### 5.2 BMA - Matriz de Gate (attention.py)

```python
# W_g es una matriz learnable por head
# Shape: (n_heads, d_head, d_head)
self.W_g = nn.Parameter(
    torch.randn(n_heads, self.d_head, self.d_head) * 0.02
)

# En el forward:
# G = sigmoid(Q @ W_g) produce gates dependientes de la query
g = torch.sigmoid(
    torch.einsum("bhtd,hde->bhte", q, self.W_g)
)
```

### 5.3 Comparacion de Eficiencia

```python
# Gated Attention Headwise - solo 8 parametros extra por capa
gate_score = query[:, :, :, num_heads]  # (B, T, H, 1)

# BMA - 32,768 parametros extra por capa (H=8, d_h=64)
W_g = Parameter(randn(8, 64, 64))  # 8 * 64 * 64 = 32,768
g = sigmoid(einsum("bhtd,hde->bhte", q, W_g))
```

---

## 6. Aplicacion en Evil Inference Code

### 6.1 Integracion Potencial

Ambos mecanismos son candidatos para Evil Inference Code:

| Mecanismo | Aplicacion | Prioridad |
|-----------|------------|-----------|
| Gated Attention | Inferencia distribuida en modelos Qwen3 | Alta |
| BMA | ModelosTransformer custom con atencion mejorada | Media |

### 6.2 Consideraciones para Rust

```rust
// Gated Attention en Rust
fn gated_attention(q: &Tensor, k: &Tensor, v: &Tensor, gate_score: &Tensor) -> Tensor {
    let a = softmax(q.matmul(&k.transpose(-2, -1)) / sqrt(d_k));
    let o = a.matmul(v);
    o * sigmoid(gate_score)  // Post-aggregation gate
}

// BMA en Rust
fn bma_attention(q: &Tensor, k: &Tensor, v: &Tensor, w_g: &Tensor) -> Tensor {
    let a = softmax(q.matmul(&k.transpose(-2, -1)) / sqrt(d_h));
    let g = sigmoid(q.matmul(w_g));  // Pre-aggregation gate
    let v_mod = g * v;
    a.matmul(&v_mod)
}
```

### 6.3 Roadmap Sugerido

1. **Fase 1:** Implementar Gated Attention en Rust (mas simple, menos parametros)
2. **Fase 2:** Implementar BMA en Rust (mayor expresividad)
3. **Fase 3:** Benchmark comparativo en modelos distribuidos
4. **Fase 4:** Integrar en pipeline de inferencia de Evil Inference Code

---

## 7. Referencias

1. Qiu et al. "Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free" arXiv:2505.06708 (NeurIPS 2025 Best Paper)

2. Gafsi. "Bilinearly Modulated Attention: Query-Conditioned Value Gating" arXiv 2025

3. Vaswani et al. "Attention Is All You Need" NeurIPS 2017

4. Fedus et al. "The Era of 1-bit LLMs: BitNet" arXiv:2402.17764

5. Qwen Team. "Qwen3 Architecture" 2025

---

*Reporte generado automaticamente por Evil Inference Code Tech Wiki*
