"""Pruebas unitarias para RoPE y Attention."""
import math, sys
import torch
import torch.nn.functional as F
from rope import RoPE
from model import Config, Attention, LLM, repeat_kv

device = "cpu"
torch.manual_seed(42)
pass_ = 0
fail_ = 0

def check(name, cond, detail=""):
    global pass_, fail_
    if cond:
        pass_ += 1
        print(f"  PASS {name}")
    else:
        fail_ += 1
        print(f"  FAIL {name}" + (f" — {detail}" if detail else ""))

def maxdiff(a, b):
    return (a - b).abs().max().item()

print("=== Config ===")
cfg = Config()
print(f"  dim={cfg.dim}, heads={cfg.heads}, kv_groups={cfg.kv_groups}, head_dim={cfg.dim//cfg.heads}")
print(f"  layers={cfg.layers}, rotary_pct={cfg.rotary_pct}")

head_dim = cfg.dim // cfg.heads
rotary_dim = int(head_dim * cfg.rotary_pct)
rotary_dim = rotary_dim - (rotary_dim % 2)
rotary_half = rotary_dim // 2
print(f"  head_dim={head_dim}, rotary_dim={rotary_dim}, rotary_half={rotary_half}")

# ============================================================
print("\n=== RoPE ===")
rope = RoPE(head_dim, max_seq_len=2048, base=10000.0, rotary_pct=cfg.rotary_pct)
rope.eval()

B, S = 2, 4
q = torch.randn(B, S, cfg.heads, head_dim)
k = torch.randn(B, S, cfg.kv_groups, head_dim)
q_rot, k_rot = rope(q, k, offset=0)

# — 1. partial: dims pasantes sin cambiar
q_pass = q[..., rotary_dim:]
q_rot_pass = q_rot[..., rotary_dim:]
check("dims pasantes idénticas", torch.allclose(q_pass, q_rot_pass))

# — 2. dims rotadas cambian
q_rot_part = q_rot[..., :rotary_dim]
q_orig_part = q[..., :rotary_dim]
check("dims rotadas cambiaron", not torch.allclose(q_rot_part, q_orig_part))

# — 3. norma preservada en dims rotadas
check("norma preservada en Q rotadas", torch.allclose(
    q_rot_part.norm(dim=-1), q_orig_part.norm(dim=-1), atol=1e-6))
check("norma preservada en K rotadas", torch.allclose(
    k_rot[..., :rotary_dim].norm(dim=-1), k[..., :rotary_dim].norm(dim=-1), atol=1e-6))

# — 4. offset consistency: rope con offset 2 en seq_len=2 vs rope offset 0 en seq_len=4 slice [2:4]
q2 = q[:, 2:4].clone()
k2 = k[:, 2:4].clone()
q2_rot, k2_rot = rope(q2, k2, offset=2)
check("offset consistency Q", torch.allclose(q_rot[:, 2:4], q2_rot, atol=1e-6))
check("offset consistency K", torch.allclose(k_rot[:, 2:4], k2_rot, atol=1e-6))

# — 5. inv_freq formula: 1/(base^(2k/head_dim))
expected = 1.0 / (10000.0 ** (torch.arange(rotary_half).float() * 2.0 / head_dim))
check("inv_freq fórmula correcta", torch.allclose(rope.inv_freq, expected, atol=1e-12))

# ============================================================
print("\n=== Attention DEBUG: step-by-step comparison ===")
# Crear dos attn IDENTICAS (clonando pesos)
attn_a = Attention(cfg)
attn_a.eval()
attn_b = Attention(cfg)
attn_b.eval()
attn_b.load_state_dict(attn_a.state_dict())

x_single = torch.randn(B, 1, cfg.dim)
x_double = torch.randn(B, 2, cfg.dim)

# forward en attn_a
out_fwd, _ = attn_a(x_single)
# forward_with_cache en attn_b
out_cache, cache = attn_b.forward_with_cache(x_single, offset=0, cache=None)

d = maxdiff(out_fwd, out_cache)
print(f"  attn_a(x) vs attn_b.forward_with_cache(x,0,None): maxdiff={d:.2e}")
print(f"  PASS={d < 1e-5}")

# Comparar internals: q, k, v antes de SDPA
with torch.no_grad():
    # forward path
    B1, T1, D1 = x_single.shape
    q1 = attn_a.q_proj(x_single).view(B1, T1, attn_a.num_heads, attn_a.head_dim)
    k1 = attn_a.k_proj(x_single).view(B1, T1, attn_a.num_kv_groups, attn_a.head_dim)
    v1 = attn_a.v_proj(x_single).view(B1, T1, attn_a.num_kv_groups, attn_a.head_dim)
    q1, k1 = attn_a.rope(q1, k1, 0)
    k1e = repeat_kv(k1, attn_a.num_heads, attn_a.num_kv_groups)
    v1e = repeat_kv(v1, attn_a.num_heads, attn_a.num_kv_groups)
    q1t = q1.transpose(1, 2)
    k1t = k1e.transpose(1, 2)
    v1t = v1e.transpose(1, 2)
    o1 = F.scaled_dot_product_attention(q1t, k1t, v1t, is_causal=True)
    o1 = o1.transpose(1, 2).contiguous().view(B1, T1, -1)

    # forward_with_cache path
    B2, S2, _ = x_single.shape
    q2 = attn_b.q_proj(x_single).view(B2, S2, attn_b.num_heads, attn_b.head_dim)
    k2 = attn_b.k_proj(x_single).view(B2, S2, attn_b.num_kv_groups, attn_b.head_dim)
    v2 = attn_b.v_proj(x_single).view(B2, S2, attn_b.num_kv_groups, attn_b.head_dim)
    q2, k2 = attn_b.rope(q2, k2, 0)
    # cache=None → k_full=k2, v_full=v2
    k2e = repeat_kv(k2, attn_b.num_heads, attn_b.num_kv_groups)
    v2e = repeat_kv(v2, attn_b.num_heads, attn_b.num_kv_groups)
    q2t = q2.transpose(1, 2)
    k2t = k2e.transpose(1, 2)
    v2t = v2e.transpose(1, 2)
    o2 = F.scaled_dot_product_attention(q2t, k2t, v2t, is_causal=(True))  # cache is None
    o2 = o2.transpose(1, 2).contiguous().view(B2, S2, -1)

print(f"  q:      maxdiff={maxdiff(q1, q2):.2e}")
print(f"  k:      maxdiff={maxdiff(k1, k2):.2e}")
print(f"  v:      maxdiff={maxdiff(v1, v2):.2e}")
print(f"  q.T:    maxdiff={maxdiff(q1t, q2t):.2e}")
print(f"  k.T:    maxdiff={maxdiff(k1t, k2t):.2e}")
print(f"  v.T:    maxdiff={maxdiff(v1t, v2t):.2e}")
print(f"  sdpa:   maxdiff={maxdiff(o1, o2):.2e}")
print(f"  o_proj: maxdiff={maxdiff(attn_a.o_proj(o1), attn_b.o_proj(o2)):.2e}")

# Ahora probar forward_with_cache con cache pre-poblado
print()
print(f"  === forward_with_cache con cache de 1 token ===")
x_pos1 = torch.randn(B, 1, cfg.dim)
x_pos2 = torch.randn(B, 1, cfg.dim)
x_both = torch.cat([x_pos1, x_pos2], dim=1)

# forward con ambos tokens (is_causal=True)
ref2, _ = attn_a.forward(x_both)
ref_first = ref2[:, 0:1]
ref_second = ref2[:, 1:2]

# forward_with_cache incremental
out_a, cache_a = attn_b.forward_with_cache(x_pos1, offset=0, cache=None)
out_b, cache_b = attn_b.forward_with_cache(x_pos2, offset=1, cache=cache_a)

print(f"  inc step1 vs ref[:,0]: maxdiff={maxdiff(out_a, ref_first):.2e}")
print(f"  inc step2 vs ref[:,1]: maxdiff={maxdiff(out_b, ref_second):.2e}")

# check cache shapes
print(f"  cache_a k shape: {cache_a[0].shape}")
print(f"  cache_b k shape: {cache_b[0].shape}")

check("DEBUG: forward == forward_with_cache attn_a vs attn_b",
      torch.allclose(out_fwd, out_cache, atol=1e-5))

# cache final debe tener K,V de ambos tokens
check("cache seqlen = 2 tras incremental", cache2[0].shape[1] == 2)
check("cache[0] == cache[1] (misma seqlen)", cache2[0].shape[1] == cache2[1].shape[1])

# ============================================================
print("\n=== Attention GQA (repeat_kv) ===")
k_gqa = torch.randn(B, S, cfg.kv_groups, head_dim)
v_gqa = torch.randn(B, S, cfg.kv_groups, head_dim)
from model import repeat_kv
k_exp = repeat_kv(k_gqa, cfg.heads, cfg.kv_groups)
v_exp = repeat_kv(v_gqa, cfg.heads, cfg.kv_groups)
check("GQA repeat_kv heads correctos", k_exp.shape[2] == cfg.heads)
check("GQA repeat_kv groups correctos", v_exp.shape[2] == cfg.heads)
# cada head del mismo grupo debe ser igual
for g in range(cfg.kv_groups):
    h0 = g * (cfg.heads // cfg.kv_groups)
    check(f"GQA group {g} heads iguales",
          torch.allclose(k_exp[:, :, h0], k_exp[:, :, h0+1], atol=1e-6))

# ============================================================
print("\n=== LLM forward vs forward_with_cache ===")
model = LLM(cfg)
model.eval()

# Con S=1: forward == forward_with_cache
ids1 = torch.randint(0, 1000, (B, 1))
logits_fwd1, _ = model(ids1)
logits_cache1, caches = model.forward_with_cache(ids1, 0, None)
check("LLM single token: forward == forward_with_cache",
      torch.allclose(logits_fwd1, logits_cache1, atol=1e-4))

# Con S>1 y cache=None: forward_with_cache usa is_causal → mismo que forward
ids4 = torch.randint(0, 1000, (B, S))
logits_fwd4, _ = model(ids4)
logits_cache4, _ = model.forward_with_cache(ids4, 0, None)
d = maxdiff(logits_fwd4, logits_cache4)
check("LLM multi token: forward == forward_with_cache",
      torch.allclose(logits_fwd4, logits_cache4, atol=1e-3), f"maxdiff={d:.2e}")

check("LLM number of caches == layers", len(caches) == cfg.layers)

# ============================================================
print("\n=== LLM incremental generation ===")
input_ids = torch.randint(0, 1000, (1, 3))
caches = None
# prompt
for i in range(3):
    logits_cached, caches = model.forward_with_cache(input_ids[:, i:i+1], i, caches)
# full forward para comparar
logits_full, _ = model(input_ids)
# después del prompt loop, el último forward predice la siguiente posición
# forward_with_cache(offset=2) con tok2 → logits para posición 3 (el "primer token nuevo")
# forward(input_ids) con seq=3 → logits para posiciones 0,1,2 (predice posiciones 1,2,3)
# así que logits_full[:, -1, :] = predicción para posición 3
check("prompt loop final logit == full forward last logit",
      torch.allclose(logits_cached[:, -1, :], logits_full[:, -1, :], atol=1e-4))

# ============================================================
print("\n=== SDPA is_causal ===")
# SDPA con is_causal=True y q_len=1, kv_len=N: query index 0 solo ve key index 0
q_test = torch.randn(1, cfg.heads, 1, head_dim)
k_test = torch.randn(1, cfg.heads, 5, head_dim)
v_test = torch.randn(1, cfg.heads, 5, head_dim)
out_causal = F.scaled_dot_product_attention(q_test, k_test, v_test, is_causal=True)
out_nomask = F.scaled_dot_product_attention(q_test, k_test, v_test, is_causal=False)
check("SDPA causal q_len=1 NO attiende todos los keys (solo key[0])",
      not torch.allclose(out_causal, out_nomask, atol=1e-5))
# verificar que causal con q_len=1 da = attender solo key[0]
k0 = k_test[:, :, :1, :]
v0 = v_test[:, :, :1, :]
out_only_k0 = F.scaled_dot_product_attention(q_test, k0, v0, is_causal=False)
check("SDPA causal q_len=1 == attender solo key[0]",
      torch.allclose(out_causal, out_only_k0, atol=1e-5))

print(f"\n{'='*40}")
print(f"Resultados: {pass_} passed, {fail_} failed")
if fail_:
    sys.exit(1)
