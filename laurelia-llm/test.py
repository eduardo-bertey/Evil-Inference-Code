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

# — 6. rotación manual: verificar fórmula de rotación 2D
rope_test = RoPE(head_dim=4, max_seq_len=10, base=10000.0, rotary_pct=1.0)
rope_test.eval()
x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
pos = 1
theta0 = pos / (10000.0 ** (2*0/4))
theta1 = pos / (10000.0 ** (2*1/4))
c0, s0 = math.cos(theta0), math.sin(theta0)
c1, s1 = math.cos(theta1), math.sin(theta1)
x_rot = rope_test(x, x, offset=pos)[0]
# RoPE code: first_half (dims 0,1) y second_half (dims 2,3)
# first[n] * cos[n] - second[n] * sin[n], luego first[n] * sin[n] + second[n] * cos[n]
# dim 0 (first[0]) y dim 2 (second[0]) son el par 0 (θ0)
# dim 1 (first[1]) y dim 3 (second[1]) son el par 1 (θ1)
f0, f1 = 1.0, 2.0   # first half = q_rot[..., :2]
s0_, s1_ = 3.0, 4.0  # second half = q_rot[..., 2:4]
expected_rot = torch.tensor([[[[
    f0*c0 - s0_*s0,    # dim 0 = first[0]*c0 - second[0]*s0
    f1*c1 - s1_*s1,    # dim 1 = first[1]*c1 - second[1]*s1
    f0*s0 + s0_*c0,    # dim 2 = first[0]*s0 + second[0]*c0
    f1*s1 + s1_*c1,    # dim 3 = first[1]*s1 + second[1]*c1
]]]])
check("rotación manual 2D coincide", torch.allclose(x_rot, expected_rot, atol=1e-6))

# — 7. propiedad relativa: mismo vector q/k rotado a distintas posiciones
rope2 = RoPE(head_dim=16, max_seq_len=10, base=10000.0, rotary_pct=cfg.rotary_pct)
rope2.eval()
q_single = torch.randn(1, 1, 1, 16)
k_single = torch.randn(1, 1, 1, 16)
q0, k0 = rope2(q_single, k_single, offset=0)
q1, k1 = rope2(q_single, k_single, offset=1)
q2, k2 = rope2(q_single, k_single, offset=2)
q3, k3 = rope2(q_single, k_single, offset=3)
q4, k4 = rope2(q_single, k_single, offset=4)
d01 = (q0 * k1).sum(-1)
d12 = (q1 * k2).sum(-1)
d23 = (q2 * k3).sum(-1)
d34 = (q3 * k4).sum(-1)
check("RoPE: q[i]·k[i+1] igual para todo i",
      torch.allclose(d01, d12, atol=1e-6) and
      torch.allclose(d12, d23, atol=1e-6) and
      torch.allclose(d23, d34, atol=1e-6))
d02 = (q0 * k2).sum(-1)
d13 = (q1 * k3).sum(-1)
d24 = (q2 * k4).sum(-1)
check("RoPE: q[i]·k[i+2] igual para todo i",
      torch.allclose(d02, d13, atol=1e-6) and
      torch.allclose(d13, d24, atol=1e-6))
check("RoPE: q[i]·k[i] preserva dot original",
      torch.allclose((q0*k0).sum(-1), (q_single*k_single).sum(-1), atol=1e-6))
check("RoPE: dot Δ=1 != Δ=2",
      not torch.allclose(d01, d02, atol=1e-6))

# ============================================================
print("\n=== Attention MINIMAL: forward vs forward_with_cache ===")
attn = Attention(cfg)
attn.eval()
x1 = torch.randn(2, 1, cfg.dim)

out1a = attn(x1)
out1b, _ = attn.forward_with_cache(x1, 0, None)
d1 = maxdiff(out1a, out1b)
print(f"  SAME module, S=1: maxdiff={d1:.2e}  {'PASS' if d1 < 1e-5 else 'FAIL'}")

# Ahora con S=2
x2 = torch.randn(2, 2, cfg.dim)
out2a = attn(x2)
out2b, _ = attn.forward_with_cache(x2, 0, None)
d2 = maxdiff(out2a, out2b)
print(f"  SAME module, S=2: maxdiff={d2:.2e}  {'PASS' if d2 < 1e-4 else 'FAIL'}")

# probar incremental manual
x_a = x1[:, :1, :]
x_b = torch.randn(2, 1, cfg.dim)
x_ab = torch.cat([x_a, x_b], dim=1)
ref_ab = attn(x_ab)

out_a, ca = attn.forward_with_cache(x_a, 0, None)
out_b, cb = attn.forward_with_cache(x_b, 1, ca)
d_a = maxdiff(out_a, ref_ab[:, 0:1])
d_b = maxdiff(out_b, ref_ab[:, 1:2])
print(f"  incremental step1 vs ref[0]: maxdiff={d_a:.2e}  {'PASS' if d_a < 1e-5 else 'FAIL'}")
print(f"  incremental step2 vs ref[1]: maxdiff={d_b:.2e}  {'PASS' if d_b < 1e-5 else 'FAIL'}")
print(f"  cache seqlen: {cb[0].shape[1]}")

# ============================================================
print("\n=== Attention matemática ===")
# atención hard: Q hace attend a K[0] (shapes: B, H, Q_len/S_len, D)
q_hard = torch.zeros(1, 1, 1, 4)
q_hard[..., 0] = 10.0
k_hard = torch.zeros(1, 1, 3, 4)
k_hard[:, :, 0, 0] = 10.0
v_hard = torch.tensor([[[[1.0, 2.0, 3.0, 4.0],
                          [5.0, 6.0, 7.0, 8.0],
                          [9.0, 10.0, 11.0, 12.0]]]])
out_hard = F.scaled_dot_product_attention(q_hard, k_hard, v_hard, is_causal=False)
check("SDPA hard attention a K[0]: output ≈ V[0]",
      torch.allclose(out_hard, v_hard[:, :, :1, :], atol=1e-4))

# atención uniforme: Q=0 → attn uniforme → output = mean(V)
q_unif = torch.zeros(1, 1, 1, 4)
k_unif = torch.zeros(1, 1, 3, 4)
v_unif = torch.tensor([[[[1.0, 0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0]]]])
out_unif = F.scaled_dot_product_attention(q_unif, k_unif, v_unif, is_causal=False)
expected_unif = v_unif.mean(dim=2, keepdim=True)
check("SDPA uniforme: output ≈ mean(V)",
      torch.allclose(out_unif, expected_unif, atol=1e-4))

# Atención hard con SDPA directo (shapes correctas) + verificar proyecciones lineales
q_hard2 = torch.zeros(1, 4, 1, 8)
q_hard2[:, :, 0, 0] = 50.0
k_hard2 = torch.zeros(1, 4, 3, 8)
k_hard2[:, :, 0, 0] = 50.0
v_hard2 = torch.arange(96, dtype=torch.float32).reshape(1, 4, 3, 8)
out_hard2 = F.scaled_dot_product_attention(q_hard2, k_hard2, v_hard2, is_causal=False)
# solo K[0] atrae atención por tener dim0=5 igual que Q, el resto es 0
check("SDPA multi-head hard attention a K[0]",
      torch.allclose(out_hard2, v_hard2[:, :, :1, :], atol=1e-3))

# Verificar shapes de proyecciones de Attention
attn_shape = Attention(cfg)
check("q_proj output shape",
      attn_shape.q_proj(torch.randn(2, 5, cfg.dim)).shape == (2, 5, cfg.heads*head_dim))
check("k_proj output shape",
      attn_shape.k_proj(torch.randn(2, 5, cfg.dim)).shape == (2, 5, cfg.kv_groups*head_dim))
check("v_proj output shape",
      attn_shape.v_proj(torch.randn(2, 5, cfg.dim)).shape == (2, 5, cfg.kv_groups*head_dim))

# ============================================================
print("\n=== Attention GQA (repeat_kv) ===")
k_gqa = torch.randn(B, S, cfg.kv_groups, head_dim)
v_gqa = torch.randn(B, S, cfg.kv_groups, head_dim)
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
