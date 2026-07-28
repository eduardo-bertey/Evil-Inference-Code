"""Pruebas unitarias para RoPE y Attention."""
import math, sys
import torch
import torch.nn.functional as F
from rope import RoPE
from model import Config, Attention, LLM

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
print("\n=== Attention (forward vs forward_with_cache) ===")
attn = Attention(cfg)
attn.eval()

x = torch.randn(B, S, cfg.dim)
logits_ref, _ = attn(x)  # forward normal con causal

# forward_with_cache offset=0, sin cache previo — debe dar igual
logits_cache, cache = attn.forward_with_cache(x, offset=0, cache=None)
check("attn forward == forward_with_cache(offset=0, cache=None)",
      torch.allclose(logits_ref, logits_cache, atol=1e-5))

cache_len = cache[0].shape[1]
check("cache length == S", cache_len == S)

# ============================================================
print("\n=== Attention incremental (KV cache) ===")
x1 = x[:, :1, :]  # primer token
x2 = x[:, 1:2, :]  # segundo token
x12 = x[:, :2, :]  # primeros dos tokens juntos

# forward normal con 2 tokens
ref_out, _ = attn.forward(x12)

# forward incremental: token1, luego token2
out1, cache1 = attn.forward_with_cache(x1, offset=0, cache=None)
out2, cache2 = attn.forward_with_cache(x2, offset=1, cache=cache1)

check("incremental: out1 == ref[0]", torch.allclose(out1, ref_out[:, 0:1], atol=1e-5))
check("incremental: out2 == ref[1]", torch.allclose(out2, ref_out[:, 1:2], atol=1e-5))

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
input_ids = torch.randint(0, 1000, (B, S))

logits_fwd, loss_fwd = model(input_ids)
logits_cached, caches = model.forward_with_cache(input_ids, 0, None)

check("LLM forward vs forward_with_cache logits",
      torch.allclose(logits_fwd, logits_cached, atol=1e-4))
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
# SDPA con is_causal=True y q_len=1, kv_len=N debe atender a todos los keys
q_test = torch.randn(1, cfg.heads, 1, head_dim)
k_test = torch.randn(1, cfg.heads, 5, head_dim)
v_test = torch.randn(1, cfg.heads, 5, head_dim)
out_causal = F.scaled_dot_product_attention(q_test, k_test, v_test, is_causal=True)
out_nomask = F.scaled_dot_product_attention(q_test, k_test, v_test, is_causal=False)
# con q_len=1 y causal, la máscara causal permite atender a TODOS (posición 0 puede ver posiciones 0..N-1)
check("SDPA causal q_len=1 attiende todos los keys",
      torch.allclose(out_causal, out_nomask, atol=1e-5))

print(f"\n{'='*40}")
print(f"Resultados: {pass_} passed, {fail_} failed")
if fail_:
    sys.exit(1)
