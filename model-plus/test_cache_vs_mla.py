import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
import torch.nn.functional as F
from mla.model import TransformerLM
from rope import apply_rope_partial

torch.manual_seed(42)

m = TransformerLM(
    vocab_size=256, d_model=128, num_layers=3,
    num_heads=12, num_kv_groups=4,
    use_mla=True, use_swiglu=True, use_x0=True,
).eval()

x = torch.randint(0, 256, (2, 8))

# ── Full forward (sin cache) ──
logits_full = m(x)

# ── MLA forward con cache latente (1 token a la vez) ──
caches = [None] * m.num_layers
logits_cached = []
for pos in range(8):
    out, caches = m.forward_with_cache(x[:, pos:pos+1], pos, caches)
    logits_cached.append(out[:, -1:, :])
logits_cached = torch.cat(logits_cached, dim=1)

diff = (logits_full - logits_cached).abs().max().item()
print(f"Full cache:         diff={diff:.6f}")

# ── MLA partial RoPE con cache latente (1 token a la vez) ──
caches2 = [None] * m.num_layers
logits_cached2 = []
for pos in range(8):
    out, caches2 = m.forward_with_cache_partial(x[:, pos:pos+1], pos, caches2, 0.25)
    logits_cached2.append(out[:, -1:, :])
logits_cached2 = torch.cat(logits_cached2, dim=1)

# Referencia: forward manual con partial RoPE en MLA
ref = m.embedding(x).clone()
for i, layer in enumerate(m.transformer.layers):
    res = ref
    h = layer.attn_norm(ref)
    Q_state, Q_rotate, K, V, K_rot = layer.attention.qkv(h)
    Q_rot, K_rot = apply_rope_partial(Q_rotate, K_rot, 0, 0.25,
        layer.attention.rope.inv_freq, layer.attention.rope.cos_cache,
        layer.attention.rope.sin_cache, layer.attention.rope.head_dim,
        layer.attention.rope.max_seq_len)
    Q = torch.cat([Q_state, Q_rot], dim=-1)
    K_rot_exp = K_rot.expand(-1, -1, layer.attention.num_kv_groups, -1)
    K = torch.cat([K, K_rot_exp], dim=-1)
    k = K.repeat_interleave(layer.attention.num_heads // layer.attention.num_kv_groups, dim=2)
    v = V.repeat_interleave(layer.attention.num_heads // layer.attention.num_kv_groups, dim=2)
    s = torch.matmul(Q.transpose(1,2), k.transpose(1,2).transpose(-2,-1)) / math.sqrt(layer.attention.qkv.qk_dim)
    mask = torch.triu(torch.full((8, 8), float("-inf")), diagonal=1)
    s = s + mask.unsqueeze(0).unsqueeze(0)
    a = F.softmax(s, dim=-1)
    o = (a @ v.transpose(1,2)).transpose(1,2)
    ref = res + layer.attention.o_proj(o)
    res2 = ref
    ref = ref + layer.ffn(layer.ffn_norm(ref))
h = m.transformer.final_norm(ref)
logits_ref = m.head(h)

diff2 = (logits_ref - logits_cached2).abs().max().item()
print(f"Partial cache:      diff={diff2:.6f}")
print("OK" if diff < 1e-4 and diff2 < 1e-4 else "DIFIERE")
