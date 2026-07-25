import sys, os, math, torch, torch.nn as nn, torch.nn.functional as F
import importlib.util
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.join(_DIR, "LLM_D3-main"))
from model import TransformerLM
from attention import repeat_kv
from LLM_2 import GPT, GPTConfig
_nano_path = os.path.join(_DIR, "nano-moe-mla-main", "steps", "03_block_model.py")
_nano_spec = importlib.util.spec_from_file_location("nano_block_model", _nano_path)
_nano_mod = importlib.util.module_from_spec(_nano_spec)
_nano_spec.loader.exec_module(_nano_mod)
MoeMlaGPT = _nano_mod.MoeMlaGPT
MoeMlaConfig = _nano_mod.MoeMlaConfig

# Import LLM_D5 (named model.py, conflict with ours)
_d5_path = os.path.join(_DIR, "LLM_D5-main", "code", "model.py")
_d5_spec = importlib.util.spec_from_file_location("llm_d5_model", _d5_path)
_d5_mod = importlib.util.module_from_spec(_d5_spec)
_d5_spec.loader.exec_module(_d5_mod)
LLM = _d5_mod.LLM
ConfigD5 = _d5_mod.Config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
txt = sys.argv[1] if len(sys.argv) > 1 else os.path.normpath(os.path.join(_DIR, "..", "input.txt"))
with open(txt, "r", encoding="utf-8") as f:
    text = f.read()
chars = sorted(set(text))
vocab_size = len(chars)
ctoi = {c:i for i,c in enumerate(chars)}

def encode(s):
    return [ctoi[c] for c in s]

tokens_1d = encode(text)
seq_len = 32
batch_size = 4

def make_batch(tokens, seq_len, batch_size):
    n = len(tokens)
    xs, ys = [], []
    for i in range(batch_size):
        offset = (i * seq_len) % (n - seq_len - 1)
        xs.append(tokens[offset:offset+seq_len])
        ys.append(tokens[offset+1:offset+seq_len+1])
    return torch.tensor(xs, dtype=torch.long, device=device), torch.tensor(ys, dtype=torch.long, device=device)

x, y = make_batch(tokens_1d, seq_len, batch_size)

d_model = 256; n_layer = 3; n_head = 4
head_dim = d_model // n_head
d_c = 32; d_c1 = 32; d_rotate = 16
num_kv_groups = 4
ffn_inter = 640  # compute_intermediate_dim(256, 4.0, SwiGLU) = 640

model_ours = TransformerLM(
    vocab_size=vocab_size, d_model=d_model, num_layers=n_layer,
    num_heads=n_head, num_kv_groups=num_kv_groups, head_dim=head_dim,
    use_swiglu=True, use_x0=False, max_seq_len=seq_len,
    residual_dropout=0.0, attn_dropout=0.0, ffn_dropout=0.0,
    use_mla=True, mla_block_size=128,
    mla_d_c=d_c, mla_d_c1=d_c1, mla_d_rotate=d_rotate,
).to(device)
model_ours_x0 = TransformerLM(
    vocab_size=vocab_size, d_model=d_model, num_layers=n_layer,
    num_heads=n_head, num_kv_groups=num_kv_groups, head_dim=head_dim,
    use_swiglu=True, use_x0=True, max_seq_len=seq_len,
    residual_dropout=0.0, attn_dropout=0.0, ffn_dropout=0.0,
    use_mla=True, mla_block_size=128,
    mla_d_c=d_c, mla_d_c1=d_c1, mla_d_rotate=d_rotate,
).to(device)

cfg_d3 = GPTConfig(
    block_size=seq_len, vocab_size=vocab_size, n_layer=n_layer,
    n_head=n_head, n_embd=d_model, dropout=0.0, ffn_dim=ffn_inter,
    bias=False, d_c=d_c, d_c1=d_c1, d_rotate=d_rotate, theta=10000.0,
    n_exp=1, top_k=1, expert_dim=d_model * 2, stride=3,
    use_aux_loss=False, use_router_z_loss=False, use_noisy_top_k=False,
    aux_loss_weight=0.0, router_z_loss_weight=0.0,
    train_capacity=1.0, eval_capacity=1.0, min_capacity=1,
    use_switch_tfm_init=False, router_use_full_prec=False,
)
model_d3 = GPT(cfg_d3).to(device)

cfg_d5 = ConfigD5()
cfg_d5.dim = d_model
cfg_d5.heads = n_head
cfg_d5.layers = n_layer
cfg_d5.ffn_dim = ffn_inter
cfg_d5.block_size = seq_len
cfg_d5.emb_num = vocab_size
cfg_d5.drop = 0.0
model_d5 = LLM(cfg_d5).to(device)

cfg_nano = MoeMlaConfig(
    vocab_size=vocab_size, block_size=seq_len, n_layer=n_layer,
    n_head=n_head, head_dim=head_dim, n_embd=d_model,
    d_rope=d_rotate, d_latent=d_c, n_kv_head=num_kv_groups,
    use_moe=False, use_mla=True,
    qk_norm=False, post_norm=False, load_balance=False, z_loss_gamma=0.0,
)
model_nano = MoeMlaGPT(cfg_nano).to(device)

def breakdown(m):
    total = sum(p.numel() for p in m.parameters())
    emb = sum(p.numel() for n,p in m.named_parameters() if 'embed' in n or 'wte' in n or 'emb' in n)
    attn = sum(p.numel() for n,p in m.named_parameters() if 'attn' in n or 'Wq' in n or 'Wk' in n or 'Wv' in n or 'Wo' in n or 'k_proj' in n or 'v_proj' in n or 'q_proj' in n or 'o_proj' in n)
    ffn = sum(p.numel() for n,p in m.named_parameters() if 'ffn' in n or 'mlp' in n or 'fc1' in n or 'fc2' in n)
    other = total - emb - attn - ffn
    return total, emb, attn, ffn, other

t_o, emb_o, att_o, ffn_o, oth_o = breakdown(model_ours)
t_x0, emb_x0, att_x0, ffn_x0, oth_x0 = breakdown(model_ours_x0)
t_d3, emb_d3, att_d3, ffn_d3, oth_d3 = breakdown(model_d3)
t_d5, emb_d5, att_d5, ffn_d5, oth_d5 = breakdown(model_d5)
t_nano, emb_nano, att_nano, ffn_nano, oth_nano = breakdown(model_nano)

fsize = os.path.getsize(txt)
print(f"Archivo: {os.path.basename(txt)} ({fsize:,} bytes)")
print(f"Tokens totales: {len(tokens_1d):,}  |  Vocab: {vocab_size} chars")
print(f"Batch: {batch_size}x{seq_len} = {batch_size*seq_len} tokens/step")
print(f"d_model={d_model} layers={n_layer} heads={n_head} head_dim={head_dim}")
print(f"MLA: d_c={d_c} d_c1={d_c1} d_rotate={d_rotate}  |  FFN inter={ffn_inter}")
print()
hdr = f"{'':>12} {'total':>10} {'emb':>10} {'attn':>10} {'ffn':>10} {'other':>10}"
print(hdr)
print(f"{'MLA (ours)':>12} {t_o:>10,} {emb_o:>10,} {att_o:>10,} {ffn_o:>10,} {oth_o:>10,}")
print(f"{'MLA +x0':>12} {t_x0:>10,} {emb_x0:>10,} {att_x0:>10,} {ffn_x0:>10,} {oth_x0:>10,}")
print(f"{'LLM_D3':>12} {t_d3:>10,} {emb_d3:>10,} {att_d3:>10,} {ffn_d3:>10,} {oth_d3:>10,}")
print(f"{'LLM_D5 (XSA)':>12} {t_d5:>10,} {emb_d5:>10,} {att_d5:>10,} {ffn_d5:>10,} {oth_d5:>10,}")
print(f"{'nano-moe-mla':>12} {t_nano:>10,} {emb_nano:>10,} {att_nano:>10,} {ffn_nano:>10,} {oth_nano:>10,}")
print()

opt_o = torch.optim.AdamW(model_ours.parameters(), lr=3e-4, weight_decay=0.01)
opt_x0 = torch.optim.AdamW(model_ours_x0.parameters(), lr=3e-4, weight_decay=0.01)
opt_d3 = torch.optim.AdamW(model_d3.parameters(), lr=3e-4, weight_decay=0.01)
opt_d5 = torch.optim.AdamW(model_d5.parameters(), lr=3e-4, weight_decay=0.01)
opt_nano = torch.optim.AdamW(model_nano.parameters(), lr=3e-4, weight_decay=0.01)

def grad_norms(model, tag=""):
    total_norm = 0.0
    norms = {}
    for n, p in model.named_parameters():
        if p.grad is not None:
            norm = p.grad.norm().item()
            norms[n] = norm
            total_norm += norm ** 2
    total_norm = total_norm ** 0.5
    top = sorted(norms.items(), key=lambda x: -x[1])[:5]
    return total_norm, top

@torch.no_grad()
def attn_entropy(model, x, tag=""):
    """Forward pass capturing per-layer attention entropy."""
    h = model.embedding(x)
    for li, layer in enumerate(model.transformer.layers):
        res = h
        h = layer.attn_norm(h)
        qkv = layer.attention.qkv
        Qs, Qr, K, V, Kr = qkv(h)
        Qr, Kr = layer.attention.rope(Qr, Kr, 0)
        Q = torch.cat([Qs, Qr], dim=-1)
        Kr_e = Kr.expand(-1, -1, layer.num_kv_groups, -1)
        K_ = torch.cat([K, Kr_e], dim=-1)
        k = repeat_kv(K_, layer.num_heads, layer.num_kv_groups)
        v = repeat_kv(V, layer.num_heads, layer.num_kv_groups)
        q = Q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(qkv.qk_dim)
        attn = F.softmax(scores, dim=-1)
        ent = -(attn * torch.log(attn.clamp(1e-10))).sum(-1).mean().item()
        print(f"  {tag} layer{li}: scores [{scores.min():.2f}, {scores.max():.2f}] mean={scores.mean():.2f}  ent={ent:.3f}")
        out = (attn @ v).transpose(1, 2)
        h = res + layer.attention.o_proj(out)
        res = h; h = layer.ffn_norm(h); h = layer.ffn(h); h = res + h
    h = model.transformer.final_norm(h)
    logits = model.head(h)
    print(f"  {tag} logits: mean={logits.mean():.3f} std={logits.std():.3f} max={logits.max():.3f}")
    return logits

print(f"{'step':>5}  {'MLA':>10}  {'MLA+x0':>10}  {'D3':>10}  {'D5(XSA)':>10}  {'nano':>10}")

# Diagnóstico: atención y logits en step 0
print("\n=== DIAG: forward pass (attention scores/entropy) ===")
attn_entropy(model_ours, x, "MLA")
print()
# Repetir forward de nano internamente
with torch.no_grad():
    h = model_nano.tok_emb(x)
    for li, blk in enumerate(model_nano.blocks):
        res = h; h = blk.norm1(h); cos, sin = model_nano.rope_cos[:x.shape[1]], model_nano.rope_sin[:x.shape[1]]
        q = blk.attn.w_q(h).view(x.shape[0], x.shape[1], 4, 80).transpose(1, 2)
        q_c, q_r = q[..., :64], q[..., 64:]
        c_kv = blk.attn.w_dkv(h)
        k_c = blk.attn.w_uk(c_kv).view(x.shape[0], x.shape[1], 4, 64).transpose(1, 2)
        v = blk.attn.w_uv(c_kv).view(x.shape[0], x.shape[1], 4, 64).transpose(1, 2)
        k_r = blk.attn.w_kr(h).view(x.shape[0], 1, x.shape[1], 16)
        q_r = _nano_mod.apply_rope(q_r, cos, sin)
        k_r = _nano_mod.apply_rope(k_r, cos, sin)
        scores = (q_c @ k_c.transpose(-2, -1) + q_r @ k_r.transpose(-2, -1)) / math.sqrt(80)
        attn = F.softmax(scores, dim=-1)
        ent = -(attn * torch.log(attn.clamp(1e-10))).sum(-1).mean().item()
        print(f"  nano layer{li}: scores [{scores.min():.2f}, {scores.max():.2f}] mean={scores.mean():.2f}  ent={ent:.3f}")
        h = res + blk.attn.w_o((attn @ v).transpose(1, 2).reshape(x.shape[0], x.shape[1], 256))
        res = h; h = blk.norm2(h); h = blk.moe(h); h = res + h
    h = model_nano.norm_f(h)
    logits = model_nano.lm_head(h)
    print(f"  nano logits: mean={logits.mean():.3f} std={logits.std():.3f} max={logits.max():.3f}")
print()

for step in range(10):

    opt_o.zero_grad()
    l = F.cross_entropy(model_ours(x).reshape(-1, vocab_size), y.reshape(-1))
    l.backward()
    g_o, top_o = grad_norms(model_ours)
    torch.nn.utils.clip_grad_norm_(model_ours.parameters(), 1.0); opt_o.step()
    l_o = l.item()

    opt_x0.zero_grad()
    l = F.cross_entropy(model_ours_x0(x).reshape(-1, vocab_size), y.reshape(-1))
    l.backward()
    g_x0, top_x0 = grad_norms(model_ours_x0)
    torch.nn.utils.clip_grad_norm_(model_ours_x0.parameters(), 1.0); opt_x0.step()
    l_x0 = l.item()

    opt_d3.zero_grad()
    out = model_d3(x, labels=y)
    l = F.cross_entropy(out.logits.reshape(-1, vocab_size), y.reshape(-1))
    l.backward(); torch.nn.utils.clip_grad_norm_(model_d3.parameters(), 1.0); opt_d3.step()
    l_d3 = l.item()

    opt_d5.zero_grad()
    out = model_d5(x, labels=y)
    l = F.cross_entropy(out.logits.reshape(-1, vocab_size), y.reshape(-1))
    l.backward(); torch.nn.utils.clip_grad_norm_(model_d5.parameters(), 1.0); opt_d5.step()
    l_d5 = l.item()

    opt_nano.zero_grad()
    _, l = model_nano(x, y)
    l.backward()
    g_nano, top_nano = grad_norms(model_nano)
    torch.nn.utils.clip_grad_norm_(model_nano.parameters(), 1.0); opt_nano.step()
    l_nano = l.item()

    print(f"{step+1:5d}  {l_o:10.6f}  {l_x0:10.6f}  {l_d3:10.6f}  {l_d5:10.6f}  {l_nano:10.6f}")

    if step == 0:
        print(f"\n  Grad norms (step 1, BEFORE clip): MLA total={g_o:.3f}  MLA+x0={g_x0:.3f}  nano={g_nano:.3f}")
        for tag, m in [("MLA", model_ours), ("nano", model_nano)]:
            print(f"  {tag} attention (before clip):")
            for n, p in m.named_parameters():
                if p.grad is not None and ('qkv' in n or 'o_proj' in n or 'W_' in n or 'w_' in n):
                    print(f"    {n.rsplit('.',1)[1]:20s} {p.grad.norm():.4f}")
