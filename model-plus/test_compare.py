import sys, os, math, torch, torch.nn as nn, torch.nn.functional as F
import importlib.util
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.join(_DIR, "LLM_D3-main"))
from model import TransformerLM as TransformerLM_MLA
sys.path.insert(0, os.path.join(_DIR, ".."))
from python.model import TransformerLM as TransformerLM_Py
from LLM_2 import GPT, GPTConfig
_nano_path = os.path.join(_DIR, "nano-moe-mla", "steps", "03_block_model.py")
_nano_spec = importlib.util.spec_from_file_location("nano_block_model", _nano_path)
_nano_mod = importlib.util.module_from_spec(_nano_spec)
_nano_spec.loader.exec_module(_nano_mod)
MoeMlaGPT = _nano_mod.MoeMlaGPT
MoeMlaConfig = _nano_mod.MoeMlaConfig

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
ffn_inter = 640

def init_nano(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# ── 7 models ──────────────────────────────────────────────────────────────

model_ours = TransformerLM_MLA(
    vocab_size=vocab_size, d_model=d_model, num_layers=n_layer,
    num_heads=n_head, num_kv_groups=num_kv_groups, head_dim=head_dim,
    use_swiglu=True, use_x0=False, max_seq_len=seq_len,
    residual_dropout=0.0, attn_dropout=0.0, ffn_dropout=0.0,
    use_mla=True, mla_block_size=128,
    mla_d_c=d_c, mla_d_c1=d_c1, mla_d_rotate=d_rotate,
).to(device)
model_ours.apply(init_nano)

model_ours_x0 = TransformerLM_MLA(
    vocab_size=vocab_size, d_model=d_model, num_layers=n_layer,
    num_heads=n_head, num_kv_groups=num_kv_groups, head_dim=head_dim,
    use_swiglu=True, use_x0=True, max_seq_len=seq_len,
    residual_dropout=0.0, attn_dropout=0.0, ffn_dropout=0.0,
    use_mla=True, mla_block_size=128,
    mla_d_c=d_c, mla_d_c1=d_c1, mla_d_rotate=d_rotate,
).to(device)
model_ours_x0.apply(init_nano)

# MoE variants (use moe_block's init depth-scaling)
def init_depth_scaled(m, n_layers=1):
    for n, p in m.named_parameters():
        if 'embed' in n and p.ndim == 2:
            nn.init.normal_(p, mean=0, std=0.02)
        elif 'weight' in n and p.ndim >= 2:
            if 'down_proj' in n or 'o_proj' in n or 'c_proj' in n:
                std = 0.02 / math.sqrt(max(1, 2 * n_layers))
            else:
                std = 0.02
            nn.init.normal_(p, mean=0, std=std)
        elif 'bias' in n:
            nn.init.zeros_(p)

model_moe = TransformerLM_MLA(
    vocab_size=vocab_size, d_model=d_model, num_layers=n_layer,
    num_heads=n_head, num_kv_groups=num_kv_groups, head_dim=head_dim,
    use_swiglu=True, use_x0=False, max_seq_len=seq_len,
    residual_dropout=0.0, attn_dropout=0.0, ffn_dropout=0.0,
    use_mla=True, mla_block_size=128,
    mla_d_c=d_c, mla_d_c1=d_c1, mla_d_rotate=d_rotate,
    use_moe=True, n_experts=4, top_k=2, n_shared=0,
    z_loss_gamma=0.001, n_dense_start=0, n_dense_end=0,
).to(device)
init_depth_scaled(model_moe, n_layer)

model_moe_sh = TransformerLM_MLA(
    vocab_size=vocab_size, d_model=d_model, num_layers=n_layer,
    num_heads=n_head, num_kv_groups=num_kv_groups, head_dim=head_dim,
    use_swiglu=True, use_x0=False, max_seq_len=seq_len,
    residual_dropout=0.0, attn_dropout=0.0, ffn_dropout=0.0,
    use_mla=True, mla_block_size=128,
    mla_d_c=d_c, mla_d_c1=d_c1, mla_d_rotate=d_rotate,
    use_moe=True, n_experts=4, top_k=2, n_shared=1,
    z_loss_gamma=0.001, n_dense_start=0, n_dense_end=0,
).to(device)
init_depth_scaled(model_moe_sh, n_layer)

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

model_py = TransformerLM_Py(
    vocab_size=vocab_size, d_model=d_model, num_layers=n_layer,
    num_heads=n_head, num_kv_groups=num_kv_groups,
    use_swiglu=True, use_x0=False, max_seq_len=seq_len,
    residual_dropout=0.0, attn_dropout=0.0, ffn_dropout=0.0,
).to(device)
model_py.apply(init_nano)

# ── Parameter breakdown ───────────────────────────────────────────────────

def breakdown(m):
    all_names = set(n for n, _ in m.named_parameters())
    total = sum(p.numel() for p in m.parameters())
    emb = sum(p.numel() for n,p in m.named_parameters() if 'embed' in n or 'wte' in n or 'emb' in n)
    attn = sum(p.numel() for n,p in m.named_parameters() if 'attn' in n or 'Wq' in n or 'Wk' in n or 'Wv' in n or 'Wo' in n or 'k_proj' in n or 'v_proj' in n or 'q_proj' in n or 'o_proj' in n)
    moe = sum(p.numel() for n,p in m.named_parameters() if 'c_fc' in n or 'c_proj' in n or 'expert_bias' in n or 'router' in n or ('shared' in n and 'ffn' in n))
    ffn = sum(p.numel() for n,p in m.named_parameters() if ('ffn' in n or 'mlp' in n or 'fc1' in n or 'fc2' in n or 'gate_proj' in n or 'up_proj' in n or 'down_proj' in n)) - moe
    other = total - emb - attn - ffn - moe
    return total, emb, attn, ffn, moe, other

models = [
    ("MLA dense", model_ours),
    ("MLA+x0", model_ours_x0),
    ("MLA+MoE", model_moe),
    ("MLA+MoE+sh", model_moe_sh),
    ("LLM_D3", model_d3),
    ("D5(XSA)", model_d5),
    ("nano-mla", model_nano),
    ("Py GQA", model_py),
]

fsize = os.path.getsize(txt)
print(f"Archivo: {os.path.basename(txt)} ({fsize:,} bytes)")
print(f"Tokens totales: {len(tokens_1d):,}  |  Vocab: {vocab_size} chars")
print(f"Batch: {batch_size}x{seq_len} = {batch_size*seq_len} tokens/step")
print(f"d_model={d_model} layers={n_layer} heads={n_head}")
print(f"MLA: d_c={d_c} d_c1={d_c1} d_rotate={d_rotate}  |  MoE: n_exp=4 top_k=2 shared={1 if model_moe_sh is not None else 0}")
print()

hdr = f"{'':>14} {'total':>10} {'emb':>10} {'attn':>10} {'ffn':>10} {'moe':>10} {'other':>10}"
print(hdr)
for name, m in models:
    t, emb, attn, ffn, moe, oth = breakdown(m)
    print(f"{name:>14} {t:>10,} {emb:>10,} {attn:>10,} {ffn:>10,} {moe:>10,} {oth:>10,}")
print()

opts = [torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01) for _, m in models]

# ── Training ──────────────────────────────────────────────────────────────

hdr2 = f"{'step':>5}  " + "  ".join(f"{name:>10}" for name, _ in models)
print(hdr2)

for step in range(10):
    losses = []
    for (name, m), opt in zip(models, opts):
        opt.zero_grad()
        aux = 0.0
        if name in ("LLM_D3", "D5(XSA)"):
            out = m(x, labels=y)
            logits = out.logits
        elif name == "nano-mla":
            _, loss = m(x, y)
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            continue
        else:
            out = m(x)
            if isinstance(out, torch.Tensor):
                logits, aux = out, 0.0
            else:
                logits, aux = out
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        if isinstance(aux, torch.Tensor):
            loss = loss + aux
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    print(f"{step+1:5d}  " + "  ".join(f"{l:10.6f}" for l in losses))
