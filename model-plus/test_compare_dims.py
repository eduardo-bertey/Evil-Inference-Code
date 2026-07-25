"""Compare MoE vs dense at different FFN dimensions.

Tests whether the router learns to route intelligently even when
expert capacity is limited (small expert_dim relative to dense_dim).
"""

import sys, os, math, torch, torch.nn as nn, torch.nn.functional as F

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.join(_DIR, ".."))

from model import TransformerLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_LAYER = 3
D_MODEL = 64
N_HEAD = 4
HEAD_DIM = D_MODEL // N_HEAD
D_C = 24
D_ROTATE = 8
SEQ_LEN = 32
BATCH = 8
N_EXPERTS = 4
TOP_K = 1
N_SHARED = 1
VOCAB = 65
STEPS = 100
STEPS_TO_LOG = [1, 5, 10, 25, 50, 100]


def make_batch():
    x = torch.randint(0, VOCAB, (BATCH, SEQ_LEN), device=device)
    y = torch.roll(x, -1, dims=1)
    y[:, -1] = 0
    return x, y


def make_model(dense_dim, moe_dim, use_moe=False):
    ffn_exp = dense_dim * 3.0 / 2.0 / D_MODEL if dense_dim else 2.0
    m = TransformerLM(
        vocab_size=VOCAB, d_model=D_MODEL, num_layers=N_LAYER,
        num_heads=N_HEAD, num_kv_groups=N_HEAD, head_dim=HEAD_DIM,
        use_swiglu=True, use_x0=False, max_seq_len=SEQ_LEN,
        ffn_expansion=ffn_exp,
        residual_dropout=0.0, attn_dropout=0.0, ffn_dropout=0.0,
        use_mla=True, mla_block_size=128,
        mla_d_c=D_C, mla_d_c1=D_C, mla_d_rotate=D_ROTATE,
        use_moe=use_moe, n_experts=N_EXPERTS, top_k=TOP_K,
        n_shared=N_SHARED if use_moe else 0,
        expert_dim=moe_dim if use_moe else None,
        z_loss_gamma=0.001 if use_moe else 0.0,
        n_dense_start=0, n_dense_end=0,
    ).to(device)
    return m


def macs_per_token(m):
    d = D_MODEL
    l0 = m.transformer.layers[0]
    head_dim = l0.head_dim
    n_heads = l0.num_heads
    n_kv = l0.num_kv_groups
    nl = m.num_layers
    kv_dim = n_kv * head_dim
    q_dim = n_heads * head_dim
    attn_proj = 2 * (d * kv_dim) + d * q_dim + d * q_dim
    attn_score = 2 * SEQ_LEN * q_dim
    attn_weight = 2 * SEQ_LEN * q_dim
    attn = attn_proj + attn_score + attn_weight
    total = 0
    for layer in m.transformer.layers:
        macs = attn
        is_moe = getattr(layer, 'use_moe', False)
        if is_moe:
            exp_dim = layer.ffn.c_fc.shape[-1] // 2
            topk = layer.ffn.top_k
            macs += 2 * 3 * d * exp_dim * topk
            if layer.ffn.n_shared > 0:
                macs += 2 * 3 * d * exp_dim
        else:
            inter_dim = layer.ffn.gate_proj.weight.shape[0]
            macs += 2 * 3 * d * inter_dim
        total += macs
    return total // nl


x, y = make_batch()

configs = [
    (48, None, False, "dense 48"),
    (48, 48, True, "moe48 t1"),
    (48, 32, True, "moe32 t1"),
    (48, 24, True, "moe24 t1"),
    (48, 16, True, "moe16 t1"),
    (32, None, False, "dense 32"),
    (32, 32, True, "moe32 t1"),
    (32, 24, True, "moe24 t1"),
    (32, 16, True, "moe16 t1"),
    (24, None, False, "dense 24"),
    (24, 24, True, "moe24 t1"),
    (24, 16, True, "moe16 t1"),
    (16, None, False, "dense 16"),
    (16, 16, True, "moe16 t1"),
]

ref_dims = sorted(set(dd for dd, _, _, _ in configs))

print(f"d_model={D_MODEL}  layers={N_LAYER}  heads={N_HEAD}  kv={N_HEAD}")
print(f"n_exp={N_EXPERTS}  top_k={TOP_K}  n_shared={N_SHARED}  steps={STEPS}")
print(f"batch={BATCH}x{SEQ_LEN}  vocab={VOCAB}")
print()

for ref_d in ref_dims:
    group = [(lbl, dd, md, moe) for dd, md, moe, lbl in configs if dd == ref_d]
    dense_lbl = [lbl for lbl, _, md, moe in group if not moe][0]

    print("=" * 70)
    print(f"  dense_ffn={ref_d}")
    print("=" * 70)
    hdr = f"{'model':>18} {'params':>8} {'MACs/tok':>9}  " + "  ".join(f"s{s:>3}" for s in STEPS_TO_LOG)
    print(hdr)

    group_results = []
    for lbl, dd, md, moe in group:
        m = make_model(dd, md, use_moe=moe)
        p = sum(p.numel() for p in m.parameters())
        macs = macs_per_token(m)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
        m.train()
        losses = []
        for step in range(STEPS):
            opt.zero_grad()
            logits, aux = m(x)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
            if isinstance(aux, torch.Tensor):
                loss = loss + aux
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            if step + 1 in STEPS_TO_LOG:
                losses.append(loss.item())
        row = f"{lbl:>18} {p:>8,} {macs:>9,}  " + "  ".join(f"{l:10.6f}" for l in losses)
        print(row)
        group_results.append((lbl, losses))

    dense_final = [r for l, r in group_results if l == dense_lbl][0][-1]
    for lbl, losses in group_results:
        if lbl == dense_lbl:
            continue
        diff = ((dense_final - losses[-1]) / dense_final) * 100
        marker = "WIN" if diff > 0 else "LOSE"
        print(f"    -> {lbl}: {diff:+.1f}% vs {dense_lbl} {marker}")
    print()
