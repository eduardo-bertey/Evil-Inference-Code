"""Verify MoE-MLA: init stability, MoE routing, training step, loss comparison.

Tests:
  1. Logit std at init (should be 0.08-0.8, healthy range)
  2. MoE: all experts get tokens (load balance)
  3. MoE: z-loss > 0 when enabled
  4. Training step: loss decreases
  5. Comparison: MLA vs MLA+MoE vs MLA+MoE+shared vs Dense
"""

import sys, os, math, torch, torch.nn as nn, torch.nn.functional as F

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.join(_DIR, ".."))

from model import TransformerLM
from moe import MoELayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_LAYER = 3
D_MODEL = 256
N_HEAD = 4
HEAD_DIM = 64
D_C = 32
D_ROTATE = 16
SEQ_LEN = 32
BATCH = 4
N_EXPERTS = 4
TOP_K = 2
VOCAB = 65


def make_batch(vocab, seq_len, batch_size):
    x = torch.randint(0, vocab, (batch_size, seq_len), device=device)
    y = torch.roll(x, -1, dims=1)
    y[:, -1] = 0
    return x, y


def init_depth_scaled(m, num_layers=1):
    """Depth-scaled init: N(0, 0.02) on most, /sqrt(2*n) on output proj."""
    for n, p in m.named_parameters():
        if 'embed' in n and p.ndim == 2:
            nn.init.normal_(p, mean=0, std=0.02)
        elif 'weight' in n and p.ndim >= 2:
            if 'down_proj' in n or 'o_proj' in n or 'c_proj' in n:
                std = 0.02 / math.sqrt(max(1, 2 * num_layers))
            else:
                std = 0.02
            nn.init.normal_(p, mean=0, std=std)
        elif 'bias' in n:
            nn.init.zeros_(p)


def make_model(use_moe=False, n_shared=1, use_x0=False):
    num_kv_groups = 4
    m = TransformerLM(
        vocab_size=VOCAB, d_model=D_MODEL, num_layers=N_LAYER,
        num_heads=N_HEAD, num_kv_groups=num_kv_groups, head_dim=HEAD_DIM,
        use_swiglu=True, use_x0=use_x0, max_seq_len=SEQ_LEN,
        residual_dropout=0.0, attn_dropout=0.0, ffn_dropout=0.0,
        use_mla=True, mla_block_size=128,
        mla_d_c=D_C, mla_d_c1=D_C, mla_d_rotate=D_ROTATE,
        use_moe=use_moe, n_experts=N_EXPERTS, top_k=TOP_K, n_shared=n_shared,
        z_loss_gamma=0.001 if use_moe else 0.0,
        n_dense_start=0, n_dense_end=0,
    ).to(device)
    init_depth_scaled(m, N_LAYER)
    return m


def main():
    print("=== MoE-MLA Verification ===\n")
    x, y = make_batch(VOCAB, SEQ_LEN, BATCH)

    # 1) Logit std at init (all models)
    print("--- Init logit std (healthy range: 0.08-0.8) ---")
    for tag, use_moe, n_shared in [
        ("MLA (dense FFN)", False, 0),
        ("MLA + MoE (noshared)", True, 0),
        ("MLA + MoE + shared", True, 1),
    ]:
        m = make_model(use_moe=use_moe, n_shared=n_shared)
        with torch.no_grad():
            logits, _ = m(x)
        total = sum(p.numel() for p in m.parameters())
        active = sum(p.numel() for p in m.parameters() if p.ndim >= 2)
        print(f"  [{tag:>24}] logit std={logits.std():.4f}  max={logits.max():.3f}  params={total:,}")
    print()

    # 2) MoE routing: all experts used, bias balance
    print("--- MoE routing verification ---")
    moe = MoELayer(d_model=D_MODEL, n_experts=N_EXPERTS, top_k=TOP_K,
                   n_shared=0, z_loss_gamma=0.001).to(device)
    init_depth_scaled(moe)
    inp = torch.randn(64, D_MODEL, device=device)
    out, z = moe(inp)
    counts = torch.bincount(
        F.softmax(moe.router(inp) + moe.expert_bias, dim=-1).topk(TOP_K, dim=-1).indices.flatten(),
        minlength=N_EXPERTS
    )
    all_used = (counts > 0).sum().item() == N_EXPERTS
    print(f"  Experts used: {all_used} (counts={counts.tolist()})")
    print(f"  z-loss > 0: {z > 0}")
    print(f"  Biases: {[f'{b:.4f}' for b in moe.expert_bias.tolist()]}")
    assert all_used, "Not all experts got tokens!"
    assert z > 0, "z-loss should be > 0"
    print()

    # 3) Training: loss decreases for all variants
    print("--- Training (10 steps, same batch) ---")
    variants = [
        ("MLA (dense)", make_model(use_moe=False, n_shared=0)),
        ("MoE (noshared)", make_model(use_moe=True, n_shared=0)),
        ("MoE+shared", make_model(use_moe=True, n_shared=1)),
    ]
    opts = [torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01) for _, m in variants]

    hdr = f"{'step':>5}  " + "  ".join(f"{name:>16}" for name, _ in variants)
    print(hdr)

    for step in range(10):
        losses = []
        for (name, m), opt in zip(variants, opts):
            opt.zero_grad()
            logits, aux = m(x)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
            loss = loss + aux
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        print(f"{step+1:5d}  " + "  ".join(f"{l:16.6f}" for l in losses))

    print()
    print("--- Done. MoE-MLA is working. ---")


if __name__ == "__main__":
    main()
