"""
Multi-Token Prediction (MTP) from scratch
=========================================

A normal LM head predicts the NEXT token (position t+1). MTP adds a SECOND head that, from
the same hidden state, predicts the token TWO ahead (t+2). Training on both gives a denser
learning signal (DeepSeek-V3 used it in pretraining), and at inference it lets you DRAFT two
tokens per step (speculative-style speedups).

    loss = CE(head1, x[t+1])  +  lambda * CE(head2, x[t+2])

This module is self-contained: the small transformer pieces it needs (RMSNorm, a GQA
attention, a SwiGLU "Expert" FFN, and the RoPE helpers) are inlined below. It builds a tiny
two-head transformer, trains it on a small text, and checks that BOTH losses start near
ln(vocab) and drop together.

Run:  python mtp.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===== inlined pieces (RMSNorm, RoPE, GQA, SwiGLU Expert) =====
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        msq = x.pow(2).mean(dim=-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(msq + self.eps))


def build_rope_cache(dim, seq_len, base=10000.0, device=None):
    idx      = torch.arange(0, dim, 2, device=device).float()
    inv_freq = 1.0 / (base ** (idx / dim))
    t        = torch.arange(seq_len, device=device).float()
    freqs    = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    return torch.stack((rx1, rx2), dim=-1).flatten(-2)


class Expert(nn.Module):
    """A SwiGLU feed-forward network (the FFN sub-layer)."""
    def __init__(self, n_embd):
        super().__init__()
        hidden = 64 * ((int(2 / 3 * 4 * n_embd) + 63) // 64)
        self.w_gate = nn.Linear(n_embd, hidden, bias=False)
        self.w_up   = nn.Linear(n_embd, hidden, bias=False)
        self.w_down = nn.Linear(hidden, n_embd, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class GQA(nn.Module):
    """Grouped-query attention: K/V are shared across query heads (n_kv_head < n_head),
    RoPE is applied to the full head_dim, with optional per-head QK-Norm for stability."""
    def __init__(self, n_embd, n_head, n_kv_head, head_dim, qk_norm=True):
        super().__init__()
        self.nh, self.nkv, self.hd = n_head, n_kv_head, head_dim
        self.rep = n_head // n_kv_head
        self.w_q = nn.Linear(n_embd, n_head    * head_dim, bias=False)
        self.w_k = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)
        self.w_v = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)
        self.w_o = nn.Linear(n_head * head_dim, n_embd, bias=False)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.w_q(x).view(B, T, self.nh,  self.hd).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)     # RoPE on the full head_dim
        k = k.repeat_interleave(self.rep, dim=1)                    # GQA: share K/V across query heads
        v = v.repeat_interleave(self.rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, self.nh * self.hd)
        return self.w_o(out)


# ===== the MTP model =====
class Block(nn.Module):
    """A plain transformer block (GQA + SwiGLU), pre-norm + residual."""
    def __init__(self, n_embd, n_head, n_kv_head, head_dim):
        super().__init__()
        self.norm1 = RMSNorm(n_embd); self.attn = GQA(n_embd, n_head, n_kv_head, head_dim)
        self.norm2 = RMSNorm(n_embd); self.ffn  = Expert(n_embd)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x


class TinyMTP(nn.Module):
    def __init__(self, vocab, n_embd=64, n_head=4, n_kv_head=2, head_dim=16, n_layer=2,
                 block_size=128, mtp_weight=0.5):
        super().__init__()
        self.mtp_weight = mtp_weight
        self.tok = nn.Embedding(vocab, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, n_kv_head, head_dim) for _ in range(n_layer)])
        self.norm = RMSNorm(n_embd)
        self.head1 = nn.Linear(n_embd, vocab, bias=False)   # predicts t+1 (the normal head)
        self.head2 = nn.Linear(n_embd, vocab, bias=False)   # predicts t+2 (the MTP head)
        cos, sin = build_rope_cache(head_dim, block_size)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, idx, t1=None, t2=None):
        B, T = idx.shape
        x = self.tok(idx)
        for blk in self.blocks:
            x = blk(x, self.cos[:T], self.sin[:T])
        x = self.norm(x)
        l1, l2 = self.head1(x), self.head2(x)               # both from the SAME hidden state
        loss = None
        if t1 is not None:
            loss1 = F.cross_entropy(l1.reshape(B * T, -1), t1.reshape(B * T))
            loss2 = F.cross_entropy(l2.reshape(B * T, -1), t2.reshape(B * T))
            loss = loss1 + self.mtp_weight * loss2
            return l1, l2, loss, (loss1.item(), loss2.item())
        return l1, l2, loss, None


# ----------------------------- TEST (self-checking) -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text = ("the mixture of experts routes each token; latent attention compresses the past. "
            "sparse where it can be, dense where it must. ") * 300
    chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
    vocab = len(chars)
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long, device=device)
    T = 64

    def batch(bs=32):
        ix = torch.randint(len(data) - T - 2, (bs,))
        x  = torch.stack([data[i:i + T]         for i in ix])     # input
        t1 = torch.stack([data[i + 1:i + 1 + T] for i in ix])     # target +1
        t2 = torch.stack([data[i + 2:i + 2 + T] for i in ix])     # target +2 (MTP)
        return x, t1, t2

    model = TinyMTP(vocab).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    print("=== Multi-Token Prediction (self-test) ===")
    x, t1, t2 = batch()
    _, _, _, (l1_0, l2_0) = model(x, t1, t2)
    print(f"initial: loss(+1) {l1_0:.3f}   loss(+2) {l2_0:.3f}   ln(vocab) {math.log(vocab):.3f}")

    steps, c1, c2 = [], [], []                                    # track both losses for the plot
    for it in range(1, 801):
        x, t1, t2 = batch()
        _, _, loss, (li1, li2) = model(x, t1, t2)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if it % 20 == 0:
            steps.append(it); c1.append(li1); c2.append(li2)

    x, t1, t2 = batch()
    _, _, _, (l1_1, l2_1) = model(x, t1, t2)
    print(f"trained: loss(+1) {l1_1:.3f}   loss(+2) {l2_1:.3f}")
    # the +2 task is genuinely harder → its loss stays higher, but BOTH must drop
    assert l1_1 < l1_0 - 0.3 and l2_1 < l2_0 - 0.2, "MTP losses didn't drop"

    # save the artifact: both heads' loss curves dropping together
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.plot(steps, c1, label="head 1  (t+1)", linewidth=1.6)
    plt.plot(steps, c2, label="head 2  (t+2, MTP)", linewidth=1.6)
    plt.xlabel("step"); plt.ylabel("cross-entropy"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.title("Multi-Token Prediction — both heads' losses drop")
    plt.tight_layout(); plt.savefig("mtp_losses.png", dpi=120); plt.close()
    print("saved loss-curve plot -> mtp_losses.png")
    print("\nOK — MTP works: two heads (t+1 and t+2), both losses drop. The +2 head adds a denser signal.")
