"""
Muon optimizer from scratch
===========================

AdamW adapts a per-weight step size from running gradient statistics. Muon (the optimizer
Moonshot used to train Kimi K2 — the first trillion-param model trained without AdamW) does
something different for 2D WEIGHT MATRICES: it takes the momentum and then ORTHOGONALIZES it
before stepping. "Orthogonalize" here = push the update's singular values toward 1 via a few
Newton-Schulz iterations (a matrix polynomial that needs no SVD). Intuition: it spreads the
update evenly across directions instead of letting a few dominate -> faster, more stable
training. Non-matrix params (norms, embeddings, biases) fall back to plain momentum/AdamW.

This module implements Muon, self-checks that it drives a tiny regression loss down, and
(via --benchmark, also run automatically after the self-test) trains the same MLP twice —
once with AdamW, once with Muon — to compare their loss curves and save `muon_vs_adamw.png`.

Run:  python muon.py
      python muon.py --benchmark
"""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


def zeropower_newtonschulz(G, steps=5):
    """Orthogonalize G (2D): return a matrix with ~the same column space but singular
    values ≈ 1, using `steps` Newton-Schulz iterations (no SVD). Coeffs from Keller Jordan."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    X = X / (X.norm() + 1e-7)                       # normalize so the iteration is stable
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T                                     # work on the shorter side
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X                           # the quintic NS step
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for grp in self.param_groups:
            for p in grp["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                buf = st.get("buf")
                if buf is None:
                    buf = st["buf"] = torch.zeros_like(p)
                buf.mul_(grp["momentum"]).add_(p.grad)      # momentum
                if p.ndim == 2:                             # MATRIX → orthogonalize the update
                    upd   = zeropower_newtonschulz(buf, grp["ns_steps"])
                    scale = max(p.size(0), p.size(1)) ** 0.5   # keep update magnitude sane
                    p.add_(upd, alpha=-grp["lr"] * scale)
                else:                                       # vector/scalar → plain momentum step
                    p.add_(buf, alpha=-grp["lr"])


def _make_task(d=32, n=256, seed=0):
    """A small linear-map regression task: recover W_true from (X, X @ W_true)."""
    torch.manual_seed(seed)
    W_true = torch.randn(d, d)
    X = torch.randn(n, d)
    Y = X @ W_true
    return X, Y


def _make_mlp(d=32, seed=0):
    torch.manual_seed(seed)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(d, d, bias=False)   # 2D weights → handled by Muon's orthogonalization
            self.l2 = nn.Linear(d, d, bias=False)

        def forward(self, x):
            return self.l2(torch.relu(self.l1(x)))

    return MLP()


def _train(model, opt, X, Y, steps=300):
    """Train `model` with `opt` for `steps` steps; return the per-step loss curve."""
    curve = []
    for _ in range(steps):
        loss = F.mse_loss(model(X), Y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        curve.append(loss.item())
    return curve


def benchmark(d=32, steps=300, out_path="muon_vs_adamw.png"):
    """Train the same MLP twice (AdamW vs Muon) on the same regression task and plot both
    loss curves on one axes. Saves `out_path` and prints the final losses."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X, Y = _make_task(d=d)

    # identical initialization for both runs (same seed) -> a fair comparison
    model_adamw = _make_mlp(d=d, seed=1)
    model_muon  = _make_mlp(d=d, seed=1)

    opt_adamw = torch.optim.AdamW(model_adamw.parameters(), lr=0.02)
    opt_muon  = Muon(model_muon.parameters(), lr=0.02)

    curve_adamw = _train(model_adamw, opt_adamw, X, Y, steps=steps)
    curve_muon  = _train(model_muon,  opt_muon,  X, Y, steps=steps)

    print("=== Benchmark: Muon vs AdamW ===")
    print(f"  AdamW  final loss: {curve_adamw[-1]:.4f}")
    print(f"  Muon   final loss: {curve_muon[-1]:.4f}")

    plt.figure(figsize=(7, 4.5))
    plt.plot(curve_adamw, label="AdamW", linewidth=1.6)
    plt.plot(curve_muon,  label="Muon",  linewidth=1.6)
    plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("MSE loss (log scale)")
    plt.title("Muon vs AdamW on a small MLP regression task")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  saved loss-curve plot -> {out_path}")
    return curve_adamw, curve_muon


# ----------------------------- TEST (self-checking) -----------------------------
if __name__ == "__main__":
    if "--benchmark" in sys.argv:
        benchmark()
        sys.exit(0)

    X, Y = _make_task(d=32)
    model = _make_mlp(d=32, seed=0)
    opt   = Muon(model.parameters(), lr=0.02)

    print("=== Muon optimizer (self-test) ===")
    loss0 = F.mse_loss(model(X), Y).item()
    for it in range(1, 301):
        loss = F.mse_loss(model(X), Y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if it % 100 == 0:
            print(f"  iter {it:3d}  loss {loss.item():.4f}")
    lossN = F.mse_loss(model(X), Y).item()

    print(f"loss: {loss0:.4f} → {lossN:.4f}")
    assert lossN < loss0 * 0.5, "Muon didn't reduce the loss enough"
    print("\nOK — Muon works: orthogonalized-momentum updates drive the loss down (no AdamW).")

    # also produce the standalone "Muon vs AdamW" comparison plot
    print()
    benchmark()
