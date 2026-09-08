"""TST (Token Superposition Training) — solo loss, variante output-only.

Predice los próximos `n_predict` tokens desde el mismo head con un solo
cross-entropy pesado (sin parámetros extra), y decae de vuelta a predicción
next-token estándar. Portado de nanogpt-aurora-tst (src/losses/tst.py).

- El vector de pesos mide SIEMPRE `n_predict`; la cola decae a 0 en vez de
  achicar el tensor (sin recompilaciones).
- Con cola en 0 el loss es bit a bit CE estándar (fase recovery = baseline).
- Opera sobre logits aplanados [N, V] y targets en orden de stream.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass
class TSTConfig:
    """Schedule del loss multi-token.

    enabled=False -> CE next-token estándar (baseline / brazo sin TST).
    mode="smooth"  -> la cola decae continuo a 0 (queda recovery_frac final como CE puro).
    mode="hard"    -> pesos bag completos hasta superposition_frac, luego CE estándar.
    """
    enabled: bool = False
    n_predict: int = 3            # tamaño del bag (máx tokens futuros predichos)
    base: float = 0.5             # base geométrica: peso[k] ~ base**k antes de decaer
    mode: str = "smooth"          # "smooth" | "hard"
    superposition_frac: float = 0.3  # usado por "hard"
    recovery_frac: float = 1.0 / 3.0  # usado por "smooth"

    def __post_init__(self):
        if self.n_predict < 1:
            raise ValueError("n_predict must be >= 1")
        if self.mode not in ("smooth", "hard"):
            raise ValueError("mode must be 'smooth' or 'hard'")
        if not (0.0 < self.recovery_frac < 1.0):
            raise ValueError("recovery_frac must be in (0, 1)")
        if not (0.0 < self.superposition_frac < 1.0):
            raise ValueError("superposition_frac must be in (0, 1)")


def mtp_weights_for_step(cfg: TSTConfig, step: int, total_steps: int, device) -> Tensor:
    """Vector de pesos longitud `n_predict` para este step (longitud estática)."""
    P = cfg.n_predict
    base_w = [cfg.base ** k for k in range(P)]  # ej. [1, 0.5, 0.25]
    if not cfg.enabled or P == 1:
        w = [1.0] + [0.0] * (P - 1)
        return torch.tensor(w, dtype=torch.float32, device=device)

    progress = min(max(step / max(total_steps, 1), 0.0), 1.0)
    w = list(base_w)

    if cfg.mode == "hard":
        if progress >= cfg.superposition_frac:
            w = [1.0] + [0.0] * (P - 1)
        return torch.tensor(w, dtype=torch.float32, device=device)

    # smooth: la cola decae a 0 en el primer (1 - recovery_frac) del train
    # (un segmento igual por orden, el mayor primero), luego CE puro.
    decay_end = max(1e-9, 1.0 - cfg.recovery_frac)
    if progress >= decay_end or P < 2:
        return torch.tensor([1.0] + [0.0] * (P - 1), dtype=torch.float32, device=device)
    seg = decay_end / (P - 1)
    for k in range(P - 1, 0, -1):
        seg_idx = P - 1 - k
        lo, hi = seg_idx * seg, (seg_idx + 1) * seg
        if progress >= hi:
            w[k] = 0.0
        elif progress < lo:
            w[k] = base_w[k]
        else:
            frac = (progress - lo) / seg
            w[k] = base_w[k] * (1.0 - frac)
    w[0] = 1.0
    return torch.tensor(w, dtype=torch.float32, device=device)


def multi_token_ce(logits_flat: Tensor, targets_flat: Tensor, mtp_weights: Tensor) -> Tensor:
    """Cross-entropy pesado next-bag, sumado sobre tokens.

    logits_flat:  [N, V] float (pasar en float32).
    targets_flat: [N] int64, targets next-token en orden de stream.
    mtp_weights:  [P] float; peso de predecir el (k+1)-ésimo token futuro.

    Con mtp_weights = [1, 0, ..., 0] equivale a
    F.cross_entropy(logits_flat, targets_flat, reduction="sum").
    """
    P = mtp_weights.numel()
    if P == 1:
        return F.cross_entropy(logits_flat, targets_flat, reduction="sum")

    N = targets_flat.size(0)
    padded = F.pad(targets_flat, (0, P - 1))          # [N + P - 1]
    idx = padded.unfold(0, P, 1)                        # [N, P]: ventana de P targets
    target_logits = logits_flat.gather(1, idx)         # [N, P]
    lse = torch.logsumexp(logits_flat, dim=-1, keepdim=True)  # [N, 1]
    ce = lse - target_logits                            # [N, P]
    # Las últimas k posiciones no tienen (k+1)-ésimo futuro válido: se enmascaran.
    for k in range(1, P):
        ce[max(N - k, 0):, k] = 0.0
    return (ce * mtp_weights).sum()
