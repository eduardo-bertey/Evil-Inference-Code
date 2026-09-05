"""MoSE: Mixture of Slimmable Experts para el FFN.

Cada experto es slimmable: una sola matriz up [I,D] / down [D,I] de la
que se usa el prefijo segun el ancho:
  25% -> I*0.25 | 50% -> I*0.50 | 75% -> I*0.75 | 100% -> I
D queda fijo; cada proyeccion usa ~width de pesos/FLOPs.

El router decide experto + ancho por token. Para que entrene con TODAS
las dims, dos mecanismos:
  1) en train, cada forward muestrea width = random.choice(widths)
     (forzado global) y el router elige experto dentro de ese ancho;
  2) perdida auxiliar de balanceo (estilo Switch, CV^2) sobre expertos
     y sobre anchos para que el router no colapse a un modo.
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class SlimmableExpert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        self.down.is_residual_proj = True

    def forward(self, x, width):
        # width = 0.25, 0.50, 0.75 o 1.0
        d = max(8, int(self.up.out_features * width))
        # 1) D -> d (prefijo de filas de up)
        h = F.linear(x, self.up.weight[:d])
        h = F.silu(h)
        # 2) d -> D (prefijo de columnas de down)
        y = F.linear(h, self.down.weight[:, :d])
        return y


class MoSE(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config.dim
        d_ff = config.ffn_dim
        self.n_experts = getattr(config, "mose_expertos", 4)
        self.widths = list(getattr(config, "mose_widths", (0.25, 0.50, 0.75, 1.0)))
        self.aux_w = getattr(config, "mose_aux_w", 0.01)

        self.experts = nn.ModuleList([
            SlimmableExpert(d_model, d_ff) for _ in range(self.n_experts)
        ])
        # Router: experto x ancho.
        self.router = nn.Linear(d_model, self.n_experts * len(self.widths))
        self.dropout = nn.Dropout(config.drop)
        self.last_aux = torch.tensor(0.0)

    def _balance_loss(self, probs_ew):
        """CV^2 sobre expertos y sobre anchos (uso uniforme)."""
        flat = probs_ew.reshape(-1, self.n_experts, len(self.widths))
        uso_e = flat.mean(dim=(0, 2))  # [E]
        uso_w = flat.mean(dim=(0, 1))  # [W]
        cv_e = (uso_e.std(unbiased=False) / (uso_e.mean() + 1e-9)) ** 2
        cv_w = (uso_w.std(unbiased=False) / (uso_w.mean() + 1e-9)) ** 2
        return (cv_e + cv_w) * self.aux_w

    def forward(self, x, width=None):
        logits = self.router(x).view(
            *x.shape[:-1], self.n_experts, len(self.widths))

        if width is not None:
            # Ancho forzado (muestreo de train): el router elige experto.
            w = self.widths.index(width) if width in self.widths else len(self.widths) - 1
            probs_e = F.softmax(logits[..., w], dim=-1)
            expert_idx = probs_e.argmax(-1)
            probs = torch.zeros_like(logits)
            probs[..., w] = probs_e
            width_idx = torch.full_like(expert_idx, w)
        else:
            # Router libre: experto + ancho por token.
            flat = logits.flatten(-2)
            probs_flat = F.softmax(flat, dim=-1)
            probs = probs_flat.view_as(logits)
            idx = probs_flat.argmax(-1)
            expert_idx = idx // len(self.widths)
            width_idx = idx % len(self.widths)

        self.last_aux = self._balance_loss(probs.detach())

        output = torch.zeros_like(x)
        for e in range(self.n_experts):
            for wi, w in enumerate(self.widths):
                mask = (expert_idx == e) & (width_idx == wi)
                if mask.any():
                    y = self.experts[e](x[mask], w)
                    output[mask] += probs[..., e, wi][mask].unsqueeze(-1) * y
        return self.dropout(output)

    @staticmethod
    def sample_width(widths):
        return random.choice(list(widths))
