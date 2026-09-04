"""MoE sobre un Linear: E expertos ruteados (top-k) + 1 compartido fijo.

Todo PyTorch estándar, sin triton/flash/CUDA avanzado. Token-wise:
compatible con cache sin código extra.

- y = compartido(x) + Σ_{e en topk} w_e · experto_e(x)
- Router: w_gate (+ w_noise con ruido solo en train), softmax,
  top-k SIN renormalizar.
- Aux (self.last_aux): cv (CV²) + switch (probs·freqs×E) + z.
"""

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoELineal(nn.Module):
    def __init__(self, dim_in, dim_out, config, residual=False):
        super().__init__()
        self.num_expertos = getattr(config, "moe_expertos", 4)
        self.topk = min(max(1, getattr(config, "moe_topk", 2)), self.num_expertos)
        self.cv_w = getattr(config, "moe_aux_w", 0.01)
        self.switch_w = getattr(config, "moe_aux_w", 0.01)
        self.z_w = getattr(config, "moe_z_w", 0.001)
        self.ruido = getattr(config, "moe_ruido", True)

        self.expertos = nn.ModuleList(
            [nn.Linear(dim_in, dim_out, bias=False) for _ in range(self.num_expertos)])
        # Compartido fijo: siempre activo (estilo DeepSeek/Qwen).
        self.compartido = nn.Linear(dim_in, dim_out, bias=False)
        self.w_gate = nn.Parameter(torch.zeros(dim_in, self.num_expertos))
        self.w_noise = nn.Parameter(torch.zeros(dim_in, self.num_expertos))
        for e in self.expertos:
            if residual:
                e.is_residual_proj = True
            else:
                e.is_attention = True
        if residual:
            self.compartido.is_residual_proj = True
        else:
            self.compartido.is_attention = True
        self.last_aux = torch.tensor(0.0)

    @staticmethod
    def _cv_cuadrado(x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor(0.0, device=x.device)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def forward(self, x):
        orig = x.shape
        xf = x.reshape(-1, orig[-1])                       # [N,Din]
        logits_r = xf @ self.w_gate
        if self.ruido and self.training:
            ruido_std = F.softplus(xf @ self.w_noise) + 1e-2
            logits_r = logits_r + torch.randn_like(logits_r) * ruido_std
        probs = torch.softmax(logits_r.float(), dim=-1).to(x.dtype)
        top_w, top_i = torch.topk(probs, k=self.topk, dim=-1)   # sin renormalizar

        out = self.compartido(xf)
        for e, exp in enumerate(self.expertos):
            w = (top_w * (top_i == e)).sum(dim=-1)         # [N] peso (0 si no elegido)
            m = w != 0
            if not m.any():
                continue
            out[m] += w[m].unsqueeze(-1) * exp(xf[m])

        if self.training:
            with torch.no_grad():
                suma = probs.sum(0)
                cv = self._cv_cuadrado(F.normalize(suma, p=1, dim=0))
                freqs = top_i.reshape(-1).bincount(minlength=self.num_expertos).float()
                self.last_dist = (freqs / freqs.sum().clamp_min(1e-9)).detach()
                switch = (F.normalize(suma, p=1, dim=0)
                          * F.normalize(freqs, p=1, dim=0)).sum() * self.num_expertos
            z = torch.logsumexp(logits_r.float(), dim=-1).pow(2).mean().to(x.dtype)
            self.last_aux = self.cv_w * cv.to(x.dtype) + self.switch_w * switch.to(x.dtype) + self.z_w * z
        else:
            self.last_aux = torch.tensor(0.0, device=x.device)
        return out.view(*orig[:-1], -1)
