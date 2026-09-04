"""Atenciones PLUS para hrm_plus: MoA-lite y Keyless.

Todo en PyTorch estándar (nn.Linear + scaled_dot_product_attention).
SIN triton, SIN flash-attention explícito, SIN kernels CUDA propios:
corre en cualquier GPU normal (T4 incluida) y en CPU.

- KeylessAttention: sin proyección K. Q' = X·WQ·WR y scores contra V
  (ver ../../MoA/keylees.md). Cache de solo-V en decoding.
- MoAAttention: E expertos de atención (idea de MoA: yikangshen/MoA)
  con router top-k por token + pérdida auxiliar de balanceo
  (estilo Switch) + z-loss. Cada experto calcula su propia atención
  GQA completa y la salida se pondera por el router.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from rope import RoPE


def repeat_kv(x, num_heads, num_kv_groups):
    if num_kv_groups == num_heads:
        return x
    return x.repeat_interleave(num_heads // num_kv_groups, dim=2)


class KeylessAttention(nn.Module):
    """Atención sin claves: O = softmax(X·WQ·WR·(X·WV)ᵀ/√d)·X·WV.

    - WR por cabeza (head_dim × head_dim), aprendida.
    - RoPE se aplica al par (Q', V) con el mismo offset: el score
      q'·v queda con posición relativa y el cache sigue siendo solo V.
    - En inferencia se puede fusionar WQ_eff = WQ·WR (ver metodo
      fusionar_wq()).
    """

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.heads
        self.num_kv_groups = getattr(config, "kv_groups", config.heads)
        self.head_dim = config.dim // config.heads

        self.q_proj = nn.Linear(config.dim, self.num_heads * self.head_dim, bias=False)
        self.wr = nn.Parameter(torch.empty(self.num_heads, self.head_dim, self.head_dim))
        self.v_proj = nn.Linear(config.dim, self.num_kv_groups * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.rope = RoPE(self.head_dim, rotary_pct=getattr(config, "rotary_pct", 0.25))
        self.attn_dropout = nn.Dropout(config.drop)

        self.q_proj.is_attention = True
        self.v_proj.is_attention = True
        self.o_proj.is_residual_proj = True
        nn.init.trunc_normal_(self.wr, std=0.02)

    def _qp(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        # Q' = Q · WR (por cabeza)
        qp = torch.einsum("bthd,hde->bthe", q, self.wr)
        return qp

    def forward(self, x):
        B, T, D = x.shape
        qp = self._qp(x)
        v = self.v_proj(x).view(B, T, self.num_kv_groups, self.head_dim)

        qp, v = self.rope(qp, v, 0)

        v = repeat_kv(v, self.num_heads, self.num_kv_groups)
        q = qp.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, v, v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(out)

    def forward_with_cache(self, x, offset, cache):
        """Cache de solo-V: cache = tensor V acumulado o None."""
        B, S_new, _ = x.shape
        qp_new = self._qp(x)
        v_new = self.v_proj(x).view(B, S_new, self.num_kv_groups, self.head_dim)
        qp_new, v_new = self.rope(qp_new, v_new, offset)

        v_full = v_new if cache is None else torch.cat([cache, v_new], dim=1)
        new_cache = v_full.clone()

        v_exp = repeat_kv(v_full, self.num_heads, self.num_kv_groups)
        q = qp_new.transpose(1, 2)
        v = v_exp.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, v, v, is_causal=(cache is None))
        out = out.transpose(1, 2).contiguous().view(B, S_new, -1)
        return self.o_proj(out), new_cache

    @torch.no_grad()
    def fusionar_wq(self):
        """Devuelve WQ_eff = WQ·WR por cabeza para inferencia rápida."""
        wq = self.q_proj.weight.view(self.num_heads, self.head_dim, -1)
        return torch.einsum("hdo,hde->heo", wq, self.wr)


class _ExpertoKeyless(nn.Module):
    """Un experto de atención keyless (sin K): Q' = Q·WR, scores vs V."""

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.heads
        self.num_kv_groups = getattr(config, "kv_groups", config.heads)
        self.head_dim = config.dim // config.heads
        self.q_proj = nn.Linear(config.dim, self.num_heads * self.head_dim, bias=False)
        self.wr = nn.Parameter(torch.empty(self.num_heads, self.head_dim, self.head_dim))
        self.v_proj = nn.Linear(config.dim, self.num_kv_groups * self.head_dim, bias=False)
        self.rope = RoPE(self.head_dim, rotary_pct=getattr(config, "rotary_pct", 0.25))
        self.attn_dropout = nn.Dropout(config.drop)
        self.q_proj.is_attention = True
        self.v_proj.is_attention = True
        nn.init.trunc_normal_(self.wr, std=0.02)

    def _qp(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        return torch.einsum("bthd,hde->bthe", q, self.wr)

    def forward(self, x):
        B, T, D = x.shape
        qp = self._qp(x)
        v = self.v_proj(x).view(B, T, self.num_kv_groups, self.head_dim)
        qp, v = self.rope(qp, v, 0)
        v = repeat_kv(v, self.num_heads, self.num_kv_groups)
        q = qp.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, v, v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )
        return out.transpose(1, 2).contiguous().view(B, T, D)


class MoAAttention(nn.Module):
    """Mixture of Attention con expertos keyless: E expertos sin K + router
    top-k por token.

    - Router: Linear(dim → E), softmax, top-k renormalizado.
    - Salida: o_proj(Σ_e w_e · experto_e(x)).
    - Pérdida auxiliar (en self.last_aux, reseteada por forward):
      balanceo E·Σ(mean(p)²) + z-loss sobre logits del router.
    - Caches solo-V (uno por experto).
    """

    def __init__(self, config):
        super().__init__()
        self.num_expertos = getattr(config, "moa_expertos", 4)
        self.topk = min(getattr(config, "moa_topk", 2), self.num_expertos)
        self.aux_w = getattr(config, "moa_aux_w", 0.01)
        self.z_w = getattr(config, "moa_z_w", 0.001)

        self.expertos = nn.ModuleList([_ExpertoKeyless(config) for _ in range(self.num_expertos)])
        self.router = nn.Linear(config.dim, self.num_expertos, bias=False)
        self.o_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.o_proj.is_residual_proj = True
        self.last_aux = torch.tensor(0.0)

    def forward(self, x):
        B, T, D = x.shape
        logits_r = self.router(x)                      # [B,T,E]
        probs = torch.softmax(logits_r.float(), dim=-1).to(x.dtype)
        top_w, top_i = torch.topk(probs, k=self.topk, dim=-1)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        # Suma ponderada de expertos (peso 0 si no fue elegido).
        out = torch.zeros_like(x)
        peso = torch.zeros(B, T, self.num_expertos, device=x.device, dtype=x.dtype)
        peso.scatter_(2, top_i, top_w)
        for e, exp in enumerate(self.expertos):
            out = out + peso[:, :, e:e+1] * exp(x)

        if self.training:
            with torch.no_grad():
                media = probs.mean(dim=(0, 1))         # [E]
                balanceo = self.num_expertos * (media * media).sum()
            zloss = torch.logsumexp(logits_r.float(), dim=-1).pow(2).mean().to(x.dtype)
            self.last_aux = self.aux_w * balanceo.to(x.dtype) + self.z_w * zloss
        else:
            self.last_aux = torch.tensor(0.0, device=x.device)
        return self.o_proj(out)

    def forward_with_cache(self, x, offset, cache):
        """cache = lista de E caches solo-V o None. Cada experto keyless
        concatena su V y atiende q'_new contra el V acumulado."""
        B, S_new, D = x.shape
        logits_r = self.router(x)
        probs = torch.softmax(logits_r.float(), dim=-1).to(x.dtype)
        top_w, top_i = torch.topk(probs, k=self.topk, dim=-1)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        caches = cache if cache is not None else [None] * self.num_expertos
        nuevas = []
        salidas = []
        for e, exp in enumerate(self.expertos):
            qp_new = exp._qp(x)
            v_new = exp.v_proj(x).view(B, S_new, exp.num_kv_groups, exp.head_dim)
            qp_new, v_new = exp.rope(qp_new, v_new, offset)
            c = caches[e]
            v_full = v_new if c is None else torch.cat([c, v_new], dim=1)
            nuevas.append(v_full.clone())
            v = repeat_kv(v_full, exp.num_heads, exp.num_kv_groups).transpose(1, 2)
            o = F.scaled_dot_product_attention(
                qp_new.transpose(1, 2), v, v, is_causal=(c is None))
            salidas.append(o.transpose(1, 2).contiguous().view(B, S_new, D))
        apilada = torch.stack(salidas, dim=2)          # [B,S,E,D]
        peso = torch.zeros(B, S_new, self.num_expertos, device=x.device, dtype=x.dtype)
        peso.scatter_(2, top_i, top_w)
        out = (apilada * peso.unsqueeze(-1)).sum(dim=2)
        self.last_aux = torch.tensor(0.0, device=x.device)
        return self.o_proj(out), nuevas
