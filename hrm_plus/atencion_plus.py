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

import os as _os
import sys as _sys
# Asegura import del mismo directorio (rope) en Colab/cualquier cwd.
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
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


class _ExpertoQ(nn.Module):
    """Medio experto keyless: aporta Q' = (X·WQ)·WR. K/V son compartidos
    (los calcula MoAAttention una sola vez)."""

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.heads
        self.head_dim = config.dim // config.heads
        self.q_proj = nn.Linear(config.dim, self.num_heads * self.head_dim, bias=False)
        self.wr = nn.Parameter(torch.empty(self.num_heads, self.head_dim, self.head_dim))
        self.q_proj.is_attention = True
        nn.init.trunc_normal_(self.wr, std=0.02)

    def qp_completa(self, x):
        """Q' sin RoPE para toda la secuencia [B,T,H,d]."""
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        return torch.einsum("bthd,hde->bthe", q, self.wr)


class MoAAttention(nn.Module):
    """Mixture of Attention fiel a MoA (yikangshen/MoA), con expertos
    keyless y torch puro (sin triton/flash/CUDA avanzado).

    - K/V COMPARTIDOS únicos (GQA); la mezcla está solo en Q:
      cada experto aporta Q' = (X·WQ_e)·WR_e.
    - Router POR CABEZA (como MoA: map →{q por head}→ reduce):
      w_gate (+ w_noise con ruido solo en train) de dim → H*E,
      softmax sobre E, top-1 por cabeza SIN renormalizar.
    - UN SOLO SDPA: Q_mezclada[t,h] = Q del experto elegido para esa
      cabeza; la ponderación por gates va DESPUÉS (lineal, exacto).
      Costo SDPA ≈ 1×.
    - Pérdidas MoA exactas: cv (CV²), switch (probs·freqs×E con
      frecuencias reales), z (logsumexp²). En self.last_aux.
    - Cache de decoding: UN solo (K, V) compartido para todos.
    """

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.heads
        self.num_kv_groups = getattr(config, "kv_groups", config.heads)
        self.head_dim = config.dim // config.heads
        self.num_expertos = getattr(config, "moa_expertos", 4)
        self.topk = min(max(1, getattr(config, "moa_topk", 2)), self.num_expertos)
        self.cv_w = getattr(config, "moa_aux_w", 0.01)
        self.switch_w = getattr(config, "moa_aux_w", 0.01)
        self.z_w = getattr(config, "moa_z_w", 0.001)
        self.ruido = getattr(config, "moa_ruido", True)

        self.expertos = nn.ModuleList([_ExpertoQ(config) for _ in range(self.num_expertos)])
        self.k_proj = nn.Linear(config.dim, self.num_kv_groups * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, self.num_kv_groups * self.head_dim, bias=False)
        # Router por cabeza: H*E logits por token (un top-1 por cabeza).
        self.w_gate = nn.Parameter(torch.zeros(config.dim, self.num_heads * self.num_expertos))
        self.w_noise = nn.Parameter(torch.zeros(config.dim, self.num_heads * self.num_expertos))
        self.rope = RoPE(self.head_dim, rotary_pct=getattr(config, "rotary_pct", 0.25))
        self.o_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.attn_dropout = nn.Dropout(config.drop)
        self.k_proj.is_attention = True
        self.v_proj.is_attention = True
        self.o_proj.is_residual_proj = True
        self.last_aux = torch.tensor(0.0)

    @staticmethod
    def _cv_cuadrado(x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor(0.0, device=x.device)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _router(self, x):
        """Top-1 experto por (token, cabeza). Devuelve:
        elige [B,Sq,H] (índice de experto), gates [B,Sq,H] (peso),
        probs [B,Sq,H,E] y logits para el aux."""
        B, Sq, _ = x.shape
        logits_r = (x @ self.w_gate).view(B, Sq, self.num_heads, self.num_expertos)
        if self.ruido and self.training:
            ruido_std = F.softplus((x @ self.w_noise).view(B, Sq, self.num_heads, self.num_expertos)) + 1e-2
            logits_r = logits_r + torch.randn_like(logits_r) * ruido_std
        probs = torch.softmax(logits_r.float(), dim=-1).to(x.dtype)
        # Top-1 por cabeza, SIN renormalizar (MoA).
        gate_w, elige = probs.max(dim=-1)   # [B,Sq,H]
        return elige, gate_w, probs, logits_r

    def _mezclar_q(self, x, elige):
        """Q_mezclada[b,t,h] = Q' del experto elegido. [B,Sq,H,d]."""
        apilada = torch.stack([e.qp_completa(x) for e in self.expertos], dim=0)
        # apilada: [E,B,Sq,H,d] -> [B,Sq,H,E,d]
        apilada = apilada.permute(1, 2, 3, 0, 4)
        idx = elige.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1, apilada.shape[-1])
        return apilada.gather(3, idx).squeeze(3)

    def _aux(self, probs, logits_r, elige, x):
        plana = probs.reshape(-1, self.num_expertos)     # [N,E]
        suma = plana.sum(0)
        with torch.no_grad():
            cv = self._cv_cuadrado(F.normalize(suma, p=1, dim=0))
            freqs = elige.reshape(-1).bincount(minlength=self.num_expertos)
            switch = (F.normalize(suma, p=1, dim=0)
                      * F.normalize(freqs, p=1, dim=0)).sum() * self.num_expertos
        z = torch.logsumexp(logits_r.float(), dim=-1).pow(2).mean().to(x.dtype)
        self.last_aux = self.cv_w * cv.to(x.dtype) + self.switch_w * switch.to(x.dtype) + self.z_w * z

    def forward(self, x):
        B, T, D = x.shape
        k = self.k_proj(x).view(B, T, self.num_kv_groups, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_groups, self.head_dim)
        k_full, v_full = self.rope(k, v, 0)

        elige, gate_w, probs, logits_r = self._router(x)
        qp = self._mezclar_q(x, elige)                   # [B,T,H,d]
        qp, _ = self.rope(qp, qp, 0)
        q = qp.transpose(1, 2)                           # [B,H,T,d]
        k = repeat_kv(k_full, self.num_heads, self.num_kv_groups).transpose(1, 2)
        v = repeat_kv(v_full, self.num_heads, self.num_kv_groups).transpose(1, 2)
        # UN SOLO SDPA (como MoA: map -> attention -> reduce).
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2)                        # [B,T,H,d]
        out = out * gate_w.unsqueeze(-1)                 # reduce: pondera por gates
        out = out.reshape(B, T, -1)

        if self.training:
            self._aux(probs, logits_r, elige, x)
        else:
            self.last_aux = torch.tensor(0.0, device=x.device)
        return self.o_proj(out)

    def forward_with_cache(self, x, offset, cache):
        """cache = (k_full, v_full) COMPARTIDO o None. UN SOLO SDPA también
        en decoding: Q mezclada por cabeza + máscara causal explícita."""
        B, S_new, D = x.shape
        k_new = self.k_proj(x).view(B, S_new, self.num_kv_groups, self.head_dim)
        v_new = self.v_proj(x).view(B, S_new, self.num_kv_groups, self.head_dim)
        k_new, v_new = self.rope(k_new, v_new, offset)
        if cache is None:
            k_full, v_full = k_new, v_new
        else:
            k_full = torch.cat([cache[0], k_new], dim=1)
            v_full = torch.cat([cache[1], v_new], dim=1)
        new_cache = (k_full.clone(), v_full.clone())
        Tkv = k_full.shape[1]

        elige, gate_w, _, _ = self._router(x)
        qp = self._mezclar_q(x, elige)
        qp, _ = self.rope(qp, qp, offset)   # nuevos contiguos: offset directo
        q = qp.transpose(1, 2)
        k = repeat_kv(k_full, self.num_heads, self.num_kv_groups).transpose(1, 2)
        v = repeat_kv(v_full, self.num_heads, self.num_kv_groups).transpose(1, 2)
        pos_q = torch.arange(offset, offset + S_new, device=x.device).unsqueeze(1)
        mask = pos_q >= torch.arange(Tkv, device=x.device).unsqueeze(0)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2) * gate_w.unsqueeze(-1)
        out = out.reshape(B, S_new, -1)
        self.last_aux = torch.tensor(0.0, device=x.device)
        return self.o_proj(out), new_cache
