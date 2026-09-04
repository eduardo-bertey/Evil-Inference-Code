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
    - Router por token: w_gate (+ w_noise con ruido solo en train),
      softmax, top-k SIN renormalizar (como MoA), scatter disperso.
    - Dispatch real: cada experto atiende solo sus tokens asignados
      (gather por experto + index_add). Costo SDPA ≈ topk×, no E×.
    - Máscara causal explícita por posiciones absolutas (los tokens
      asignados no son contiguos: is_causal=True sería incorrecto).
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
        self.w_gate = nn.Parameter(torch.zeros(config.dim, self.num_expertos))
        self.w_noise = nn.Parameter(torch.zeros(config.dim, self.num_expertos))
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
        """Gates dispersos [B,Sq,E] + probs/logits para el aux."""
        logits_r = x @ self.w_gate
        if self.ruido and self.training:
            ruido_std = F.softplus(x @ self.w_noise) + 1e-2
            logits_r = logits_r + torch.randn_like(logits_r) * ruido_std
        probs = torch.softmax(logits_r.float(), dim=-1).to(x.dtype)
        top_w, top_i = torch.topk(probs, k=self.topk, dim=-1)
        # MoA NO renormaliza el top-k.
        gates = torch.zeros_like(probs)
        gates.scatter_(2, top_i, top_w)
        return gates, top_i, top_w, probs, logits_r

    def forward(self, x):
        B, T, D = x.shape
        k = self.k_proj(x).view(B, T, self.num_kv_groups, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_groups, self.head_dim)
        # RoPE sobre secuencia completa (posiciones 0..T-1 correctas).
        k_full, v_full = self.rope(k, v, 0)
        pos_q = torch.arange(T, device=x.device)
        return self._nucleo(x, pos_q, k_full, v_full)

    def _nucleo(self, x, pos_q, k_full, v_full):
        """Dispatch disperso por experto con Q' de cada experto rotada por
        RoPE completo ANTES del gather (posiciones absolutas correctas en
        training, donde x es la secuencia completa)."""
        B, Sq, D = x.shape
        Tkv = k_full.shape[1]
        gates, top_i, _, probs, logits_r = self._router(x)

        k = repeat_kv(k_full, self.num_heads, self.num_kv_groups).transpose(1, 2)
        v = repeat_kv(v_full, self.num_heads, self.num_kv_groups).transpose(1, 2)
        base = torch.arange(Tkv, device=x.device).unsqueeze(0)

        out = torch.zeros(B, Sq, self.num_heads * self.head_dim,
                          device=x.device, dtype=x.dtype)
        for b in range(B):
            for e, exp in enumerate(self.expertos):
                toca = (top_i[b] == e).any(dim=-1)
                n = int(toca.sum())
                if n == 0:
                    continue
                qe_full = exp.qp_completa(x[b:b+1])              # [1,Sq,H,d]
                qe_full, _ = self.rope(qe_full, qe_full, 0)     # offset 0: x es la secuencia completa
                qe = qe_full[:, toca].transpose(1, 2)            # [1,H,n,d]
                pos = pos_q[toca].unsqueeze(1)
                mask = pos >= base
                oe = F.scaled_dot_product_attention(
                    qe, k[b:b+1], v[b:b+1], attn_mask=mask,
                    dropout_p=self.attn_dropout.p if self.training else 0.0,
                )
                oe = oe.transpose(1, 2).reshape(1, n, -1)
                out[b][toca] += gates[b][toca][:, e:e+1] * oe[0]

        if self.training:
            with torch.no_grad():
                plana = probs.reshape(-1, self.num_expertos)
                suma = plana.sum(0)
                cv = self._cv_cuadrado(F.normalize(suma, p=1, dim=0))
                freqs = gates.reshape(-1, self.num_expertos).gt(0).float().sum(0)
                switch = (F.normalize(suma, p=1, dim=0)
                          * F.normalize(freqs, p=1, dim=0)).sum() * self.num_expertos
            z = torch.logsumexp(logits_r.float(), dim=-1).pow(2).mean().to(x.dtype)
            self.last_aux = self.cv_w * cv.to(x.dtype) + self.switch_w * switch.to(x.dtype) + self.z_w * z
        else:
            self.last_aux = torch.tensor(0.0, device=x.device)
        return self.o_proj(out)

    def forward_with_cache(self, x, offset, cache):
        """cache = (k_full, v_full) COMPARTIDO o None. Los tokens nuevos
        llegan contiguos (offset..offset+Sq-1): RoPE con offset directo."""
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

        pos_q = torch.arange(offset, offset + S_new, device=x.device)
        out = self._nucleo_cache(x, pos_q, k_full, v_full)
        self.last_aux = torch.tensor(0.0, device=x.device)
        return self.o_proj(out), new_cache

    def _nucleo_cache(self, x, pos_q, k_full, v_full):
        """Dispatch disperso donde Q' usa RoPE con offset directo (tokens
        nuevos contiguos). Sin aux (solo inferencia)."""
        B, Sq, D = x.shape
        Tkv = k_full.shape[1]
        gates, top_i, _, _, _ = self._router(x)

        k = repeat_kv(k_full, self.num_heads, self.num_kv_groups).transpose(1, 2)
        v = repeat_kv(v_full, self.num_heads, self.num_kv_groups).transpose(1, 2)
        base = torch.arange(Tkv, device=x.device).unsqueeze(0)

        out = torch.zeros(B, Sq, self.num_heads * self.head_dim,
                          device=x.device, dtype=x.dtype)
        for b in range(B):
            for e, exp in enumerate(self.expertos):
                toca = (top_i[b] == e).any(dim=-1)
                n = int(toca.sum())
                if n == 0:
                    continue
                qe = exp.qp_completa(x[b:b+1])[:, toca]          # [1,n,H,d]
                qe, _ = self.rope(qe, qe, int(pos_q[toca][0].item()))
                qe = qe.transpose(1, 2)
                pos = pos_q[toca].unsqueeze(1)
                mask = pos >= base
                oe = F.scaled_dot_product_attention(
                    qe, k[b:b+1], v[b:b+1], attn_mask=mask)
                oe = oe.transpose(1, 2).reshape(1, n, -1)
                out[b][toca] += gates[b][toca][:, e:e+1] * oe[0]
        return out
