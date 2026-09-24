"""Rotary Position Embeddings (RoPE) — parcial.

Soporta RoPE parcial (rotary_pct) para rotar solo una fracción de head_dim.
Compatible con inferencia KV cache via offset.
"""

import math
import torch
import torch.nn as nn


class RoPE(nn.Module):
    """RoPE con soporte parcial.

    Args:
        head_dim: dimensión de cada head (debe ser par)
        max_seq_len: longitud máxima precomputada
        base: frecuencia base
        rotary_pct: fracción de dims a rotar (0.0-1.0). 1.0 = full RoPE.
    """

    def __init__(self, head_dim, max_seq_len=2048, base=10000.0, rotary_pct=1.0):
        super().__init__()
        assert head_dim % 2 == 0
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.rotary_pct = rotary_pct

        if rotary_pct >= 1.0:
            self.rotary_dim = head_dim
        else:
            self.rotary_dim = int(head_dim * rotary_pct)
            self.rotary_dim = self.rotary_dim - (self.rotary_dim % 2)
        self.rotary_half = self.rotary_dim // 2

        inv_freq = 1.0 / (base ** (torch.arange(0, self.rotary_half).float() * 2.0 / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, q, k, offset=0):
        """Aplica RoPE parcial a Q y K.

        Args:
            q: (batch, seq_len, num_heads, head_dim)
            k: (batch, seq_len, num_kv_groups, head_dim)
            offset: posición inicial
        Returns:
            (q_rotated, k_rotated)
        """
        if self.rotary_dim == 0:
            return q, k

        seq_len = q.shape[1]
        end = offset + seq_len

        if end > self.max_seq_len:
            self._build_cache(end)
            self.max_seq_len = end

        cos = self.cos_cached[offset:end].unsqueeze(0).unsqueeze(2)
        sin = self.sin_cached[offset:end].unsqueeze(0).unsqueeze(2)

        q_out = self._rotate_partial(q, cos, sin)
        k_out = self._rotate_partial(k, cos, sin)
        return q_out, k_out

    def _rotate_partial(self, x, cos, sin):
        """Rotea solo las primeras rotary_dim dimensiones, el resto pasa sin cambio."""
        rd = self.rotary_dim
        hh = self.rotary_half

        q_rot = x[..., :rd]
        q_pass = x[..., rd:]

        qr_first = q_rot[..., :hh]
        qr_second = q_rot[..., hh:rd]

        q_rotated = torch.cat([
            qr_first * cos - qr_second * sin,
            qr_first * sin + qr_second * cos,
        ], dim=-1)

        return torch.cat([q_rotated, q_pass], dim=-1)

    @staticmethod
    def _apply_rotation(x, cos, sin):
        """RoPE full (100%)."""
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
