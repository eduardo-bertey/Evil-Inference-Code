"""Gated Attention — Headwise and Elementwise variants.

Based on "Gated Attention for Large Language Models: Non-linearity, Sparsity,
and Attention-Sink-Free" (NeurIPS 2025 Best Paper).

Implements post-SDPA gating with query-dependent sparse gates:
  - Headwise: one scalar gate per attention head
  - Elementwise: one gate per element in the attention output

Compatible with Attention interface (forward, forward_with_cache,
forward_with_cache_partial) and with MLA.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from rope import RoPE, apply_rope_partial
from cache_kv import KVCache
from attention import repeat_kv, QKVProjection, OutputProjection


class GatedAttention(nn.Module):
    """Grouped Query Attention + post-SDPA gating.

    Supports headwise (1 scalar per head) or elementwise (d_head values per
    head) gating derived from an extra projection of Q.

    The gate is computed as sigmoid(gate_score) and multiplied onto the
    attention output *after* aggregation (post-SDPA).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_groups: int,
        head_dim: int,
        max_seq_len: int = 2048,
        rope_base: float = 10000.0,
        rope_scaling: float = 1.0,
        causal: bool = True,
        dropout: float = 0.0,
        attn_logit_cap: float | None = None,
        bias: bool = False,
        gated_type: str = "headwise",  # "headwise" | "elementwise"
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim
        self.causal = causal
        self.attn_logit_cap = attn_logit_cap
        self.gated_type = gated_type

        self.inv_sqrt_head_dim = 1.0 / math.sqrt(head_dim)

        # Q projection: extra outputs for gate scores
        if gated_type == "headwise":
            # 1 scalar per head per kv_group
            q_out = num_heads * head_dim + num_kv_groups
        elif gated_type == "elementwise":
            # d_head values per head per kv_group
            q_out = num_heads * head_dim + num_kv_groups * head_dim
        else:
            raise ValueError(f"Unknown gated_type: {gated_type}")

        self.q_proj = nn.Linear(d_model, q_out, bias=bias)
        self.k_proj = nn.Linear(d_model, num_kv_groups * head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, num_kv_groups * head_dim, bias=bias)
        self.o_proj = OutputProjection(d_model, num_heads, head_dim, bias)

        self.rope = RoPE(head_dim, max_seq_len, rope_base, rope_scaling)
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def _split_q_gate(self, q_raw: torch.Tensor, B: int, S: int):
        """Split raw Q projection into query states and gate scores.

        Args:
            q_raw: (B, S, q_out)
        Returns:
            q: (B, S, num_heads, head_dim)
            gate_score: (B, S, num_kv_groups, 1) for headwise
                        (B, S, num_kv_groups, head_dim) for elementwise
        """
        if self.gated_type == "headwise":
            num_kv_groups = self.num_kv_groups
            q, gate = torch.split(q_raw, [self.num_heads * self.head_dim, num_kv_groups], dim=-1)
            gate_score = gate.reshape(B, S, num_kv_groups, 1)
        else:  # elementwise
            num_kv_groups = self.num_kv_groups
            q, gate = torch.split(
                q_raw, [self.num_heads * self.head_dim, num_kv_groups * self.head_dim], dim=-1
            )
            gate_score = gate.reshape(B, S, num_kv_groups, self.head_dim)

        q = q.reshape(B, S, self.num_heads, self.head_dim)
        return q, gate_score

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Full attention forward (training, no cache)."""
        B, S, _ = x.shape

        q_raw = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q, gate_score = self._split_q_gate(q_raw, B, S)
        q, k = self.rope(q, k, offset)

        k = repeat_kv(k, self.num_heads, self.num_kv_groups)
        v = repeat_kv(v, self.num_heads, self.num_kv_groups)

        q = q.transpose(1, 2)   # (B, H, S, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.inv_sqrt_head_dim

        if self.attn_logit_cap is not None:
            scores = torch.tanh(scores / self.attn_logit_cap) * self.attn_logit_cap

        if self.causal and S > 1:
            scores = self._apply_causal_mask(scores, S)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)

        # Transpose back: (B, H, S, D) -> (B, S, H, D)
        attn_output = attn_output.transpose(1, 2)

        # Post-SDPA gating: expand gate to match (B, S, H, D)
        gate_expanded = self._expand_gate(gate_score, B, S, self.num_heads, self.head_dim)
        attn_output = attn_output * torch.sigmoid(gate_expanded)

        return self.o_proj(attn_output)

    def forward_with_cache(
        self,
        x: torch.Tensor,
        offset: int,
        cache: KVCache | None,
    ) -> tuple[torch.Tensor, KVCache]:
        """Attention with KV cache for autoregressive generation."""
        B, S_new, _ = x.shape

        q_raw = self.q_proj(x)
        k_new = self.k_proj(x)
        v_new = self.v_proj(x)

        q, gate_score = self._split_q_gate(q_raw, B, S_new)
        q, k_new = self.rope(q, k_new, offset)

        if cache is not None:
            k_full = torch.cat([cache.cached_k, k_new], dim=1)
            v_full = torch.cat([cache.cached_v, v_new], dim=1)
        else:
            k_full = k_new
            v_full = v_new

        new_cache = KVCache(cached_k=k_full.clone(), cached_v=v_full.clone())

        k_expanded = repeat_kv(k_full, self.num_heads, self.num_kv_groups)
        v_expanded = repeat_kv(v_full, self.num_heads, self.num_kv_groups)

        q = q.transpose(1, 2)
        k = k_expanded.transpose(1, 2)
        v = v_expanded.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.inv_sqrt_head_dim

        if self.attn_logit_cap is not None:
            scores = torch.tanh(scores / self.attn_logit_cap) * self.attn_logit_cap

        q_len = q.shape[2]
        kv_len = k.shape[2]
        if self.causal and q_len > 1:
            scores = self._apply_causal_mask_with_offset(scores, q_len, kv_len)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2)

        gate_expanded = self._expand_gate(gate_score, B, S_new, self.num_heads, self.head_dim)
        attn_output = attn_output * torch.sigmoid(gate_expanded)

        return self.o_proj(attn_output), new_cache

    def forward_with_cache_partial(
        self,
        x: torch.Tensor,
        offset: int,
        cache: KVCache | None,
        rotary_pct: float,
    ) -> tuple[torch.Tensor, KVCache]:
        """Attention with KV cache + partial RoPE."""
        B, S_new, _ = x.shape

        q_raw = self.q_proj(x)
        k_new = self.k_proj(x)
        v_new = self.v_proj(x)

        q, gate_score = self._split_q_gate(q_raw, B, S_new)

        q, k_new = apply_rope_partial(
            q, k_new, offset, rotary_pct,
            self.rope.inv_freq, self.rope.cos_cache, self.rope.sin_cache,
            self.head_dim, self.rope.max_seq_len,
        )

        if cache is not None:
            k_full = torch.cat([cache.cached_k, k_new], dim=1)
            v_full = torch.cat([cache.cached_v, v_new], dim=1)
        else:
            k_full = k_new
            v_full = v_new

        new_cache = KVCache(cached_k=k_full.clone(), cached_v=v_full.clone())

        k_expanded = repeat_kv(k_full, self.num_heads, self.num_kv_groups)
        v_expanded = repeat_kv(v_full, self.num_heads, self.num_kv_groups)

        q = q.transpose(1, 2)
        k = k_expanded.transpose(1, 2)
        v = v_expanded.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.inv_sqrt_head_dim

        if self.attn_logit_cap is not None:
            scores = torch.tanh(scores / self.attn_logit_cap) * self.attn_logit_cap

        q_len = q.shape[2]
        kv_len = k.shape[2]
        if self.causal and q_len > 1:
            scores = self._apply_causal_mask_with_offset(scores, q_len, kv_len)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2)

        gate_expanded = self._expand_gate(gate_score, B, S_new, self.num_heads, self.head_dim)
        attn_output = attn_output * torch.sigmoid(gate_expanded)

        return self.o_proj(attn_output), new_cache

    def _expand_gate(
        self,
        gate_score: torch.Tensor,
        B: int,
        S: int,
        num_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        """Expand gate scores to match (B, S, H, D).

        gate_score: (B, S, num_kv_groups, 1) for headwise
                    (B, S, num_kv_groups, head_dim) for elementwise
        Returns: (B, S, num_heads, head_dim)
        """
        if self.gated_type == "headwise":
            # (B, S, num_kv_groups, 1) -> (B, S, num_heads, 1)
            repeats = self.num_heads // self.num_kv_groups
            g = gate_score.repeat_interleave(repeats, dim=2)
            # (B, S, num_heads, 1) -> (B, S, num_heads, head_dim)
            g = g.expand(-1, -1, -1, head_dim)
        else:  # elementwise
            # (B, S, num_kv_groups, head_dim) -> (B, S, num_heads, head_dim)
            repeats = self.num_heads // self.num_kv_groups
            g = gate_score.repeat_interleave(repeats, dim=2)
        return g

    def _apply_causal_mask(self, scores: torch.Tensor, seq_len: int) -> torch.Tensor:
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
            diagonal=1,
        )
        return scores + mask.unsqueeze(0).unsqueeze(0)

    def _apply_causal_mask_with_offset(
        self, scores: torch.Tensor, q_len: int, kv_len: int
    ) -> torch.Tensor:
        offset = kv_len - q_len
        mask = torch.triu(
            torch.full((q_len, kv_len), float("-inf"), device=scores.device),
            diagonal=offset + 1,
        )
        return scores + mask.unsqueeze(0).unsqueeze(0)
