"""Gated Attention — Headwise and Elementwise variants.

Based on "Gated Attention for Large Language Models: Non-linearity, Sparsity,
and Attention-Sink-Free" (NeurIPS 2025 Best Paper).

Implements post-SDPA gating with query-dependent sparse gates:
  - Headwise: one scalar gate per attention head
  - Elementwise: one gate per element in the attention output

Also includes MLAGatedAttention: MLA compression + post-SDPA gating.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from rope import RoPE, apply_rope_partial
from cache_kv import KVCache
from attention import repeat_kv, QKVProjection, OutputProjection
from mla_attention import QKVProjectionMLA, OutputProjectionMLA, RMSNorm
from bma import BMAFilter


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
        use_bma: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim
        self.causal = causal
        self.attn_logit_cap = attn_logit_cap
        self.gated_type = gated_type
        self.use_bma = use_bma

        # BMA filter
        if use_bma:
            from bma import BMAFilter
            self.bma_filter = BMAFilter(num_heads, head_dim)

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

        # BMA: pre-aggregation gating
        if self.use_bma:
            v = self.bma_filter(q, v)

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

        # BMA: pre-aggregation gating
        if self.use_bma:
            v = self.bma_filter(q, v)

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

        # BMA: pre-aggregation gating
        if self.use_bma:
            v = self.bma_filter(q, v)

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


class MLAGatedAttention(nn.Module):
    """MLA + Gated Attention: latent compression + post-SDPA gating.

    Combines MLA's KV compression for cache efficiency with gated
    attention's post-SDPA gating for better attention patterns.

    Cache stores: (C_KV, K_rotate_raw) — same as MLA.
    Gate scores are derived from Q_state (content part of Q).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_groups: int = 0,
        head_dim: int | None = None,
        max_seq_len: int = 2048,
        rope_base: float = 10000.0,
        rope_scaling: float = 1.0,
        causal: bool = True,
        dropout: float = 0.0,
        attn_logit_cap: float | None = None,
        bias: bool = False,
        d_c: int | None = None,
        d_c1: int | None = None,
        d_rotate: int | None = None,
        block_size: int = 128,
        use_xsa: bool = False,
        qk_norm: bool = True,
        gated_type: str = "headwise",  # "headwise" | "elementwise"
        use_bma: bool = False,
    ):
        super().__init__()
        if num_kv_groups == 0:
            num_kv_groups = num_heads
        if head_dim is None:
            head_dim = d_model // num_heads
        if d_c is None:
            d_c = max(32, d_model // 6)
        if d_c1 is None:
            d_c1 = max(32, d_model // 6)
        if d_rotate is None:
            d_rotate = max(16, d_model // 12)

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim
        self.d_rotate = d_rotate
        self.d_c = d_c
        self.d_c1 = d_c1
        self.causal = causal
        self.attn_logit_cap = attn_logit_cap
        self.block_size = block_size
        self.use_xsa = use_xsa
        self.qk_norm = qk_norm
        self.gated_type = gated_type
        self.use_bma = use_bma

        # BMA filter (pre-aggregation gating)
        if use_bma:
            self.bma_filter = BMAFilter(num_heads, head_dim)

        # QK-Norm
        if qk_norm:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)

        # MLA QKV projection (shared latent)
        self.qkv = QKVProjectionMLA(d_model, num_heads, num_kv_groups, head_dim,
                                     d_c, d_c1, d_rotate, bias)
        self.o_proj = OutputProjectionMLA(d_model, num_heads, head_dim, bias)

        # Gate projection from Q_state content
        # Gate scores come from the content part of Q, not the rotary part
        if gated_type == "headwise":
            # 1 scalar per head
            self.gate_proj = nn.Linear(d_c1, num_heads, bias=bias)
        elif gated_type == "elementwise":
            # d_head values per head
            self.gate_proj = nn.Linear(d_c1, num_heads * head_dim, bias=bias)
        else:
            raise ValueError(f"Unknown gated_type: {gated_type}")

        # RoPE on rotary dimension
        self.rope = RoPE(head_dim=d_rotate, max_seq_len=max_seq_len,
                         base=rope_base, scaling_factor=rope_scaling)
        self.rope.head_dim = d_rotate
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        if causal:
            mask = torch.triu(torch.full((max_seq_len, max_seq_len), float("-inf")), diagonal=1)
            self.register_buffer("causal_mask", mask, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, RMSNorm):
            nn.init.ones_(m.weight)

    def _decoupled_scores(self, Q_state, Q_rot, K_state, K_rot, seq_len, kv_len=None, causal=None):
        """Decoupled content + RoPE scoring (same as MLA)."""
        nh, nkv, hd, dr = self.num_heads, self.num_kv_groups, self.head_dim, self.d_rotate
        scale_c = 1.0 / math.sqrt(self.d_c) if self.qk_norm else 1.0 / math.sqrt(hd)
        scale_r = 1.0 / math.sqrt(dr)

        k_c = repeat_kv(K_state, nh, nkv).transpose(1, 2)
        q_c = Q_state.transpose(1, 2)
        s_c = torch.matmul(q_c, k_c.transpose(-2, -1)) * scale_c

        q_r = Q_rot.transpose(1, 2)
        k_r = K_rot.transpose(1, 2).expand(-1, nh, -1, -1)
        s_r = torch.matmul(q_r, k_r.transpose(-2, -1)) * scale_r

        scores = s_c + s_r
        if self.attn_logit_cap is not None:
            scores = torch.tanh(scores / self.attn_logit_cap) * self.attn_logit_cap

        is_causal = self.causal if causal is None else causal
        if is_causal:
            if kv_len is None:
                kv_len = K_state.shape[1]
            if seq_len > 1:
                if seq_len == kv_len:
                    if seq_len <= self.causal_mask.shape[0]:
                        scores = scores + self.causal_mask[:seq_len, :seq_len]
                    else:
                        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"),
                                                      device=scores.device), diagonal=1)
                        scores = scores + mask
                else:
                    mask = torch.triu(
                        torch.full((seq_len, kv_len), float("-inf"), device=scores.device),
                        diagonal=kv_len - seq_len + 1
                    )
                    scores = scores + mask
        return scores

    def _compute_gate(self, C_Q: torch.Tensor, B: int, S: int):
        """Compute gate scores from MLA's C_Q latent.

        C_Q: (B, S, d_c1) — the content latent
        Returns: (B, S, num_heads, 1) for headwise or (B, S, num_heads, head_dim) for elementwise
        """
        gate_raw = self.gate_proj(C_Q)  # (B, S, num_heads) or (B, S, num_heads * head_dim)
        if self.gated_type == "headwise":
            return gate_raw.reshape(B, S, self.num_heads, 1)
        else:
            return gate_raw.reshape(B, S, self.num_heads, self.head_dim)

    def _expand_gate(self, gate_score: torch.Tensor):
        """Expand gate to (B, S, num_heads, head_dim)."""
        if self.gated_type == "headwise":
            return gate_score.expand(-1, -1, -1, self.head_dim)
        return gate_score  # already (B, S, H, D) for elementwise

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Full forward (training, no cache)."""
        B, T, _ = x.shape

        # MLA: compress and decompress
        Q_state, Q_rotate, K, V, K_rotate = self.qkv(x)
        Q_rotate, K_rotate = self.rope(Q_rotate, K_rotate, offset)

        # Gate from C_Q (before W_up_q expansion, we need C_Q)
        # Re-derive C_Q from x: C_Q = norm(W_down(x)[:d_c1])
        down = self.qkv.W_down(x)
        C_Q, _, _ = down.split([self.qkv.d_c1, self.qkv.d_c, self.qkv.d_rotate], dim=-1)
        C_Q = self.qkv.norm_cq(C_Q)
        gate_score = self._compute_gate(C_Q, B, T)

        if self.qk_norm:
            Q_state = self.q_norm(Q_state)
            K = self.k_norm(K)

        scores = self._decoupled_scores(Q_state, Q_rotate, K, K_rotate, T)
        attn_w = F.softmax(scores, dim=-1)
        attn_w = self.attn_dropout(attn_w)
        v = repeat_kv(V, self.num_heads, self.num_kv_groups).transpose(1, 2)

        # BMA: pre-aggregation gating (modulate V before SDPA)
        if self.use_bma:
            v = self.bma_filter(Q_state.transpose(1, 2), v)

        attn_out = torch.matmul(attn_w, v)  # (B, nh, T, hd)

        # XSA (orthogonal projection removal)
        if self.use_xsa:
            Vn = F.normalize(v, dim=-1)
            attn_out = attn_out - (attn_out * Vn).sum(dim=-1, keepdim=True) * Vn

        # Post-SDPA gating
        attn_out = attn_out.transpose(1, 2)  # (B, T, nh, hd)
        gate_expanded = self._expand_gate(gate_score)
        attn_out = attn_out * torch.sigmoid(gate_expanded)

        return self.o_proj(attn_out)

    def forward_with_cache(self, x: torch.Tensor, offset: int, cache) -> tuple[torch.Tensor, tuple]:
        """Forward with MLA latent cache + gated attention."""
        B, S_new, _ = x.shape

        # MLA compress
        down = self.qkv.W_down(x)
        C_Q_new, C_KV_new, K_rot_raw = down.split([self.qkv.d_c1, self.qkv.d_c, self.qkv.d_rotate], dim=-1)
        C_Q_new = self.qkv.norm_cq(C_Q_new)
        C_KV_new = self.qkv.norm_ckv(C_KV_new)

        # Gate from C_Q
        gate_score = self._compute_gate(C_Q_new, B, S_new)

        # Q from C_Q
        q_up = self.qkv.W_up_q(C_Q_new)
        Q_state, Q_rot_raw = q_up.split([self.num_heads * self.head_dim, self.num_heads * self.d_rotate], dim=-1)
        Q_state = Q_state.reshape(B, S_new, self.num_heads, self.head_dim)
        Q_rot_raw = Q_rot_raw.reshape(B, S_new, self.num_heads, self.d_rotate)
        Q_rot = self.rope.apply_to_single(Q_rot_raw, offset=offset)

        # Cache
        if cache is not None:
            C_KV_full = torch.cat([cache[0], C_KV_new], dim=1)
            K_rot_full = torch.cat([cache[1], K_rot_raw], dim=1)
        else:
            C_KV_full = C_KV_new
            K_rot_full = K_rot_raw
        S_full = C_KV_full.shape[1]

        # KV decompress
        kv_up = self.qkv.W_up_kv(C_KV_full)
        K_state, V_state = kv_up.chunk(2, dim=-1)
        K_state = K_state.reshape(B, S_full, self.num_kv_groups, self.head_dim)
        V_state = V_state.reshape(B, S_full, self.num_kv_groups, self.head_dim)

        K_rot = self.rope.apply_to_single(K_rot_full.unsqueeze(2), offset=0)

        # Attention
        if self.qk_norm:
            Q_state = self.q_norm(Q_state)
            K_state = self.k_norm(K_state)
        scores = self._decoupled_scores(Q_state, Q_rot, K_state, K_rot, S_new, S_full, S_new > 1)
        attn_w = F.softmax(scores, dim=-1)
        attn_w = self.attn_dropout(attn_w)
        v = repeat_kv(V_state, self.num_heads, self.num_kv_groups).transpose(1, 2)

        # BMA: pre-aggregation gating
        if self.use_bma:
            v = self.bma_filter(Q_state.transpose(1, 2), v)

        attn_out = torch.matmul(attn_w, v)
        if self.use_xsa:
            v_new = v[:, :, -S_new:, :]
            Vn = F.normalize(v_new, dim=-1)
            attn_out = attn_out - (attn_out * Vn).sum(dim=-1, keepdim=True) * Vn

        # Post-SDPA gating
        attn_out = attn_out.transpose(1, 2)
        gate_expanded = self._expand_gate(gate_score)
        attn_out = attn_out * torch.sigmoid(gate_expanded)

        return self.o_proj(attn_out), (C_KV_full, K_rot_full)

    def forward_with_cache_partial(self, x: torch.Tensor, offset: int, cache, rotary_pct: float):
        """Forward with MLA latent cache + partial RoPE + gated attention."""
        B, S_new, _ = x.shape

        down = self.qkv.W_down(x)
        C_Q_new, C_KV_new, K_rot_raw = down.split([self.qkv.d_c1, self.qkv.d_c, self.qkv.d_rotate], dim=-1)
        C_Q_new = self.qkv.norm_cq(C_Q_new)
        C_KV_new = self.qkv.norm_ckv(C_KV_new)

        gate_score = self._compute_gate(C_Q_new, B, S_new)

        q_up = self.qkv.W_up_q(C_Q_new)
        Q_state, Q_rot_raw = q_up.split([self.num_heads * self.head_dim, self.num_heads * self.d_rotate], dim=-1)
        Q_state = Q_state.reshape(B, S_new, self.num_heads, self.head_dim)
        Q_rot_raw = Q_rot_raw.reshape(B, S_new, self.num_heads, self.d_rotate)

        Q_rot, _ = apply_rope_partial(Q_rot_raw, Q_rot_raw, offset, rotary_pct,
            self.rope.inv_freq, self.rope.cos_cache, self.rope.sin_cache,
            self.rope.head_dim, self.rope.max_seq_len)

        if cache is not None:
            C_KV_full = torch.cat([cache[0], C_KV_new], dim=1)
            K_rot_full = torch.cat([cache[1], K_rot_raw], dim=1)
        else:
            C_KV_full = C_KV_new
            K_rot_full = K_rot_raw
        S_full = C_KV_full.shape[1]

        kv_up = self.qkv.W_up_kv(C_KV_full)
        K_state, V_state = kv_up.chunk(2, dim=-1)
        K_state = K_state.reshape(B, S_full, self.num_kv_groups, self.head_dim)
        V_state = V_state.reshape(B, S_full, self.num_kv_groups, self.head_dim)

        _, K_rot = apply_rope_partial(K_rot_full.unsqueeze(2), K_rot_full.unsqueeze(2), 0, rotary_pct,
            self.rope.inv_freq, self.rope.cos_cache, self.rope.sin_cache,
            self.rope.head_dim, self.rope.max_seq_len)

        if self.qk_norm:
            Q_state = self.q_norm(Q_state)
            K_state = self.k_norm(K_state)
        scores = self._decoupled_scores(Q_state, Q_rot, K_state, K_rot, S_new, S_full, S_new > 1)
        attn_w = F.softmax(scores, dim=-1)
        attn_w = self.attn_dropout(attn_w)
        v = repeat_kv(V_state, self.num_heads, self.num_kv_groups).transpose(1, 2)

        # BMA: pre-aggregation gating
        if self.use_bma:
            v = self.bma_filter(Q_state.transpose(1, 2), v)

        attn_out = torch.matmul(attn_w, v)

        if self.use_xsa:
            v_new = v[:, :, -S_new:, :]
            Vn = F.normalize(v_new, dim=-1)
            attn_out = attn_out - (attn_out * Vn).sum(dim=-1, keepdim=True) * Vn

        attn_out = attn_out.transpose(1, 2)
        gate_expanded = self._expand_gate(gate_score)
        attn_out = attn_out * torch.sigmoid(gate_expanded)

        return self.o_proj(attn_out), (C_KV_full, K_rot_full)
