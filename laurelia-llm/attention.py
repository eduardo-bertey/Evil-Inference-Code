"""Grouped Query Attention + XSA.

GQA standard con RoPE, causal mask, KV cache, y XSA (orthogonal projection removal).
Sin MLA, sin BMA, sin gated.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from rope import RoPE, apply_rope_partial


def repeat_kv(x, num_heads, num_kv_groups):
    if num_kv_groups == num_heads:
        return x
    return x.repeat_interleave(num_heads // num_kv_groups, dim=2)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        msq = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * msq).type_as(x) * self.weight


class Attention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_groups=0, head_dim=None,
                 max_seq_len=2048, rope_base=10000.0, rope_scaling=1.0,
                 causal=True, dropout=0.0, attn_logit_cap=None, bias=False,
                 use_xsa=False, qk_norm=True, rotary_pct=1.0):
        super().__init__()
        if num_kv_groups == 0:
            num_kv_groups = num_heads
        if head_dim is None:
            head_dim = d_model // num_heads

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim
        self.causal = causal
        self.attn_logit_cap = attn_logit_cap
        self.use_xsa = use_xsa
        self.qk_norm = qk_norm
        self.rotary_pct = rotary_pct

        self.q_proj = nn.Linear(d_model, num_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, num_kv_groups * head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, num_kv_groups * head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * head_dim, d_model, bias=bias)

        self.rope = RoPE(head_dim, max_seq_len, rope_base, rope_scaling)
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        if qk_norm:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)

        if causal:
            self.register_buffer("causal_mask", None, persistent=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, RMSNorm):
                nn.init.ones_(m.weight)

    def _apply_rope(self, q, k, offset):
        if self.rotary_pct >= 1.0:
            return self.rope(q, k, offset)
        return apply_rope_partial(
            q, k, offset, self.rotary_pct,
            self.rope.inv_freq, self.rope.cos_cache, self.rope.sin_cache,
            self.head_dim, self.rope.max_seq_len,
        )

    def forward(self, x, offset=0):
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.num_kv_groups, self.head_dim)
        v = self.v_proj(x).view(B, S, self.num_kv_groups, self.head_dim)

        q, k = self._apply_rope(q, k, offset)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        k = repeat_kv(k, self.num_heads, self.num_kv_groups)
        v = repeat_kv(v, self.num_heads, self.num_kv_groups)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        if self.attn_logit_cap is not None:
            scores = torch.tanh(scores / self.attn_logit_cap) * self.attn_logit_cap

        if self.causal and S > 1:
            if self.causal_mask is None or self.causal_mask.shape[0] < S:
                self.causal_mask = torch.triu(
                    torch.full((S, S), float("-inf"), device=x.device, dtype=x.dtype), diagonal=1
                )
            scores = scores + self.causal_mask[:S, :S].unsqueeze(0).unsqueeze(0)

        attn_w = F.softmax(scores, dim=-1)
        attn_w = self.attn_dropout(attn_w)
        attn_out = torch.matmul(attn_w, v)

        if self.use_xsa:
            Vn = F.normalize(v, dim=-1)
            attn_out = attn_out - (attn_out * Vn).sum(dim=-1, keepdim=True) * Vn

        attn_out = attn_out.transpose(1, 2)
        return self.o_proj(attn_out.reshape(B, S, -1))

    def forward_with_cache(self, x, offset, cache):
        B, S_new, _ = x.shape

        q_new = self.q_proj(x).view(B, S_new, self.num_heads, self.head_dim)
        k_new = self.k_proj(x).view(B, S_new, self.num_kv_groups, self.head_dim)
        v_new = self.v_proj(x).view(B, S_new, self.num_kv_groups, self.head_dim)

        q_new, k_new = self._apply_rope(q_new, k_new, offset)

        if cache is not None:
            k_full = torch.cat([cache[0], k_new], dim=1)
            v_full = torch.cat([cache[1], v_new], dim=1)
        else:
            k_full = k_new
            v_full = v_new

        new_cache = (k_full, v_full)
        S_full = k_full.shape[1]

        if self.qk_norm:
            q_new = self.q_norm(q_new)
            k_full = self.k_norm(k_full)

        k_exp = repeat_kv(k_full, self.num_heads, self.num_kv_groups)
        v_exp = repeat_kv(v_full, self.num_heads, self.num_kv_groups)

        q = q_new.transpose(1, 2)
        k = k_exp.transpose(1, 2)
        v = v_exp.transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        if self.attn_logit_cap is not None:
            scores = torch.tanh(scores / self.attn_logit_cap) * self.attn_logit_cap

        if self.causal and S_new > 1:
            offset_mask = S_full - S_new
            mask = torch.triu(
                torch.full((S_new, S_full), float("-inf"), device=scores.device, dtype=scores.dtype),
                diagonal=offset_mask + 1,
            )
            scores = scores + mask.unsqueeze(0).unsqueeze(0)

        attn_w = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_w, v)

        if self.use_xsa:
            Vn = F.normalize(v[:, :, -S_new:, :], dim=-1)
            attn_out = attn_out - (attn_out * Vn).sum(dim=-1, keepdim=True) * Vn

        attn_out = attn_out.transpose(1, 2)
        return self.o_proj(attn_out.reshape(B, S_new, -1)), new_cache
