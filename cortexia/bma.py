"""Bilinearly Modulated Attention (BMA).

Based on "Bilinearly Modulated Attention" (ICLR 2026).
Applies query-conditioned value gating BEFORE attention aggregation:
  g = sigmoid(Q @ W_g)   per-head bilinear gate
  v = g * V               modulate values
  output = attn_weights @ v

Standalone module that can be combined with MLA, Gated Attention, or standard GQA.
"""

import torch
import torch.nn as nn


class BMAFilter(nn.Module):
    """Per-head bilinear gate on V before aggregation.

    Usage:
        bma = BMAFilter(num_heads, head_dim)
        v = bma(q, v)  # q: (B, H, S, D), v: (B, H, S, D)
    """

    def __init__(self, num_heads: int, head_dim: int, bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.W_g = nn.Parameter(torch.empty(num_heads, head_dim, head_dim))
        nn.init.normal_(self.W_g, std=0.02)

    def forward(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Modulate V with bilinear gate from Q.

        Args:
            q: (B, H, S, D)
            v: (B, H, S, D)
        Returns:
            v_modulated: (B, H, S, D)
        """
        # g = sigmoid(Q @ W_g) per head
        g = torch.sigmoid(torch.einsum("bhtd,hde->bhte", q, self.W_g))
        return g * v
