"""Mixture-of-Experts with bmm routing, shared experts, bias trick, z-loss, capacity.

Architecture per token:
  1. Router scores = x @ W_router (no bias)
  2. Bias trick: scores += expert_bias (non-learned, updated via load feedback)
  3. softmax → top-k selection → capacity clamp → combine
  4. Routed output = Σ(weight_e * expert_e(x)) over top-k
  5. Shared experts = always-on experts (capture common knowledge)
  6. Final = routed + shared

Design choices (best-of-both from nano-moe-mla + LLM-D3):
  - bmm-based expert params: (n_exp, d_model, 2*expert_dim) + (n_exp, expert_dim, d_model)
  - Bias trick for load balancing (DeepSeek, loss-free)
  - Router z-loss (ST-MoE) to keep logits bounded
  - Capacity factor with token dropping (D3-style)
  - Shared experts (nano/DeepSeekMoE style)
  - SwiGLU activation in every expert
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _swiglu(x, gate):
    return F.silu(gate) * x


class ExpertSwiGLU(nn.Module):
    """Single dense SwiGLU FFN, used standalone or as shared expert."""

    def __init__(self, d_model, expert_dim, bias=False):
        super().__init__()
        self.w_gate = nn.Linear(d_model, expert_dim, bias=bias)
        self.w_up = nn.Linear(d_model, expert_dim, bias=bias)
        self.w_down = nn.Linear(expert_dim, d_model, bias=bias)

    def forward(self, x):
        return self.w_down(_swiglu(self.w_up(x), self.w_gate(x)))


class MoELayer(nn.Module):
    """Sparse Mixture-of-Experts with capacity, bias trick, z-loss, and shared experts.

    Args:
        d_model: Model dimension
        n_experts: Total routed experts
        top_k: Experts activated per token
        n_shared: Always-on shared experts (default 1)
        expert_dim: Intermediate dim per expert (default: 2*d_model)
        capacity_factor: Max capacity multiplier (default 1.25)
        z_loss_gamma: Router z-loss weight (0 to disable, default 0.001)
        bias_decay: Expert bias adaptation rate (default 1e-3)
        bias: Use bias in Linear layers (default False)
    """

    def __init__(self, d_model, n_experts, top_k, n_shared=1, expert_dim=None,
                 capacity_factor=1.25, z_loss_gamma=0.001, bias_decay=0.1,
                 bias=False, noise_std=0.01, head_dim_qk=None):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.n_shared = n_shared
        self.capacity_factor = capacity_factor
        self.z_loss_gamma = z_loss_gamma
        self.bias_decay = bias_decay
        self.noise_std = noise_std

        if expert_dim is None:
            expert_dim = 2 * d_model

        # Proyección de estado recurrente (c, n) del mLSTM para dar contexto al MoE
        state_in = (head_dim_qk or d_model // 2) * 2
        self.state_proj = nn.Linear(state_in, d_model, bias=False)

        # Router
        self.router = nn.Linear(d_model, n_experts, bias=False)

        # Expert biases for load balancing (non-learned, updated via feedback)
        self.register_buffer("expert_bias", torch.zeros(n_experts))

        # Routed experts — batched parameters for bmm
        self.c_fc = nn.Parameter(torch.randn(n_experts, d_model, 2 * expert_dim) * 0.02)
        self.c_proj = nn.Parameter(torch.randn(n_experts, expert_dim, d_model) * 0.02)
        self.b_fc = nn.Parameter(torch.zeros(n_experts, 2 * expert_dim)) if bias else None
        self.b_proj = nn.Parameter(torch.zeros(n_experts, d_model)) if bias else None

        # Shared experts
        self.shared = nn.ModuleList([ExpertSwiGLU(d_model, expert_dim, bias) for _ in range(n_shared)])

        # Routing stats (tracked during forward)
        self.register_buffer("last_counts", torch.zeros(n_experts, dtype=torch.long))
        self.last_total = 0

    def _batched_expert_forward(self, x, expert_idx):
        """Run selected tokens through their chosen expert via bmm.

        Args:
            x: (N, d_model) — selected tokens
            expert_idx: int — expert index (all tokens go to same expert)
        Returns:
            (N, d_model) — expert outputs
        """
        if x.shape[0] == 0:
            return x

        # Scalar index avoids materializing (N, d_model, 2*edim)
        w_fc = self.c_fc[expert_idx]  # (d_model, 2*edim)
        w_proj = self.c_proj[expert_idx]  # (edim, d_model)

        h = x @ w_fc  # (N, 2*edim)

        if self.b_fc is not None:
            h = h + self.b_fc[expert_idx]

        gate, up = h.chunk(2, dim=-1)
        h = _swiglu(up, gate)  # (N, edim)

        out = h @ w_proj  # (N, d_model)

        if self.b_proj is not None:
            out = out + self.b_proj[expert_idx]

        return out

    def _update_expert_bias(self, counts, n_tokens):
        """Update load-balancing biases via feedback (no loss).
        Normalised delta: expert_bias decays toward the target proportion,
        scaled by bias_decay per token to decouple from batch size.
        """
        target = n_tokens / self.n_experts
        load = counts.float()
        delta = self.bias_decay * (target - load) / max(n_tokens, 1)
        self.expert_bias.add_(delta.to(self.expert_bias.dtype))

    def _router_z_loss(self, logits):
        """z-loss: penalize large router logits to stabilize training."""
        if self.z_loss_gamma <= 0:
            return 0.0
        logsumexp = torch.logsumexp(logits, dim=-1)
        z_loss = self.z_loss_gamma * (logsumexp ** 2).mean()
        return z_loss

    def forward(self, x, state=None):
        """Forward MoE.

        Args:
            x: (B, T, d_model)
            state: (c, n, m) from mLSTM, optional context
        Returns:
            output: (B, T, d_model)
            aux_loss: Router z-loss (scalar, 0 if disabled)
        """
        orig_ndim = x.ndim
        if orig_ndim == 2:
            x = x.unsqueeze(1)
        B, T, C = x.shape
        N = B * T
        xf = x.reshape(N, C)

        # State context: inject mLSTM recurrent state into tokens
        if state is not None:
            c, n = state[0], state[1]  # (B, NH, qk, hv), (B, NH, qk, 1)
            c_pool = c.mean(dim=(1, 3))  # (B, qk)
            n_pool = n.mean(dim=(1, 3))  # (B, qk)
            ctx = torch.cat([c_pool, n_pool], dim=-1)  # (B, 2*qk) = (B, d_model)
            ctx = self.state_proj(ctx).unsqueeze(1)  # (B, 1, d_model)
            xf = xf + ctx.repeat(1, T, 1).reshape(N, C)

        # 1) Router scores
        scores = self.router(xf)  # (N, n_experts)

        # Noisy top-k: add gaussian noise during training for exploration
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(scores) * self.noise_std
            scores = scores + noise

        # 2) Bias trick: add non-learned bias
        biased_scores = scores + self.expert_bias.unsqueeze(0)

        # 3) Softmax + top-k
        probs = F.softmax(biased_scores, dim=-1)
        topk_w, topk_i = probs.topk(self.top_k, dim=-1)  # (N, top_k) each
        topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-9)

        # 4) Capacity
        cap = max(1, int(math.ceil(self.top_k * N / self.n_experts * self.capacity_factor)))

        # 5) Route tokens to experts with capacity
        out = torch.zeros_like(xf)
        n_dropped = 0
        for e in range(self.n_experts):
            # Which tokens selected expert e?
            sel_mask = (topk_i == e).any(dim=-1)  # (N,)
            tok_idx = sel_mask.nonzero(as_tuple=True)[0]
            if tok_idx.numel() == 0:
                continue

            # Enforce capacity
            if tok_idx.numel() > cap:
                # Keep only first cap tokens (router chose them with highest prob)
                order = probs[tok_idx, e].argsort(descending=True)
                tok_idx = tok_idx[order[:cap]]
                n_dropped += tok_idx.numel() - cap

            # Get routing weights for these tokens
            w = topk_w[tok_idx]  # (n_sel, top_k)
            sel = (topk_i[tok_idx] == e)  # (n_sel, top_k) bool
            w_e = (w * sel).sum(dim=-1)  # (n_sel,) weight for expert e

            # Expert forward via bmm
            expert_out = self._batched_expert_forward(xf[tok_idx], e)
            out[tok_idx] += w_e.unsqueeze(-1) * expert_out

        # 6) Shared experts
        for sh in self.shared:
            out += sh(xf)

        # 7) Update biases for next step
        with torch.no_grad():
            counts = torch.bincount(topk_i.flatten(), minlength=self.n_experts)
            self.last_counts = counts.clone()
            self.last_total = N
            self._update_expert_bias(counts, N)

        # 8) z-loss
        aux_loss = self._router_z_loss(scores)

        out = out.reshape(B, T, C)
        if orig_ndim == 2:
            out = out.squeeze(1)
        return out, aux_loss

    def balance_str(self):
        """Return string like 'exp%: 25/25/25/25' for the last forward."""
        total = self.last_total or 1
        pcts = [f"{c*100//total}%" for c in self.last_counts]
        return f"exp: {'/'.join(pcts)}"


class DenseFFN(nn.Module):
    """Dense SwiGLU FFN, used as fallback when MoE is disabled."""

    def __init__(self, d_model, intermediate_dim, dropout=0.0, bias=False):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_dim, bias=bias)
        self.up_proj = nn.Linear(d_model, intermediate_dim, bias=bias)
        self.down_proj = nn.Linear(intermediate_dim, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        return self.down_proj(self.dropout(_swiglu(self.up_proj(x), self.gate_proj(x))))
