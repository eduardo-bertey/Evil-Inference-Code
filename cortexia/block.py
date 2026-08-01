"""Transformer Block compatible with Rust blocks::trasformer::layer.

Architecture per layer:
  x -> RMSNorm -> Attention(GQA + RoPE) -> +residual
    -> RMSNorm -> FeedForward(SwiGLU) -> +residual -> output

Also includes the full Transformer stack with final RMSNorm.

Grouped shared cache:
  - cache_every=N: cada N capas, una MLA layer (producer) proyecta
    Q_state, Q_rotate, K_state, V, K_rotate y las comparte con el grupo
  - Las capas intermedias son SharedCacheTransformerLayer que leen esas
    proyecciones del producer (sin q_proj ni W_up_kv propios)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import Attention, repeat_kv
from gated_attention import GatedAttention, MLAGatedAttention
from mla_attention import MultiHeadLatentAttentionGQA
from cache_kv import KVCache
from rope import RoPE
from bma import BMAFilter


class RMSNorm(nn.Module):
    """RMSNorm compatible with burn::nn::RmsNorm."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, d_model: int, intermediate_dim: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_dim, bias=bias)
        self.up_proj = nn.Linear(d_model, intermediate_dim, bias=bias)
        self.down_proj = nn.Linear(intermediate_dim, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        h = gate * up
        h = self.dropout(h)
        return self.down_proj(h)


class StandardFFN(nn.Module):
    """Standard FFN: x -> up -> GELU -> dropout -> down"""

    def __init__(self, d_model: int, intermediate_dim: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        self.up_proj = nn.Linear(d_model, intermediate_dim, bias=bias)
        self.down_proj = nn.Linear(intermediate_dim, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.gelu(self.up_proj(x))
        h = self.dropout(h)
        return self.down_proj(h)


def compute_intermediate_dim(
    d_model: int,
    expansion_factor: float = 4.0,
    use_swiglu: bool = True,
    round_to: int = 64,
) -> int:
    """Compute FFN intermediate dimension matching Rust FeedForwardConfig."""
    if use_swiglu:
        raw = int(expansion_factor * d_model * 2.0 / 3.0)
    else:
        raw = int(expansion_factor * d_model)
    return ((raw + round_to - 1) // round_to) * round_to


class TransformerLayer(nn.Module):
    """Single Transformer decoder layer with optional MLA."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_groups: int = 0,
        head_dim: int | None = None,
        ffn_expansion: float = 4.0,
        use_swiglu: bool = True,
        max_seq_len: int = 2048,
        rope_base: float = 10000.0,
        rope_scaling: float = 1.0,
        causal: bool = True,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        attn_logit_cap: float | None = None,
        bias: bool = False,
        norm_eps: float = 1e-5,
        ffn_round_to: int = 64,
        use_sparse_attn: bool = False,
        num_selected_blocks: int = 16,
        use_mla: bool = False,
        mla_d_c: int | None = None,
        mla_d_c1: int | None = None,
        mla_d_rotate: int | None = None,
        mla_block_size: int = 128,
        use_gated_attn: bool = False,
        gated_type: str = "headwise",
        use_xsa: bool = False,
        qk_norm: bool = True,
        use_sandwich_norm: bool = False,
        use_bma: bool = False,
    ):
        super().__init__()

        if num_kv_groups == 0:
            num_kv_groups = num_heads
        if head_dim is None:
            head_dim = d_model // num_heads

        self.attn_norm = RMSNorm(d_model, eps=norm_eps)
        self.use_mla = use_mla
        if use_mla and use_gated_attn:
            self.attention = MLAGatedAttention(
                d_model=d_model, num_heads=num_heads, num_kv_groups=num_kv_groups,
                head_dim=head_dim, max_seq_len=max_seq_len, rope_base=rope_base,
                rope_scaling=rope_scaling, causal=causal, dropout=attn_dropout,
                attn_logit_cap=attn_logit_cap, bias=bias,
                d_c=mla_d_c, d_c1=mla_d_c1, d_rotate=mla_d_rotate,
                block_size=mla_block_size, use_xsa=use_xsa, qk_norm=qk_norm,
                gated_type=gated_type, use_bma=use_bma,
            )
        elif use_mla:
            self.attention = MultiHeadLatentAttentionGQA(
                d_model=d_model, num_heads=num_heads, num_kv_groups=num_kv_groups,
                head_dim=head_dim, max_seq_len=max_seq_len, rope_base=rope_base,
                rope_scaling=rope_scaling, causal=causal, dropout=attn_dropout,
                attn_logit_cap=attn_logit_cap, bias=bias,
                d_c=mla_d_c, d_c1=mla_d_c1, d_rotate=mla_d_rotate,
                block_size=mla_block_size, use_xsa=use_xsa, qk_norm=qk_norm,
                use_bma=use_bma,
            )
        elif use_gated_attn:
            self.attention = GatedAttention(
                d_model=d_model, num_heads=num_heads, num_kv_groups=num_kv_groups,
                head_dim=head_dim, max_seq_len=max_seq_len, rope_base=rope_base,
                rope_scaling=rope_scaling, causal=causal, dropout=attn_dropout,
                attn_logit_cap=attn_logit_cap, bias=bias, gated_type=gated_type,
                use_bma=use_bma,
            )
        else:
            self.attention = Attention(
                d_model=d_model, num_heads=num_heads, num_kv_groups=num_kv_groups,
                head_dim=head_dim, max_seq_len=max_seq_len, rope_base=rope_base,
                rope_scaling=rope_scaling, causal=causal, dropout=attn_dropout,
                attn_logit_cap=attn_logit_cap, bias=bias,
            )
        self.use_sparse_attn = False
        self.ffn_norm = RMSNorm(d_model, eps=norm_eps)
        inter_dim = compute_intermediate_dim(d_model, ffn_expansion, use_swiglu, ffn_round_to)
        if use_swiglu:
            self.ffn = SwiGLUFFN(d_model, inter_dim, ffn_dropout, bias)
        else:
            self.ffn = StandardFFN(d_model, inter_dim, ffn_dropout, bias)
        self.residual_dropout = nn.Dropout(residual_dropout) if residual_dropout > 0.0 else nn.Identity()

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim
        self.causal = causal
        self.attn_logit_cap = attn_logit_cap

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Forward pass with Pre-Norm residual connections."""
        residual = x
        h = self.attn_norm(x)
        h = self.attention(h, offset)
        h = self.residual_dropout(h)
        x = residual + h

        residual = x
        h = self.ffn_norm(x)
        h = self.ffn(h)
        h = self.residual_dropout(h)
        return residual + h

    def forward_produce_cache(self, x: torch.Tensor, offset: int = 0) -> tuple[torch.Tensor, tuple | None]:
        """Forward for MLA producer layers: returns (output, shared_proj).

        Extrae de pre-attention h (el mismo input que recibe MLA qkv) las 5
        proyecciones compartidas para las capas del grupo:
          (Q_state, Q_rotate_raw, K_state, V, K_rotate_raw)
        y luego corre attention.
        """
        residual = x
        h = self.attn_norm(x)

        if self.use_mla and hasattr(self.attention, 'qkv'):
            qkv = self.attention.qkv
            B, T, _ = x.shape
            down = qkv.W_down(h)
            C_Q, C_KV, K_rotate_raw = down.split([qkv.d_c1, qkv.d_c, qkv.d_rotate], dim=-1)
            C_Q = qkv.norm_cq(C_Q)
            C_KV = qkv.norm_ckv(C_KV)

            q_up = qkv.W_up_q(C_Q)
            Q_state, Q_rotate_raw = q_up.split([self.num_heads * self.head_dim, self.num_heads * qkv.d_rotate], dim=-1)
            Q_state = Q_state.reshape(B, T, self.num_heads, self.head_dim)
            Q_rotate_raw = Q_rotate_raw.reshape(B, T, self.num_heads, qkv.d_rotate)

            kv_up = qkv.W_up_kv(C_KV)
            K_state, V = kv_up.chunk(2, dim=-1)
            K_state = K_state.reshape(B, T, self.num_kv_groups, self.head_dim)
            V = V.reshape(B, T, self.num_kv_groups, self.head_dim)

            K_rotate_raw = K_rotate_raw.unsqueeze(2)
            shared_proj = (Q_state, Q_rotate_raw, K_state, V, K_rotate_raw)
        else:
            shared_proj = None

        h_attn = self.attention(h, offset)
        h_attn = self.residual_dropout(h_attn)
        x = residual + h_attn

        residual = x
        h = self.ffn_norm(x)
        h = self.ffn(h)
        h = self.residual_dropout(h)
        return residual + h, shared_proj

    def forward_with_cache(
        self,
        x: torch.Tensor,
        offset: int,
        cache: KVCache | None,
    ) -> tuple[torch.Tensor, KVCache]:
        """Forward with KV cache."""
        residual = x
        h = self.attn_norm(x)
        h, new_cache = self.attention.forward_with_cache(h, offset, cache)
        h = self.residual_dropout(h)
        x = residual + h

        residual = x
        h = self.ffn_norm(x)
        h = self.ffn(h)
        h = self.residual_dropout(h)
        return residual + h, new_cache


class SharedCacheTransformerLayer(nn.Module):
    """Capa que lee Q/K/V ya proyectados de una cache compartida (MLA producer).

    Mismo scoring decoupled que MLA:
      - Q_state + Q_rotate, K_state + V, K_rotate_raw: todos del producer del grupo
      - RoPE solo en Q_rotate y K_rotate_raw
      - q_norm/k_norm/o_proj/gate_proj: propios por capa
      - score = Q_state·K_state/sqrt(d_c) + Q_rotate·K_rotate/sqrt(d_rotate)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_groups: int,
        head_dim: int,
        d_c: int,
        d_rotate: int,
        max_seq_len: int = 2048,
        rope_base: float = 10000.0,
        rope_scaling: float = 1.0,
        causal: bool = True,
        dropout: float = 0.0,
        attn_logit_cap: float | None = None,
        bias: bool = False,
        norm_eps: float = 1e-5,
        ffn_expansion: float = 4.0,
        use_swiglu: bool = True,
        ffn_dropout: float = 0.0,
        ffn_round_to: int = 64,
        use_gated_attn: bool = False,
        gated_type: str = "headwise",
        qk_norm: bool = True,
        use_bma: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim
        self.d_c = d_c
        self.d_rotate = d_rotate
        self.causal = causal
        self.attn_logit_cap = attn_logit_cap
        self.qk_norm = qk_norm
        self.use_bma = use_bma
        self.use_gated_attn = use_gated_attn
        self.gated_type = gated_type

        # Q, K, V ya vienen proyectados por el MLA producer del grupo (cache)

        # RoPE solo para dimensiones de rotacion
        self.rope = RoPE(head_dim=d_rotate, max_seq_len=max_seq_len, base=rope_base, scaling_factor=rope_scaling)

        # QK-Norm
        if qk_norm:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)

        # BMA filter
        if use_bma:
            self.bma_filter = BMAFilter(num_heads, head_dim)

        # Gate projection
        if use_gated_attn:
            if gated_type == "headwise":
                self.gate_proj = nn.Linear(num_heads * head_dim, num_heads, bias=bias)
            else:
                self.gate_proj = nn.Linear(num_heads * head_dim, num_heads * head_dim, bias=bias)

        self.o_proj = nn.Linear(num_heads * head_dim, d_model, bias=bias)
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Norms
        self.attn_norm = RMSNorm(d_model, eps=norm_eps)
        self.ffn_norm = RMSNorm(d_model, eps=norm_eps)

        # FFN
        inter_dim = compute_intermediate_dim(d_model, ffn_expansion, use_swiglu, ffn_round_to)
        if use_swiglu:
            self.ffn = SwiGLUFFN(d_model, inter_dim, ffn_dropout, bias)
        else:
            self.ffn = StandardFFN(d_model, inter_dim, ffn_dropout, bias)

        self.residual_dropout = nn.Identity()

        if causal:
            mask = torch.triu(torch.full((max_seq_len, max_seq_len), float("-inf")), diagonal=1)
            self.register_buffer("causal_mask", mask, persistent=False)

    def _decoupled_scores(self, Q_state, Q_rotate, K_state, K_rotate, seq_len, kv_len=None, causal_mask=None):
        """Decoupled content + RoPE scoring, igual que MLA."""
        nh, nkv, hd, dr = self.num_heads, self.num_kv_groups, self.head_dim, self.d_rotate
        scale_c = 1.0 / math.sqrt(self.d_c) if self.qk_norm else 1.0 / math.sqrt(hd)
        scale_r = 1.0

        k_c = repeat_kv(K_state, nh, nkv).transpose(1, 2)
        q_c = Q_state.transpose(1, 2)
        s_c = torch.matmul(q_c, k_c.transpose(-2, -1)) * scale_c

        q_r = Q_rotate.transpose(1, 2)
        k_r = K_rotate.transpose(1, 2).expand(-1, nh, -1, -1)
        s_r = torch.matmul(q_r, k_r.transpose(-2, -1)) * scale_r

        scores = s_c + s_r
        if self.attn_logit_cap is not None:
            scores = torch.tanh(scores / self.attn_logit_cap) * self.attn_logit_cap

        is_causal = self.causal if causal_mask is None else causal_mask
        if is_causal:
            if kv_len is None:
                kv_len = K_state.shape[1]
            if seq_len > 1:
                if seq_len == kv_len:
                    if seq_len <= self.causal_mask.shape[0]:
                        scores = scores + self.causal_mask[:seq_len, :seq_len]
                    else:
                        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1)
                        scores = scores + mask
                else:
                    mask = torch.triu(
                        torch.full((seq_len, kv_len), float("-inf"), device=scores.device),
                        diagonal=kv_len - seq_len + 1
                    )
                    scores = scores + mask
        return scores

    def forward(self, x: torch.Tensor, cache: tuple, offset: int = 0) -> torch.Tensor:
        """Training: lee (Q_state, Q_rotate_raw, K_state, V, K_rotate_raw) del producer."""
        B, T, _ = x.shape
        h = self.attn_norm(x)

        # Q/K/V desde la cache compartida del producer del grupo
        Q_state, Q_rotate_raw, K_state, V, K_rotate_raw = cache

        # RoPE solo en dimensiones de rotacion
        Q_rotate, K_rotate = self.rope(Q_rotate_raw, K_rotate_raw, offset)

        if self.qk_norm:
            Q_state = self.q_norm(Q_state)
            K_state = self.k_norm(K_state)

        scores = self._decoupled_scores(Q_state, Q_rotate, K_state, K_rotate, T)
        attn_w = F.softmax(scores, dim=-1)
        attn_w = self.attn_dropout(attn_w)

        n = self.num_heads // self.num_kv_groups
        V_exp = V.repeat_interleave(n, 2).transpose(1, 2)

        if self.use_bma:
            V_exp = self.bma_filter(Q_state.transpose(1, 2), V_exp)

        attn_out = torch.matmul(attn_w, V_exp)

        # Gated: post-aggregation gating
        if self.use_gated_attn:
            gate_raw = self.gate_proj(Q_state.transpose(1, 2).reshape(B, T, -1))
            if self.gated_type == "headwise":
                gate = gate_raw.permute(0, 2, 1).unsqueeze(-1)  # (B, H, T, 1)
            else:
                gate = gate_raw.reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            attn_out = attn_out * torch.sigmoid(gate)

        attn_out = attn_out.transpose(1, 2)
        h_attn = self.o_proj(attn_out.reshape(B, T, -1))
        x = x + h_attn

        # FFN
        residual = x
        h = self.ffn_norm(x)
        h = self.ffn(h)
        x = residual + h
        return x

    def forward_with_cache(self, x: torch.Tensor, offset: int, cache: tuple) -> torch.Tensor:
        """Inference: lee (Q_state, Q_rotate_raw, K_state, V, K_rotate_raw) del producer."""
        B, S_new, _ = x.shape
        h = self.attn_norm(x)

        if cache is not None:
            Q_state, Q_rotate_raw, K_state, V, K_rotate_raw = cache
            T = K_state.shape[1]
            # RoPE: Q_rotate con offset, K_rotate acumulado con offset=0
            Q_rotate = self.rope.apply_to_single(Q_rotate_raw, offset=offset)
            K_rotate = self.rope.apply_to_single(K_rotate_raw, offset=0)
        else:
            T = 0
            Q_state = torch.zeros(B, S_new, self.num_heads, self.head_dim, device=x.device)
            Q_rotate_raw = torch.zeros(B, S_new, self.num_heads, self.d_rotate, device=x.device)
            Q_rotate = self.rope.apply_to_single(Q_rotate_raw, offset=offset)
            K_state = torch.zeros(B, 0, self.num_kv_groups, self.head_dim, device=x.device)
            V = torch.zeros(B, 0, self.num_kv_groups, self.head_dim, device=x.device)
            K_rotate = torch.zeros(B, 0, 1, self.d_rotate, device=x.device)

        if self.qk_norm:
            Q_state = self.q_norm(Q_state)
            if T > 0:
                K_state = self.k_norm(K_state)

        scores = self._decoupled_scores(Q_state, Q_rotate, K_state, K_rotate, S_new, T, S_new > 1)
        attn_w = F.softmax(scores, dim=-1)
        attn_w = self.attn_dropout(attn_w)

        n = self.num_heads // self.num_kv_groups
        V_exp = V.repeat_interleave(n, 2).transpose(1, 2)

        if self.use_bma:
            V_exp = self.bma_filter(Q_state.transpose(1, 2), V_exp)

        attn_out = torch.matmul(attn_w, V_exp)

        if self.use_gated_attn:
            gate_raw = self.gate_proj(Q_state.transpose(1, 2).reshape(B, S_new, -1))
            if self.gated_type == "headwise":
                gate = gate_raw.permute(0, 2, 1).unsqueeze(-1)
            else:
                gate = gate_raw.reshape(B, S_new, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            attn_out = attn_out * torch.sigmoid(gate)

        attn_out = attn_out.transpose(1, 2)
        h_attn = self.o_proj(attn_out.reshape(B, S_new, -1))
        x = x + h_attn

        residual = x
        h = self.ffn_norm(x)
        h = self.ffn(h)
        x = residual + h
        return x


class Transformer(nn.Module):
    """Transformer stack with N layers + final RMSNorm.

    Soporta cache compartida cada `cache_every` capas:
      - cache_every=1: MLA cada capa (MLA full)
      - cache_every=N: MLA cada N capas, shared layers intermedias
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        num_kv_groups: int = 0,
        head_dim: int | None = None,
        ffn_expansion: float = 4.0,
        use_swiglu: bool = True,
        max_seq_len: int = 2048,
        rope_base: float = 10000.0,
        rope_scaling: float = 1.0,
        causal: bool = True,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        attn_logit_cap: float | None = None,
        bias: bool = False,
        norm_eps: float = 1e-5,
        ffn_round_to: int = 64,
        use_sparse_attn: bool = False,
        num_selected_blocks: int = 16,
        use_mla: bool = False,
        mla_d_c: int | None = None,
        mla_d_c1: int | None = None,
        mla_d_rotate: int | None = None,
        mla_block_size: int = 128,
        use_gated_attn: bool = False,
        gated_type: str = "headwise",
        use_xsa: bool = False,
        qk_norm: bool = True,
        use_sandwich_norm: bool = False,
        use_bma: bool = False,
        cache_every: int = 1,
    ):
        super().__init__()
        if num_kv_groups == 0:
            num_kv_groups = num_heads
        if head_dim is None:
            head_dim = d_model // num_heads
        if mla_d_c is None:
            mla_d_c = max(32, d_model // 6)
        if mla_d_rotate is None:
            mla_d_rotate = max(16, d_model // 12)

        self.cache_every = cache_every
        self.num_layers = num_layers
        self.d_model = d_model
        self.use_mla = use_mla
        self.use_sparse_attn = use_sparse_attn
        self.head_dim = head_dim
        self.num_kv_groups = num_kv_groups

        # Indices de capas que producen cache
        self.cache_producer_indices = list(range(0, num_layers, cache_every))
        self.is_cache_producer = [i in self.cache_producer_indices for i in range(num_layers)]

        # Crear capas
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if self.is_cache_producer[i]:
                # MLA layer: produce C_KV cache
                layer = TransformerLayer(
                    d_model=d_model, num_heads=num_heads, num_kv_groups=num_kv_groups,
                    head_dim=head_dim, ffn_expansion=ffn_expansion, use_swiglu=use_swiglu,
                    max_seq_len=max_seq_len, rope_base=rope_base, rope_scaling=rope_scaling,
                    causal=causal, attn_dropout=attn_dropout, ffn_dropout=ffn_dropout,
                    residual_dropout=residual_dropout, attn_logit_cap=attn_logit_cap,
                    bias=bias, norm_eps=norm_eps, ffn_round_to=ffn_round_to,
                    use_sparse_attn=use_sparse_attn, num_selected_blocks=num_selected_blocks,
                    use_mla=use_mla, mla_d_c=mla_d_c, mla_d_c1=mla_d_c1,
                    mla_d_rotate=mla_d_rotate, mla_block_size=mla_block_size,
                    use_gated_attn=use_gated_attn, gated_type=gated_type,
                    use_xsa=use_xsa, qk_norm=qk_norm, use_sandwich_norm=use_sandwich_norm,
                    use_bma=use_bma,
                )
            else:
                # Shared cache layer: lee C_KV de cache compartida
                layer = SharedCacheTransformerLayer(
                    d_model=d_model, num_heads=num_heads, num_kv_groups=num_kv_groups,
                    head_dim=head_dim, d_c=mla_d_c, d_rotate=mla_d_rotate,
                    max_seq_len=max_seq_len, rope_base=rope_base, rope_scaling=rope_scaling,
                    causal=causal, dropout=attn_dropout, attn_logit_cap=attn_logit_cap,
                    bias=bias, norm_eps=norm_eps, ffn_expansion=ffn_expansion,
                    use_swiglu=use_swiglu, ffn_dropout=ffn_dropout, ffn_round_to=ffn_round_to,
                    use_gated_attn=use_gated_attn, gated_type=gated_type,
                    qk_norm=qk_norm, use_bma=use_bma,
                )
            self.layers.append(layer)

        self.final_norm = RMSNorm(d_model, eps=norm_eps)

    def _get_group_idx(self, layer_idx: int) -> int:
        """Get cache group index for a given layer index."""
        group_idx = 0
        for gi, pi in enumerate(self.cache_producer_indices):
            if pi < layer_idx:
                group_idx = gi
            else:
                break
        return group_idx

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Forward through all layers + final norm. Shared cache training."""
        num_groups = len(self.cache_producer_indices)
        shared_caches = [None] * num_groups

        for i, layer in enumerate(self.layers):
            if self.is_cache_producer[i]:
                # MLA layer: produce (Q_state, Q_rotate_raw, K_state, V, K_rotate_raw)
                x, shared_proj = layer.forward_produce_cache(x, offset)
                if shared_proj is not None:
                    group_idx = self.cache_producer_indices.index(i)
                    shared_caches[group_idx] = shared_proj
            else:
                # Shared cache layer: lee Q/K/V proyectados del producer de su grupo
                group_idx = self._get_group_idx(i)
                x = layer(x, shared_caches[group_idx], offset)

        return self.final_norm(x)

    def forward_with_cache(
        self,
        x: torch.Tensor,
        offset: int,
        caches: list,
    ) -> tuple[torch.Tensor, list]:
        """Forward with shared caches for autoregressive generation.

        caches: list por grupo de (mla_cache, shared_proj), o None.
        """
        num_groups = len(self.cache_producer_indices)
        new_caches = list(caches) if caches is not None else [None] * num_groups

        for i, layer in enumerate(self.layers):
            if self.is_cache_producer[i]:
                group_idx = self.cache_producer_indices.index(i)
                residual = x
                h_pre = layer.attn_norm(x)
                if new_caches[group_idx] is not None:
                    mla_cache = new_caches[group_idx][0]
                else:
                    mla_cache = None
                h, new_mla_cache = layer.attention.forward_with_cache(h_pre, offset, mla_cache)
                h = layer.residual_dropout(h)
                x = residual + h

                residual = x
                h = layer.ffn_norm(x)
                h = layer.ffn(h)
                x = residual + h

                # Proyecciones compartidas para las capas del grupo
                shared_proj = self._producer_shared_proj(layer, h_pre, new_mla_cache, new_caches[group_idx])
                new_caches[group_idx] = (new_mla_cache, shared_proj)
            else:
                # Shared cache layer: lee Q/K/V del producer de su grupo
                group_idx = self._get_group_idx(i)
                shared_cache = new_caches[group_idx][1] if new_caches[group_idx] is not None else None
                x = layer.forward_with_cache(x, offset, shared_cache)

        return self.final_norm(x), new_caches

    def _producer_shared_proj(self, layer, h_pre, new_mla_cache, prev_group_cache):
        """Construye (Q_state, Q_rotate_raw, K_state, V, K_rotate_raw) del MLA producer.

        Q es del token nuevo (h_pre); K_state/V/K_rotate_raw acumulan la secuencia
        completa desde C_KV_full (ya normalizado) y el K_rotate_raw acumulado.
        """
        qkv = layer.attention.qkv
        B, S_new, _ = h_pre.shape
        down = qkv.W_down(h_pre)
        _, _, K_rot_raw_new = down.split([qkv.d_c1, qkv.d_c, qkv.d_rotate], dim=-1)
        C_Q_new = down[:, :, :qkv.d_c1]
        C_Q_new = qkv.norm_cq(C_Q_new)
        q_up = qkv.W_up_q(C_Q_new)
        Q_state, Q_rot_raw = q_up.split([layer.num_heads * layer.head_dim, layer.num_heads * qkv.d_rotate], dim=-1)
        Q_state = Q_state.reshape(B, S_new, layer.num_heads, layer.head_dim)
        Q_rot_raw = Q_rot_raw.reshape(B, S_new, layer.num_heads, qkv.d_rotate)
        K_rot_raw_new = K_rot_raw_new.unsqueeze(2)

        C_KV_full = new_mla_cache[0]
        kv_up = qkv.W_up_kv(C_KV_full)
        K_state, V = kv_up.chunk(2, dim=-1)
        T = C_KV_full.shape[1]
        K_state = K_state.reshape(B, T, layer.num_kv_groups, layer.head_dim)
        V = V.reshape(B, T, layer.num_kv_groups, layer.head_dim)

        if prev_group_cache is not None:
            K_rot_raw_full = torch.cat([prev_group_cache[1][4], K_rot_raw_new], dim=1)
        else:
            K_rot_raw_full = K_rot_raw_new
        return (Q_state, Q_rot_raw, K_state, V, K_rot_raw_full)
