"""Transformer Block compatible with Rust blocks::trasformer::layer.

Architecture per layer:
  x -> RMSNorm -> Attention(GQA + RoPE) -> +residual
    -> RMSNorm -> FeedForward(SwiGLU) -> +residual -> output

Also includes the full Transformer stack with final RMSNorm.

Grouped shared cache:
  - cache_every=N: cada N capas, una MLA layer produce C_KV
  - Las capas intermedias son SharedCacheTransformerLayer con su propio W_up_kv
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

    def forward_produce_cache(self, x: torch.Tensor, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward for MLA producer layers: returns (output, C_KV).

        Extracts C_KV from pre-attention h (same input as MLA qkv), then runs attention.
        C_KV is the normalized latent stored in cache for shared layers.
        """
        residual = x
        h = self.attn_norm(x)

        # Extract C_KV BEFORE attention (same h that MLA qkv receives)
        if self.use_mla and hasattr(self.attention, 'qkv'):
            down = self.attention.qkv.W_down(h)
            C_KV = down[:, :, self.attention.qkv.d_c1:self.attention.qkv.d_c1 + self.attention.qkv.d_c]
            C_KV = self.attention.qkv.norm_ckv(C_KV)
        else:
            C_KV = None

        h_attn = self.attention(h, offset)
        h_attn = self.residual_dropout(h_attn)
        x = residual + h_attn

        residual = x
        h = self.ffn_norm(x)
        h = self.ffn(h)
        h = self.residual_dropout(h)
        return residual + h, C_KV

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
    """Capa que lee C_KV de una cache compartida (producida por MLA layer).

    - Q viene de x via q_proj propio (sin compresion latent)
    - K,V se descomprimen de C_KV via W_up_kv propio (pesos diferentes por capa)
    - RoPE se aplica a K desde el cache raw
    - Compatible con MLA cache: C_KV
    - Soporta BMA (pre-aggregation) y Gated (post-aggregation)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_groups: int,
        head_dim: int,
        d_c: int,
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
        self.causal = causal
        self.attn_logit_cap = attn_logit_cap
        self.qk_norm = qk_norm
        self.use_bma = use_bma
        self.use_gated_attn = use_gated_attn
        self.gated_type = gated_type

        # Q projection: directo desde x
        self.q_proj = nn.Linear(d_model, num_heads * head_dim, bias=bias)

        # K,V decompression: C_KV -> K, V (SUS pesos, diferentes a MLA layer)
        self.W_up_kv = nn.Linear(d_c, 2 * num_kv_groups * head_dim, bias=bias)

        # RoPE para Q y K
        self.rope = RoPE(head_dim=head_dim, max_seq_len=max_seq_len, base=rope_base, scaling_factor=rope_scaling)

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

    def forward(self, x: torch.Tensor, C_KV: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Training: lee C_KV de cache compartida, descomprime con sus pesos."""
        B, T, _ = x.shape
        h = self.attn_norm(x)

        # Q directo desde x
        Q = self.q_proj(h).reshape(B, T, self.num_heads, self.head_dim)

        # K,V desde cache compartida via SUS pesos
        kv_up = self.W_up_kv(C_KV)
        K, V = kv_up.chunk(2, dim=-1)
        K = K.reshape(B, T, self.num_kv_groups, self.head_dim)
        V = V.reshape(B, T, self.num_kv_groups, self.head_dim)

        # RoPE para Q y K
        Q, K = self.rope(Q, K, offset)

        if self.qk_norm:
            Q = self.q_norm(Q)
            K = self.k_norm(K)

        # GQA expand
        n = self.num_heads // self.num_kv_groups
        K_exp = K.repeat_interleave(n, 2)
        V_exp = V.repeat_interleave(n, 2)

        Q, K_exp, V_exp = [t.transpose(1, 2) for t in (Q, K_exp, V_exp)]

        # BMA: pre-aggregation gating
        if self.use_bma:
            V_exp = self.bma_filter(Q, V_exp)

        # Attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K_exp.transpose(-2, -1)) * scale

        if self.attn_logit_cap is not None:
            scores = torch.tanh(scores / self.attn_logit_cap) * self.attn_logit_cap

        if self.causal:
            if T <= self.causal_mask.shape[0]:
                scores = scores + self.causal_mask[:T, :T]
            else:
                mask = torch.triu(torch.full((T, T), float("-inf"), device=scores.device), diagonal=1)
                scores = scores + mask

        attn_w = F.softmax(scores, dim=-1)
        attn_w = self.attn_dropout(attn_w)
        attn_out = torch.matmul(attn_w, V_exp)

        # Gated: post-aggregation gating
        if self.use_gated_attn:
            gate_raw = self.gate_proj(Q.transpose(1, 2).reshape(B, T, -1))
            if self.gated_type == "headwise":
                gate = gate_raw.reshape(B, T, self.num_heads, 1)
            else:
                gate = gate_raw.reshape(B, T, self.num_heads, self.head_dim)
            gate = gate.expand(-1, -1, -1, self.head_dim)
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

    def forward_with_cache(self, x: torch.Tensor, offset: int, C_KV: torch.Tensor) -> torch.Tensor:
        """Inference: lee C_KV completo de cache compartida."""
        B, S_new, _ = x.shape
        h = self.attn_norm(x)

        Q = self.q_proj(h).reshape(B, S_new, self.num_heads, self.head_dim)

        if C_KV is not None and C_KV.shape[1] > 0:
            T = C_KV.shape[1]
            kv_up = self.W_up_kv(C_KV)
            K, V = kv_up.chunk(2, dim=-1)
            K = K.reshape(B, T, self.num_kv_groups, self.head_dim)
            V = V.reshape(B, T, self.num_kv_groups, self.head_dim)
            Q, K = self.rope(Q, K, offset)
        else:
            T = 0
            K = torch.zeros(B, 0, self.num_kv_groups, self.head_dim, device=x.device)
            V = torch.zeros(B, 0, self.num_kv_groups, self.head_dim, device=x.device)

        if self.qk_norm:
            Q = self.q_norm(Q)
            if T > 0:
                K = self.k_norm(K)

        n = self.num_heads // self.num_kv_groups
        K_exp = K.repeat_interleave(n, 2)
        V_exp = V.repeat_interleave(n, 2)

        Q, K_exp, V_exp = [t.transpose(1, 2) for t in (Q, K_exp, V_exp)]

        if self.use_bma:
            V_exp = self.bma_filter(Q, V_exp)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K_exp.transpose(-2, -1)) * scale

        if self.attn_logit_cap is not None:
            scores = torch.tanh(scores / self.attn_logit_cap) * self.attn_logit_cap

        if self.causal and S_new > 1:
            kv_len = K_exp.shape[2]
            if S_new == kv_len:
                if S_new <= self.causal_mask.shape[0]:
                    scores = scores + self.causal_mask[:S_new, :S_new]
                else:
                    mask = torch.triu(torch.full((S_new, S_new), float("-inf"), device=scores.device), diagonal=1)
                    scores = scores + mask
            else:
                mask = torch.triu(
                    torch.full((S_new, kv_len), float("-inf"), device=scores.device),
                    diagonal=kv_len - S_new + 1
                )
                scores = scores + mask

        attn_w = F.softmax(scores, dim=-1)
        attn_w = self.attn_dropout(attn_w)
        attn_out = torch.matmul(attn_w, V_exp)

        if self.use_gated_attn:
            gate_raw = self.gate_proj(Q.transpose(1, 2).reshape(B, S_new, -1))
            if self.gated_type == "headwise":
                gate = gate_raw.reshape(B, S_new, self.num_heads, 1)
            else:
                gate = gate_raw.reshape(B, S_new, self.num_heads, self.head_dim)
            gate = gate.expand(-1, -1, -1, self.head_dim)
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
                    head_dim=head_dim, d_c=mla_d_c,
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
                # MLA layer: produce C_KV
                x, C_KV = layer.forward_produce_cache(x, offset)
                if C_KV is not None:
                    group_idx = self.cache_producer_indices.index(i)
                    if shared_caches[group_idx] is None:
                        shared_caches[group_idx] = C_KV
                    else:
                        shared_caches[group_idx] = torch.cat([shared_caches[group_idx], C_KV], 1)
            else:
                # Shared cache layer: lee C_KV de cache correspondiente
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

        caches: list of C_KV per cache group, or None
        """
        num_groups = len(self.cache_producer_indices)
        new_caches = list(caches) if caches is not None else [None] * num_groups

        for i, layer in enumerate(self.layers):
            if self.is_cache_producer[i]:
                # MLA layer: produce C_KV
                group_idx = self.cache_producer_indices.index(i)
                residual = x
                h = layer.attn_norm(x)
                h, new_cache = layer.attention.forward_with_cache(h, offset, new_caches[group_idx])
                h = layer.residual_dropout(h)
                x = residual + h

                residual = x
                h = layer.ffn_norm(x)
                h = layer.ffn(h)
                x = residual + h

                # new_cache is (C_KV_full, K_rot_full) — store C_KV only
                if new_cache is not None:
                    new_caches[group_idx] = new_cache[0]  # C_KV only
            else:
                # Shared cache layer: lee del cache de su grupo
                group_idx = self._get_group_idx(i)
                x = layer.forward_with_cache(x, offset, new_caches[group_idx])

        return self.final_norm(x), new_caches
