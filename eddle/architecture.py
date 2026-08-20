import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


DTYPE_MAP = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


@dataclass
class TransformerConfig:
    vocab_size: int = 8192
    d_model: int = 512
    num_heads: int = 8
    num_kv_heads: int = 4
    num_encoder_layers: int = 12
    num_decoder_layers: int = 8
    d_ff: int = 2048
    max_seq_len: int = 1024
    pad_token_id: int = 0
    rope_theta: float = 10000.0
    dtype: str = "bfloat16"
    activation: str = "drelu"
    dropout_rate: float = 0.1
    contrastive_dim: int = 128
    no_feedforward: bool = True

    @property
    def torch_dtype(self):
        return DTYPE_MAP[self.dtype]


def precompute_rope_freqs(head_dim: int, seq_len: int, theta: float = 10000.0, device=None):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    angles = torch.outer(t, freqs)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    T = x.shape[2]
    half = x.shape[-1] // 2
    cos = cos[:T][None, None, :, :]
    sin = sin[:T][None, None, :, :]
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class ZCRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, dtype=torch.bfloat16):
        super().__init__()
        self.eps = eps
        self.dtype = dtype
        self.scale = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        out = ((1 + self.scale) * x / rms)
        return out.to(self.dtype)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, num_kv_heads, d_model, num_layers, dtype=torch.bfloat16):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.num_layers = num_layers
        self.dtype = dtype

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.head_dim * num_kv_heads, bias=False)
        self.v_proj = nn.Linear(d_model, self.head_dim * num_kv_heads, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.q_norm = ZCRMSNorm(self.head_dim, dtype=dtype)
        self.k_norm = ZCRMSNorm(self.head_dim, dtype=dtype)

        self._init_residual(self.out_proj, num_layers)

    @staticmethod
    def _init_residual(module, num_layers):
        nn.init.normal_(module.weight, std=0.02 / math.sqrt(2 * num_layers))

    def forward(self, q_input, kv_input, mask=None, rope=None):
        B = q_input.shape[0]
        dtype = self.dtype

        q = self.q_proj(q_input.to(dtype))
        k = self.k_proj(kv_input.to(dtype))
        v = self.v_proj(kv_input.to(dtype))

        q = q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        repeats = self.num_heads // self.num_kv_heads
        if repeats > 1:
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        if rope is not None:
            cos, sin = rope
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

        if mask is not None:
            attn_weights = attn_weights.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.head_dim)
        return self.out_proj(out.to(dtype))


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, num_layers, dtype=torch.bfloat16, activation="drelu"):
        super().__init__()
        self.dtype = dtype
        self.activation = activation
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        nn.init.normal_(self.down_proj.weight, std=0.02 / math.sqrt(2 * num_layers))

    def forward(self, x, ffn_mask=None):
        dtype = self.dtype
        gate = self.gate_proj(x.to(dtype))
        up = self.up_proj(x.to(dtype))
        if self.activation == "swiglu":
            h = F.silu(gate) * up
        elif self.activation == "geglu":
            h = F.gelu(gate) * up
        else:
            h = F.relu(gate) * F.relu(up)
        if ffn_mask is not None:
            h = h * ffn_mask[:, None, :]
        return self.down_proj(h.to(dtype))


class EncoderBlock(nn.Module):
    def __init__(self, num_heads, num_kv_heads, d_model, d_ff, num_layers,
                 dtype=torch.bfloat16, activation="drelu", dropout_rate=0.0,
                 no_feedforward=True):
        super().__init__()
        self.dtype = dtype
        self.dropout_rate = dropout_rate
        self.no_feedforward = no_feedforward

        self.norm1 = ZCRMSNorm(d_model, dtype=dtype)
        self.self_attn = MultiHeadAttention(num_heads, num_kv_heads, d_model, num_layers, dtype)
        self.drop = nn.Dropout(dropout_rate)
        self.attn_gate = nn.Parameter(torch.zeros(()))

        if not no_feedforward:
            self.norm2 = ZCRMSNorm(d_model, dtype=dtype)
            self.ffn = FeedForward(d_model, d_ff, num_layers, dtype, activation)
            self.ffn_gate = nn.Parameter(torch.zeros(()))
            self.ffn_drop = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None, rope=None, ffn_mask=None):
        dtype = self.dtype
        gate = torch.sigmoid(self.attn_gate).to(dtype)
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, mask=mask, rope=rope)
        x = residual + gate * self.drop(x)

        if not self.no_feedforward:
            ffn_gate = torch.sigmoid(self.ffn_gate).to(dtype)
            residual = x
            x = self.norm2(x)
            x = self.ffn(x, ffn_mask=ffn_mask)
            x = residual + ffn_gate * self.ffn_drop(x)

        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dtype = config.torch_dtype
        d_ff = config.d_ff or config.d_model * 4
        total_layers = config.num_encoder_layers + config.num_decoder_layers
        self.layers = nn.ModuleList([
            EncoderBlock(
                config.num_heads, config.num_kv_heads, config.d_model, d_ff,
                total_layers, dtype, config.activation, config.dropout_rate,
                config.no_feedforward,
            )
            for _ in range(config.num_encoder_layers)
        ])
        self.final_norm = ZCRMSNorm(config.d_model, dtype=dtype)

    def forward(self, x, mask=None, rope=None, ffn_mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask, rope=rope, ffn_mask=ffn_mask)
        x = self.final_norm(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, num_heads, num_kv_heads, d_model, d_ff, num_layers,
                 dtype=torch.bfloat16, activation="drelu", dropout_rate=0.0,
                 no_feedforward=True):
        super().__init__()
        self.dtype = dtype
        self.dropout_rate = dropout_rate
        self.no_feedforward = no_feedforward

        self.norm1 = ZCRMSNorm(d_model, dtype=dtype)
        self.self_attn = MultiHeadAttention(num_heads, num_kv_heads, d_model, num_layers, dtype)
        self.self_drop = nn.Dropout(dropout_rate)
        self.self_attn_gate = nn.Parameter(torch.zeros(()))

        self.norm2 = ZCRMSNorm(d_model, dtype=dtype)
        self.cross_attn = MultiHeadAttention(num_heads, num_kv_heads, d_model, num_layers, dtype)
        self.cross_drop = nn.Dropout(dropout_rate)
        self.cross_attn_gate = nn.Parameter(torch.zeros(()))

        if not no_feedforward:
            self.norm3 = ZCRMSNorm(d_model, dtype=dtype)
            self.ffn = FeedForward(d_model, d_ff, num_layers, dtype, activation)
            self.ffn_gate = nn.Parameter(torch.zeros(()))
            self.ffn_drop = nn.Dropout(dropout_rate)

    def forward(self, x, encoder_out, self_mask=None, cross_mask=None, rope=None, ffn_mask=None):
        dtype = self.dtype

        self_gate = torch.sigmoid(self.self_attn_gate).to(dtype)
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, mask=self_mask, rope=rope)
        x = residual + self_gate * self.self_drop(x)

        cross_gate = torch.sigmoid(self.cross_attn_gate).to(dtype)
        residual = x
        x = self.norm2(x)
        x = self.cross_attn(x, encoder_out, mask=cross_mask)
        x = residual + cross_gate * self.cross_drop(x)

        if not self.no_feedforward:
            ffn_gate = torch.sigmoid(self.ffn_gate).to(dtype)
            residual = x
            x = self.norm3(x)
            x = self.ffn(x, ffn_mask=ffn_mask)
            x = residual + ffn_gate * self.ffn_drop(x)

        return x


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dtype = config.torch_dtype
        d_ff = config.d_ff or config.d_model * 4
        total_layers = config.num_encoder_layers + config.num_decoder_layers
        self.layers = nn.ModuleList([
            DecoderBlock(
                config.num_heads, config.num_kv_heads, config.d_model, d_ff,
                total_layers, dtype, config.activation, config.dropout_rate,
                config.no_feedforward,
            )
            for _ in range(config.num_decoder_layers)
        ])
        self.final_norm = ZCRMSNorm(config.d_model, dtype=dtype)

    def forward(self, x, encoder_out, self_mask=None, cross_mask=None, rope=None, ffn_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_out, self_mask=self_mask, cross_mask=cross_mask,
                      rope=rope, ffn_mask=ffn_mask)
        x = self.final_norm(x)
        return x


class SimpleAttentionNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dtype = config.torch_dtype
        d_ff = config.d_ff or config.d_model * 4

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.embed_scale = math.sqrt(config.d_model)

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.contrastive_hidden = nn.Linear(config.d_model, config.d_model // 4, bias=True)
        self.contrastive_proj = nn.Linear(config.d_model // 4, config.contrastive_dim, bias=False)

        nn.init.normal_(self.contrastive_hidden.weight, std=0.02)
        nn.init.normal_(self.contrastive_proj.weight, std=0.02)

        self.log_temp = nn.Parameter(torch.zeros(()))

        self._rope_cache = {}

    def _rope(self, seq_len, device=None):
        key = (seq_len, device)
        if key not in self._rope_cache:
            head_dim = self.config.d_model // self.config.num_heads
            self._rope_cache[key] = precompute_rope_freqs(
                head_dim, seq_len, self.config.rope_theta, device)
        return self._rope_cache[key]

    def encode_text(self, src, src_mask=None, ffn_mask=None):
        dtype = self.config.torch_dtype
        x = self.embedding(src.long()) * self.embed_scale
        x = x.to(dtype)
        rope = self._rope(src.shape[1], src.device)
        out = self.encoder(x, mask=src_mask, rope=rope, ffn_mask=ffn_mask)
        return out

    def encode(self, src, src_mask=None):
        return self.encode_text(src, src_mask=src_mask)

    def decode(self, tgt, encoder_out, self_mask=None, cross_mask=None):
        dtype = self.config.torch_dtype
        x = self.embedding(tgt.long()) * self.embed_scale
        x = x.to(dtype)
        rope = self._rope(tgt.shape[1], tgt.device)
        x = self.decoder(x, encoder_out, self_mask=self_mask, cross_mask=cross_mask, rope=rope)
        logits = x.float() @ self.embedding.weight.T
        return logits

    def _mean_pool(self, encoder_out, enc_mask):
        if enc_mask is not None:
            mask_2d = enc_mask[:, 0, 0, :]
        else:
            mask_2d = torch.ones(encoder_out.shape[:2], device=encoder_out.device, dtype=encoder_out.dtype)
        mask_3d = mask_2d[:, :, None]
        summed = (encoder_out * mask_3d).sum(dim=1)
        counts = mask_2d.sum(dim=1, keepdim=True).clamp(min=1.0)
        return summed / counts

    def encode_contrastive(self, tokens):
        src_mask = make_padding_mask(tokens, self.config.pad_token_id)
        encoder_out = self.encode_text(tokens, src_mask=src_mask)
        pooled = self._mean_pool(encoder_out, src_mask)
        h = F.relu(self.contrastive_hidden(pooled))
        projected = self.contrastive_proj(h)
        denom = torch.sqrt(projected.float().pow(2).sum(-1, keepdim=True) + 1e-12)
        return (projected / denom.to(projected.dtype))

    def forward_contrastive(self, query_tokens, tool_tokens):
        q_emb = self.encode_contrastive(query_tokens)
        t_emb = self.encode_contrastive(tool_tokens)
        return q_emb, t_emb, self.log_temp

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, cross_mask=None):
        encoder_out = self.encode_text(src, src_mask=src_mask)
        cm = cross_mask if cross_mask is not None else src_mask
        logits = self.decode(tgt, encoder_out, self_mask=tgt_mask, cross_mask=cm)
        return logits

    def forward_masked(self, src, tgt, src_mask=None, tgt_mask=None, cross_mask=None,
                       ffn_mask=None):
        dtype = self.config.torch_dtype
        encoder_out = self.encode_text(src, src_mask=src_mask, ffn_mask=ffn_mask)
        cm = cross_mask if cross_mask is not None else src_mask

        x = self.embedding(tgt.long()) * self.embed_scale
        x = x.to(dtype)
        rope = self._rope(tgt.shape[1], tgt.device)
        x_f32 = self.decoder(x, encoder_out, self_mask=tgt_mask, cross_mask=cm,
                             rope=rope, ffn_mask=ffn_mask).float()
        logits = x_f32 @ self.embedding.weight.T
        return logits


def make_causal_mask(seq_len, device=None):
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    return mask[None, None, :, :]


def make_padding_mask(tokens, pad_token_id):
    mask = tokens != pad_token_id
    return mask[:, None, None, :]


def make_packing_mask(seg_ids):
    mask = (seg_ids[:, :, None] == seg_ids[:, None, :]) & (seg_ids[:, :, None] > 0)
    return mask[:, None, :, :]


def make_causal_packing_mask(seg_ids):
    T = seg_ids.shape[1]
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=seg_ids.device))
    block = (seg_ids[:, :, None] == seg_ids[:, None, :]) & (seg_ids[:, :, None] > 0)
    return (block & causal[None, :, :])[:, None, :, :]


def make_cross_packing_mask(enc_seg_ids, dec_seg_ids):
    mask = (dec_seg_ids[:, :, None] == enc_seg_ids[:, None, :]) & (dec_seg_ids[:, :, None] > 0)
    return mask[:, None, :, :]
