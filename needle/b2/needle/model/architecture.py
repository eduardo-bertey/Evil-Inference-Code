import math
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp
import jax.nn.initializers as jinit
import flax.linen as nn

from . import quantize as _quantize
from .quantize import fake_quant_act


def _aq(x, quant):
    if quant is False:
        return x
    return jax.lax.cond(quant, fake_quant_act, lambda t: t, x)


def default_init():
    return jinit.normal(stddev=0.02)


def residual_init(num_layers):
    return jinit.normal(stddev=0.02 / math.sqrt(2 * num_layers))


DTYPE_MAP = {"float32": jnp.float32, "bfloat16": jnp.bfloat16, "float16": jnp.float16}

PRESETS = {
    "needle": dict(d_model=768, num_heads=12, num_kv_heads=6, num_layers=27,
                   engram_layers=(2, 15)),
}
PRESETS["base"] = dict(d_model=512, num_heads=8, num_kv_heads=4, num_layers=27,
                       engram_layers=(2, 15))


class ZCRMSNorm(nn.Module):
    epsilon: float = 1e-6
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", jinit.zeros, (x.shape[-1],))
        rms = jnp.sqrt(jnp.mean(x.astype(jnp.float32) ** 2, axis=-1, keepdims=True) + self.epsilon)
        return ((1 + scale) * x / rms).astype(self.dtype)



@dataclass
class TransformerConfig:
    vocab_size: int = 8192
    d_model: int = 512
    attn_dim: int = 0
    num_heads: int = 8
    num_kv_heads: int = 4
    num_layers: int = 12
    max_seq_len: int = 2048
    pad_token_id: int = 0
    contrastive_dim: int = 128
    rope_theta: float = 100000.0
    dtype: str = "bfloat16"
    flash: bool = True
    engram_orders: tuple = (2, 3)
    engram_heads: int = 0
    engram_slots: int = 8192
    engram_layers: tuple = (2, 15)
    mhc_lanes: int = 4
    kv_window: int = 0
    kv_bits: int = 8
    act_bits: int = 8
    weight_bits: str = ""

    def __init__(self, **kwargs):
        valid = {f.name for f in self.__dataclass_fields__.values()}
        for k, v in kwargs.items():
            if k in valid:
                setattr(self, k, v)
        self.attn_dim = self.attn_dim or self.d_model
        self.engram_layers = tuple(self.engram_layers)

    @property
    def jax_dtype(self):
        return DTYPE_MAP[self.dtype]


def precompute_rope_freqs(head_dim, seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2).astype(jnp.float32) / head_dim))
    t = jnp.arange(seq_len).astype(jnp.float32)
    angles = jnp.outer(t, freqs)
    return jnp.cos(angles), jnp.sin(angles)


def apply_rope(x, cos, sin):
    T = x.shape[2]
    half = x.shape[-1] // 2
    cos = cos[:T][None, None, :, :]
    sin = sin[:T][None, None, :, :]
    x1 = x[..., :half]
    x2 = x[..., half:]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(x.dtype)


ENGRAM_SUB_DIM = 128
ENGRAM_CONV_TAPS = 4
_ENGRAM_SEED = 0x9E3779B9
_ENGRAM_PRIME = 0x01000193


def engram_geometry(config):
    orders = tuple(config.engram_orders)
    heads = config.engram_heads or max(1, config.d_model // (len(orders) * ENGRAM_SUB_DIM))
    sub_dim = config.d_model // (len(orders) * heads)
    return orders, heads, sub_dim


def _shift_right(x, offset):
    if offset == 0:
        return x
    pad = [(0, 0)] * x.ndim
    pad[1] = (offset, 0)
    return jnp.pad(x, pad)[:, : x.shape[1]]


def _mask_diag(mask, offset):
    m = mask[:, 0]
    T = m.shape[-1]
    if offset >= T:
        return jnp.zeros(m.shape[:-2] + (T,), m.dtype)
    d = jnp.diagonal(m, offset=-offset, axis1=-2, axis2=-1)
    if offset == 0:
        return d
    return jnp.pad(d, ((0, 0), (offset, 0)))


def engram_indices(tokens, orders, heads, slots):
    u = tokens.astype(jnp.uint32)
    idx = []
    for oi, order in enumerate(orders):
        for h in range(heads):
            seed = (_ENGRAM_SEED * (oi * heads + h + 1)) & 0xFFFFFFFF
            acc = jnp.full_like(u, jnp.uint32(seed))
            for j in range(order):
                acc = (acc ^ _shift_right(u, j)) * jnp.uint32(_ENGRAM_PRIME)
            acc = acc ^ (acc >> jnp.uint32(15))
            idx.append((acc % jnp.uint32(slots)).astype(jnp.int32))
    return jnp.stack(idx, axis=-1)


def _rms_unit(x, epsilon=1e-6):
    xf = x.astype(jnp.float32)
    return xf * jax.lax.rsqrt(jnp.mean(xf ** 2, axis=-1, keepdims=True) + epsilon)


def _conv_identity_init(key, shape, dtype=jnp.float32):
    return jnp.zeros(shape, dtype).at[0].set(1.0)


def _sinkhorn(logits, iters=20):
    log_K = logits
    for _ in range(iters):
        log_K = log_K - jax.nn.logsumexp(log_K, axis=-1, keepdims=True)
        log_K = log_K - jax.nn.logsumexp(log_K, axis=-2, keepdims=True)
    return jnp.exp(log_K)


def _res_identity_init(key, shape, dtype=jnp.float32):
    return jnp.broadcast_to(4.0 * jnp.eye(shape[-1], dtype=dtype), shape)


class Engram(nn.Module):
    d_model: int
    num_tables: int
    slots: int
    sub_dim: int
    num_layers: int
    conv_dilation: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, indices, ngram_ok, tap_ok, quant=False):
        tables = self.param("embedding", default_init(),
                            (self.num_tables, self.slots, self.sub_dim))
        fetched = tables[jnp.arange(self.num_tables), indices]
        fetched = fetched * ngram_ok[..., None]
        e = fetched.reshape(*indices.shape[:2], self.num_tables * self.sub_dim)
        e = _aq(e.astype(self.dtype), quant)
        k = nn.Dense(self.d_model, dtype=self.dtype, use_bias=False,
                     kernel_init=default_init(), name="key_proj")(e)
        v = nn.Dense(self.d_model, dtype=self.dtype, use_bias=False,
                     kernel_init=residual_init(self.num_layers), name="value_proj")(e)
        taps = self.param("taps", _conv_identity_init,
                          (ENGRAM_CONV_TAPS, self.d_model)).astype(self.dtype)
        v = sum(taps[j] * _shift_right(v, j * self.conv_dilation) * tap_ok[j][..., None]
                for j in range(ENGRAM_CONV_TAPS))
        return k, v


class MultiHeadAttention(nn.Module):
    num_heads: int
    num_kv_heads: int
    d_model: int
    num_layers: int
    dtype: jnp.dtype = jnp.bfloat16
    flash: bool = True
    attn_dim: int = 0

    @nn.compact
    def __call__(self, x, mask=None, rope=None, quant=False):
        attn_dim = self.attn_dim or self.d_model
        head_dim = attn_dim // self.num_heads
        kv_dim = self.num_kv_heads * head_dim
        B = x.shape[0]

        x = _aq(x, quant)
        q = nn.Dense(attn_dim, dtype=self.dtype, use_bias=False, kernel_init=default_init(), name="q_proj")(x)
        k = nn.Dense(kv_dim, dtype=self.dtype, use_bias=False, kernel_init=default_init(), name="k_proj")(x)
        v = nn.Dense(kv_dim, dtype=self.dtype, use_bias=False, kernel_init=default_init(), name="v_proj")(x)

        q = q.reshape(B, -1, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, -1, self.num_kv_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, -1, self.num_kv_heads, head_dim).transpose(0, 2, 1, 3)

        q = ZCRMSNorm(dtype=self.dtype, name="q_norm")(q)
        k = ZCRMSNorm(dtype=self.dtype, name="k_norm")(k)

        if rope is not None:
            cos, sin = rope
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        k = _quantize.maybe_quant_kv(k, quant)
        v = _quantize.maybe_quant_kv(v, quant)

        if self.flash:
            impl = ("cudnn" if jax.default_backend() == "gpu"
                    and q.dtype in (jnp.bfloat16, jnp.float16) else None)
            out = jax.nn.dot_product_attention(
                q.transpose(0, 2, 1, 3),
                k.transpose(0, 2, 1, 3),
                v.transpose(0, 2, 1, 3),
                mask=mask,
                implementation=impl,
            )
            out = out.reshape(B, -1, attn_dim)
        else:
            repeats = self.num_heads // self.num_kv_heads
            if repeats > 1:
                k = jnp.repeat(k, repeats, axis=1)
                v = jnp.repeat(v, repeats, axis=1)

            scale = jnp.sqrt(jnp.float32(head_dim))
            attn_weights = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / scale

            if mask is not None:
                attn_weights = jnp.where(mask, attn_weights, jnp.finfo(attn_weights.dtype).min)

            attn_weights = nn.softmax(attn_weights, axis=-1)

            out = jnp.matmul(attn_weights, v)
            out = out.transpose(0, 2, 1, 3).reshape(B, -1, attn_dim)

        out = out * nn.sigmoid(
            nn.Dense(attn_dim, dtype=self.dtype, use_bias=False,
                     kernel_init=default_init(), name="gate_proj")(x))
        out = _aq(out, quant)
        return nn.Dense(self.d_model, dtype=self.dtype, use_bias=False, kernel_init=residual_init(self.num_layers), name="out_proj")(out)


def _walsh_matrix(n):
    H = np.array([[1.0]], dtype=np.float32)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return jnp.asarray(H / np.sqrt(n))


class HadamardMLP(nn.Module):
    d_model: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        n = 1 << (self.d_model - 1).bit_length()
        H = _walsh_matrix(n).astype(self.dtype)
        d1 = self.param("d1", jinit.ones, (n,)).astype(self.dtype)
        d2 = self.param("d2", jinit.ones, (n,)).astype(self.dtype)
        d3 = self.param("d3", jinit.constant(0.02), (n,)).astype(self.dtype)
        pad = n - self.d_model
        z = jnp.pad(x, ((0, 0), (0, 0), (0, pad))) if pad else x
        z = (d1 * z) @ H
        z = nn.silu(d2 * z) @ H
        return (d3 * z)[..., : self.d_model]


class Block(nn.Module):
    num_heads: int
    num_kv_heads: int
    d_model: int
    num_layers: int
    dtype: jnp.dtype = jnp.bfloat16
    flash: bool = True
    attn_dim: int = 0

    def _gate(self, name):
        return nn.sigmoid(self.param(name, jinit.zeros, ())).astype(self.dtype)

    @nn.compact
    def __call__(self, x, mask=None, rope=None, quant=False, engram_kv=None, site_flags=None):
        if engram_kv is not None:
            ek, ev = engram_kv
            alpha = nn.sigmoid(jnp.einsum("btd,sbtd->sbt", _rms_unit(x), _rms_unit(ek))
                               / math.sqrt(self.d_model))
            x = x + jnp.einsum("s,sbt,sbtd->btd", site_flags.astype(jnp.float32),
                               alpha, ev.astype(jnp.float32)).astype(x.dtype)

        skip = x
        x = ZCRMSNorm(dtype=self.dtype)(x)
        x = MultiHeadAttention(self.num_heads, self.num_kv_heads, self.d_model, self.num_layers,
                               self.dtype, self.flash, attn_dim=self.attn_dim,
                               name="self_attn")(x, mask=mask, rope=rope, quant=quant)
        x = ZCRMSNorm(dtype=self.dtype, name="post_attn_norm")(x)
        x = skip + self._gate("attn_gate") * x

        skip = x
        x = ZCRMSNorm(dtype=self.dtype, name="pre_hada_norm")(x)
        x = HadamardMLP(self.d_model, self.dtype, name="hadamard_mlp")(x)
        return skip + x


class _ScanBody(nn.Module):
    num_heads: int
    num_kv_heads: int
    d_model: int
    num_layers: int
    dtype: jnp.dtype = jnp.bfloat16
    flash: bool = True
    collect_hidden: bool = False
    attn_dim: int = 0

    @nn.compact
    def __call__(self, x, xs, mask, rope, quant, engram_kv):
        site_flags, hc = xs
        block = Block(
            self.num_heads, self.num_kv_heads, self.d_model, self.num_layers,
            self.dtype, self.flash, attn_dim=self.attn_dim,
            name="block",
        )
        B, T, n, C = x.shape
        xf = x.astype(jnp.float32)
        nx = _rms_unit(x.reshape(B, T, n * C))
        hpre = nn.sigmoid(hc["a_pre"] * (nx @ hc["phi_pre"].astype(jnp.float32))
                          + hc["b_pre"] + hc["pre_off"])
        u = jnp.einsum("btn,btnc->btc", hpre, xf).astype(self.dtype)
        y = block(u, mask=mask, rope=rope, quant=quant,
                  engram_kv=engram_kv, site_flags=site_flags) - u
        hpost = 2 * nn.sigmoid(hc["a_post"] * (nx @ hc["phi_post"].astype(jnp.float32))
                               + hc["b_post"] + hc["post_off"])
        res = nx @ hc["phi_res"].astype(jnp.float32)
        hres = _sinkhorn(hc["a_res"] * res.reshape(B, T, n, n) + hc["b_res"])
        new_x = (jnp.einsum("btij,btjc->btic", hres, xf)
                 + hpost[..., None] * y.astype(jnp.float32)[:, :, None, :]).astype(self.dtype)
        out = jnp.mean(new_x, axis=2) if self.collect_hidden else None
        return new_x, out


class Stack(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, x, mask=None, rope=None, engram_kv=None, collect_hidden=False,
                 quant=False):
        cfg = self.config
        dt = cfg.jax_dtype
        x = x.astype(dt)

        site_flags = None
        if engram_kv is not None:
            flags = np.zeros((cfg.num_layers, len(cfg.engram_layers)), np.float32)
            for s, layer in enumerate(cfg.engram_layers):
                flags[layer, s] = 1.0
            site_flags = jnp.asarray(flags)

        n, L, nC = cfg.mhc_lanes, cfg.num_layers, cfg.mhc_lanes * cfg.d_model
        lane = np.eye(n, dtype=np.float32)[np.arange(L) % n]
        hc = {
            "phi_pre": self.param("mhc_phi_pre", default_init(), (L, nC, n)),
            "phi_post": self.param("mhc_phi_post", default_init(), (L, nC, n)),
            "phi_res": self.param("mhc_phi_res", default_init(), (L, nC, n * n)),
            "b_pre": self.param("mhc_b_pre", jinit.zeros, (L, n)),
            "b_post": self.param("mhc_b_post", jinit.zeros, (L, n)),
            "b_res": self.param("mhc_b_res", _res_identity_init, (L, n, n)),
            "a_pre": self.param("mhc_a_pre", jinit.constant(0.01), (L,)),
            "a_post": self.param("mhc_a_post", jinit.constant(0.01), (L,)),
            "a_res": self.param("mhc_a_res", jinit.constant(0.01), (L,)),
            "pre_off": jnp.asarray(8 * lane - 4),
            "post_off": jnp.asarray(-4 * (1 - lane)),
        }
        x = jnp.broadcast_to(x[:, :, None, :], (*x.shape[:2], n, x.shape[-1]))

        ScanBlock = nn.scan(
            nn.remat(_ScanBody),
            variable_axes={"params": 0},
            split_rngs={"params": True},
            length=cfg.num_layers,
            in_axes=(0, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast),
        )
        x, hidden = ScanBlock(
            cfg.num_heads, cfg.num_kv_heads, cfg.d_model, cfg.num_layers, dt,
            cfg.flash, collect_hidden, attn_dim=cfg.attn_dim,
            name="layers",
        )(x, (site_flags, hc), mask, rope, quant, engram_kv)

        x = jnp.mean(x, axis=2)
        x = ZCRMSNorm(dtype=dt, name="final_norm")(x)
        return x, hidden


def probe_pool(cells, probes, keep=None, dtype=jnp.bfloat16):
    b, t, l, d = cells.shape
    cells = cells.reshape(b, t * l, d)
    if keep is not None:
        keep = jnp.repeat(keep, l, axis=1)
    scores = jnp.einsum("bcd,kd->bkc", cells.astype(jnp.float32),
                        probes.astype(jnp.float32)) / math.sqrt(d)
    if keep is not None:
        scores = jnp.where(keep[:, None, :] > 0, scores, -jnp.inf)
    w = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("bkc,bcd->bkd", w, cells.astype(jnp.float32)
                      ).reshape(b, -1).astype(dtype)


class ContrastiveHead(nn.Module):
    d_model: int
    out_dim: int
    dtype: jnp.dtype = jnp.bfloat16
    temp_init: float = 0.07

    PROBES = 4

    @nn.compact
    def __call__(self, cells, keep=None):
        pooled = probe_pool(cells, self.param("probes", default_init(),
                                              (self.PROBES, cells.shape[-1])),
                            keep, self.dtype)
        p = nn.Dense(self.out_dim, dtype=self.dtype, use_bias=False,
                     kernel_init=default_init(), name="proj")(pooled)
        log_temp = self.param("log_temp", jinit.constant(math.log(self.temp_init)), ())
        denom = jnp.sqrt(jnp.sum(p.astype(jnp.float32) ** 2, axis=-1, keepdims=True) + 1e-12)
        return (p / denom.astype(p.dtype)), log_temp


class ConfidenceHead(nn.Module):
    dtype: jnp.dtype = jnp.bfloat16

    PROBES = 8

    @nn.compact
    def __call__(self, cells, keep=None):
        pooled = probe_pool(cells, self.param("probes", default_init(),
                                              (self.PROBES, cells.shape[-1])),
                            keep, self.dtype)
        logit = nn.Dense(1, dtype=self.dtype, use_bias=True,
                         kernel_init=default_init(), name="proj")(pooled)
        return logit[..., 0].astype(jnp.float32)


class SimpleAttentionNetwork(nn.Module):
    config: TransformerConfig

    def setup(self):
        cfg = self.config
        self.embedding = nn.Embed(cfg.vocab_size, cfg.d_model, embedding_init=jinit.normal(stddev=0.02))
        self.embed_scale = math.sqrt(cfg.d_model)
        self.stack = Stack(cfg)
        self.contrastive_head = ContrastiveHead(
            cfg.d_model, cfg.contrastive_dim, cfg.jax_dtype)
        self.confidence_head = ConfidenceHead(cfg.jax_dtype)
        assert all(l < cfg.num_layers for l in cfg.engram_layers)
        orders, heads, sub_dim = engram_geometry(cfg)
        self.engrams = [
            Engram(cfg.d_model, len(orders) * heads, cfg.engram_slots,
                   sub_dim, cfg.num_layers, max(orders), cfg.jax_dtype)
            for _ in cfg.engram_layers
        ]
        self.mtp_combine = nn.Dense(cfg.d_model, dtype=cfg.jax_dtype, use_bias=False,
                                    kernel_init=default_init())
        self.mtp_block = Block(cfg.num_heads, cfg.num_kv_heads, cfg.d_model,
                               cfg.num_layers, cfg.jax_dtype, cfg.flash,
                               attn_dim=cfg.attn_dim)
        self.mtp_emb_norm = ZCRMSNorm(dtype=cfg.jax_dtype)
        self.mtp_final_norm = ZCRMSNorm(dtype=cfg.jax_dtype)

    def _rope(self, seq_len):
        cfg = self.config
        head_dim = (cfg.attn_dim or cfg.d_model) // cfg.num_heads
        return precompute_rope_freqs(head_dim, seq_len, cfg.rope_theta)

    def _engram_kv(self, tokens, mask, quant):
        if not self.engrams:
            return None
        orders, heads, _ = engram_geometry(self.config)
        indices = engram_indices(tokens, orders, heads, self.config.engram_slots)
        ngram_ok = jnp.stack([_mask_diag(mask, o - 1) for o in orders for _ in range(heads)],
                             axis=-1)
        tap_ok = jnp.stack([_mask_diag(mask, j * max(orders)) for j in range(ENGRAM_CONV_TAPS)])
        pairs = [e(indices, ngram_ok, tap_ok, quant=quant) for e in self.engrams]
        return jnp.stack([k for k, _ in pairs]), jnp.stack([v for _, v in pairs])

    def __call__(self, tokens, mask=None, quant=False, return_mtp=False):
        if mask is None:
            mask = make_causal_mask(tokens.shape[1])
        x = self.embedding(tokens) * self.embed_scale
        rope = self._rope(tokens.shape[1])
        engram_kv = self._engram_kv(tokens, mask, quant)
        x, _ = self.stack(x, mask=mask, rope=rope, engram_kv=engram_kv, quant=quant)
        logits = _aq(x, quant).astype(jnp.float32) @ self.embedding.embedding.T
        if not (return_mtp or self.is_initializing()):
            return logits

        nxt = jnp.pad(tokens[:, 1:], ((0, 0), (0, 1)))
        e2 = self.mtp_emb_norm(self.embedding(nxt) * self.embed_scale)
        m = self.mtp_combine(_aq(jnp.concatenate([x, e2], axis=-1), quant))
        m = self.mtp_block(m, mask=mask, rope=rope, quant=quant)
        m = self.mtp_final_norm(m)
        mtp_logits = _aq(m, quant).astype(jnp.float32) @ self.embedding.embedding.T
        if not return_mtp:
            return logits
        return logits, mtp_logits

    def hidden_cells(self, tokens, quant=False, window=0, sink=None):
        cfg = self.config
        mask = (make_causal_mask(tokens.shape[1])
                & make_padding_mask(tokens, cfg.pad_token_id))
        if window:
            pos = jnp.arange(tokens.shape[1])
            recent = ((pos[:, None] - pos[None, :]) < window)[None, None, :, :]
            keep = recent if sink is None else (recent | sink[:, None, None, :])
            mask = mask & keep
        x0 = self.embedding(tokens) * self.embed_scale
        rope = self._rope(tokens.shape[1])
        engram_kv = self._engram_kv(tokens, mask, quant)
        _, hidden = self.stack(x0, mask=mask, rope=rope, engram_kv=engram_kv,
                               quant=quant, collect_hidden=True)
        return jnp.stack([x0, *hidden], axis=2)

    def _encode_contrastive(self, tokens, quant=False, window=0, sink=None):
        cells = jax.lax.stop_gradient(
            self.hidden_cells(tokens, quant=quant, window=window, sink=sink))
        keep = (tokens != self.config.pad_token_id).astype(jnp.float32)
        return self.contrastive_head(cells, keep=keep)

    def encode_contrastive(self, tokens, quant=False, window=0, sink=None):
        return self._encode_contrastive(tokens, quant=quant, window=window,
                                        sink=sink)[0]

    def forward_contrastive(self, query_tokens, tool_tokens, quant=False):
        q_emb, log_temp = self._encode_contrastive(query_tokens, quant=quant)
        t_emb, _ = self._encode_contrastive(tool_tokens, quant=quant)
        return q_emb, t_emb, log_temp

    def forward_confidence(self, tokens, quant=False, window=0, sink=None):
        cells = jax.lax.stop_gradient(
            self.hidden_cells(tokens, quant=quant, window=window, sink=sink))
        keep = (tokens != self.config.pad_token_id).astype(jnp.float32)
        return self.confidence_head(cells, keep=keep)

    def hidden_states(self, tokens, mask=None):
        if mask is None:
            mask = make_causal_mask(tokens.shape[1])
        x = self.embedding(tokens) * self.embed_scale
        rope = self._rope(tokens.shape[1])
        engram_kv = self._engram_kv(tokens, mask, False)
        _, hidden = self.stack(x, mask=mask, rope=rope, engram_kv=engram_kv, collect_hidden=True)
        return hidden.astype(jnp.float32)


def make_causal_mask(seq_len):
    mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
    return mask[None, None, :, :]


def make_padding_mask(tokens, pad_token_id):
    mask = tokens != pad_token_id
    return mask[:, None, None, :]


KV_BUDGET_BYTES = 11 * 1024 * 1024 + 512 * 1024
KV_GROUP = 32
KV_WINDOW_MIN = 160


def kv_budget_window(config):
    head_dim = (getattr(config, "attn_dim", 0) or config.d_model) // config.num_heads
    kv = config.num_kv_heads * head_dim
    d, L = config.d_model, config.num_layers
    sites = len(tuple(getattr(config, "engram_layers", (2, 15))))
    per_pos = (L * (2 * kv + 2 * (kv // KV_GROUP) * 4)
               + sites * (d + (d // KV_GROUP) * 4))
    window = (KV_BUDGET_BYTES // per_pos) // KV_GROUP * KV_GROUP
    return max(KV_WINDOW_MIN, min(window, config.max_seq_len))


def effective_kv_window(config):
    budget = kv_budget_window(config)
    return min(budget, config.kv_window) if config.kv_window else budget


def make_causal_packing_mask(seg_ids, prefix=None, window=0):
    T = seg_ids.shape[1]
    causal = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
    block = (seg_ids[:, :, None] == seg_ids[:, None, :]) & (seg_ids[:, :, None] > 0)
    mask = block & causal[None, :, :]
    if window:
        pos = jnp.arange(T)
        recent = (pos[:, None] - pos[None, :]) < window
        sink = (jnp.zeros_like(seg_ids, dtype=jnp.bool_) if prefix is None
                else prefix > 0)
        mask = mask & (recent[None, :, :] | sink[:, None, :])
    return mask[:, None, :, :]
