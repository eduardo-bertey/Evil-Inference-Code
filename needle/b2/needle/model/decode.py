import math
from collections import namedtuple

import jax
import jax.numpy as jnp

from .architecture import (precompute_rope_freqs, apply_rope, _walsh_matrix,
                           engram_geometry, engram_indices, _rms_unit, _shift_right,
                           _sinkhorn, ENGRAM_CONV_TAPS)


def _doc_prefix_len(ids):
    return 0

DecodeCfg = namedtuple("DecodeCfg", [
    "d_model", "num_heads", "num_kv_heads", "num_layers", "attn_dim", "mhc_lanes",
    "engram_layers", "engram_orders", "engram_heads", "engram_slots",
    "kv_window",
])


def _attn_width(config):
    return getattr(config, "attn_dim", 0) or config.d_model


def decode_cfg(config, kv_window=0):
    orders, heads, _ = engram_geometry(config)
    return DecodeCfg(config.d_model, config.num_heads, config.num_kv_heads,
                     config.num_layers, _attn_width(config), config.mhc_lanes,
                     tuple(config.engram_layers), tuple(orders), heads,
                     config.engram_slots, int(kv_window))


def init_kv_cache(config, batch, max_len):
    head_dim = _attn_width(config) // config.num_heads
    shape = (config.num_layers, batch, config.num_kv_heads, max_len, head_dim)
    z = jnp.zeros(shape, jnp.float32)
    return z, z


def _zcrms(x, scale, eps=1e-6):
    rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (1.0 + scale) * x / rms


def _layer(params, i):
    b = params["stack"]["layers"]["block"]
    sa = b["self_attn"]
    ha = b["hadamard_mlp"]
    lp = {
        "pre": b["ZCRMSNorm_0"]["scale"][i],
        "q": sa["q_proj"]["kernel"][i], "k": sa["k_proj"]["kernel"][i],
        "v": sa["v_proj"]["kernel"][i], "o": sa["out_proj"]["kernel"][i],
        "attn_gate_proj": sa["gate_proj"]["kernel"][i],
        "qn": sa["q_norm"]["scale"][i], "kn": sa["k_norm"]["scale"][i],
        "post": b["post_attn_norm"]["scale"][i], "attn_gate": b["attn_gate"][i],
        "pre_hada": b["pre_hada_norm"]["scale"][i],
        "hd1": ha["d1"][i], "hd2": ha["d2"][i], "hd3": ha["d3"][i],
    }
    return lp


def _attn_cached(x, lp, k_cache_l, v_cache_l, pos, cos_s, sin_s, cfg, key_valid,
                 flash=False, sink=None):
    B, S, _ = x.shape
    H, KV = cfg.num_heads, cfg.num_kv_heads
    hd = cfg.attn_dim // H

    q = (x @ lp["q"]).reshape(B, S, H, hd).transpose(0, 2, 1, 3)
    k = (x @ lp["k"]).reshape(B, S, KV, hd).transpose(0, 2, 1, 3)
    v = (x @ lp["v"]).reshape(B, S, KV, hd).transpose(0, 2, 1, 3)
    q = _zcrms(q, lp["qn"]); k = _zcrms(k, lp["kn"])
    q = apply_rope(q, cos_s, sin_s); k = apply_rope(k, cos_s, sin_s)

    k_cache_l = jax.lax.dynamic_update_slice(k_cache_l, k, (0, 0, pos, 0))
    v_cache_l = jax.lax.dynamic_update_slice(v_cache_l, v, (0, 0, pos, 0))
    max_len = k_cache_l.shape[2]

    if flash and S > 1:
        causal = jnp.tril(jnp.ones((S, S), bool))
        if cfg.kv_window:
            rows = jnp.arange(S)
            causal = causal & ((rows[:, None] - rows[None, :]) < cfg.kv_window)
            if sink is not None:
                causal = causal | (jnp.tril(jnp.ones((S, S), bool))
                                   & sink[:, :S][:, None, :])
        if key_valid is None:
            m = causal[None, None, :, :] if causal.ndim == 2 else causal[:, None, :, :]
        else:
            c = causal[None] if causal.ndim == 2 else causal
            m = (c & key_valid[:, :S][:, None, :]) | jnp.eye(S, dtype=bool)[None]
            m = m[:, None, :, :]
        impl = ("cudnn" if jax.default_backend() == "gpu"
                and q.dtype in (jnp.bfloat16, jnp.float16) else None)
        out = jax.nn.dot_product_attention(
            q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3),
            mask=m, implementation=impl)
        out = out.reshape(B, S, cfg.attn_dim)
    else:
        reps = H // KV
        scale = math.sqrt(hd)
        qg = q.reshape(B, KV, reps, S, hd)
        aw = jnp.einsum("bgrsd,bgld->bgrsl", qg, k_cache_l) / scale
        qpos = pos + jnp.arange(S)
        kpos = jnp.arange(max_len)
        causal = kpos[None, :] <= qpos[:, None]
        if cfg.kv_window:
            recent = (qpos[:, None] - kpos[None, :]) < cfg.kv_window
            keep = recent[None] if sink is None else (recent[None] | sink[:, None, :])
            causal = causal[None] & keep
        else:
            causal = causal[None]
        if key_valid is None:
            mask = causal[:, None, None, :, :]
        else:
            mask = (causal & key_valid[:, None, :])[:, None, None, :, :]
        aw = jnp.where(mask, aw, jnp.finfo(aw.dtype).min)
        aw = jax.nn.softmax(aw, axis=-1)
        out = jnp.einsum("bgrsl,bgld->bgrsd", aw, v_cache_l)
        out = out.reshape(B, H, S, hd).transpose(0, 2, 1, 3).reshape(B, S, cfg.attn_dim)

    out = out * jax.nn.sigmoid(x @ lp["attn_gate_proj"])
    return out @ lp["o"], k_cache_l, v_cache_l


def _hadamard(z, d1, d2, d3, d_model):
    n = 1 << (d_model - 1).bit_length()
    H = _walsh_matrix(n).astype(z.dtype)
    pad = n - d_model
    if pad:
        z = jnp.pad(z, ((0, 0), (0, 0), (0, pad)))
    z = (d1 * z) @ H
    z = jax.nn.silu(d2 * z) @ H
    return (d3 * z)[..., :d_model]


def _block_cached(x, lp, k_cache_l, v_cache_l, pos, cos_s, sin_s, cfg, key_valid,
                  flash=False, sink=None):
    skip = x
    h = _zcrms(x, lp["pre"])
    attn, k_cache_l, v_cache_l = _attn_cached(h, lp, k_cache_l, v_cache_l, pos,
                                              cos_s, sin_s, cfg, key_valid, flash, sink)
    attn = _zcrms(attn, lp["post"])
    x = skip + jax.nn.sigmoid(lp["attn_gate"]) * attn

    skip = x
    h = _zcrms(x, lp["pre_hada"])
    x = skip + _hadamard(h, lp["hd1"], lp["hd2"], lp["hd3"], cfg.d_model)
    return x, k_cache_l, v_cache_l


def _engram_window(cfg):
    return ENGRAM_CONV_TAPS * max(cfg.engram_orders) if cfg.engram_layers else 0


def _engram_kv(params, cfg, hist, hist_valid, pos, S):
    W = _engram_window(cfg)
    B = hist.shape[0]
    padded = jnp.pad(hist, ((0, 0), (W, 0)))
    win = jax.lax.dynamic_slice(padded, (0, pos), (B, W + S))
    pv = jnp.pad(hist_valid, ((0, 0), (W, 0)))
    win_valid = jax.lax.dynamic_slice(pv, (0, pos), (B, W + S)).astype(jnp.float32)

    orders, heads = cfg.engram_orders, cfg.engram_heads
    idx = engram_indices(win, orders, heads, cfg.engram_slots)
    ngram_ok = jnp.stack([_shift_right(win_valid, o - 1)
                          for o in orders for _ in range(heads)], axis=-1)
    tap_ok = jnp.stack([_shift_right(win_valid, j * max(orders))
                        for j in range(ENGRAM_CONV_TAPS)])

    ks, vs = [], []
    for site in range(len(cfg.engram_layers)):
        ep = params[f"engrams_{site}"]
        tables = ep["embedding"]
        fetched = tables[jnp.arange(tables.shape[0]), idx] * ngram_ok[..., None]
        e = fetched.reshape(B, W + S, -1)
        k = e @ ep["key_proj"]["kernel"]
        v = e @ ep["value_proj"]["kernel"]
        taps = ep["taps"]
        v = sum(taps[j] * _shift_right(v, j * max(orders)) * tap_ok[j][..., None]
                for j in range(ENGRAM_CONV_TAPS))
        ks.append(k[:, W:]); vs.append(v[:, W:])
    return jnp.stack(ks), jnp.stack(vs)


def _mhc(params, cfg):
    st = params["stack"]
    n, L = cfg.mhc_lanes, cfg.num_layers
    lane = jnp.eye(n, dtype=jnp.float32)[jnp.arange(L) % n]
    return {"phi_pre": st["mhc_phi_pre"], "phi_post": st["mhc_phi_post"],
            "phi_res": st["mhc_phi_res"], "b_pre": st["mhc_b_pre"],
            "b_post": st["mhc_b_post"], "b_res": st["mhc_b_res"],
            "a_pre": st["mhc_a_pre"], "a_post": st["mhc_a_post"],
            "a_res": st["mhc_a_res"], "pre_off": 8 * lane - 4,
            "post_off": -4 * (1 - lane)}


def _forward_cached(params, cfg, tokens, k_cache, v_cache, pos, cos, sin, key_valid=None,
                    flash=False, hist=None, hist_valid=None, sink=None):
    emb = params["embedding"]["embedding"].astype(jnp.float32)
    x = emb[tokens] * math.sqrt(cfg.d_model)
    B, S = tokens.shape
    cos_s = jax.lax.dynamic_slice_in_dim(cos, pos, S, axis=0)
    sin_s = jax.lax.dynamic_slice_in_dim(sin, pos, S, axis=0)

    ekv = None
    if cfg.engram_layers:
        h = tokens if hist is None else hist
        hv = jnp.ones(h.shape, bool) if hist_valid is None else hist_valid
        ekv = _engram_kv(params, cfg, h, hv,
                         jnp.asarray(0, jnp.int32) if hist is None else pos, S)

    n, C = cfg.mhc_lanes, cfg.d_model
    hc = _mhc(params, cfg)
    x = jnp.broadcast_to(x[:, :, None, :], (B, S, n, C))

    new_k, new_v = [], []
    for i in range(cfg.num_layers):
        nx = _rms_unit(x.reshape(B, S, n * C))
        hpre = jax.nn.sigmoid(hc["a_pre"][i] * (nx @ hc["phi_pre"][i])
                              + hc["b_pre"][i] + hc["pre_off"][i])
        u = jnp.einsum("btn,btnc->btc", hpre, x)
        bx = u
        if ekv is not None and i in cfg.engram_layers:
            site = cfg.engram_layers.index(i)
            ek, ev = ekv
            alpha = jax.nn.sigmoid(
                jnp.einsum("btd,btd->bt", _rms_unit(u), _rms_unit(ek[site]))
                / math.sqrt(C))
            bx = u + alpha[..., None] * ev[site]
        y, kc, vc = _block_cached(bx, _layer(params, i), k_cache[i], v_cache[i],
                                  pos, cos_s, sin_s, cfg, key_valid, flash, sink)
        y = y - u
        hpost = 2 * jax.nn.sigmoid(hc["a_post"][i] * (nx @ hc["phi_post"][i])
                                   + hc["b_post"][i] + hc["post_off"][i])
        res = nx @ hc["phi_res"][i]
        hres = _sinkhorn(hc["a_res"][i] * res.reshape(B, S, n, n) + hc["b_res"][i])
        x = (jnp.einsum("btij,btjc->btic", hres, x)
             + hpost[..., None] * y[:, :, None, :])
        new_k.append(kc); new_v.append(vc)
    x = jnp.mean(x, axis=2)
    x = _zcrms(x, params["stack"]["final_norm"]["scale"])
    logits = x @ emb.T
    return logits, jnp.stack(new_k), jnp.stack(new_v)


forward_cached = jax.jit(_forward_cached, static_argnums=(1, 9))


def _f32(params):
    return jax.tree.map(lambda a: jnp.asarray(a, jnp.float32), params)


def generate_cached(config, params, tokenizer, prompt, max_new_tokens=96, max_len=None,
                    kv_window=0):
    from .tokenizer import BOS_ID, EOS_ID
    params = _f32(params)
    prompt_ids = ([BOS_ID] + tokenizer.encode(prompt))[:config.max_seq_len - 1]
    max_len = max_len or min(config.max_seq_len, len(prompt_ids) + max_new_tokens)
    head_dim = _attn_width(config) // config.num_heads
    cos, sin = precompute_rope_freqs(head_dim, max_len, config.rope_theta)

    dcfg = decode_cfg(config, kv_window=kv_window)
    kc, vc = init_kv_cache(config, 1, max_len)
    hist = jnp.zeros((1, max_len), jnp.int32).at[0, :len(prompt_ids)].set(
        jnp.asarray(prompt_ids, jnp.int32))
    valid = jnp.ones((1, max_len), bool)
    sink = None
    if kv_window:
        import numpy as _np
        sk = _np.zeros((1, max_len), bool)
        sk[0, :_doc_prefix_len(prompt_ids)] = True
        sink = jnp.asarray(sk)
    toks = jnp.asarray([prompt_ids], jnp.int32)
    logits, kc, vc = forward_cached(params, dcfg, toks, kc, vc,
                                    jnp.asarray(0, jnp.int32), cos, sin, None, False,
                                    hist, valid, sink)
    nxt = int(jnp.argmax(logits[0, -1]))

    out, pos = [], len(prompt_ids)
    for _ in range(max_new_tokens):
        if nxt == EOS_ID or pos >= max_len:
            break
        out.append(nxt)
        hist = hist.at[0, pos].set(nxt)
        logits, kc, vc = forward_cached(params, dcfg, jnp.asarray([[nxt]], jnp.int32),
                                        kc, vc, jnp.asarray(pos, jnp.int32), cos, sin,
                                        None, False, hist, valid, sink)
        nxt = int(jnp.argmax(logits[0, -1]))
        pos += 1
    return tokenizer.decode(out)


def _shard(x, nd):
    return x.reshape(nd, x.shape[0] // nd, *x.shape[1:])


def _sample(last, rng, temperature):
    logp = jax.nn.log_softmax(last, axis=-1)
    nxt = jnp.where(temperature > 0,
                    jax.random.categorical(rng, last / jnp.maximum(temperature, 1e-6), axis=-1),
                    jnp.argmax(last, axis=-1)).astype(jnp.int32)
    tok_logp = jnp.take_along_axis(logp, nxt[:, None], axis=-1)[:, 0]
    return nxt, tok_logp


def _make_rollout(cfg, max_len, scan_len):
    hd = cfg.attn_dim // cfg.num_heads

    def rollout(params, buf, valid, sink, cos, sin, rng, temperature):
        B, max_p = buf.shape
        z = jnp.zeros((cfg.num_layers, B, cfg.num_kv_heads, max_len, hd), jnp.float32)
        hist = jnp.zeros((B, max_len), jnp.int32).at[:, :max_p].set(buf)
        logits, kc, vc = _forward_cached(params, cfg, buf, z, z,
                                         jnp.asarray(0, jnp.int32), cos, sin, valid,
                                         flash=True, hist=hist, hist_valid=valid,
                                         sink=sink)
        rng, sk = jax.random.split(rng)
        t0, lp0 = _sample(logits[:, -1], sk, temperature)

        def body(carry, _):
            kc, vc, last, pos, rng, hist = carry
            hist = jax.lax.dynamic_update_slice(hist, last[:, None], (0, pos))
            logits, kc, vc = _forward_cached(params, cfg, last[:, None], kc, vc,
                                             pos, cos, sin, valid,
                                             hist=hist, hist_valid=valid, sink=sink)
            rng, sk = jax.random.split(rng)
            tok, lp = _sample(logits[:, -1], sk, temperature)
            return (kc, vc, tok, pos + 1, rng, hist), (tok, lp)

        init = (kc, vc, t0, jnp.asarray(max_p, jnp.int32), rng, hist)
        _, (toks, lps) = jax.lax.scan(body, init, None, length=scan_len)
        toks = jnp.concatenate([t0[None], toks], axis=0)
        lps = jnp.concatenate([lp0[None], lps], axis=0)
        return toks.T, lps.T
    return rollout


_rollout_cache = {}


def _rollout_fn(cfg, max_len, scan_len):
    key = (cfg, max_len, scan_len)
    if key not in _rollout_cache:
        _rollout_cache[key] = jax.pmap(_make_rollout(cfg, max_len, scan_len),
                                       in_axes=(0, 0, 0, 0, None, None, 0, None))
    return _rollout_cache[key]


def batched_generate(config, params, tokenizer, prompts, max_new_tokens=96,
                     temperature=0.0, seed=0, pad_to=None, kv_window=0):
    from .tokenizer import BOS_ID, EOS_ID, PAD_ID
    import numpy as _np

    params = _f32(params)
    dcfg = decode_cfg(config, kv_window=kv_window)
    cap = max(1, config.max_seq_len - max_new_tokens)
    enc = [([BOS_ID] + tokenizer.encode(p))[:cap] for p in prompts]
    n_real = len(enc)

    nd = jax.local_device_count()
    B = ((n_real + nd - 1) // nd) * nd
    while len(enc) < B:
        enc.append([BOS_ID])
    max_p = min(cap, pad_to) if pad_to else max(len(e) for e in enc)
    max_len = min(config.max_seq_len, max_p + max_new_tokens)
    scan_len = max(max_len - max_p - 1, 0)
    if scan_len == 0:
        raise ValueError(
            f"longest prompt is {max_p} tokens, leaving no room for "
            f"{max_new_tokens} new ones inside max_seq_len {config.max_seq_len} - "
            f"the whole batch would generate a single token. Drop prompts longer "
            f"than max_seq_len - max_new_tokens before calling.")

    buf = _np.full((B, max_p), PAD_ID, _np.int32)
    valid = _np.zeros((B, max_len), bool)
    sink = _np.zeros((B, max_len), bool)
    for b, e in enumerate(enc):
        start = max_p - len(e)
        buf[b, start:] = e
        valid[b, start:] = True
        if kv_window:
            sink[b, start:start + _doc_prefix_len(e)] = True
    valid[:, max_p:] = True

    head_dim = _attn_width(config) // config.num_heads
    cos, sin = precompute_rope_freqs(head_dim, max_len, config.rope_theta)

    pr = jax.tree.map(lambda x: jnp.broadcast_to(x, (nd,) + x.shape), params)
    rollout = _rollout_fn(dcfg, max_len, scan_len)
    rngs = jax.random.split(jax.random.PRNGKey(seed), nd)
    toks, lps = rollout(pr, _shard(jnp.asarray(buf), nd), _shard(jnp.asarray(valid), nd),
                        _shard(jnp.asarray(sink), nd), cos, sin, rngs,
                        jnp.asarray(temperature, jnp.float32))
    toks = _np.asarray(toks).reshape(B, -1)
    lps = _np.asarray(lps).reshape(B, -1)

    keep = _np.cumprod(toks != EOS_ID, axis=1).astype(bool)
    texts, out_ids, out_lps = [], [], []
    for b in range(n_real):
        ids = toks[b][keep[b]].tolist()
        out_ids.append(ids)
        out_lps.append(lps[b][keep[b]].tolist())
        texts.append(tokenizer.decode(ids))
    return texts, out_ids, out_lps
