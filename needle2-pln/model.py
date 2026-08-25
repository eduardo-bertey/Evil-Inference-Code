"""SimpleAttentionNetwork (SAN) — puerto PyTorch de needle 2 (cactus-compute/needle).

Arquitectura v2: Hadamard MLP en vez de FFN, GQA + RoPE full (theta 1e5),
memoria KV engram (hash n-gram), hyper-conexiones multi-lane con Sinkhorn,
ZCRMSNorm, sandwich-norm con gates, MTP (multi-token prediction),
ConfidenceHead/ContrastiveHead incluidos (se activan en finetune).
Inferencia con ventana deslizante KV (kv_window). API idéntica a
laurelia-llm/model.py para que pretrain.py sirva tal cual.
"""

import math, inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _ckpt
from rope import RoPE


class Config:
    drop = 0.0
    dim = 768                # d_model
    heads = 8                # num_heads (ratio v2)
    kv_groups = 4            # num_kv_heads (GQA)
    layers = 16              # num_layers
    block_size = 1024        # max_seq_len entrenamiento
    emb_num = 32000          # vocab_size BPE laurelia
    rotary_pct = 1.0         # v2 usa RoPE full
    rope_theta = 100000.0

    engram_orders = (2, 3)
    engram_slots = 8192
    engram_layers = (3, 12)  # sitios dentro de 16 capas (v2: (2,15)/27)
    mhc_lanes = 4            # hyper-conexiones
    kv_window = 256          # ventana deslizante al inferir

    batch_size: int = 8
    grad_acc: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    warm_up: int = 50


ENGRAM_SUB_DIM = 128
ENGRAM_CONV_TAPS = 4
_ENGRAM_SEED = 0x9E3779B9
_ENGRAM_PRIME = 0x01000193


def _trunc_normal(t, std):
    torch.nn.init.trunc_normal_(t, mean=0.0, std=std, a=-2 * std, b=2 * std)


def rms_unit(x, epsilon=1e-6):
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + epsilon)


def shift_right(x, offset):
    """jnp._shift_right: x[:, t] <- x[:, t-offset], ceros al frente (dim 1 = tiempo)."""
    if offset == 0:
        return x
    out = F.pad(x, [0, 0] * (x.dim() - 2) + [offset, 0] + [0, 0])
    return out[:, : x.shape[1]]


def mask_diag(mask, offset):
    """Diagonal -offset de máscara (B,1,T,T) -> (B,T), como jnp._mask_diag."""
    m = mask[:, 0]
    T = m.shape[-1]
    if offset >= T:
        return torch.zeros(m.shape[:-2] + (T,), dtype=m.dtype, device=m.device)
    d = m.diagonal(offset=-offset, dim1=-2, dim2=-1)
    if offset == 0:
        return d
    return F.pad(d, (offset, 0))


def sinkhorn(logits, iters=20):
    logK = logits.float()
    for _ in range(iters):
        logK = logK - logK.logsumexp(-1, keepdim=True)
        logK = logK - logK.logsumexp(-2, keepdim=True)
    return logK.exp()


def walsh_matrix(n, device=None):
    H = torch.ones(1, 1, device=device)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1),
                       torch.cat([H, -H], dim=1)], dim=0)
    return H / math.sqrt(n)


def engram_geometry(config):
    orders = tuple(config.engram_orders)
    heads = max(1, config.dim // (len(orders) * ENGRAM_SUB_DIM))
    sub_dim = config.dim // (len(orders) * heads)
    return orders, heads, sub_dim


def engram_indices(tokens, orders, heads, slots):
    """Hash FNV-style del v2 sobre n-grams -> índices (B,T,num_tables)."""
    u = tokens.long() & 0xFFFFFFFF
    idx = []
    for oi, order in enumerate(orders):
        for h in range(heads):
            seed = (_ENGRAM_SEED * (oi * heads + h + 1)) & 0xFFFFFFFF
            acc = torch.full_like(u, seed)
            for j in range(order):
                sh = shift_right(u, j)
                acc = ((acc ^ sh) * _ENGRAM_PRIME) & 0xFFFFFFFF
            acc = acc ^ (acc >> 15)
            idx.append(acc % slots)
    return torch.stack(idx, dim=-1)


class ZCRMSNorm(nn.Module):
    """RMSNorm con scale inicializado en cero: out = (1+scale)*x/rms."""

    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        return ((1 + self.scale) * x.float() / rms).to(x.dtype)


class Engram(nn.Module):
    """Tabla hash n-gram -> proyección key/value con conv de taps identidad."""

    def __init__(self, config):
        super().__init__()
        orders, heads, sub_dim = engram_geometry(config)
        num_tables = len(orders) * heads
        res_std = 0.02 / math.sqrt(2 * config.layers)
        self.num_tables, self.sub_dim, self.d_model = num_tables, sub_dim, config.dim
        self.slots = config.engram_slots
        self.dilation = max(orders)

        self.tables = nn.Parameter(torch.empty(num_tables, self.slots, sub_dim))
        _trunc_normal(self.tables, 0.02)
        self.key_proj = nn.Linear(num_tables * sub_dim, config.dim, bias=False)
        _trunc_normal(self.key_proj.weight, 0.02)
        self.value_proj = nn.Linear(num_tables * sub_dim, config.dim, bias=False)
        _trunc_normal(self.value_proj.weight, res_std)
        self.taps = nn.Parameter(torch.zeros(ENGRAM_CONV_TAPS, config.dim))
        with torch.no_grad():
            self.taps[0] = 1.0

    def forward(self, indices, ngram_ok, tap_ok):
        ar = torch.arange(self.num_tables, device=indices.device).view(1, 1, -1)
        fetched = self.tables[ar, indices]                          # (B,T,N,Dsub)
        e = fetched.float() * ngram_ok[..., None].float()
        e = e.reshape(*indices.shape[:2], self.num_tables * self.sub_dim)
        e = e.to(self.key_proj.weight.dtype)
        k = self.key_proj(e)
        v = self.value_proj(e)
        v = sum(self.taps[j] * shift_right(v, j * self.dilation) * tap_ok[j][..., None]
                for j in range(ENGRAM_CONV_TAPS))
        return k, v


class Attention(nn.Module):
    """GQA + qk ZCRMSNorm + RoPE full + gate sigmoide (v2 MultiHeadAttention)."""

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.heads
        self.num_kv_groups = config.kv_groups
        self.head_dim = config.dim // config.heads
        attn_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_groups * self.head_dim
        res_std = 0.02 / math.sqrt(2 * config.layers)

        self.q_proj = nn.Linear(config.dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, kv_dim, bias=False)
        self.gate_proj = nn.Linear(config.dim, attn_dim, bias=False)
        self.o_proj = nn.Linear(attn_dim, config.dim, bias=False)
        for w in (self.q_proj.weight, self.k_proj.weight, self.v_proj.weight,
                  self.gate_proj.weight):
            _trunc_normal(w, 0.02)
        _trunc_normal(self.o_proj.weight, res_std)

        self.q_norm = ZCRMSNorm(self.head_dim)
        self.k_norm = ZCRMSNorm(self.head_dim)
        self.rope = RoPE(self.head_dim, base=config.rope_theta,
                         rotary_pct=getattr(config, "rotary_pct", 1.0))

    def _split_qkv(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_groups, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_groups, self.head_dim)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = self.rope(q, k, 0)
        return q, k, v

    @staticmethod
    def _expand_kv(k, v, num_heads):
        if k.shape[1] == num_heads:
            return k, v
        rep = num_heads // k.shape[1]
        return k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)

    def forward(self, x, mask):
        q, k, v = self._split_qkv(x)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))       # (B,H,T,Dh)
        k, v = self._expand_kv(k, v, self.num_heads)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        B, _, T, _ = out.shape
        out = out.transpose(1, 2).reshape(B, T, -1)
        out = out * torch.sigmoid(self.gate_proj(x))
        return self.o_proj(out)

    def forward_with_cache(self, x, offset, cache, window=0):
        B, S_new, _ = x.shape
        q = self.q_proj(x).view(B, S_new, self.num_heads, self.head_dim)
        k_new = self.k_proj(x).view(B, S_new, self.num_kv_groups, self.head_dim)
        v_new = self.v_proj(x).view(B, S_new, self.num_kv_groups, self.head_dim)
        q, k_new = self.q_norm(q), self.k_norm(k_new)
        q, k_new = self.rope(q, k_new, offset)

        if cache is not None:
            k_full = torch.cat([cache[0], k_new], dim=1)
            v_full = torch.cat([cache[1], v_new], dim=1)
        else:
            k_full, v_full = k_new, v_new
        if window and k_full.shape[1] > window:
            k_full, v_full = k_full[:, -window:], v_full[:, -window:]
        new_cache = (k_full, v_full)

        q = q.transpose(1, 2)
        k, v = self._expand_kv(k_full.transpose(1, 2), v_full.transpose(1, 2),
                               self.num_heads)
        out = F.scaled_dot_product_attention(q, k, v,
                                             is_causal=(cache is None and S_new > 1))
        out = out.transpose(1, 2).reshape(B, S_new, -1)
        out = out * torch.sigmoid(self.gate_proj(x))
        return self.o_proj(out), new_cache


def hadamard_mlp(x, H, d1, d2, d3, d_model):
    n = H.shape[0]
    z = x
    if n != d_model:
        z = F.pad(z, (0, n - d_model))
    z = (d1 * z) @ H
    z = F.silu(d2 * z) @ H
    return (d3 * z)[..., :d_model]


class Block(nn.Module):
    """Bloque v2: sandwich attention con gate escalar + Hadamard MLP."""

    def __init__(self, config):
        super().__init__()
        self.d_model = config.dim
        self.attn = Attention(config)
        self.attn_gate = nn.Parameter(torch.zeros(()))
        self.norm1 = ZCRMSNorm(config.dim)
        self.post_attn_norm = ZCRMSNorm(config.dim)
        self.pre_hada_norm = ZCRMSNorm(config.dim)

        n = 1 << (config.dim - 1).bit_length()
        self.register_buffer("walsh", walsh_matrix(n), persistent=False)
        self.d1 = nn.Parameter(torch.ones(n))
        self.d2 = nn.Parameter(torch.ones(n))
        self.d3 = nn.Parameter(torch.full((n,), 0.02))

    def _engram_inject(self, x, engram_kv, site_flag):
        if engram_kv is None:
            return x
        ek, ev = engram_kv
        alpha = torch.sigmoid(
            torch.einsum("btd,sbtd->sbt", rms_unit(x), rms_unit(ek))
            / math.sqrt(self.d_model))
        return x + torch.einsum("s,sbt,sbtd->btd",
                                site_flag.flatten(), alpha, ev)

    def forward(self, x, mask, engram_kv=None, site_flag=None):
        x = self._engram_inject(x, engram_kv, site_flag)
        skip = x
        h = self.attn(self.norm1(x), mask)
        h = self.post_attn_norm(h)
        x = skip + torch.sigmoid(self.attn_gate) * h
        return x + hadamard_mlp(self.pre_hada_norm(x), self.walsh,
                                self.d1, self.d2, self.d3, self.d_model)

    def forward_with_cache(self, x, offset, cache, engram_kv=None, site_flag=None,
                           window=0):
        x = self._engram_inject(x, engram_kv, site_flag)
        skip = x
        h, new_cache = self.attn.forward_with_cache(self.norm1(x), offset, cache,
                                                    window=window)
        h = self.post_attn_norm(h)
        x = skip + torch.sigmoid(self.attn_gate) * h
        x = x + hadamard_mlp(self.pre_hada_norm(x), self.walsh,
                             self.d1, self.d2, self.d3, self.d_model)
        return x, new_cache


def probe_pool(cells, probes, keep=None):
    b, t, l, d = cells.shape
    cells = cells.reshape(b, t * l, d).float()
    if keep is not None:
        keep = keep.repeat_interleave(l, dim=1)
    scores = torch.einsum("bcd,kd->bkc", cells, probes.float()) / math.sqrt(d)
    if keep is not None:
        scores = torch.where(keep[:, None, :] > 0, scores,
                             torch.finfo(scores.dtype).min)
    w = torch.softmax(scores, dim=-1)
    return torch.einsum("bkc,bcd->bkd", w, cells).reshape(b, -1)


class ContrastiveHead(nn.Module):
    PROBES = 4

    def __init__(self, d_model, out_dim):
        super().__init__()
        self.probes = nn.Parameter(torch.empty(self.PROBES, d_model))
        _trunc_normal(self.probes, 0.02)
        self.proj = nn.Linear(d_model, out_dim, bias=False)
        _trunc_normal(self.proj.weight, 0.02)
        self.log_temp = nn.Parameter(torch.tensor(math.log(0.07)))

    def forward(self, cells, keep=None):
        pooled = probe_pool(cells, self.probes, keep)
        p = self.proj(pooled.to(self.probes.dtype)).float()
        denom = torch.sqrt(p.pow(2).sum(-1, keepdim=True) + 1e-12)
        return p / denom.to(p.dtype), self.log_temp


class ConfidenceHead(nn.Module):
    PROBES = 8

    def __init__(self, d_model):
        super().__init__()
        self.probes = nn.Parameter(torch.empty(self.PROBES, d_model))
        _trunc_normal(self.probes, 0.02)
        self.proj = nn.Linear(d_model, 1, bias=True)
        _trunc_normal(self.proj.weight, 0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, cells, keep=None):
        pooled = probe_pool(cells, self.probes, keep)
        return self.proj(pooled.to(self.probes.dtype))[..., 0].float()


class LLM(nn.Module):
    """SimpleAttentionNetwork completo. API laurelia: forward/forward_with_cache/generate."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        cfg = config
        L, n = cfg.layers, cfg.mhc_lanes
        assert all(l < cfg.layers for l in cfg.engram_layers)

        self.embeddings = nn.Embedding(cfg.emb_num, cfg.dim)
        _trunc_normal(self.embeddings.weight, 0.02)
        self.embed_scale = math.sqrt(cfg.dim)

        # Hyper-conexiones (params por capa, equivale al Stack escaneado del v2)
        nC = cfg.mhc_lanes * cfg.dim
        self.phi_pre = nn.Parameter(torch.empty(L, nC, n)); _trunc_normal(self.phi_pre, 0.02)
        self.phi_post = nn.Parameter(torch.empty(L, nC, n)); _trunc_normal(self.phi_post, 0.02)
        self.phi_res = nn.Parameter(torch.empty(L, nC, n * n)); _trunc_normal(self.phi_res, 0.02)
        self.b_pre = nn.Parameter(torch.zeros(L, n))
        self.b_post = nn.Parameter(torch.zeros(L, n))
        self.b_res = nn.Parameter(torch.stack([4.0 * torch.eye(n) for _ in range(L)]))
        self.a_pre = nn.Parameter(torch.full((L,), 0.01))
        self.a_post = nn.Parameter(torch.full((L,), 0.01))
        self.a_res = nn.Parameter(torch.full((L,), 0.01))

        lane = torch.eye(n)[torch.arange(L) % n]               # (L,n) one-hot
        self.register_buffer("pre_off", 8 * lane - 4, persistent=False)
        self.register_buffer("post_off", -4 * (1 - lane), persistent=False)
        flags = torch.zeros(L, len(tuple(cfg.engram_layers)))
        for s, layer in enumerate(cfg.engram_layers):
            flags[layer, s] = 1.0
        self.register_buffer("site_flags", flags, persistent=False)

        self.blocks = nn.ModuleList(Block(cfg) for _ in range(L))
        self.final_norm = ZCRMSNorm(cfg.dim)

        self.engrams = nn.ModuleList(Engram(cfg) for _ in cfg.engram_layers)

        # MTP (multi-token prediction, segundo token siguiente)
        self.mtp_combine = nn.Linear(cfg.dim * 2, cfg.dim, bias=False)
        _trunc_normal(self.mtp_combine.weight, 0.02)
        self.mtp_block = Block(cfg)
        self.mtp_emb_norm = ZCRMSNorm(cfg.dim)
        self.mtp_final_norm = ZCRMSNorm(cfg.dim)

        # Heads de finetune (retrieval + confianza): congelados en pretraining
        self.contrastive_head = ContrastiveHead(cfg.dim, 128)
        self.confidence_head = ConfidenceHead(cfg.dim)
        for p in (list(self.contrastive_head.parameters())
                  + list(self.confidence_head.parameters())):
            p.requires_grad_(False)

        print("Number of parameters: %.2fM"
              % (sum(p.numel() for p in self.parameters()) / 1e6,))

    # ---------- helpers ----------

    def _engram_kv(self, tokens, mask):
        orders, heads, _ = engram_geometry(self.config)
        indices = engram_indices(tokens, orders, heads, self.config.engram_slots)
        ngram_ok = torch.stack(
            [mask_diag(mask, o - 1) for o in orders for _ in range(heads)], dim=-1)
        tap_ok = torch.stack(
            [mask_diag(mask, j * max(orders)) for j in range(ENGRAM_CONV_TAPS)], dim=0)
        pairs = [e(indices, ngram_ok, tap_ok) for e in self.engrams]
        return torch.stack([k for k, _ in pairs]), torch.stack([v for _, v in pairs])

    def _hc_step(self, l, xf, block_call, cdtype):
        """Una capa de hyper-conexiones multi-lane (compartida train/inferencia)."""
        cfg = self.config
        n = cfg.mhc_lanes
        nx = rms_unit(xf.reshape(*xf.shape[:2], n * cfg.dim))
        hpre = torch.sigmoid(nx @ self.phi_pre[l].float()
                             + self.b_pre[l].float() + self.pre_off[l].float())
        u = torch.einsum("btn,btnc->btc", hpre, xf).to(cdtype)
        y = block_call(u) - u
        hpost = 2 * torch.sigmoid(nx @ self.phi_post[l].float()
                                  + self.b_post[l].float() + self.post_off[l].float())
        res = (nx @ self.phi_res[l].float()).reshape(*xf.shape[:2], n, n)
        hres = sinkhorn(self.a_res[l] * res + self.b_res[l].float())
        new_x = torch.einsum("btij,btjc->btic", hres, xf) \
            + hpost[..., None] * y.float()[:, :, None, :]
        return new_x.to(xf.dtype)

    def _stack(self, x, mask, engram_kv):
        """Hyper-conexiones multi-lane: mezcla 4 streams residuales por capa."""
        cfg = self.config
        L, n = cfg.layers, cfg.mhc_lanes
        cdtype = self.embeddings.weight.dtype
        xf = x.unsqueeze(2).float().expand(*x.shape[:2], n, x.shape[-1]).contiguous()
        for l in range(L):
            site_flag = self.site_flags[l] if engram_kv is not None else None
            ekv = engram_kv if (site_flag is not None and site_flag.sum() > 0) else None

            def block_call(u, l=l, ekv=ekv, sf=site_flag):
                return self.blocks[l](u, mask, engram_kv=ekv,
                                      site_flag=sf if ekv is not None else None)

            xf = self._hc_step(l, xf, block_call, cdtype)
        x = self.final_norm(xf.mean(dim=2)).to(cdtype)
        return x

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas,
                                      **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def _ce_chunked(self, h, targets, chunk=2048):
        """CE sin materializar logits completos: recompute por chunk en backward."""
        emb_w = self.embeddings.weight
        hf = h.reshape(-1, h.size(-1))
        tf = targets.reshape(-1)
        N = hf.shape[0]

        def _chunk_loss(hc, tc):
            logits_c = (hc @ emb_w.T).float()
            return F.cross_entropy(logits_c, tc, reduction="sum")

        total = None
        for i in range(0, N, chunk):
            lc = _ckpt(_chunk_loss, hf[i:i + chunk], tf[i:i + chunk],
                       use_reentrant=False)
            total = lc if total is None else total + lc
        return total / max(N, 1)

    # ---------- entrenamiento ----------

    def forward(self, input_ids, labels=None):
        cfg = self.config
        B, T = input_ids.shape
        x = self.embeddings(input_ids) * self.embed_scale
        mask = torch.ones(B, 1, T, T, dtype=torch.bool, device=x.device).tril()
        engram_kv = self._engram_kv(input_ids, mask)
        x = self._stack(x, mask, engram_kv)

        loss = None
        if labels is not None:
            loss = self._ce_chunked(x, labels)

            nxt = torch.cat([input_ids[:, 1:],
                             torch.zeros(B, 1, dtype=input_ids.dtype, device=x.device)],
                            dim=1)
            e2 = self.mtp_emb_norm(self.embeddings(nxt) * self.embed_scale)
            m = self.mtp_combine(torch.cat([x, e2], dim=-1))
            m = self.mtp_block(m, mask)
            m = self.mtp_final_norm(m)
            loss = loss + self._ce_chunked(m, nxt)

        logits = x @ self.embeddings.weight.T if labels is None else None
        return logits, loss

    # ---------- inferencia (ventana deslizante KV) ----------

    def _init_caches(self):
        return {"attn": [None] * self.config.layers, "tokens": None}

    def forward_with_cache(self, input_ids, offset, caches):
        cfg = self.config
        B, T = input_ids.shape
        x = self.embeddings(input_ids) * self.embed_scale
        window = cfg.kv_window or 0

        if caches["tokens"] is None:
            caches["tokens"] = input_ids.clone()
        else:
            caches["tokens"] = torch.cat([caches["tokens"], input_ids], dim=1)
        keep = max(window, 64) + max(cfg.engram_orders) ** 2 + T
        caches["tokens"] = caches["tokens"][:, -keep:]

        tok_buf = caches["tokens"]
        Tb = tok_buf.shape[1]
        buf_mask = torch.ones(B, 1, Tb, Tb, dtype=torch.bool, device=x.device).tril()
        ek_full, ev_full = self._engram_kv(tok_buf, buf_mask)
        if window and Tb > window:
            ek_full, ev_full = ek_full[:, :, -window:], ev_full[:, :, -window:]

        xf = x.unsqueeze(2).float().expand(*x.shape[:2], cfg.mhc_lanes,
                                           x.shape[-1]).contiguous()
        new_attn = []
        for l in range(cfg.layers):
            site_flag = self.site_flags[l]
            ekv = (ek_full, ev_full) if site_flag.sum() > 0 else None

            def block_call(u, l=l, ekv=ekv, sf=site_flag):
                out, kc = self.blocks[l].forward_with_cache(
                    u, offset, caches["attn"][l],
                    engram_kv=ekv, site_flag=sf if ekv is not None else None,
                    window=window)
                new_attn.append(kc)
                return out

            xf = self._hc_step(l, xf, block_call, self.embeddings.weight.dtype)
        caches["attn"] = new_attn
        x = self.final_norm(xf.mean(dim=2)).to(self.embeddings.weight.dtype)

        logits = x @ self.embeddings.weight.T
        return logits, caches

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=0.8, top_k=50,
                 top_p=0.9, repetition_penalty=1.1, eos_token_id=None):
        caches = self._init_caches()
        prompt_len = input_ids.shape[1]

        for i in range(prompt_len):
            logits, caches = self.forward_with_cache(input_ids[:, i:i + 1], i, caches)

        for gen_i in range(max_new_tokens):
            logits_last = logits[:, -1, :] / temperature

            if repetition_penalty != 1.0:
                for tok in input_ids[0].unique():
                    if logits_last[0, tok] > 0:
                        logits_last[0, tok] /= repetition_penalty
                    else:
                        logits_last[0, tok] *= repetition_penalty

            if top_k > 0:
                v, _ = torch.topk(logits_last, min(top_k, logits_last.size(-1)))
                logits_last[logits_last < v[:, [-1]]] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits_last, descending=True)
                cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                msk = cumulative - torch.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[msk] = float("-inf")
                logits_last.scatter_(1, sorted_idx, sorted_logits)

            probs = torch.softmax(logits_last, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)

            if eos_token_id is not None and next_tok.item() == eos_token_id:
                break

            input_ids = torch.cat([input_ids, next_tok], dim=1)
            logits, caches = self.forward_with_cache(next_tok, prompt_len + gen_i, caches)

        return input_ids
