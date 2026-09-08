"""TransformerLM — Denso MLA + MoSE por router, basado en LLM_350M_DENSE.

- Atención MLA (de moe-plus): latente KV comprimido, RoPE decoupled,
  caché latente (C_KV, K_rot) en vez de K/V completos.
- FFN denso slimmable estilo MoSE (sin MoE): 1 router lineal por capa elige
  entre 4 anchos (full/75/50/25). Train: Gumbel-ST; inferencia: argmax.
- Loss: CE estándar o TST next-bag (ver tst.py).
- Init adaptativo por capa, weight tying, KV cache para inferencia.
"""

import math, inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from rope import RoPE
from tst import multi_token_ce


class Config:
    drop = 0.0
    dim = 768
    heads = 12
    kv_groups = 4
    layers = 16
    ffn_dim = 3072
    block_size = 1024
    emb_num = 32000
    rotary_pct = 0.25  # compat (MLA usa RoPE full en d_rotate)

    # MLA
    mla_d_c = 128
    mla_d_c1 = 128
    mla_d_rotate = 64
    attn_logit_cap = 30.0

    # MoSE (router de ancho por capa, sin MoE)
    mose_widths = (1.0, 0.75, 0.50, 0.25)
    mose_cost_w = 1e-3  # penalización costo esperado (prefiere angosto si empata)
    mose_tau = 1.0      # temperatura Gumbel-ST en train

    batch_size: int = 6
    grad_acc: int = 6
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    warm_up: int = 50


def repeat_kv(x, num_heads, num_kv_groups):
    if num_kv_groups == num_heads:
        return x
    return x.repeat_interleave(num_heads // num_kv_groups, dim=2)


# ─── MLA (port de moe-plus/mla_attention.py) ─────────────────────────────

class QKVProjectionMLA(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_groups, head_dim, d_c, d_c1, d_rotate, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim
        self.d_c = d_c
        self.d_c1 = d_c1
        self.d_rotate = d_rotate

        self.W_down = nn.Linear(d_model, d_c1 + d_c + d_rotate, bias=bias)
        self.norm_cq = nn.RMSNorm(d_c1, eps=1e-6)
        self.norm_ckv = nn.RMSNorm(d_c, eps=1e-6)
        self.W_up_q = nn.Linear(d_c1, num_heads * (head_dim + d_rotate), bias=bias)
        self.W_up_kv = nn.Linear(d_c, 2 * num_kv_groups * head_dim, bias=bias)

        self.W_down.is_attention = True
        self.W_up_q.is_attention = True
        self.W_up_kv.is_attention = True

    def forward(self, x):
        B, S, _ = x.shape
        down = self.W_down(x)
        C_Q, C_KV, K_rotate = down.split([self.d_c1, self.d_c, self.d_rotate], dim=-1)

        C_Q = self.norm_cq(C_Q)
        C_KV = self.norm_ckv(C_KV)

        q_up = self.W_up_q(C_Q)
        Q_state, Q_rotate = q_up.split([self.num_heads * self.head_dim, self.num_heads * self.d_rotate], dim=-1)
        Q_state = Q_state.reshape(B, S, self.num_heads, self.head_dim)
        Q_rotate = Q_rotate.reshape(B, S, self.num_heads, self.d_rotate)

        kv_up = self.W_up_kv(C_KV)
        K, V = kv_up.chunk(2, dim=-1)
        K = K.reshape(B, S, self.num_kv_groups, self.head_dim)
        V = V.reshape(B, S, self.num_kv_groups, self.head_dim)
        K_rotate = K_rotate.reshape(B, S, 1, self.d_rotate)

        return Q_state, Q_rotate, K, V, K_rotate


class OutputProjectionMLA(nn.Module):
    def __init__(self, d_model, num_heads, head_dim, bias=False):
        super().__init__()
        self.o_proj = nn.Linear(num_heads * head_dim, d_model, bias=bias)
        self.o_proj.is_residual_proj = True

    def forward(self, x):
        B, S, NH, QK = x.shape
        return self.o_proj(x.reshape(B, S, NH * QK))


class MLAAttention(nn.Module):
    """Multi-head Latent Attention con GQA (DeepSeek MLA adaptado)."""

    def __init__(self, config):
        super().__init__()
        d_model = config.dim
        num_heads = config.heads
        num_kv_groups = config.kv_groups
        head_dim = d_model // num_heads
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim
        self.d_rotate = config.mla_d_rotate
        self.causal = True
        self.attn_logit_cap = getattr(config, "attn_logit_cap", 30.0)
        max_seq_len = config.block_size

        self.qkv = QKVProjectionMLA(
            d_model, num_heads, num_kv_groups, head_dim,
            config.mla_d_c, config.mla_d_c1, config.mla_d_rotate, bias=False)
        self.o_proj = OutputProjectionMLA(d_model, num_heads, head_dim, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.rope = RoPE(head_dim=self.d_rotate, max_seq_len=max_seq_len,
                         base=10000.0, rotary_pct=1.0)
        self.attn_dropout = nn.Dropout(config.drop)

        mask = torch.triu(torch.full((max_seq_len, max_seq_len), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def _scores(self, Q_state, Q_rot, K_state, K_rot, seq_len, kv_len=None, causal=True, q_off=0):
        """Decoupled content/sqrt(d_c) + rope (QK-norm: rope sin escala extra).

        q_off: posición global de la primera query (para chunks: máscara j<=q_off+r).
        """
        scale_c = 1.0 / math.sqrt(self.qkv.d_c)
        k_c = repeat_kv(K_state, self.num_heads, self.num_kv_groups).transpose(1, 2)
        q_c = Q_state.transpose(1, 2)
        s_c = torch.matmul(q_c, k_c.transpose(-2, -1)) * scale_c

        q_r = Q_rot.transpose(1, 2)
        k_r = K_rot.transpose(1, 2).expand(-1, self.num_heads, -1, -1)
        s_r = torch.matmul(q_r, k_r.transpose(-2, -1))

        scores = s_c + s_r
        if self.attn_logit_cap is not None:
            scores = torch.tanh(scores / self.attn_logit_cap) * self.attn_logit_cap

        if causal:
            if kv_len is None:
                kv_len = K_state.shape[1]
            if seq_len > 1:
                if seq_len == kv_len and seq_len <= self.causal_mask.shape[0]:
                    scores = scores + self.causal_mask[:seq_len, :seq_len]
                else:
                    mask = torch.triu(
                        torch.full((seq_len, kv_len), float("-inf"), device=scores.device),
                        diagonal=q_off + 1)
                    scores = scores + mask
        return scores

    def _attend(self, Q_state, Q_rot, K_state, V_state, K_rot, q_len, kv_len, causal):
        Q_state = self.q_norm(Q_state)
        K_state = self.k_norm(K_state)
        scores = self._scores(Q_state, Q_rot, K_state, K_rot, q_len, kv_len, causal)
        attn_w = F.softmax(scores, dim=-1)
        attn_w = self.attn_dropout(attn_w)
        v = repeat_kv(V_state, self.num_heads, self.num_kv_groups).transpose(1, 2)
        return torch.matmul(attn_w, v).transpose(1, 2)  # (B, T, nh, hd)

    def forward(self, x):
        Q_state, Q_rotate, K, V, K_rotate = self.qkv(x)
        Q_rotate, K_rotate = self.rope(Q_rotate, K_rotate, 0)
        T = x.shape[1]
        out = self._attend(Q_state, Q_rotate, K, V, K_rotate, T, T, True)
        return self.o_proj(out)

    def forward_with_cache(self, x, offset, cache):
        """Caché latente (C_KV, K_rot_raw)."""
        B, S_new, _ = x.shape
        down = self.qkv.W_down(x)
        C_Q_new, C_KV_new, K_rot_raw = down.split(
            [self.qkv.d_c1, self.qkv.d_c, self.qkv.d_rotate], dim=-1)
        C_Q_new = self.qkv.norm_cq(C_Q_new)
        C_KV_new = self.qkv.norm_ckv(C_KV_new)

        q_up = self.qkv.W_up_q(C_Q_new)
        Q_state, Q_rot_raw = q_up.split(
            [self.num_heads * self.head_dim, self.num_heads * self.d_rotate], dim=-1)
        Q_state = Q_state.reshape(B, S_new, self.num_heads, self.head_dim)
        Q_rot_raw = Q_rot_raw.reshape(B, S_new, self.num_heads, self.d_rotate)
        Q_rot = self.rope.apply_single(Q_rot_raw, offset=offset)

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

        K_rot = self.rope.apply_single(K_rot_full.unsqueeze(2), offset=0)

        out = self._attend(Q_state, Q_rot, K_state, V_state, K_rot,
                           S_new, S_full, S_new > 1)
        return self.o_proj(out), (C_KV_full, K_rot_full)


# ─── MoSE denso: MLP slimmable + router de ancho por capa ─────────────────

class SlimMLP(nn.Module):
    """FFN SwiGLU denso con ancho slimmable (prefijo, estilo MoSE, sin MoE)."""

    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.dim, 2 * config.ffn_dim, bias=False)
        self.fc2 = nn.Linear(config.ffn_dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.drop)
        self.full_dim = config.ffn_dim
        self.fc2.is_residual_proj = True

    def width_dim(self, width):
        if width is None:
            return self.full_dim
        return max(8, int(self.full_dim * width))

    def forward(self, x, width=None):
        d = self.width_dim(width)
        I = self.full_dim
        if d >= I:
            h = self.fc1(x)
            a, gate = h.chunk(2, dim=-1)
        else:
            w1 = self.fc1.weight
            a = F.linear(x, w1[:d])
            gate = F.linear(x, w1[I:I + d])
        h = a * F.silu(gate)
        if d >= I:
            return self.dropout(self.fc2(h))
        return self.dropout(F.linear(h, self.fc2.weight[:, :d]))


class WidthRouter(nn.Module):
    """Router mínimo por capa: 1 lineal dim→4 anchos (full/75/50/25).

    Train: Gumbel-ST (pasa hard, gradiente soft) + penalización de costo.
    Eval: argmax.
    """

    def __init__(self, config):
        super().__init__()
        self.widths = tuple(getattr(config, "mose_widths", (1.0, 0.75, 0.50, 0.25)))
        self.cost_w = getattr(config, "mose_cost_w", 1e-3)
        self.tau = getattr(config, "mose_tau", 1.0)
        self.proj = nn.Linear(config.dim, len(self.widths), bias=True)
        self.last_idx = 0  # último ancho elegido (para log)

    def forward(self, h):
        pooled = h.float().mean(dim=(0, 1))  # una decisión por capa y forward
        logits = self.proj(pooled) / self.tau
        probs = F.softmax(logits, dim=-1)
        wvec = torch.tensor(self.widths, device=h.device, dtype=probs.dtype)
        if self.training:
            u = torch.rand_like(probs).clamp_min(1e-9)
            g = -torch.log(-torch.log(u))
            y_hard = torch.zeros_like(probs).scatter_(
                0, (logits + g).argmax(-1, keepdim=True), 1.0)
            y = (y_hard - probs).detach() + probs  # ST: hard pasa, soft gradientea
        else:
            y = torch.zeros_like(probs).scatter_(
                0, probs.argmax(-1, keepdim=True), 1.0)
        idx = int(y.argmax(-1).item()) if not self.training else int(
            (logits + g).argmax(-1).item())
        self.last_idx = idx
        width = self.widths[idx]
        # Escala ST: vale 1.0 en forward, lleva gradiente del task-loss al router.
        scale = (y @ wvec) / width if self.training else None
        aux = self.cost_w * (probs @ wvec)  # prefiere angosto si empata calidad
        return width, scale, aux


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.dim)
        self.attn = MLAAttention(config)
        self.ln_2 = nn.RMSNorm(config.dim)
        self.mlp = SlimMLP(config)
        self.router = WidthRouter(config)

    def forward(self, x, width=None):
        x = x + self.attn(self.ln_1(x))
        h = self.ln_2(x)
        if width is None:
            w, scale, aux = self.router(h)
        else:
            w, scale, aux = width, None, torch.zeros((), device=x.device)
        out = self.mlp(h, w)
        if scale is not None:
            out = out * scale
        x = x + out
        return x, aux

    def forward_with_cache(self, x, offset, cache, width=None):
        h = self.ln_1(x)
        h, new_cache = self.attn.forward_with_cache(h, offset, cache)
        x = x + h
        h2 = self.ln_2(x)
        if width is None:
            w, _, _ = self.router(h2)
        else:
            w = width
        x = x + self.mlp(h2, w)
        return x, new_cache


class LLM(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.embeddings = nn.Embedding(config.emb_num, config.dim)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.norm_f = nn.RMSNorm(config.dim)

        self.lm_head = nn.Linear(config.dim, config.emb_num, bias=False)
        self.embeddings.weight = self.lm_head.weight

        self.apply(self._init_weights)
        print("Number of parameters: %.2fM" % (sum(p.numel() for p in self.parameters()) / 1e6,))

    @torch.no_grad()
    def _init_weights(self, module):
        n_layer = self.config.layers

        if isinstance(module, nn.Linear):
            if module is self.lm_head:
                return

            w_fan_in = module.weight.shape[-1]
            base_std = (1.0 / w_fan_in) ** 0.5

            if hasattr(module, 'is_residual_proj'):
                final_std = base_std / math.sqrt(2 * n_layer)
            elif hasattr(module, 'is_attention'):
                final_std = base_std * 0.7
            else:
                final_std = base_std

            torch.nn.init.trunc_normal_(
                module.weight, mean=0.0, std=final_std, a=-2*final_std, b=2*final_std
            )

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.trunc_normal_(
                module.weight, mean=0.0, std=0.02, a=-0.04, b=0.04
            )

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def forward(self, input_ids, labels=None, mtp_weights=None, width=None):
        """width: ancho global forzado (None = router por capa)."""
        x = self.embeddings(input_ids)

        aux_total = 0.0
        for block in self.blocks:
            x, aux = block(x, width)
            aux_total = aux_total + aux

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            if mtp_weights is not None:
                N = logits.size(0) * logits.size(1)
                tot = multi_token_ce(logits.float().reshape(N, -1),
                                     labels.reshape(-1), mtp_weights)
                loss = tot / max(N, 1)  # por token (igual escala que CE mean)
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.0, reduction="mean")
                loss = loss_fct(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                )

        return logits, loss, aux_total

    def width_report(self):
        """Conteo de capas por ancho elegido (full/75/50/25)."""
        n = len(self.config.mose_widths)
        counts = [0] * n
        for b in self.blocks:
            counts[b.router.last_idx] += 1
        return counts

    def forward_with_cache(self, input_ids, offset, caches, width=None):
        x = self.embeddings(input_ids)
        new_caches = []
        for i, block in enumerate(self.blocks):
            cache = caches[i] if caches is not None and i < len(caches) else None
            x, new_cache = block.forward_with_cache(x, offset, cache, width)
            new_caches.append(new_cache)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits, new_caches

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=0.8, top_k=50,
                 top_p=0.9, repetition_penalty=1.1, eos_token_id=None, width=None):
        caches = None
        prompt_len = input_ids.shape[1]

        for i in range(prompt_len):
            logits, caches = self.forward_with_cache(input_ids[:, i:i+1], i, caches, width)

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
                mask = cumulative - torch.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[mask] = float("-inf")
                logits_last.scatter_(1, sorted_idx, sorted_logits)

            probs = torch.softmax(logits_last, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)

            if eos_token_id is not None and next_tok.item() == eos_token_id:
                break

            input_ids = torch.cat([input_ids, next_tok], dim=1)
            logits, caches = self.forward_with_cache(next_tok, prompt_len + gen_i, caches, width)

        return input_ids
