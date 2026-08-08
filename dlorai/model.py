"""dLoRA-MoE — MoE sobre base laurelia-llm (PROYECTO-dLoRA-MoE.md).

Implementa:
  - Atención Q–K=V (BrainChip): un solo kv_proj, sin v_proj; v = k (rotado).
    Cache KV de un solo tensor (50% menos).
  - XSA (Apple): Z = Y - (Y*Vn).sum(-1,keepdim)*Vn con Vn = normalize(v,-1).
  - LISA-SM: escalado espectral de Q/K por zona de profundidad, antes de SDPA.
  - Gated Attention G1 (headwise): g = sigmoid(proj(Z)), 1 escalar por head.
  - MoE: base denso compartido (SwiGLU, corre en todo token) + router estilo
    moe-mla (bias trick, top-k, capacity, z-loss) que consulta top_k adaptadores
    dLoRA (DoRA low-rank: solo A, B y magnitud m, sin W0 propio).
    W_eff,e = base + m_e * (B_e·A_e/r).
  - Capas pares (0-indexed impares) comparten el K de la capa anterior (sin kv_proj).
  - Residual aditivo estándar (sin AttnRes).
"""

import math, inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from rope import RoPE


class Config:
    drop = 0.0
    dim = 768
    heads = 12
    kv_groups = 4
    layers = 16
    ffn_dim = 3072
    block_size = 1024
    emb_num = 32000
    rotary_pct = 0.25

    # LISA Spectral Modulation
    sm_eps = 1e-6
    sm_zones = [(0.9, 0.9, 0.3), (1.0, 1.1, 0.6), (1.2, 1.3, 1.0)]  # shallow/mid/deep

    # G1 headwise gate
    g1_headwise = True

    # MoE
    num_experts = 4
    top_k = 2
    capacity = 1.25
    z_loss = 0.001
    shared_experts = 1
    moe_noise = 0.005
    load_balance_gamma = 0.0
    bias_decay = 0.1

    # dLoRA (DoRA)
    lora_rank = 64
    lora_alpha = 64

    # capas pares comparten K (0-indexed impar: 1,3,5,...)
    share_k_even = True

    batch_size: int = 4
    grad_acc: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    warm_up: int = 50


def repeat_kv(x, num_heads, num_kv_groups):
    if num_kv_groups == num_heads:
        return x
    return x.repeat_interleave(num_heads // num_kv_groups, dim=2)


def _sm_zone_params(layer_idx, zones):
    """Zonas adaptadas a 16 capas (LISA usa umbrales para ~40 capas)."""
    if layer_idx < 6:
        return zones[0]
    if layer_idx < 12:
        return zones[1]
    return zones[2]


class DoraLinear(nn.Module):
    """Adaptador dLoRA (DoRA low-rank): ΔW = (B·A)·(alpha/r) con magnitud m.

    Sin W0 propio: el base denso compartido (ExpertSwiGLU) provee la capa full-rank
    que corre en todos los tokens; cada adaptador solo aporta su delta low-rank.
    Init: A ~ N(0, sigma), B = 0, m = 1 → aporte nulo al inicio.
    """

    def __init__(self, in_f, out_f, r, alpha, bias=False):
        super().__init__()
        self.a = nn.Parameter(torch.randn(in_f, r) * 0.02)
        self.b = nn.Parameter(torch.zeros(r, out_f))
        self.m = nn.Parameter(torch.ones(1))
        self.scale = alpha / r
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None

    def forward(self, x):
        out = self.m * (x @ (self.a @ self.b)) * self.scale
        if self.bias is not None:
            out = out + self.bias
        return out


class DoraExpert(nn.Module):
    """SwiGLU FFN con dLoRA (DoRA) en cada matriz."""

    def __init__(self, d_model, expert_dim, r, alpha):
        super().__init__()
        self.w_gate = DoraLinear(d_model, expert_dim, r, alpha)
        self.w_up = DoraLinear(d_model, expert_dim, r, alpha)
        self.w_down = DoraLinear(expert_dim, d_model, r, alpha)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class ExpertSwiGLU(nn.Module):
    """SwiGLU denso, usado como shared expert."""

    def __init__(self, d_model, expert_dim, bias=False):
        super().__init__()
        self.w_gate = nn.Linear(d_model, expert_dim, bias=bias)
        self.w_up = nn.Linear(d_model, expert_dim, bias=bias)
        self.w_down = nn.Linear(expert_dim, d_model, bias=bias)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MoELayer(nn.Module):
    """MoE con base denso compartido + adaptadores dLoRA.

    Router estilo moe-mla: bias trick (no-learned), noisy top-k, capacity y
    z-loss. El base denso (ExpertSwiGLU) corre en todos los tokens; los
    top_k adaptadores low-rank (DoraExpert) se suman ponderados por el router.
    """

    def __init__(self, config):
        super().__init__()
        self.n_experts = config.num_experts
        self.top_k = config.top_k
        self.capacity_factor = config.capacity
        self.z_loss_gamma = config.z_loss
        self.load_balance_gamma = config.load_balance_gamma
        self.bias_decay = config.bias_decay
        self.noise_std = config.moe_noise

        self.router = nn.Linear(config.dim, self.n_experts, bias=False)
        self.register_buffer("expert_bias", torch.zeros(self.n_experts))
        self.experts = nn.ModuleList([
            DoraExpert(config.dim, config.ffn_dim, config.lora_rank, config.lora_alpha)
            for _ in range(self.n_experts)
        ])
        self.base = ExpertSwiGLU(config.dim, config.ffn_dim)

        self.register_buffer("last_counts", torch.zeros(self.n_experts, dtype=torch.long))
        self.last_total = 0

    def _router_z_loss(self, logits):
        if self.z_loss_gamma <= 0:
            return torch.tensor(0.0, device=logits.device)
        logsumexp = torch.logsumexp(logits, dim=-1)
        return self.z_loss_gamma * (logsumexp ** 2).mean()

    def _update_expert_bias(self, counts, n_tokens):
        target = n_tokens / self.n_experts
        load = counts.float()
        delta = self.bias_decay * (target - load) / max(n_tokens, 1)
        self.expert_bias.add_(delta.to(self.expert_bias.dtype))

    def forward(self, x):
        B, T, C = x.shape
        N = B * T
        xf = x.reshape(N, C)

        scores = self.router(xf)
        if self.training and self.noise_std > 0:
            scores = scores + torch.randn_like(scores) * self.noise_std

        biased_scores = scores + self.expert_bias.unsqueeze(0)
        probs = F.softmax(biased_scores, dim=-1)

        aux = self._router_z_loss(scores)
        if self.load_balance_gamma > 0:
            p_mean = probs.mean(dim=0)
            target = torch.full_like(p_mean, 1.0 / float(self.n_experts))
            aux = aux + self.load_balance_gamma * ((p_mean - target) ** 2).sum()

        topk_w, topk_i = probs.topk(self.top_k, dim=-1)
        topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-9)

        cap = max(1, int(math.ceil(self.top_k * N / self.n_experts * self.capacity_factor)))

        out = self.base(xf)
        for e in range(self.n_experts):
            sel_mask = (topk_i == e).any(dim=-1)
            tok_idx = sel_mask.nonzero(as_tuple=True)[0]
            if tok_idx.numel() == 0:
                continue
            if tok_idx.numel() > cap:
                order = probs[tok_idx, e].argsort(descending=True)
                tok_idx = tok_idx[order[:cap]]
            w = topk_w[tok_idx]
            sel = (topk_i[tok_idx] == e)
            w_e = (w * sel).sum(dim=-1)
            out[tok_idx] += w_e.unsqueeze(-1) * self.experts[e](xf[tok_idx])

        with torch.no_grad():
            counts = torch.bincount(topk_i.flatten(), minlength=self.n_experts)
            self.last_counts = counts.clone()
            self.last_total = N
            self._update_expert_bias(counts, N)

        return out.reshape(B, T, C), aux

    def balance_str(self):
        total = self.last_total or 1
        pcts = [f"{c*100//total}%" for c in self.last_counts]
        return f"exp: {'/'.join(pcts)}"


class Attention(nn.Module):
    """Atención Q–K=V + XSA + G1-headwise + LISA-SM. Capas pares comparten K."""

    def __init__(self, config, layer_idx, share_k=False):
        super().__init__()
        self.num_heads = config.heads
        self.num_kv_groups = config.kv_groups
        self.head_dim = config.dim // config.heads
        self.causal = True
        self.share_k = share_k

        self.q_proj = nn.Linear(config.dim, self.num_heads * self.head_dim, bias=False)
        if not share_k:
            self.kv_proj = nn.Linear(config.dim, self.num_kv_groups * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.rope = RoPE(self.head_dim, rotary_pct=getattr(config, 'rotary_pct', 0.25))
        self.attn_dropout = nn.Dropout(config.drop)

        self.gate = nn.Linear(self.head_dim, 1, bias=False)
        self.gate.is_gate_proj = True

        self.sm_eps = getattr(config, 'sm_eps', 1e-6)
        alpha, beta, gamma = _sm_zone_params(layer_idx, getattr(config, 'sm_zones',
                                                                [(0.9, 0.9, 0.3), (1.0, 1.1, 0.6), (1.2, 1.3, 1.0)]))
        self.sm_alpha = alpha
        self.sm_beta = beta
        self.sm_gamma = gamma

        self.q_proj.is_attention = True
        if not share_k:
            self.kv_proj.is_attention = True
        self.o_proj.is_residual_proj = True

    def _lisa_sm(self, q, k):
        """Escalado espectral de Q/K por token/head antes de SDPA."""
        trace_q = (q * q).sum(dim=-1)
        trace_k = (k * k).sum(dim=-1)
        norm_q = 1 + self.sm_gamma / torch.log(trace_q + self.sm_eps)
        norm_k = 1 + self.sm_gamma / torch.log(trace_k + self.sm_eps)
        q = q * norm_q.unsqueeze(-1) * self.sm_alpha
        k = k * norm_k.unsqueeze(-1) * self.sm_beta
        return q, k

    @staticmethod
    def _xsa(y, v):
        B, H, S, D = y.shape
        vn = F.normalize(v, p=2, dim=-1)
        vn = vn[:, :, -S:, :]
        return y - (y * vn).sum(dim=-1, keepdim=True) * vn

    def _post(self, z):
        z = z.transpose(1, 2).contiguous()
        g = torch.sigmoid(self.gate(z))
        z = z * g
        return z.view(z.shape[0], z.shape[1], -1)

    def forward(self, x, k_shared=None):
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)

        if self.share_k:
            q, _ = self.rope(q, q, 0)
            kv = k_shared
        else:
            kv = self.kv_proj(x).view(B, T, self.num_kv_groups, self.head_dim)
            q, kv = self.rope(q, kv, 0)
        k = kv
        v = kv

        q, k = self._lisa_sm(q, k)

        k = repeat_kv(k, self.num_heads, self.num_kv_groups)
        v = repeat_kv(v, self.num_heads, self.num_kv_groups)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )
        z = self._xsa(y, v)
        z = self._post(z)

        return self.o_proj(z), kv

    def forward_with_cache(self, x, offset, cache, k_shared_full=None):
        B, S_new, _ = x.shape
        q_new = self.q_proj(x).view(B, S_new, self.num_heads, self.head_dim)

        if self.share_k:
            q_new, _ = self.rope(q_new, q_new, offset)
            k_full = k_shared_full
            new_cache = None
        else:
            kv_new = self.kv_proj(x).view(B, S_new, self.num_kv_groups, self.head_dim)
            q_new, kv_new = self.rope(q_new, kv_new, offset)
            if cache is not None:
                k_full = torch.cat([cache[0], kv_new], dim=1)
            else:
                k_full = kv_new
            new_cache = (k_full.clone(),)

        k = k_full
        v = k_full

        q, k = self._lisa_sm(q_new, k)

        k_exp = repeat_kv(k, self.num_heads, self.num_kv_groups)
        v_exp = repeat_kv(v, self.num_heads, self.num_kv_groups)

        q = q.transpose(1, 2)
        k = k_exp.transpose(1, 2)
        v = v_exp.transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=(offset == 0),
        )
        z = self._xsa(y, v)
        z = self._post(z)

        return self.o_proj(z), k_full, new_cache


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        share_k = getattr(config, "share_k_even", True) and (layer_idx % 2 == 1)
        self.ln_1 = nn.RMSNorm(config.dim)
        self.attn = Attention(config, layer_idx, share_k=share_k)
        self.ln_2 = nn.RMSNorm(config.dim)
        self.moe = MoELayer(config)

    def forward(self, x, k_shared):
        h = self.ln_1(x)
        attn_out, k_out = self.attn(h, k_shared)
        x = x + attn_out
        moe_out, aux = self.moe(self.ln_2(x))
        x = x + moe_out
        return x, k_out, aux

    def forward_with_cache(self, x, offset, cache, k_shared_full):
        h = self.ln_1(x)
        attn_out, k_out_full, new_cache = self.attn.forward_with_cache(h, offset, cache, k_shared_full)
        x = x + attn_out
        x = x + self.moe(self.ln_2(x))[0]
        return x, k_out_full, new_cache


class LLM(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.embeddings = nn.Embedding(config.emb_num, config.dim)
        self.blocks = nn.ModuleList([Block(config, i) for i in range(config.layers)])
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
            if hasattr(module, 'is_gate_proj'):
                torch.nn.init.zeros_(module.weight)
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

    def forward(self, input_ids, labels=None):
        x = self.embeddings(input_ids)
        aux_total = 0
        k_prev = None
        for block in self.blocks:
            x, k_prev, aux = block(x, k_prev)
            aux_total = aux_total + aux

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.0, reduction="mean")
            loss = loss_fct(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )
            loss = loss + aux_total

        return logits, loss

    def forward_with_cache(self, input_ids, offset, caches):
        x = self.embeddings(input_ids)
        new_caches = []
        k_prev_full = None
        for i, block in enumerate(self.blocks):
            cache = caches[i] if caches is not None and i < len(caches) else None
            x, k_prev_full, new_cache = block.forward_with_cache(x, offset, cache, k_prev_full)
            new_caches.append(new_cache)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits, new_caches

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=0.8, top_k=50,
                 top_p=0.9, repetition_penalty=1.1, eos_token_id=None):
        caches = None
        prompt_len = input_ids.shape[1]

        for i in range(prompt_len):
            logits, caches = self.forward_with_cache(input_ids[:, i:i+1], i, caches)

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
            logits, caches = self.forward_with_cache(next_tok, prompt_len + gen_i, caches)

        return input_ids
