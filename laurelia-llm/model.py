"""TransformerLM — Dense GQA, basado en LLM_350M_DENSE.

Init adaptativo por capa, weight tying, KV cache para inferencia.
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

    batch_size: int = 4
    grad_acc: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    warm_up: int = 50


def repeat_kv(x, num_heads, num_kv_groups):
    if num_kv_groups == num_heads:
        return x
    return x.repeat_interleave(num_heads // num_kv_groups, dim=2)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.heads
        self.num_kv_groups = config.kv_groups
        self.head_dim = config.dim // config.heads
        self.causal = True

        self.q_proj = nn.Linear(config.dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, self.num_kv_groups * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, self.num_kv_groups * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.rope = RoPE(self.head_dim, rotary_pct=getattr(config, 'rotary_pct', 0.25))
        self.attn_dropout = nn.Dropout(config.drop)

        self.q_proj.is_attention = True
        self.k_proj.is_attention = True
        self.v_proj.is_attention = True
        self.o_proj.is_residual_proj = True

    def forward(self, x):
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_groups, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_groups, self.head_dim)

        q, k = self.rope(q, k, 0)

        k = repeat_kv(k, self.num_heads, self.num_kv_groups)
        v = repeat_kv(v, self.num_heads, self.num_kv_groups)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        att_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )

        att_output = att_output.transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(att_output)

    def forward_with_cache(self, x, offset, cache):
        B, S_new, _ = x.shape

        q_new = self.q_proj(x).view(B, S_new, self.num_heads, self.head_dim)
        k_new = self.k_proj(x).view(B, S_new, self.num_kv_groups, self.head_dim)
        v_new = self.v_proj(x).view(B, S_new, self.num_kv_groups, self.head_dim)

        q_new, k_new = self.rope(q_new, k_new, offset)

        if cache is not None:
            k_full = torch.cat([cache[0], k_new], dim=1)
            v_full = torch.cat([cache[1], v_new], dim=1)
        else:
            k_full = k_new
            v_full = v_new

        new_cache = (k_full.clone(), v_full.clone())

        k_exp = repeat_kv(k_full, self.num_heads, self.num_kv_groups)
        v_exp = repeat_kv(v_full, self.num_heads, self.num_kv_groups)

        q = q_new.transpose(1, 2)
        k = k_exp.transpose(1, 2)
        v = v_exp.transpose(1, 2)

        att_output = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=(cache is None),
        )

        att_output = att_output.transpose(1, 2).contiguous().view(B, S_new, -1)
        return self.o_proj(att_output), new_cache


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.dim, 2 * config.ffn_dim, bias=False)
        self.fc2 = nn.Linear(config.ffn_dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.drop)
        self.fc2.is_residual_proj = True

    def forward(self, x):
        x = self.fc1(x)
        x, gate = x.chunk(2, dim=-1)
        x = x * F.silu(gate)
        return self.dropout(self.fc2(x))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.dim)
        self.attn = Attention(config)
        self.ln_2 = nn.RMSNorm(config.dim)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def forward_with_cache(self, x, offset, cache):
        h = self.ln_1(x)
        h, new_cache = self.attn.forward_with_cache(h, offset, cache)
        x = x + h
        x = x + self.mlp(self.ln_2(x))
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

    def forward(self, input_ids, labels=None):
        x = self.embeddings(input_ids)

        for block in self.blocks:
            x = block(x)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.0, reduction="mean")
            loss = loss_fct(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

        return logits, loss

    def forward_with_cache(self, input_ids, offset, caches):
        x = self.embeddings(input_ids)
        new_caches = []
        for i, block in enumerate(self.blocks):
            cache = caches[i] if caches is not None and i < len(caches) else None
            x, new_cache = block.forward_with_cache(x, offset, cache)
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
