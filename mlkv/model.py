"""MLKV — Multi-Layer Key-Value sharing sobre base laurelia-llm (dense GQA).

Basado en https://github.com/zaydzuhri/pythia-mlkv:
- GQA normal: heads Q propias por capa, kv_groups KV heads.
- MLKV: además, K/V se comparten entre capas del mismo grupo.
  Solo la primera capa de cada grupo tiene k_proj/v_proj;
  las demás capas reusan las K/V pasadas (ya RoPEadas — mismas posiciones).
- KV cache en inferencia: num_kv_layers entradas en vez de layers.

Grupos con layers=12:
  num_kv_layers=12 -> 12 grupos × 1 capa  (GQA puro, sin compartir entre capas)
  num_kv_layers=6  -> 6 grupos × 2 capas  (default)
  num_kv_layers=4  -> 4 grupos × 3 capas
  num_kv_layers=3  -> 3 grupos × 4 capas
  num_kv_layers=2  -> 2 grupos × 6 capas
  num_kv_layers=1  -> 1 grupo × 12 capas
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
    layers = 12
    ffn_dim = 3072
    block_size = 1024
    emb_num = 32000
    rotary_pct = 0.25

    # MLKV: cantidad de grupos que tienen K/V propias (capas/grupo = layers/num_kv_layers)
    num_kv_layers = 6

    batch_size: int = 10
    grad_acc: int = 6
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    warm_up: int = 50


def repeat_kv(x, num_heads, num_kv_groups):
    if num_kv_groups == num_heads:
        return x
    return x.repeat_interleave(num_heads // num_kv_groups, dim=2)


class Attention(nn.Module):
    def __init__(self, config, has_kv=True):
        super().__init__()
        self.has_kv = has_kv
        self.num_heads = config.heads
        self.num_kv_groups = config.kv_groups
        self.head_dim = config.dim // config.heads
        self.causal = True

        self.q_proj = nn.Linear(config.dim, self.num_heads * self.head_dim, bias=False)
        if has_kv:
            self.k_proj = nn.Linear(config.dim, self.num_kv_groups * self.head_dim, bias=False)
            self.v_proj = nn.Linear(config.dim, self.num_kv_groups * self.head_dim, bias=False)
            self.k_proj.is_attention = True
            self.v_proj.is_attention = True
        self.o_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.rope = RoPE(self.head_dim, rotary_pct=getattr(config, 'rotary_pct', 0.25))
        self.attn_dropout = nn.Dropout(config.drop)

        self.q_proj.is_attention = True
        self.o_proj.is_residual_proj = True

    def forward(self, x, passed_kv=None):
        """Training/full forward. passed_kv=(k,v) ya RoPEadas de la capa previa del grupo."""
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)

        if self.has_kv or passed_kv is None:
            k = self.k_proj(x).view(B, T, self.num_kv_groups, self.head_dim)
            v = self.v_proj(x).view(B, T, self.num_kv_groups, self.head_dim)
            q, k = self.rope(q, k, 0)
            out_kv = (k, v)
        else:
            q, _ = self.rope(q, q, 0)
            k, v = passed_kv
            out_kv = passed_kv

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
        return self.o_proj(att_output), out_kv

    def forward_with_cache(self, x, offset, cache, passed_kv=None):
        """Inferencia incremental. cache=(k_full,v_full) solo para capas con KV propias."""
        B, S_new, _ = x.shape

        q_new = self.q_proj(x).view(B, S_new, self.num_heads, self.head_dim)

        if self.has_kv:
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
            used_k, used_v = k_full, v_full
        else:
            _, q_new = self.rope(q_new, q_new, offset)
            new_cache = None
            used_k, used_v = passed_kv

        k_exp = repeat_kv(used_k, self.num_heads, self.num_kv_groups)
        v_exp = repeat_kv(used_v, self.num_heads, self.num_kv_groups)

        q = q_new.transpose(1, 2)
        k = k_exp.transpose(1, 2)
        v = v_exp.transpose(1, 2)

        att_output = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=(cache is None),
        )

        att_output = att_output.transpose(1, 2).contiguous().view(B, S_new, -1)
        return self.o_proj(att_output), new_cache, (used_k, used_v)


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
    def __init__(self, config, has_kv=True):
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.dim)
        self.attn = Attention(config, has_kv=has_kv)
        self.ln_2 = nn.RMSNorm(config.dim)
        self.mlp = MLP(config)

    def forward(self, x, passed_kv=None):
        h = self.ln_1(x)
        h, out_kv = self.attn(h, passed_kv)
        x = x + h
        x = x + self.mlp(self.ln_2(x))
        return x, out_kv

    def forward_with_cache(self, x, offset, cache, passed_kv=None):
        h = self.ln_1(x)
        h, new_cache, out_kv = self.attn.forward_with_cache(h, offset, cache, passed_kv)
        x = x + h
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache, out_kv


class LLM(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        assert config.layers % config.num_kv_layers == 0, \
            f"layers={config.layers} debe ser múltiplo de num_kv_layers={config.num_kv_layers}"

        # Capas que hospedan K/V: primera siempre, resto distribuidas parejo.
        # Ej: layers=12, num_kv_layers=6 -> key_value_layers=[0, 2, 4, 6, 8, 10]
        self.key_value_layers = [0] + [
            int((i + 1) * (config.layers / config.num_kv_layers))
            for i in range(config.num_kv_layers - 1)
        ]

        self.embeddings = nn.Embedding(config.emb_num, config.dim)
        self.blocks = nn.ModuleList([
            Block(config, has_kv=(i in self.key_value_layers))
            for i in range(config.layers)
        ])
        self.norm_f = nn.RMSNorm(config.dim)

        self.lm_head = nn.Linear(config.dim, config.emb_num, bias=False)
        self.embeddings.weight = self.lm_head.weight

        self.apply(self._init_weights)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"MLKV: {n_params/1e6:.2f}M params, {config.layers} capas, "
              f"{config.num_kv_layers} grupos de {config.layers // config.num_kv_layers}")
        print(f"  key_value_layers: {self.key_value_layers}")
        print(f"  KV cache: {config.num_kv_layers} entradas "
              f"({config.layers / config.num_kv_layers:.2f}x menos que dense)")

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

        passed_kv = None
        for i, block in enumerate(self.blocks):
            if i in self.key_value_layers:
                passed_kv = None  # nuevo grupo: K/V frescas
            x, passed_kv = block(x, passed_kv)

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
        """caches: lista de num_kv_layers entradas (una por grupo), o None al inicio."""
        x = self.embeddings(input_ids)
        new_caches = []
        group_idx = -1
        passed_kv = None
        for i, block in enumerate(self.blocks):
            cache = None
            if i in self.key_value_layers:
                group_idx += 1
                passed_kv = None
                if caches is not None and group_idx < len(caches):
                    cache = caches[group_idx]
            x, new_cache, passed_kv = block.forward_with_cache(x, offset, cache, passed_kv)
            if i in self.key_value_layers:
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


def kv_cache_bytes(config, seq_len, dtype=2):
    """Bytes del KV cache para seq_len tokens."""
    head_dim = config.dim // config.heads
    per_entry = 2 * config.kv_groups * head_dim * seq_len * dtype  # k+v
    return per_entry * config.num_kv_layers
