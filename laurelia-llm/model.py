"""TransformerLM: Dense GQA + XSA.

Sin MLA, sin BMA, sin gated, sin MoE.
"""

import math
import torch
import torch.nn as nn

from attention import Attention, RMSNorm


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, inter_dim, dropout=0.0, bias=False):
        super().__init__()
        self.w_gate = nn.Linear(d_model, inter_dim, bias=bias)
        self.w_up = nn.Linear(d_model, inter_dim, bias=bias)
        self.w_down = nn.Linear(inter_dim, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        gate = torch.nn.functional.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.dropout(self.w_down(gate * up))


class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_groups=0, head_dim=None,
                 max_seq_len=2048, rope_base=10000.0, rope_scaling=1.0,
                 causal=True, dropout=0.0, ffn_dropout=0.0,
                 attn_logit_cap=None, bias=False, norm_eps=1e-5,
                 ffn_expansion=4.0, use_swiglu=True, ffn_round_to=64,
                 use_xsa=False, qk_norm=True, use_sandwich_norm=False,
                 rotary_pct=1.0):
        super().__init__()
        if num_kv_groups == 0:
            num_kv_groups = num_heads
        if head_dim is None:
            head_dim = d_model // num_heads

        self.use_sandwich_norm = use_sandwich_norm

        self.attn_norm = RMSNorm(d_model, eps=norm_eps)
        if use_sandwich_norm:
            self.attn_norm_out = RMSNorm(d_model, eps=norm_eps)

        self.attention = Attention(
            d_model=d_model, num_heads=num_heads, num_kv_groups=num_kv_groups,
            head_dim=head_dim, max_seq_len=max_seq_len, rope_base=rope_base,
            rope_scaling=rope_scaling, causal=causal, dropout=dropout,
            attn_logit_cap=attn_logit_cap, bias=bias,
            use_xsa=use_xsa, qk_norm=qk_norm, rotary_pct=rotary_pct,
        )

        inter_dim = int(d_model * ffn_expansion)
        inter_dim = ((inter_dim + ffn_round_to - 1) // ffn_round_to) * ffn_round_to

        self.ffn_norm = RMSNorm(d_model, eps=norm_eps)
        if use_sandwich_norm:
            self.ffn_norm_out = RMSNorm(d_model, eps=norm_eps)

        if use_swiglu:
            self.ffn = SwiGLUFFN(d_model, inter_dim, ffn_dropout, bias)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, inter_dim, bias=bias),
                nn.GELU(),
                nn.Dropout(ffn_dropout),
                nn.Linear(inter_dim, d_model, bias=bias),
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, RMSNorm):
                nn.init.ones_(m.weight)

    def forward(self, x, offset=0):
        h = self.attn_norm(x)
        h = self.attention(h, offset)
        if self.use_sandwich_norm:
            h = self.attn_norm_out(h)
        x = x + h

        h = self.ffn_norm(x)
        h = self.ffn(h)
        if self.use_sandwich_norm:
            h = self.ffn_norm_out(h)
        x = x + h
        return x

    def forward_with_cache(self, x, offset, cache):
        h = self.attn_norm(x)
        h, new_cache = self.attention.forward_with_cache(h, offset, cache)
        if self.use_sandwich_norm:
            h = self.attn_norm_out(h)
        x = x + h

        h = self.ffn_norm(x)
        h = self.ffn(h)
        if self.use_sandwich_norm:
            h = self.ffn_norm_out(h)
        x = x + h
        return x, new_cache


class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, num_kv_groups=0,
                 head_dim=None, ffn_expansion=4.0, use_swiglu=True,
                 max_seq_len=2048, rope_base=10000.0, rope_scaling=1.0,
                 causal=True, dropout=0.0, ffn_dropout=0.0,
                 attn_logit_cap=None, bias=False, norm_eps=1e-5,
                 ffn_round_to=64, use_xsa=False, qk_norm=True,
                 use_sandwich_norm=False, rotary_pct=1.0):
        super().__init__()
        if num_kv_groups == 0:
            num_kv_groups = num_heads
        if head_dim is None:
            head_dim = d_model // num_heads

        self.layers = nn.ModuleList([
            TransformerLayer(
                d_model=d_model, num_heads=num_heads, num_kv_groups=num_kv_groups,
                head_dim=head_dim, max_seq_len=max_seq_len, rope_base=rope_base,
                rope_scaling=rope_scaling, causal=causal, dropout=dropout,
                ffn_dropout=ffn_dropout, attn_logit_cap=attn_logit_cap,
                bias=bias, norm_eps=norm_eps, ffn_expansion=ffn_expansion,
                use_swiglu=use_swiglu, ffn_round_to=ffn_round_to,
                use_xsa=use_xsa, qk_norm=qk_norm,
                use_sandwich_norm=use_sandwich_norm,
                rotary_pct=rotary_pct,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model, eps=norm_eps)

    def forward(self, x, offset=0):
        for layer in self.layers:
            x = layer(x, offset)
        return self.final_norm(x)

    def forward_with_cache(self, x, offset, caches):
        new_caches = []
        for i, layer in enumerate(self.layers):
            cache = caches[i] if caches is not None and i < len(caches) else None
            x, new_cache = layer.forward_with_cache(x, offset, cache)
            new_caches.append(new_cache)
        return x, new_caches


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads,
                 num_kv_groups=0, head_dim=None, ffn_expansion=4.0,
                 use_swiglu=True, use_x0=False, max_seq_len=2048,
                 rope_base=10000.0, rope_scaling=1.0, causal=True,
                 attn_dropout=0.0, ffn_dropout=0.0, residual_dropout=0.0,
                 attn_logit_cap=None, bias=False, norm_eps=1e-5,
                 ffn_round_to=64, use_xsa=False, qk_norm=True,
                 use_sandwich_norm=False, noise_std=0.0,
                 use_mla=False, use_moe=False, use_gated_attn=False,
                 gated_type="headwise", use_bma=False, cache_every=1,
                 mla_d_c=None, mla_d_c1=None, mla_d_rotate=None,
                 mla_block_size=128, rotary_pct=1.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)

        self.transformer = Transformer(
            num_layers=num_layers, d_model=d_model, num_heads=num_heads,
            num_kv_groups=num_kv_groups, head_dim=head_dim,
            ffn_expansion=ffn_expansion, use_swiglu=use_swiglu,
            max_seq_len=max_seq_len, rope_base=rope_base,
            rope_scaling=rope_scaling, causal=causal,
            dropout=attn_dropout, ffn_dropout=ffn_dropout,
            attn_logit_cap=attn_logit_cap, bias=bias, norm_eps=norm_eps,
            ffn_round_to=ffn_round_to, use_xsa=use_xsa, qk_norm=qk_norm,
            use_sandwich_norm=use_sandwich_norm,
            rotary_pct=rotary_pct,
        )

        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.emb_weight = self.embedding.weight
        self.head.weight = self.embedding.weight

        self.use_x0 = use_x0
        self.noise_std = noise_std

    def forward(self, x):
        h = self.embedding(x)
        if self.training and self.noise_std > 0:
            h = h + torch.randn_like(h) * self.noise_std
        h = self.transformer(h, 0)
        logits = self.head(h)
        return logits, 0.0

    def forward_with_cache(self, x, offset, caches):
        h = self.embedding(x)
        h, new_caches = self.transformer.forward_with_cache(h, offset, caches)
        logits = self.head(h)
        return logits, new_caches

    @torch.no_grad()
    def generate(self, x, max_new_tokens=100, temperature=0.8, top_k=50,
                 top_p=0.9, repetition_penalty=1.1):
        caches = None
        prompt_len = x.shape[1]

        for i in range(prompt_len):
            logits, caches = self.forward_with_cache(x[:, i:i+1], i, caches)

        for _ in range(max_new_tokens):
            logits_last = logits[:, -1, :] / temperature

            if repetition_penalty != 1.0:
                for tok in x[0].unique():
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
            x = torch.cat([x, next_tok], dim=1)
            logits, caches = self.forward_with_cache(next_tok, prompt_len + _ , caches)

        return x
