"""Cortexia: 3 transformers para comparar cache compartida vs normal.

Transformer 1: GQA normal — cada capa con su propia cache
Transformer 2: Cache unica (solo capa 1) + BMA — capas 2-3 consultan cache 1
Transformer 3: Cache unica (solo capa 1) + Gated — capas 2-3 consultan cache 1

Config: 3 capas, 128 dim, 8 heads, 4 kv groups, head_dim=16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class RoPE(nn.Module):
    def __init__(self, dim, max_len=512, base=10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        self.register_buffer("cos", freqs.cos())
        self.register_buffer("sin", freqs.sin())
    def forward(self, q, k, offset=0):
        T = q.shape[1]
        cos = self.cos[offset:offset+T].unsqueeze(0).unsqueeze(2)
        sin = self.sin[offset:offset+T].unsqueeze(0).unsqueeze(2)
        def rot(x):
            x1 = x[..., 0::2]
            x2 = x[..., 1::2]
            return torch.stack([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1).flatten(-2)
        return rot(q), rot(k)


class BMAFilter(nn.Module):
    def __init__(self, num_heads, head_dim):
        super().__init__()
        self.W_g = nn.Parameter(torch.empty(num_heads, head_dim, head_dim))
        nn.init.normal_(self.W_g, std=0.02)
    def forward(self, q, v):
        g = torch.sigmoid(torch.einsum("bhtd,hde->bhte", q, self.W_g))
        return g * v


class GatedAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_groups, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        q_out = num_heads * head_dim + num_kv_groups
        self.q_proj = nn.Linear(d_model, q_out)
        self.k_proj = nn.Linear(d_model, num_kv_groups * head_dim)
        self.v_proj = nn.Linear(d_model, num_kv_groups * head_dim)
        self.o_proj = nn.Linear(num_heads * head_dim, d_model)
        self.rope = RoPE(head_dim)
    def forward(self, x, offset=0):
        B, S, _ = x.shape
        q_raw = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q, gate = torch.split(q_raw, [self.num_heads * self.head_dim, self.num_kv_groups], dim=-1)
        q = q.reshape(B, S, self.num_heads, self.head_dim)
        gate = gate.reshape(B, S, self.num_kv_groups, 1)
        q, k = self.rope(q, k, offset)
        def repeat(x):
            n = self.num_heads // self.num_kv_groups
            return x.repeat_interleave(n, dim=2)
        k, v = repeat(k), repeat(v)
        q, k, v = [t.transpose(1,2) for t in (q, k, v)]
        scores = (q @ k.transpose(-2,-1)) * self.scale
        if S > 1:
            mask = torch.triu(torch.full((S, S), float("-inf"), device=x.device), diagonal=1)
            scores = scores + mask
        out = (F.softmax(scores, -1) @ v).transpose(1,2).flatten(2)
        gate_exp = gate.expand(-1, -1, self.num_heads, self.head_dim).flatten(2)
        out = out * torch.sigmoid(gate_exp)
        return self.o_proj(out)


class StandardGQALayer(nn.Module):
    """Transformer layer GQA normal — cada capa tiene su propia cache."""
    def __init__(self, d_model, num_heads, num_kv_groups, head_dim):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.scale = 1.0 / math.sqrt(head_dim)
        self.q_proj = nn.Linear(d_model, num_heads * head_dim)
        self.k_proj = nn.Linear(d_model, num_kv_groups * head_dim)
        self.v_proj = nn.Linear(d_model, num_kv_groups * head_dim)
        self.o_proj = nn.Linear(num_heads * head_dim, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.rope = RoPE(head_dim)
    def forward(self, x, offset=0):
        B, S, _ = x.shape
        h = self.norm1(x)
        q = self.q_proj(h).reshape(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(h).reshape(B, S, self.num_kv_groups, self.head_dim)
        v = self.v_proj(h).reshape(B, S, self.num_kv_groups, self.head_dim)
        q, k = self.rope(q, k, offset)
        n = self.num_heads // self.num_kv_groups
        k, v = k.repeat_interleave(n, 2), v.repeat_interleave(n, 2)
        q, k, v = [t.transpose(1,2) for t in (q, k, v)]
        scores = (q @ k.transpose(-2,-1)) * self.scale
        if S > 1:
            mask = torch.triu(torch.full((S, S), float("-inf"), device=x.device), diagonal=1)
            scores = scores + mask
        h2 = (F.softmax(scores, -1) @ v).transpose(1,2).flatten(2)
        x = x + self.o_proj(h2)
        x = x + self.ffn(self.norm2(x))
        return x

    def forward_with_cache(self, x, offset, cache):
        B, S_new, _ = x.shape
        h = self.norm1(x)
        q = self.q_proj(h).reshape(B, S_new, self.num_heads, self.head_dim)
        k_new = self.k_proj(h).reshape(B, S_new, self.num_kv_groups, self.head_dim)
        v_new = self.v_proj(h).reshape(B, S_new, self.num_kv_groups, self.head_dim)
        q, k_new = self.rope(q, k_new, offset)
        if cache is not None:
            k_full = torch.cat([cache[0], k_new], 1)
            v_full = torch.cat([cache[1], v_new], 1)
        else:
            k_full, v_full = k_new, v_new
        n = self.num_heads // self.num_kv_groups
        k_exp, v_exp = k_full.repeat_interleave(n, 2), v_full.repeat_interleave(n, 2)
        q, k_exp, v_exp = [t.transpose(1,2) for t in (q, k_exp, v_exp)]
        scores = (q @ k_exp.transpose(-2,-1)) * self.scale
        kv_len = k_exp.shape[2]
        if S_new > 1:
            mask = torch.triu(torch.full((S_new, kv_len), float("-inf"), device=x.device), diagonal=kv_len - S_new + 1)
            scores = scores + mask
        h2 = (F.softmax(scores, -1) @ v_exp).transpose(1,2).flatten(2)
        x = x + self.o_proj(h2)
        x = x + self.ffn(self.norm2(x))
        return x, (k_full, v_full)


class SharedCacheLayer(nn.Module):
    """Capa que consulta una cache COMPARTIDA (de capa 1) con sus propias proyecciones.

    - Cache viene de layer 1 (k_shared, v_shared)
    - Esta capa proyecta sus propios Q, K, V desde x
    - K y V de la cache se re-proyectan con las proyecciones locales de esta capa
    - Usa BMA o Gated para mejorar la atencion
    """
    def __init__(self, d_model, num_heads, num_kv_groups, head_dim, use_bma=False, use_gated=False):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.scale = 1.0 / math.sqrt(head_dim)
        self.use_bma = use_bma
        self.use_gated = use_gated

        # Proyecciones locales (proyectan x local + re-proyectan cache compartida)
        self.q_proj = nn.Linear(d_model, num_heads * head_dim)
        self.k_proj = nn.Linear(d_model, num_kv_groups * head_dim)
        self.v_proj = nn.Linear(d_model, num_kv_groups * head_dim)
        self.o_proj = nn.Linear(num_heads * head_dim, d_model)

        # Reproyeccion de cache: transformar K,V de capa1 al espacio de esta capa
        self.k_reproj = nn.Linear(num_kv_groups * head_dim, num_kv_groups * head_dim)
        self.v_reproj = nn.Linear(num_kv_groups * head_dim, num_kv_groups * head_dim)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.rope = RoPE(head_dim)

        if use_bma:
            self.bma = BMAFilter(num_heads, head_dim)
        if use_gated:
            # Gate score desde Q
            self.gate_proj = nn.Linear(d_model, num_heads)

    def forward(self, x, offset=0):
        """Training forward — self-attention local sin cache compartida."""
        B, S, _ = x.shape
        h = self.norm1(x)
        q = self.q_proj(h).reshape(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(h).reshape(B, S, self.num_kv_groups, self.head_dim)
        v = self.v_proj(h).reshape(B, S, self.num_kv_groups, self.head_dim)
        q, k = self.rope(q, k, offset)
        n = self.num_heads // self.num_kv_groups
        k, v = k.repeat_interleave(n, 2), v.repeat_interleave(n, 2)
        q, k, v = [t.transpose(1,2) for t in (q, k, v)]
        if self.use_bma:
            v = self.bma(q, v)
        scores = (q @ k.transpose(-2,-1)) * self.scale
        if S > 1:
            mask = torch.triu(torch.full((S, S), float("-inf"), device=x.device), diagonal=1)
            scores = scores + mask
        h2 = (F.softmax(scores, -1) @ v).transpose(1,2).flatten(2)
        if self.use_gated:
            gate = self.gate_proj(h).unsqueeze(-1).expand(-1, -1, self.head_dim)
            h2 = h2 * torch.sigmoid(gate)
        x = x + self.o_proj(h2)
        x = x + self.ffn(self.norm2(x))
        return x

    def forward_with_cache(self, x, offset, shared_cache):
        """Consulta cache compartida de layer 1 con proyecciones locales."""
        B, S_new, _ = x.shape
        h = self.norm1(x)

        # Q local
        q = self.q_proj(h).reshape(B, S_new, self.num_heads, self.head_dim)

        # K, V locales (de x actual, sin cache propia)
        k_local = self.k_proj(h).reshape(B, S_new, self.num_kv_groups, self.head_dim)
        v_local = self.v_proj(h).reshape(B, S_new, self.num_kv_groups, self.head_dim)

        # RoPE solo en q y k_local
        q, k_local = self.rope(q, k_local, offset)

        if shared_cache is not None:
            # Cache de layer 1 ya tiene RoPE aplicado
            k_cached = self.k_reproj(shared_cache[0])
            v_cached = self.v_reproj(shared_cache[1])
        else:
            k_cached, v_cached = k_local, v_local

        # Concatenar cache (ya con RoPE) + local (con RoPE)
        k_full = torch.cat([k_cached, k_local], 1)
        v_full = torch.cat([v_cached, v_local], 1)

        n = self.num_heads // self.num_kv_groups
        k_exp = k_full.repeat_interleave(n, 2)
        v_exp = v_full.repeat_interleave(n, 2)
        q, k_exp, v_exp = [t.transpose(1,2) for t in (q, k_exp, v_exp)]

        # BMA: pre-aggregation gating
        if self.use_bma:
            v_exp = self.bma(q, v_exp)

        scores = (q @ k_exp.transpose(-2,-1)) * self.scale
        kv_len = k_exp.shape[2]
        if S_new > 1:
            mask = torch.triu(torch.full((S_new, kv_len), float("-inf"), device=x.device), diagonal=kv_len - S_new + 1)
            scores = scores + mask

        h2 = (F.softmax(scores, -1) @ v_exp).transpose(1,2).flatten(2)

        # Gated: post-aggregation gating
        if self.use_gated:
            gate = self.gate_proj(h).unsqueeze(-1).expand(-1, -1, self.head_dim)
            h2 = h2 * torch.sigmoid(gate)

        x = x + self.o_proj(h2)
        x = x + self.ffn(self.norm2(x))
        return x


class CortexiaTransformer1(nn.Module):
    """Transformer 1: GQA normal — cada capa con su propia cache."""
    def __init__(self, vocab_size, d_model=128, num_layers=3, num_heads=8, num_kv_groups=4):
        super().__init__()
        head_dim = d_model // num_heads
        self.emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            StandardGQALayer(d_model, num_heads, num_kv_groups, head_dim)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        h = self.emb(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(self.norm(h))
    def generate(self, prompt, max_new=100, temperature=0.8):
        self.eval()
        with torch.no_grad():
            x = prompt.unsqueeze(0)
            caches = [None] * len(self.layers)
            offset = 0
            for _ in range(max_new):
                logits, caches = self.forward_with_cache(x[:, -1:], offset, caches)
                offset += 1
                probs = F.softmax(logits[:, -1] / temperature, -1)
                next_tok = torch.multinomial(probs, 1)
                x = torch.cat([x, next_tok], 1)
            return x.squeeze(0)
    def forward_with_cache(self, x, offset, caches):
        h = self.emb(x)
        new_caches = []
        for layer, cache in zip(self.layers, caches):
            h, new_cache = layer.forward_with_cache(h, offset, cache)
            new_caches.append(new_cache)
        return self.head(self.norm(h)), new_caches


class CortexiaTransformer2(nn.Module):
    """Transformer 2: Cache unica (layer 1) + BMA — capas 2-3 consultan cache 1."""
    def __init__(self, vocab_size, d_model=128, num_layers=3, num_heads=8, num_kv_groups=4):
        super().__init__()
        head_dim = d_model // num_heads
        self.emb = nn.Embedding(vocab_size, d_model)
        # Capa 1: GQA normal (produce la cache compartida)
        self.layer1 = StandardGQALayer(d_model, num_heads, num_kv_groups, head_dim)
        # Capas 2-3: consultan cache de layer 1 con BMA
        self.layers_rest = nn.ModuleList([
            SharedCacheLayer(d_model, num_heads, num_kv_groups, head_dim, use_bma=True)
            for _ in range(num_layers - 1)
        ])
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        h = self.emb(x)
        h = self.layer1(h)
        for layer in self.layers_rest:
            h = layer(h, 0)
        return self.head(self.norm(h))
    def generate(self, prompt, max_new=100, temperature=0.8):
        self.eval()
        with torch.no_grad():
            x = prompt.unsqueeze(0)
            shared_cache = None
            offset = 0
            for _ in range(max_new):
                logits, shared_cache = self.forward_with_cache(x[:, -1:], offset, shared_cache)
                offset += 1
                probs = F.softmax(logits[:, -1] / temperature, -1)
                next_tok = torch.multinomial(probs, 1)
                x = torch.cat([x, next_tok], 1)
            return x.squeeze(0)
    def forward_with_cache(self, x, offset, shared_cache):
        h = self.emb(x)
        # Layer 1 produce la cache
        h, shared_cache = self.layer1.forward_with_cache(h, offset, shared_cache)
        # Capas 2-3 consultan cache compartida
        for layer in self.layers_rest:
            h = layer.forward_with_cache(h, offset, shared_cache)
        return self.head(self.norm(h)), shared_cache


class CortexiaTransformer3(nn.Module):
    """Transformer 3: Cache unica (layer 1) + Gated — capas 2-3 consultan cache 1."""
    def __init__(self, vocab_size, d_model=128, num_layers=3, num_heads=8, num_kv_groups=4):
        super().__init__()
        head_dim = d_model // num_heads
        self.emb = nn.Embedding(vocab_size, d_model)
        # Capa 1: GQA normal (produce la cache compartida)
        self.layer1 = StandardGQALayer(d_model, num_heads, num_kv_groups, head_dim)
        # Capas 2-3: consultan cache de layer 1 con Gated
        self.layers_rest = nn.ModuleList([
            SharedCacheLayer(d_model, num_heads, num_kv_groups, head_dim, use_gated=True)
            for _ in range(num_layers - 1)
        ])
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        h = self.emb(x)
        h = self.layer1(h)
        for layer in self.layers_rest:
            h = layer(h, 0)
        return self.head(self.norm(h))
    def generate(self, prompt, max_new=100, temperature=0.8):
        self.eval()
        with torch.no_grad():
            x = prompt.unsqueeze(0)
            shared_cache = None
            offset = 0
            for _ in range(max_new):
                logits, shared_cache = self.forward_with_cache(x[:, -1:], offset, shared_cache)
                offset += 1
                probs = F.softmax(logits[:, -1] / temperature, -1)
                next_tok = torch.multinomial(probs, 1)
                x = torch.cat([x, next_tok], 1)
            return x.squeeze(0)
    def forward_with_cache(self, x, offset, shared_cache):
        h = self.emb(x)
        # Layer 1 produce la cache
        h, shared_cache = self.layer1.forward_with_cache(h, offset, shared_cache)
        # Capas 2-3 consultan cache compartida
        for layer in self.layers_rest:
            h = layer.forward_with_cache(h, offset, shared_cache)
        return self.head(self.norm(h)), shared_cache
