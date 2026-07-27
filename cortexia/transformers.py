"""Cortexia: 3 transformers para comparar cache compartida vs normal.

Transformer 1: GQA normal — cada capa con su propia cache
Transformer 2: MLA compartida (capa 1 comprime, capas 2+ descomprimen) + BMA
Transformer 3: MLA compartida (capa 1 comprime, capas 2+ descomprimen) + Gated

Cache almacena C_KV (latente comprimido). Cada capa aplica sus propios W_up_kv
para descomprimir K,V diferentes del MISMO latente.
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

    def forward(self, q, k, offset=0, k_offset=None):
        if k_offset is None:
            k_offset = offset
        T_q = q.shape[1]
        T_k = k.shape[1]
        cos_q = self.cos[offset:offset+T_q].unsqueeze(0).unsqueeze(2)
        sin_q = self.sin[offset:offset+T_q].unsqueeze(0).unsqueeze(2)
        cos_k = self.cos[k_offset:k_offset+T_k].unsqueeze(0).unsqueeze(2)
        sin_k = self.sin[k_offset:k_offset+T_k].unsqueeze(0).unsqueeze(2)
        def rot(x, cos, sin):
            x1 = x[..., 0::2]
            x2 = x[..., 1::2]
            return torch.stack([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1).flatten(-2)
        return rot(q, cos_q, sin_q), rot(k, cos_k, sin_k)


class BMAFilter(nn.Module):
    def __init__(self, num_heads, head_dim):
        super().__init__()
        self.W_g = nn.Parameter(torch.empty(num_heads, head_dim, head_dim))
        nn.init.normal_(self.W_g, std=0.02)
    def forward(self, q, v):
        g = torch.sigmoid(torch.einsum("bhtd,hde->bhte", q, self.W_g))
        return g * v


class StandardGQALayer(nn.Module):
    """Capa GQA normal — cada capa tiene su propia cache."""
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


class MLALayer(nn.Module):
    """Capa 1: comprime x en latente C_KV, descomprime con sus pesos propios.

    Flujo:
      C_KV = W_down(norm(x))          ← compresión a latente
      K,V = W_up_kv(C_KV)             ← descompresión propia
      Q   = q_proj(x)                 ← query propio
      attend(Q, K, V)

    Retorna (h, C_KV) para que capas 2+ lean el latente.
    """
    def __init__(self, d_model, num_heads, num_kv_groups, head_dim, latent_dim):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.latent_dim = latent_dim
        self.scale = 1.0 / math.sqrt(head_dim)

        self.W_down = nn.Linear(d_model, latent_dim, bias=False)
        self.W_up_kv = nn.Linear(latent_dim, num_kv_groups * head_dim, bias=False)
        self.q_proj = nn.Linear(d_model, num_heads * head_dim)
        self.o_proj = nn.Linear(num_heads * head_dim, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.rope = RoPE(head_dim)

    def forward(self, x, offset=0):
        B, S, _ = x.shape
        h = self.norm1(x)

        C_KV = self.W_down(h)
        kv = self.W_up_kv(C_KV)
        k = kv.reshape(B, S, self.num_kv_groups, self.head_dim)
        v = kv.reshape(B, S, self.num_kv_groups, self.head_dim)
        q = self.q_proj(h).reshape(B, S, self.num_heads, self.head_dim)

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
        return x, C_KV

    def forward_with_cache(self, x, offset, cache):
        B, S_new, _ = x.shape
        h = self.norm1(x)

        C_KV_new = self.W_down(h)
        if cache is not None:
            C_KV_full = torch.cat([cache, C_KV_new], 1)
        else:
            C_KV_full = C_KV_new

        T = C_KV_full.shape[1]
        kv = self.W_up_kv(C_KV_full)
        k = kv.reshape(B, T, self.num_kv_groups, self.head_dim)
        v = kv.reshape(B, T, self.num_kv_groups, self.head_dim)
        q = self.q_proj(h).reshape(B, S_new, self.num_heads, self.head_dim)

        q, k = self.rope(q, k, offset, k_offset=0)
        n = self.num_heads // self.num_kv_groups
        k_exp = k.repeat_interleave(n, 2)
        v_exp = v.repeat_interleave(n, 2)
        q, k_exp, v_exp = [t.transpose(1,2) for t in (q, k_exp, v_exp)]

        scores = (q @ k_exp.transpose(-2,-1)) * self.scale
        kv_len = k_exp.shape[2]
        if S_new > 1:
            mask = torch.triu(torch.full((S_new, kv_len), float("-inf"), device=x.device), diagonal=kv_len - S_new + 1)
            scores = scores + mask
        h2 = (F.softmax(scores, -1) @ v_exp).transpose(1,2).flatten(2)

        x = x + self.o_proj(h2)
        x = x + self.ffn(self.norm2(x))
        return x, C_KV_full


class SharedCacheLayer(nn.Module):
    """Capas 2+: lee C_KV del cache (layer 1), descomprime con sus propios W_up_kv.

    Cada capa tiene sus propios pesos de descompresión → K,V diferentes
    del MISMO latente C_KV. Q viene de x vía q_proj propio.

    Flujo:
      K,V = W_up_kv(C_KV)   ← descomprime con SUS pesos (diferentes a layer 1)
      Q   = q_proj(x)       ← query propio
      attend(Q, K, V)
    """
    def __init__(self, d_model, num_heads, num_kv_groups, head_dim, latent_dim, use_bma=False, use_gated=False):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.latent_dim = latent_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        self.use_bma = use_bma
        self.use_gated = use_gated

        self.q_proj = nn.Linear(d_model, num_heads * head_dim)
        self.W_up_kv = nn.Linear(latent_dim, num_kv_groups * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.rope = RoPE(head_dim)

        if use_bma:
            self.bma = BMAFilter(num_heads, head_dim)
        if use_gated:
            self.gate_proj = nn.Linear(d_model, num_heads)

    def forward(self, x, C_KV, offset=0):
        """Training: lee C_KV de layer 1, descomprime con sus pesos, atiende."""
        B, S, _ = x.shape
        h = self.norm1(x)

        q = self.q_proj(h).reshape(B, S, self.num_heads, self.head_dim)

        kv = self.W_up_kv(C_KV)
        k = kv.reshape(B, S, self.num_kv_groups, self.head_dim)
        v = kv.reshape(B, S, self.num_kv_groups, self.head_dim)

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

        h2_4d = F.softmax(scores, -1) @ v

        if self.use_gated:
            gate = self.gate_proj(h).unsqueeze(-1).permute(0, 2, 1, 3).expand(-1, -1, -1, self.head_dim)
            h2_4d = h2_4d * torch.sigmoid(gate)

        h2 = h2_4d.transpose(1,2).flatten(2)
        x = x + self.o_proj(h2)
        x = x + self.ffn(self.norm2(x))
        return x

    def forward_with_cache(self, x, offset, shared_cache):
        """Inference: lee C_KV completo del cache, descomprime con sus pesos."""
        B, S_new, _ = x.shape
        h = self.norm1(x)

        q = self.q_proj(h).reshape(B, S_new, self.num_heads, self.head_dim)

        if shared_cache is not None:
            T = shared_cache.shape[1]
            kv = self.W_up_kv(shared_cache)
            k = kv.reshape(B, T, self.num_kv_groups, self.head_dim)
            v = kv.reshape(B, T, self.num_kv_groups, self.head_dim)
        else:
            k = torch.zeros(B, 0, self.num_kv_groups, self.head_dim, device=x.device)
            v = torch.zeros(B, 0, self.num_kv_groups, self.head_dim, device=x.device)

        q, k = self.rope(q, k, offset, k_offset=0)
        n = self.num_heads // self.num_kv_groups
        k_exp = k.repeat_interleave(n, 2)
        v_exp = v.repeat_interleave(n, 2)
        q, k_exp, v_exp = [t.transpose(1,2) for t in (q, k_exp, v_exp)]

        if self.use_bma:
            v_exp = self.bma(q, v_exp)

        scores = (q @ k_exp.transpose(-2,-1)) * self.scale
        kv_len = k_exp.shape[2]
        if S_new > 1:
            mask = torch.triu(torch.full((S_new, kv_len), float("-inf"), device=x.device), diagonal=kv_len - S_new + 1)
            scores = scores + mask

        h2_4d = F.softmax(scores, -1) @ v_exp

        if self.use_gated:
            gate = self.gate_proj(h).unsqueeze(-1).permute(0, 2, 1, 3).expand(-1, -1, -1, self.head_dim)
            h2_4d = h2_4d * torch.sigmoid(gate)

        h2 = h2_4d.transpose(1,2).flatten(2)
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
            prompt_len = x.shape[1]
            for i in range(prompt_len):
                logits, caches = self.forward_with_cache(x[:, i:i+1], offset, caches)
                offset += 1
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
    """Transformer 2: MLA compartida + BMA — capa 1 comprime, capas 2+ descomprimen."""
    def __init__(self, vocab_size, d_model=128, num_layers=3, num_heads=8, num_kv_groups=4):
        super().__init__()
        head_dim = d_model // num_heads
        latent_dim = num_kv_groups * head_dim // 2
        self.emb = nn.Embedding(vocab_size, d_model)
        self.layer1 = MLALayer(d_model, num_heads, num_kv_groups, head_dim, latent_dim)
        self.layers_rest = nn.ModuleList([
            SharedCacheLayer(d_model, num_heads, num_kv_groups, head_dim, latent_dim, use_bma=True)
            for _ in range(num_layers - 1)
        ])
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.emb(x)
        h, C_KV = self.layer1(h)
        for layer in self.layers_rest:
            h = layer(h, C_KV)
        return self.head(self.norm(h))

    def generate(self, prompt, max_new=100, temperature=0.8):
        self.eval()
        with torch.no_grad():
            x = prompt.unsqueeze(0)
            shared_cache = None
            offset = 0
            prompt_len = x.shape[1]
            for i in range(prompt_len):
                logits, shared_cache = self.forward_with_cache(x[:, i:i+1], offset, shared_cache)
                offset += 1
            for _ in range(max_new):
                logits, shared_cache = self.forward_with_cache(x[:, -1:], offset, shared_cache)
                offset += 1
                probs = F.softmax(logits[:, -1] / temperature, -1)
                next_tok = torch.multinomial(probs, 1)
                x = torch.cat([x, next_tok], 1)
            return x.squeeze(0)

    def forward_with_cache(self, x, offset, shared_cache):
        h = self.emb(x)
        h, shared_cache = self.layer1.forward_with_cache(h, offset, shared_cache)
        for layer in self.layers_rest:
            h = layer.forward_with_cache(h, offset, shared_cache)
        return self.head(self.norm(h)), shared_cache


class CortexiaTransformer3(nn.Module):
    """Transformer 3: MLA compartida + Gated — capa 1 comprime, capas 2+ descomprimen."""
    def __init__(self, vocab_size, d_model=128, num_layers=3, num_heads=8, num_kv_groups=4):
        super().__init__()
        head_dim = d_model // num_heads
        latent_dim = num_kv_groups * head_dim // 2
        self.emb = nn.Embedding(vocab_size, d_model)
        self.layer1 = MLALayer(d_model, num_heads, num_kv_groups, head_dim, latent_dim)
        self.layers_rest = nn.ModuleList([
            SharedCacheLayer(d_model, num_heads, num_kv_groups, head_dim, latent_dim, use_gated=True)
            for _ in range(num_layers - 1)
        ])
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.emb(x)
        h, C_KV = self.layer1(h)
        for layer in self.layers_rest:
            h = layer(h, C_KV)
        return self.head(self.norm(h))

    def generate(self, prompt, max_new=100, temperature=0.8):
        self.eval()
        with torch.no_grad():
            x = prompt.unsqueeze(0)
            shared_cache = None
            offset = 0
            prompt_len = x.shape[1]
            for i in range(prompt_len):
                logits, shared_cache = self.forward_with_cache(x[:, i:i+1], offset, shared_cache)
                offset += 1
            for _ in range(max_new):
                logits, shared_cache = self.forward_with_cache(x[:, -1:], offset, shared_cache)
                offset += 1
                probs = F.softmax(logits[:, -1] / temperature, -1)
                next_tok = torch.multinomial(probs, 1)
                x = torch.cat([x, next_tok], 1)
            return x.squeeze(0)

    def forward_with_cache(self, x, offset, shared_cache):
        h = self.emb(x)
        h, shared_cache = self.layer1.forward_with_cache(h, offset, shared_cache)
        for layer in self.layers_rest:
            h = layer.forward_with_cache(h, offset, shared_cache)
        return self.head(self.norm(h)), shared_cache
