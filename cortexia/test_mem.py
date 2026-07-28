"""Test:3 variantes — cache compartida MLA.

  T1: GQA Normal —12 capas, cada una con su cache
  T2: MLA Compartida + BMA + Gated — 1 MLA + 11 SharedCache (pesos propios)
  T3: MLA Compartida Basico — 1 MLA + 11 SharedKV (MISMOS K,V de layer 1)
"""

import torch
import torch.nn.functional as F
import time
import sys
import os
import math

sys.path.insert(0, os.path.dirname(__file__))
from transformers import (
    StandardGQALayer, MLALayer, SharedCacheLayer,
    CortexiaTransformer1,
)
from transformers import RMSNorm, RoPE
import torch.nn as nn


class CortexiaTransformer2BG(nn.Module):
    """MLA compartida + BMA + Gated juntos."""
    def __init__(self, vocab_size, d_model=128, num_layers=3, num_heads=8, num_kv_groups=4):
        super().__init__()
        head_dim = d_model // num_heads
        latent_dim = num_kv_groups * head_dim // 2
        self.emb = nn.Embedding(vocab_size, d_model)
        self.layer1 = MLALayer(d_model, num_heads, num_kv_groups, head_dim, latent_dim)
        self.layers_rest = nn.ModuleList([
            SharedCacheLayer(d_model, num_heads, num_kv_groups, head_dim, latent_dim,
                             use_bma=True, use_gated=True)
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


class SharedKVLayer(nn.Module):
    """Capa que comparte el MISMO W_up_kv de layer 1 — mismos K,V, solo Q propio."""
    def __init__(self, d_model, num_heads, num_kv_groups, head_dim, shared_W_up_kv):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.scale = 1.0 / math.sqrt(head_dim)
        self.q_proj = nn.Linear(d_model, num_heads * head_dim)
        self.o_proj = nn.Linear(num_heads * head_dim, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.rope = RoPE(head_dim)
        self.W_up_kv = shared_W_up_kv  # referencia compartida, no parametro propio

    def forward(self, x, C_KV, offset=0):
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
        scores = (q @ k.transpose(-2,-1)) * self.scale
        if S > 1:
            mask = torch.triu(torch.full((S, S), float("-inf"), device=x.device), diagonal=1)
            scores = scores + mask
        h2 = (F.softmax(scores, -1) @ v).transpose(1,2).flatten(2)
        x = x + self.o_proj(h2)
        x = x + self.ffn(self.norm2(x))
        return x

    def forward_with_cache(self, x, offset, shared_cache):
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
        scores = (q @ k_exp.transpose(-2,-1)) * self.scale
        kv_len = k_exp.shape[2]
        if S_new > 1:
            mask = torch.triu(torch.full((S_new, kv_len), float("-inf"), device=x.device), diagonal=kv_len - S_new + 1)
            scores = scores + mask
        h2 = (F.softmax(scores, -1) @ v_exp).transpose(1,2).flatten(2)
        x = x + self.o_proj(h2)
        x = x + self.ffn(self.norm2(x))
        return x


class CortexiaTransformer3Shared(nn.Module):
    """MLA compartida basico — todas las capas usan los MISMOS K,V de layer 1."""
    def __init__(self, vocab_size, d_model=128, num_layers=3, num_heads=8, num_kv_groups=4):
        super().__init__()
        head_dim = d_model // num_heads
        latent_dim = num_kv_groups * head_dim // 2
        self.emb = nn.Embedding(vocab_size, d_model)
        self.layer1 = MLALayer(d_model, num_heads, num_kv_groups, head_dim, latent_dim)
        self.layers_rest = nn.ModuleList([
            SharedKVLayer(d_model, num_heads, num_kv_groups, head_dim,
                          shared_W_up_kv=self.layer1.W_up_kv)
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


# ── Texto de prueba ────────────────────────────────────────────────────────
TEXT = """
El principito vivia en un planeta muy pequeno junto a su rosa favorita.
Cada dia limpiaba los volcanes y arrancaba las brotas de baobabs.
Un dia decidio viajar por el universo visitando planetas de adultos.
El primer planeta tenia un rey que solo daba ordenes absurdas.
El segundo planeta tenia un vanidoso que solo queria aplausos.
El tercer planeta tenia un borracho que bebia para olvidar que tenia verguenza de beber.
El cuarto planeta tenia un hombre de negocios que contaba estrellas.
El quinto planeta tenia un farolero que apagaba y encendia la luz cada minuto.
El sexto planeta tenia un geografo que nunca habia explorado nada.
El septimo planeta era la Tierra donde el principito encontro una serpiente.
La serpiente le dijo que podia enviarlo de vuelta a su planeta.
El principito encontro un jardin de rosas y se entristecio porque su rosa no era unica.
Un zorro le enseno que lo esencial es invisible a los ojos.
Solo se ve bien con el corazon.
El principito cuidaba su rosa porque era unica para el.
El zorro le regalo un secreto las cosas se vuelven importantes porque dedicaste tiempo a ellas.
El principito regreso a su planeta para cuidar su rosa.
"""


def char_level_encode(text):
    chars = sorted(list(set(text)))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    data = torch.tensor([char2idx[c] for c in text], dtype=torch.long)
    return data, char2idx, idx2char, len(chars)


def make_batches(data, seq_len, batch_size):
    n = (len(data) - 1) // seq_len
    data = data[:n * seq_len + 1]
    x = data[:-1].view(n, seq_len)
    y = data[1:].view(n, seq_len)
    return x, y


def train_model(model, name, data, vocab_size, idx2char, seq_len=64, batch_size=16, epochs=30, lr=3e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    x_batch, y_batch = make_batches(data, seq_len, batch_size)
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Params: {params:,}")
    print(f"  Data: {len(data):,} chars | Batch: {x_batch.shape}")
    print(f"{'='*60}")

    model.train()
    t0 = time.time()
    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        for i in range(0, x_batch.shape[0], batch_size):
            xb = x_batch[i:i+batch_size]
            yb = y_batch[i:i+batch_size]
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg = total_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  e{epoch+1:3d} | loss={avg:.4f} | {elapsed:.1f}s")

    # Generar texto
    model.eval()
    prompt = torch.tensor([ord(c) % vocab_size for c in "El principito"], dtype=torch.long, device=device)
    prompt = prompt.clamp(0, vocab_size - 1)
    try:
        generated = model.generate(prompt, max_new=100, temperature=0.8)
        gen_text = "".join([idx2char.get(i.item(), "?") for i in generated])
    except Exception as e:
        gen_text = f"[Error generando: {e}]"

    return {
        "name": name,
        "loss": avg,
        "time": time.time() - t0,
        "params": params,
        "generated": gen_text,
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data, char2idx, idx2char, vocab_size = char_level_encode(TEXT)
    print(f"Vocab: {vocab_size} | Data: {len(data):,} chars")

    SEQ_LEN = 64
    BATCH = 16
    EPOCHS = 30
    LAYERS = 12
    DIM = 128
    HEADS = 8
    KV_GROUPS = 4

    t1 = CortexiaTransformer1(vocab_size, DIM, LAYERS, HEADS, KV_GROUPS)
    t2 = CortexiaTransformer2BG(vocab_size, DIM, LAYERS, HEADS, KV_GROUPS)
    t3 = CortexiaTransformer3Shared(vocab_size, DIM, LAYERS, HEADS, KV_GROUPS)

    results = []
    results.append(train_model(t1, "T1: GQA Normal (12 capas, cache por capa)", data, vocab_size, idx2char, SEQ_LEN, BATCH, EPOCHS))
    results.append(train_model(t2, "T2: MLA+BMA+Gated (1+11, pesos propios)", data, vocab_size, idx2char, SEQ_LEN, BATCH, EPOCHS))
    results.append(train_model(t3, "T3: MLA Basico (1+11, mismos K,V)", data, vocab_size, idx2char, SEQ_LEN, BATCH, EPOCHS))

    print(f"\n{'='*60}")
    print(f"  RESULTADOS")
    print(f"{'='*60}")
    for r in results:
        print(f"\n  {r['name']}")
        print(f"    Params: {r['params']:,}")
        print(f"    Loss:   {r['loss']:.4f}")
        print(f"    Tiempo: {r['time']:.1f}s")
        print(f"    Texto:  {r['generated'][:120]}...")
    print()
