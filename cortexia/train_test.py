"""Cortexia: Test character-level — 3 transformers comparados.

Entrena 3 modelos en el mismo texto y compara loss, tiempo, y generacion.

Modelos:
  T1: GQA normal — cache por capa
  T2: Cache unica (layer 1) + BMA
  T3: Cache unica (layer 1) + Gated
"""

import torch
import torch.nn.functional as F
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from transformers import CortexiaTransformer1, CortexiaTransformer2, CortexiaTransformer3


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


def train_model(model, name, data, vocab_size, seq_len=64, batch_size=16, epochs=20, lr=3e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    x_batch, y_batch = make_batches(data, seq_len, batch_size)
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
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
    # Usar solo caracteres que existen en vocab
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
        "params": sum(p.numel() for p in model.parameters()),
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
    t2 = CortexiaTransformer2(vocab_size, DIM, LAYERS, HEADS, KV_GROUPS)
    t3 = CortexiaTransformer3(vocab_size, DIM, LAYERS, HEADS, KV_GROUPS)

    results = []
    results.append(train_model(t1, "T1: GQA Normal (cache por capa)", data, vocab_size, SEQ_LEN, BATCH, EPOCHS))
    results.append(train_model(t2, "T2: Cache Unica + BMA", data, vocab_size, SEQ_LEN, BATCH, EPOCHS))
    results.append(train_model(t3, "T3: Cache Unica + Gated", data, vocab_size, SEQ_LEN, BATCH, EPOCHS))

    print(f"\n{'='*60}")
    print(f"  RESULTADOS")
    print(f"{'='*60}")
    for r in results:
        print(f"\n  {r['name']}")
        print(f"    Params: {r['params']:,}")
        print(f"    Loss:   {r['loss']:.4f}")
        print(f"    Tiempo: {r['time']:.1f}s")
        print(f"    Texto:  {r['generated'][:80]}...")
    print()
