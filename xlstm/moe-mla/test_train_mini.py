"""Mini test: train 3-layer xLSTM MoE on input.txt (no HF, no checkpoint)."""
import sys, os, time, math, torch
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.join(_DIR, "..", ".."))

from mlstm_kernels_mock import install_mock; install_mock()
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from xlstm_model import xLSTMMoEModel

d_model = 128
num_layers = 3
num_heads = 4
seq_len = 64
batch_size = 4
lr = 3e-4
steps = 5000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# BPE tokenizer from input.txt
input_path = os.path.join(_DIR, "..", "..", "rust", "input.txt")
# Also try relative to repo root
if not os.path.exists(input_path):
    input_path = os.path.join(_DIR, "..", "..", "..", "rust", "input.txt")

tok = Tokenizer(models.BPE(unk_token="[UNK]"))
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(vocab_size=1000, special_tokens=["[UNK]", "[EOS]"])
tok.train([input_path], trainer=trainer)
vocab_size = tok.get_vocab_size()
print(f"Vocab: {vocab_size}")

# Model: 3 layers, first 2 dense, last 1 MoE
model = xLSTMMoEModel(
    vocab_size=vocab_size, d_model=d_model, num_layers=num_layers,
    num_heads=num_heads, moe_at=[2], n_experts=4, top_k=1, n_shared=1,
    noise_std=0.01, max_seq_len=seq_len,
).to(device)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

opt = torch.optim.AdamW(model.parameters(), lr=lr)

# Tokenize text
with open(input_path, "r", encoding="utf-8") as f:
    text = f.read()[:100000]
tokens = tok.encode(text).ids
print(f"Tokens: {len(tokens)}")

# Train loop
t0 = time.time()
max_start = len(tokens) - seq_len - 1
for step in range(steps):
    idx = (step * seq_len * batch_size) % max_start
    x_list, y_list = [], []
    for i in range(batch_size):
        off = idx + i * seq_len
        if off + seq_len + 1 > len(tokens):
            off = max_start - (batch_size - i) * seq_len
        x_list.append(torch.tensor(tokens[off:off+seq_len], dtype=torch.long))
        y_list.append(torch.tensor(tokens[off+1:off+seq_len+1], dtype=torch.long))
    x = torch.stack(x_list).to(device)
    y = torch.stack(y_list).to(device)

    logits, aux_loss = model(x)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    (loss + aux_loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()

    if step % 50 == 0:
        dt = time.time() - t0
        print(f"s{step} loss {loss.item():.4f} aux {aux_loss.item():.6f} {dt:.1f}s")
        for blk in model.blocks:
            if hasattr(blk, "moe") and hasattr(blk.moe, "last_counts"):
                bal = blk.moe.balance_str()
                print(f"  L{blk._layer_idx}: {bal}")
        t0 = time.time()

# Generate sample
model.eval()
prompt_ids = tok.encode("hola").ids
x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
out = model.generate(x, max_new_tokens=50, temperature=1.0, top_k=20)
print(f"Generated: {tok.decode(out[0].tolist())}")
model.train()

print("Done!")
