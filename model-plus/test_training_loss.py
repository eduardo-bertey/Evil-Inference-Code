"""Test: run one training batch, compare Rust forward_train_partial_rope vs Python."""
import os, sys, math
import torch
import torch.nn.functional as F
import numpy as np
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from python.model import TransformerLM

device = torch.device("cpu")

# Load model from safetensors
model = TransformerLM(d_model=256, num_layers=6, num_heads=8, num_kv_groups=4,
                       vocab_size=16000, max_seq_len=128, use_swiglu=True, use_x0=True).to(device)
safetensors_path = os.path.join(os.path.dirname(__file__), "model_test.safetensors")
from safetensors.torch import load_file
state = load_file(safetensors_path)
model.load_state_dict(state, strict=False)
del state
model.eval()

# Load tokenizer and create one batch from input.txt
tokenizer = Tokenizer.from_file(os.path.join(os.path.dirname(__file__), "tokenizer.json"))
text_path = os.path.join(os.path.dirname(__file__), "..", "xorIA", "input.txt")
with open(text_path, "r", encoding="utf-8") as f:
    text = f.read()
tokens = tokenizer.encode(text).ids
print(f"Total tokens: {len(tokens)}")

batch_size = 1
seq_len = 128
stride = seq_len
batch_idx = 0  # first batch

x_flat = []
y_flat = []
for i in range(batch_size):
    start = (batch_idx + i) * stride
    for j in range(seq_len):
        x_flat.append(tokens[start + j])
        y_flat.append(tokens[start + j + 1])
input_ids = torch.tensor(x_flat, dtype=torch.long).view(batch_size, seq_len)
targets = torch.tensor(y_flat, dtype=torch.long).view(batch_size, seq_len)
print(f"Input shape: {input_ids.shape}, Targets shape: {targets.shape}")

# ---- Standard forward (what Python training uses) ----
with torch.no_grad():
    logits_std = model(input_ids)
loss_std = F.cross_entropy(logits_std.view(-1, 16000), targets.view(-1))
print(f"\nPython standard forward loss: {loss_std.item():.4f}")

# ---- forward_train_partial_rope at 100% ----
with torch.no_grad():
    logits_partial = model.forward_train_partial_rope(input_ids, rotary_pct=1.0)
loss_partial = F.cross_entropy(logits_partial.view(-1, 16000), targets.view(-1))
print(f"Python partial_rope forward loss: {loss_partial.item():.4f}")

# Diff between the two
diff = (logits_std - logits_partial).abs().max().item()
print(f"Max diff between std and partial: {diff:.6f}")
if diff > 0.01:
    print("WARNING: forward_train_partial_rope != standard forward!")
else:
    print("OK: forward_train_partial_rope matches standard forward")

# Save for Rust
out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_data")
os.makedirs(out_dir, exist_ok=True)
np.save(os.path.join(out_dir, "training_input_ids.npy"), input_ids.numpy().astype(np.int32))
np.save(os.path.join(out_dir, "training_targets.npy"), targets.numpy().astype(np.int32))
np.save(os.path.join(out_dir, "logits_py_std.npy"), logits_std.numpy().astype(np.float32))
np.save(os.path.join(out_dir, "logits_py_partial.npy"), logits_partial.numpy().astype(np.float32))
print(f"\nSaved to {out_dir}/")

# ---- Compare with Rust logits if available ----
rust_path = os.path.join(out_dir, "logits_rust_partial.npy")
if os.path.exists(rust_path):
    rust_logits = np.load(rust_path).flatten()
    py_logits = logits_partial.numpy().astype(np.float32).flatten()
    n = min(len(py_logits), len(rust_logits))
    max_diff = np.abs(py_logits[:n] - rust_logits[:n]).max()
    mse = np.mean((py_logits[:n] - rust_logits[:n])**2)
    print(f"\n--- Rust vs Python comparison ---")
    print(f"Elements: {n}")
    print(f"Max diff: {max_diff:.6f}")
    print(f"MSE:      {mse:.10f}")
    print(f"Py [:5]:  {py_logits[:5]}")
    print(f"Rust[:5]: {rust_logits[:5]}")
    if max_diff < 0.01:
        print("OK: Rust forward_train_partial_rope matches Python")
    else:
        print(f"WARNING: max diff {max_diff} > 0.01")
