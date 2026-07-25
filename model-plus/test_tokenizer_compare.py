"""Compare token IDs and logits: Python vs Rust."""
import os, sys
import numpy as np
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Python tokenizer (ByteLevel)
py_tok = Tokenizer.from_file(os.path.join(os.path.dirname(__file__), "tokenizer.json"))

# Read a sample from input.txt
text_path = os.path.join(os.path.dirname(__file__), "..", "xorIA", "input.txt")
with open(text_path, "r", encoding="utf-8") as f:
    text = f.read()

sample = text[:5000]  # first 5000 chars
py_ids = py_tok.encode(sample).ids[:128]  # first 128 tokens
print(f"Python token IDs: {len(py_ids)} tokens")
print(f"First 10: {py_ids[:10]}")

out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_data")

# ---- Tokenizer comparison ----
rust_path = os.path.join(out_dir, "rust_token_ids.npy")
if os.path.exists(rust_path):
    rust_ids = np.load(rust_path).tolist()
    n = min(len(py_ids), len(rust_ids))
    match = sum(1 for i in range(n) if py_ids[i] == rust_ids[i])
    print(f"\n--- Tokenizer Comparison ---")
    print(f"Python tokens: {len(py_ids)}")
    print(f"Rust tokens:   {len(rust_ids)}")
    print(f"Matching: {match}/{n} ({100*match/n:.1f}%)")
    if match == n:
        print("OK: Tokenizers match!")
    else:
        first_diff = next(i for i in range(n) if py_ids[i] != rust_ids[i])
        print(f"First diff at position {first_diff}: Python={py_ids[first_diff]}, Rust={rust_ids[first_diff]}")
else:
    print(f"No rust_token_ids.npy found")

# ---- Logits comparison ----
rust_logits_path = os.path.join(out_dir, "rust_logits_full.npy")
if os.path.exists(rust_logits_path):
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from python.model import TransformerLM

    model = TransformerLM(d_model=256, num_layers=6, num_heads=8, num_kv_groups=4,
                           vocab_size=16000, max_seq_len=128, use_swiglu=True).to("cpu")
    safetensors_path = os.path.join(os.path.dirname(__file__), "model_test.safetensors")
    from safetensors.torch import load_file
    state = load_file(safetensors_path)
    model.load_state_dict(state, strict=False)
    del state
    model.eval()

    input_ids = torch.tensor([py_ids], dtype=torch.long)
    with torch.no_grad():
        py_logits = model(input_ids)

    rust_logits = np.load(rust_logits_path).astype(np.float32).flatten()
    py_logits_np = py_logits.numpy().astype(np.float32).flatten()
    n = min(len(py_logits_np), len(rust_logits))
    max_diff = np.abs(py_logits_np[:n] - rust_logits[:n]).max()
    mse = np.mean((py_logits_np[:n] - rust_logits[:n])**2)
    print(f"\n--- Logits Comparison ---")
    print(f"Elements: {n}")
    print(f"Max diff: {max_diff:.6f}")
    print(f"MSE:      {mse:.10f}")
    print(f"Py [:5]:  {py_logits_np[:5]}")
    print(f"Rust[:5]: {rust_logits[:5]}")
    if max_diff < 0.01:
        print("OK: Logits match!")
    else:
        print("ERROR: Logits differ!")
else:
    print(f"No rust_logits_full.npy found")
