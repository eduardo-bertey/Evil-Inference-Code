"""Verify that model init does NOT have logit over-amplification bug.

The bug: embedding init N(0, 1/sqrt(d_model)) + Kaiming on Linear layers
→ logit std ~1.0-1.2 (should be ~0.3-0.5)
→ softmax becomes near one-hot → model memorizes like a database.

Pass if: logit_std < 0.8 and logit_max < 5.0 for any d_model/vocab combo.
"""

import sys, os, math, torch, torch.nn as nn

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.join(_DIR, ".."))
from mla.model import TransformerLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check_model(d_model, num_layers, num_heads, vocab_size, tag="", num_kv_groups=None):
    if num_kv_groups is None:
        num_kv_groups = 2
        while num_heads % num_kv_groups != 0 and num_kv_groups > 1:
            num_kv_groups -= 1
    head_dim = d_model // num_heads
    m = TransformerLM(
        vocab_size=vocab_size, d_model=d_model, num_layers=num_layers,
        num_heads=num_heads, num_kv_groups=num_kv_groups, head_dim=head_dim,
        use_swiglu=True, use_x0=False, max_seq_len=128,
        residual_dropout=0.0, attn_dropout=0.0, ffn_dropout=0.0,
        use_mla=True, mla_block_size=128,
        mla_d_c=32, mla_d_c1=32, mla_d_rotate=16,
    ).to(device)
    x = torch.randint(0, vocab_size, (4, 64), device=device)
    with torch.no_grad():
        logits = m(x)
        std, mx = logits.std().item(), logits.max().item()
        ok = "PASS" if std < 0.8 and mx < 5.0 else "FAIL"
        print(f"  [{ok}] {tag or f'd_model={d_model} vocab={vocab_size}'}: "
              f"std={std:.3f}  max={mx:.3f}  "
              f"{'(OVERFIT BUG)' if std >= 0.8 else ''}")
        return std < 0.8 and mx < 5.0

def main():
    tests = 0
    passed = 0
    print("=== Init verification (logit over-amplification bug) ===\n")

    # Configs matching train.py (num_kv_groups=4, num_heads=12)
    tests += 1
    if check_model(d_model=768, num_layers=25, num_heads=12, vocab_size=32000,
                   num_kv_groups=4,
                   tag="train.py (d_model=768, 25lay, vocab=32k)"):
        passed += 1

    # Configs matching test_compare.py (num_heads=4, GQA not active)
    tests += 1
    if check_model(d_model=256, num_layers=3, num_heads=4, vocab_size=65,
                   tag="test_compare (d_model=256, 3lay, vocab=65)"):
        passed += 1

    # Edge cases
    tests += 1
    if check_model(d_model=256, num_layers=3, num_heads=4, vocab_size=50000,
                   tag="edge: d_model=256, vocab=50k"):
        passed += 1

    tests += 1
    if check_model(d_model=512, num_layers=6, num_heads=8, vocab_size=32000,
                   num_kv_groups=4,
                   tag="edge: d_model=512, 6lay, vocab=32k"):
        passed += 1

    tests += 1
    if check_model(d_model=1024, num_layers=12, num_heads=16, vocab_size=32000,
                   num_kv_groups=4,
                   tag="edge: d_model=1024, 12lay, vocab=32k"):
        passed += 1

    print(f"\n{'='*50}")
    print(f"Result: {passed}/{tests} passed")
    if passed == tests:
        print("OK — No over-amplification bug detected.")
    else:
        print("BUG DETECTED — embedding or linear init needs N(0,0.02).")
        sys.exit(1)

if __name__ == "__main__":
    main()
