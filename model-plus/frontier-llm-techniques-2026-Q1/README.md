# frontier-llm-techniques-2026-Q1

From-scratch PyTorch implementations of techniques that are at the **frontier of open LLMs as of
2026‑Q1** — each a self-contained, self-checking module. The date is a snapshot: a technique earns
"frontier" by still shipping in the newest models, not by when it was invented.

- **Muon optimizer** — used to train **Kimi K2 / K2.5** (Moonshot, trillion-param MoE, Jan 2026).
- **Multi-Token Prediction** — shipped in **DeepSeek‑V3/V4, Gemma 4, GLM‑5.1, Qwen 3.x** (2026).
- **BPE tokenizer** — included as a **foundational base technique** (universal since GPT‑2), not a
  frontier item; it's here because the other two are studied alongside a tokenizer.

Each module runs on its own with no cross-imports, executes a small self-test on `python <module>.py`,
and asserts that the technique does what it claims.

## Frontier

### Muon optimizer — `muon.py`

Muon orthogonalizes the momentum of 2D weight matrices before each step, pushing the update's singular
values toward 1 via a few Newton-Schulz iterations (no SVD). This spreads the update across directions
instead of letting a few dominate. Non-matrix parameters fall back to a plain momentum step.

```
python muon.py              # self-test, then write the benchmark plot
python muon.py --benchmark  # benchmark only
```

The benchmark trains the same MLP regression task twice — once with AdamW, once with Muon — for ~300
steps each and saves `muon_vs_adamw.png` comparing the two loss curves on the same axes.

**Result** (single RTX 3090): at the same step budget Muon reaches a **lower final loss than AdamW**
(e.g. ≈ **2.03 vs 2.24** on the toy task) — the same orthogonalized-momentum that trains Kimi K2 at
trillion-parameter scale.

<p align="center"><img src="muon_vs_adamw.png" width="520" alt="Muon vs AdamW loss curves" /></p>

### Multi-Token Prediction — `mtp.py`

A normal language-model head predicts the next token (t+1). MTP adds a second head that predicts the
token two ahead (t+2) from the same hidden state, giving a denser training signal (and, at inference,
draft tokens for speculative decoding). The module inlines the small transformer pieces it needs
(RMSNorm, GQA attention, SwiGLU FFN, RoPE) so it has no external dependencies.

```
python mtp.py
```

**Result:** both heads' cross-entropy drops together over training; the t+2 head stays a little higher
(predicting two ahead is genuinely harder) — exactly as expected.

<p align="center"><img src="mtp_losses.png" width="520" alt="MTP — both heads' losses drop" /></p>

## Base technique

### BPE tokenizer — `bpe.py`

Byte-pair encoding starts from the 256 raw bytes and repeatedly merges the most frequent adjacent pair
into a new token, packing common chunks into single tokens for shorter sequences. Foundational rather
than frontier, but included as the from-scratch reference (the "how it works" receipt).

```
python bpe.py
```

**Example output** — the self-test learns merges on an embedded sample, then verifies tokenization is
lossless and measures the win over char-level:

```
learned 256 merges  →  vocab 512
round-trip exact?: True
compression: ~2x  fewer tokens than char-level
sample merges: ['e ', 'th', 's ', 't ', 'ou', ', ', 'd ', 'er']
```

## Artifacts

Every run saves what it produces (and the plots above are committed so they render here):
`muon.py → muon_vs_adamw.png`, `mtp.py → mtp_losses.png`. PNGs are kept; any checkpoints (`.pt`) are gitignored.

## Setup

```
pip install -r requirements.txt
```

---

These modules were factored out of the companion repos **modern-nanoGPT** (dense transformer) and
**nano-moe-mla** (sparse MoE+MLA + measurement). This repo is meant as a series: a new
`frontier-llm-techniques-YYYY-Qn` can snapshot whatever is at the frontier in a later period.
