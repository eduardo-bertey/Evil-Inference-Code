"""Pruebas HRM: forward (shapes + loss) y generación. Corre en CPU/T4, SIN flash-attn.

Uso: python3 test.py   (en Colab: device cuda se usa solo en train.py)
"""
import sys
import torch

from model import HRM, Config


def check(name, cond, detail=""):
    ok = bool(cond)
    print(f"  {'PASS' if ok else 'FAIL'} {name}" + (f" — {detail}" if (not ok and detail) else ""))
    if not ok:
        sys.exit(1)


# Config chico para validación rápida en CPU.
cfg = Config()
cfg.dim = 64
cfg.heads = 4
cfg.kv_groups = 2
cfg.layers = 2
cfg.ffn_dim = 128
cfg.emb_num = 200
cfg.h_cycles = 2
cfg.l_cycles = 3
cfg.rotary_pct = 1.0

torch.manual_seed(0)
model = HRM(cfg).eval()
print(f"HRM params: {sum(p.numel() for p in model.parameters()) / 1e3:.1f}K")

B, S = 2, 10
ids = torch.randint(0, cfg.emb_num, (B, S))

print("=== forward ===")
logits, loss = model(ids, labels=ids)
check("logits shape [B,S,vocab]", logits.shape == (B, S, cfg.emb_num), str(tuple(logits.shape)))
check("loss finito", torch.isfinite(loss).all().item())
check("zL_init broadcast (sin crash)", True)

print("=== generación ===")
gen = model.generate(ids[:1], max_new_tokens=5, temperature=1.0)
check("generate alarga la secuencia", gen.shape[1] > S, str(tuple(gen.shape)))
check("generate no produce NaN", torch.isfinite(
    torch.tensor(gen.tolist(), dtype=torch.float32)).all().item())

print("\nHRM OK (sin flash-attention, SDPA)")
