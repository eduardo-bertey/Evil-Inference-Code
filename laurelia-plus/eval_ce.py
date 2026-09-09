"""Eval CE zero-shot del ckpt TST (rev laurelia-plus) sin entrenar.

Prueba: si el CE puro del ckpt TST da ~18 sin un solo update, el pico es
pesos-con-otra-tarea (shift plegado->desplegado), no bug de codigo.
Uso en Colab: !python eval_ce.py  (pide token HF y bloque)
"""
import os
import sys
import torch

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

from tokenizers import Tokenizer
from model import LLM, Config
from huggingface import HFManager
import importlib
train_data = importlib.import_module("train-data")
from train import BPEWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

config = Config()
hf = HFManager(repo_id="ScortexIA/laurelia", revision="laurelia-plus")
hf._get_token()
hf.login_global()

tok_path = os.path.join(_DIR, "tokenizer.json")
if os.path.exists(tok_path):
    tokenizer = BPEWrapper(Tokenizer.from_file(tok_path))
else:
    local_tok = hf.download_tokenizer(tok_path)
    tokenizer = BPEWrapper(Tokenizer.from_file(local_tok))
config.emb_num = tokenizer.vocab_size
print(f"Vocab: {tokenizer.vocab_size}")

model = LLM(config).to(device).to(dtype=torch.float32)
model.eval()

ckpt_path = os.path.join(_DIR, "checkpoint.pt")
if not (os.path.exists(ckpt_path) and os.path.getsize(ckpt_path) > 0):
    ok = hf.download_checkpoint(ckpt_path)
    if not ok:
        sys.exit("no se pudo bajar checkpoint de laurelia-plus")
ckpt = torch.load(ckpt_path, map_location="cpu")
ckpt["model"].pop("head.emb_weight", None)
model.load_state_dict(ckpt["model"], strict=False)
print(f"ckpt: step {ckpt.get('step')} tst {ckpt.get('tst_tokens', 0):,} dense {ckpt.get('dense_tokens', 0):,}")
del ckpt

bi = input("Block: ").strip()
sd = train_data.TrainData(block_idx=int(bi) if bi else 0)
sd.load_tokens(tokenizer)
tokens = sd.get_tokens()
seq_len = config.block_size
n_seq = (len(tokens) - seq_len - 1) // seq_len
print(f"tokens: {len(tokens):,} seqs: {n_seq}")

loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
tot, n = 0.0, 0
with torch.no_grad():
    for i in range(n_seq):
        idx = i * seq_len
        x = torch.tensor([tokens[idx + j] for j in range(seq_len)], dtype=torch.long, device=device).unsqueeze(0)
        y = torch.tensor([tokens[idx + j + 1] for j in range(seq_len)], dtype=torch.long, device=device).unsqueeze(0)
        logits, _, _ = model(x, labels=None, fold=1)
        l = loss_fct(logits.view(-1, logits.size(-1)), y.view(-1)).item()
        tot += l
        n += 1
        if n % 50 == 0:
            print(f"  seq {n}/{n_seq} ce={l:.4f} media={tot / n:.4f}")
print(f"CE ZERO-SHOT media bloque: {tot / max(n, 1):.4f} (aleatorio ~{__import__('math').log(tokenizer.vocab_size):.2f})")
