import sys, os, time, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from model import TransformerLM
from tokenizers import Tokenizer

print("start")
tok = Tokenizer.from_file(os.path.join(os.path.dirname(__file__), "tokenizer.json"))
print("tok", tok.get_vocab_size())

with open(r"C:\Users\Emabe\Documents\GitHub\xlstm\rust\input.txt", encoding="utf-8") as f:
    tokens = tok.encode(f.read()).ids
print("tokens", len(tokens))
tokens = torch.tensor(tokens, dtype=torch.long)

model = TransformerLM(vocab_size=tok.get_vocab_size(), d_model=128, num_layers=3,
    num_heads=8, num_kv_groups=4, use_swiglu=True, use_x0=True,
    max_seq_len=512, residual_dropout=0.0, attn_dropout=0.0, ffn_dropout=0.0,
    mla_block_size=128)
print("model", sum(p.numel() for p in model.parameters()))

opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
model.train()

for step in range(10):
    i = step * 8 * 512
    x = tokens[i:i+512].unsqueeze(0)
    y = tokens[i+1:i+513].unsqueeze(0)
    opt.zero_grad()
    logits = model.forward_train_partial_rope(x, rotary_pct=0.25)
    loss = F.cross_entropy(logits.view(-1, tok.get_vocab_size()), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    print(f"step {step} loss {loss.item():.4f}")
    sys.stdout.flush()

print("DONE")
