"""Inferencia rápida K2-Horizon-0.9B (IFM).
Uso Colab: !python infer.py "tu pregunta" [--max-new 512]
"""
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "IFM/K2-Horizon-0.9B"

prompt = sys.argv[1] if len(sys.argv) > 1 else "¿Quién eres?"
max_new = 512
for a in sys.argv[2:]:
    if a.startswith("--max-new"):
        max_new = int(a.split("=")[1])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE, device_map="auto", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True,
)
model.eval()

msgs = [{"role": "user", "content": prompt}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                               chat_template_kwargs={"reasoning_effort": "high"})
inp = tok(text, return_tensors="pt").to(model.device)
inp.pop("token_type_ids", None)
with torch.no_grad():
    out = model.generate(**inp, max_new_tokens=max_new, temperature=0.6,
                         top_p=0.95, do_sample=True)
gen = out[0][inp["input_ids"].shape[1]:]
print(tok.decode(gen, skip_special_tokens=False))
