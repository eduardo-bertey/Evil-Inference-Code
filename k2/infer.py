"""Chat K2-Horizon-0.9B (IFM). Loop interactivo en GPU.
Uso Colab: !python k2/infer.py [--max-new 2048]
Escribí tu mensaje y Enter. 'salir' para terminar.
"""
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "IFM/K2-Horizon-0.9B"
max_new = 2048
for a in sys.argv[1:]:
    if a.startswith("--max-new"):
        max_new = int(a.split("=")[1])

assert torch.cuda.is_available(), "sin CUDA: este script exige GPU"
print(f"GPU: {torch.cuda.get_device_name(0)}")

tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE, device_map="cuda:0", dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True,
)
model.eval()
print(f"Modelo en: {model.device} | VRAM: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
print("Chat listo (salir = terminar).")

history = []
while True:
    try:
        msg = input("\nvos> ").strip()
    except EOFError:
        break
    if msg.lower() in ("salir", "exit", "quit"):
        break
    if not msg:
        continue
    history.append({"role": "user", "content": msg})
    text = tok.apply_chat_template(history, tokenize=False, add_generation_prompt=True,
                                   chat_template_kwargs={"reasoning_effort": "high"})
    inp = tok(text, return_tensors="pt").to(model.device)
    inp.pop("token_type_ids", None)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new, temperature=0.6,
                             top_p=0.95, do_sample=True,
                             repetition_penalty=1.1, no_repeat_ngram_size=16)
    gen = out[0][inp["input_ids"].shape[1]:]
    ans = tok.decode(gen, skip_special_tokens=False)
    print(f"\nk2> {ans}")
    print(f"[VRAM: {torch.cuda.memory_allocated() / 1e9:.2f}GB]")
    history.append({"role": "assistant", "content": ans[-1500:]})
