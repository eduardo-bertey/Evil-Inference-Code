"""Chat K2 persistente para Colab: el modelo queda cargado entre celdas.
Celda 1:  from k2.chat import ask  (carga 1 vez, ~2.2GB se quedan en VRAM)
Celda 2+: ask("tu pregunta")  (reusa, sin recargar)
"""
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "IFM/K2-Horizon-0.9B"
_model = None
_tok = None
_history = []


def load():
    global _model, _tok
    if _model is not None:
        return _model, _tok
    assert torch.cuda.is_available(), "sin CUDA"
    import transformers
    print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | transformers {transformers.__version__} (tarjeta valida 5.15.0+torch 2.13.0)")
    _tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    try:
        _model = AutoModelForCausalLM.from_pretrained(
            BASE, device_map="cuda:0", dtype=torch.bfloat16,
            low_cpu_mem_usage=True, trust_remote_code=True,
            attn_implementation="sdpa",  # T4: mem-efficient > eager (FA2 no existe en sm75)
        )
        print("attn: sdpa")
    except Exception as e:
        print(f"attn sdpa no soportado ({e}), eager")
        _model = AutoModelForCausalLM.from_pretrained(
            BASE, device_map="cuda:0", dtype=torch.bfloat16,
            low_cpu_mem_usage=True, trust_remote_code=True,
        )
    _model.eval()
    _model.config.use_cache = True
    try:
        _model = torch.compile(_model, mode="reduce-overhead")
        print("torch.compile: on")
    except Exception as e:
        print(f"torch.compile no ({e}), sigo sin compilar")
    print(f"VRAM: {torch.cuda.memory_allocated() / 1e9:.2f}GB (queda cargado)")
    return _model, _tok


def ask(msg, max_new=2048, reset=False, effort="high"):
    global _history
    model, tok = load()
    if reset:
        _history = []
    _history.append({"role": "user", "content": msg})
    if effort == "off":
        # Tarjeta HF: para generate plano, parsers off (sin reasoning).
        text = tok.apply_chat_template(_history, tokenize=False, add_generation_prompt=True)
        temp = 1.0
    else:
        text = tok.apply_chat_template(_history, tokenize=False, add_generation_prompt=True,
                                       chat_template_kwargs={"reasoning_effort": effort})
        temp = 0.6
    inp = tok(text, return_tensors="pt").to(model.device)
    inp.pop("token_type_ids", None)
    import time
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new, temperature=temp,
                             top_p=0.95, do_sample=True,
                             repetition_penalty=1.1, use_cache=True)
    dt = time.time() - t0
    new_toks = out[0].shape[0] - inp["input_ids"].shape[1]
    ans = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=False)
    # Historial truncado: el pensamiento crudo no vuelve entero (retroalimenta loops).
    _history.append({"role": "assistant", "content": ans[-1500:]})
    print(ans)
    print(f"[{new_toks} toks en {dt:.1f}s = {new_toks / max(dt, 1e-3):.1f} t/s | VRAM: {torch.cuda.memory_allocated() / 1e9:.2f}GB]")
    return ans


def reset():
    global _history
    _history = []
    print("historial limpio (modelo sigue en VRAM)")
