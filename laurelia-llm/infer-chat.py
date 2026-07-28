"""Interactive chat with KV cache inference.

Comandos:
  /temp <float>   — temperature (default 0.7)
  /topk <int>     — top-k (default 40)
  /topp <float>   — top-p (default 0.9)
  /rep <float>    — repetition penalty (default 1.2)
  /ctx <int>      — max context/cache length (default 1024)
  /length <int>   — max tokens a generar (default 200)
  /reset          — limpiar cache
  /exit           — salir
"""

import sys, os
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
import torch
from model import LLM, Config
from tokenizers import Tokenizer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config()

    tok_path = os.path.join(_DIR, "tokenizer.json")
    if not os.path.exists(tok_path):
        sys.exit("No tokenizer found. Train first.")
    tok = Tokenizer.from_file(tok_path)
    config.emb_num = tok.get_vocab_size()
    eos_id = tok.token_to_id("eos_token")

    model = LLM(config).to(device)

    ckpt_path = os.path.join(_DIR, "checkpoint.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        ckpt["model"].pop("head.emb_weight", None)
        model.load_state_dict(ckpt["model"], strict=False)
        step = ckpt.get("step", 0)
        print(f"Loaded checkpoint step {step}")
        del ckpt
    else:
        sys.exit("No checkpoint found.")

    model.eval()

    temperature = 0.7
    top_k = 40
    top_p = 0.9
    repetition_penalty = 1.2
    max_new = 200
    max_ctx = 1024

    print(f"Chat (Ctrl+C to exit)")
    print(f"  temp={temperature} top_k={top_k} top_p={top_p} rep={repetition_penalty} ctx={max_ctx} length={max_new}")
    print(f"  /temp /topk /topp /rep /ctx /reset /exit")

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not prompt:
            continue

        if prompt.startswith("/"):
            parts = prompt.split()
            cmd = parts[0].lower()
            val = parts[1] if len(parts) > 1 else None

            if cmd == "/exit":
                break
            elif cmd == "/temp" and val:
                temperature = float(val)
                print(f"  temperature = {temperature}")
            elif cmd == "/topk" and val:
                top_k = int(val)
                print(f"  top_k = {top_k}")
            elif cmd == "/topp" and val:
                top_p = float(val)
                print(f"  top_p = {top_p}")
            elif cmd == "/rep" and val:
                repetition_penalty = float(val)
                print(f"  repetition_penalty = {repetition_penalty}")
            elif cmd == "/ctx" and val:
                max_ctx = int(val)
                print(f"  max_ctx = {max_ctx}")
            elif cmd in ("/length", "/len") and val:
                max_new = int(val)
                print(f"  max_new_tokens = {max_new}")
            elif cmd == "/reset":
                print("  Cache cleared (restart script to fully reset)")
            else:
                print(f"  Unknown: {cmd}")
            continue

        ids = tok.encode(prompt).ids
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(x, max_new_tokens=max_new, temperature=temperature,
                                 top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
                                 eos_token_id=eos_id)
        print(f"Bot: {tok.decode(out[0].tolist(), skip_special_tokens=False)}")


if __name__ == "__main__":
    main()
