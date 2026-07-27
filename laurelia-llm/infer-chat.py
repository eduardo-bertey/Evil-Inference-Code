"""Infer-Chat: Chat interactivo con cache persistente.

Comandos:
  /temp <valor>     - Cambiar temperatura (default: 0.8)
  /topk <valor>     - Cambiar top_k (default: 50)
  /topp <valor>     - Cambiar top_p (default: 0.9)
  /rep <valor>      - Cambiar repetition_penalty (default: 1.1)
  /ctx <valor>      - Max tokens de contexto para cache (default: 2048)
  /gen <valor>      - Max tokens a generar por respuesta (default: 200)
  /clear            - Limpiar cache (nueva conversación)
  /stats            - Mostrar configuración actual
  /help             - Mostrar ayuda
  /salir            - Salir
"""

import sys, os, time, torch

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.join(_DIR, ".."))

from model import TransformerLM
from tokenizers import Tokenizer


class ChatConfig:
    def __init__(self):
        self.temperature = 0.8
        self.top_k = 50
        self.top_p = 0.9
        self.rep_penalty = 1.1
        self.max_ctx = 2048
        self.max_gen = 200

    def show(self):
        print(f"  temp={self.temperature} top_k={self.top_k} top_p={self.top_p} "
              f"rep={self.rep_penalty} ctx={self.max_ctx} gen={self.max_gen}")


def load_model(device):
    tok_path = os.path.join(_DIR, "tokenizer.json")
    ckpt_path = os.path.join(_DIR, "checkpoint.pt")

    tok = Tokenizer.from_file(tok_path)
    vocab_size = tok.get_vocab_size()

    model = TransformerLM(
        vocab_size=vocab_size, d_model=768, num_layers=24,
        num_heads=12, num_kv_groups=4,
        use_swiglu=True, max_seq_len=2048,
        attn_logit_cap=30, use_xsa=True, qk_norm=True,
        use_sandwich_norm=True, use_mla=False, cache_every=1,
    )

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        ckpt["model"].pop("head.emb_weight", None)
        model.load_state_dict(ckpt["model"], strict=False)
        step = ckpt.get("step", 0)
        print(f"  Checkpoint: step {step}")
        del ckpt
    else:
        print("  No checkpoint found, random weights")

    return model.to(device), tok


def parse_command(line):
    parts = line.strip().split()
    if not parts or not parts[0].startswith("/"):
        return None, []
    cmd = parts[0].lower()
    args = parts[1:]
    return cmd, args


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, tokenizer = load_model(device)
    model.eval()

    cfg = ChatConfig()
    caches = None
    prompt_len = 0
    history_tokens = []

    print("\n── Infer-Chat ──")
    print("  Escribe /help para comandos")
    cfg.show()
    print()

    with torch.no_grad():
        while True:
            try:
                user_input = input("Tú: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nAdiós!")
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                cmd, args = parse_command(user_input)

                if cmd == "/help":
                    print(__doc__)

                elif cmd == "/temp" and args:
                    cfg.temperature = float(args[0])
                    print(f"  temp={cfg.temperature}")

                elif cmd == "/topk" and args:
                    cfg.top_k = int(args[0])
                    print(f"  top_k={cfg.top_k}")

                elif cmd == "/topp" and args:
                    cfg.top_p = float(args[0])
                    print(f"  top_p={cfg.top_p}")

                elif cmd == "/rep" and args:
                    cfg.rep_penalty = float(args[0])
                    print(f"  rep={cfg.rep_penalty}")

                elif cmd == "/ctx" and args:
                    cfg.max_ctx = int(args[0])
                    print(f"  ctx={cfg.max_ctx}")

                elif cmd == "/gen" and args:
                    cfg.max_gen = int(args[0])
                    print(f"  gen={cfg.max_gen}")

                elif cmd == "/clear":
                    caches = None
                    prompt_len = 0
                    history_tokens = []
                    print("  Cache limpiada")

                elif cmd == "/stats":
                    cfg.show()
                    n_tok = len(history_tokens)
                    print(f"  tokens en cache: {n_tok}")

                elif cmd == "/salir":
                    print("Adiós!")
                    break

                else:
                    print(f"  Comando desconocido: {cmd}")
                continue

            tokens = tokenizer.encode(user_input)
            history_tokens.extend(tokens)

            if len(history_tokens) > cfg.max_ctx:
                overflow = len(history_tokens) - cfg.max_ctx
                history_tokens = history_tokens[overflow:]
                caches = None
                prompt_len = 0

            x = torch.tensor([tokens], dtype=torch.long, device=device)

            if caches is None:
                prompt_len = len(history_tokens)
                for i in range(prompt_len):
                    inp = torch.tensor([[history_tokens[i]]], dtype=torch.long, device=device)
                    _, caches = model.forward_with_cache(inp, i, caches)
            else:
                offset = prompt_len
                for i, tok_id in enumerate(tokens):
                    inp = torch.tensor([[tok_id]], dtype=torch.long, device=device)
                    _, caches = model.forward_with_cache(inp, offset + i, caches)
                prompt_len = len(history_tokens)

            gen_start = time.time()
            generated = []

            last_logits = None
            for step_i in range(cfg.max_gen):
                if last_logits is None:
                    inp = torch.tensor([[history_tokens[-1]]], dtype=torch.long, device=device)
                    logits, caches = model.forward_with_cache(inp, prompt_len - 1, caches)
                else:
                    inp = torch.tensor([[generated[-1]]], dtype=torch.long, device=device)
                    logits, caches = model.forward_with_cache(inp, prompt_len + step_i, caches)

                last_logits = logits[:, -1, :] / cfg.temperature

                if cfg.rep_penalty != 1.0:
                    all_prev = torch.tensor(history_tokens + generated, device=device)
                    for tok in all_prev.unique():
                        if last_logits[0, tok] > 0:
                            last_logits[0, tok] /= cfg.rep_penalty
                        else:
                            last_logits[0, tok] *= cfg.rep_penalty

                if cfg.top_k > 0:
                    v, _ = torch.topk(last_logits, min(cfg.top_k, last_logits.size(-1)))
                    last_logits[last_logits < v[:, [-1]]] = float("-inf")

                if cfg.top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(last_logits, descending=True)
                    cumprobs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    mask = cumprobs - torch.softmax(sorted_logits, dim=-1) >= cfg.top_p
                    sorted_logits[mask] = float("-inf")
                    last_logits.scatter_(1, sorted_idx, sorted_logits)

                probs = torch.softmax(last_logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1).item()
                generated.append(next_tok)
                history_tokens.append(next_tok)
                prompt_len = len(history_tokens)

                if next_tok == tokenizer.token_to_id("[EOS]") if hasattr(tokenizer, 'token_to_id') else False:
                    break

            gen_time = time.time() - gen_start
            text = tokenizer.decode(generated)
            tps = len(generated) / max(gen_time, 0.001)
            print(f"Bot: {text}  [{tps:.0f} tok/s]\n")


if __name__ == "__main__":
    main()
