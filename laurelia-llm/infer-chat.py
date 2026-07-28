"""Interactive chat with KV cache inference."""

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
    print("Chat (Ctrl+C to exit)")
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt:
            continue
        ids = tok.encode(prompt)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(x, max_new_tokens=100, temperature=0.7, top_k=40, top_p=0.9,
                                 repetition_penalty=1.2)
        print(f"Bot: {tok.decode(out[0].tolist(), skip_special_tokens=False)}")


if __name__ == "__main__":
    main()
