"""Laurelia Fine-tune ES — alpaca-cleaned-es (test ~10MB) -> fine-checkpoint.pt -> chat.

Reutiliza el checkpoint base del laurelia (checkpoint.pt) — si no existe local, lo baja de HF.
El resultado se guarda y sube SOLO como `fine-checkpoint.pt` (nunca toca checkpoint.pt).
"""

import sys, os, time, json, random, argparse
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.join(_DIR, ".."))
import torch
from model import LLM, Config
from huggingface import HFManager
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

REPO = "ScortexIA/laurelia"
REV = "filosofia"
DATASET = "pinzhenchen/alpaca-cleaned-es"
DATA_FILE = "alpaca_data_cleaned.es.json"

PROMPT_HEADER = "### Instrucción:\n"
INPUT_HEADER = "\n## Entrada:\n"
RESP_HEADER = "\n### Respuesta:\n"


class BPEWrapper:
    def __init__(self, tok):
        self.tokenizer = tok
        self.vocab_size = tok.get_vocab_size()
    def encode(self, text):
        return self.tokenizer.encode(text).ids
    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)


def chat_prompt(instruction, inp):
    p = PROMPT_HEADER + instruction.strip()
    if inp and inp.strip():
        p += INPUT_HEADER + inp.strip()
    return p + RESP_HEADER


def build_sample(ex, tok, eos_id, block_size):
    inst = ex.get("instruction", "")
    inp = ex.get("input", "")
    out = ex.get("output", "")
    prompt_ids = tok.encode(chat_prompt(inst, inp))
    resp_ids = tok.encode(out)
    resp_full = resp_ids + ([eos_id] if eos_id is not None else [])
    x = (prompt_ids + resp_full)[:block_size]
    y = [-100] * len(x)
    for k, t in enumerate(resp_full):
        pos = len(prompt_ids) - 1 + k
        if 0 <= pos < len(y):
            y[pos] = t
    return x, y


@torch.no_grad()
def generate_sample(model, tokenizer, device, prompt="hola", max_new=40, eos_id=None):
    model.eval()
    x = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens=max_new, temperature=0.7, top_k=40,
                         top_p=0.9, repetition_penalty=1.2, eos_token_id=eos_id)
    model.train()
    return tokenizer.decode(out[0].tolist())


def chat_loop(model, tokenizer, device, eos_id):
    model.eval()
    print("\n=== CHAT (fine-checkpoint) ===")
    while True:
        p = input("\n> ")
        if p.strip().lower() in ("exit", "salir", "quit"):
            break
        prompt_ids = tokenizer.encode(chat_prompt(p, ""))
        x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        out = model.generate(x, max_new_tokens=150, temperature=0.7, top_k=40,
                             top_p=0.9, repetition_penalty=1.2, eos_token_id=eos_id)
        gen = out[0][len(prompt_ids):].tolist()
        print(tokenizer.decode(gen))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tok_path = os.path.join(_DIR, "tokenizer.json")
    if not os.path.exists(tok_path):
        sys.exit("No tokenizer.json found")
    tokenizer = BPEWrapper(Tokenizer.from_file(tok_path))
    eos_id = tokenizer.tokenizer.token_to_id("eos_token")
    print(f"Vocab: {tokenizer.vocab_size}  eos_id: {eos_id}")

    fine_path = os.path.join(_DIR, "fine-checkpoint.pt")

    mode = "e"
    if os.path.exists(fine_path):
        ans = input("¿Entrenar (e) o inferir (i)? [e]: ").strip().lower()
        if ans in ("i", "inferir", "infer", "chat"):
            mode = "i"
        else:
            mode = "e"
    else:
        print("No hay fine-checkpoint.pt; modo entrenar.")

    if mode == "i":
        ckpt = torch.load(fine_path, map_location="cpu")
        config = Config()
        config.emb_num = tokenizer.vocab_size
        model = LLM(config).to(device)
        ckpt["model"].pop("head.emb_weight", None)
        model.load_state_dict(ckpt["model"], strict=False)
        del ckpt
        print(f"Fine-checkpoint cargado: {fine_path}")
        chat_loop(model, tokenizer, device, eos_id)
        return

    prec = input("Precision (n=f32, b=bf16): ").strip().lower()
    dtype = torch.bfloat16 if prec == "b" else torch.float32
    print(f"  Compute: {dtype}")

    hf = HFManager(repo_id=REPO, revision=REV)
    hf._get_token()

    config = Config()
    config.emb_num = tokenizer.vocab_size
    lr = float(sys.argv[sys.argv.index("--lr") + 1]) if "--lr" in sys.argv else 3e-5
    model = LLM(config).to(device).to(dtype)
    optimizer = model.configure_optimizers(0.1, lr, (0.9, 0.95), "cuda")

    ckpt_path = os.path.join(_DIR, "checkpoint.pt")
    if os.path.exists(ckpt_path):
        print(f"checkpoint.pt existe; no se descarga nada -> {ckpt_path}")
    else:
        print("checkpoint.pt no existe; descargando base desde HF...")
        if not hf.download_checkpoint(ckpt_path):
            sys.exit("No se pudo descargar checkpoint.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt["model"].pop("head.emb_weight", None)
    model.load_state_dict(ckpt["model"], strict=False)
    print(f"Base cargada desde {ckpt_path}")

    block_size = config.block_size
    data_path = os.path.join(_DIR, DATA_FILE)
    if not os.path.exists(data_path):
        print(f"Descargando dataset {DATASET}...")
        data_path = hf_hub_download(repo_id=DATASET, filename=DATA_FILE, repo_type="dataset")
        print(f"Dataset en {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    n_examples = int(sys.argv[sys.argv.index("--examples") + 1]) if "--examples" in sys.argv else 15000
    skip = int(sys.argv[sys.argv.index("--skip") + 1]) if "--skip" in sys.argv else 0
    repeats = int(sys.argv[sys.argv.index("--repeats") + 1]) if "--repeats" in sys.argv else 1
    step_cap = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 0
    batch_size = 6
    grad_acc = 6
    print(f"Ejemplos: {n_examples} | skip: {skip} | repeats: {repeats} | bs: {batch_size} | ga: {grad_acc}")

    samples = [build_sample(ex, tokenizer, eos_id, block_size) for ex in data[skip:skip + n_examples]]
    raw = data[skip:skip + n_examples]
    print(f"Muestras preparadas: {len(samples)}")

    model.train()
    step = 0
    stop = False
    asked = False
    ask_step = 30
    t0 = time.time()
    order = list(range(len(samples)))
    random.Random(1).shuffle(order)
    for pass_i in range(1, repeats + 1):
        random.Random(pass_i).shuffle(order)
        mb_count = 0
        for idx in range(0, len(order), batch_size):
            mb = order[idx:idx + batch_size]
            xs, ys = [], []
            max_len = max(len(samples[i][0]) for i in mb)
            for i in mb:
                x, y = samples[i]
                x = x + [0] * (max_len - len(x))
                y = y + [-100] * (max_len - len(y))
                xs.append(x)
                ys.append(y)
            xb = torch.tensor(xs, dtype=torch.long, device=device)
            yb = torch.tensor(ys, dtype=torch.long, device=device)
            logits, loss = model(xb, labels=yb)
            (loss / grad_acc).backward()
            loss_val = loss.item()
            del logits, loss
            mb_count += 1
            if mb_count % grad_acc == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                print(f"s{step} loss {loss_val:.4f} grad {grad_norm:.3f} " +
                      f"{len(samples)*block_size/ (time.time()-t0):.0f}t/s")
                r = raw[(step - 1) % len(raw)]
                print(f"  [{step}] Q: {r.get('instruction','')[:80]!r} -> A: {r.get('output','')[:80]!r}")
                q = r.get("instruction", "")
                gen = generate_sample(model, tokenizer, device, prompt=chat_prompt(q, ""), max_new=40, eos_id=eos_id)
                print(f"  [{step}] GEN: {gen!r}")
                if step_cap and step >= step_cap:
                    stop = True
                    break
                if step == ask_step and not asked:
                    asked = True
                    ans = input(f"\nStep {step}. ¿Continuar entrenando (c) o guardar y chatear (g)? ").strip().lower()
                    if ans in ("g", "chat", "guardar"):
                        stop = True
                        break
        if stop:
            break

    sample = generate_sample(model, tokenizer, device, eos_id=eos_id)
    print(f"  >>> {sample}")

    fine_path = os.path.join(_DIR, "fine-checkpoint.pt")
    state = model.state_dict()
    state.pop("head.emb_weight", None)
    torch.save({"step": step, "model": state}, fine_path)
    print(f"Guardado {fine_path}")
    print(f"Subiendo fine-checkpoint.pt a {REPO}@{REV} ...")
    hf.upload_checkpoint(fine_path, tokenizer_path=tok_path, step=step)

    chat_loop(model, tokenizer, device, eos_id)


if __name__ == "__main__":
    main()