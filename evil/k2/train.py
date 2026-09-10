"""Fine-tune K2-Horizon-0.9B con LoFT (LoRA + LoFTAdamW) sobre datos laurelia-plus.

Uso Colab:
  !python train.py
Pide: repo HF destino, rama, bloque inicial, steps, lr.
Guarda f16 + tokenizer + .py del modelo base y sube a tu repo (estilo laurelia-llm).

Requiere: transformers, peft, huggingface_hub, torch.
LoFT vive en ../loft/loft_optim. Datos: ../../Evil-Inference-Code/laurelia-plus.
"""
import os
import sys
import time

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, "..", "loft"))
sys.path.insert(0, os.path.join(_DIR, "..", "..", "Evil-Inference-Code", "laurelia-plus"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from huggingface_hub import HfApi, create_repo, upload_folder, snapshot_download

from loft_optim.optimizer import LoFTAdamW
import importlib
train_data = importlib.import_module("train-data")

BASE = "IFM/K2-Horizon-0.9B"
SEQ = 2048
BS = 2
GA = 8
RANK = 8  # LoFT exige rank == alpha (sin scaling)
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class TokWrap:
    def __init__(self, tok):
        self.tok = tok
        self.vocab_size = tok.vocab_size

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False)


def get_token(repo):
    tok = os.environ.get("HF_TOKEN")
    if tok:
        print("HF token desde entorno")
        return tok
    print(f"\nNo HF token found. Enter token for {repo} (write access):")
    tok = input("Token: ").strip()
    try:
        from huggingface_hub import login
        login(token=tok)
        print("HF token aplicado globalmente para descargas")
    except Exception as e:
        print(f"  login fallo ({e}), sigo con token en memoria")
    return tok


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    repo = input("Repo HF destino (ej. ScortexIA/k2-es): ").strip()
    rev = input("Rama [k2-loft]: ").strip() or "k2-loft"
    token = get_token(repo)
    api = HfApi()

    print(f"Cargando {BASE}...")
    hf_tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if hf_tok.pad_token is None:
        hf_tok.pad_token = hf_tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE, device_map="auto", dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    cfg = LoraConfig(r=RANK, lora_alpha=RANK, target_modules=TARGETS,
                     lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    # Optimizer LoFT (receta del notebook hf_implementation).
    for g in model.parameters():
        pass
    opt = LoFTAdamW(
        model.parameters(), lr=1e-4, weight_decay=0.0, eps=1e-4,
        betas=(0.9, 0.999), model=model,
        lora_A_name="lora_A", lora_B_name="lora_B",
        alternate_update=True, rescale_grads=True,
        reproject_momentum=True, reproject_second_moment=True,
    )
    lr_in = input("lr [1e-4]: ").strip()
    if lr_in:
        for g in opt.param_groups:
            g["lr"] = float(lr_in)

    bi = input("Block [0]: ").strip()
    sd = train_data.TrainData(block_idx=int(bi) if bi else 0)
    sd.load_tokens(TokWrap(hf_tok))

    max_steps = int(input("Steps [500]: ").strip() or 500)
    step = 0
    epoch = 0
    t0 = time.time()
    model.train()
    while step < max_steps:
        tokens = sd.get_tokens()
        n_seq = (len(tokens) - SEQ - 1) // SEQ
        if n_seq <= 0:
            epoch += 1
            sd.next_block()
            continue
        for bs0 in range(0, n_seq, BS):
            if step >= max_steps:
                break
            be = min(bs0 + BS, n_seq)
            xs, ys = [], []
            for i in range(bs0, be):
                idx = i * SEQ
                xs.append(torch.tensor(tokens[idx:idx + SEQ], dtype=torch.long))
                ys.append(torch.tensor(tokens[idx + 1:idx + 1 + SEQ], dtype=torch.long))
            x = torch.stack(xs).to(device)
            y = torch.stack(ys).to(device)
            out = model(input_ids=x, labels=y)
            (out.loss / GA).backward()
            if ((bs0 // BS + 1) % GA == 0) or be >= n_seq:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                step += 1
                if step % 10 == 0:
                    dt = time.time() - t0
                    print(f"s{step} loss {out.loss.item():.4f} "
                          f"{BS * GA * SEQ / max(dt, 1e-3):.0f}t/s bloque {sd.block_idx}")
                    t0 = time.time()
        epoch += 1
        sd.next_block()

    # Merge LoRA -> f16 -> subir (estilo laurelia-llm).
    print("Merge LoRA + f16...")
    merged = model.merge_and_unload().half()
    out_dir = os.path.join(_DIR, "k2_out")
    os.makedirs(out_dir, exist_ok=True)
    merged.save_pretrained(out_dir)
    hf_tok.save_pretrained(out_dir)
    # .py del modelo base (custom_code): sin esto el repo no carga.
    try:
        base_files = snapshot_download(repo_id=BASE, allow_patterns=["*.py"],
                                       local_dir=out_dir)
        print(f"  .py base: {base_files}")
    except Exception as e:
        print(f"  no se pudieron bajar .py base: {e}")
    create_repo(repo_id=repo, exist_ok=True, private=False, token=token)
    try:
        api.create_branch(repo_id=repo, branch=rev, token=token)
    except Exception:
        pass
    upload_folder(repo_id=repo, folder_path=out_dir, revision=rev, token=token)
    print(f"Subido a {repo}@{rev}")


if __name__ == "__main__":
    main()
