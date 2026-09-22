"""Full fine-tune K2-Horizon-0.9B sobre rebuild/ (wiki 3MB + fine 2MB + tuit 1MB por bloque).

Uso Colab:
  !python rebuild/train.py
Pide: repo HF destino, rama, bloque inicial, steps, lr.
Guarda modelo + tokenizer + .py del modelo base y sube a tu repo.

Requiere: transformers, huggingface_hub, torch. Sin PEFT/LoRA: entrena todos los pesos.
"""

import importlib
import json
import os
import sys
import time

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
train_data = importlib.import_module("train-data")
import prism

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, create_repo, upload_folder, snapshot_download
import mose
from k2_mose import ensure_k2_mose

BASE = "IFM/K2-Horizon-0.9B"
SEQ = 2048
BS = 2
GA = 8


class TokWrap:
    def __init__(self, tok):
        self.tok = tok
        self.vocab_size = tok.vocab_size

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False)


def get_token(repo):
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if tok:
        print("HF token desde entorno")
        return tok
    print(f"\nNo HF token found. Enter token for {repo} (write access):")
    import getpass
    tok = getpass.getpass("Token: ").strip()
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

    # Un solo modelo: base y finetune viven en el mismo repo@branch.
    repo, rev = "ScortexIA/laurelia", "k2"
    print(f"Modelo unico: {repo}@{rev}")
    token = get_token(repo)
    api = HfApi()

    print(f"Verificando K2-MoSE en {repo}@{rev}...")
    model, hf_tok = ensure_k2_mose(
        repo=repo, branch=rev, token=token,
        build_dir=os.path.join(_DIR, "k2_mose_build"))
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    # Estado persistente: si se reinicia, sigue donde quedo.
    state_path = os.path.join(_DIR, "rebuild_state.json")
    saved = {"block": 0, "step": 0}
    if os.path.exists(state_path):
        try:
            with open(state_path) as f:
                saved.update(json.load(f))
            print(f"Estado previo: bloque {saved['block']} step {saved['step']}")
        except Exception as e:
            print(f"  estado ilegible ({e}), arranco en 0")

    bi = input(f"Block [{saved['block']}]: ").strip()
    sd = train_data.TrainData(block_idx=int(bi) if bi else saved["block"])
    sd.load_tokens(TokWrap(hf_tok))

    k_in = input("Capas PRiSM k (1 por bloque contiguo) [4]: ").strip()
    k_sel = int(k_in) if k_in else 4
    lr_in = input("lr [2e-5]: ").strip()
    lr_val = float(lr_in) if lr_in else 2e-5

    def save_state():
        with open(state_path, "w") as f:
            json.dump({"block": sd.block_idx, "step": step}, f)

    def setup_block():
        """PRiSM de nuevo en cada bloque + optimizer nuevo.

        Libera grads viejos y estado Adam anterior (si no, la VRAM se llena).
        """
        for p in model.parameters():
            p.grad = None
        mose.set_force_full(model, True)  # seleccion sobre denso exacto
        selected = prism.run_prism_selection(
            model, sd.get_tokens(), k_sel, SEQ, device, hf_tok.pad_token_id)
        mose.set_force_full(model, False)
        prism.freeze_except_layers(model, selected)
        n_params = prism.trainable_parameter_count(model)
        print(f"PRiSM bloque {sd.block_idx}: entrenan capas {selected} "
              f"(+embed/norms/head) | {n_params:,} params")
        new_opt = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad), lr=lr_val)
        new_opt.zero_grad()
        return new_opt

    def advance_block(old_opt):
        del old_opt
        sd.next_block()
        save_state()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return setup_block()

    step = int(saved.get("step", 0))
    opt = setup_block()
    print("Optimizer: AdamW (solo entrenables, nuevo por bloque)")
    save_state()

    run_steps = int(input("Steps [500]: ").strip() or 500)
    max_steps = step + run_steps
    epoch = 0
    aux_log = 0.0
    aux_r_log = 0.0
    t0 = time.time()
    model.train()
    print("Entrenando... (el primer step tarda, CUDA calienta)", flush=True)
    while step < max_steps:
        tokens = sd.get_tokens()
        n_seq = (len(tokens) - SEQ - 1) // SEQ
        if n_seq <= 0:
            epoch += 1
            opt = advance_block(opt)
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
            # Eq.(6): dos forwards por microbatch (router + full), promedio.
            # Un backward por forward: mitad de pico de memoria. /GA acumula.
            mose.set_force_full(model, False)
            out_r = model(input_ids=x, labels=y)
            aux_r = mose.collect_aux_loss(model)
            (0.5 * (out_r.loss + aux_r) / GA).backward()
            loss_r = float(out_r.loss.detach())
            aux_r_log = float(aux_r.detach()) if torch.is_tensor(aux_r) else float(aux_r)
            del out_r, aux_r
            mose.set_force_full(model, True)
            out_f = model(input_ids=x, labels=y)
            aux_f = mose.collect_aux_loss(model)
            mose.set_force_full(model, False)
            (0.5 * (out_f.loss + aux_f) / GA).backward()
            loss_f = float(out_f.loss.detach())
            aux_log = float(aux_f.detach()) if torch.is_tensor(aux_f) else float(aux_f)
            del out_f, aux_f
            # Escalares para el log (promedio Eq.6, sin grafo)
            loss = torch.tensor(0.5 * (loss_r + loss_f))
            if ((bs0 // BS + 1) % GA == 0) or be >= n_seq:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                step += 1
                if step == 1 or step % 10 == 0:
                    dt = time.time() - t0
                    print(f"s{step} loss {loss.item():.4f} "
                          f"mose_aux r={aux_r_log:.5f} f={aux_log:.5f} "
                          f"{BS * GA * SEQ / max(dt, 1e-3):.0f}t/s bloque {sd.block_idx}")
                    t0 = time.time()
        epoch += 1
        opt = advance_block(opt)

    save_state()
    # Guardar pesos full + subir.
    print("Guardando modelo...")
    out_dir = os.path.join(_DIR, "k2_out")
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    hf_tok.save_pretrained(out_dir)
    # .pt estilo moe-flash: pesos + step/epoch/block adentro.
    torch.save({"model": model.state_dict(), "step": step,
                "epoch": epoch, "block": sd.block_idx},
               os.path.join(out_dir, "k2_checkpoint.pt"))
    print(f"  k2_checkpoint.pt: step {step} epoch {epoch} block {sd.block_idx}")
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
