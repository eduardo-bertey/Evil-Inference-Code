"""k2moe — fine-tune top-down por bloques de capas sobre FineWeb (para Colab).

Ejecuta:
  !python k2moe/train.py

Pregunta solo lo básico: token, precisión, cuántas capas
entrenar por turno, salto de bloque (loss por tiempo o steps), lr, steps y
block de datos. Repo y rama son fijos: ScortexIA/laurelia@k2moe.

Entrena SOLO el bloque activo (N capas, de arriba hacia abajo). Todo lo demás
queda congelado: otras capas (densas o MoE ya convertidas), embeddings, norms
y lm_head. Con d2dmoe progresivo, el bloque que entra se convierte dense→MoE
en ese momento y se entrenan sus routers (resto congelado). Al agotar todos
los bloques el modelo queda completo.

Checkpoints en HF (modelo + optimizer + estado): si ya existe uno (local o
en el repo/rama) te pregunta si quieres continuar desde ahí.

Archivos del proyecto:
  train.py      — este entry (prompts + loop)
  layers.py     — freeze top-down por bloques
  fineweb.py    — FineWeb2-HQ (spa_Latn) en bloques
  hf.py         — token/repo HF, uploads/downloads
  plot.py       — train_history.json + plot_train_step_N.png
  checkpoint.py — k2moe_checkpoint.pt + k2moe_state.json, push a HF
  d2dmoe.py     — núcleo D2DMoE (MIT) para convertir capas a MoE
"""

import os
import sys
import time
from collections import deque

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

import torch

import checkpoint
import fineweb
import hf
import layers as layerlib
import plot

BASE = "IFM/K2-Horizon-0.9B"
REPO = "ScortexIA/laurelia"  # repo único del proyecto (como rebuild/); cambia solo esto
BRANCH = "k2moe"              # rama fija
EVAL_EVERY = 20        # cada cuantos steps se mira el avg_loss
LOSS_WINDOW = 20       # ventana para el avg_loss
SAVE_MINUTES = 20.0    # cada cuantos minutos sube checkpoint a HF


class TokWrap:
    def __init__(self, tok):
        self.tok = tok
        self.vocab_size = tok.vocab_size

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    repo = REPO
    branch = BRANCH
    token = hf.hf_token(repo)
    api = hf.hf_api(token)
    hf.hf_ensure_repo_branch(api, repo, branch, token)

    prec = input("Precision (n=f32, b=bf16) [b]: ").strip().lower() or "b"
    dtype = torch.bfloat16 if prec == "b" else torch.float32

    layers_per_block = int(input("Cuantas capas entrenar por turno (desde arriba) [2]: ").strip() or 2)
    salto = input("Avance de bloque (l=loss por tiempo, s=steps) [l]: ").strip().lower() or "l"
    if salto == "s":
        steps_per_block = int(input("Steps por bloque [200]: ").strip() or 200)
        block_minutes, loss_delta = None, None
        print(f"Avance por steps: cada {steps_per_block} steps")
    else:
        block_minutes = float(input("Minutos por bloque antes de saltar [10]: ").strip() or 10)
        loss_delta = float(input("Caida de avg_loss para saltar antes [0.05]: ").strip() or 0.05)
        steps_per_block = None
        print(f"Avance por loss: {block_minutes} min por bloque o caida de {loss_delta}")

    seq_len = int(input("Seq len [1024]: ").strip() or 1024)
    bs = int(input("Batch size [1]: ").strip() or 1)
    ga = int(input("Grad accum [8]: ").strip() or 8)
    lr = float(input("lr [2e-5]: ").strip() or 2e-5)
    max_steps = int(input("Steps [1000]: ").strip() or 1000)

    # ¿checkpoint previo?
    state = checkpoint.default_state()
    ck = checkpoint.load_checkpoint()
    if ck is None:
        ck = None
        if checkpoint.download_checkpoint(repo, branch, token):
            ck = checkpoint.load_checkpoint()
    if ck is not None:
        print(f"\nCheckpoint encontrado: step {ck.get('step')} | "
              f"bloque datos {ck.get('block_idx')} | bloque capas {ck.get('layer_block_idx')}")
        cont = input("¿Continuar desde ahí? [s/n] [s]: ").strip().lower() or "s"
        if cont == "s":
            state.update({k: ck.get(k, state[k]) for k in
                          ("step", "block_idx", "layer_block_idx",
                           "steps_in_block", "block_base_avg",
                           "block_start_time", "arch", "done")})
        else:
            ck = None
            print("Empezando de cero (el checkpoint se sobrescribirá al guardar).")

    if state["done"]:
        print("Este checkpoint ya completó todos los bloques.")
        return

    # ¿d2dmoe progresivo? (la receta viaja guardada en el checkpoint; al
    # retomar se reaplican las MISMAS conversiones antes de cargar pesos)
    if state.get("arch") is None:
        resp = input("¿Aplicar d2dmoe progresivo (convertir a MoE bloque a bloque)? [n]: ").strip().lower() or "n"
        if resp == "s":
            n_exp = int(input("Expertos por capa [32]: ").strip() or 32)
            swap = input("¿Cambiar SiLU/GELU -> ReLU en las capas convertidas? [s]: ").strip().lower() or "s"
            r_steps = int(input("Steps de router por bloque convertido (0 = no entrenar routers) [500]: ").strip() or 500)
            r_lr = float(input("lr router [1e-3]: ").strip() or 1e-3)
            r_width = int(input("Ancho del router [512]: ").strip() or 512)
            tau = float(input("Tau dynk para inferencia final [0.9]: ").strip() or 0.9)
            state["arch"] = {"moe": True, "num_experts": n_exp,
                             "swap_relu": swap == "s", "converted_blocks": 0,
                             "router_steps": r_steps, "router_lr": r_lr,
                             "router_width": r_width, "dynk_tau": tau}
    arch = state.get("arch")
    if arch:
        print(f"d2dmoe progresivo: {arch['num_experts']} expertos/capa | "
              f"swap_relu={arch['swap_relu']} | bloques convertidos: {arch['converted_blocks']}")

    bi = input(f"Block de datos [{state['block_idx']}]: ").strip()
    state["block_idx"] = int(bi) if bi else state["block_idx"]

    # modelo
    print(f"\nCargando tokenizer {BASE}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"Cargando {BASE}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=dtype, low_cpu_mem_usage=True,
        device_map="auto", trust_remote_code=True)
    model.config.use_cache = False

    layers = layerlib.get_layers(model)
    blocks = layerlib.top_down_blocks(len(layers), layers_per_block)
    print(f"{len(layers)} capas | bloques top-down: {blocks}")

    # d2dmoe (dense -> MoE) bloque a bloque: reaplica conversiones al retomar,
    # o convierte el bloque superior en un run nuevo
    arch = state.get("arch")

    def convert_block(k, split_weights):
        import d2dmoe
        if arch["swap_relu"]:
            d2dmoe.swap_ffn_activation_to_relu(model, blocks[k])
        d2dmoe.replace_with_moes(model, arch["num_experts"], blocks[k],
                                 split_weights=split_weights)
        arch["converted_blocks"] = k + 1

    def train_block_routers(k):
        if not arch or arch.get("router_steps", 0) <= 0:
            return
        import d2dmoe
        print(f"Entrenando routers del bloque {k + 1} ({arch['router_steps']} steps)...")
        d2dmoe.add_routers(model, layer_ids=blocks[k], width=arch["router_width"])
        d2dmoe.train_routers(model, sd.get_tokens(), seq_len, device,
                             steps=arch["router_steps"], lr=arch["router_lr"],
                             width=arch["router_width"],
                             layer_ids=blocks[k])

    fresh_moe = bool(arch) and arch.get("converted_blocks", 0) == 0

    if arch and arch["converted_blocks"] != min(state["layer_block_idx"] + 1, len(blocks)):
        print(f"  AVISO: receta MoE dice {arch['converted_blocks']} bloque(s) convertidos pero "
              f"el estado está en bloque_capas {state['layer_block_idx'] + 1}; se usa la receta.")

    if arch:
        k_done = arch["converted_blocks"]
        if k_done > 0:
            print(f"Reaplicando dense->MoE en {k_done} bloque(s) ya convertidos...")
            for i in range(k_done):
                convert_block(i, split_weights=False)
        else:
            print("dense -> MoE: convirtiendo el bloque superior (bloque 1)...")
            convert_block(0, split_weights=True)

    # datos primero: el router training los necesita
    sd = fineweb.FineWebBlocks(block_idx=state["block_idx"])
    sd.load_tokens(TokWrap(tok))
    if fresh_moe:
        train_block_routers(0)

    if ck is not None:
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        print(f"Pesos cargados (missing={len(missing)}, unexpected={len(unexpected)})")
        if unexpected:
            print(f"  AVISO: {len(unexpected)} claves del checkpoint no existen en el modelo — ¿receta MoE distinta?")
        del ck
        if device.type == "cuda":
            torch.cuda.empty_cache()

    cur = min(state["layer_block_idx"], len(blocks) - 1)
    trainable_layers = layerlib.freeze_all_except_block(model, layers, cur, blocks)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Entrenando SOLO bloque {cur + 1}/{len(blocks)} (capas {trainable_layers}) | "
          f"resto congelado | params: {n_params:,}")

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    if state["step"] > 0:
        ck = checkpoint.load_checkpoint()
        if ck and "optimizer" in ck:
            try:
                opt.load_state_dict(ck["optimizer"])
                print("Optimizer cargado.")
            except Exception as e:
                print(f"  optimizer no cargado ({e}), AdamW fresco")
            del ck

    step = state["step"]
    recent = deque(maxlen=LOSS_WINDOW)
    t0 = time.time()
    last_push = time.time()
    last_report = step
    model.train()
    print("Entrenando...", flush=True)

    while step < max_steps and not state["done"]:
        tokens = sd.get_tokens()
        n_seq = (len(tokens) - seq_len - 1) // seq_len
        if n_seq <= 0:
            sd.next_block()
            state["block_idx"] = sd.block_idx
            continue
        for bs0 in range(0, n_seq, bs):
            if step >= max_steps or state["done"]:
                break
            be = min(bs0 + bs, n_seq)
            xs, ys = [], []
            for i in range(bs0, be):
                idx = i * seq_len
                xs.append(torch.tensor(tokens[idx:idx + seq_len], dtype=torch.long))
                ys.append(torch.tensor(tokens[idx + 1:idx + 1 + seq_len], dtype=torch.long))
            x = torch.stack(xs).to(device)
            y = torch.stack(ys).to(device)
            out = model(input_ids=x, labels=y)
            (out.loss / ga).backward()
            if ((bs0 // bs + 1) % ga == 0) or be >= n_seq:
                gn = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad()
                step += 1
                state["step"] = step
                state["steps_in_block"] += 1
                loss_val = float(out.loss.detach())
                recent.append(loss_val)
                tps = (step - last_report) * bs * ga * seq_len / max(time.time() - t0, 1e-3)
                plot.log_history(step, loss_val, lr, tps, float(gn), state["layer_block_idx"])
                if step % 10 == 0:
                    print(f"s{step} loss {loss_val:.4f} | bloque_capas {state['layer_block_idx'] + 1}/{len(blocks)} "
                          f"| bloque_datos {sd.block_idx} | {tps:.0f}t/s", flush=True)
                    t0 = time.time()
                    last_report = step
                # ¿salto de bloque?
                if step % EVAL_EVERY == 0:
                    avg = sum(recent) / len(recent)
                    if state["block_base_avg"] is None:
                        state["block_base_avg"] = avg
                        if not state["block_start_time"]:
                            state["block_start_time"] = time.time()
                        print(f"  avg_loss bloque (referencia): {avg:.4f}")
                    advance = False
                    if salto == "s":
                        advance = state["steps_in_block"] >= steps_per_block
                        if advance:
                            print(f"  {steps_per_block} steps en el bloque: avanzo")
                    else:
                        drop = state["block_base_avg"] - avg
                        elapsed = (time.time() - state["block_start_time"]) / 60
                        if drop >= loss_delta:
                            print(f"  avg_loss cayó {drop:.4f} (>= {loss_delta}): avanzo de bloque")
                            advance = True
                        elif elapsed >= block_minutes:
                            print(f"  {elapsed:.1f} min en el bloque sin bajar {loss_delta}: avanzo de bloque")
                            advance = True
                        else:
                            print(f"  avg_loss {avg:.4f} | caida {drop:.4f} | {elapsed:.1f}/{block_minutes} min")
                    if advance:
                        state["layer_block_idx"] += 1
                        # dense -> MoE: el bloque que entra a entrenar se convierte AHORA
                        if arch and arch["converted_blocks"] == state["layer_block_idx"]:
                            print(f"dense -> MoE: convirtiendo bloque {state['layer_block_idx'] + 1} "
                                  f"(capas {sorted(blocks[state['layer_block_idx']])})...")
                            convert_block(state["layer_block_idx"], split_weights=True)
                            train_block_routers(state["layer_block_idx"])
                        state["steps_in_block"] = 0
                        state["block_base_avg"] = None
                        state["block_start_time"] = time.time()
                        if state["layer_block_idx"] >= len(blocks):
                            state["done"] = True
                            print("Todos los bloques entrenados.")
                            if arch:
                                import d2dmoe
                                d2dmoe.set_moe_forward_mode(model, "dynk",
                                                            tau=arch.get("dynk_tau", 0.9))
                            plot.plot(step)
                            checkpoint.save_checkpoint(model, opt, state)
                            checkpoint.push_checkpoint(api, repo, branch, token, step)
                            model.save_pretrained(checkpoint.OUT_DIR)
                            tok.save_pretrained(checkpoint.OUT_DIR)
                            hf.hf_upload_folder(repo, branch, checkpoint.OUT_DIR, token,
                                                f"k2moe done step {step}")
                            print(f"Fin. Subido a {repo}@{branch}")
                            return
                        trainable_layers = layerlib.freeze_all_except_block(
                            model, layers, state["layer_block_idx"], blocks)
                        print(f"AVANCE -> bloque {state['layer_block_idx'] + 1}/{len(blocks)} | "
                              f"capas: {trainable_layers} (resto congelado)")
                        opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
                        recent.clear()
                        plot.plot(step)
                        checkpoint.save_checkpoint(model, opt, state)
                        checkpoint.push_checkpoint(api, repo, branch, token, step)
                        last_push = time.time()
                # push periódico
                if time.time() - last_push > 60 * SAVE_MINUTES:
                    plot.plot(step)
                    checkpoint.save_checkpoint(model, opt, state)
                    checkpoint.push_checkpoint(api, repo, branch, token, step)
                    last_push = time.time()
        sd.next_block()
        state["block_idx"] = sd.block_idx

    state["block_idx"] = sd.block_idx
    if arch:
        import d2dmoe
        d2dmoe.set_moe_forward_mode(model, "dynk", tau=arch.get("dynk_tau", 0.9))
    plot.plot(step)
    checkpoint.save_checkpoint(model, opt, state)
    model.save_pretrained(checkpoint.OUT_DIR)
    tok.save_pretrained(checkpoint.OUT_DIR)
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=BASE, allow_patterns=["*.py"], local_dir=checkpoint.OUT_DIR)
    except Exception as e:
        print(f"  no se pudieron bajar .py base: {e}")
    checkpoint.push_checkpoint(api, repo, branch, token, step, with_weights_dir=True)
    print(f"Fin. step {step} | Subido a {repo}@{branch}")


if __name__ == "__main__":
    main()
