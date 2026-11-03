"""k2moe — fine-tune top-down por bloques de capas sobre FineWeb (para Colab).

Ejecuta:
  !python k2moe/train.py

Pregunta solo lo básico: repo HF, rama, token, precisión, capas por bloque,
salto de bloque (loss o steps), lr, steps y block de datos.

Empieza entrenando el bloque SUPERIOR de capas (+ embeddings/norms/lm_head).
El salto al siguiente bloque hacia abajo se elige por:
  - loss por tiempo: avanza si avg_loss cae >= delta, o si el bloque lleva
    X minutos sin bajar suficiente
  - steps elegidos: avanza cada N steps
Al agotar todos los bloques el modelo queda completo.

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

    repo = input("Repo HF destino [ScortexIA/laurelia]: ").strip() or "ScortexIA/laurelia"
    branch = input("Rama [k2moe]: ").strip() or "k2moe"
    token = hf.hf_token(repo)
    api = hf.hf_api(token)
    hf.hf_ensure_repo_branch(api, repo, branch, token)

    prec = input("Precision (n=f32, b=bf16) [b]: ").strip().lower() or "b"
    dtype = torch.bfloat16 if prec == "b" else torch.float32

    layers_per_block = int(input("Capas por bloque (desde arriba) [2]: ").strip() or 2)
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
                           "block_start_time", "done")})
        else:
            ck = None
            print("Empezando de cero (el checkpoint se sobrescribirá al guardar).")

    if state["done"]:
        print("Este checkpoint ya completó todos los bloques.")
        return

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
    if ck is not None:
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        print(f"Pesos cargados (missing={len(missing)}, unexpected={len(unexpected)})")
        del ck
        if device.type == "cuda":
            torch.cuda.empty_cache()

    layers = layerlib.get_layers(model)
    blocks = layerlib.top_down_blocks(len(layers), layers_per_block)
    print(f"{len(layers)} capas | bloques top-down: {blocks}")
    n_active = min(state["layer_block_idx"] + 1, len(blocks))
    trainable_layers = layerlib.apply_freeze(model, layers, n_active, blocks)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Bloques activos: {n_active}/{len(blocks)} | capas entrenables: {trainable_layers} | "
          f"params: {n_params:,}")

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

    sd = fineweb.FineWebBlocks(block_idx=state["block_idx"])
    sd.load_tokens(TokWrap(tok))

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
                        state["steps_in_block"] = 0
                        state["block_base_avg"] = None
                        state["block_start_time"] = time.time()
                        if state["layer_block_idx"] >= len(blocks):
                            state["done"] = True
                            print("Todos los bloques entrenados.")
                            plot.plot(step)
                            checkpoint.save_checkpoint(model, opt, state)
                            checkpoint.push_checkpoint(api, repo, branch, token, step)
                            model.save_pretrained(checkpoint.OUT_DIR)
                            tok.save_pretrained(checkpoint.OUT_DIR)
                            hf.hf_upload_folder(repo, branch, checkpoint.OUT_DIR, token,
                                                f"k2moe done step {step}")
                            print(f"Fin. Subido a {repo}@{branch}")
                            return
                        n_active = state["layer_block_idx"] + 1
                        trainable_layers = layerlib.apply_freeze(model, layers, n_active, blocks)
                        print(f"AVANCE -> bloques {n_active}/{len(blocks)} | capas: {trainable_layers}")
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
