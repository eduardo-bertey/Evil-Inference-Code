"""inferk2.py — solo router training sobre K2-Horizon-0.9B (d2dmoe progresivo).

Carga el modelo OFICIAL IFM/K2-Horizon-0.9B, baja 10MB de FineWeb y por cada
bloque de 2 capas (de arriba hacia abajo) convierte dense->MoE y entrena SOLO
su router (2500 steps, MSE contra norma de expertos — Eq.3 del paper).
Al final: chat con el modelo; en caliente puedes cambiar dynk/tau/top-k.

Checkpoints en ScortexIA/laurelia@tempk2.

Ejecuta:  !python k2moe/inferk2.py
"""

import json
import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

import torch

import fineweb
import hf
import layers as layerlib
import d2dmoe

BASE = "IFM/K2-Horizon-0.9B"
REPO = "ScortexIA/laurelia"
BRANCH = "tempk2"
CKPT = os.path.join(_DIR, "inferk2_checkpoint.pt")
STATE_PATH = os.path.join(_DIR, "inferk2_state.json")
BLOCK_MB = 10.0
OUT_DIR = os.path.join(_DIR, "inferk2_out")


def default_state():
    return {"blocks_done": 0, "step": 0}


def save_state(state):
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_PATH)


def load_state():
    return json.load(open(STATE_PATH)) if os.path.exists(STATE_PATH) else default_state()


def save_ckpt(model, tok, state):
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save({"model": model.state_dict(), **state}, CKPT)
    tok.save_pretrained(OUT_DIR)
    save_state(state)


def push_ckpt(api, token, msg):
    hf.hf_upload_file(api, REPO, BRANCH, CKPT, "inferk2_checkpoint.pt", token, msg)
    hf.hf_upload_file(api, REPO, BRANCH, STATE_PATH, "inferk2_state.json", token, msg)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    token = hf.hf_token(REPO)
    api = hf.hf_api(token)
    hf.hf_ensure_repo_branch(api, REPO, BRANCH, token)

    prec = input("Precision (n=f32, b=bf16) [b]: ").strip().lower() or "b"
    dtype = torch.bfloat16 if prec == "b" else torch.float32
    layers_per_block = int(input("Capas por bloque [2]: ").strip() or 2)
    n_experts = int(input("Expertos por capa [32]: ").strip() or 32)
    swap = input("¿Cambiar SiLU/GELU -> ReLU? [s]: ").strip().lower() or "s"
    swap_relu = swap == "s"
    router_steps = int(input("Steps de router por bloque [2500]: ").strip() or 2500)
    router_lr = float(input("lr router [1e-3]: ").strip() or 1e-3)
    router_width = int(input("Ancho del router [512]: ").strip() or 512)
    tau = float(input("Tau dynk para inferencia [0.5]: ").strip() or 0.5)

    # ¿continuar desde checkpoint tempk2?
    state = default_state()
    have = os.path.exists(CKPT)
    if not have:
        have = hf.hf_download(REPO, BRANCH, "inferk2_checkpoint.pt", CKPT, token)
    if have:
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        print(f"\nCheckpoint encontrado en tempk2: bloques {ck.get('blocks_done')} hechos")
        cont = input("¿Continuar desde ahí? [s/n] [s]: ").strip().lower() or "s"
        if cont == "s":
            state.update(ck)
        else:
            ck = None
    else:
        ck = None

    # tokenizer/model oficial
    print(f"\nCargando tokenizer {BASE}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"Cargando {BASE}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=dtype, low_cpu_mem_usage=True, device_map="auto",
        trust_remote_code=True)
    model.config.use_cache = False

    # FineWeb 10MB SOLO para el router training
    print("Abriendo FineWeb (10MB, solo router training)...")

    class TokWrap:
        def __init__(self, tok):
            self.tok = tok
            self.vocab_size = tok.vocab_size

        def encode(self, text):
            return self.tok.encode(text, add_special_tokens=False)

    sd = fineweb.FineWebBlocks(block_idx=0, block_mb=BLOCK_MB)
    sd.load_tokens(TokWrap(tok))

    layers = layerlib.get_layers(model)
    blocks = layerlib.top_down_blocks(len(layers), layers_per_block)
    N = len(blocks)
    print(f"{len(layers)} capas | {N} bloques de {layers_per_block}")

    def convert_block(k, split_weights):
        if swap_relu:
            d2dmoe.swap_ffn_activation_to_relu(model, blocks[k])
        d2dmoe.replace_with_moes(model, n_experts, blocks[k], split_weights=split_weights)

    # reanudar: reaplicar conversiones ya hechas (sin k-means; pesos desde CKPT)
    k_done = state["blocks_done"]
    for i in range(k_done):
        convert_block(i, split_weights=False)
    if ck is not None:
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        print(f"Pesos cargados (missing={len(missing)}, unexpected={len(unexpected)})")
        del ck
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # entrenar bloques que faltan
    for k in range(k_done, N):
        print(f"\n=== Bloque {k + 1}/{N} | capas {sorted(blocks[k])} ===")
        print("  dense -> MoE (k-means)...")
        convert_block(k, split_weights=True)
        print("  router training...")
        d2dmoe.add_routers(model, layer_ids=blocks[k], width=router_width)
        d2dmoe.train_routers(model, sd.get_tokens(), 1024, device,
                             steps=router_steps, lr=router_lr,
                             width=router_width, layer_ids=blocks[k])
        state["blocks_done"] = k + 1
        state["step"] += router_steps
        save_ckpt(model, tok, state)
        push_ckpt(api, token, f"inferk2 block {k + 1}/{N}")
        d2dmoe.set_moe_forward_mode(model, "dynk", tau=tau)

    print("\nConversion completa.")
    d2dmoe.set_moe_forward_mode(model, "dynk", tau=tau)
    save_ckpt(model, tok, state)
    push_ckpt(api, token, "inferk2 final")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=BASE, allow_patterns=["*.py"], local_dir=OUT_DIR)
    except Exception as e:
        print(f"  no se pudieron bajar .py base: {e}")
    hf.hf_upload_folder(REPO, BRANCH, OUT_DIR, token, "inferk2 final")

    # chat
    chat(model, tok, device, state)


def chat(model, tok, device, state):
    print("\nListo para chatear. Comandos:")
    print("  :tau <0-1]   dynamic-k con ese tau      :dynk [tau]  dynk (def 0.5)")
    print("  :topk <k>    top-k fijo                 :all          denso-equivalente")
    print("  :usage       expertos ejecutados/token   :exit         salir")
    print("  :help        esta ayuda")
    messages = []
    while True:
        try:
            user = input("\ntú> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.startswith(":"):
            if _cmd(user[1:].strip(), model, tok, device) == "exit":
                print("Hasta luego.")
                return
            continue
        messages.append({"role": "user", "content": user})
        text = tok.apply_chat_template(messages, add_generation_prompt=True,
                                        tokenize=False)
        inputs = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512,
                                 temperature=0.6, top_p=0.95, do_sample=True,
                                 pad_token_id=tok.eos_token_id)
        reply = tok.decode(out[0][inputs.input_ids.shape[1]:],
                           skip_special_tokens=True)
        print(f"\nk2> {reply}")
        messages.append({"role": "assistant", "content": reply})


def _cmd(cmd, model, tok, device):
    parts = cmd.split()
    name = parts[0].lower() if parts else ""
    try:
        if name == "exit" or name == "quit" or name == "q":
            return "exit"
        if name == "help":
            print("  :tau <0-1] | :dynk [tau] | :topk <k> | :all | :usage | :exit")
            return
        if name == "tau" and len(parts) > 1:
            t = float(parts[1])
            t = min(max(t, 1e-3), 1.0)
            d2dmoe.set_moe_forward_mode(model, "dynk", tau=t)
            return
        if name == "dynk":
            t = float(parts[1]) if len(parts) > 1 else 0.5
            d2dmoe.set_moe_forward_mode(model, "dynk", tau=t)
            return
        if name == "topk" and len(parts) > 1:
            k = int(parts[1])
            if k < 1:
                raise ValueError("k>=1")
            d2dmoe.set_moe_forward_mode(model, "topk", k=k)
            return
        if name == "all":
            d2dmoe.set_moe_forward_mode(model, "all")
            return
        if name == "usage":
            ids = tok("hola", return_tensors="pt").input_ids.to(device)
            d2dmoe.print_dynk_usage(model, ids)
            return
        print("Comando no reconocido; prueba :help")
    except ValueError as e:
        print(f"Valor inválido: {e}")


if __name__ == "__main__":
    main()
