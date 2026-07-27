"""Laurelia LLM Train: Dense + XSA, sin MLA.

Arquitectura: GQA standard + XSA (orthogonal projection removal)
en TODAS las capas. Sin MLA, sin BMA, sin Gated. Dense.
"""

import sys, os, time, math, inspect, torch
import torch.nn.functional as F
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.join(_DIR, ".."))
from model import TransformerLM
from dataset import download_wikipedia_50mb, StreamingDataset
from huggingface import HFManager, PeriodicPusher
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from plot import PlotManager


class BPEWrapper:
    def __init__(self, tok):
        self.tokenizer = tok
        self.vocab_size = tok.get_vocab_size()
    def encode(self, text):
        return self.tokenizer.encode(text).ids
    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)


@torch.no_grad()
def generate_sample(model, tokenizer, device, prompt="hola", max_new=30):
    model.eval()
    x = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens=max_new, temperature=0.7, top_k=40, top_p=0.9,
                         repetition_penalty=1.2)
    model.train()
    return tokenizer.decode(out[0].tolist())


def train_tokenizer_from_wiki(vocab_size, output_path):
    wiki = download_wikipedia_50mb()
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["eos_token"])
    with open(wiki, "r", encoding="utf-8") as f:
        tok.train_from_iterator([f.read()], trainer=trainer)
    tok.save(output_path)
    return output_path


def get_lr(step, total, warmup, lr):
    if step < warmup:
        return lr * (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return lr * (0.2 + 0.8 * (1.0 + math.cos(math.pi * t)) / 2.0)


# ─── Config Laurelia LLM (Dense + XSA) ────────────────────────────────────────
d_model = 768
num_layers = 24
num_heads = 12
num_kv_groups = 4
head_dim = d_model // num_heads
seq_len = 1024
max_seq_len = 65536
batch_size = 4
grad_accum = 16
lr = 25e-5
num_epochs = 200000
warmup_steps = 50
bpe_vocab = 32000

# Dense GQA + XSA (sin MLA, sin BMA, sin Gated)
use_mla = False
use_xsa = True
qk_norm = True
use_sandwich_norm = True
noise_std = 0.01

# Dense only (sin MoE)
use_moe = False

# FFN
dense_dim = None

# Data Mix
mezcla = True

plot_interval = 256

tok_path = os.path.join(_DIR, "tokenizer.json")


def main():
    test_mode = len(sys.argv) > 1 and sys.argv[1].endswith(".txt")
    txt_path = sys.argv[1] if test_mode else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── HF ──────────────────────────────────────────────────────────────────
    repo_id = "ScortexIA/laurelia"
    revision = "laurelia-llm"
    hf = pusher = None
    if not test_mode:
        hf = HFManager(repo_id=repo_id, revision=revision)
        hf._get_token()
        pusher = PeriodicPusher(hf, interval_minutes=20)
    pm = PlotManager(hf if not test_mode else None, save_dir=_DIR, plot_interval=plot_interval)

    # ── Precision: FP16 ─────────────────────────────────────────────────────
    amp = False
    if test_mode:
        dtype = torch.float32
    else:
        prec = input("Precision (n=f32, f=f16, b=bf16, a=amp(f16+master-f32)): ").strip().lower()
        if prec == "b":
            dtype = torch.bfloat16
        elif prec == "f":
            dtype = torch.float16
        elif prec == "a":
            dtype = torch.float16
            amp = True
        else:
            dtype = torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if device.type == "cuda" else torch.amp.GradScaler("cpu", enabled=amp)
    master = "f32 (master)" if amp else str(dtype)
    print(f"  Compute: {dtype}  |  Weights: {master}  |  AMP: {amp}")

    # ── Attention: Dense GQA + XSA ────────────────────────────────────────
    print(f"  Attention: GQA+XSA (dense)")
    print(f"  Dense only (no MoE) | {num_layers} capas")

    # ── Tokenizer ──────────────────────────────────────────────────────────
    tokenizer = None
    if os.path.exists(tok_path):
        tokenizer = BPEWrapper(Tokenizer.from_file(tok_path))
    elif hf and hf.tokenizer_exists():
        try:
            local_tok = hf.download_tokenizer(tok_path)
            tokenizer = BPEWrapper(Tokenizer.from_file(local_tok))
        except:
            pass
    if tokenizer is None:
        if hf:
            train_tokenizer_from_wiki(bpe_vocab, tok_path)
            tokenizer = BPEWrapper(Tokenizer.from_file(tok_path))
            hf.upload_tokenizer(tok_path, os.path.join(_DIR, "tokenizer_config.json"))
        else:
            sys.exit("No tokenizer found")
    print(f"Vocab: {tokenizer.vocab_size}")

    # ── Model ───────────────────────────────────────────────────────────────
    if dense_dim is not None:
        ffn_expansion = dense_dim * 3.0 / 2.0 / d_model
    else:
        ffn_expansion = 4.0

    model = TransformerLM(
        vocab_size=tokenizer.vocab_size, d_model=d_model, num_layers=num_layers,
        num_heads=num_heads, num_kv_groups=num_kv_groups, head_dim=head_dim,
        use_swiglu=True, use_x0=False, max_seq_len=max_seq_len, rotary_pct=0.25,
        ffn_expansion=ffn_expansion,
        residual_dropout=0.0, attn_dropout=0.0, ffn_dropout=0.0,
        use_mla=use_mla, use_xsa=use_xsa, qk_norm=qk_norm,
        attn_logit_cap=30, use_sandwich_norm=use_sandwich_norm, noise_std=noise_std,
        use_moe=False, cache_every=1,
    ).to(device).to(dtype=dtype)

    # Weight decay groups
    emb_params = [model.embedding.weight]
    other_decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2 and p.requires_grad and 'embedding' not in n]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2 and p.requires_grad]
    optim_groups = [
        {"params": emb_params, "weight_decay": 0.01, "lr_scale": 0.25},
        {"params": other_decay_params, "weight_decay": 0.01},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device.type == "cuda"
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(optim_groups, lr=lr, betas=(0.9, 0.95))
        print(f"Optimizer: AdamW 8-bit (bitsandbytes)")
    except ImportError:
        opt = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), fused=use_fused)
        print(f"Optimizer: AdamW fused={use_fused}")
    print(f"  decay={len(other_decay_params)} param tensors, emb_lr=lr/4, nodecay={len(nodecay_params)}")

    # ── Checkpoint ─────────────────────────────────────────────────────────
    step = 0
    epoch = 0
    ckpt_block = 0
    ckpt_path = os.path.join(_DIR, "checkpoint.pt")

    if not test_mode:
        loaded = False
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu')
            ckpt["model"].pop("head.emb_weight", None)
            model.load_state_dict(ckpt["model"], strict=False)
            step = ckpt.get("step", 0)
            epoch = ckpt.get("epoch", 0)
            ckpt_block = ckpt.get("block", 0)
            del ckpt
            torch.cuda.empty_cache()
            print(f"Loaded checkpoint: step {step} epoch {epoch} block {ckpt_block}")
            loaded = True
        elif hf and hf.download_checkpoint(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu')
            ckpt["model"].pop("head.emb_weight", None)
            model.load_state_dict(ckpt["model"], strict=False)
            step = ckpt.get("step", 0)
            epoch = ckpt.get("epoch", 0)
            ckpt_block = ckpt.get("block", 0)
            del ckpt
            torch.cuda.empty_cache()
            print(f"Loaded HF checkpoint: step {step} epoch {epoch} block {ckpt_block}")
            loaded = True

        if loaded:
            print("\n── Generation test ──")
            for p in ["hola", "que es la inteligencia artificial", "en un lugar de la mancha", "hoy hace mucho calor"]:
                sample = generate_sample(model, tokenizer, device, prompt=p, max_new=50)
                print(f"  [{p}] → {sample}")
            print("── End test ──\n")

    # ── Data ────────────────────────────────────────────────────────────────
    if test_mode:
        with open(txt_path, "r", encoding="utf-8") as f:
            all_tokens = tokenizer.encode(f.read())
        n = len(all_tokens)
        epochs_do = 10
        total_steps = ((n - seq_len - 1) // (batch_size * seq_len)) * epochs_do
    else:
        bi = input(f"Block [{ckpt_block}]: ").strip()
        block_idx = int(bi) if bi else ckpt_block
        sd = StreamingDataset(block_mb=3.0, block_idx=block_idx, mezcla=mezcla)
        print(f"Mix {'enabled' if mezcla else 'disabled'} (FineWeb2-HQ)")
        sd.load_tokens(tokenizer)
        n = len(sd.get_tokens())
        tokens_per_epoch = (n - seq_len - 1) // seq_len
        total_steps = (tokens_per_epoch // batch_size) * num_epochs
        epochs_do = num_epochs

    # ── Stats ──────────────────────────────────────────────
    emb_p = model.embedding.weight.numel()
    layer_p = sum(p.numel() for l in model.transformer.layers for p in l.parameters())
    norm_p = model.transformer.final_norm.weight.numel()
    total_p = emb_p + layer_p + norm_p
    print(f"Params: emb={emb_p:,} + {num_layers}capas={layer_p:,} + norm={norm_p} = {total_p:,}")
    print(f"dim={d_model} lay={num_layers} heads={num_heads} kv={num_kv_groups} seq={seq_len} bs={batch_size} ga={grad_accum} lr={lr}")
    print(f"GQA+XSA: {num_kv_groups}kv groups | XSA enabled")
    print(f"Tokens: {n:,} | Steps total: {total_steps}")

    # ── Train loop ──────────────────────────────────────────────────────────
    model.train()
    t0 = time.time()
    last_rpt_time = t0
    last_rpt_step = 0

    epoch = 0
    torch.cuda.empty_cache()
    while True:
        if test_mode:
            tokens = all_tokens
        else:
            tokens = sd.get_tokens()

        n_seq = (len(tokens) - seq_len - 1) // seq_len
        if n_seq <= 0:
            print(f"Block too small ({len(tokens)} tokens), skipping epoch {epoch}")
            epoch += 1
            if epoch >= epochs_do:
                break
            continue

        micro = 0
        for batch_start in range(0, n_seq, batch_size):
            if step >= total_steps:
                break
            batch_end = min(batch_start + batch_size, n_seq)
            x_list, y_list = [], []
            for i in range(batch_start, batch_end):
                idx = i * seq_len
                x = torch.tensor([tokens[idx + j] for j in range(seq_len)], dtype=torch.long, device=device).unsqueeze(0)
                y = torch.tensor([tokens[idx + j + 1] for j in range(seq_len)], dtype=torch.long, device=device).unsqueeze(0)
                x_list.append(x)
                y_list.append(y)
            x = torch.cat(x_list, dim=0)
            y = torch.cat(y_list, dim=0)

            if micro == 0:
                lr_curr = get_lr(step, total_steps, warmup_steps, lr)
                for pg in opt.param_groups:
                    pg["lr"] = lr_curr * pg.get("lr_scale", 1.0)
                opt.zero_grad()

            logits, aux_loss = model(x)
            loss = F.cross_entropy(logits.reshape(-1, tokenizer.vocab_size), y.reshape(-1))
            (loss / grad_accum).backward()
            micro += 1

            if micro >= grad_accum:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)

                if step % 10 == 0:
                    layer_grads = {}
                    for name, param in model.named_parameters():
                        if param.grad is None:
                            continue
                        g = param.grad
                        parts = name.split('.')
                        if len(parts) > 2 and parts[0] == 'transformer' and parts[1] == 'layers':
                            layer_key = f"L{parts[2]}"
                        elif parts[0] == 'transformer' and parts[1] == 'head':
                            layer_key = 'head'
                        else:
                            layer_key = parts[0]
                        if layer_key not in layer_grads:
                            layer_grads[layer_key] = []
                        layer_grads[layer_key].append(g.norm().item())

                    layer_norms = []
                    for layer_key, norms in layer_grads.items():
                        avg_norm = sum(norms) / len(norms)
                        layer_norms.append((layer_key, avg_norm))
                    layer_norms.sort(key=lambda x: x[1], reverse=True)

                    grad_report = " | ".join(
                        f"{k}={v:.3g}" for k, v in layer_norms
                    )
                    print(f"  Grad norms: {grad_report}")

                opt.step()
                opt.zero_grad()
                step += 1
                micro = 0

                if step % 10 == 0:
                    now = time.time()
                    tok = (step - last_rpt_step) * batch_size * grad_accum * seq_len
                    tps = tok / max(now - last_rpt_time, 0.001)
                    print(f"e{epoch} s{step} loss {loss.item():.4f} lr {lr_curr:.6f} {tps:.0f}t/s")
                    last_rpt_time = now
                    last_rpt_step = step
                    pm.log(step, loss.item(), lr_curr, tps)

                if not test_mode and step % 50 == 0:
                    t_gen = time.time()
                    sample = generate_sample(model, tokenizer, device)
                    gen_tps = 100 / (time.time() - t_gen)
                    print(f"  >>> {sample}  [{gen_tps:.0f} tok/s]")

                if not test_mode and pusher and (time.time() - pusher.last_push) >= pusher.interval:
                    state = model.state_dict()
                    state.pop("head.emb_weight", None)
                    ckpt = {"step": step, "epoch": epoch, "block": sd.block_idx if not test_mode else 0, "model": state}
                    torch.save(ckpt, ckpt_path)
                    pusher.maybe_push(ckpt_path, None, tok_path, step)
                    pm.plot(step)
                    pm.upload(step)

        if micro > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            opt.zero_grad()
            step += 1
            micro = 0

        epoch += 1
        print(f"── Epoch {epoch} done: {step} steps ──")
        if epoch >= epochs_do:
            break

        if not test_mode:
            tokens = None
            sd.next_block()
            total_tokens = len(sd.get_tokens())
            n_seq = (total_tokens - seq_len - 1) // seq_len

    if not test_mode and hf:
        ckpt = {"step": step, "epoch": epoch, "block": sd.block_idx, "model": model.state_dict()}
        torch.save(ckpt, ckpt_path)
        hf.upload_checkpoint(ckpt_path, tok_path, step)

    print(f"Done! {step} steps in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
