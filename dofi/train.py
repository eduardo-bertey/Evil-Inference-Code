"""Dofi Training — Entrenamiento por bloques con DiffusionBlocks.

Transformer autoregressivo de 16 capas, 4 bloques de 4 capas.
XSA + K=V, sin LISA.
Cada bloque se entrena independientemente con denoising loss.
Usa TrainData de laurelia (wiki 3MB + fine 2MB + tuit 1MB por bloque).
"""

import sys, os, time, math, json, random
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import norm

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from model import DofiLLM, Config
from huggingface import HFManager, PeriodicPusher
from plot import PlotManager
import importlib
train_data = importlib.import_module("train-data")
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


class BPEWrapper:
    def __init__(self, tok):
        self.tokenizer = tok
        self.vocab_size = tok.get_vocab_size()
    def encode(self, text):
        return self.tokenizer.encode(text).ids
    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)


# ── Sigma scheduling ──────────────────────────────────────────────

def get_block_sigmas(config):
    """Equi-probability partitioning: divide σ en B bloques por masa de probabilidad igual."""
    cdf_min = norm.cdf((np.log(config.sigma_min) - config.p_mean) / config.p_std)
    cdf_max = norm.cdf((np.log(config.sigma_max) - config.p_mean) / config.p_std)
    boundaries = np.linspace(cdf_min, cdf_max, config.num_blocks + 1)
    sigmas = np.exp(config.p_mean + config.p_std * norm.ppf(boundaries))
    return sigmas


def sample_sigma_in_block(block_idx, block_sigmas, gamma=0.05):
    """Muestra σ del rango del bloque con overlap γ."""
    sigma_min_b = block_sigmas[block_idx]
    sigma_max_b = block_sigmas[block_idx + 1]

    log_range = np.log(sigma_max_b) - np.log(sigma_min_b)
    log_min = np.log(sigma_min_b) - gamma * log_range
    log_max = np.log(sigma_max_b) + gamma * log_range

    cdf_min = norm.cdf((log_min - (-1.2)) / 1.2)
    cdf_max = norm.cdf((log_max - (-1.2)) / 1.2)
    u = np.random.uniform(max(cdf_min, 0.0), min(cdf_max, 1.0))
    log_sigma = -1.2 + 1.2 * norm.ppf(u)
    return float(max(np.exp(log_sigma), 0.05))


def edm_weight(sigma, sigma_data=0.5):
    """EDM weighting: w(σ) = (σ² + sigma_data²) / (σ·sigma_data)²."""
    sigma = np.maximum(sigma, 1e-8)
    return (sigma**2 + sigma_data**2) / (sigma * sigma_data)**2


# ── Tokenizer ─────────────────────────────────────────────────────

def train_tokenizer_from_wiki(vocab_size, output_path):
    from dataset import download_tokenizer_corpus
    corpus = download_tokenizer_corpus()
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["eos_token"])
    with open(corpus, "r", encoding="utf-8") as f:
        tok.train_from_iterator([f.read()], trainer=trainer)
    tok.save(output_path)
    return output_path


# ── Training ──────────────────────────────────────────────────────

config = Config()
tok_path = os.path.join(_DIR, "tokenizer.json")
plot_interval = 256


@torch.no_grad()
def generate_sample(model, tokenizer, device, prompt="hola", max_new=50):
    model.eval()
    x = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens=max_new, temperature=0.7, top_k=40,
                         sigma=0.002, use_ode=False)
    model.train()
    return tokenizer.decode(out[0].tolist())


def generate_mode():
    """Modo generación: carga checkpoint y genera."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tok_path = os.path.join(_DIR, "tokenizer.json")
    tokenizer = BPEWrapper(Tokenizer.from_file(tok_path))
    config.emb_num = tokenizer.vocab_size

    model = DofiLLM(config).to(device)
    ckpt_path = os.path.join(_DIR, "checkpoint.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        ckpt["model"].pop("lm_head.weight", None)
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"Loaded checkpoint step {ckpt.get('step', '?')}")
    model.eval()

    prompts = ["hola", "que es la inteligencia artificial", "en un lugar de la mancha"]
    for p in prompts:
        sample = generate_sample(model, tokenizer, device, prompt=p, max_new=80)
        print(f"  [{p}] → {sample}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--generate":
        generate_mode()
        return

    test_mode = len(sys.argv) > 1 and sys.argv[1].endswith(".txt")
    txt_path = sys.argv[1] if test_mode else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    repo_id = "ScortexIA/laurelia"
    revision = "dofi"
    hf = pusher = None
    if not test_mode:
        hf = HFManager(repo_id=repo_id, revision=revision)
        hf._get_token()
        pusher = PeriodicPusher(hf, interval_minutes=20)
    pm = PlotManager(hf if not test_mode else None, save_dir=_DIR, plot_interval=plot_interval)

    if test_mode:
        dtype = torch.float32
    else:
        prec = input("Precision (n=f32, b=bf16): ").strip().lower()
        dtype = torch.bfloat16 if prec == "b" else torch.float32
    print(f"  Compute: {dtype}")

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
            train_tokenizer_from_wiki(config.emb_num, tok_path)
            tokenizer = BPEWrapper(Tokenizer.from_file(tok_path))
            hf.upload_tokenizer(tok_path)
        else:
            sys.exit("No tokenizer found")
    config.emb_num = tokenizer.vocab_size
    print(f"Vocab: {tokenizer.vocab_size}")

    model = DofiLLM(config).to(device).to(dtype=dtype)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )

    # Sigma scheduling
    block_sigmas = get_block_sigmas(config)
    print(f"Block sigmas: {block_sigmas}")

    ckpt_path = os.path.join(_DIR, "checkpoint.pt")
    step = 0
    ckpt_block = 0

    if not test_mode:
        loaded = False
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu')
            ckpt["model"].pop("lm_head.weight", None)
            model.load_state_dict(ckpt["model"], strict=False)
            step = ckpt.get("step", 0)
            ckpt_block = ckpt.get("block_idx", 0)
            del ckpt
            torch.cuda.empty_cache()
            print(f"Loaded checkpoint: step {step} block {ckpt_block}")
            loaded = True
        elif hf and hf.download_checkpoint(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu')
            ckpt["model"].pop("lm_head.weight", None)
            model.load_state_dict(ckpt["model"], strict=False)
            step = ckpt.get("step", 0)
            ckpt_block = ckpt.get("block_idx", 0)
            del ckpt
            torch.cuda.empty_cache()
            print(f"Loaded HF checkpoint: step {step} block {ckpt_block}")
            loaded = True

        if loaded:
            print("\n── Generation test ──")
            for p in ["hola", "que es la inteligencia artificial", "en un lugar de la mancha"]:
                sample = generate_sample(model, tokenizer, device, prompt=p, max_new=50)
                print(f"  [{p}] → {sample}")
            print("── End test ──\n")

    # Dataset
    seq_len = config.block_size
    if test_mode:
        with open(txt_path, "r", encoding="utf-8") as f:
            all_tokens = tokenizer.encode(f.read())
        n = len(all_tokens)
        total_steps = ((n - seq_len - 1) // (config.batch_size * seq_len)) * 10
    else:
        bi = input(f"Block [{ckpt_block}]: ").strip()
        block_idx = int(bi) if bi else ckpt_block
        sd = train_data.TrainData(block_idx=block_idx)
        sd.load_tokens(tokenizer)
        tokens = sd.get_tokens()
        n = len(tokens)
        n_seq = (n - seq_len - 1) // seq_len
        steps_per_epoch = n_seq // config.batch_size
        total_steps = steps_per_epoch * 200000

    # WSD schedule
    num_warmup = config.warm_up
    num_decay = int(total_steps * 0.15)
    num_stable = total_steps - num_warmup - num_decay

    def lr_lambda(step):
        if step < num_warmup:
            return float(step) / float(max(1, num_warmup))
        if step < num_warmup + num_stable:
            return 1.0
        progress = float(step - num_warmup - num_stable) / float(max(1, num_decay))
        progress = min(1.0, progress)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    emb_p = model.embeddings.weight.numel()
    layer_p = sum(p.numel() for b in model.blocks for p in b.parameters())
    print(f"Params: emb={emb_p:,} + {config.layers}capas={layer_p:,} = {emb_p + layer_p:,}")
    print(f"dim={config.dim} lay={config.layers} heads={config.heads} kv={config.kv_groups} "
          f"blocks={config.num_blocks} seq={seq_len} bs={config.batch_size} lr={config.learning_rate}")
    print(f"Tokens: {n:,}")

    model.train()
    t0 = time.time()
    last_rpt_time = t0
    last_rpt_step = 0
    seq_block_counter = 0

    while True:
        if test_mode:
            tokens = all_tokens
            n_seq = (len(tokens) - seq_len - 1) // seq_len
        else:
            tokens = sd.get_tokens()
            n_seq = (len(tokens) - seq_len - 1) // seq_len

        if n_seq <= 0:
            if not test_mode:
                sd.next_block()
            continue

        for batch_start in range(0, n_seq, config.batch_size):
            if step >= total_steps:
                break
            batch_end = min(batch_start + config.batch_size, n_seq)

            # 1. Eleg bloque
            if config.sequential_blocks:
                dblock_idx = seq_block_counter % config.num_blocks
                seq_block_counter += 1
            else:
                dblock_idx = random.randint(0, config.num_blocks - 1)

            # 2. Muestrear σ del rango del bloque
            sigma_np = sample_sigma_in_block(dblock_idx, block_sigmas, gamma=config.gamma)
            sigma = torch.tensor(sigma_np, device=device)

            # 3. Preparar batch
            x_list, y_list = [], []
            for i in range(batch_start, batch_end):
                idx = i * seq_len
                x = torch.tensor(tokens[idx:idx + seq_len], dtype=torch.long, device=device).unsqueeze(0)
                y = torch.tensor(tokens[idx + 1:idx + seq_len + 1], dtype=torch.long, device=device).unsqueeze(0)
                x_list.append(x)
                y_list.append(y)
            input_ids = torch.cat(x_list, dim=0)
            target_ids = torch.cat(y_list, dim=0)

            # 4. Forward por el bloque
            if config.sequential_blocks:
                total_loss = 0.0
                for b in range(config.num_blocks):
                    sigma_np = sample_sigma_in_block(b, block_sigmas, gamma=config.gamma)
                    sigma = torch.tensor(sigma_np, device=device)
                    logits = model.forward_block(b, input_ids, sigma, target_ids=target_ids)
                    loss_b = F.cross_entropy(logits.view(-1, config.emb_num), target_ids.view(-1))
                    w = float(edm_weight(sigma_np, config.sigma_data))
                    total_loss = total_loss + loss_b * w
                    del logits
                loss = total_loss / config.num_blocks
            else:
                logits = model.forward_block(dblock_idx, input_ids, sigma, target_ids=target_ids)
                loss = F.cross_entropy(logits.view(-1, config.emb_num), target_ids.view(-1))
                w = float(edm_weight(sigma_np, config.sigma_data))
                loss = loss * w

            # 6. Backward
            (loss / config.grad_acc).backward()
            loss_val = loss.item()
            del logits, loss

            if (batch_start // config.batch_size + 1) % config.grad_acc == 0 or batch_end >= n_seq:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                lr_curr = float(scheduler.get_last_lr()[0])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if step % 10 == 0:
                    now = time.time()
                    tok = (step - last_rpt_step) * config.batch_size * config.grad_acc * seq_len
                    tps = tok / max(now - last_rpt_time, 0.001)
                    print(f"s{step} loss {loss_val:.4f} w={w:.3f} lr {lr_curr:.6f} "
                          f"grad {grad_norm:.3f} dblock={dblock_idx} σ={sigma_np:.4f} {tps:.0f}t/s")
                    last_rpt_time = now
                    last_rpt_step = step
                    pm.log(step, loss_val, lr_curr, tps)

                if not test_mode and step % 300 == 0:
                    sample = generate_sample(model, tokenizer, device)
                    print(f"  >>> {sample}")

                if not test_mode and pusher and (time.time() - pusher.last_push) >= pusher.interval:
                    state = model.state_dict()
                    state.pop("lm_head.weight", None)
                    ckpt = {"step": step, "block_idx": sd.block_idx if not test_mode else 0, "model": state}
                    torch.save(ckpt, ckpt_path)
                    pusher.maybe_push(ckpt_path, None, None, step)
                    pm.plot(step)
                    pm.upload(step)

        if not test_mode:
            sd.next_block()

    if not test_mode and hf:
        state = model.state_dict()
        state.pop("lm_head.weight", None)
        ckpt = {"step": step, "block_idx": sd.block_idx if not test_mode else 0, "model": state}
        torch.save(ckpt, ckpt_path)
        hf.upload_checkpoint(ckpt_path, step=step)

    print(f"Done! {step} steps in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
