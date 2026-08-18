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
sys.path.insert(0, os.path.join(_DIR, ".."))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from model import DofiLLM, Config
import importlib
train_data = importlib.import_module("train-data")


# ── HF Token ─────────────────────────────────────────────────────

def get_hf_token():
    import getpass
    from huggingface_hub import login as hf_login
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        token = getpass.getpass("HF token (write a ScortexIA/laurelia): ").strip()
    os.environ["HF_TOKEN"] = token
    hf_login(token=token)
    print("HF token aplicado globalmente")
    return token


# ── Sigma scheduling ──────────────────────────────────────────────

def get_block_sigmas(config):
    """Equi-probability partitioning: divide σ en B bloques por masa de probabilidad igual."""
    cdf_min = norm.cdf((np.log(config.sigma_min) - config.p_mean) / config.p_std)
    cdf_max = norm.cdf((np.log(config.sigma_max) - config.p_mean) / config.p_std)
    boundaries = np.linspace(cdf_min, cdf_max, config.num_blocks + 1)
    sigmas = np.exp(config.p_mean + config.p_std * norm.ppf(boundaries))
    return sigmas


def sample_sigma_in_block(block_idx, block_sigmas, gamma=0.05, size=1):
    """Muestra σ del rango del bloque con overlap γ."""
    sigma_min_b = block_sigmas[block_idx]
    sigma_max_b = block_sigmas[block_idx + 1]

    log_range = np.log(sigma_max_b) - np.log(sigma_min_b)
    log_min = np.log(sigma_min_b) - gamma * log_range
    log_max = np.log(sigma_max_b) + gamma * log_range

    cdf_min = norm.cdf((log_min - (-1.2)) / 1.2)
    cdf_max = norm.cdf((log_max - (-1.2)) / 1.2)
    u = np.random.uniform(max(cdf_min, 0.0), min(cdf_max, 1.0), size=size)
    log_sigma = -1.2 + 1.2 * norm.ppf(u)
    return np.exp(log_sigma).astype(np.float32)


def edm_weight(sigma, sigma_data=0.5):
    """EDM weighting: w(σ) = (σ² + sigma_data²) / (σ·sigma_data)²."""
    sigma = np.maximum(sigma, 1e-8)
    return (sigma**2 + sigma_data**2) / (sigma * sigma_data)**2


# ── BPE Tokenizer ────────────────────────────────────────────────

class BPEWrapper:
    def __init__(self, tok):
        self.tokenizer = tok
        self.vocab_size = tok.get_vocab_size()
    def encode(self, text):
        return self.tokenizer.encode(text).ids
    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)


def get_tokenizer(config):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    tok_path = os.path.join(_DIR, "tokenizer_test_16k.json")
    wiki = os.path.join(_DIR, "wiki_tokenizer_50mb.txt")

    # Descargar wiki si no existe
    import importlib
    wikipedia = importlib.import_module("wikipedia")
    if not os.path.exists(wiki) or os.path.getsize(wiki) < 50_000_000:
        wikipedia.download_wikipedia_50mb(wiki)

    if not os.path.exists(tok_path):
        print("Entrenando BPE 16k...")
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=config.emb_num, special_tokens=["eos_token"])

        def iter_chunks(path, mb=5):
            with open(path, "rb") as fh:
                while True:
                    chunk = fh.read(mb * 1024 * 1024)
                    if not chunk:
                        break
                    yield chunk.decode("utf-8", errors="ignore")

        tok.train_from_iterator(iter_chunks(wiki), trainer=trainer)
        tok.save(tok_path)

    tokenizer = BPEWrapper(Tokenizer.from_file(tok_path))
    config.emb_num = tokenizer.vocab_size
    print(f"Vocab: {tokenizer.vocab_size}")
    return tokenizer


# ── Training ──────────────────────────────────────────────────────

def train(config):
    # HF token al inicio
    get_hf_token()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: dim={config.dim} lay={config.layers} heads={config.heads} "
          f"kv={config.kv_groups} blocks={config.num_blocks} "
          f"seq={config.block_size} bs={config.batch_size} lr={config.learning_rate}")

    # Tokenizer
    tokenizer = get_tokenizer(config)

    # Modelo
    model = DofiLLM(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )

    # Sigma scheduling
    block_sigmas = get_block_sigmas(config)
    print(f"Block sigmas: {block_sigmas}")

    # Dataset de laurelia
    bi = input("Block [0]: ").strip()
    block_idx = int(bi) if bi else 0
    sd = train_data.TrainData(block_idx=block_idx)
    sd.load_tokens(tokenizer)
    tokens = sd.get_tokens()
    n = len(tokens)
    print(f"Tokens: {n:,}")

    # Scheduler
    seq_len = config.block_size
    n_seq = (n - seq_len - 1) // seq_len
    steps_per_epoch = n_seq // config.batch_size
    total_steps = steps_per_epoch * 200000
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

    # Training loop
    model.train()
    t0 = time.time()
    step = 0
    history = []

    while True:
        tokens = sd.get_tokens()
        n_seq = (len(tokens) - seq_len - 1) // seq_len

        if n_seq <= 0:
            sd.next_block()
            continue

        for batch_start in range(0, n_seq, config.batch_size):
            batch_end = min(batch_start + config.batch_size, n_seq)

            # 1. Eleg bloque al azar
            block_idx = random.randint(0, config.num_blocks - 1)

            # 2. Muestrear σ del rango del bloque
            sigma_np = sample_sigma_in_block(block_idx, block_sigmas, gamma=config.gamma)
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
            logits = model.forward_block(block_idx, input_ids, sigma)

            # 5. Loss: weighted cross-entropy
            loss = F.cross_entropy(
                logits.view(-1, config.emb_num),
                target_ids.view(-1),
            )
            w = float(edm_weight(sigma_np, config.sigma_data))
            loss = loss * w

            # 6. Backward
            (loss / config.grad_acc).backward()

            if (batch_start // config.batch_size + 1) % config.grad_acc == 0 or batch_end >= n_seq:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                lr_curr = float(scheduler.get_last_lr()[0])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if step % 100 == 0:
                    now = time.time()
                    tps = step * config.batch_size * config.grad_acc * seq_len / max(now - t0, 0.001)
                    print(f"s{step} loss {loss.item():.4f} w={w:.3f} lr {lr_curr:.6f} "
                          f"grad {grad_norm:.3f} block={block_idx} σ={sigma_np[0]:.4f} {tps:.0f}t/s")
                    history.append({"step": step, "loss": loss.item(), "block": block_idx,
                                    "sigma": float(sigma_np[0]), "lr": lr_curr})

        sd.next_block()

    print(f"Done! {step} steps in {time.time()-t0:.1f}s")

    hist_path = os.path.join(_DIR, "train_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History: {hist_path}")

    return model


if __name__ == "__main__":
    config = Config()
    train(config)
