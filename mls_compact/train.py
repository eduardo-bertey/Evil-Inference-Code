"""Laurelia LLM Train — Dense GQA, basado en LLM_350M_DENSE.

bf16, AdamW fused, WSD schedule, HF upload. Datos: pares HLS (hls.lsh_hash)
en vez de dataset: entrada vector de 23 (hash 0/1) -> salida vector de 24
(dato 0/1). Sin secuencia ni autoregresion: seq=1, una pasada por los blocks.
"""

import sys, os, time, math, random, torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
from model import LLM, Config
from huggingface import HFManager, PeriodicPusher
from hls import BITS_IN, BITS_WIN, lsh_hash

SALT = 0x5A17


def bits_of(value: int, n: int):
    return [(value >> (n - 1 - i)) & 1 for i in range(n)]


def make_pool(pool_size: int, seed: int):
    r = random.Random(seed)
    pairs = []
    for _ in range(pool_size):
        a1, b2 = lsh_hash(r, salt=SALT)
        pairs.append((bits_of(a1, BITS_WIN), bits_of(b2, BITS_IN)))
    return pairs


def make_batch(pairs, bs, rng, device):
    x = torch.empty((bs, BITS_WIN), dtype=torch.float32)
    y = torch.empty((bs, BITS_IN, 1), dtype=torch.float32)
    for j in range(bs):
        h, d = pairs[rng.randrange(len(pairs))]
        x[j] = torch.tensor(h, dtype=torch.float32)
        y[j, :, 0] = torch.tensor(d, dtype=torch.float32)
    return x.to(device), y.to(device)


def make_batch_fresh(bs, rng, device):
    """Pares nuevos cada batch: datos infinitos (el hash sale del dato)."""
    x = torch.empty((bs, BITS_WIN), dtype=torch.float32)
    y = torch.empty((bs, BITS_IN, 1), dtype=torch.float32)
    for j in range(bs):
        a1, b2 = lsh_hash(rng, salt=SALT)
        x[j] = torch.tensor(bits_of(a1, BITS_WIN), dtype=torch.float32)
        y[j, :, 0] = torch.tensor(bits_of(b2, BITS_IN), dtype=torch.float32)
    return x.to(device), y.to(device)


def forward_bits(model, x):
    """Vector 23 -> in_proj -> (B,1,dim) -> blocks -> head -> (B,24,1)."""
    h = model.in_proj(x).unsqueeze(1)
    for block in model.blocks:
        h = block(h)
    h = model.norm_f(h)
    return model.bit_head(h).squeeze(1).unsqueeze(-1)


@torch.no_grad()
def evaluate(model, pairs, device, n=256):
    model.eval()
    correct = total = 0
    for i in range(0, min(n, len(pairs)), config.batch_size):
        batch = pairs[i:i + config.batch_size]
        x = torch.tensor([h for h, _ in batch], dtype=torch.float32, device=device)
        y = torch.tensor([d for _, d in batch], dtype=torch.float32, device=device).unsqueeze(-1)
        pred = (forward_bits(model, x) > 0).float()
        correct += int((pred == y).sum())
        total += y.numel()
    model.train()
    return correct / max(total, 1)


@torch.no_grad()
def show_sample(model, pairs, device, idx=0):
    model.eval()
    h, d = pairs[idx]
    x = torch.tensor([h], dtype=torch.float32, device=device)
    p1 = torch.sigmoid(forward_bits(model, x).float().squeeze(-1))[0]
    print("  hash :", "".join(str(b) for b in h))
    print("  real :", "".join(str(b) for b in d))
    print("  P(1) :", " ".join(f"{v:.3f}" for v in p1.tolist()))
    print("  P(0) :", " ".join(f"{1 - v:.3f}" for v in p1.tolist()))
    model.train()


def get_wsd_schedule(optimizer, num_warmup, num_stable, num_decay, min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < num_warmup:
            return float(step) / float(max(1, num_warmup))
        if step < num_warmup + num_stable:
            return 1.0
        progress = float(step - num_warmup - num_stable) / float(max(1, num_decay))
        progress = min(1.0, progress)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


config = Config()
config.dim = 512
config.layers = 12
config.heads = 8
config.kv_groups = 4
config.ffn_dim = 2048
config.batch_size = 256
config.grad_acc = 2
ckpt_path = os.path.join(_DIR, "checkpoint.pt")
plot_interval = 256


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    repo_id = "ScortexIA/laurelia"
    revision = "hls2"
    hf = HFManager(repo_id=repo_id, revision=revision)
    hf._get_token()
    hf.login_global()
    pusher = PeriodicPusher(hf, interval_minutes=20)

    prec = input("Precision (n=f32, b=bf16) [n]: ").strip().lower()
    dtype = torch.bfloat16 if prec == "b" else torch.float32
    print(f"  Compute: {dtype}")

    steps = int(input("Steps [2000]: ").strip() or 2000)
    lr_in = input("lr [3e-4]: ").strip()
    if lr_in:
        config.learning_rate = float(lr_in)

    val_pairs = make_pool(64, seed=99)
    rng = random.Random()
    print(f"Datos HLS: infinitos (par nuevo por batch, semilla {time.time_ns()}) | "
          f"entrada vector {BITS_WIN} -> salida vector {BITS_IN}")

    model = LLM(config)
    model.in_proj = torch.nn.Linear(BITS_WIN, config.dim, bias=False)
    model.bit_head = torch.nn.Linear(config.dim, BITS_IN, bias=False)
    model.embeddings.weight.requires_grad_(False)
    model.lm_head.weight.requires_grad_(False)
    model = model.to(device).to(dtype=dtype)

    optimizer = model.configure_optimizers(config.weight_decay, config.learning_rate, config.betas, "cuda")

    step = 0
    epoch = 0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt["model"], strict=False)
        step = ckpt.get("step", 0)
        epoch = ckpt.get("epoch", 0)
        del ckpt
        torch.cuda.empty_cache()
        print(f"Loaded checkpoint: step {step} epoch {epoch}")

    num_warmup = config.warm_up
    num_decay = max(1, int(steps * 0.15))
    num_stable = max(0, steps - num_warmup - num_decay)
    scheduler = get_wsd_schedule(optimizer, num_warmup, num_stable, num_decay)

    layer_p = sum(p.numel() for b in model.blocks for p in b.parameters())
    print(f"Params: {config.layers}capas={layer_p:,} + in_proj={model.in_proj.weight.numel():,} + bit_head={model.bit_head.weight.numel():,}")
    print(f"dim={config.dim} lay={config.layers} heads={config.heads} kv={config.kv_groups} bs={config.batch_size} ga={config.grad_acc} lr={config.learning_rate}")
    print(f"Entrada: vector {BITS_WIN} (hash 0/1) | Salida: vector {BITS_IN} (dato 0/1)")

    loss_fct = torch.nn.BCEWithLogitsLoss()
    model.train()
    t0 = time.time()
    last_rpt_time = t0
    last_rpt_step = 0
    micro = 0

    while step < steps:
        x, y = make_batch_fresh(config.batch_size, rng, device)
        logits = forward_bits(model, x)
        loss = loss_fct(logits.float(), y)
        (loss / config.grad_acc).backward()
        loss_val = loss.item()
        del logits, loss
        micro += 1

        if micro % config.grad_acc == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            lr_curr = scheduler.get_last_lr()[0]
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            if step == 1 or step % 10 == 0:
                now = time.time()
                tok = (step - last_rpt_step) * config.batch_size * config.grad_acc
                tps = tok / max(now - last_rpt_time, 0.001)
                acc = evaluate(model, val_pairs, device)
                print(f"s{step} loss {loss_val:.4f} acc_bits {acc:.3f} lr {lr_curr:.6f} grad {grad_norm:.3f} {tps:.0f}ej/s")
                last_rpt_time = now
                last_rpt_step = step

            if step % 50 == 0:
                state = model.state_dict()
                ckpt = {"step": step, "epoch": epoch, "block": 0, "model": state}
                torch.save(ckpt, ckpt_path)

            if pusher and (time.time() - pusher.last_push) >= pusher.interval:
                state = model.state_dict()
                ckpt = {"step": step, "epoch": epoch, "block": 0, "model": state}
                torch.save(ckpt, ckpt_path)
                pusher.maybe_push(ckpt_path, None, None, step)

        epoch += 1

    state = model.state_dict()
    ckpt = {"step": step, "epoch": epoch, "block": 0, "model": state}
    torch.save(ckpt, ckpt_path)
    hf.upload_checkpoint(ckpt_path, None, None, step)
    print(f"Acc final: {evaluate(model, val_pairs, device):.3f}")
    show_sample(model, val_pairs, device)
    print(f"Done! {step} steps in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
