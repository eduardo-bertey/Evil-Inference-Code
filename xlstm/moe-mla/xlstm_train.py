"""xLSTM MoE Training — mLSTM + MoE (uses transformer tokenizer, same revision)."""
import sys, os, time, math, inspect, torch
import torch.nn.functional as F
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.join(_DIR, ".."))

from mlstm_kernels_mock import install_mock; install_mock()
from xlstm_model import xLSTMMoEModel
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
def generate_sample(model, tokenizer, device, prompt="hola", max_new=100):
    model.eval()
    x = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens=max_new, temperature=1.0, top_k=50)
    model.train()
    return tokenizer.decode(out[0].tolist())

def get_lr(step, total, warmup, lr):
    if step < warmup:
        return lr * (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return lr * (0.2 + 0.8 * (1.0 + math.cos(math.pi * t)) / 2.0)

# ─── Config ──────────────────────────────────────────────────────────────
d_model = 512
num_layers = 16
num_heads = 8
seq_len = 512
batch_size = 8
grad_accum = 8
lr = 3e-4
num_epochs = 200000
warmup_steps = 50
bpe_vocab = 32000
tok_path = os.path.join(_DIR, "tokenizer.json")
ckpt_path = os.path.join(_DIR, "xlstm_checkpoint.pt")
plot_dir = _DIR
test_mode = False
n_experts = 4
top_k = 1
n_shared = 1
expert_dim = None
capacity_factor = 1.0
z_loss_gamma = 0.001
bias_decay = 0.1
noise_std = 0.01

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    repo_id = "ScortexIA/laurelia"
    revision = "moe-mla"
    hf = pusher = None
    if not test_mode:
        hf = HFManager(repo_id=repo_id, revision=revision)
        hf._get_token()
        pusher = PeriodicPusher(hf, interval_minutes=20)
    pm = PlotManager(hf if not test_mode else None, save_dir=_DIR, plot_interval=200,
                     json_prefix="xlstm_")

    prec = input("Precision (n=f32, f=f16, b=bf16): ").strip().lower()
    dtype = {"b": torch.bfloat16, "f": torch.float16}.get(prec, torch.float32)
    print(f"  Compute: {dtype}")

    if os.path.exists(tok_path):
        tokenizer = BPEWrapper(Tokenizer.from_file(tok_path))
    elif hf and hf.tokenizer_exists(filename="tokenizer.json"):
        local_tok = hf.download_tokenizer(tok_path, remote_filename="tokenizer.json")
        tokenizer = BPEWrapper(Tokenizer.from_file(local_tok))
    else:
        wiki = download_wikipedia_50mb()
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=bpe_vocab, special_tokens=["eos_token"])
        with open(wiki, "r", encoding="utf-8") as f:
            tok.train_from_iterator([f.read()], trainer=trainer)
        tok.save(tok_path)
        tokenizer = BPEWrapper(Tokenizer.from_file(tok_path))
    print(f"Vocab: {tokenizer.vocab_size}")

    n_dense_init = 3
    n_dense_final = 3
    moe_at = list(range(n_dense_init, num_layers - n_dense_final))
    model = xLSTMMoEModel(
        vocab_size=tokenizer.vocab_size, d_model=d_model, num_layers=num_layers,
        num_heads=num_heads, moe_at=moe_at, n_experts=n_experts, top_k=top_k,
        n_shared=n_shared, expert_dim=expert_dim, capacity_factor=capacity_factor,
        z_loss_gamma=z_loss_gamma, bias_decay=bias_decay, noise_std=noise_std,
        max_seq_len=seq_len,
    ).to(device).to(dtype=dtype)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {total_params:,}")

    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2 and p.requires_grad]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2 and p.requires_grad]
    optim_groups = [
        {"params": decay_params, "weight_decay": 0.01},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device.type == "cuda"
    opt = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), fused=use_fused)
    print(f"AdamW fused={use_fused} | decay={len(decay_params)} tensors, nodecay={len(nodecay_params)}")

    # Checkpoint
    step = 0
    epoch = 0
    ckpt_block = 0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        ckpt["model"].pop("head.emb_weight", None)
        model.load_state_dict(ckpt["model"], strict=False)
        step = ckpt.get("step", 0)
        epoch = ckpt.get("epoch", 0)
        ckpt_block = ckpt.get("block", 0)
        del ckpt
        torch.cuda.empty_cache()
        print(f"Loaded checkpoint: step {step} epoch {epoch} block {ckpt_block}")
    elif hf and hf.download_checkpoint(ckpt_path, filename="xlstm_checkpoint.pt"):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        ckpt["model"].pop("head.emb_weight", None)
        model.load_state_dict(ckpt["model"], strict=False)
        step = ckpt.get("step", 0)
        epoch = ckpt.get("epoch", 0)
        ckpt_block = ckpt.get("block", 0)
        del ckpt
        torch.cuda.empty_cache()
        print(f"Loaded HF checkpoint: step {step} epoch {epoch} block {ckpt_block}")
    else:
        print("No checkpoint found, starting fresh")

    # Data
    bi = input(f"Block [{ckpt_block}]: ").strip()
    block_idx = int(bi) if bi else ckpt_block
    sd = StreamingDataset(block_mb=3.0, block_idx=block_idx)
    sd.load_tokens(tokenizer)
    n = len(sd.get_tokens())
    tokens_per_epoch = (n - seq_len - 1) // seq_len
    total_steps = (tokens_per_epoch // batch_size) * num_epochs
    print(f"Tokens per step: {batch_size * seq_len}")
    print(f"Total steps: {total_steps}")

    # Train
    model.train()
    t0 = time.time()
    last_rpt_time = t0
    last_rpt_step = step

    epoch = 0
    while True:
        tokens = sd.get_tokens()
        n_seq = (len(tokens) - seq_len - 1) // seq_len
        if n_seq <= 0:
            epoch += 1
            if epoch >= num_epochs:
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
                x_list.append(x); y_list.append(y)
            x = torch.cat(x_list, dim=0); y = torch.cat(y_list, dim=0)

            if micro == 0:
                lr_curr = get_lr(step, total_steps, warmup_steps, lr)
                for pg in opt.param_groups:
                    pg["lr"] = lr_curr
                opt.zero_grad()

            logits, aux_loss = model(x)
            loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), y.view(-1))
            loss = loss + aux_loss
            (loss / grad_accum).backward()
            micro += 1

            if micro >= grad_accum:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                opt.step()
                step += 1
                micro = 0

                if step % 10 == 0:
                    now = time.time()
                    tok = (step - last_rpt_step) * batch_size * grad_accum * seq_len
                    tps = tok / max(now - last_rpt_time, 0.001)
                    balance_strs = []
                    moe_dist = {}
                    for blk in model.blocks:
                        if hasattr(blk, "moe") and hasattr(blk.moe, "last_counts"):
                            ffn = blk.moe
                            total = ffn.last_total or 1
                            pcts = (ffn.last_counts.float() / total * 100).tolist()
                            moe_dist[f"L{blk._layer_idx}"] = pcts
                            balance_strs.append(f"L{blk._layer_idx}:{ffn.balance_str()}")
                    bal = " | ".join(balance_strs[:5])
                    print(f"e{epoch} s{step} loss {loss.item():.4f} lr {lr_curr:.6f} {tps:.0f}t/s")
                    if bal:
                        print(f"  MoE balance: {bal}")
                    last_rpt_time = now
                    last_rpt_step = step
                    pm.log(step, loss.item(), lr_curr, tps, aux_loss.item(),
                           grad_norm=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                           moe_dist=moe_dist if moe_dist else None)

                if step > 0 and step % 50 == 0:
                    t_gen = time.time()
                    sample = generate_sample(model, tokenizer, device)
                    gen_tps = 100 / (time.time() - t_gen)
                    print(f"  >>> {sample}  [{gen_tps:.0f} tok/s]")

                if not test_mode and pusher and (time.time() - pusher.last_push) >= pusher.interval:
                    state = model.state_dict()
                    state.pop("head.emb_weight", None)
                    ckpt = {"step": step, "epoch": epoch, "block": sd.block_idx, "model": state}
                    torch.save(ckpt, ckpt_path)
                    pusher.maybe_push(ckpt_path, None, None, step)
                    pm.plot(step)
                    pm.plot_grad_moe(step)
                    pm.upload(step)

        if micro > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            step += 1

        epoch += 1
        print(f"-- Epoch {epoch} done: {step} steps --")
        if epoch >= num_epochs:
            break

        sd.next_block()
        tokens = sd.get_tokens()

    if not test_mode and hf:
        ckpt = {"step": step, "epoch": epoch, "block": sd.block_idx, "model": model.state_dict()}
        torch.save(ckpt, ckpt_path)
        hf.upload_checkpoint(ckpt_path, None, tok_path, step)

    print(f"Done! {step} steps in {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
