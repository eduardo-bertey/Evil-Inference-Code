"""Test de convergencia: filosofia con 4 capas, BPE 16k, seq 2k, batch 8.
Entrena sobre los 50MB de wiki ES. Al final guarda train_history.json y
plot_test_filo.png.
"""

import sys, os, time, json, math, torch

_DIR = os.path.dirname(os.path.abspath(__file__))
_FILO = os.path.join(_DIR, "..", "filosofia")
sys.path.insert(0, _DIR)
sys.path.insert(0, _FILO)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from model import LLM, Config
from dataset import download_wikipedia_50mb
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


class BPEWrapper:
    def __init__(self, tok):
        self.tokenizer = tok
        self.vocab_size = tok.get_vocab_size()
    def encode(self, text):
        return self.tokenizer.encode(text).ids
    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)


def main():
    config = Config()
    config.dim = 512
    config.heads = 16
    config.layers = 4
    config.emb_num = 16384
    config.block_size = 2048
    config.batch_size = 8
    config.grad_acc = 1
    config.warm_up = 20
    config.learning_rate = 3e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: dim={config.dim} lay={config.layers} heads={config.heads} kv={config.kv_groups} "
          f"seq={config.block_size} bs={config.batch_size} lr={config.learning_rate}")

    tok_path = os.path.join(_DIR, "tokenizer_test_16k.json")
    wiki = os.path.join(_DIR, "wiki_tokenizer_50mb.txt")
    download_wikipedia_50mb(wiki)

    def iter_chunks(path, mb=5):
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(mb * 1024 * 1024)
                if not chunk:
                    break
                yield chunk.decode("utf-8", errors="ignore")

    if not os.path.exists(tok_path):
        print("Entrenando BPE 16k sobre los 50MB de wiki...")
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=config.emb_num, special_tokens=["eos_token"])
        tok.train_from_iterator(iter_chunks(wiki), trainer=trainer)
        tok.save(tok_path)
    tokenizer = BPEWrapper(Tokenizer.from_file(tok_path))
    config.emb_num = tokenizer.vocab_size
    print(f"Vocab: {tokenizer.vocab_size}")
    print(f"Wiki bytes: {os.path.getsize(wiki)}")

    model = LLM(config).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = model.configure_optimizers(config.weight_decay, config.learning_rate, config.betas, "cuda")

    print("Generando ids de a 5MB...")
    tok_chunks = []
    for chunk in iter_chunks(wiki):
        tok_chunks.append(torch.tensor(tokenizer.encode(chunk), dtype=torch.long))
    tokens = torch.cat(tok_chunks)
    print(f"Tokens: {len(tokens):,}")
    seq_len = config.block_size
    n_seq = (len(tokens) - seq_len - 1) // seq_len
    steps_per_epoch = max(1, n_seq // config.batch_size)
    epochs = 1
    total_steps = steps_per_epoch * epochs
    print(f"steps_per_epoch={steps_per_epoch} epochs={epochs} total_steps={total_steps}")

    num_warmup = config.warm_up
    num_decay = int(total_steps * 0.15)
    num_stable = max(0, total_steps - num_warmup - num_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / max(1, num_warmup)) if step < num_warmup else 1.0,
    )

    history = []
    model.train()
    t0 = time.time()
    step = 0
    done = False
    while not done:
        for batch_start in range(0, n_seq, config.batch_size):
            if step >= total_steps:
                done = True
                break
            batch_end = min(batch_start + config.batch_size, n_seq)
            x_list, y_list = [], []
            for i in range(batch_start, batch_end):
                idx = i * seq_len
                x = tokens[idx:idx + seq_len].unsqueeze(0).to(device)
                y = tokens[idx + 1:idx + seq_len + 1].unsqueeze(0).to(device)
                x_list.append(x)
                y_list.append(y)
            x = torch.cat(x_list, dim=0)
            y = torch.cat(y_list, dim=0)

            logits, loss = model(x, labels=y)
            loss.backward()
            loss_val = loss.item()
            del logits, loss

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            lr_curr = scheduler.get_last_lr()[0]
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            if step % 10 == 0 or done:
                now = time.time()
                tps = step * config.batch_size * seq_len / max(now - t0, 0.001)
                print(f"s{step} loss {loss_val:.4f} lr {lr_curr:.6f} grad {grad_norm:.3f} {tps:.0f}t/s")
                history.append({"step": step, "loss": loss_val, "lr": lr_curr, "grad_norm": grad_norm})

    print(f"Done! {step} steps in {time.time()-t0:.1f}s")

    hist_path = os.path.join(_DIR, "train_history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"train_history.json: {hist_path}")

    plot_path = os.path.join(_DIR, "plot_test_filo.png")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        steps = [e["step"] for e in history]
        losses = [e["loss"] for e in history]
        plt.figure(figsize=(10, 5))
        plt.plot(steps, losses, label="loss", color="tab:blue")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title(f"Test filosofia convergencia ({config.layers} capas, seq {config.block_size})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(plot_path, dpi=100, bbox_inches="tight")
        print(f"Plot: {plot_path}")
    except ImportError as e:
        print(f"No matplotlib: {e}")

    print("Subiendo a ScortexIA/laurelia@doc-llm...")
    from huggingface_hub import HfApi
    import getpass
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        token = getpass.getpass("HF token (con write a ScortexIA/laurelia): ").strip()
    api = HfApi(token=token)
    for f in [hist_path, plot_path]:
        api.upload_file(
            path_or_fileobj=f,
            path_in_repo=os.path.basename(f),
            repo_id="ScortexIA/laurelia",
            revision="doc-llm",
            commit_message=f"test_filo step {step}",
        )
        print(f"  Subido {os.path.basename(f)}")


if __name__ == "__main__":
    main()
