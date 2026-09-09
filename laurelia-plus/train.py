"""Laurelia Plus Train — Denso MLA + MoSE por router + TST opcional.

bf16, streaming dataset, AdamW fused, WSD schedule, HF upload.
"""

import sys, os, time, math, inspect, torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.join(_DIR, ".."))
from model import LLM, Config
from tst import TSTConfig, mtp_weights_for_step
import importlib
train_data = importlib.import_module("train-data")
from wikipedia import download_wikipedia_50mb
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
tok_path = os.path.join(_DIR, "tokenizer.json")
plot_interval = 256


def main():
    test_mode = len(sys.argv) > 1 and sys.argv[1].endswith(".txt")
    txt_path = sys.argv[1] if test_mode else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    repo_id = "ScortexIA/laurelia"
    revision = "laurelia-plus"
    hf = pusher = None
    if not test_mode:
        hf = HFManager(repo_id=repo_id, revision=revision)
        hf._get_token()
        hf.login_global()
        pusher = PeriodicPusher(hf, interval_minutes=20)
    pm = PlotManager(hf if not test_mode else None, save_dir=_DIR, plot_interval=plot_interval)

    if test_mode:
        dtype = torch.float32
        tst_cfg = TSTConfig(enabled=False)
    else:
        prec = input("Precision (n=f32, b=bf16): ").strip().lower()
        dtype = torch.bfloat16 if prec == "b" else torch.float32
        tst_on = input("Habilitar TST? (s/n): ").strip().lower()
        tst_cfg = TSTConfig(enabled=(tst_on == "s"), n_predict=4, mode="fijo")
        config.batch_size = 8 if tst_cfg.enabled else 6
    print(f"  Compute: {dtype}")
    print(f"  TST: {'ON' if tst_cfg.enabled else 'OFF'} (n={tst_cfg.n_predict} base={tst_cfg.base} {tst_cfg.mode} fold={4 if tst_cfg.enabled else 1})")

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
            hf.upload_tokenizer(tok_path, os.path.join(_DIR, "tokenizer_config.json"))
        else:
            sys.exit("No tokenizer found")
    config.emb_num = tokenizer.vocab_size
    print(f"Vocab: {tokenizer.vocab_size}")

    model = LLM(config).to(device).to(dtype=dtype)

    optimizer = model.configure_optimizers(config.weight_decay, config.learning_rate, config.betas, "cuda")

    ckpt_path = os.path.join(_DIR, "checkpoint.pt")
    step = 0
    epoch = 0
    ckpt_block = 0
    tst_tokens = 0
    dense_tokens = 0

    if not test_mode:
        loaded = False
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu')
            ckpt["model"].pop("head.emb_weight", None)
            model.load_state_dict(ckpt["model"], strict=False)
            step = ckpt.get("step", 0)
            epoch = ckpt.get("epoch", 0)
            ckpt_block = ckpt.get("block", 0)
            tst_tokens = ckpt.get("tst_tokens", 0)
            dense_tokens = ckpt.get("dense_tokens", 0)
            del ckpt
            torch.cuda.empty_cache()
            print(f"Loaded checkpoint: step {step} epoch {epoch} block {ckpt_block} tst {tst_tokens:,} dense {dense_tokens:,}")
            loaded = True
        elif hf and hf.download_checkpoint(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu')
            ckpt["model"].pop("head.emb_weight", None)
            model.load_state_dict(ckpt["model"], strict=False)
            step = ckpt.get("step", 0)
            epoch = ckpt.get("epoch", 0)
            ckpt_block = ckpt.get("block", 0)
            tst_tokens = ckpt.get("tst_tokens", 0)
            dense_tokens = ckpt.get("dense_tokens", 0)
            del ckpt
            torch.cuda.empty_cache()
            print(f"Loaded HF checkpoint: step {step} epoch {epoch} block {ckpt_block} tst {tst_tokens:,} dense {dense_tokens:,}")
            loaded = True

        if loaded:
            print("\n── Generation test ──")
            for p in ["hola", "que es la inteligencia artificial", "en un lugar de la mancha"]:
                sample = generate_sample(model, tokenizer, device, prompt=p, max_new=50)
                print(f"  [{p}] → {sample}")
            print(f"  Widths router (full/75/50/25): {model.width_report()}")
            print("── End test ──\n")

    if test_mode:
        with open(txt_path, "r", encoding="utf-8") as f:
            all_tokens = tokenizer.encode(f.read())
        n = len(all_tokens)
        total_steps = ((n - config.block_size - 1) // (config.batch_size * config.block_size)) * 10
        seq_len = config.block_size
    else:
        bi = input(f"Block [{ckpt_block}]: ").strip()
        block_idx = int(bi) if bi else ckpt_block
        sd = train_data.TrainData(block_idx=block_idx)
        sd.load_tokens(tokenizer)
        tokens = sd.get_tokens()
        n = len(tokens)
        seq_len = config.block_size * (2 if tst_cfg.enabled else 1)
        n_seq = (n - seq_len - 1) // seq_len
        steps_per_epoch = n_seq // config.batch_size
        total_steps = steps_per_epoch * 200000
        epochs_do = 200000

    num_warmup = config.warm_up
    num_decay = int(total_steps * 0.15)
    num_stable = total_steps - num_warmup - num_decay
    scheduler = get_wsd_schedule(optimizer, num_warmup, num_stable, num_decay)

    emb_p = model.embeddings.weight.numel()
    layer_p = sum(p.numel() for b in model.blocks for p in b.parameters())
    norm_p = model.norm_f.weight.numel()
    head_p = model.lm_head.weight.numel()
    print(f"Params: emb={emb_p:,} + {config.layers}capas={layer_p:,} + norm={norm_p} = {emb_p + layer_p + norm_p:,}")
    print(f"dim={config.dim} lay={config.layers} heads={config.heads} kv={config.kv_groups} seq={seq_len} bs={config.batch_size} ga={config.grad_acc} lr={config.learning_rate}")
    gqa_cpt = config.kv_groups * (config.dim // config.heads) * 2
    mla_cpt = config.mla_d_c + config.mla_d_rotate
    print(f"MLA: d_c={config.mla_d_c} d_c1={config.mla_d_c1} d_rot={config.mla_d_rotate} | cache: {gqa_cpt}→{mla_cpt}B/tok ({100*(1-mla_cpt/gqa_cpt):.0f}%)")
    print(f"MoSE: anchos={config.mose_widths} cost_w={config.mose_cost_w} tau={config.mose_tau}")
    print(f"Tokens: {n:,}")

    model.train()
    t0 = time.time()
    last_rpt_time = t0
    last_rpt_step = 0

    while True:
        if test_mode:
            tokens = all_tokens
            n_seq = (len(tokens) - seq_len - 1) // seq_len
        else:
            tokens = sd.get_tokens()
            n_seq = (len(tokens) - seq_len - 1) // seq_len

        if n_seq <= 0:
            epoch += 1
            if not test_mode:
                sd.next_block()
            continue

        for batch_start in range(0, n_seq, config.batch_size):
            if step >= total_steps:
                break
            batch_end = min(batch_start + config.batch_size, n_seq)
            x_list, y_list = [], []
            for i in range(batch_start, batch_end):
                idx = i * seq_len
                x = torch.tensor([tokens[idx + j] for j in range(seq_len)], dtype=torch.long, device=device).unsqueeze(0)
                y = torch.tensor([tokens[idx + j + 1] for j in range(seq_len)], dtype=torch.long, device=device).unsqueeze(0)
                x_list.append(x)
                y_list.append(y)
            x = torch.cat(x_list, dim=0)
            y = torch.cat(y_list, dim=0)

            tok_n = x.numel()
            mtp_w = None
            fold = 1
            if tst_cfg.enabled:
                FOLD = 4
                w = torch.tensor([1.0, 0.5, 0.25, 0.125], device=x.device)
                w = w / w.sum()  # media: paridad de escala con CE (estilo paper)
                mtp_w = w
                fold = FOLD
                tst_tokens += tok_n
                mtp_tag = f"fold{FOLD} {[round(float(v), 3) for v in w.tolist()]}"
            else:
                dense_tokens += tok_n
                mtp_tag = ""
            logits, loss, aux = model(x, labels=y, mtp_weights=mtp_w, fold=fold)
            ((loss + aux) / config.grad_acc).backward()
            loss_val = loss.item()
            del logits, loss

            if (batch_start // config.batch_size + 1) % config.grad_acc == 0 or batch_end >= n_seq:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                lr_curr = scheduler.get_last_lr()[0]
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if step % 10 == 0:
                    now = time.time()
                    tok = (step - last_rpt_step) * config.batch_size * config.grad_acc * seq_len
                    tps = tok / max(now - last_rpt_time, 0.001)
                    wr = model.width_report()
                    print(f"s{step} loss {loss_val:.4f} lr {lr_curr:.6f} grad {grad_norm:.3f} {tps:.0f}t/s tst {tst_tokens/1e6:.1f}M/{dense_tokens/1e6:.1f}M W{wr} {mtp_tag}")
                    last_rpt_time = now
                    last_rpt_step = step
                    pm.log(step, loss_val, lr_curr, tps)

                if not test_mode and step % 50 == 0:
                    sample = generate_sample(model, tokenizer, device)
                    print(f"  >>> {sample}")

                if not test_mode and pusher and (time.time() - pusher.last_push) >= pusher.interval:
                    state = model.state_dict()
                    state.pop("head.emb_weight", None)
                    ckpt = {"step": step, "epoch": epoch, "block": sd.block_idx if not test_mode else 0,
                            "tst_tokens": tst_tokens, "dense_tokens": dense_tokens, "model": state}
                    torch.save(ckpt, ckpt_path)
                    pusher.maybe_push(ckpt_path, None, tok_path, step)
                    pm.plot(step)
                    pm.upload(step)

        epoch += 1
        if not test_mode:
            sd.next_block()

    if not test_mode and hf:
        ckpt = {"step": step, "epoch": epoch, "block": sd.block_idx,
                "tst_tokens": tst_tokens, "dense_tokens": dense_tokens,
                "model": model.state_dict()}
        torch.save(ckpt, ckpt_path)
        hf.upload_checkpoint(ckpt_path, tok_path, step)

    print(f"Done! {step} steps in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
