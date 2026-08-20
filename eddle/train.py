import math
import os
import pickle
import time
import sys

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from .architecture import (
    SimpleAttentionNetwork, TransformerConfig,
    make_causal_mask, make_padding_mask,
    make_packing_mask, make_causal_packing_mask, make_cross_packing_mask,
)
from .tokenizer import get_tokenizer, train_tokenizer, TOKENIZER_PREFIX, DEFAULT_MAX_ENC_LEN, DEFAULT_MAX_DEC_LEN
from .optim import Muon, get_param_groups, wsd_schedule
from .dataset import (
    load_tool_calls_from_hf, prepare_tool_call_pairs,
    make_streaming_loader, ToolCallDataset, StreamTextDataset,
    get_wiki_blocks, get_fineweb_blocks, get_tweet_blocks,
)
from .quantize import quantize_params

_DIR = os.path.dirname(os.path.abspath(__file__))

LOSS_WEIGHT_MAP = np.array([1.0, 3.0, 2.0, 1.5], dtype=np.float32)


def train(
    dataset_mode="evil",
    vocab_size=8192,
    d_model=512,
    num_heads=8,
    num_kv_heads=4,
    num_enc_layers=12,
    num_dec_layers=8,
    d_ff=None,
    max_enc_len=1024,
    max_dec_len=512,
    batch_size=8,
    epochs=1,
    lr=3e-4,
    muon_lr=0.02,
    warmup_ratio=0.05,
    decay_ratio=0.15,
    eval_every=100,
    checkpoint_dir="checkpoints",
    checkpoint_interval_min=20,
    hf_repo=None,
    hf_revision="eddle",
    seed=42,
    precision="int4",
    contrastive_weight=0.1,
    block_size=512,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    if d_ff is None:
        d_ff = d_model * 4

    os.makedirs(checkpoint_dir, exist_ok=True)

    tokenizer_path = TOKENIZER_PREFIX + ".model"
    if not os.path.exists(tokenizer_path):
        print("Training tokenizer...")
        if dataset_mode == "evil":
            blocks = get_wiki_blocks(max_bytes=50_000_000)
            corpus_path = os.path.join(_DIR, "_tokenizer_corpus.txt")
            with open(corpus_path, "w") as f:
                for b in blocks:
                    f.write(b)
            train_tokenizer(vocab_size=vocab_size, corpus_path=corpus_path)
            os.remove(corpus_path)
        else:
            samples = load_tool_calls_from_hf("train", max_samples=5000)
            corpus_path = os.path.join(_DIR, "_tokenizer_corpus.txt")
            with open(corpus_path, "w") as f:
                for ex in samples:
                    f.write(ex.get("query", "") + "\n")
                    f.write(ex.get("tools", "") + "\n")
                    f.write(ex.get("answers", "") + "\n")
            train_tokenizer(vocab_size=vocab_size, corpus_path=corpus_path)
            os.remove(corpus_path)

    tokenizer = get_tokenizer()
    actual_vocab_size = tokenizer.vocab_size
    print(f"Tokenizer vocab: {actual_vocab_size}")

    config = TransformerConfig(
        vocab_size=actual_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        num_encoder_layers=num_enc_layers,
        num_decoder_layers=num_dec_layers,
        d_ff=d_ff,
        max_seq_len=max(max_enc_len, max_dec_len),
        dtype="bfloat16" if device != "cpu" else "float32",
        contrastive_dim=128,
        no_feedforward=True,
    )

    model = SimpleAttentionNetwork(config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    if dataset_mode == "evil":
        loader = make_streaming_loader(tokenizer, block_size=block_size, batch_size=batch_size, mode="evil")
        total_steps = epochs * len(loader)
    else:
        samples = load_tool_calls_from_hf("train", max_samples=None)
        enc, dec_in, dec_tgt = prepare_tool_call_pairs(samples, tokenizer, max_enc_len, max_dec_len)
        ds = ToolCallDataset(enc, dec_in, dec_tgt)
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
        total_steps = epochs * len(loader)

    warmup_steps = max(1, int(total_steps * warmup_ratio))
    param_groups = get_param_groups(model, muon_lr=muon_lr, adam_lr=lr)

    optimizer_muon = Muon(param_groups[0], lr=muon_lr, momentum=0.95, weight_decay=0.01)
    optimizer_adam = torch.optim.AdamW(param_groups[1], lr=lr, betas=(0.9, 0.95))
    schedule = wsd_schedule(1.0, total_steps, warmup_steps, decay_ratio)

    global_step = 0
    train_start = time.time()
    losses = []

    print(f"\nTraining: {total_steps} steps, {warmup_steps} warmup, lr={lr}, muon_lr={muon_lr}")

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in pbar:
            factor = schedule(global_step)
            for pg in optimizer_muon.param_groups:
                pg["lr"] = muon_lr * factor
            for pg in optimizer_adam.param_groups:
                pg["lr"] = lr * factor

            if dataset_mode == "evil":
                x, y, _ = batch
                x, y = x.to(device), y.to(device)
                logits = model(x, y[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, actual_vocab_size), y[:, 1:].reshape(-1))
            else:
                enc, dec_in, dec_tgt = batch
                enc, dec_in, dec_tgt = enc.to(device), dec_in.to(device), dec_tgt.to(device)
                logits = model(enc, dec_in[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, actual_vocab_size), dec_tgt[:, 1:].reshape(-1))

            optimizer_muon.zero_grad()
            optimizer_adam.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_muon.step()
            optimizer_adam.step()

            losses.append(loss.item())
            global_step += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}", step=global_step)

            if global_step % eval_every == 0:
                avg = np.mean(losses[-eval_every:])
                ppl = math.exp(min(avg, 20))
                print(f"\n  Step {global_step}: loss={avg:.4f} ppl={ppl:.2f}")

    total_time = (time.time() - train_start) / 60
    avg_loss = np.mean(losses) if losses else 0
    print(f"\nDone: {global_step} steps, loss={avg_loss:.4f}, time={total_time:.1f}min")

    ckpt_path = os.path.join(checkpoint_dir, f"needle_{global_step}.pt")
    torch.save({
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "config": config.__dict__,
        "step": global_step,
        "loss": avg_loss,
    }, ckpt_path)
    print(f"Saved: {ckpt_path}")
    return ckpt_path
