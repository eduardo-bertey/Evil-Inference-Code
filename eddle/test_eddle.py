#!/usr/bin/env python3
"""test_eddle.py — Train + infer needle (PyTorch), push checkpoints to HF every 20 min.

Usage:
    python test_eddle.py                         # evil dataset (wiki+fineweb+tweets)
    python test_eddle.py --dataset original      # Cactus-Compute/tool-calls
    python test_eddle.py --vocab-size 16000      # custom vocab
    python test_eddle.py --d-model 256           # smaller model
"""
import argparse
import os
import sys
import time
import math

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

import torch
from eddle.train import train
from eddle.run import load_checkpoint, generate
from eddle.tokenizer import get_tokenizer, DEFAULT_MAX_ENC_LEN, DEFAULT_MAX_DEC_LEN


def run_inference(checkpoint_path, device="cpu"):
    model, config = load_checkpoint(checkpoint_path, device=device)
    tokenizer = get_tokenizer()
    model.eval()

    queries = [
        ("What is the weather in San Francisco?",
         '[{"name":"get_weather","description":"Get weather","parameters":{"location":{"type":"string","required":true}}}]'),
        ("Send an email to john@example.com",
         '[{"name":"send_email","description":"Send email","parameters":{"to":{"type":"string","required":true},"body":{"type":"string","required":true}}}]'),
    ]
    print("\nInference tests:")
    for q, t in queries:
        print(f"  Q: {q}")
        result = generate(model, tokenizer, q, tools=t, device=device)
        print(f"  A: {result}")
    del model


def main():
    parser = argparse.ArgumentParser(description="Needle PyTorch trainer + HF push")
    parser.add_argument("--dataset", choices=["evil", "original"], default="evil",
                        help="Dataset: evil (wiki+fineweb+tweets) or original (Cactus-Compute/tool-calls)")
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--num-enc-layers", type=int, default=12)
    parser.add_argument("--num-dec-layers", type=int, default=8)
    parser.add_argument("--max-enc-len", type=int, default=DEFAULT_MAX_ENC_LEN)
    parser.add_argument("--max-dec-len", type=int, default=DEFAULT_MAX_DEC_LEN)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_eddle")
    parser.add_argument("--checkpoint-interval-min", type=int, default=20)
    parser.add_argument("--hf-repo", type=str, default=None)
    parser.add_argument("--hf-revision", type=str, default="eddle")
    parser.add_argument("--precision", type=str, default="int4", choices=["int4", "int8"])
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--block-size", type=int, default=512)
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"EDDLE — Needle PyTorch")
    print(f"  Dataset:     {args.dataset}")
    print(f"  Vocab:       {args.vocab_size}")
    print(f"  Model:       d={args.d_model}, heads={args.num_heads}/{args.num_kv_heads}kv")
    print(f"  Layers:      {args.num_enc_layers} enc / {args.num_dec_layers} dec")
    print(f"  Device:      {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"{'='*50}\n")

    ckpt = train(
        dataset_mode=args.dataset,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        num_enc_layers=args.num_enc_layers,
        num_dec_layers=args.num_dec_layers,
        max_enc_len=args.max_enc_len,
        max_dec_len=args.max_dec_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        muon_lr=args.muon_lr,
        seed=args.seed,
        eval_every=args.eval_every,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval_min=args.checkpoint_interval_min,
        hf_repo=args.hf_repo,
        hf_revision=args.hf_revision,
        precision=args.precision,
        contrastive_weight=args.contrastive_weight,
        block_size=args.block_size,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_inference(ckpt, device=device)


if __name__ == "__main__":
    main()
