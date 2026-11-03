"""Estado + checkpoint: k2moe_checkpoint.pt y k2moe_state.json (local y HF)."""

import json
import os
import sys
import time

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

import torch

import hf
import plot

CKPT = os.path.join(_DIR, "k2moe_checkpoint.pt")
STATE_PATH = os.path.join(_DIR, "k2moe_state.json")
OUT_DIR = os.path.join(_DIR, "k2moe_out")


def default_state():
    return {"step": 0, "block_idx": 0, "layer_block_idx": 0,
            "steps_in_block": 0, "block_base_avg": None,
            "block_start_time": None, "done": False}


def save_state(state):
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_PATH)


def save_checkpoint(model, opt, state):
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                **state}, CKPT)
    save_state(state)
    return CKPT


def load_checkpoint():
    """Devuelve el dict del checkpoint local (o None)."""
    if not os.path.exists(CKPT):
        return None
    return torch.load(CKPT, map_location="cpu", weights_only=False)


def download_checkpoint(repo, branch, token):
    return hf.hf_download(repo, branch, "k2moe_checkpoint.pt", CKPT, token)


def push_checkpoint(api, repo, branch, token, step, with_weights_dir=False):
    msg = f"k2moe step {step}"
    hf.hf_upload_file(api, repo, branch, CKPT, "k2moe_checkpoint.pt", token, msg)
    hf.hf_upload_file(api, repo, branch, STATE_PATH, "k2moe_state.json", token, msg)
    if os.path.exists(plot.HIST_PATH):
        hf.hf_upload_file(api, repo, branch, plot.HIST_PATH, "train_history.json", token, msg)
    p = plot.plot_path(step)
    if os.path.exists(p):
        hf.hf_upload_file(api, repo, branch, p, os.path.basename(p), token, msg)
    if with_weights_dir:
        hf.hf_upload_folder(repo, branch, OUT_DIR, token, msg)
