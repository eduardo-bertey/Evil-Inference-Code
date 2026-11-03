"""Historial de entrenamiento + plot (train_history.json, plot_train_step_N.png)."""

import json
import os
import time

_DIR = os.path.dirname(os.path.abspath(__file__))
HIST_PATH = os.path.join(_DIR, "train_history.json")


def plot_path(step):
    return os.path.join(_DIR, f"plot_train_step_{step}.png")


def log_history(step, loss, lr, tps, grad_norm, layer_block):
    hist = []
    if os.path.exists(HIST_PATH):
        try:
            with open(HIST_PATH) as f:
                hist = json.load(f)
        except Exception:
            hist = []
    hist.append({"step": step, "loss": loss, "lr": lr, "tps": tps,
                 "grad_norm": grad_norm, "layer_block": layer_block,
                 "time": time.time()})
    with open(HIST_PATH, "w") as f:
        json.dump(hist, f, indent=2)


def plot(step):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    if not os.path.exists(HIST_PATH):
        return None
    with open(HIST_PATH) as f:
        hist = json.load(f)
    if len(hist) < 2:
        return None
    steps = [e["step"] for e in hist]
    losses = [e["loss"] for e in hist]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, losses, color="tab:blue")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    for i in range(1, len(hist)):
        if hist[i].get("layer_block") != hist[i - 1].get("layer_block"):
            ax.axvline(hist[i]["step"], color="tab:red", linestyle="--", alpha=0.5)
    ax.set_title(f"k2moe loss (step {step}, rojo = salto de bloque)")
    ax.grid(True, alpha=0.3)
    path = plot_path(step)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path
