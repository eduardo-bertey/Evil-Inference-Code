"""doc-plot.py — compara loss de laurelia-llm, filosofia y moe-mla.

Descarga los train_history.json de ScortexIA/laurelia (revisions
laurelia-llm, filosofia, moe-mla), plotea las 3 curvas de loss crudas
(step distinto cada una) y sube el PNG a ScortexIA/laurelia@doc-llm.
"""

import os
import argparse

REPO = "ScortexIA/laurelia"
BRANCH = "doc-llm"
MODELS = ["laurelia-llm", "filosofia", "moe-mla"]
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(SAVE_DIR, "compare_loss.png"))
    ap.add_argument("--no-upload", action="store_true",
                    help="solo guardar el PNG, no subir a HF")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from huggingface_hub import hf_hub_download, HfApi

    history = {}
    for model in MODELS:
        try:
            path = hf_hub_download(repo_id=REPO, filename="train_history.json",
                                   revision=model)
            import json
            with open(path) as f:
                history[model] = json.load(f)
            print(f"{model}: {len(history[model])} entries")
        except Exception as e:
            print(f"{model}: ERROR {e}")
            history[model] = []

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"laurelia-llm": "tab:blue", "filosofia": "tab:orange", "moe-mla": "tab:green"}
    for model, entries in history.items():
        if not entries:
            continue
        steps = [e["step"] for e in entries]
        losses = [e["loss"] for e in entries]
        ax.plot(steps, losses, label=model, linewidth=1.2, color=colors.get(model))
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Train loss: laurelia-llm vs filosofia vs moe-mla")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(args.out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.out}")

    if not args.no_upload:
        import getpass
        api = HfApi()
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if not token:
            print(f"\nNo HF token found. Enter token for {REPO} (write access):")
            token = getpass.getpass("Token: ").strip()
            if not token:
                raise ValueError("Token requerido para subir a HuggingFace")
        api.create_branch(repo_id=REPO, branch=BRANCH, token=token)
        api.upload_file(
            path_or_fileobj=args.out,
            path_in_repo=os.path.basename(args.out),
            repo_id=REPO,
            revision=BRANCH,
            token=token,
            commit_message="compare loss laurelia-llm/filosofia/moe-mla",
        )
        print(f"Uploaded to {REPO}@{BRANCH}")


if __name__ == "__main__":
    main()
