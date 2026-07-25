import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from model import Config, LLM


# ─────────────────────────────────────────────────────────────────────────────
# Sampling helpers
# ─────────────────────────────────────────────────────────────────────────────
def sample_next_token(
    logits: torch.Tensor,       # [1, vocab]
    context_ids: torch.Tensor,  # [1, T]  — used for repetition penalty
    temperature: float = 1.0,
    top_k: int = 30,
    top_p: float = 0.8,
    repetition_penalty: float = 1.15,
) -> torch.Tensor:              # returns [1, 1]
    """
    Proper nucleus (top-p), top-k sampling with temperature and repetition penalty.
    Includes the fix for negative logits during repetition penalty.
    """
    logits = logits.clone().float()           # work in fp32 for numerical safety

    # ── 1. Repetition penalty ─────────────────────────────────────────────────
    if repetition_penalty != 1.0:
        for token_id in set(context_ids[0].tolist()):
            # FIX: If logit is positive, reduce it; if negative, make it more negative
            if logits[0, token_id] > 0:
                logits[0, token_id] /= repetition_penalty
            else:
                logits[0, token_id] *= repetition_penalty

    # ── 2. Temperature ────────────────────────────────────────────────────────
    if temperature != 1.0:
        logits = logits / max(temperature, 1e-8)

    # ── 3. Top-K ──────────────────────────────────────────────────────────────
    if top_k is not None:
        values, indices = torch.topk(logits, top_k)
        logits = torch.full_like(logits, float('-inf'))
        logits.scatter_(1, indices, values)

    # ── 4. Top-P (nucleus) sampling ───────────────────────────────────────────
    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = probs.cumsum(dim=-1)

        cutoff = cumulative_probs > top_p
        cutoff[:, 1:] = cutoff[:, :-1].clone()
        cutoff[:, 0] = False

        sorted_logits[cutoff] = float('-inf')
        logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)

    # ── 5. Sample ─────────────────────────────────────────────────────────────
    probs    = F.softmax(logits, dim=-1)
    next_tok = torch.multinomial(probs, num_samples=1)   # [1, 1]
    return next_tok


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation function
# ─────────────────────────────────────────────────────────────────────────────
def generate_and_plot_activations(
    prompt_text:        str,
    ckpt_path:          str  = None,
    num_tokens:         int  = 50,
    temperature:        float = 1.0,
    top_k:              int   = 30,
    top_p:              float = 0.8,
    repetition_penalty: float = 1.15,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading Tokenizer and Model...")

    # ── Tokenizer & model ─────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained("data_mixed/custom_multilingual_tokenizer")
    tokenizer.pad_token = tokenizer.eos_token

    config = Config()
    model  = LLM(config).to(device)

    if ckpt_path:
        try:
            state_dict       = torch.load(ckpt_path, map_location=device)
            # strip torch.compile prefix if checkpoint was saved from compiled model
            clean_state_dict = {
                k.replace("_orig_mod.", ""): v for k, v in state_dict.items()
            }
            model.load_state_dict(clean_state_dict, strict=True)
            print(f"✅ Model loaded from {ckpt_path}")
        except FileNotFoundError:
            print("⚠️  Checkpoint not found — running with random weights.")

    model.eval()

    # ── Tokenise prompt ───────────────────────────────────────────────────────
    input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)

    # ── Activation storage & hooks ────────────────────────────────────────────
    attn_activations = {i: [] for i in range(config.layers)}
    mlp_activations  = {i: [] for i in range(config.layers)}
    hooks            = []

    def make_hook(layer_idx, storage):
        def hook(module, inp, out):
            tensor_out = out[0] if isinstance(out, tuple) else out
            # Only the last token position — the one being predicted
            norm = tensor_out[:, -1, :].norm(p=2, dim=-1).mean().item()
            storage[layer_idx].append(norm)
        return hook

    for i, block in enumerate(model.blocks):
        hooks.append(block.attn.register_forward_hook(make_hook(i, attn_activations)))
        hooks.append(block.mlp.register_forward_hook(make_hook(i, mlp_activations)))

    # ── Generation loop ───────────────────────────────────────────────────────
    print(f"\nGenerating {num_tokens} tokens for prompt: '{prompt_text}'")
    print(f"Settings → temp={temperature}, top_k={top_k}, top_p={top_p}, rep_penalty={repetition_penalty}")
    print("\nOutput: ", end="", flush=True)

    with torch.inference_mode():
        for _ in range(num_tokens):
            # Truncate context to block_size to avoid OOM on long sequences
            context = input_ids[:, -config.block_size:]

            out    = model(input_ids=context)
            logits = out.logits                    # [1, T, vocab]

            # ── FIX: use correct robust sampling ──────────────────────────────
            next_tok = sample_next_token(
                logits[:, -1, :],                  # [1, vocab] — last position
                context_ids=input_ids,             # full context for rep penalty
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )                                      # [1, 1]

            # Print token as it generates
            decoded = tokenizer.decode(next_tok[0], skip_special_tokens=True)
            print(decoded, end="", flush=True)

            # EOS check
            if next_tok.item() == tokenizer.eos_token_id:
                print("\n[EOS reached]")
                break

            input_ids = torch.cat([input_ids, next_tok], dim=1)

    print("\n\n✅ Generation complete.")

    # ── Remove hooks ──────────────────────────────────────────────────────────
    for h in hooks:
        h.remove()

    # ── Average activations across generation steps ───────────────────────────
    layers   = list(range(config.layers))
    avg_attn = [sum(attn_activations[i]) / max(len(attn_activations[i]), 1) for i in layers]
    avg_mlp  = [sum(mlp_activations[i])  / max(len(mlp_activations[i]),  1) for i in layers]

    # ── Plot ──────────────────────────────────────────────────────────────────
    plt.figure(figsize=(12, 6))
    plt.plot(layers, avg_attn,
             label=f"Avg Attention (over {num_tokens} tokens)",
             color="#3498db", marker="o", linewidth=2.5)
    plt.plot(layers, avg_mlp,
             label=f"Avg MLP (over {num_tokens} tokens)",
             color="#e67e22", marker="s", linewidth=2.5)

    plt.title(
        f"Layer-wise Activation Magnitudes (Averaged over {num_tokens} steps)\n"
        f"temp={temperature}, top_k={top_k}, top_p={top_p}, rep_penalty={repetition_penalty}",
        fontsize=14, fontweight="bold", pad=15,
    )
    plt.xlabel("Layer Index", fontsize=13)
    plt.ylabel("Mean L2 Norm", fontsize=13)
    plt.xticks(layers)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()

    out_path = "activation_plot_avg.png"
    plt.savefig(out_path, dpi=300)
    print(f"📊 Activation plot saved as '{out_path}'")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Quick multi-prompt test (no activation plot, just generation)
# ─────────────────────────────────────────────────────────────────────────────
def quick_test(ckpt_path: str, prompts: list[str], num_tokens: int = 100):
    """Run several prompts quickly without the activation overhead."""
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("data_mixed/custom_multilingual_tokenizer")
    tokenizer.pad_token = tokenizer.eos_token

    config = Config()
    model  = LLM(config).to(device).eval()

    state_dict       = torch.load(ckpt_path, map_location=device)
    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict, strict=True)
    print(f"✅ Model loaded\n")

    with torch.inference_mode():
        for prompt in prompts:
            input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
            print(f"Prompt : {prompt}")
            print("Output : ", end="", flush=True)

            for _ in range(num_tokens):
                context  = input_ids[:, -config.block_size:]
                out      = model(input_ids=context)
                next_tok = sample_next_token(
                    out.logits[:, -1, :],
                    context_ids=input_ids,
                    temperature=1.0,
                    top_k=30,
                    top_p=0.8,
                    repetition_penalty=1.15
                )
                decoded = tokenizer.decode(next_tok[0], skip_special_tokens=True)
                print(decoded, end="", flush=True)
                if next_tok.item() == tokenizer.eos_token_id:
                    break
                input_ids = torch.cat([input_ids, next_tok], dim=1)

            print("\n" + "─" * 60)

def analyze_and_plot_weights(ckpt_path: str):
    """
    Extracts and plots weight statistics (max, min, median, std) for each layer block.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nLoading Model for Weight Analysis...")
    
    config = Config()
    model = LLM(config).to(device)

    if ckpt_path:
        try:
            state_dict = torch.load(ckpt_path, map_location=device)
            clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(clean_state_dict, strict=True)
            print(f"✅ Model loaded from {ckpt_path}")
        except FileNotFoundError:
            print("⚠️ Checkpoint not found — running with random weights.")

    layers = list(range(config.layers))
    
    # Storage for the statistics
    stats = {
        "max": [],
        "min": [],
        "median": [],
        "mean": [],
        "std": []
    }

    for i in layers:
        layer_weights = []
        # Iterate through all parameters and grab the ones belonging to block 'i'
        for name, param in model.named_parameters():
            if f"blocks.{i}." in name and "weight" in name:
                # Flatten the weights and move to CPU
                layer_weights.append(param.detach().cpu().view(-1))
        
        if layer_weights:
            # Concatenate all weights for this layer into one massive 1D tensor
            concat_weights = torch.cat(layer_weights)
            
            stats["max"].append(concat_weights.max().item())
            stats["min"].append(concat_weights.min().item())
            stats["median"].append(concat_weights.median().item())
            stats["mean"].append(concat_weights.mean().item())
            stats["std"].append(concat_weights.std().item())
        else:
            print(f"⚠️ No weights found for layer {i}")
            for k in stats: stats[k].append(0)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plt.figure(figsize=(12, 6))
    
    # Plot Min & Max
    plt.plot(layers, stats["max"], label="Max", color="#e74c3c", linestyle="--", marker="^")
    plt.plot(layers, stats["min"], label="Min", color="#2980b9", linestyle="--", marker="v")
    
    # Plot Median
    plt.plot(layers, stats["median"], label="Median", color="#27ae60", linewidth=3, marker="o")
    
    # Plot Standard Deviation as a shaded region around the Mean
    mean_arr = torch.tensor(stats["mean"])
    std_arr = torch.tensor(stats["std"])
    plt.fill_between(
        layers, 
        (mean_arr - std_arr).tolist(), 
        (mean_arr + std_arr).tolist(),
        color="#f1c40f", alpha=0.3, label="Mean ± 1 Std Dev"
    )

    plt.title("Weight Distributions per Layer (Attn + MLP combined)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Layer Index", fontsize=13)
    plt.ylabel("Weight Value", fontsize=13)
    plt.xticks(layers)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend(fontsize=12, loc="upper right")
    plt.tight_layout()

    out_path = "weight_distribution.png"
    plt.savefig(out_path, dpi=300)
    print(f"📊 Weight distribution plot saved as '{out_path}'")
    plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    CKPT = "model_save/checkpoint-104657/pytorch_model.bin"

    # ── 1. Weight Distribution Analysis ───────────────────────────────────────
    analyze_and_plot_weights(ckpt_path=CKPT)

    # ── 2. Activation plot (single prompt) ───────────────────────────────────
    generate_and_plot_activations(
        prompt_text="The quick brown fox jumps over the lazy",
        ckpt_path=CKPT,
        num_tokens=50,
        temperature=1.0,
        top_k=30,
        top_p=0.8,
        repetition_penalty=1.0,
    )

    # ── 3. Quick multi-prompt sanity check ────────────────────────────────────
    quick_test(
        ckpt_path=CKPT,
        prompts=[
            "The quick brown fox jumps over the lazy",
            "In mathematics, a prime number is",
            "Столица Франции — Париж, и",
            "Фотосинтез — это процесс, посредством которого",
            "Amir Temur buyuk sarkarda va",
            "Bugun juda yahshi obhavo",
        ],
        num_tokens=100,
    )