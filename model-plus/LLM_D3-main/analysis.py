
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Import your custom classes
from LLM_2 import GPT, GPTConfig, Block

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_tokenizer():
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading Model...")
    config = GPTConfig() 
    model = GPT(config).to(device)

    ckpt_path = "model_save/checkpoint-48833/pytorch_model.bin"
    try:
        state_dict = torch.load(ckpt_path, map_location=device)
        clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state_dict, strict=True)
        print(f"✅ Model loaded from {ckpt_path}")
    except FileNotFoundError:
        print(f"⚠️ Checkpoint not found at {ckpt_path}. Running with random weights.")
    
    model.eval()
    return model, tokenizer, config

@torch.no_grad()
def generate_streaming(model, tokenizer, prompt, max_new_tokens=100, temperature=1.0, top_k=50, top_p=0.9):
    """Enhanced streaming generation with Top-K and Top-P (Nucleus) sampling."""
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    
    print(f"\n🚀 Generating {max_new_tokens} tokens (Temp: {temperature}, Top-K: {top_k}, Top-P: {top_p})...")
    print(f"Prompt: {prompt}", end="", flush=True)
    
    for _ in range(max_new_tokens):
        out = model(input_ids)
        logits = out.logits[:, -1, :] / max(temperature, 1e-5)
        
        # Apply Top-K
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        # Apply Top-P (Nucleus Sampling)
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to keep the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        
        token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
        print(token_text, end="", flush=True)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
            
    print("\n\n--- Generation Complete ---\n")
    return input_ids

def plot_weight_histograms(model, config):
    """Generates a 6x4 grid of histograms for all 24 layers."""
    print("Generating Weight Histograms (24 layers)...")
    n_layers = config.n_layer
    cols = 4
    rows = (n_layers + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(22, 4 * rows))
    fig.suptitle('Weight Distributions: Attn + MoE + Router', fontsize=24, fontweight='bold', y=1.02)
    axes = axes.flatten()

    for i in range(n_layers):
        ax = axes[i]
        block = model.transformer.h[i]
        
        # Extract all weights (linear layers, experts, routers)
        weights = torch.cat([p.data.view(-1) for n, p in block.named_parameters() if 'weight' in n]).cpu().float().numpy()
        
        ax.hist(weights, bins=80, color='#34495e', alpha=0.7, edgecolor='black', linewidth=0.2)
        ax.set_title(f'Layer {i} (Std: {np.std(weights):.4f})', fontsize=12)
        ax.axvline(0, color='red', linestyle='--', alpha=0.6) # Symmetry check
        ax.grid(axis='y', alpha=0.2)

    plt.tight_layout()
    plt.savefig('weight_histograms.png', dpi=300)
    print("📊 Weight histograms saved as 'weight_histograms.png'")

def run_diagnostics_and_plot(model, config, input_ids):
    """Activation Norms, Weight Stats Bar Chart, and Expert Heatmap."""
    attn_norms, mlp_norms, expert_usage, weight_stats = [], [], {}, []
    hooks = []

    # 1. Collect Weight Stats
    for i, block in enumerate(model.transformer.h):
        p_flat = torch.cat([p.data.view(-1) for n, p in block.named_parameters() if 'weight' in n])
        weight_stats.append({
            'mean': p_flat.mean().item(), 'std': p_flat.std().item(),
            'max': p_flat.max().item(), 'min': p_flat.min().item()
        })

    # 2. Setup Hooks for Activations
    def get_hook(storage):
        return lambda m, inp, out: storage.append(out.norm(p=2, dim=-1).mean().item())

    for i, block in enumerate(model.transformer.h):
        hooks.append(block.attn.register_forward_hook(get_hook(attn_norms)))
        hooks.append(block.mlp.register_forward_hook(get_hook(mlp_norms)))
        if hasattr(block.mlp, 'router'):
            def r_hook(m, inp, out, idx=i): expert_usage[idx] = out[0].detach().cpu().numpy()
            hooks.append(block.mlp.router.register_forward_hook(r_hook))

    # 3. Forward pass
    with torch.no_grad(): _ = model(input_ids)
    for h in hooks: h.remove()

    # --- Plotting ---
    fig = plt.figure(figsize=(18, 16))
    fig.suptitle('LLM_2 Architecture Diagnostics', fontsize=20, fontweight='bold')

    # A: Activations
    ax1 = plt.subplot(3, 1, 1)
    layers = np.arange(config.n_layer)
    ax1.plot(layers, attn_norms, label='Attn Out Norm', marker='o')
    ax1.plot(layers, mlp_norms, label='MLP/MoE Out Norm', marker='s')
    # Highlight MoE range
    ax1.axvspan(config.stride-0.5, config.n_layer-config.stride-0.5, color='gray', alpha=0.1, label='MoE Range')
    ax1.set_title('Activation Norms (Signal Strength)'); ax1.legend(); ax1.grid(alpha=0.3)

    # B: Weight Stats Bar Chart
    ax2 = plt.subplot(3, 1, 2)
    width = 0.2
    ax2.bar(layers - 1.5*width, [s['min'] for s in weight_stats], width, label='Min', color='#e74c3c')
    ax2.bar(layers - 0.5*width, [s['mean'] for s in weight_stats], width, label='Mean', color='#3498db')
    ax2.bar(layers + 0.5*width, [s['std'] for s in weight_stats], width, label='Std', color='#f1c40f')
    ax2.bar(layers + 1.5*width, [s['max'] for s in weight_stats], width, label='Max', color='#2ecc71')
    ax2.set_title('Weight Stats (Outlier Detection)'); ax2.legend(); ax2.grid(axis='y', alpha=0.3)

    # C: MoE Heatmap
    ax3 = plt.subplot(3, 1, 3)
    if expert_usage:
        moe_idxs = sorted(expert_usage.keys())
        matrix = np.array([expert_usage[l] for l in moe_idxs])
        pct = (matrix / matrix.sum(axis=1, keepdims=True)) * 100
        sns.heatmap(pct, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax3,
                    xticklabels=[f"Exp {i}" for i in range(config.n_exp)],
                    yticklabels=[f"Layer {l}" for l in moe_idxs])
        ax3.set_title('Expert Routing Utilization (%)')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig('full_diagnostics.png', dpi=300)
    print("📊 Analysis complete. View 'full_diagnostics.png'")

if __name__ == "__main__":
    model, tokenizer, config = load_model_and_tokenizer()
    
    # Run Generation
    prompt = "The future of artificial intelligence requires us to"
    ids = generate_streaming(model, tokenizer, prompt, temperature=0.8, top_p=0.9)
    
    # Run Visualizations
    run_diagnostics_and_plot(model, config, ids)
    plot_weight_histograms(model, config)