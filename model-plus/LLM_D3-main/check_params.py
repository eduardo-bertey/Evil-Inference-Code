import torch
import torch.nn as nn
from LLM_2 import GPT, GPTConfig, MLPExperts, Router

def estimate_model_stats(model):
    config = model.config
    total_params = sum(p.numel() for p in model.parameters())
    
    # We calculate active parameters by identifying MoE components
    active_params = 0
    moe_layers_count = 0
    
    for name, module in model.named_modules():
        # Handle leaf modules (parameters belong to these)
        if len(list(module.children())) == 0:
            # Check if this is part of an expert layer
            is_expert = any(isinstance(m, MLPExperts) for m in model.modules() if name.startswith(tuple(n for n, _ in model.named_modules() if m == _)))
            
            # This is a bit complex, let's simplify logic:
            # Active = Total - (Inactive Experts)
            pass

    # Simplified Logic: Iterate through blocks and check for MoE
    base_params = 0 # Non-MoE parameters
    moe_router_params = 0
    expert_params_total = 0
    
    # 1. Embeddings and Head
    base_params += model.transformer.wte.weight.numel()
    base_params += model.transformer.ln_f.weight.numel()
    # Note: lm_head and wte are tied, so we don't double count if weight is shared
    
    # 2. Iterate Blocks
    for block in model.transformer.h:
        # MLA is always active
        base_params += sum(p.numel() for p in block.ln_1.parameters())
        base_params += sum(p.numel() for p in block.attn.parameters())
        base_params += sum(p.numel() for p in block.ln_2.parameters())
        
        if hasattr(block.mlp, 'router'): # It's a MOELayer
            moe_layers_count += 1
            # Router is always active
            moe_router_params += sum(p.numel() for p in block.mlp.router.parameters())
            # Expert parameters
            expert_total = sum(p.numel() for p in block.mlp.experts.parameters())
            expert_params_total += expert_total
        else:
            # Standard MLP is always active
            base_params += sum(p.numel() for p in block.mlp.parameters())

    # Calculate Expert Sparsity
    # For one MoE layer: active_expert_params = (top_k / n_exp) * total_expert_params
    active_expert_params = (config.top_k / config.n_exp) * expert_params_total
    
    total_active = base_params + moe_router_params + active_expert_params
    
    print(f"--- Model Statistics ---")
    print(f"Total Parameters:    {total_params / 1e6:.2f}M")
    print(f"Active Parameters:   {total_active / 1e6:.2f}M")
    print(f"Sparsity Ratio:      {1 - (total_active / total_params):.2%}")
    print(f"MoE Layers:          {moe_layers_count} (out of {config.n_layer})")
    print(f"Experts per Layer:   {config.n_exp} (Top-{config.top_k} active)")

# Usage
config = GPTConfig() # Uses your default 16 experts / 3 top_k
model = GPT(config)
estimate_model_stats(model)