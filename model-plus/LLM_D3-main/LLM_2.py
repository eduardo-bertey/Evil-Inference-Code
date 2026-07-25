import math
import inspect
from dataclasses import dataclass
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple
import inspect

from transformers.modeling_outputs import CausalLMOutput
from manager import MANAGER

def precompute_freqs_cis(config):
    # We now return cos and sin directly instead of a complex polar tensor
    freqs = 1.0 / (config.theta ** (torch.arange(0, config.d_rotate, 2)[: (config.d_rotate // 2)].float() / config.d_rotate))
    t = torch.arange(config.block_size, device=freqs.device)
    freqs = torch.outer(t, freqs).float() # [seq_len, d_rotate/2]
    
    # Cos and Sin are what Inductor can easily optimize
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    
    # Repeat along the last dimension to match the d_rotate size
    # [seq_len, d_rotate/2] -> [seq_len, d_rotate]
    cos = torch.repeat_interleave(cos, 2, dim=-1)
    sin = torch.repeat_interleave(sin, 2, dim=-1)
    return cos, sin

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    # x: [..., d_rotate]
    # Split into [x1, x2, x3, x4...] -> x1, x2 are pairs
    # We use the interleaving pattern: [-x2, x1, -x4, x3...]
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

def apply_rotary_emb(xq, xk, freqs_cos, freqs_sin):
    # Reshape freqs for broadcasting: [seq_len, d_rotate] -> [1, seq_len, 1, d_rotate]
    # This matches (batch, seq, head, dim)
    cos = freqs_cos[:xq.shape[1]].view(1, xq.shape[1], 1, xq.shape[-1])
    sin = freqs_sin[:xq.shape[1]].view(1, xq.shape[1], 1, xq.shape[-1])

    # The RoPE formula: x_out = x * cos + rotate_half(x) * sin
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)

    return xq_out.type_as(xq), xk_out.type_as(xk)

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.n_embd
        self.num_head = config.n_head
        self.d_head = self.d_model // self.num_head
        
        self.d_c = config.d_c
        self.d_c1 = config.d_c1
        self.d_rotate = config.d_rotate
        
        # ==========================================
        # FUSION 1: All Projections from 'x'
        # Replaces DQ_proj, DKV_proj, and RK_proj
        # ==========================================
        self.W_down = nn.Linear(
            self.d_model, 
            self.d_c1 + self.d_c + self.d_rotate, 
            bias=config.bias
        )
        
        # ==========================================
        # FUSION 2: All Q Up-Projections from 'C_Q'
        # Replaces UQ_proj and RQ_proj
        # ==========================================
        self.W_up_q = nn.Linear(
            self.d_c1, 
            self.d_model + (self.num_head * self.d_rotate), 
            bias=config.bias
        )
        
        # ==========================================
        # FUSION 3: All KV Up-Projections from 'C_KV'
        # Replaces UK_proj and UV_proj (STILL STRICTLY SEPARATE WEIGHTS)
        # ==========================================
        self.W_up_kv = nn.Linear(
            self.d_c, 
            self.d_model + self.d_model, # d_model for K, d_model for V
            bias=config.bias
        )

        # Output projection and Regularization
        self.output_proj = nn.Linear(self.d_model, self.d_model, bias=config.bias)
        self.output_proj.output_proj_marker = True
        self.dropout = nn.Dropout(config.dropout)
        self.attn_dropout_p = config.dropout

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        cos, sin = precompute_freqs_cis(config)
        self.register_buffer("freqs_cos", cos, persistent=False)
        self.register_buffer("freqs_sin", sin, persistent=False)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # ---------------------------------------------------------
        # 1. KERNEL 1: Down-project everything at once
        # ---------------------------------------------------------
        down_out = self.W_down(x)
        # Split into the 3 exact latents your math requires
        C_Q, C_KV, K_rotate = down_out.split(
            [self.d_c1, self.d_c, self.d_rotate], dim=-1
        )

        # ---------------------------------------------------------
        # 2. KERNEL 2: Up-project Query content and RoPE
        # ---------------------------------------------------------
        q_up_out = self.W_up_q(C_Q)
        Q_state, Q_rotate = q_up_out.split(
            [self.d_model, self.num_head * self.d_rotate], dim=-1
        )
        Q_state = Q_state.view(batch_size, seq_len, self.num_head, self.d_head)
        Q_rotate = Q_rotate.view(batch_size, seq_len, self.num_head, self.d_rotate)

        # ---------------------------------------------------------
        # 3. KERNEL 3: Up-project Key and Value content independently
        # ---------------------------------------------------------
        kv_up_out = self.W_up_kv(C_KV)
        K_state, V_state = kv_up_out.split(
            [self.d_model, self.d_model], dim=-1
        )
        K_state = K_state.view(batch_size, seq_len, self.num_head, self.d_head)
        V_state = V_state.view(batch_size, seq_len, self.num_head, self.d_head)
        
        # Prepare shared RoPE Key
        K_rotate = K_rotate.view(batch_size, seq_len, 1, self.d_rotate).expand(-1, -1, self.num_head, -1)

        # ---------------------------------------------------------
        # 4. Apply RoPE, Concatenate, and Attention
        # ---------------------------------------------------------
        Q_rotate, K_rotate = apply_rotary_emb(
            Q_rotate, 
            K_rotate, 
            self.freqs_cos, 
            self.freqs_sin
        )

        Q = torch.cat([Q_state, Q_rotate], dim=-1).transpose(1, 2)
        K = torch.cat([K_state, K_rotate], dim=-1).transpose(1, 2)
        V = V_state.transpose(1, 2)

        if self.flash:
            att_output = F.scaled_dot_product_attention(
                Q, K, V, 
                dropout_p=self.attn_dropout_p if self.training else 0.0, 
                is_causal=True
            )
        else:
            scaler = 1.0 / math.sqrt(self.d_head + self.d_rotate)
            att_matrix = (Q @ K.transpose(-2, -1)) * scaler
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).view(1, 1, seq_len, seq_len)
            att_matrix = att_matrix.masked_fill(mask == 0, float('-inf'))
            att_score = self.dropout(F.softmax(att_matrix, dim=-1))
            att_output = att_score @ V

        att_output = att_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.output_proj(att_output)

class Router(nn.Module):
    def __init__(self, config):
        super().__init__()

        # router settings
        self.top_k = config.top_k
        self.n_exp = config.n_exp
        assert self.top_k >= 1 and self.top_k <= config.n_exp
        self.use_noisy_top_k = config.use_noisy_top_k
        self.train_capacity = config.train_capacity
        self.eval_capacity = config.eval_capacity
        self.min_capacity = config.min_capacity
        self.router_use_full_prec = config.router_use_full_prec

        # auxiliary / load balancing loss settings
        self.use_aux_loss = config.use_aux_loss
        self.use_router_z_loss = config.use_router_z_loss

        # linear projection for (noisy) softmax gating
        # no bias is used, see page 4 eq (4) in (https://arxiv.org/abs/1701.06538)
        self.w_g = nn.Linear(config.n_embd, config.n_exp, bias=False)
        self.w_g.router_marker = True
        self.w_noise = nn.Linear(config.n_embd, config.n_exp, bias=False) if self.use_noisy_top_k else None
    
    def forward(self, x):
        # optionally run the router in full precision to avoid instability during training
        # see discussion on pg. 9 here: https://arxiv.org/abs/2101.03961
        # setting enabled to False in autocast automatically puts everything in float32
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu' # for later use in torch.autocast
        ctx = nullcontext() if not self.router_use_full_prec else torch.amp.autocast(device_type=device_type, enabled=False)

        with ctx:
            B, T, _ = x.size()
            num_tokens = B * T

            # eq (4) in (https://arxiv.org/abs/1701.06538)
            logits = self.w_g(x)  # [B, T, n_exp]
            if self.use_noisy_top_k:
                # optionally add noise into the router
                noise = F.softplus(self.w_noise(x))
                noise *= torch.randn_like(noise)
                logits += noise

            # router z loss, computed on logits (before softmax)
            # this loss prevents router logits from becoming too large
            if self.use_router_z_loss:
                z_loss = self.compute_router_z_loss(logits)
                MANAGER.add_router_z_loss(z_loss)

            # find top k experts for each token
            top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1) # [B, T, k]

            # normalize expert probabilities
            # Question: should we normalize over all experts or just top-k?
            # we choose to normalize over top-k, other option is commented out below

            # Shazeer et al (https://arxiv.org/abs/1701.06538) does only topk
            # see page 4 eq (3)-(5), the code for this is commented out below
            router_probs = torch.full_like(logits, float('-inf'))  # [B, T, n_exp]
            router_probs.scatter_(-1, top_k_indices, top_k_logits)
            router_probs = F.softmax(router_probs, dim=-1)

            # # normalize all router logits (not just top-k) via softmax      
            # router_probs = F.softmax(logits, dim=-1)

            # compute auxiliary load balancing loss
            # this loss encourages equal probability assigned to each expert
            # and equal load balancing of tokens assigned to each expert
            if self.use_aux_loss:
                aux_loss = self.compute_aux_loss(router_probs, top_k_indices)
                MANAGER.add_aux_loss(aux_loss)

            # compute expert capacity
            exp_capacity = self.get_capacity(num_tokens)

            # make a multi-hot mask of chosen experts, size [B, T, n_exp]
            # entries are 0 if expert not chosen and 1 if expert chosen
            exp_mask = F.one_hot(top_k_indices, num_classes=self.n_exp)  # [B, T, k, n_exp]
            exp_mask = exp_mask.view(num_tokens, self.top_k, self.n_exp)  # [B * T, k, n_exp]
            exp_mask = exp_mask.permute(1, 0, 2) # [k, B * T, n_exp]

            # compute cumulative sum of each token over experts, this stores
            # the index of each token within the batch of each expert
            # NOTE: cumsum should count all top-1 first, top-2 second, etc.
            # so that we prioritize top experts when dropping tokens (this is
            # done by putting k dimension first for the reshape operation)
            exp_rank = exp_mask.reshape(self.top_k * num_tokens, self.n_exp)  # [k * B * T, n_exp]
            exp_rank = torch.cumsum(exp_rank, dim=0) - 1  # cumulative sum of expert selections [k * B * T, n_exp]
            exp_rank = exp_rank.reshape(self.top_k, num_tokens, self.n_exp)  # [k, B * T, n_exp]

            # mask out (set to zero) entries that go beyond expert capacity
            # compute amount of used capacity by taking a sum over mask
            exp_mask *= torch.lt(exp_rank, exp_capacity) # [k, B * T, n_exp]
            used_capacity = torch.sum(exp_mask, dim=(0, 1)) # [n_exp]

            # mask rank to only include tokens that are selected
            # perform a sum so each row only contains index of token
            # for the expert that is selected in that row
            # result is a matrix that contains the position of each token
            # in the batch of its corresponding expert
            exp_rank = torch.sum(exp_mask * exp_rank, dim=-1)  # [k, B * T]

            # mask probabilities to only include selected experts
            router_probs = router_probs.view(num_tokens, self.n_exp)[None, :] # [1, B * T, n_exp]
            exp_weights = exp_mask * router_probs # [k, B * T, n_exp]

            # convert rank into one-hot vectors over the available capacity
            # stores the position of each token within the capacity of the selected expert
            exp_rank_sc = F.one_hot(exp_rank, num_classes=exp_capacity) # [k, B * T, exp_capacity]

            # create a vector that stores, for each token, the weight of selected
            # experts at token's position in the capacity of that expert
            # size of tensor is [B * T, n_exp, exp_capacity]
            cb_weight = torch.sum(exp_weights.unsqueeze(3) * exp_rank_sc.unsqueeze(2), dim=0)
            sec_mask = cb_weight.bool() # binary mask of selected experts for each token
            return used_capacity, cb_weight, sec_mask
    
    def compute_aux_loss(self, expert_probs: torch.Tensor, indices: torch.Tensor):
        """
        Computes Switch Transformer auxiliary loss (https://arxiv.org/abs/2101.03961)
        See equations (4)-(6) on page 7
        """

        # equation (5): compute ratio of tokens allocated to each expert
        # total number of tokens is defined as total tokens in batch * k
        # (k = 1) for the Switch Transformer
        with torch.no_grad():
            one_hot_indices = F.one_hot(indices, num_classes=self.n_exp)  # [B, T, k, n_exp]
            one_hot_indices = torch.sum(one_hot_indices.float(), dim=2)  # [B, T, n_exp] (sum over k dimension)
            tokens_per_expert = torch.mean(one_hot_indices.float(), dim=(0, 1))

        # equation (6): compute ratio of router probability allocated to each expert
        prob_per_expert = torch.mean(expert_probs.float(), dim=(0, 1))

        # equation (4): take a scaled dot product between prob/token allocation vectors
        # multiply the result by the number of experts
        return self.n_exp * torch.sum(prob_per_expert * tokens_per_expert)
    
    def compute_router_z_loss(self, logits: torch.Tensor):
        """
        Computes ST-MoE router z loss (https://arxiv.org/abs/2202.08906)
        See equation (5) on page 7
        """
    
        # exponentiate logits, sum logits of each expert, take log, and square
        # code below is the same as:
        # > z_loss = torch.exp(logits)
        # > z_loss = torch.sum(z_loss, dim=-1)
        # > z_loss = torch.log(z_loss) ** 2.0
        z_loss = torch.logsumexp(logits, dim=-1) ** 2.0  # [B, T, n_exp]

        # sum over all tokens and divide by total number of tokens
        return torch.mean(z_loss)

    def get_capacity(self, tokens_per_batch):
        # expert capacity is given by (tokens_per_batch / num_experts) * capacity_factor
        # see eq (3) in Switch Transformer (https://arxiv.org/abs/2101.03961)
        capacity_factor = self.train_capacity if self.training else self.eval_capacity
        capacity = math.floor(self.top_k * capacity_factor * tokens_per_batch / self.n_exp)
        capacity += capacity % 2 
        capacity = max(capacity, self.min_capacity)
        assert capacity > 0
        return int(capacity)

# FEEDFORWARD
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.fc1 = nn.Linear(config.n_embd, 2 * config.ffn_dim, bias=config.bias)
        self.swish = nn.SiLU() 
        self.fc2 = nn.Linear(config.ffn_dim, config.n_embd, bias=config.bias)
        self.fc2.output_proj_marker = True
        
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        # nn.init.xavier_uniform_(self.fc1.weight, gain=math.sqrt(2.0))
        # nn.init.xavier_uniform_(self.fc2.weight, gain=1.0)

    def forward(self, x):
        x = self.fc1(x)

        # Inline SwiGLU: Split the doubled dimension and apply gate
        x, gate = x.chunk(2, dim=-1)
        x = x * self.swish(gate)
        
        x = self.dropout1(x)
        x = self.fc2(x)
        return self.dropout2(x)
    

class MLPExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_exp = config.n_exp
        self.n_embd = config.n_embd
        self.bias = config.bias

        self.c_fc = nn.Parameter(torch.empty(self.n_exp, self.n_embd, 2 * config.expert_dim))
        self.c_proj = nn.Parameter(torch.empty(self.n_exp, config.expert_dim, self.n_embd))
        
        self.swish = nn.SiLU()
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        x = torch.bmm(x, self.c_fc)

        x, gate = x.chunk(2, dim=-1)
        x = x * self.swish(gate)

        x = torch.bmm(x, self.c_proj)
        
        return self.dropout(x)

class MOELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = Router(config) # (noisy) top k router
        self.experts = MLPExperts(config) # group of MLPs (experts)

    def forward(self, x: torch.Tensor):
        B, T, n_embd = x.size() 
        num_tokens = (B * T)

        used_capacity, exp_weight, exp_mask = self.router(x)

        x = x.view(num_tokens, n_embd)

        # [n_exp, exp_capacity, B * T] * [B * T, n_embd] -> [n_exp, exp_capacity, n_embd]
        exp_batches = exp_mask.permute(1, 2, 0).type_as(x) @ x

        exp_out = self.experts(exp_batches) # [n_exp, exp_capacity, n_embd]

        # aggregate expert outputs based on router weights
        # eq (2) on page 4 of ST-MoE (https://arxiv.org/abs/2202.08906)
        # similar equations are used for other MoE papers
        exp_weight = exp_weight.view(num_tokens, -1) # [B * T, n_exp * exp_capacity]
        exp_out = exp_out.view(-1, n_embd) # [n_exp * exp_capacity, n_embd] 
        output = exp_weight @ exp_out # [B * T, n_embd]

        return output.view(B, T, n_embd)

class Block(nn.Module):

    def __init__(self, config, use_moe=False):
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.n_embd)
        self.attn = MultiHeadLatentAttention(config)
        self.ln_2 = nn.RMSNorm(config.n_embd)
        if use_moe:
            self.mlp = MOELayer(config)
        else:
            self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 2048
    vocab_size: int = 50304 
    n_layer: int = 24         # Keep it deep for logic
    n_head: int = 5          # 64-dim heads (640 / 10) are extremely fast
    n_embd: int = 640         # The "Sweet Spot" for 350M
    dropout: float = 0.0      # Set to 0.0 for pre-training
    ffn_dim: int = 640*4       # 4x Embedding
    bias: bool = False 

    # MLA - High Efficiency
    d_c: int = 128           
    d_c1: int = 128
    d_rotate: int = 64    
    theta: float = 10000.0

    # MoE - Maximally Smart
    n_exp: int = 6            # 6 experts is ideal for a 350M model
    top_k: int = 2
    expert_dim: int = 640*2    # 2x Embedding (Powerful Experts)
    stride: int = 3           # Dense base (Layers 0-3 and 24-27)
    
    # Stability (Standard Production Settings)
    use_aux_loss: bool = True
    use_router_z_loss: bool = True
    use_noisy_top_k: bool = True
    aux_loss_weight: float = 0.01
    router_z_loss_weight: float = 0.001
    train_capacity: float = 1.25
    eval_capacity: float = 2.0
    min_capacity: int = 8
    use_switch_tfm_init: bool = True
    switch_tfm_init_scale: float = 1.0
    router_use_full_prec: bool = True

    # Training Hyperparameters
    batch_size: int = 8      # Increased from 16 since 350M is lighter
    grad_acc: int = 64        # Global batch = 1024
    num_train_epochs: int = 1 # 5 Epochs is best for 20B tokens / 350M model
    learning_rate: float = 3e-4 # Slightly higher LR for smaller models
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    warm_up: int = 1000       # Shorter warmup for 350M
    
    eos_token_id = 0
    bos_token_id = 0
    pad_token_id = 0


class HybridOptimizer(torch.optim.Optimizer):
        def __init__(self, optimizers):
            self.optimizers = optimizers
            self.param_groups = []
            for opt in self.optimizers:
                self.param_groups.extend(opt.param_groups)
    
        def step(self, closure=None):
            loss = None
            if closure is not None:
                loss = closure()
            for opt in self.optimizers:
                opt.step()
            return loss
    
        def zero_grad(self, set_to_none=True):
            for opt in self.optimizers:
                opt.zero_grad(set_to_none=set_to_none)
    
        def state_dict(self):
            return [opt.state_dict() for opt in self.optimizers]
    
        def load_state_dict(self, state_dict):
            for opt, sd in zip(self.optimizers, state_dict):
                opt.load_state_dict(sd)

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.can_return_loss = True
        self.accepts_loss_kwargs = False

        if config.n_exp == 1:
            blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        else:
            blocks = []
            for i in range(config.n_layer):
                use_moe = False if (i < config.stride or i > config.n_layer - config.stride)  else True
                blocks.append(Block(config, use_moe=use_moe))
            blocks = nn.ModuleList(blocks)

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = blocks,
            ln_f = nn.RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight 
        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('experts.c_proj'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    @torch.no_grad()
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if self.config.use_switch_tfm_init:
                scale = self.config.switch_tfm_init_scale
                w_fan_in = module.weight.shape[-1]
                base_std = (scale / w_fan_in) ** 0.5
                
                # 1. Output Projections (End of residual branch)
                if hasattr(module, 'output_proj_marker'):
                    # Depth scaling to prevent residual explosion
                    res_std = base_std / math.sqrt(2 * self.config.n_layer)
                    
                    if hasattr(module, 'is_attention'):
                        # MLA Attention Recovery (Compensate for Softmax and compression)
                        attention_gain = 2.0 
                        final_std = res_std * attention_gain
                    else:
                        # Standard Dense MLP Output
                        final_std = res_std
                
                # 2. Router Gating
                elif hasattr(module, 'router_marker'):
                    # Routers must start uniform and small
                    final_std = 0.01
                    
                # 3. Input Projections
                else:
                    if hasattr(module, 'is_swiglu'):
                        # SwiGLU cuts variance in half, boost by sqrt(2)
                        final_std = base_std * math.sqrt(2.0)
                    else:
                        final_std = base_std

                torch.nn.init.trunc_normal_(
                    module.weight, mean=0.0, std=final_std, a=-2*final_std, b=2*final_std
                )
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, MLPExperts):
            if self.config.use_switch_tfm_init:
                scale = self.config.switch_tfm_init_scale

                # UP-PROJECTION (c_fc): Apply SwiGLU boost
                c_fc_fan_in = module.c_fc.shape[-2]
                base_fc_std = (scale / c_fc_fan_in) ** 0.5
                final_fc_std = base_fc_std * math.sqrt(2.0)
                torch.nn.init.trunc_normal_(
                    module.c_fc, mean=0.0, std=final_fc_std, a=-2*final_fc_std, b=2*final_fc_std
                )

                # DOWN-PROJECTION (c_proj): Apply Depth Scaling + MoE Sparsity Boost
                c_proj_fan_in = module.c_proj.shape[-2]
                base_proj_std = (scale / c_proj_fan_in) ** 0.5
                res_proj_std = base_proj_std / math.sqrt(2 * self.config.n_layer)
                
                moe_scale = math.sqrt(self.config.n_exp / self.config.top_k)
                final_proj_std = res_proj_std * moe_scale
                
                torch.nn.init.trunc_normal_(
                    module.c_proj, mean=0.0, std=final_proj_std, a=-2*final_proj_std, b=2*final_proj_std
                )
            else:
                torch.nn.init.normal_(module.c_fc, mean=0.0, std=0.02)
                torch.nn.init.normal_(module.c_proj, mean=0.0, std=0.02)

            # NOTE: Your updated MLPExperts doesn't show bias, but if you have it:
            if hasattr(module, 'fc_bias') and module.fc_bias is not None:
                torch.nn.init.zeros_(module.fc_bias)
            if hasattr(module, 'proj_bias') and module.proj_bias is not None:
                torch.nn.init.zeros_(module.proj_bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    

    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        _, t = input_ids.size()
        assert t <= self.config.block_size, f"Sequence length {t} exceeds block size {self.config.block_size}"

        x = self.transformer.wte(input_ids) 
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if labels is not None:
            logits = self.lm_head(x)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(
                ignore_index=-100,
                label_smoothing=0.1, 
                reduction='mean'
            )

            main_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )

            loss = main_loss
            
            if self.config.n_exp > 1:
                if self.config.use_aux_loss:
                    loss += self.config.aux_loss_weight * MANAGER.aggregate_aux_loss()
                    MANAGER.reset_aux_loss()
                
                if self.config.use_router_z_loss:
                    loss += self.config.router_z_loss_weight * MANAGER.aggregate_router_z_loss()
                    MANAGER.reset_router_z_loss()
        else:
            logits = self.lm_head(x[:, [-1], :]) 
            loss = None

        return CausalLMOutput(loss=loss, logits=logits)
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # TODO: add expert config
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        # add an extra check for "bias" string to account for bias terms in MoE layers
        decay_params = [p for n, p in param_dict.items() if (p.dim() >= 2 and not n.endswith('bias'))]
        nodecay_params = [p for n, p in param_dict.items() if (p.dim() < 2 or n.endswith('bias'))]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # Correctly unpack the dataclass output
            outputs = self(idx_cond)
            logits = outputs.logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx