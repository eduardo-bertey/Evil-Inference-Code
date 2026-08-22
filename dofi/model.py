"""Dofi — Transformer autoregressivo con DiffusionBlocks.

Basado en https://github.com/metric-space/AutoregressiveDiffusionBlocks
- GPT2-style con positional embeddings (no RoPE, para concat clean+noised)
- Q, K, V separados
- XSA
- AdaLN para condicionamiento σ
- Custom attn mask: noisy tokens no ven otros noisy tokens
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dblock_modules import get_discrete_sigmas


class Config:
    dim = 512
    heads = 8
    kv_groups = 4
    layers = 16
    num_blocks = 4
    layers_per_block = 4
    ffn_dim = 2048
    block_size = 2048
    emb_num = 32000
    drop = 0.0

    # DiffusionBlocks
    sigma_min = 5.0
    sigma_max = 10.0
    p_mean = -1.2
    p_std = 1.2
    gamma = 0.1
    sigma_data = 0.5

    # Condicionamiento
    cond_hidden_size = 128

    noise_on_labels = True
    sequential_blocks = True

    # Training
    batch_size = 8
    grad_acc = 1
    learning_rate = 3e-4
    weight_decay = 0.1
    betas = (0.9, 0.95)
    warm_up = 20


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half) / half
        ).to(t.device)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
            dtype=next(self.parameters()).dtype
        )
        return self.mlp(t_freq)


class AdaLN(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.net = nn.Sequential(nn.SiLU(), nn.Linear(in_features, out_features, bias))

    def forward(self, x):
        return self.net(x)


def repeat_kv(x, num_heads, num_kv_groups):
    if num_kv_groups == num_heads:
        return x
    return x.repeat_interleave(num_heads // num_kv_groups, dim=2)


class Attention(nn.Module):
    """QKV attention + XSA con mask custom."""

    def __init__(self, config):
        super().__init__()
        nx = config.dim
        self.n_head = config.heads
        self.n_kv_groups = config.kv_groups
        self.head_dim = nx // self.n_head

        self.q_proj = nn.Linear(nx, nx, bias=False)
        self.k_proj = nn.Linear(nx, config.kv_groups * self.head_dim, bias=False)
        self.v_proj = nn.Linear(nx, config.kv_groups * self.head_dim, bias=False)
        self.o_proj = nn.Linear(nx, nx, bias=False)

    @staticmethod
    def _xsa(y, v):
        B, H, S, D = y.shape
        vn = F.normalize(v, p=2, dim=-1)
        vn = vn[:, :, -S:, :]
        return y - (y * vn).sum(dim=-1, keepdim=True) * vn

    @staticmethod
    def _derive_mask(original_mask, noise_mask):
        """Mask custom: clean causal, noisy solo ve clean hasta su posición."""
        mask_dim = original_mask.shape[-1]
        mask = original_mask.new_zeros(mask_dim, mask_dim)

        i = int(original_mask.sum())
        ni = int(noise_mask.sum())

        if ni == 0:
            return torch.tril(mask.new_ones(mask_dim, mask_dim))

        mask[i:i+ni, :ni] = torch.tril(mask.new_ones(ni, ni))

        for z in range(ni):
            noisy_pos = i + z
            mask[noisy_pos, 1 + z] = False
            mask[noisy_pos, noisy_pos] = True

        return mask

    def forward(self, x, original_mask, noise_mask):
        B, T, D = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_groups, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_groups, self.head_dim)
        k = repeat_kv(k, self.n_head, self.n_kv_groups).transpose(1, 2)
        v = repeat_kv(v, self.n_head, self.n_kv_groups).transpose(1, 2)

        masks = []
        for i in range(B):
            masks.append(self._derive_mask(original_mask[i], noise_mask[i]))
        masks = torch.stack(masks, dim=0)
        masks = masks[:, None, :, :].to(dtype=q.dtype)

        w = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        w = w * masks - 1e20 * (1 - masks)
        w = F.softmax(w, dim=-1)

        y = torch.matmul(w, v)
        y = self._xsa(y, v)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)

        return self.o_proj(y)

    def forward_inference(self, x):
        """Inference: causal mask normal."""
        B, T, D = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_groups, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_groups, self.head_dim)
        k = repeat_kv(k, self.n_head, self.n_kv_groups).transpose(1, 2)
        v = repeat_kv(v, self.n_head, self.n_kv_groups).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = self._xsa(y, v)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)

        return self.o_proj(y)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        nx = config.dim
        self.adaLN_modulation = AdaLN(config.cond_hidden_size, 6 * nx)
        self.ln_1 = nn.LayerNorm(nx)
        self.attn = Attention(config)
        self.ln_2 = nn.LayerNorm(nx)
        self.ffn = nn.Sequential(
            nn.Linear(nx, config.ffn_dim),
            nn.SiLU(),
            nn.Linear(config.ffn_dim, nx),
        )

    def forward(self, x, conditioning, original_mask, noise_mask):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(conditioning).chunk(6, dim=-1)

        h = self.ln_1(x)
        h = modulate(h, shift_msa, scale_msa)
        attn_out = self.attn(h, original_mask, noise_mask)
        x = x + gate_msa.unsqueeze(1) * attn_out

        h = self.ln_2(x)
        h = modulate(h, shift_mlp, scale_mlp)
        ffn_out = self.ffn(h)
        x = x + gate_mlp.unsqueeze(1) * ffn_out

        return x

    def forward_inference(self, x, conditioning):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(conditioning).chunk(6, dim=-1)

        h = self.ln_1(x)
        h = modulate(h, shift_msa, scale_msa)
        attn_out = self.attn.forward_inference(h)
        x = x + gate_msa.unsqueeze(1) * attn_out

        h = self.ln_2(x)
        h = modulate(h, shift_mlp, scale_mlp)
        ffn_out = self.ffn(h)
        x = x + gate_mlp.unsqueeze(1) * ffn_out

        return x


class DofiLLM(nn.Module):
    """Transformer AR con DiffusionBlocks."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.emb_num, config.dim)
        self.wpe = nn.Embedding(config.block_size, config.dim)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])

        self.norm_f = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.emb_num, bias=False)
        self.lm_head.weight = self.wte.weight

        self.time_embedder = TimestepEmbedder(config.cond_hidden_size)
        self.adaLN_final = AdaLN(config.cond_hidden_size, 2 * config.dim)

        self._init_dit()

        print(f"Dofi: {sum(p.numel() for p in self.parameters())/1e6:.2f}M params, "
              f"{config.layers} capas, {config.num_blocks} bloques de {config.layers_per_block}")

    def _init_dit(self):
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation.net[-1].weight)
            nn.init.zeros_(block.adaLN_modulation.net[-1].bias)

    def get_block_layers(self, block_idx):
        start = block_idx * self.config.layers_per_block
        return list(range(start, start + self.config.layers_per_block))

    def normalize_embeddings(self, x):
        return F.normalize(x, p=2, dim=-1)

    def get_embeddings(self, input_ids):
        return self.wte(input_ids)

    def add_position_embeddings(self, inputs, offset=0):
        seq_dim = 1 if inputs.ndim == 3 else 0
        seq_len = inputs.size(seq_dim)
        position_ids = torch.arange(
            offset, seq_len + offset, dtype=torch.long, device=inputs.device
        )
        position_embeds = self.wpe(position_ids)
        return position_embeds + inputs

    def forward_block(self, block_idx, input_ids, sigma, target_ids=None):
        """Training forward: concat [clean, noised*c_in] con mask custom."""
        sigma_val = sigma if isinstance(sigma, float) else sigma.item()
        sigma_data = self.config.sigma_data

        c_in = 1.0 / (sigma_val**2 + sigma_data**2)**0.5
        c_out = sigma_val * sigma_data / (sigma_val**2 + sigma_data**2)**0.5
        c_skip = sigma_data**2 / (sigma_val**2 + sigma_data**2)
        c_noise = 0.25 * math.log(max(sigma_val, 1e-8))
        sigma_cond = self.time_embedder(
            torch.tensor([c_noise], device=input_ids.device, dtype=torch.float32)
        )

        # Clean embeddings: positions 0..L-1
        original = self.normalize_embeddings(self.get_embeddings(input_ids))
        original = self.add_position_embeddings(original, offset=0)

        if self.config.noise_on_labels and target_ids is not None:
            # Noised embeddings: shifted by 1, positions 1..L
            noised_tokens = target_ids
            noised = self.normalize_embeddings(self.get_embeddings(noised_tokens))
            noise = torch.randn_like(noised) * sigma_val
            noised = noised + noise
            noised = self.add_position_embeddings(noised, offset=1)

            # Concat: [clean, noised * c_in]
            x = torch.cat([original, noised * c_in], dim=1)

            # Masks (ambos del largo total L+N)
            L = original.shape[1]
            N = noised.shape[1]
            total = L + N
            original_mask = torch.zeros(input_ids.shape[0], total, device=input_ids.device)
            original_mask[:, :L] = 1.0
            noise_mask = torch.zeros(input_ids.shape[0], total, device=input_ids.device)
            noise_mask[:, L:] = 1.0

            # Noise tensor for EDM output (zeros for clean, actual noise for noised)
            noise_full = torch.cat([
                torch.zeros_like(original),
                noise,
            ], dim=1)
        else:
            x = self.add_position_embeddings(original, offset=0)
            L = x.shape[1]
            original_mask = torch.ones(input_ids.shape[0], L, device=input_ids.device)
            noise_mask = torch.ones(input_ids.shape[0], L, device=input_ids.device)
            noise_full = torch.randn_like(x) * sigma_val

        # Forward through assigned block layers
        layer_indices = self.get_block_layers(block_idx)
        for i in layer_indices:
            x = self.blocks[i](x, sigma_cond, original_mask, noise_mask)

        # Final norm + AdaLN
        x = self.norm_f(x)
        shift, scale = self.adaLN_final(sigma_cond).chunk(2, dim=-1)
        x = modulate(x, shift, scale)

        # EDM output on ALL positions
        model_out = x * c_out + noise_full * c_skip
        logits = self.lm_head(model_out)

        return logits, noise_mask

    def forward(self, input_ids, sigma=None, block_idx=None):
        if block_idx is not None:
            return self.forward_block(block_idx, input_ids, sigma, target_ids=input_ids)

        sigma_cond = self.time_embedder(sigma.unsqueeze(0) if sigma.dim() == 0 else sigma)
        x = self.normalize_embeddings(self.get_embeddings(input_ids))
        x = self.add_position_embeddings(x, offset=0)

        for block in self.blocks:
            x = block.forward_inference(x, sigma_cond)

        x = self.norm_f(x)
        shift, scale = self.adaLN_final(sigma_cond).chunk(2, dim=-1)
        x = modulate(x, shift, scale)

        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def ode_solve(self, input_ids, num_steps=4):
        """ODE inference: match reference diffusion_step."""
        B = input_ids.shape[0]
        hidden_size = self.config.dim
        sigma_data = self.config.sigma_data

        # Clean embeddings
        x = self.normalize_embeddings(self.get_embeddings(input_ids))
        x = self.add_position_embeddings(x, offset=0)

        L = x.shape[1]

        # Sigmas for diffusion steps
        sigmas = get_discrete_sigmas(num_steps, self.config.sigma_min, self.config.sigma_max, dblock=True)
        sigmas = sigmas.to(input_ids.device)

        # Start from noise (L-1 positions, shifted tokens)
        z = torch.randn(B, L - 1, hidden_size, device=input_ids.device, dtype=x.dtype)
        z *= torch.sqrt(1.0 + sigmas[0] ** 2.0)

        for i in range(len(sigmas) - 1):
            sigma = sigmas[i]
            next_sigma = sigmas[i + 1]
            block_idx = min(i, self.config.num_blocks - 1)

            # Concat [clean, z] with causal mask (inference = full causal)
            seq = torch.cat([x, z], dim=1)
            seq = self.add_position_embeddings(seq, offset=0)

            sigma_cond = self.time_embedder(sigma.unsqueeze(0))

            for j in self.get_block_layers(block_idx):
                seq = self.blocks[j].forward_inference(seq, sigma_cond)

            # EDM output
            seq = self.norm_f(seq)
            shift, scale = self.adaLN_final(sigma_cond).chunk(2, dim=-1)
            seq = modulate(seq, shift, scale)

            c_out = sigma * sigma_data / (sigma**2 + sigma_data**2)**0.5
            c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)

            z_out = seq[:, L:] * c_out + z * c_skip
            logits = self.lm_head(z_out)

            # Euler step
            probs = F.softmax(logits, dim=-1)
            denoised = probs @ self.wte.weight
            d = (z - denoised) / sigma
            dt = next_sigma - sigma
            z = z + dt * d

        # Final pass through ALL blocks
        sigma = sigmas[-1]
        c_out = sigma * sigma_data / (sigma**2 + sigma_data**2)**0.5
        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)

        seq = torch.cat([x, z], dim=1)
        seq = self.add_position_embeddings(seq, offset=0)
        sigma_cond = self.time_embedder(sigma.unsqueeze(0))

        for block in self.blocks:
            seq = block.forward_inference(seq, sigma_cond)

        seq = self.norm_f(seq)
        shift, scale = self.adaLN_final(sigma_cond).chunk(2, dim=-1)
        seq = modulate(seq, shift, scale)

        z_out = seq[:, L:] * c_out + z * c_skip
        logits = self.lm_head(z_out)
        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=0.8, top_k=40,
                 ode_steps=4):
        device = input_ids.device

        for _ in range(max_new_tokens):
            ctx = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids
            logits = self.ode_solve(ctx, num_steps=ode_steps)
            next_logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
