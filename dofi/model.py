"""Dofi — Transformer autoregressivo con DiffusionBlocks.

Adaptado de dlorai/model.py:
  - Q–K=V (BrainChip): k = v, sin v_proj
  - XSA (Apple): Z = Y - (Y*Vn).sum(-1,keepdim)*Vn
  - Sin LISA (deshabilitado)
  - Sin MoE (solo base denso)
  - 16 capas, 4 bloques de 4 capas
  - AdaLN para condicionamiento de σ (diffusion blocks)
  - Capas pares comparten K
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from rope import RoPE
from dblock_modules import get_discrete_sigmas


class Config:
    dim = 512
    heads = 8
    kv_groups = 4
    layers = 16
    num_blocks = 4  # 16 capas / 4 bloques = 4 capas por bloque
    layers_per_block = 4
    ffn_dim = 2048
    block_size = 2048
    emb_num = 32000
    rotary_pct = 0.25
    drop = 0.0

    # DiffusionBlocks
    sigma_min = 0.002
    sigma_max = 80.0
    p_mean = -1.2
    p_std = 1.2
    gamma = 0.05  # overlap entre bloques
    sigma_data = 0.5

    # Capas pares comparten K
    share_k_even = True

    # Condicionamiento
    cond_hidden_size = 64  # dim // 8

    # DiffusionBlocks: noise on labels (True) or input (False)
    noise_on_labels = True  # True = como DiffusionBlocks original (necesita retrain)

    # Training
    batch_size = 8
    grad_acc = 1
    learning_rate = 3e-4
    weight_decay = 0.1
    betas = (0.9, 0.95)
    warm_up = 20


def repeat_kv(x, num_heads, num_kv_groups):
    if num_kv_groups == num_heads:
        return x
    return x.repeat_interleave(num_heads // num_kv_groups, dim=2)


def sinusoidal_embedding(timesteps, dim):
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    return torch.cat([emb.sin(), emb.cos()], dim=-1)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, sigma):
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor(sigma, dtype=torch.float32)
        if sigma.dim() == 0:
            sigma = sigma.unsqueeze(0)
        t_emb = sinusoidal_embedding(sigma, self.mlp[0].in_features)
        return self.mlp(t_emb)


class AdaLN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Sequential(nn.SiLU(), nn.Linear(in_features, out_features))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Atención Q–K=V + XSA. Sin LISA. Capas pares comparten K."""

    def __init__(self, config, layer_idx, share_k=False):
        super().__init__()
        self.num_heads = config.heads
        self.num_kv_groups = config.kv_groups
        self.head_dim = config.dim // config.heads
        self.share_k = share_k

        self.q_proj = nn.Linear(config.dim, self.num_heads * self.head_dim, bias=False)
        if not share_k:
            self.kv_proj = nn.Linear(config.dim, self.num_kv_groups * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.rope = RoPE(self.head_dim, rotary_pct=getattr(config, 'rotary_pct', 0.25))
        self.attn_dropout = nn.Dropout(config.drop)

    @staticmethod
    def _xsa(y, v):
        B, H, S, D = y.shape
        vn = F.normalize(v, p=2, dim=-1)
        vn = vn[:, :, -S:, :]
        return y - (y * vn).sum(dim=-1, keepdim=True) * vn

    def forward(self, x, k_shared=None):
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)

        if self.share_k and k_shared is not None:
            q, _ = self.rope(q, q, 0)
            kv = k_shared
        else:
            kv = self.kv_proj(x).view(B, T, self.num_kv_groups, self.head_dim)
            q, kv = self.rope(q, kv, 0)
        k = kv
        v = kv

        k = repeat_kv(k, self.num_heads, self.num_kv_groups)
        v = repeat_kv(v, self.num_heads, self.num_kv_groups)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )
        z = self._xsa(y, v)
        z = z.transpose(1, 2).contiguous().view(B, T, -1)

        return self.o_proj(z), kv

    def forward_with_cache(self, x, offset, cache, k_shared_full=None):
        B, S_new, _ = x.shape
        q_new = self.q_proj(x).view(B, S_new, self.num_heads, self.head_dim)

        if self.share_k:
            q_new, _ = self.rope(q_new, q_new, offset)
            k_full = k_shared_full
            new_cache = None
        else:
            kv_new = self.kv_proj(x).view(B, S_new, self.num_kv_groups, self.head_dim)
            q_new, kv_new = self.rope(q_new, kv_new, offset)
            if cache is not None:
                k_full = torch.cat([cache[0], kv_new], dim=1)
            else:
                k_full = kv_new
            new_cache = (k_full.clone(),)

        k = k_full
        v = k_full

        k_exp = repeat_kv(k, self.num_heads, self.num_kv_groups)
        v_exp = repeat_kv(v, self.num_heads, self.num_kv_groups)

        q = q_new.transpose(1, 2)
        k = k_exp.transpose(1, 2)
        v = v_exp.transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=(offset == 0),
        )
        z = self._xsa(y, v)
        z = z.transpose(1, 2).contiguous().view(B, S_new, -1)

        return self.o_proj(z), k_full, new_cache


class Block(nn.Module):
    """Bloque transformer con AdaLN para condicionamiento σ."""

    def __init__(self, config, layer_idx, share_k=False):
        super().__init__()
        self.layer_idx = layer_idx

        # AdaLN: 6 parámetros (shift/scale/gate para attn y FFN)
        self.adaLN_attn = AdaLN(config.cond_hidden_size, 3 * config.dim)
        self.adaLN_ffn = AdaLN(config.cond_hidden_size, 3 * config.dim)

        self.ln_1 = nn.LayerNorm(config.dim)
        self.attn = Attention(config, layer_idx, share_k=share_k)
        self.ln_2 = nn.LayerNorm(config.dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.dim, config.ffn_dim),
            nn.SiLU(),
            nn.Linear(config.ffn_dim, config.dim),
        )

    def forward(self, x, sigma_cond, k_shared=None):
        # AdaLN para attention
        shift_msa, scale_msa, gate_msa = self.adaLN_attn(sigma_cond).chunk(3, dim=-1)

        h = self.ln_1(x)
        h = modulate(h, shift_msa, scale_msa)
        attn_out, kv = self.attn(h, k_shared=k_shared)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # AdaLN para FFN
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_ffn(sigma_cond).chunk(3, dim=-1)

        h = self.ln_2(x)
        h = modulate(h, shift_mlp, scale_mlp)
        ffn_out = self.ffn(h)
        x = x + gate_mlp.unsqueeze(1) * ffn_out

        return x, kv

    def forward_with_cache(self, x, offset, cache, k_shared_full, sigma_cond):
        # Para inference: mismo forward pero con cache
        shift_msa, scale_msa, gate_msa = self.adaLN_attn(sigma_cond).chunk(3, dim=-1)

        h = self.ln_1(x)
        h = modulate(h, shift_msa, scale_msa)
        attn_out, k_full, new_cache = self.attn.forward_with_cache(h, offset, cache, k_shared_full)
        x = x + gate_msa.unsqueeze(1) * attn_out

        shift_mlp, scale_mlp, gate_mlp = self.adaLN_ffn(sigma_cond).chunk(3, dim=-1)

        h = self.ln_2(x)
        h = modulate(h, shift_mlp, scale_mlp)
        ffn_out = self.ffn(h)
        x = x + gate_mlp.unsqueeze(1) * ffn_out

        return x, k_full, new_cache


class DofiLLM(nn.Module):
    """Transformer autoregressivo con DiffusionBlocks.

    16 capas, 4 bloques de 4 capas.
    Cada bloque se entrena independientemente con condicionamiento σ.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Embedding compartido
        self.embeddings = nn.Embedding(config.emb_num, config.dim)

        # 16 capas transformer
        self.blocks = nn.ModuleList([
            Block(config, i, share_k=(config.share_k_even and (i % 2 == 1)))
            for i in range(config.layers)
        ])

        self.norm_f = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.emb_num, bias=False)
        self.embeddings.weight = self.lm_head.weight

        # Timestep embedder (σ → condición)
        self.timestep_embedder = TimestepEmbedder(256, config.cond_hidden_size)

        # Zero-init para AdaLN y lm_head (DiT best practice)
        self._init_dit()

        print(f"Dofi: {sum(p.numel() for p in self.parameters())/1e6:.2f}M params, "
              f"{config.layers} capas, {config.num_blocks} bloques de {config.layers_per_block}")

    def _init_dit(self):
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_attn.net[-1].weight)
            nn.init.zeros_(block.adaLN_attn.net[-1].bias)
            nn.init.zeros_(block.adaLN_ffn.net[-1].weight)
            nn.init.zeros_(block.adaLN_ffn.net[-1].bias)

    def get_block_layers(self, block_idx):
        """Retorna los índices de capas para un bloque dado."""
        start = block_idx * self.config.layers_per_block
        return list(range(start, start + self.config.layers_per_block))

    def forward_block(self, block_idx, input_ids, sigma, target_ids=None):
        """Forward solo por un bloque de capas (para training).

        noise_on_labels=True (DiffusionBlocks original):
          - Ruido en embeddings de labels (target_ids), input limpio
          - EDM parameterization: c_in, c_noise
        noise_on_labels=False:
          - Ruido en embeddings de input (input_ids)
        """
        sigma_val = sigma if isinstance(sigma, float) else sigma.item()
        sigma_data = self.config.sigma_data

        # EDM parameterization
        c_in = 1.0 / (sigma_val**2 + sigma_data**2)**0.5
        c_noise = 0.25 * math.log(max(sigma_val, 1e-8))
        sigma_cond = self.timestep_embedder(torch.tensor(c_noise, device=input_ids.device, dtype=torch.float32))

        x = self.embeddings(input_ids)

        if self.config.noise_on_labels and target_ids is not None:
            # DiffusionBlocks: ruido en LABEL embeddings
            z = self.embeddings(target_ids)
            if sigma_val > 0:
                zt = z + torch.randn_like(z) * sigma_val
            else:
                zt = z
            x = x + zt * c_in
        else:
            # Original: ruido en INPUT embeddings
            if sigma_val > 0:
                x = x + torch.randn_like(x) * sigma_val * c_in

        layer_indices = self.get_block_layers(block_idx)
        k_prev = None
        for i in layer_indices:
            x, kv = self.blocks[i](x, sigma_cond, k_shared=k_prev)
            k_prev = kv

        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits

    def forward(self, input_ids, sigma=None, block_idx=None):
        """Forward completo o por bloque.

        Si block_idx es None: forward completo (inference).
        Si block_idx es dado: forward solo por ese bloque (training).
        """
        if block_idx is not None:
            return self.forward_block(block_idx, input_ids, sigma)

        # Forward completo (inference)
        sigma_cond = self.timestep_embedder(sigma)
        x = self.embeddings(input_ids)

        k_prev = None
        for block in self.blocks:
            x, kv = block(x, sigma_cond, k_shared=k_prev)
            k_prev = kv

        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits

    def forward_with_cache(self, input_ids, offset, caches, sigma):
        """Forward con KV cache para inference."""
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor(sigma, device=input_ids.device, dtype=torch.float32)
        c_noise = 0.25 * torch.log(sigma)
        sigma_cond = self.timestep_embedder(c_noise)
        x = self.embeddings(input_ids)

        new_caches = []
        k_prev_full = None
        for i, block in enumerate(self.blocks):
            cache = caches[i] if caches is not None and i < len(caches) else None
            x, k_prev_full, new_cache = block.forward_with_cache(
                x, offset, cache, k_prev_full, sigma_cond
            )
            new_caches.append(new_cache)

        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits, new_caches

    @torch.no_grad()
    def ode_solve(self, input_ids, num_steps=4, sigma_min=0.002, sigma_max=80.0):
        """ODE para noise_on_labels: denoising iterativo con EDM parameterization.

        Cada paso corre UN bloque (4 capas), total = 4 bloques = 16 capas.
        """
        context = self.embeddings(input_ids)  # (B, L, D)
        B, L, D = context.shape
        sigma_data = self.config.sigma_data

        sigmas = get_discrete_sigmas(num_steps, sigma_min, sigma_max, dblock=True)
        sigmas = sigmas.to(input_ids.device)

        zt = torch.randn(B, 1, D, device=input_ids.device, dtype=context.dtype) * sigmas[0]

        for i in range(len(sigmas) - 1):
            sigma = sigmas[i]
            next_sigma = sigmas[i + 1]
            block_idx = min(i, self.config.num_blocks - 1)

            # EDM parameterization
            c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
            c_out = sigma * sigma_data / (sigma**2 + sigma_data**2)**0.5
            c_in = 1.0 / (sigma**2 + sigma_data**2)**0.5
            c_noise = 0.25 * torch.log(sigma)

            # Forward: context + zt escalado por c_in
            x = torch.cat([context, zt * c_in], dim=1)
            sc = self.timestep_embedder(c_noise)
            k_prev = None
            for j in self.get_block_layers(block_idx):
                x, kv = self.blocks[j](x, sc, k_shared=k_prev)
                k_prev = kv

            # EDM output en la posición del target (última)
            model_out = x[:, -1:, :] * c_out + zt * c_skip
            logits = self.lm_head(model_out)

            # Convertir a embedding para Euler step
            probs = F.softmax(logits / 0.7, dim=-1)
            denoised = probs @ self.embeddings.weight

            # Euler step
            d = (zt - denoised) / sigma
            dt = next_sigma - sigma
            zt = zt + dt * d

        # Forward final por TODOS los bloques con EDM
        sigma = sigmas[-1]
        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
        c_out = sigma * sigma_data / (sigma**2 + sigma_data**2)**0.5
        c_in = 1.0 / (sigma**2 + sigma_data**2)**0.5
        c_noise = 0.25 * torch.log(sigma)

        x = torch.cat([context, zt * c_in], dim=1)
        sc = self.timestep_embedder(c_noise)
        k_prev = None
        for block in self.blocks:
            x, kv = block(x, sc, k_shared=k_prev)
            k_prev = kv
        model_out = x[:, -1:, :] * c_out + zt * c_skip
        logits = self.lm_head(model_out)
        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=0.8, top_k=40,
                 sigma=0.002, use_ode=False, ode_steps=20):
        """Generación.

        noise_on_labels: siempre usa ODE (denoising iterativo).
        noise_on_input: KV cache normal (rápido).
        """
        device = input_ids.device

        if self.config.noise_on_labels or use_ode:
            # ODE: denoising iterativo para noise_on_labels
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
        else:
            offset = 0
            caches = None
            for _ in range(max_new_tokens):
                logits, caches = self.forward_with_cache(input_ids[:, -1:], offset, caches, sigma)
                offset += input_ids.shape[1]
                next_logits = logits[:, -1, :] / temperature
                if top_k > 0:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, [-1]]] = float('-inf')
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
