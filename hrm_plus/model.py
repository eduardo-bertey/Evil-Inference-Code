"""HRM (Hierarchical Reasoning Model) adaptado desde el transformer normal de laurelia.

- Reusa Block/Attention (GQA + RoPE) de laurelia, que ya usa
  torch.nn.functional.scaled_dot_product_attention (SIN flash-attention)
  -> compatible con T4 CUDA (Colab).
- Arquitectura: dos niveles recurrentes (H = high, L = low). En cada ciclo
  H el nivel L itera L_cycles veces inyectando z_H en z_L (suma), y luego H
  se actualiza inyectando z_L en z_H. Al final z_H pasa por norm + lm_head.
- El estado latente z_L arranca desde un buffer aprendido (zL_init) que se
  difunde a toda la secuencia.

No se copió código de HRM-Text: se reimplementó el esquema jerárquico sobre
los bloques SDPA que ya funcionaban en laurelia.
"""

import math
import inspect
import os as _os
import sys as _sys
# Colab/terminal: el script puede correrse desde otro cwd; asegura que los
# módulos del mismo directorio (rope, moe_lineal) se puedan importar.
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import torch
import torch.nn as nn
import torch.nn.functional as F
from rope import RoPE
from moe_lineal import MoELineal


class Config:
    # --- transformer base (igual que laurelia) ---
    drop = 0.0
    dim = 768
    heads = 12
    kv_groups = 4
    ffn_dim = 3072
    block_size = 1024
    emb_num = 32000
    rotary_pct = 0.25

    # --- HRM: niveles recurrentes ---
    # --- HRM+: atención Keyless (sin K, cache solo-V) con MoE en V y O
    # (Q densa; hallazgo SwitchHead). 4 expertos top-2 + 1 compartido
    # fijo por MoE. FFN intacto. Todo torch puro, sin CUDA avanzado.
    moe_expertos = 4
    moe_topk = 2
    moe_aux_w = 0.03
    moe_z_w = 0.01
    moe_ruido = True
    # capas POR nivel (H y L usan este TransformerCore)
    layers = 4
    h_cycles = 2
    l_cycles = 4
    # Retropropagación truncada (BPTT): solo las últimas `bp_steps` pasadas
    # guardan gradiente. Las anteriores corren sin grafo (se libera VRAM).
    # 10 pasadas con grad simultáneas revientan la T4; con esto cabe fácil.
    bp_steps = 2

    batch_size: int = 8
    grad_acc: int = 6
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    warm_up: int = 50


def repeat_kv(x, num_heads, num_kv_groups):
    if num_kv_groups == num_heads:
        return x
    return x.repeat_interleave(num_heads // num_kv_groups, dim=2)


class Attention(nn.Module):
    """Keyless + MoE en V y O (Q densa, hallazgo SwitchHead: Q/K normales
    + MoE en V+O rinde mejor). Sin K: Q' = X·WQ·WR, scores contra V,
    cache solo-V. FFN intacto. Un solo SDPA."""
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.heads
        self.num_kv_groups = config.kv_groups
        self.head_dim = config.dim // config.heads
        self.causal = True

        self.q_proj = nn.Linear(config.dim, self.num_heads * self.head_dim, bias=False)
        self.q_proj.is_attention = True
        self.v_proj = MoELineal(config.dim, self.num_kv_groups * self.head_dim, config)
        self.o_proj = MoELineal(config.dim, config.dim, config, residual=True)
        self.wr = nn.Parameter(torch.empty(self.num_heads, self.head_dim, self.head_dim))
        nn.init.trunc_normal_(self.wr, std=0.02)
        self.rope = RoPE(self.head_dim, rotary_pct=getattr(config, "rotary_pct", 0.25))
        self.attn_dropout = nn.Dropout(config.drop)

    def _qp(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        return torch.einsum("bthd,hde->bthe", q, self.wr)

    @torch.no_grad()
    def fusionar_wq(self):
        """WQ_eff = WQ·WR por cabeza (keylees.md §8). WR es la K
        compactada en el espacio de V: en inferencia Q' = X·WQ_eff
        en un paso en vez de X→WQ→WR."""
        wq = self.q_proj.weight.view(self.num_heads, self.head_dim, -1)
        return torch.einsum("hdo,hde->heo", wq, self.wr)

    def forward(self, x):
        B, T, D = x.shape
        qp = self._qp(x)
        v = self.v_proj(x).view(B, T, self.num_kv_groups, self.head_dim)

        qp, v = self.rope(qp, v, 0)

        v = repeat_kv(v, self.num_heads, self.num_kv_groups)

        q = qp.transpose(1, 2)
        v = v.transpose(1, 2)

        att_output = F.scaled_dot_product_attention(
            q, v, v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )

        att_output = att_output.transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(att_output)

    def forward_with_cache(self, x, offset, cache):
        B, S_new, _ = x.shape
        qp_new = self._qp(x)
        v_new = self.v_proj(x).view(B, S_new, self.num_kv_groups, self.head_dim)
        qp_new, v_new = self.rope(qp_new, v_new, offset)

        if cache is None:
            v_full = v_new
        else:
            v_full = torch.cat([cache, v_new], dim=1)
        new_cache = v_full.clone()

        v_exp = repeat_kv(v_full, self.num_heads, self.num_kv_groups)
        q = qp_new.transpose(1, 2)
        v = v_exp.transpose(1, 2)

        att_output = F.scaled_dot_product_attention(q, v, v, is_causal=(cache is None))
        att_output = att_output.transpose(1, 2).contiguous().view(B, S_new, -1)
        return self.o_proj(att_output), new_cache


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.dim, 2 * config.ffn_dim, bias=False)
        self.fc2 = nn.Linear(config.ffn_dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.drop)
        self.fc2.is_residual_proj = True

    def forward(self, x):
        x = self.fc1(x)
        x, gate = x.chunk(2, dim=-1)
        x = x * F.silu(gate)
        return self.dropout(self.fc2(x))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.dim)
        self.attn = Attention(config)
        self.ln_2 = nn.RMSNorm(config.dim)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def forward_with_cache(self, x, offset, cache):
        h = self.ln_1(x)
        h, new_cache = self.attn.forward_with_cache(h, offset, cache)
        x = x + h
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache


class TransformerCore(nn.Module):
    """Un 'nivel' (H o L) del HRM: stack de Block + norm final."""
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.norm_f = nn.RMSNorm(config.dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.norm_f(x)


class HRM(nn.Module):
    """Hierarchical Reasoning Model.

    forward(input_ids, labels=None) -> (logits, loss)
    Generación autoregresiva simple (re-ejecuta el forward completo por token,
    sin KV-cache, para mantenerlo simple y correcto en T4).
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.embeddings = nn.Embedding(config.emb_num, config.dim)
        # Niveles recurrentes (cada uno es un TransformerCore de config.layers)
        self.H_level = TransformerCore(config)
        self.L_level = TransformerCore(config)

        self.norm_f = nn.RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.emb_num, bias=False)
        # weight tying
        self.embeddings.weight = self.lm_head.weight

        # Estado latente inicial del nivel L (se difunde a [B, S, dim] al sumar)
        self.zL_init = nn.Buffer(torch.empty(config.dim))
        torch.nn.init.trunc_normal_(self.zL_init, std=1.0)

        self.apply(self._init_weights)
        # alias para que train.py (laurelia) cuente params sin tocarlo
        self.blocks = self.L_level.blocks
        print("HRM params: %.2fM" % (sum(p.numel() for p in self.parameters()) / 1e6,))

    @torch.no_grad()
    def _init_weights(self, module):
        n_layer = self.config.layers * (self.config.h_cycles + self.config.h_cycles * self.config.l_cycles)
        if isinstance(module, nn.Linear):
            if module is self.lm_head:
                return
            w_fan_in = module.weight.shape[-1]
            base_std = (1.0 / w_fan_in) ** 0.5
            if hasattr(module, "is_residual_proj"):
                final_std = base_std / math.sqrt(2 * max(1, n_layer))
            elif hasattr(module, "is_attention"):
                final_std = base_std * 0.7
            else:
                final_std = base_std
            torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=final_std, a=-2 * final_std, b=2 * final_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def _recur(self, z_H, z_L):
        """Recurrencia jerárquica con retropropagación truncada (BPTT).

        Se corren TODOS los ciclos en forward, pero solo las últimas `bp_steps`
        pasadas conservan el grafo de autograd; el resto corre con
        torch.set_grad_enabled(False). Así la VRAM no escala con (H_cycles x
        L_cycles) sino con unas pocas pasadas, evitando OOM en T4.
        En inferencia (sin grad) todo corre en no_grad de todos modos.
        """
        cfg = self.config
        total_L = cfg.h_cycles * cfg.l_cycles
        bp = max(1, getattr(cfg, "bp_steps", 2))
        H_bp = min(cfg.h_cycles, max(1, bp - 1))
        L_bp = max(1, bp - H_bp)

        seq_k = 0
        for i in range(cfg.h_cycles):
            for _k in range(cfg.l_cycles):
                keep = seq_k >= total_L - L_bp
                with torch.set_grad_enabled(torch.is_grad_enabled() and keep):
                    z_L = self.L_level(z_L + z_H)  # inyección: z_H entra en z_L
                seq_k += 1
            keep_h = i >= cfg.h_cycles - H_bp
            with torch.set_grad_enabled(torch.is_grad_enabled() and keep_h):
                z_H = self.H_level(z_H + z_L)  # inyección: z_L entra en z_H
        return z_H

    def forward(self, input_ids, labels=None):
        z_H = self.embeddings(input_ids)        # [B, S, dim]
        z_L = self.zL_init                     # [dim] -> broadcast
        z_H = self._recur(z_H, z_L)
        h = self.norm_f(z_H)
        logits = self.lm_head(h)

        aux = self.aux_total()
        self.ultimo_aux = aux.detach() if torch.is_tensor(aux) else torch.tensor(0.0)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.0, reduction="mean")
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            if torch.is_tensor(aux) and aux.numel() == 1 and aux.item() != 0.0:
                loss = loss + aux
        return logits, loss

    def aux_total(self):
        """Suma last_aux de todos los bloques MoA (0.0 si no hay)."""
        total = torch.tensor(0.0)
        for m in self.modules():
            a = getattr(m, "last_aux", None)
            if torch.is_tensor(a) and a.numel() == 1:
                total = total.to(a.device) + a
        return total

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=0.8, top_k=50,
                 top_p=0.9, repetition_penalty=1.1, eos_token_id=None):
        for _ in range(max_new_tokens):
            logits, _ = self.forward(input_ids)
            logits_last = logits[:, -1, :] / max(temperature, 1e-6)

            if repetition_penalty != 1.0:
                for tok in input_ids[0].unique():
                    if logits_last[0, tok] > 0:
                        logits_last[0, tok] /= repetition_penalty
                    else:
                        logits_last[0, tok] *= repetition_penalty

            if top_k > 0:
                v, _ = torch.topk(logits_last, min(top_k, logits_last.size(-1)))
                logits_last[logits_last < v[:, [-1]]] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits_last, descending=True)
                cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cumulative - torch.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[mask] = float("-inf")
                logits_last.scatter_(1, sorted_idx, sorted_logits)

            probs = torch.softmax(logits_last, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            if eos_token_id is not None and next_tok.item() == eos_token_id:
                break
            input_ids = torch.cat([input_ids, next_tok], dim=1)
        return input_ids


# Alias para no tocar train.py (laurelia importa `from model import LLM, Config`)
LLM = HRM
