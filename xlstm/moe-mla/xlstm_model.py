"""xLSTM Large MoE Model: embedding → [MoE/dense blocks] → norm → lm_head."""
import sys, os, math
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, "..", ".."))
import torch
from torch import nn
from xlstm.xlstm_large.model import xLSTMLargeConfig, mLSTMBlock
from xlstm.xlstm_large.components import RMSNorm, soft_cap
from xlstm_moe_block import xLSTMMoEBlock

class xLSTMMoEModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_layers=8, num_heads=4,
                 moe_at=None, n_experts=4, top_k=1, n_shared=1, expert_dim=None,
                 capacity_factor=1.0, z_loss_gamma=0.001, bias_decay=0.1,
                 noise_std=0.0, max_seq_len=1024):
        super().__init__()
        if moe_at is None:
            moe_at = list(range(num_layers))

        config = xLSTMLargeConfig(
            embedding_dim=d_model, num_heads=num_heads, num_blocks=num_layers,
            vocab_size=vocab_size, use_bias=False,
        )

        moe_cfg = {
            "expert_dim": expert_dim,
            "n_experts": n_experts,
            "top_k": top_k,
            "n_shared": n_shared,
            "capacity_factor": capacity_factor,
            "z_loss_gamma": z_loss_gamma,
            "bias_decay": bias_decay,
            "noise_std": noise_std,
        }

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            if i in moe_at:
                block = xLSTMMoEBlock(config, moe_cfg, layer_idx=i)
            else:
                block = mLSTMBlock(config)
            self.blocks.append(block)

        self.out_norm = RMSNorm(
            num_features=d_model, eps=config.norm_eps,
            use_weight=True, use_bias=False,
            force_float32_reductions=config.norm_reduction_force_float32,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * len(self.blocks)))
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        x = self.embedding(idx)
        aux_loss = 0.0
        mlstm_state = {i: None for i in range(len(self.blocks))}
        for i, block in enumerate(self.blocks):
            if isinstance(block, xLSTMMoEBlock):
                x, loss, mlstm_state[i] = block(x, mlstm_state[i])
                aux_loss = aux_loss + loss
            else:
                x, mlstm_state[i] = block(x, mlstm_state[i])
        x = self.out_norm(x)
        logits = self.lm_head(x)
        logits = soft_cap(logits, 30.0)
        return logits, aux_loss

    def generate(self, idx, max_new_tokens=50, temperature=1.0, top_k=50):
        for _ in range(max_new_tokens):
            logits, _ = self.forward(idx[:, -1024:])
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                vals, _ = torch.topk(logits, top_k)
                logits[logits < vals[:, -1:]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_tok], dim=-1)
        return idx
