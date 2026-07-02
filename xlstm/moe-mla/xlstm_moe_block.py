"""xLSTM Large MoE block: RMSNorm → mLSTM → +residual → RMSNorm → MoE → +residual."""
import sys, os, math
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, "..", ".."))
import torch
from torch import nn
from xlstm.xlstm_large.model import mLSTMLayer, mLSTMLayerConfig
from xlstm.xlstm_large.components import RMSNorm
from mlstm_kernels.torch.backend_module import mLSTMBackendConfig
from moe import MoELayer

class xLSTMMoEBlock(nn.Module):
    """mLSTM block (Large) with MoE replacing FeedForward."""

    def __init__(self, config, moe_cfg, layer_idx=0):
        super().__init__()
        self._layer_idx = layer_idx
        self.norm_mlstm = RMSNorm(
            num_features=config.embedding_dim, eps=config.norm_eps,
            use_weight=True, use_bias=False,
            force_float32_reductions=config.norm_reduction_force_float32,
        )
        self.mlstm_layer = mLSTMLayer(mLSTMLayerConfig(
            embedding_dim=config.embedding_dim, num_heads=config.num_heads,
            use_bias=config.use_bias, norm_eps=config.norm_eps,
            norm_reduction_force_float32=config.norm_reduction_force_float32,
            qk_dim_factor=config.qk_dim_factor, v_dim_factor=config.v_dim_factor,
            gate_soft_cap=config.gate_soft_cap,
            weight_mode=config.weight_mode,
            mlstm_backend=mLSTMBackendConfig(
                chunkwise_kernel=config.chunkwise_kernel,
                sequence_kernel=config.sequence_kernel,
                step_kernel=config.step_kernel, mode=config.mode,
                chunk_size=config.chunk_size,
                return_last_states=config.return_last_states,
                autocast_kernel_dtype=config.autocast_kernel_dtype,
                eps=config.eps,
                inference_state_dtype=config.inference_state_dtype,
            ),
        ))
        self.norm_moe = RMSNorm(
            num_features=config.embedding_dim, eps=config.norm_eps,
            use_weight=True, use_bias=False,
            force_float32_reductions=config.norm_reduction_force_float32,
        )
        qk_dim = int(config.embedding_dim * config.qk_dim_factor)
        head_dim_qk = qk_dim // config.num_heads
        self.moe = MoELayer(
            d_model=config.embedding_dim,
            expert_dim=moe_cfg.get("expert_dim"),
            n_experts=moe_cfg["n_experts"],
            top_k=moe_cfg["top_k"],
            n_shared=moe_cfg["n_shared"],
            capacity_factor=moe_cfg.get("capacity_factor", 1.0),
            z_loss_gamma=moe_cfg.get("z_loss_gamma", 0.001),
            bias_decay=moe_cfg.get("bias_decay", 0.1),
            noise_std=moe_cfg.get("noise_std", 0.0),
            head_dim_qk=head_dim_qk,
        )

    def forward(self, x, state=None):
        x_mlstm = self.norm_mlstm(x)
        x_mlstm, state = self.mlstm_layer(x_mlstm, state)
        x = x + x_mlstm
        x_moe = self.norm_moe(x)
        x_moe, aux_loss = self.moe(x_moe, state)
        x = x + x_moe
        return x, aux_loss, state

    def step(self, x, state=None):
        x_mlstm, state = self.mlstm_layer(self.norm_mlstm(x), state)
        x = x + x_mlstm
        x_moe, _ = self.moe(self.norm_moe(x), state)
        x = x + x_moe
        return x, state
