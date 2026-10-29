/// Transformer block con MoE — espejo de `moe-plus/moe_block.py::LayerWithMoE` (inferencia).
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use super::attention::Attention;
use super::cache::KVCache;
use super::config::Config;
use super::moe::{DenseFFN, MoELayer};
use super::rms_norm::RMSNorm;

pub enum Ffn {
    Dense(DenseFFN),
    Moe(MoELayer),
}

pub struct MoEBlock {
    pub attn_norm: RMSNorm,
    pub attn: Attention,
    pub ffn_norm: RMSNorm,
    pub ffn: Ffn,
}

impl MoEBlock {
    pub fn new(vb: VarBuilder, cfg: &Config, use_moe: bool) -> Result<Self> {
        let dim = cfg.d_model;
        let hidden = ((dim as f64 * cfg.ffn_expansion) as usize / 64) * 64;
        let hidden = hidden.max(64);
        let attn = Attention::new(
            vb.pp("attn"), dim, cfg.num_heads, cfg.num_kv_groups,
            cfg.head_dim_or(), cfg.max_seq_len, cfg.rope_base, cfg.rope_scaling,
            true, false,
        )?;
        let attn_norm = RMSNorm::new(dim, cfg.norm_eps, vb.pp("attn_norm"))?;
        let ffn_norm = RMSNorm::new(dim, cfg.norm_eps, vb.pp("ffn_norm"))?;
        let ffn = if use_moe {
            Ffn::Moe(MoELayer::new(
                vb.pp("ffn"), dim, cfg.n_experts, cfg.top_k, cfg.n_shared,
                if cfg.expert_dim == 0 { hidden } else { cfg.expert_dim },
                None, false,
            )?)
        } else {
            Ffn::Dense(DenseFFN::new(vb.pp("ffn"), dim, hidden, false)?)
        };
        Ok(Self { attn_norm, attn, ffn_norm, ffn })
    }

    pub fn forward(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let residual = x.clone();
        let h = self.attn_norm.forward(x)?;
        let h = self.attn.forward(&h, offset)?;
        let x = (residual + h)?;
        let residual = x.clone();
        let h = self.ffn_norm.forward(&x)?;
        let h = match &self.ffn {
            Ffn::Dense(d) => d.forward(&h)?,
            Ffn::Moe(m) => m.forward(&h, None)?.0,
        };
        Ok((residual + h)?)
    }

    pub fn forward_with_cache(&self, x: &Tensor, offset: usize, cache: Option<&KVCache>) -> Result<(Tensor, KVCache)> {
        let residual = x.clone();
        let h = self.attn_norm.forward(x)?;
        let (h, new_cache) = self.attn.forward_with_cache(&h, offset, cache)?;
        let x = (residual + h)?;
        let residual = x.clone();
        let h = self.ffn_norm.forward(&x)?;
        let h = match &self.ffn {
            Ffn::Dense(d) => d.forward(&h)?,
            Ffn::Moe(m) => m.forward(&h, None)?.0,
        };
        Ok(((residual + h)?, new_cache))
    }
}
