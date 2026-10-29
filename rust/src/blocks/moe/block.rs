/// Transformer block (dense) — espejo de `moe-plus/block.py` (pre-norm).
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use super::attention::Attention;
use super::mlp::Mlp;
use super::rms_norm::RMSNorm;

pub struct Block {
    pub attn_norm: RMSNorm,
    pub attn: Attention,
    pub ffn_norm: RMSNorm,
    pub mlp: Mlp,
}

impl Block {
    pub fn new(vb: VarBuilder, cfg: &super::config::Config) -> Result<Self> {
        let dim = cfg.d_model;
        let hidden = (dim as f64 * cfg.ffn_expansion) as usize;
        let hidden = (hidden / 64) * 64;
        let hidden = hidden.max(64);
        let attn = Attention::new(
            vb.pp("attn"),
            dim,
            cfg.num_heads,
            cfg.num_kv_groups,
            cfg.head_dim_or(),
            cfg.max_seq_len,
            cfg.rope_base,
            cfg.rope_scaling,
            true,
            false,
        )?;
        let attn_norm = RMSNorm::new(dim, cfg.norm_eps, vb.pp("attn_norm"))?;
        let ffn_norm = RMSNorm::new(dim, cfg.norm_eps, vb.pp("ffn_norm"))?;
        let mlp = Mlp::new(vb.pp("mlp"), dim, hidden, false)?;
        Ok(Self { attn_norm, attn, ffn_norm, mlp })
    }

    pub fn forward(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let residual = x.clone();
        let h = self.attn_norm.forward(x)?;
        let h = self.attn.forward(&h, offset)?;
        let x = (residual + h)?;
        let residual = x.clone();
        let h = self.ffn_norm.forward(&x)?;
        let h = self.mlp.forward(&h)?;
        Ok((residual + h)?)
    }

    pub fn forward_with_cache(&self, x: &Tensor, offset: usize, cache: Option<&super::cache::KVCache>) -> Result<(Tensor, super::cache::KVCache)> {
        let residual = x.clone();
        let h = self.attn_norm.forward(x)?;
        let (h, new_cache) = self.attn.forward_with_cache(&h, offset, cache)?;
        let x = (residual + h)?;
        let residual = x.clone();
        let h = self.ffn_norm.forward(&x)?;
        let h = self.mlp.forward(&h)?;
        Ok(((residual + h)?, new_cache))
    }
}
