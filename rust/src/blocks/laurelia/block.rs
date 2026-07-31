/// Transformer Block — espejo de `laurelia-llm/model.py::Block`.
///
/// x = x + attn(ln_1(x)); x = x + mlp(ln_2(x))

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use super::attention::Attention;
use super::cache::KVCache;
use super::mlp::MLP;
use super::rms_norm::RMSNorm;

pub struct Block {
    pub ln_1: RMSNorm,
    pub attn: Attention,
    pub ln_2: RMSNorm,
    pub mlp: MLP,
}

impl Block {
    pub fn new(
        vb: VarBuilder,
        dim: usize,
        heads: usize,
        kv_groups: usize,
        head_dim: usize,
        ffn_dim: usize,
        block_size: usize,
        rotary_pct: f64,
        eps: f32,
    ) -> Result<Self> {
        let ln_1 = RMSNorm::new(dim, eps, vb.pp("ln_1"))?;
        let attn = Attention::new(
            vb.pp("attn"),
            dim,
            heads,
            kv_groups,
            head_dim,
            block_size,
            rotary_pct,
            true,
        )?;
        let ln_2 = RMSNorm::new(dim, eps, vb.pp("ln_2"))?;
        let mlp = MLP::new(vb.pp("mlp"), dim, ffn_dim)?;
        Ok(Self { ln_1, attn, ln_2, mlp })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.attn.forward(&self.ln_1.forward(x)?, 0)?;
        let x = (x + h)?;
        let h = self.mlp.forward(&self.ln_2.forward(&x)?)?;
        let x = (x + h)?;
        Ok(x)
    }

    pub fn forward_with_cache(
        &self,
        x: &Tensor,
        offset: usize,
        cache: Option<&KVCache>,
    ) -> Result<(Tensor, KVCache)> {
        let h = self.ln_1.forward(x)?;
        let (h, new_cache) = self.attn.forward_with_cache(&h, offset, cache)?;
        let x = (x + h)?;
        let h = self.mlp.forward(&self.ln_2.forward(&x)?)?;
        let x = (x + h)?;
        Ok((x, new_cache))
    }
}
