/// TransformerLM — espejo de `moe-plus/model.py::TransformerLM` (hibrido denso/MoE).
use candle_core::{Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder};

use super::config::Config;
use super::moe_block::MoEBlock;
use super::rms_norm::RMSNorm;

pub enum Layer {
    Moe(MoEBlock),
}

impl Layer {
    pub fn forward(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        match self {
            Layer::Moe(b) => b.forward(x, offset),
        }
    }
    pub fn forward_with_cache(&self, x: &Tensor, offset: usize, cache: Option<&super::cache::KVCache>) -> Result<(Tensor, super::cache::KVCache)> {
        match self {
            Layer::Moe(b) => b.forward_with_cache(x, offset, cache),
        }
    }
}

pub struct LLM {
    pub config: Config,
    pub embedding: Embedding,
    pub layers: Vec<Layer>,
    pub final_norm: RMSNorm,
}

impl LLM {
    pub fn new(vb: VarBuilder, cfg: Config) -> Result<Self> {
        let embedding = candle_nn::embedding(cfg.vocab_size, cfg.d_model, vb.pp("embedding"))?;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let is_dense = i < cfg.n_dense_start || i >= cfg.num_layers - cfg.n_dense_end;
            let use_moe = cfg.use_moe && !is_dense;
            let layer = MoEBlock::new(vb.pp(format!("layers.{}", i)), &cfg, use_moe)?;
            layers.push(Layer::Moe(layer));
        }
        let final_norm = RMSNorm::new(cfg.d_model, cfg.norm_eps, vb.pp("final_norm"))?;
        Ok(Self { config: cfg, embedding, layers, final_norm })
    }

    pub fn forward(&self, input_ids: &Tensor, offset: usize) -> Result<Tensor> {
        let mut x = self.embedding.forward(input_ids)?;
        for layer in &self.layers {
            x = layer.forward(&x, offset)?;
        }
        let x = self.final_norm.forward(&x)?;
        let w = self.embedding.embeddings();
        x.matmul(&w.t()?)
    }

    pub fn forward_with_cache(&self, input_ids: &Tensor, offset: usize, caches: &mut Vec<Option<super::cache::KVCache>>) -> Result<Tensor> {
        let mut x = self.embedding.forward(input_ids)?;
        for (i, layer) in self.layers.iter().enumerate() {
            let (h, new_cache) = layer.forward_with_cache(&x, offset, caches[i].as_ref())?;
            x = h;
            caches[i] = Some(new_cache);
        }
        let x = self.final_norm.forward(&x)?;
        let w = self.embedding.embeddings();
        x.matmul(&w.t()?)
    }
}
