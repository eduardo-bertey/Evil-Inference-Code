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

    pub fn device(&self) -> &candle_core::Device {
        self.embedding.embeddings().device()
    }

    pub fn generate(
        &self,
        input_ids: &Tensor,
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        repetition_penalty: f32,
    ) -> Result<Tensor> {
        let prompt_len = input_ids.dim(1)?;
        let mut caches: Vec<Option<crate::blocks::moe::cache::KVCache>> =
            vec![None; self.layers.len()];
        let mut tokens: Vec<u32> = input_ids.get(0)?.to_vec1()?;
        let mut logits = Tensor::zeros(
            (1, 1, self.config.vocab_size),
            candle_core::DType::F32,
            self.device(),
        )?;
        for i in 0..prompt_len {
            let t = Tensor::from_vec(vec![tokens[i]], (1, 1), self.device())?;
            logits = self.forward_with_cache(&t, i, &mut caches)?;
        }

        let mut gen: Vec<u32> = Vec::new();
        for gen_i in 0..max_new_tokens {
            let next = self.sample(&logits, &tokens, temperature, top_k, top_p, repetition_penalty)?;
            tokens.push(next);
            gen.push(next);

            let t = Tensor::from_vec(vec![next], (1, 1), self.device())?;
            logits = self.forward_with_cache(&t, prompt_len + gen_i, &mut caches)?;
        }

        let all = [tokens[..prompt_len].to_vec(), gen].concat();
        let n = all.len();
        Ok(Tensor::from_vec(all, (1, n), self.device())?)
    }

    fn sample(
        &self,
        logits: &Tensor,
        tokens: &[u32],
        temperature: f32,
        top_k: usize,
        top_p: f32,
        repetition_penalty: f32,
    ) -> Result<u32> {
        let mut logits: Vec<f32> = logits
            .squeeze(0)?
            .squeeze(0)?
            .to_dtype(candle_core::DType::F32)?
            .to_vec1()?;
        let vocab = logits.len();

        let t = temperature.max(1e-6);
        if t != 1.0 {
            for v in logits.iter_mut() {
                *v /= t;
            }
        }

        if repetition_penalty != 1.0 {
            let mut seen = vec![false; vocab];
            for &tok in tokens {
                let i = tok as usize;
                if i >= vocab || seen[i] {
                    continue;
                }
                seen[i] = true;
                if logits[i] > 0.0 {
                    logits[i] /= repetition_penalty;
                } else {
                    logits[i] *= repetition_penalty;
                }
            }
        }

        let mut order: Vec<usize> = (0..vocab).collect();
        if top_k > 0 && top_k < vocab {
            order.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
            order.truncate(top_k);
            let mut keep = vec![false; vocab];
            for &i in &order {
                keep[i] = true;
            }
            for i in 0..vocab {
                if !keep[i] {
                    logits[i] = f32::NEG_INFINITY;
                }
            }
        }

        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = logits
            .iter()
            .map(|&v| {
                if v == f32::NEG_INFINITY {
                    0.0
                } else {
                    (v - max).exp()
                }
            })
            .collect();
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for p in probs.iter_mut() {
                *p /= sum;
            }
        }

        if top_p < 1.0 {
            order.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal));
            let mut cum = 0.0;
            let mut keep = vec![false; vocab];
            for &i in &order {
                cum += probs[i];
                keep[i] = true;
                if cum >= top_p {
                    break;
                }
            }
            for i in 0..vocab {
                if !keep[i] {
                    probs[i] = 0.0;
                }
            }
        }

        let u = rand::random::<f32>();
        let mut cum = 0.0;
        for i in 0..vocab {
            cum += probs[i];
            if u < cum {
                return Ok(i as u32);
            }
        }
        Ok((vocab - 1) as u32)
    }
}
