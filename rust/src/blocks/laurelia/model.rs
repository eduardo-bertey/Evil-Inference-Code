/// LLM — espejo de `laurelia-llm/model.py::LLM`, solo inferencia.
///
/// Embeddings + N blocks + norm_f + lm_head tied a embeddings.
/// generate(): temperature, top_k, top_p, repetition_penalty, eos.

use candle_core::IndexOp;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder};

use super::block::Block;
use super::cache::KVCache;
use super::config::Config;
use super::rms_norm::RMSNorm;

pub struct LLM {
    pub config: Config,
    pub embeddings: Embedding,
    pub blocks: Vec<Block>,
    pub norm_f: RMSNorm,
    pub lm_head: candle_nn::Linear,
}

impl LLM {
    pub fn new(vb: VarBuilder, config: &Config) -> Result<Self> {
        let dim = config.dim;
        let head_dim = config.head_dim();
        let eps = 1e-5;

        let embeddings = candle_nn::Embedding::new(
            vb.get((config.emb_num, dim), "embeddings.weight")?,
            dim,
        );

        let mut blocks = Vec::with_capacity(config.layers);
        for i in 0..config.layers {
            let block = Block::new(
                vb.pp(format!("blocks.{i}")),
                dim,
                config.heads,
                config.kv_groups,
                head_dim,
                config.ffn_dim,
                config.block_size,
                config.rotary_pct,
                eps,
            )?;
            blocks.push(block);
        }

        let norm_f = RMSNorm::new(dim, eps, vb.pp("norm_f"))?;

        let lm_head = candle_nn::Linear::new(
            vb.get((config.emb_num, dim), "lm_head.weight")
                .or_else(|_| vb.get((config.emb_num, dim), "head.emb_weight"))
                .or_else(|_| vb.get((config.emb_num, dim), "embeddings.weight"))?,
            None,
        );

        Ok(Self {
            config: config.clone(),
            embeddings,
            blocks,
            norm_f,
            lm_head,
        })
    }

    pub fn device(&self) -> &Device {
        self.embeddings.embeddings().device()
    }

    pub fn dtype(&self) -> DType {
        self.embeddings.embeddings().dtype()
    }

    /// Forward completo (prefill de toda la secuencia, sin cache).
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let mut x = self.embeddings.forward(input_ids)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        let x = self.norm_f.forward(&x)?;
        self.lm_head.forward(&x)
    }

    /// Forward con KV cache.
    ///
    /// input_ids: (B, S_new). offset: posición inicial.
    /// caches: Vec con un KVCache por capa (None en prefill).
    pub fn forward_with_cache(
        &self,
        input_ids: &Tensor,
        offset: usize,
        caches: &mut Vec<Option<KVCache>>,
    ) -> Result<Tensor> {
        let mut x = self.embeddings.forward(input_ids)?;
        for (i, block) in self.blocks.iter().enumerate() {
            let cache = caches[i].as_ref();
            let (h, new_cache) = block.forward_with_cache(&x, offset, cache)?;
            caches[i] = Some(new_cache);
            x = h;
        }
        let x = self.norm_f.forward(&x)?;
        self.lm_head.forward(&x)
    }

    /// Muestra el token siguiente con top_k + top_p + repetition_penalty.
    /// Todo el muestreo en Rust puro (Vec<f32>), mismo flujo que el Python.
    fn sample(
        &self,
        logits: &Tensor, // (1, 1, vocab)
        tokens: &[u32],
        temperature: f32,
        top_k: usize,
        top_p: f32,
        repetition_penalty: f32,
    ) -> Result<u32> {
        let mut logits: Vec<f32> = logits
            .squeeze(0)?
            .squeeze(0)?
            .to_dtype(DType::F32)?
            .to_vec1()?;
        let vocab = logits.len();

        if temperature != 1.0 {
            for v in logits.iter_mut() {
                *v /= temperature;
            }
        }

        // repetition_penalty sobre tokens ya vistos (solo el último bastaría,
        // pero replicamos el Python sobre todos)
        if repetition_penalty != 1.0 {
            let mut seen = vec![false; vocab];
            for &t in tokens {
                let t = t as usize;
                if t >= vocab || seen[t] {
                    continue;
                }
                seen[t] = true;
                if logits[t] > 0.0 {
                    logits[t] /= repetition_penalty;
                } else {
                    logits[t] *= repetition_penalty;
                }
            }
        }

        let mut order: Vec<usize> = (0..vocab).collect();

        // top_k: dejar solo los top_k por logit
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

        // softmax (estable) + top_p nucleus
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
        for p in probs.iter_mut() {
            *p /= sum;
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

        // multinomial
        let u = rand::random::<f32>();
        let mut cum = 0.0;
        for i in 0..vocab {
            cum += probs[i];
            if u < cum {
                return Ok(i as u32);
            }
        }
        // fallback: el de mayor prob
        let mut best = 0usize;
        for i in 1..vocab {
            if probs[i] > probs[best] {
                best = i;
            }
        }
        Ok(best as u32)
    }

    /// Generación autoregresiva con KV cache (mismo flujo que el Python).
    pub fn generate(
        &self,
        input_ids: &Tensor, // (1, prompt_len)
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        repetition_penalty: f32,
        eos_token_id: Option<u32>,
    ) -> Result<Tensor> {
        let prompt_len = input_ids.dim(1)?;
        let mut caches: Vec<Option<KVCache>> = vec![None; self.blocks.len()];
        let mut tokens: Vec<u32> = input_ids.i((0, ..))?.to_vec1()?;
        let mut logits = Tensor::zeros((1, 1, self.config.emb_num), DType::F32, self.device())?;
        for i in 0..prompt_len {
            let t = Tensor::from_vec(vec![tokens[i]], (1, 1), self.device())?;
            logits = self.forward_with_cache(&t, i, &mut caches)?;
        }

        let mut gen: Vec<u32> = Vec::new();
        for gen_i in 0..max_new_tokens {
            let next = self.sample(&logits, &tokens, temperature, top_k, top_p, repetition_penalty)?;
            if eos_token_id == Some(next) {
                break;
            }
            tokens.push(next);
            gen.push(next);

            let t = Tensor::from_vec(vec![next], (1, 1), self.device())?;
            logits = self.forward_with_cache(&t, prompt_len + gen_i, &mut caches)?;
        }

        let all = [tokens[..prompt_len].to_vec(), gen].concat();
        let n = all.len();
        Ok(Tensor::from_vec(all, (1, n), self.device())?)
    }
}
