/// GQA Attention — espejo de `moe-plus/attention.py` (GQA; MLA queda como TODO).
use candle_core::{Device, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use super::cache::KVCache;
use super::rope::RoPE;

pub fn repeat_kv(x: &Tensor, num_heads: usize, num_kv_groups: usize) -> Result<Tensor> {
    if num_kv_groups == num_heads {
        return Ok(x.clone());
    }
    let repeats = num_heads / num_kv_groups;
    let (b, s, g, d) = x.dims4()?;
    let x = x.unsqueeze(3)?.expand((b, s, g, repeats, d))?;
    x.reshape((b, s, g * repeats, d))
}

pub struct Attention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub rope: RoPE,
    pub num_heads: usize,
    pub num_kv_groups: usize,
    pub head_dim: usize,
    pub causal: bool,
    pub bias: bool,
}

impl Attention {
    pub fn new(
        vb: VarBuilder,
        dim: usize,
        heads: usize,
        kv_groups: usize,
        head_dim: usize,
        block_size: usize,
        rope_base: f64,
        rope_scaling: f64,
        causal: bool,
        bias: bool,
    ) -> Result<Self> {
        let q_proj = if bias {
            candle_nn::linear(dim, heads * head_dim, vb.pp("q_proj"))?
        } else {
            candle_nn::linear_no_bias(dim, heads * head_dim, vb.pp("q_proj"))?
        };
        let k_proj = if bias {
            candle_nn::linear(dim, kv_groups * head_dim, vb.pp("k_proj"))?
        } else {
            candle_nn::linear_no_bias(dim, kv_groups * head_dim, vb.pp("k_proj"))?
        };
        let v_proj = if bias {
            candle_nn::linear(dim, kv_groups * head_dim, vb.pp("v_proj"))?
        } else {
            candle_nn::linear_no_bias(dim, kv_groups * head_dim, vb.pp("v_proj"))?
        };
        let o_proj = if bias {
            candle_nn::linear(heads * head_dim, dim, vb.pp("o_proj"))?
        } else {
            candle_nn::linear_no_bias(heads * head_dim, dim, vb.pp("o_proj"))?
        };

        let rope = RoPE::new(head_dim, block_size, rope_base * rope_scaling, &vb.device())?;

        Ok(Self {
            q_proj, k_proj, v_proj, o_proj, rope,
            num_heads: heads, num_kv_groups: kv_groups, head_dim, causal, bias,
        })
    }

    fn sdpa(&self, q: &Tensor, k: &Tensor, v: &Tensor, is_causal: bool) -> Result<Tensor> {
        let scale = (self.head_dim as f64).sqrt() as f32;
        let scale_t = Tensor::new(scale, q.device())?.reshape((1,))?;
        let mut scores = q.matmul(&k.transpose(2, 3)?)?.broadcast_div(&scale_t)?;
        if is_causal {
            let (_, _, q_len, kv_len) = scores.dims4()?;
            let device = scores.device();
            let mut data = vec![0.0f32; q_len * kv_len];
            let diag = kv_len as i64 - q_len as i64;
            for i in 0..q_len {
                for j in 0..kv_len {
                    if (j as i64) > (i as i64 + diag) {
                        data[i * kv_len + j] = f32::NEG_INFINITY;
                    }
                }
            }
            let mask = Tensor::from_vec(data, (q_len, kv_len), device)?;
            scores = scores.broadcast_add(&mask)?;
        }
        let attn_w = candle_nn::ops::softmax(&scores, 3)?;
        attn_w.matmul(v)
    }

    pub fn forward(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let q = self.q_proj.forward(x)?.reshape((b, s, self.num_heads, self.head_dim))?;
        let k = self.k_proj.forward(x)?.reshape((b, s, self.num_kv_groups, self.head_dim))?;
        let v = self.v_proj.forward(x)?.reshape((b, s, self.num_kv_groups, self.head_dim))?;

        let (q, k) = self.rope.forward(&q, &k, offset)?;

        let k = repeat_kv(&k, self.num_heads, self.num_kv_groups)?;
        let v = repeat_kv(&v, self.num_heads, self.num_kv_groups)?;

        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        let out = self.sdpa(&q, &k, &v, self.causal && s > 1)?;
        let out = out.transpose(1, 2)?.reshape((b, s, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&out)
    }

    pub fn forward_with_cache(&self, x: &Tensor, offset: usize, cache: Option<&KVCache>) -> Result<(Tensor, KVCache)> {
        let (b, s_new, _) = x.dims3()?;
        let q_new = self.q_proj.forward(x)?.reshape((b, s_new, self.num_heads, self.head_dim))?;
        let k_new = self.k_proj.forward(x)?.reshape((b, s_new, self.num_kv_groups, self.head_dim))?;
        let v_new = self.v_proj.forward(x)?.reshape((b, s_new, self.num_kv_groups, self.head_dim))?;

        let (q_new, k_new) = self.rope.forward(&q_new, &k_new, offset)?;

        let (k_full, v_full) = match cache {
            Some(c) => (Tensor::cat(&[c.cached_k.clone(), k_new.clone()], 1)?,
                        Tensor::cat(&[c.cached_v.clone(), v_new.clone()], 1)?),
            None => (k_new.clone(), v_new.clone()),
        };
        let new_cache = KVCache::new(b, self.num_kv_groups, self.head_dim, x.device())?;
        // rebuild cache with full tensors
        let new_cache = KVCache { cached_k: k_full.clone(), cached_v: v_full.clone() };

        let k_exp = repeat_kv(&k_full, self.num_heads, self.num_kv_groups)?;
        let v_exp = repeat_kv(&v_full, self.num_heads, self.num_kv_groups)?;

        let q = q_new.transpose(1, 2)?;
        let k = k_exp.transpose(1, 2)?;
        let v = v_exp.transpose(1, 2)?;

        let out = self.sdpa(&q, &k, &v, self.causal && cache.is_none() && s_new > 1)?;
        let out = out.transpose(1, 2)?.reshape((b, s_new, self.num_heads * self.head_dim))?;
        let out = self.o_proj.forward(&out)?;
        Ok((out, new_cache))
    }
}
