/// GQA Attention — espejo de `laurelia-llm/model.py::Attention`.
///
/// q_proj: (dim -> heads*head_dim)
/// k_proj: (dim -> kv_groups*head_dim)
/// v_proj: (dim -> kv_groups*head_dim)
/// o_proj: (heads*head_dim -> dim)
/// RoPE parcial en Q y K. Escala 1/sqrt(head_dim).

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
    // (B, S, G, 1, D) -> (B, S, G, repeats, D) -> (B, S, G*repeats, D)
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
}

impl Attention {
    pub fn new(
        vb: VarBuilder,
        dim: usize,
        heads: usize,
        kv_groups: usize,
        head_dim: usize,
        block_size: usize,
        rotary_pct: f64,
        causal: bool,
    ) -> Result<Self> {
        let q_proj = candle_nn::linear_no_bias(dim, heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear_no_bias(dim, kv_groups * head_dim, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear_no_bias(dim, kv_groups * head_dim, vb.pp("v_proj"))?;
        let o_proj = candle_nn::linear_no_bias(heads * head_dim, dim, vb.pp("o_proj"))?;

        let rope = RoPE::new(head_dim, block_size, 10000.0, rotary_pct, &vb.device())?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rope,
            num_heads: heads,
            num_kv_groups: kv_groups,
            head_dim,
            causal,
        })
    }

    /// SDPA manual: scores = q·k^T / sqrt(head_dim), causal si cache es None.
    fn sdpa(
        &self,
        q: &Tensor, // (B, heads, S_new, head_dim)
        k: &Tensor, // (B, heads, S_full, head_dim)
        v: &Tensor, // (B, heads, S_full, head_dim)
        is_causal: bool,
    ) -> Result<Tensor> {
        let scale = (self.head_dim as f64).sqrt() as f32;
        let scale_t = Tensor::new(scale, q.device())?.reshape((1,))?;
        let mut scores = q.matmul(&k.transpose(2, 3)?)?.broadcast_div(&scale_t)?;

        if is_causal {
            let (_, _, q_len, kv_len) = scores.dims4()?;
            let device = scores.device();
            let mask = self.causal_mask(q_len, kv_len, device)?;
            scores = scores.broadcast_add(&mask)?;
        }

        let attn_w = candle_nn::ops::softmax(&scores, 3)?;
        let out = attn_w.matmul(v)?; // (B, heads, S_new, head_dim)
        Ok(out)
    }

    fn causal_mask(&self, q_len: usize, kv_len: usize, device: &Device) -> Result<Tensor> {
        // fila i ve columnas j <= i (diagonal 1)
        let mut data = vec![0.0f32; q_len * kv_len];
        let diag = kv_len as i64 - q_len as i64;
        for i in 0..q_len {
            for j in 0..kv_len {
                let jj = j as i64;
                if jj > (i as i64 + diag) {
                    data[i * kv_len + j] = f32::NEG_INFINITY;
                }
            }
        }
        Tensor::from_vec(data, (q_len, kv_len), device)
    }

    /// Attention sin cache (prefill completo).
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

    /// Attention con KV cache.
    ///
    /// x: (B, S_new, dim). offset: posición inicial.
    /// is_causal solo cuando cache es None (prefill), igual que el Python.
    pub fn forward_with_cache(
        &self,
        x: &Tensor,
        offset: usize,
        cache: Option<&KVCache>,
    ) -> Result<(Tensor, KVCache)> {
        let (b, s_new, _) = x.dims3()?;

        let q_new = self.q_proj.forward(x)?.reshape((b, s_new, self.num_heads, self.head_dim))?;
        let k_new = self.k_proj.forward(x)?.reshape((b, s_new, self.num_kv_groups, self.head_dim))?;
        let v_new = self.v_proj.forward(x)?.reshape((b, s_new, self.num_kv_groups, self.head_dim))?;

        let (q_new, k_new) = self.rope.forward(&q_new, &k_new, offset)?;

        let (k_full, v_full) = match cache {
            Some(c) => (
                Tensor::cat(&[c.cached_k.clone(), k_new.clone()], 1)?,
                Tensor::cat(&[c.cached_v.clone(), v_new.clone()], 1)?,
            ),
            None => (k_new.clone(), v_new.clone()),
        };

        let new_cache = KVCache::new(k_full.clone(), v_full.clone());

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
