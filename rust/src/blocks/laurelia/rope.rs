/// RoPE parcial — espejo de `laurelia-llm/rope.py`.
///
/// Rota solo las primeras `rotary_dim` dimensiones de head_dim
/// (`rotary_dim = int(head_dim * rotary_pct)` redondeado a par).
/// Compatible con inferencia KV cache vía `offset`.

use candle_core::{Device, Result, Tensor};

pub struct RoPE {
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub rotary_half: usize,
    pub max_seq_len: usize,
    pub base: f64,
    pub rotary_pct: f64,
    pub cos_cache: Tensor,
    pub sin_cache: Tensor,
}

impl RoPE {
    pub fn new(head_dim: usize, max_seq_len: usize, base: f64, rotary_pct: f64, device: &Device) -> Result<Self> {
        assert!(head_dim % 2 == 0, "head_dim must be even, got {}", head_dim);

        let rotary_dim = if rotary_pct >= 1.0 {
            head_dim
        } else {
            let rd = (head_dim as f64 * rotary_pct) as usize;
            rd - (rd % 2)
        };
        let rotary_half = rotary_dim / 2;

        let inv_freq: Vec<f32> = (0..rotary_half)
            .map(|k| 1.0 / (base.powf(2.0 * k as f64 / head_dim as f64)) as f32)
            .collect();

        let t: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
        let t = Tensor::new(t.as_slice(), device)?.unsqueeze(1)?; // (S, 1)
        let inv = Tensor::new(inv_freq.as_slice(), device)?.unsqueeze(0)?; // (1, rotary_half)
        let freqs = t.matmul(&inv)?; // (S, rotary_half)

        let cos_cache = freqs.cos()?;
        let sin_cache = freqs.sin()?;

        Ok(Self {
            head_dim,
            rotary_dim,
            rotary_half,
            max_seq_len,
            base,
            rotary_pct,
            cos_cache,
            sin_cache,
        })
    }

    /// Slice cos/sin para [offset, offset+seq_len), shape (1, S, 1, rotary_half).
    fn cache_slice(&self, offset: usize, seq_len: usize, device: &Device) -> Result<(Tensor, Tensor)> {
        let end = offset + seq_len;
        if end > self.max_seq_len {
            let rebuilt = Self::new(self.head_dim, end, self.base, self.rotary_pct, device)?;
            let cos = rebuilt.cos_cache.narrow(0, offset, seq_len)?.unsqueeze(0)?.unsqueeze(2)?;
            let sin = rebuilt.sin_cache.narrow(0, offset, seq_len)?.unsqueeze(0)?.unsqueeze(2)?;
            return Ok((cos, sin));
        }
        let cos = self.cos_cache.narrow(0, offset, seq_len)?.unsqueeze(0)?.unsqueeze(2)?;
        let sin = self.sin_cache.narrow(0, offset, seq_len)?.unsqueeze(0)?.unsqueeze(2)?;
        Ok((cos, sin))
    }

    /// Rota solo las primeras `rotary_dim` dims; el resto pasa sin cambio.
    fn rotate_partial(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        if self.rotary_dim == 0 {
            return Ok(x.clone());
        }
        let rd = self.rotary_dim;
        let hh = self.rotary_half;
        let last = x.dim(3)?;

        let x_rot = x.narrow(3, 0, rd)?;
        let x_pass = x.narrow(3, rd, last - rd)?;

        let first = x_rot.narrow(3, 0, hh)?;
        let second = x_rot.narrow(3, hh, hh)?;

        let rotated_first = ((first.clone().broadcast_mul(cos)?) - (second.clone().broadcast_mul(sin)?))?;
        let rotated_second = ((first.broadcast_mul(sin)?) + (second.broadcast_mul(cos)?))?;
        let rotated = Tensor::cat(&[rotated_first, rotated_second], 3)?;

        Tensor::cat(&[rotated, x_pass], 3)
    }

    /// Aplica RoPE a Q y K.
    ///
    /// q: (B, S, num_heads, head_dim)
    /// k: (B, S, num_kv_groups, head_dim)
    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(1)?;
        let (cos, sin) = self.cache_slice(offset, seq_len, q.device())?;

        let q_out = self.rotate_partial(q, &cos, &sin)?;
        let k_out = self.rotate_partial(k, &cos, &sin)?;

        Ok((q_out, k_out))
    }

    /// Aplica RoPE a un solo tensor.
    pub fn apply_to_single(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let seq_len = x.dim(1)?;
        let (cos, sin) = self.cache_slice(offset, seq_len, x.device())?;
        self.rotate_partial(x, &cos, &sin)
    }
}
