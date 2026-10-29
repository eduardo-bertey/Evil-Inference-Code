/// RoPE — espejo de `moe-plus/rope.py` (rotacion completa, scaling).
/// Reusa la implementacion de laurelia con rotary_pct=1.0.
use candle_core::{Device, Result, Tensor};

pub struct RoPE {
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub base: f64,
    pub cos_cache: Tensor,
    pub sin_cache: Tensor,
}

impl RoPE {
    pub fn new(head_dim: usize, max_seq_len: usize, base: f64, device: &Device) -> Result<Self> {
        assert!(head_dim % 2 == 0, "head_dim must be even, got {}", head_dim);
        let half = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|k| 1.0 / (base.powf(2.0 * k as f64 / head_dim as f64)) as f32)
            .collect();
        let t: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
        let t = Tensor::new(t.as_slice(), device)?.unsqueeze(1)?;
        let inv = Tensor::new(inv_freq.as_slice(), device)?.unsqueeze(0)?;
        let freqs = t.matmul(&inv)?;
        let cos_cache = freqs.cos()?;
        let sin_cache = freqs.sin()?;
        Ok(Self { head_dim, max_seq_len, base, cos_cache, sin_cache })
    }

    pub fn forward(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(1)?;
        let half = self.head_dim / 2;
        let (cos, sin) = if offset + seq_len <= self.max_seq_len {
            let c = self.cos_cache.narrow(0, offset, seq_len)?.unsqueeze(0)?.unsqueeze(2)?;
            let s = self.sin_cache.narrow(0, offset, seq_len)?.unsqueeze(0)?.unsqueeze(2)?;
            (c, s)
        } else {
            let positions: Vec<f32> = (offset..offset + seq_len).map(|i| i as f32).collect();
            let t = Tensor::new(positions.as_slice(), q.device())?.unsqueeze(1)?;
            let inv: Vec<f32> = (0..half).map(|k| 1.0 / (self.base.powf(2.0 * k as f64 / self.head_dim as f64)) as f32).collect();
            let inv = Tensor::new(inv.as_slice(), q.device())?.unsqueeze(0)?;
            let freqs = t.matmul(&inv)?;
            (freqs.cos()?.unsqueeze(0)?.unsqueeze(2)?, freqs.sin()?.unsqueeze(0)?.unsqueeze(2)?)
        };

        let q_out = apply_rot(q, &cos, &sin, self.head_dim)?;
        let k_out = apply_rot(k, &cos, &sin, self.head_dim)?;
        Ok((q_out, k_out))
    }
}

fn apply_rot(x: &Tensor, cos: &Tensor, sin: &Tensor, head_dim: usize) -> Result<Tensor> {
    let half = head_dim / 2;
    let x_first = x.narrow(3, 0, half)?;
    let x_second = x.narrow(3, half, half)?;
    let out_first = (x_first.broadcast_mul(cos)? - x_second.broadcast_mul(sin)?)?;
    let out_second = (x_first.broadcast_mul(sin)? + x_second.broadcast_mul(cos)?)?;
    Tensor::cat(&[out_first, out_second], 3)
}
