/// KV cache — espejo de `moe-plus/cache_kv.py::KVCache`.
use candle_core::{Result, Tensor};

#[derive(Clone, Debug)]
pub struct KVCache {
    pub cached_k: Tensor,
    pub cached_v: Tensor,
}

impl KVCache {
    pub fn new(batch: usize, num_kv_groups: usize, head_dim: usize, device: &candle_core::Device) -> Result<Self> {
        let cached_k = Tensor::zeros((batch, 0, num_kv_groups, head_dim), candle_core::DType::F32, device)?;
        let cached_v = Tensor::zeros((batch, 0, num_kv_groups, head_dim), candle_core::DType::F32, device)?;
        Ok(Self { cached_k, cached_v })
    }

    pub fn seq_len(&self) -> usize {
        self.cached_k.dim(1).unwrap_or(0)
    }

    pub fn append(&self, k_new: Tensor, v_new: Tensor) -> Result<Self> {
        if self.cached_k.dim(1)? == 0 {
            return Ok(Self { cached_k: k_new, cached_v: v_new });
        }
        let k = Tensor::cat(&[&self.cached_k, &k_new], 1)?;
        let v = Tensor::cat(&[&self.cached_v, &v_new], 1)?;
        Ok(Self { cached_k: k, cached_v: v })
    }

    pub fn trim_prefix(&self, remove: usize) -> Result<Self> {
        let seq = self.cached_k.dim(1)?;
        if remove == 0 || remove >= seq {
            return Ok(self.clone());
        }
        let k = self.cached_k.narrow(1, remove, seq - remove)?.contiguous()?;
        let v = self.cached_v.narrow(1, remove, seq - remove)?.contiguous()?;
        Ok(Self { cached_k: k, cached_v: v })
    }

    pub fn keep_last(&self, keep: usize) -> Result<Self> {
        let seq = self.cached_k.dim(1)?;
        if keep == 0 || keep >= seq {
            return Ok(self.clone());
        }
        let start = seq - keep;
        let k = self.cached_k.narrow(1, start, keep)?.contiguous()?;
        let v = self.cached_v.narrow(1, start, keep)?.contiguous()?;
        Ok(Self { cached_k: k, cached_v: v })
    }
}
