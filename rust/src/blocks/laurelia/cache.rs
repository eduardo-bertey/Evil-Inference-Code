/// KV cache para una capa de atención.
///
/// Shapes: cached_k, cached_v: (batch, accumulated_seq_len, num_kv_groups, head_dim)
use candle_core::{Result, Tensor};

#[derive(Clone, Debug)]
pub struct KVCache {
    pub cached_k: Tensor,
    pub cached_v: Tensor,
}

impl KVCache {
    pub fn new(cached_k: Tensor, cached_v: Tensor) -> Self {
        Self { cached_k, cached_v }
    }

    pub fn seq_len(&self) -> usize {
        self.cached_k.dim(1).unwrap_or(0)
    }

    /// Quita los primeros `remove` tokens del cache.
    pub fn trim_prefix(&self, remove: usize) -> Result<KVCache> {
        let seq = self.cached_k.dim(1)?;
        if remove == 0 || remove >= seq {
            return Ok(self.clone());
        }
        let k = self.cached_k.narrow(1, remove, seq - remove)?.contiguous()?;
        let v = self.cached_v.narrow(1, remove, seq - remove)?.contiguous()?;
        Ok(KVCache { cached_k: k, cached_v: v })
    }

    /// Deja solo los últimos `keep` tokens del cache.
    pub fn keep_last(&self, keep: usize) -> Result<KVCache> {
        let seq = self.cached_k.dim(1)?;
        if keep == 0 || keep >= seq {
            return Ok(self.clone());
        }
        let start = seq - keep;
        let k = self.cached_k.narrow(1, start, keep)?.contiguous()?;
        let v = self.cached_v.narrow(1, start, keep)?.contiguous()?;
        Ok(KVCache { cached_k: k, cached_v: v })
    }
}
