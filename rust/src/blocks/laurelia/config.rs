/// Laurelia Config — espejo de `laurelia-llm/model.py::Config`.
#[derive(Clone, Debug)]
pub struct Config {
    pub dim: usize,
    pub heads: usize,
    pub kv_groups: usize,
    pub layers: usize,
    pub ffn_dim: usize,
    pub block_size: usize,
    pub emb_num: usize,
    pub rotary_pct: f64,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.dim / self.heads
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            dim: 768,
            heads: 12,
            kv_groups: 4,
            layers: 16,
            ffn_dim: 3072,
            block_size: 1024,
            emb_num: 32000,
            rotary_pct: 0.25,
        }
    }
}
