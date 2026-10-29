/// Config de MoE — espejo de `moe-plus/model.py::TransformerLM`.
#[derive(Clone, Debug)]
pub struct Config {
    pub vocab_size: usize,
    pub d_model: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_groups: usize,
    pub head_dim: usize,
    pub ffn_expansion: f64,
    pub max_seq_len: usize,
    pub rope_base: f64,
    pub rope_scaling: f64,
    pub norm_eps: f64,
    pub use_moe: bool,
    pub n_experts: usize,
    pub top_k: usize,
    pub n_shared: usize,
    pub expert_dim: usize,
    pub capacity_factor: f64,
    pub z_loss_gamma: f64,
    pub bias_decay: f64,
    pub noise_std: f64,
    pub n_dense_start: usize,
    pub n_dense_end: usize,
    pub use_mla: bool,
    pub use_x0: bool,
    pub use_swiglu: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            d_model: 768,
            num_layers: 16,
            num_heads: 12,
            num_kv_groups: 4,
            head_dim: 64,
            ffn_expansion: 4.0,
            max_seq_len: 2048,
            rope_base: 10000.0,
            rope_scaling: 1.0,
            norm_eps: 1e-5,
            use_moe: true,
            n_experts: 8,
            top_k: 2,
            n_shared: 1,
            expert_dim: 0,
            capacity_factor: 1.25,
            z_loss_gamma: 0.001,
            bias_decay: 1e-3,
            noise_std: 0.01,
            n_dense_start: 3,
            n_dense_end: 3,
            use_mla: false,
            use_x0: false,
            use_swiglu: true,
        }
    }
}

impl Config {
    pub fn head_dim_or(&self) -> usize {
        if self.head_dim == 0 {
            self.d_model / self.num_heads
        } else {
            self.head_dim
        }
    }
}
