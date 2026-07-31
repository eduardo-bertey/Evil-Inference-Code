/// MLP SwiGLU — espejo de `laurelia-llm/model.py::MLP`.
///
/// fc1: dim -> 2*ffn_dim, chunk en (x, gate), x * silu(gate), fc2: ffn_dim -> dim

use candle_core::{Result, Tensor};
use candle_nn::{Module, VarBuilder};

#[derive(Clone, Debug)]
pub struct MLP {
    pub fc1: candle_nn::Linear,
    pub fc2: candle_nn::Linear,
    pub ffn_dim: usize,
}

impl MLP {
    pub fn new(vb: VarBuilder, dim: usize, ffn_dim: usize) -> Result<Self> {
        let fc1 = candle_nn::linear_no_bias(dim, 2 * ffn_dim, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear_no_bias(ffn_dim, dim, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2, ffn_dim })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let h = self.fc1.forward(x)?.reshape((b, s, 2, self.ffn_dim))?;
        let x = h.narrow(2, 0, 1)?.squeeze(2)?;
        let gate = h.narrow(2, 1, 1)?.squeeze(2)?;
        let gate = candle_nn::ops::silu(&gate)?;
        let out = (x * gate)?;
        self.fc2.forward(&out)
    }
}
