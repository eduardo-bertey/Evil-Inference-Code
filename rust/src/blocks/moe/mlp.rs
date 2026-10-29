/// MLP (SwiGLU) — espejo de `moe-plus` FFN.
use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

pub struct Mlp {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

impl Mlp {
    pub fn new(vb: VarBuilder, dim: usize, hidden: usize, bias: bool) -> Result<Self> {
        let (g, u, d) = if bias {
            (candle_nn::linear(dim, hidden, vb.pp("gate_proj"))?,
             candle_nn::linear(dim, hidden, vb.pp("up_proj"))?,
             candle_nn::linear(hidden, dim, vb.pp("down_proj"))?)
        } else {
            (candle_nn::linear_no_bias(dim, hidden, vb.pp("gate_proj"))?,
             candle_nn::linear_no_bias(dim, hidden, vb.pp("up_proj"))?,
             candle_nn::linear_no_bias(hidden, dim, vb.pp("down_proj"))?)
        };
        Ok(Self { gate_proj: g, up_proj: u, down_proj: d })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        // silu(gate) * up
        let gate = candle_nn::ops::silu(&gate)?;
        let hidden = (gate * up)?;
        self.down_proj.forward(&hidden)
    }
}
