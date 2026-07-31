/// RMSNorm — espejo de `torch.nn.RMSNorm`.
///
/// out = x * rsqrt(mean(x^2, dim=-1) + eps) * weight

use candle_core::{DType, Result, Tensor};
use candle_nn::VarBuilder;

#[derive(Clone, Debug)]
pub struct RMSNorm {
    pub weight: Tensor,
    pub eps: f32,
}

impl RMSNorm {
    pub fn new(dim: usize, eps: f32, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.to_dtype(DType::F32)?;
        let last = x.rank() - 1;
        let msq = (x.sqr()?.mean_keepdim(last)? + self.eps)?.rsqrt()?;
        let out = x.mul(&msq)?.mul(&self.weight.to_dtype(DType::F32)?)?;
        out
    }
}
