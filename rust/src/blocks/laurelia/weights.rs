/// Carga de pesos safetensors → LLM de Candle.
///
/// Los tensores PyTorch están en formato (out, in); `candle_nn::linear`
/// y `Embedding::new` los consumen directamente con VarBuilder.

use candle_core::{Device, Result};
use candle_nn::VarBuilder;

use super::config::Config;
use super::model::LLM;

pub struct Weights;

impl Weights {
    /// Carga un safetensors local y construye el LLM.
    ///
    /// `dtype`: F32 o BF16 para los pesos.
    pub fn load(path: &str, config: &Config, dtype: candle_core::DType, device: &Device) -> Result<LLM> {
        let tensors = candle_core::safetensors::load(path, device)?;
        let vb = VarBuilder::from_safetensors(&tensors, dtype, device);
        LLM::new(vb, config)
    }

    /// Carga y devuelve los tensores crudos (para debug).
    pub fn load_tensors(
        path: &str,
        device: &Device,
    ) -> Result<std::collections::HashMap<String, candle_core::Tensor>> {
        candle_core::safetensors::load(path, device)
    }
}
