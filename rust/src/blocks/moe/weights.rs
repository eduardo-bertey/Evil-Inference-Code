/// Carga de pesos — espejo de `laurelia/weights.rs`.
use candle_core::{Device, Result, DType};
use candle_nn::VarBuilder;

use super::config::Config;
use super::model::LLM;

pub struct Weights;

impl Weights {
    pub fn load(path: &str, config: &Config, dtype: DType, device: &Device) -> Result<LLM> {
        let tensors = candle_core::safetensors::load(path, device)?;
        let vb = VarBuilder::from_tensors(tensors, dtype, device);
        LLM::new(vb, config.clone())
    }

    pub fn load_pth(path: &str, config: &Config, dtype: DType, device: &Device) -> Result<LLM> {
        use candle_core::pickle;
        let tensors = match pickle::read_all_with_key(path, Some("model")) {
            Ok(t) => t,
            Err(_) => pickle::read_all(path)?,
        };
        let map: std::collections::HashMap<String, candle_core::Tensor> =
            tensors.into_iter().collect();
        let vb = VarBuilder::from_tensors(map, dtype, device);
        LLM::new(vb, config.clone())
    }
}
