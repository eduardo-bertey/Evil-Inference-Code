/// Tokenizer BPE — wrapper del crate `tokenizers` (ya en Cargo.toml).
///
/// Igual que `BPEWrapper` de `laurelia-llm/train.py`.

use tokenizers::Tokenizer;

#[derive(Clone)]
pub struct LaureliaTokenizer {
    tok: Tokenizer,
}

impl LaureliaTokenizer {
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let tok = Tokenizer::from_file(path).map_err(|e| format!("tokenizer: {e}"))?;
        Ok(Self { tok })
    }

    pub fn vocab_size(&self) -> usize {
        self.tok.get_vocab_size(true)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let enc = self
            .tok
            .encode(text, true)
            .map_err(|e| format!("encode: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, Box<dyn std::error::Error>> {
        let s = self
            .tok
            .decode(ids, true)
            .map_err(|e| format!("decode: {e}"))?;
        Ok(s)
    }
}
