<div align="center">

# Evil Inference Code

Distributed inference engine for Transformers, minGRU, minLSTM, and xLSTM — with BitNet (1.58-bit) support.

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
![Rust](https://img.shields.io/badge/Language-Rust-orange)

</div>

## About

Evil Inference Code is a high-performance distributed inference system implementing multiple neural network architectures in both Python and Rust:

- **Transformers** — standard attention-based models
- **minGRU** — minimal gated recurrent units
- **minLSTM** — minimal long short-term memory networks
- **xLSTM** — extended LSTM with exponential gating and matrix memory
- **BitNet (1.58-bit)** — ternary quantization for efficient inference

The system is designed for **distributed inference** across multiple devices/nodes, enabling large-scale model deployment with low latency.

## Project Structure

```
evil-inference-code/
├── src/           # Rust implementation (core inference engine)
├── python/        # Python bindings and model definitions
├── models/        # Model configurations and weights
└── examples/      # Usage examples
```

## Requirements

- Rust (latest stable)
- Python 3.10+
- PyTorch >= 2.0

## Installation

### Rust

```bash
cargo build --release
```

### Python

```bash
pip install -r requirements.txt
```

## Quick Start

```rust
// Rust distributed inference example
use evil_inference::distributed::InferenceCluster;

let cluster = InferenceCluster::new(vec!["node1:8080", "node2:8080"]);
let model = cluster.load_model("path/to/model");
let output = model.generate(&input_tokens);
```

```python
# Python inference example
from evil_inference import DistributedInference

engine = DistributedInference(nodes=["node1:8080", "node2:8080"])
model = engine.load_model("path/to/model")
output = model.generate(input_tokens)
```

## Supported Architectures

| Architecture | Quantization | Distributed |
|-------------|--------------|-------------|
| Transformer | FP16, INT8, BitNet | ✓ |
| minGRU | FP16, INT8 | ✓ |
| minLSTM | FP16, INT8 | ✓ |
| xLSTM | FP16, INT8, BitNet | ✓ |

## References

- **xLSTM:** [Extended Long Short-Term Memory](https://arxiv.org/abs/2405.04517)
- **minGRU/minLSTM:** [Were RNNs All We Needed?](https://arxiv.org/abs/2405.21060)
- **BitNet:** [The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764)

## License

Apache-2.0
