# LLM\_D3: A Sparse 350M Architecture Trained on 50B Tokens

This repository contains the implementation of **LLM\_D3**, a decoder-only Large Language Model trained from scratch on 50 billion tokens of the C4 English-only dataset. It features a modern, high-performance architecture optimized for efficiency, combining **Mixture of Experts (MoE)**, **Multi-head Latent Attention (MLA)**, and **Rotary Positional Embeddings (RoPE)**.

Designed for genuine generalization over rote memorization, the model was trained using a single-epoch pass, achieving a **33% zero-shot HellaSwag** score. Following instruction fine-tuning, it serves as a capable assistant with strong general reasoning and factual recall.

-----

## 📊 Model Statistics

| Metric | Value |
| :--- | :--- |
| **Total Parameters** | 358.74M |
| **Active Parameters** | 171.96M |
| **Sparsity Ratio** | 52.06% |
| **Training Data** | 50B Tokens (C4 English) |
| **Architecture** | MLA + Sparse MoE + RoPE |

-----

## for weights
huggingface: firdavsus/LLM_D3

## 🏗️ Architecture Details

The model utilizes a custom GPT implementation (`LLM_2.py`) with several key architectural innovations focused on compute efficiency and memory optimization.

### Multi-head Latent Attention (MLA)

To solve the memory bottleneck of the KV cache, LLM\_D3 implements **Multi-head Latent Attention**.

  * **Latent Compression**: Query and KV states are compressed into a lower-dimensional latent space before being up-projected for attention calculations.
  * **Throughput**: This reduces the memory footprint of the KV cache during inference while maintaining the performance of standard Multi-Head Attention.

### Sparse Mixture of Experts (MoE)

LLM\_D3 uses a sparse MoE architecture for 19 out of its 24 layers.

  * **Expert Configuration**: Each MoE layer contains **6 experts**, with a **Top-2** routing mechanism active for every token.
  * **Hybrid Stability Sandwich**: For improved training stability, the **first 3 layers** and **last 2 layers** are initialized as standard dense MLP blocks rather than MoE layers.
  * **Routing**: Uses a noisy Top-K router with auxiliary load-balancing and router z-loss to prevent expert collapse and ensure balanced utilization across the 19 MoE blocks.

### Positional Encoding

  * **RoPE**: Rotary Positional Embeddings are applied to ensure better handling of long-range dependencies and superior sequence positioning compared to traditional learned embeddings.

-----

## 📈 Training & Evaluation

### Pre-training Setup

  * **Policy**: Single-epoch pass on 50B tokens (no repetition) to prioritize feature extraction and generalization.
  * **Batch Size**: 1M tokens effective batch size for high gradient stability.
  * **Schedule**: Warmup-Stable-Decay (WSD) / Stepped Cosine Decay with a 1,000-step warmup.
  * **Optimizer**: AdamW with hardware-optimized settings.

### Benchmarks

| Benchmark | Setting | Score |
| :--- | :--- | :--- |
| **HellaSwag** | Zero-shot | **33%** |

### Fine-tuning

Fine-tuned on the `alpaca-cleaned` dataset using an Instruction-Input-Response format.

  * **Strengths**: Strong general reasoning, factual consistency, and instruction adherence.
  * **Known Limitations**: The model currently struggles with complex arithmetic. Additionally, an initialization anomaly in the final 2 layers resulted in a signal spike at the end of the network; while the model remains functional and capable, this is a known area for future refinement.

-----

## 🖼️ Visualizations

### Pre-Training Curves
![Pre-Training](images/training_curves_with_eval.png)

*50k steps on a 50B token corpus with 1M token effective batch size.*

### Diagnostics & Utilization
![Model-analysis](images/full_diagnostics.png)
![Model-analysis](images/weight_histograms.png)
*Visualizing weight distribution and expert utilization. Current routing shows healthy balance with utilization under 33%.*

-----

## 🛠️ Usage

### Inference

Interact with the model using the `test.py` script, which includes Top-K, Top-P, and repetition penalty sampling.

```bash
python test.py
```

### Fine-tuning

To replicate the instruction tuning on your own dataset:

1.  Format your data following the Alpaca template in `fine_tune.py`.
2.  Execute:

<!-- end list -->

```bash
python fine_tune.py
```

-----

## 📂 Repository Structure

  * `LLM_2.py`: Core architecture (MLA, MoE, RoPE).
  * `train.py`: Pre-training logic and WSD scheduler.
  * `fine_tune.py`: Instruction tuning implementation.
  * `manager.py`: MoE auxiliary loss tracking.
  * `check_params.py`: Active vs. total parameter counter.
  * `eval.py`: HellaSwag evaluation suite.
  * `analysis.py` / `show.py`: Diagnostic and visualization tools.

-----

*Note: This model was developed as a research exploration into efficient sparse architectures. Verify all mathematical outputs manually.*

### References

  * [nanoMoE Implementation](https://www.google.com/search?q=https://github.com/avm-avm/nanoMoE)
  * [MLA Implementation Guide](https://medium.com/@atulit23/implementing-multi-head-latent-attention-from-scratch-in-python-1e14d03fbc91)
  * [DeepSeek-V3 Research (MoE/MLA Foundations)](https://arxiv.org/abs/2412.19437)
  * [Lets build LLM](https://medium.com/@bogdan.su/in-this-article-we-will-build-our-llm-which-i-called-lightlm-from-scratch-choose-the-optimal-c1e1839668db)
