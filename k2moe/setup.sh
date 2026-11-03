#!/usr/bin/env bash
# k2moe setup: instala las dependencias que falten (Colab/Linux).
# Uso en Colab:  !bash k2moe/setup.sh
# No toca torch/CUDA — Colab ya lo trae (reinstalar podría romper CUDA).
set -euo pipefail

need() {
    python -c "import $1" 2>/dev/null && echo "OK  $1" || {
        echo "... instalando $2"
        pip install -q "$2"
    }
}

need transformers "transformers"
need datasets "datasets"
need huggingface_hub "huggingface_hub"
need matplotlib "matplotlib"
need k_means_constrained "k-means-constrained"

echo "Listo. Corre:  !python k2moe/train.py"
