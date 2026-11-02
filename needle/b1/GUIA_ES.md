# Needle 1 (b1): inferencia y entrenamiento

Esta carpeta contiene el snapshot original de Needle del repositorio oficial (`1807b1d`). Es una implementación JAX/Flax de encoder-decoder, descrita por Cactus como atención pura, sin FFN. El código de entrenamiento está en `needle/training/`.

## Instalación

Usá Python 3.11 o superior y un entorno independiente. Desde esta carpeta:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

JAX depende del sistema operativo y del hardware; en Windows puede convenir WSL/Linux. Para CUDA/TPU revisá `pyproject.toml`, `requirements.txt` y `docs/tpu.md`; no instales el entorno de b1 junto con b2 o b3.

## Inferencia

El comando `run` requiere un checkpoint compatible con esta versión:

```powershell
needle run --checkpoint checkpoints\needle.pkl --query "¿Qué tiempo hace en Buenos Aires?" --tools '[{"name":"get_weather","description":"Consulta el clima de una ciudad","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]'
```

También se puede usar desde Python:

```python
from needle.model.run import load_checkpoint, generate
from needle.model.architecture import EncoderDecoderTransformer
from needle.dataset.dataset import get_tokenizer

params, config = load_checkpoint("checkpoints/needle.pkl")
model = EncoderDecoderTransformer(config)
tokenizer = get_tokenizer()
result = generate(
    model, params, tokenizer,
    query="¿Qué tiempo hace en Buenos Aires?",
    tools='[{"name":"get_weather","parameters":{"location":"string"}}]',
    stream=False,
)
print(result)
```

El checkpoint y el tokenizer deben corresponder a b1. Los pesos no están incluidos en el snapshot; consultá los enlaces oficiales del `README.md` antes de descargar un modelo.

## Entrenamiento

El CLI de este snapshot ofrece dos caminos:

- **Ajuste supervisado para llamadas a herramientas**: `needle train`.
- **Pretraining**: `needle pretrain`; requiere preparar los datos y recursos de cómputo antes de ejecutarlo.

Consultá los argumentos completos con:

```powershell
needle train --help
needle pretrain --help
needle tokenize --help
```

Ejemplo de ajuste con una configuración pequeña; `train` usa los datos preparados por el pipeline del proyecto:

```powershell
needle tokenize
needle train --epochs 1 --batch-size 2 --checkpoint-dir checkpoints
```

Para reanudar, añadí `--checkpoint checkpoints\<archivo>.pkl`. El pretraining oficial publicado fue de gran escala; una corrida local de prueba no reproduce ese entrenamiento ni su calidad. Revisá `docs/`, `needle/dataset/` y `needle/training/` para el formato y preparación de los datos.

## Archivos de referencia

- Inferencia: `needle/model/run.py`
- Modelo: `needle/model/architecture.py`
- Ajuste supervisado: `needle/training/train.py`
- Pretraining: `needle/training/pretrain.py`
- Dataset/tokenización: `needle/dataset/`
