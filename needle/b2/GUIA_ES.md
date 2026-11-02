# Needle 2 (b2): inferencia y fine-tuning

Esta carpeta es el snapshot oficial `v2.0.1`. Needle 2 está pensado para tool calling y extracción estructurada. Su arquitectura combina atención GQA con MLP Hadamard, memoria Engram y conexiones multi-lane: **no es solo atención**.

## Instalación

Desde esta carpeta, en un entorno independiente:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

El primer uso puede necesitar internet para descargar el motor/recursos desde Hugging Face. Los archivos de pesos no están incluidos en este snapshot.

## Inferencia

La API Python permite declarar herramientas como esquemas JSON y pedir la llamada estructurada:

```python
import needle

tools = [{
    "name": "get_weather",
    "description": "Consulta el clima de una ciudad.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}]

agent = needle.Needle(tools=tools)
response = agent.complete("¿Qué tiempo hace en Buenos Aires?")
print(response)
```

`complete()` devuelve la respuesta estructurada; `run()` además ejecuta las funciones Python que se registraron con `@needle.tool`. Empezá con `agent.complete()` si solo querés inspeccionar la llamada sin ejecutar acciones.

## Fine-tuning

El paquete de b2 ofrece ajuste local LoRA sobre ejemplos JSONL; no es un script de pretraining desde cero. Las dependencias JAX/Flax de entrenamiento ya están declaradas en `pyproject.toml`; prepará ejemplos acordes al formato descrito en `README.md`:

```powershell
needle finetune datos.jsonl --epochs 3 --out checkpoints\needle2-lora.pkl
```

Para exportar un modelo con el adaptador, usá un checkpoint compatible con b2:

```powershell
needle build checkpoints\needle2.pkl --lora checkpoints\needle2-lora.pkl --out modelo.cact
```

Si tus nombres/formato de checkpoint difieren, verificá los argumentos con `needle build --help`.

## Archivos principales

- Inferencia: `needle/__init__.py`, `needle/agent/`
- Arquitectura: `needle/model/architecture.py`
- Fine-tuning/exportación: `needle/model/finetune.py`
- Ejemplos y formato de datos: `README.md`
