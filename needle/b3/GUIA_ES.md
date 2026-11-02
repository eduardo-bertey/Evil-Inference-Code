# Needle 3 (b3): inferencia y fine-tuning

Esta carpeta contiene el snapshot oficial `main` (`b4fc11b`). Needle 3 sirve para llamadas a herramientas, extracción estructurada y embeddings. No es attention-only: usa una MLP Hadamard tipo Monarch, atención GQA, Engram y conexiones multi-lane. La arquitectura es escalonada: se pueden exportar profundidades desde 2 capas hasta el modelo completo.

## Instalación

Desde esta carpeta, en un entorno separado:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

El motor y los pesos se descargan cuando se necesitan. Para compilar o entrenar localmente instalá las dependencias opcionales:

```powershell
python -m pip install -e ".[train]"
```

## Inferencia

Definí herramientas como funciones anotadas. `complete()` devuelve la respuesta estructurada; `run()` ejecuta las funciones registradas y continúa el ciclo de herramientas:

```python
import needle

@needle.tool
def get_weather(city: str):
    """Consulta el clima de una ciudad."""
    return {"city": city, "temperature_c": 22}

agent = needle.Needle(tools=[get_weather])
print(agent.complete("¿Qué tiempo hace en Buenos Aires?"))
# Para permitir que el agente ejecute get_weather:
# print(agent.run("¿Qué tiempo hace en Buenos Aires?"))
```

Las herramientas tienen efectos reales si se llama a `run()`: revisá sus argumentos y permisos antes de conectarlas a servicios o dispositivos.

## Fine-tuning

El entrenamiento local publicado para Needle 3 es LoRA con el modelo base congelado, no pretraining desde cero. Prepará un JSONL con ejemplos `query`/`answers` o el formato de conversación que documenta el README. Después:

```powershell
needle finetune datos.jsonl --epochs 10 --out adapter.safetensors
needle build --lora adapter.safetensors --layers 8 --out needle3-ajustado.cact
```

`--layers 8` exporta un rung de ocho capas; quitá el parámetro para usar la profundidad completa, si el build de este snapshot lo permite. El entrenamiento en la plataforma Cactus es otra opción y requiere credenciales/API; no las pongas en archivos versionados.

## Archivos principales

- Inferencia: `needle/__init__.py`, `needle/model/run.py`
- Arquitectura: `needle/model/architecture.py`
- Fine-tuning: `needle/model/finetune.py`
- CLI: `needle/cli.py`
- Referencia de formatos/comandos: `README.md`
