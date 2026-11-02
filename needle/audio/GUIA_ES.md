# Audio: Whistle para ASR, inferencia y fine-tuning

Esta carpeta conserva el snapshot oficial de Needle que incluye el modelo **Whistle**. Whistle hace reconocimiento de voz: transforma audio en texto. No es text-to-speech (no genera audio/voz).

## Requisitos y formato de entrada

- Audio mono de 16 kHz; WAV es la opción más sencilla.
- La documentación oficial informa inglés, alemán, francés, español, italiano, neerlandés y polaco.
- Hasta 30 segundos se procesa en una pasada; clips más largos se procesan por ventanas.
- El modelo/engine de audio no se incluye aquí como pesos descargados: la librería los obtiene al primer uso o se pueden cargar/exportar explícitamente.

Instalación para inferencia:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Para captura de micrófono o entrenamiento:

```powershell
python -m pip install -e ".[train,mic]"
```

## Inferencia de audio a texto

```python
import needle

result = needle.transcribe(
    "grabacion.wav",
    language="es",
    word_timestamps=True,
    keywords=["Laurelia", "Buenos Aires"],
)
print(result["text"])
print(result)
```

Se puede usar `needle.Whistle()` para mantener un objeto de modelo y transcribir varias veces. Para streaming, consultá `needle.stream(...)` y los ejemplos en `needle/agent/whistle.py`. El CLI de micrófono es:

```powershell
needle whistle playground
```

Silencio y ruido constante deberían producir una transcripción vacía, pero sonidos no verbales —en especial música— pueden transcribirse como texto. Para micrófonos siempre activos, agregá VAD o una wake word.

## Audio + llamadas de herramientas

Whistle transcribe; Needle 3 decide qué herramienta corresponde. El motor puede cargar ambos modelos y devolver transcripción más llamadas en el mismo resultado:

```powershell
needle --model needle3.cact --model whistle.cact --tools tools.json --audio grabacion.wav
```

La interfaz Python también tiene `needle.complete(...)` con audio; verificá los parámetros exactos para la plataforma elegida con `needle --help` y `needle whistle --help`. La transcripción queda marcada como campos `audio_` en el resultado combinado.

## Fine-tuning de Whistle

El ajuste local es LoRA sobre las proyecciones de atención; no entrena el modelo entero desde cero. Prepará **tres archivos JSONL** (train, validation y test), con una línea por clip:

```json
{"audio":"audio/clip.wav","text":"apaga las luces de la cocina","language":"es"}
```

Los paths de audio deben existir al ejecutar el comando. Luego:

```powershell
needle whistle finetune train.jsonl validation.jsonl test.jsonl --out adapter.safetensors
needle whistle build --lora adapter.safetensors --out whistle-ajustado.cact
```

Para añadir nombres, términos o caracteres poco comunes, revisá `--tokens` en `needle whistle finetune --help`; el README upstream describe el formato (un elemento por línea). La evaluación del test se realiza durante el fine-tuning; no reutilices test como entrenamiento.

## Código y documentación fuente

- Inferencia/modelo: `needle/agent/whistle.py`, `needle/model/whistle.py`
- Fine-tuning: `needle/model/whistle_finetune.py`
- CLI: `needle/cli.py`
- Referencia oficial y benchmarks: `README.md`

Fuentes oficiales: [Whistle](https://huggingface.co/Cactus-Compute/whistle) y [Needle](https://github.com/cactus-compute/needle).
