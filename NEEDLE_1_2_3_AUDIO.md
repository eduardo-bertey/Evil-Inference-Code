# Needle 1, 2 y 3: arquitectura y audio

Resumen de las fuentes oficiales de Cactus Compute consultadas el 31 de octubre de 2026, contrastadas con las implementaciones presentes en este repositorio.

## Resumen ejecutivo

- **Needle 1**: modelo pequeño de 26 millones de parámetros, encoder-decoder y explícitamente descrito como *pure attention (no FFN)*. Es el que más se acerca a “solo atención”.
- **Needle 2**: 45 millones de parámetros, orientado a tool calling y extracción estructurada. No es solo atención: combina GQA, memoria Engram y una MLP Hadamard.
- **Needle 3**: generación más reciente, con una arquitectura escalonada de 2 a 20 capas; también incluye MLP Hadamard, Engram y conexiones multi-lane. Añade embeddings de texto y una cabeza de confianza calibrada.
- **Audio**: lo proporciona **Whistle**, un modelo separado de reconocimiento de voz (audio a texto). Needle 3 y Whistle pueden cargarse juntos en el mismo motor para transcribir una instrucción hablada y devolver llamadas a herramientas en una sola operación. Esto no es síntesis de voz (texto a audio).

## Comparación

| Versión | Tamaño y forma | Arquitectura según las fuentes | Capacidades destacadas |
|---|---|---|---|
| **Needle 1** | 26M parámetros; `d_model=512`; vocabulario de 8.192; 12 capas encoder + 8 decoder. | Encoder-decoder con GQA, RoPE y residuales con gates. La tarjeta del modelo lo describe expresamente como **atención pura, sin FFN**. | Tool calling, con foco en ejecución local y tamaño reducido. |
| **Needle 2** | 45M parámetros; archivo cuantizado de aproximadamente 14 MB; alrededor de 28 MB de RAM de sesión según el proyecto. | *Simple Attention Network* densa: atención GQA, **MLP Hadamard**, memoria Engram basada en n-gramas y conexiones residuales multi-lane. Usa gramática compilada para restringir la salida a esquemas válidos. | Tool calling, uso de dispositivos y extracción estructurada. |
| **Needle 3** | Modelo de hasta 121M parámetros, empaquetado en archivos de 8–29 MB; se puede desplegar a distintas profundidades entre 2 y 20 capas. | *Laddered Simple Attention Network*: **MLP Hadamard tipo Monarch** (no solo atención), GQA con taps convolucionales causales, Engram y conexiones multi-lane. La fuente atribuye la menor carga de cómputo, frente al total de parámetros, a que gran parte reside en Engram. | Tool calling, extracción estructurada, embeddings de texto y confianza calibrada. Salidas restringidas por gramática a partir de los esquemas. |

### ¿“Simple Attention Network” significa solo atención?

No. El nombre describe la familia de modelos, no la ausencia absoluta de componentes adicionales. Needle 1 sí se anuncia como atención pura y sin FFN. En cambio, la documentación oficial de Needle 2 y 3 especifica una **MLP Hadamard**, además de atención. Por tanto, Needle 2/3 no cumplen el requisito de “atención solamente”.

La implementación local confirma esta diferencia:

- `eddle/architecture.py` implementa bloques con opción `no_feedforward`; su configuración de entrenamiento la activa, consistente con la variante de atención pura asociada a Needle 1.
- `needle2-pln/model.py` llama `hadamard_mlp` después de la atención en cada bloque y crea módulos Engram.
- `needle2-pln/pretrain.py` entrena ese modelo local `LLM`. Esta carpeta es una implementación de este repositorio; sus valores de configuración no deben confundirse con las cifras oficiales del modelo publicado.

## Cómo funciona el modo de audio

### Componentes

- **Whistle** es el modelo de speech-to-text/ASR: recibe audio y produce texto. La tarjeta oficial informa un archivo de 16,9 MB y ejecución en CPU, usando el mismo formato de contenedor y motor que Needle.
- Acepta audio mono de 16 kHz; procesa hasta 30 segundos en una pasada y usa ventanas para audios más largos. La documentación publica soporte para inglés, alemán, francés, español, italiano, neerlandés y polaco.
- También documenta transcripción en streaming, timestamps por palabra con inicio/fin/probabilidad, sesgo por palabras clave y embeddings de audio.
- **Needle 3** interpreta la transcripción frente a las herramientas declaradas por la aplicación. Al cargar ambos modelos, el motor puede recibir audio y devolver en un objeto JSON tanto las llamadas a herramientas como los campos de audio/transcripción.

### Flujo

`audio del usuario → Whistle transcribe → Needle 3 interpreta el texto según las herramientas/esquemas → JSON con transcripción y llamadas`

La integración se presenta como una sola llamada al motor y puede favorecer valores enumerados en el esquema durante la transcripción. No implica que Needle 3 sea, por sí mismo, un modelo de audio de extremo a extremo: la transcripción corresponde a Whistle.

### Límites a tener en cuenta

- **No es TTS**: las fuentes revisadas describen reconocimiento/transcripción, no generación de voz.
- El soporte de streaming se refiere a transcripción incremental; la documentación dice que consolida palabras cuando dos pasadas coinciden.
- La documentación advierte que silencio y ruido constante deben dar texto vacío, pero sonidos no verbales —especialmente música— todavía pueden convertirse en texto. Para micrófonos siempre activos recomienda VAD o una wake word.
- Las cifras de tamaño, velocidad y benchmarks aquí citadas son las publicadas por Cactus Compute; no se midieron independientemente para este resumen.

## Conclusión práctica

Si el requisito es **un modelo estrictamente de atención, sin MLP/FFN**, Needle 1 es la versión que la documentación oficial presenta de esa manera. Needle 2 y Needle 3 no son attention-only. Para una interfaz hablada que ejecute herramientas, la combinación documentada es **Whistle + Needle 3**; si lo requerido es que el modelo genere audio, las fuentes consultadas no respaldan esa capacidad.

## Fuentes

Fuentes principales del fabricante:

1. [Tarjeta de Needle 1 en Hugging Face](https://huggingface.co/Cactus-Compute/needle) — arquitectura encoder-decoder y descripción “pure attention (no FFN)”.
2. [Tarjeta de Needle 2 en Hugging Face](https://huggingface.co/Cactus-Compute/needle2) — tamaño, uso y componentes de Simple Attention Network.
3. [Tarjeta de Needle 3 en Hugging Face](https://huggingface.co/Cactus-Compute/needle3) — capacidades, arquitectura escalonada, confianza y formatos de despliegue.
4. [Tarjeta de Whistle en Hugging Face](https://huggingface.co/Cactus-Compute/whistle) — ASR, formatos de audio, idiomas, streaming, timestamps e integración con Needle.
5. [Repositorio oficial cactus-compute/needle](https://github.com/cactus-compute/needle) — implementación, CLI, interfaces y documentación actual del motor.

Implementaciones locales contrastadas:

- [eddle/architecture.py](./eddle/architecture.py)
- [eddle/train.py](./eddle/train.py)
- [needle2-pln/model.py](./needle2-pln/model.py)
- [needle2-pln/pretrain.py](./needle2-pln/pretrain.py)
