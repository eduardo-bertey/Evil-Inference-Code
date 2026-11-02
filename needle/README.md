# Needle dentro de Evil-Inference-Code

Se agregaron snapshots del repositorio oficial `cactus-compute/needle`, con cada versión en su propia carpeta. Los archivos `README.md` y `LICENSE` de upstream se conservaron; las guías `GUIA_ES.md` son notas locales en español.

| Carpeta | Snapshot del código oficial | Qué contiene |
|---|---|---|
| [b1](./b1/) | `1807b1d` (12 mayo 2026) | Needle original: encoder-decoder de atención pura; incluye pretraining y fine-tuning en el código público. |
| [b2](./b2/) | tag `v2.0.1` | Needle 2; inferencia y fine-tuning LoRA local. |
| [b3](./b3/) | `main`, commit `b4fc11b` | Needle 3; inferencia, fine-tuning LoRA y exportación. |
| [audio](./audio/) | `main`, commit `b4fc11b` | Whistle para ASR/audio, fine-tuning y exportación; incluye el cliente Needle actual. |

## Guías

- [Needle 1: inferencia y entrenamiento](./b1/GUIA_ES.md)
- [Needle 2: inferencia y fine-tuning](./b2/GUIA_ES.md)
- [Needle 3: inferencia y fine-tuning](./b3/GUIA_ES.md)
- [Whistle/audio: transcripción, combinación con Needle y fine-tuning](./audio/GUIA_ES.md)

## Importante

- Son snapshots separados para comparar y mantener las versiones; instalá cada una desde su propia carpeta, no mezcles sus entornos Python.
- Los pesos grandes y las bibliotecas de ejecución de Cactus no están copiados en estas carpetas. Algunas funciones los descargan desde Hugging Face al primer uso; otras requieren que se indique un checkpoint local.
- En Needle 2/3, “entrenar” en el paquete público normalmente significa **fine-tuning**, no volver a preentrenar el modelo desde cero. Needle 1 sí conserva scripts de entrenamiento/pretraining, pero replicar el entrenamiento publicado requiere datos y hardware especializado.
- El soporte de audio es reconocimiento de voz con **Whistle** (audio → texto), no generación de voz.

Fuentes: [repositorio oficial](https://github.com/cactus-compute/needle), [Needle 1](https://huggingface.co/Cactus-Compute/needle), [Needle 2](https://huggingface.co/Cactus-Compute/needle2), [Needle 3](https://huggingface.co/Cactus-Compute/needle3) y [Whistle](https://huggingface.co/Cactus-Compute/whistle).
