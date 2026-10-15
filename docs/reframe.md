# Reencuadre (MOLI-50)

El worker sigue al sujeto activo y recorta a 9:16, 1:1 o 16:9. Si no llega `reframe`, o llega `mode=center` con `layout=single` (o `auto`), el filtro es el de siempre:

```
crop=ih*9/16:ih,scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2
```

16:9 usa `crop=iw:iw*9/16` y el scale/pad de 1920x1080. 1:1 no existía: el centro nuevo es un cuadrado `min(iw,ih)` a 1080x1080.

## Cómo se activa

El VPS manda `reframe` dentro del render spec (versión 3, sin subir versión). El defecto es centro. Para seguir al hablante, en `source_metadata.rules_overrides`:

```json
{"reframe": {"mode": "auto", "layout": "auto", "sample_fps": 8}}
```

`mode`: `center` | `face` | `object` | `auto`.
`layout`: `single` | `split` | `screenshare` | `fit` | `auto`.
`smoothing`: número (solo EMA) o `{ema, deadzone_frac, max_shift_frac}`.
Defecto: EMA 0.45, zona muerta 3 % del lado corto del crop, y como máximo un 5 % de ese lado por frame de salida. El frame 0 puede saltar al sujeto; a partir de ahí el paso está acotado. `sample_fps` se recorta a 5–10 (defecto 8).

`auto` aguanta un layout `hold_s` segundos (defecto 0.75, tope 3) antes de cambiar. Split apila en 9:16 y 1:1, y pone lado a lado en 16:9. Screenshare deja la pantalla grande (arriba, o a la izquierda en 16:9) y la cara en el otro hueco. Fit escala el plano sobre una copia desenfocada (`boxblur=20:2`).

El hablante sale de la diarización. El VPS pone `diarization: true` solo si la campaña tiene `rules_overrides.reframe.mode` distinto de `center`. Los turnos van en `reframe.speakers` (tiempos del clip). Si no hay turnos, se usa la cara más grande o el objeto de más peso. El emparejado hablante↔cara usa el movimiento de la boca; si no hay señal, de izquierda a derecha. Si `plan_reframe` lanza, el render sigue con el crop centrado y `applied.reframe.fallback` queda en `error`.

Si MediaPipe o YOLO no están, o el detector falla, y el layout es `single` o `auto`, el render vuelve al crop centrado. No tumba el job.

## Métrica

`applied.reframe.speaker_in_frame` es la fracción de muestras (5–10 fps) en las que el centro de la cara activa cae dentro de su ventana de crop. Las muestras sin sujeto no cuentan. `max_shift_px` es el tope; `max_shift_observed_px` es el paso máximo medido entre frames de salida.

En el test de entrevista (1920x1080, dos caras, un cambio de hablante a los 5 s, clip de 20 s, tope 5 %) la fracción queda por encima de 0.90 y ningún paso supera el tope. Un tope minúsculo deja la métrica por debajo de 0.90 a propósito: no está hardcodeada.

## Detectores y GPU

No van en `requirements.txt`. Están comentados para no pisar torch:

- `mediapipe` (Apache-2.0): Face Detector, modelo `blaze_face_full_range.tflite`. En Windows el wheel corre en CPU.
- `ultralytics` (AGPL-3.0, uso interno, no se vende el código): YOLO11n + ByteTrack. Usa CUDA (`device=0`) cuando `torch.cuda` está disponible. Una persona de YOLO cuenta como objeto, no pisa una cara de MediaPipe.

El encode de salida sigue en libx264 (medium, crf 23). No se ha pasado a nvenc: un encoder distinto cambiaría el resultado de `mode=center`.

Medido en esta máquina el 2026-10-13 (RTX 5070 Ti, torch 2.8.0+cu128, CUDA disponible, MediaPipe y Ultralytics no instalados). Mismo asset sintético 960x540, ventana 0–12 s, salida 1080x1920:

| paso | tiempo |
| --- | --- |
| plan `mode=center` | 0.000 s |
| encode centro (`render_s`) | 1.264 s |
| plan seguimiento (`plan_s`, incluye `detect_s` 0.001 s) | 0.009 s |
| encode seguimiento (`render_s`) | 1.376 s |

El seguimiento de ese clip trae `speaker_in_frame` 0.927 y paso máximo 15.0 px sobre un tope de 15.1 px. A 1 s el cuadro es el bloque rojo del hablante A; a 5 s, el verde de B. El centro, en los dos instantes, es el gris del hueco (RGB ~126). Vídeos: `docs/reframe/before.mp4` y `docs/reframe/after.mp4`. Se regeneran con `python scripts/reframe_demo.py`.

`applied.reframe.gpu` del camino inyectado no abre torch. El nombre de arriba sale de `torch.cuda.get_device_name(0)` en el mismo script.
