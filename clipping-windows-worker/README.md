# clipping-windows-worker

Windows GPU Worker para el sistema automático de clipping.

Un servicio que corre permanentemente en un PC Windows con GPU NVIDIA,
recibe trabajos desde un servidor central (VPS) y los ejecuta localmente
usando **WhisperX** (transcripción con GPU) y **FFmpeg/FFprobe** (vídeo).

> **Principio de arquitectura:** el servidor central decide *qué* hacer;
> el worker decide *cómo* ejecutarlo técnicamente.

---

## Arquitectura

```
                    VPS
                     │
              JOB API / QUEUE
                     │
                TAILSCALE
                     │
             WINDOWS WORKER
                     │
       ┌─────────────┼─────────────┐
       │             │             │
       ▼             ▼             ▼
    WhisperX       FFmpeg       FFprobe
       │             │             │
       └─────────────┼─────────────┘
                     │
                     ▼
                  RESULT
                     │
                     ▼
                    VPS
```

Flujo interno de un job:

```
API → Worker → Job Runner → Job Handler → FFmpeg / WhisperX → Result
```

El worker **no** contiene lógica de negocio de campañas. Es un ejecutor
independiente que consulta trabajos, los procesa y devuelve resultados.

---

## Requisitos

- **Python 3.11 o superior** (el código usa sintaxis moderna de tipado).
- **FFmpeg / FFprobe** en el `PATH` (o configura `FFMPEG_PATH`/`FFPROBE_PATH`).
- **NVIDIA GPU + CUDA** (para WhisperX).
- **WhisperX** instalado con soporte CUDA.

### Instalación

```bash
# 1. Crear y activar un entorno virtual
python -m venv .venv
.venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Instalar WhisperX con soporte CUDA
pip install whisperx

# 4. Configurar
copy .env.example .env
# edita .env con tus valores reales
```

---

## Configuración (`.env`)

| Variable | Descripción | Default |
|---|---|---|
| `WORKER_ID` | Identidad del worker | `windows-worker-01` |
| `API_BASE_URL` | URL de la API central | `https://internal-api.example.com` |
| `API_TOKEN` | Token de autenticación | `CHANGE_ME` |
| `POLL_INTERVAL` | Segundos entre consultas de jobs | `5` |
| `HEARTBEAT_INTERVAL` | Segundos entre heartbeats | `30` |
| `MAX_CONCURRENT_JOBS` | Jobs simultáneos (MVP: 1) | `1` |
| `WORKING_DIRECTORY` | Directorio de datos | `C:\ClippingWorker\data` |
| `DEVICE` | `cuda` o `cpu` | `cuda` |
| `COMPUTE_TYPE` | Precisión de WhisperX | `float16` |
| `ALLOW_CPU_FALLBACK` | ¿Usar CPU si no hay CUDA? | `false` |
| `AUTO_CLEANUP` | Limpieza automática | `true` |
| `JOB_RETENTION_HOURS` | Retención de jobs completados | `24` |
| `LOG_LEVEL` | Nivel de log | `INFO` |
| `RENDER_OUTPUT_FPS` | FPS de salida del render | `30.0` |
| `CLIP_STORAGE_ROOT` | Raíz del storage de clips por campaña | `C:\CODIANT\clipping\storage\clips` |
| `FFMPEG_PATH` / `FFPROBE_PATH` | Rutas a los binarios (vacío = PATH) | — |
| `GOG_PATH` / `GOG_ACCOUNT` / `GOG_KEYRING_PASSWORD` | gog CLI para descargas de Drive | — |
| `PUBLISH_STATE_DIR` | Registro local de publicaciones reales (idempotencia en reintentos) | `<WORKING_DIRECTORY>\publish_state` |
| `YOUTUBE_CLIENT_ID` / `YOUTUBE_CLIENT_SECRET` / `YOUTUBE_REFRESH_TOKEN` | OAuth de YouTube | — |
| `YOUTUBE_PRIVACY` | `private` / `unlisted` / `public` | `private` |
| `INSTAGRAM_ACCESS_TOKEN` / `INSTAGRAM_IG_USER_ID` | Graph API (Reels) | — |
| `TIKTOK_ACCESS_TOKEN` / `TIKTOK_PRIVACY` | Content Posting API | — / `SELF_ONLY` |

> **Seguridad:** nunca subas `.env` a Git. Usa `.env.example` como plantilla.

---

## Ejecución

```bash
python run.py
```

Salida esperada al arrancar:

```
INFO Worker starting
INFO Worker ID: windows-worker-01
INFO OS: Windows 11
INFO GPU detected: NVIDIA RTX 5070 Ti
INFO CUDA available: true
INFO FFmpeg available
INFO WhisperX available
INFO Worker registered
INFO Waiting for jobs...
```

---

## Modo desarrollo (Mock API)

Para probar el worker sin un backend real, usa `API_BASE_URL=mock`:

```env
API_BASE_URL=mock
```

Esto activa el `MockAPIClient`, que permite encolar jobs manualmente.
Puedes encolar jobs de prueba desde un script o REPL:

```python
from app.config import Settings
from app.services.api_client import MockAPIClient

client = MockAPIClient(Settings())
client.enqueue_health()
client.enqueue_transcribe(r"C:\videos\mi_video.mp4")
```

### Demo de flujo completo (end-to-end, sin VPS)

Con `API_BASE_URL=mock`, ejecuta una secuencia realista de extremo a extremo
sobre un vídeo local (transcribe → render → qa) recorriendo los estados reales
del worker (claimed → processing → completed → upload):

```bash
.venv\Scripts\python demo_full_flow.py "C:\ruta\al\video.mp4"
```

Produce `transcript.json`, un clip `.mp4` en 9:16 con subtítulos + watermark,
y el resultado del QA, subiendo cada resultado al `MockAPIClient`.

> **Nota GPU (RTX 5070 Ti / Blackwell):** la transcripción necesita torch con
> la build CUDA correcta. En esta máquina se usó:
> `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`

### Notas de implementación (whisperx 3.8+ y FFmpeg en Windows)

- **whisperx 3.8+**: `load_model()` devuelve un pipeline; la transcripción es
  `pipeline.transcribe(audio, ...)`, no `whisperx.transcribe()`. El audio se
  pre-carga a numpy (16 kHz) para evitar el decoding interno (torchcodec), que
  en Windows puede estar roto.
- **FFmpeg + rutas Windows**: los filtros `subtitles`/`drawtext`/`ass` no
  aceptan rutas con drive colon (`C:\...`). Por eso `render_clip` referencia los
  archivos de subtítulos por nombre relativo y ejecuta ffmpeg con `cwd` en su
  carpeta. Con watermark (2 entradas) se usa `-filter_complex` + `-map`.
- `-t <dur>` debe colocarse **después de todos los inputs** para limitar el
  clip; si va antes de un `-i` extra se interpreta como opción de input y
  procesa el vídeo completo.

---

## Jobs soportados

| Tipo | Descripción | Herramienta |
|---|---|---|
| `health` | Reporta estado del sistema, GPU y tools | — |
| `download` | Descarga archivos por streaming + SHA256 | httpx |
| `transcribe` | Transcripción con timestamps y diarización | WhisperX |
| `render` | Recorta, reencuadra, subtítulos y watermark | FFmpeg |
| `qa` | Validación técnica del vídeo | FFprobe |

### Añadir un nuevo job

1. Crea `app/jobs/mi_job.py` con una clase que herede de `BaseJob`.
2. Implementa `execute()` devolviendo un `dict`.
3. Regístrala en `app/services/job_manager.py`:

```python
JOB_HANDLERS = {
    ...
    "mi_job": MiJob,
}
```

No necesitas tocar el runner ni el worker loop.

---

## Estructura del proyecto

```
clipping-windows-worker/
├── app/
│   ├── main.py            # Lifecycle + worker loop
│   ├── config.py          # Configuración (pydantic-settings)
│   ├── models/            # Modelos Pydantic
│   ├── services/          # API client, job runner/manager, system info
│   ├── jobs/              # Handlers de jobs
│   ├── tools/             # Wrappers de ffmpeg/ffprobe/whisperx
│   └── utils/             # Logging, paths, subprocess
├── data/                  # Datos generados (no versionar)
├── tests/
├── requirements.txt
├── .env.example
└── run.py
```

Cada job tiene su propio directorio aislado:

```
data/jobs/job_123/
├── input/
├── temp/
├── output/
└── logs/
```

---

## Tests

```bash
pytest
```

Los tests cubren lógica no dependiente de GPU (config, modelos, mock API,
utilidades). Los jobs de GPU/FFmpeg se prueban de forma independiente.

---

## Notas de producción

- El worker usa **polling** (`GET /worker/jobs/next`), no requiere puertos
  públicos abiertos en el PC Windows.
- Se recomienda conectar VPS y worker mediante **Tailscale** (red privada).
- Los jobs fallidos **no** matan al worker: se registran, se reportan y el
  loop continúa.
- La limpieza automática no borra resultados hasta que el servidor confirma
  su recepción.
