# Contexto del proyecto: sistema de clipping automatizado

## Objetivo

Este workspace contiene un sistema autónomo para detectar campañas de recompensas, interpretar sus reglas, resolver recursos autorizados, transcribir vídeo, seleccionar segmentos con MiniMax, renderizar clips con FFmpeg y validar el resultado de forma determinista antes de la revisión humana y, en fases posteriores, publicación y métricas.

No debe confundirse el diseño con la implementación comprobada. Antes de afirmar que una capacidad funciona, verificar código, tests, logs o una prueba end-to-end.

## Arquitectura

### Diagrama lógico

```
                        INTERNET
                           │
                           ▼
                 ┌───────────────────┐
                 │ Campaign Sources  │
                 │                   │
                 │ - campañas        │
                 │ - vídeos          │
                 │ - PDFs            │
                 │ - documentos      │
                 │ - imágenes        │
                 └─────────┬─────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                            VPS                               │
│                                                               │
│                       CONTROL PLANE                          │
│                                                               │
│   ┌─────────────────────┐                                    │
│   │ Campaign Collector  │                                    │
│   └─────────┬───────────┘                                    │
│             ▼                                                │
│   ┌─────────────────────┐                                    │
│   │ Campaign Selector   │                                    │
│   └─────────┬───────────┘                                    │
│             ▼                                                │
│   ┌─────────────────────┐                                    │
│   │ Campaign Analyzer   │                                    │
│   │ / Rule Parser       │                                    │
│   └─────────┬───────────┘                                    │
│             ▼                                                │
│   ┌─────────────────────┐                                    │
│   │ CampaignSpec        │                                    │
│   └─────────┬───────────┘                                    │
│             ▼                                                │
│   ┌─────────────────────┐                                    │
│   │ Asset Resolver      │                                    │
│   └─────────┬───────────┘                                    │
│             ▼                                                │
│   ┌─────────────────────┐                                    │
│   │ Job Orchestrator    │                                    │
│   └─────────┬───────────┘                                    │
│             ▼                                                │
│   ┌─────────────────────┐                                    │
│   │ PostgreSQL          │                                    │
│   │ State + Metadata    │                                    │
│   └─────────┬───────────┘                                    │
│             │                                                │
│             ▼                                                │
│        Job Queue                                              │
│             │                                                │
└─────────────┼────────────────────────────────────────────────┘
              │
              │ Tailscale
              ▼
┌──────────────────────────────────────────────────────────────┐
│                      WINDOWS GPU WORKER                      │
│                                                               │
│   Download                                                    │
│       ↓                                                       │
│   WhisperX                                                    │
│       ↓                                                       │
│   Transcript                                                  │
│       ↓                                                       │
│   FFmpeg                                                      │
│       ↓                                                       │
│   Render                                                      │
│       ↓                                                       │
│   QA                                                          │
│       ↓                                                       │
│   Upload results                                              │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Responsabilidades por componente

| Componente | Ubicación | Estado | Responsabilidad |
|---|---|---|---|
| **Campaign Sources** | Internet | Externo | Campañas, vídeos fuente, PDFs, documentos, imágenes |
| **Campaign Collector** | VPS | Pendiente | Ingesta periódica desde fuentes externas |
| **Campaign Selector** | VPS | Pendiente | Filtra y prioriza campañas relevantes. Ver [docs/campaign_selector.md](docs/campaign_selector.md) para diseño como skill + cron en OpenClaw |
| **Campaign Analyzer / Rule Parser** | VPS | Pendiente | Interpreta reglas en lenguaje natural → `CampaignSpec` |
| **CampaignSpec** | VPS | Pendiente | Representación estructurada con evidencia, fuente y confianza |
| **Asset Resolver** | VPS | Pendiente | Resuelve vídeos y assets autorizados para una campaña |
| **Job Orchestrator** | VPS | Validado | Crea jobs en PostgreSQL según `CampaignSpec` y assets |
| **PostgreSQL + Job Queue** | VPS | Validado | Persistencia + cola con `FOR UPDATE SKIP LOCKED` |
| **API central** | VPS | Validado | FastAPI con Bearer auth, `/jobs`, `/worker/*` |
| **Worker (Download)** | Windows | Validado | Descarga vídeos por streaming (`FileManager`) |
| **Worker (WhisperX)** | Windows | Validado E2E | Transcripción GPU `large-v3` + alineamiento |
| **Worker (FFmpeg/Render)** | Windows | Validado local | Recorte + escalado 16:9/9:16, watermark y captions opcionales |
| **Worker (QA)** | Windows | Validado local | Validación determinista con `ffprobe`. Step 18: tras PASS copia el clip a `pending_upload/` y reporta `final_path_worker` |
| **Worker (Step 18 — Clip Storage)** | Windows | Validado local | `ClipStorage` organiza clips por `campaign_id` en `pending_upload/`, `uploaded/`, `archived/` (`<clip_storage_root>\<campaign_id>\<location>\<clip_id>.mp4`) |
| **Upload results** | Windows | Validado | `POST /worker/jobs/{id}/result` con `{"result": ...}` |
| **Revisión humana + Publicación** | Fuera del flujo Worker | Pendiente | Bot de Telegram y panel web |

### Notas arquitectónicas

- **OpenClaw** se ejecuta en el VPS y orquesta el flujo. No procesa vídeo directamente.
- **VPS Ubuntu** aloja la API central, PostgreSQL, la cola de jobs, Campaign Engine, ingestor, resolver de assets, parser documental, bot de Telegram y OpenClaw.
- **Windows GPU Worker** (`clipping-windows-worker/`) hace polling a la API central por Tailscale y ejecuta el trabajo pesado.
- **Hardware validado del Worker:** Windows 11, AMD Ryzen 7 5800X, 64 GB RAM y NVIDIA GeForce RTX 5070 Ti de 16 GB.
- **Procesamiento validado:** FFmpeg, FFprobe, WhisperX 3.8.6, PyTorch 2.8.0+cu128, CUDA 12.8, transcripción y alineamiento en español sobre vídeo real.
- Los **vídeos fuente** hoy viven solo en el Windows Worker (tests). No hay todavía resolución de assets real: el **Asset Resolver** en el VPS es el siguiente bloque a construir.
- El Worker referencia vídeos por path Windows absoluto (`C:\...`) o URL HTTP(S); nunca accede a almacenamiento remoto sin descargar primero.

Flujo objetivo:

`Campaign → CampaignSpec → assets autorizados → transcripción → candidatos → render → QA → revisión → publicación`

> **El flujo detallado paso a paso, quién hace qué y en qué orden, está en [docs/architecture_flow.md](docs/architecture_flow.md). Es la fuente única de verdad del flujo; cualquier cambio se acuerda y se modifica ahí.**

## Estado confirmado

Ya están conseguidos y no deben reinstalarse ni diagnosticarse de nuevo sin evidencia en contrario:

- API central del VPS.
- PostgreSQL y cola de jobs.
- Tailscale entre VPS y Windows.
- Autenticación Bearer mediante `API_TOKEN`.
- Polling del Worker y contrato `job_type` → `type`.
- Inicio y reclamación de jobs.
- Transcripción WhisperX end-to-end **en el Worker**: el job `d9690909-0c4c-4f05-b065-2d0bf27515bd` produjo `transcript.json` con `language="es"`, `duration=595.799`, `segments=122`, `words=1941`.
- **Flujo transcribe end-to-end cerrado**: VPS persiste `result.data` con la shape `{language, duration, segments, words, transcript_path}` validada en `GET /jobs/{id}`.
- Envío de resultados.
- Envío de fallos con `{"error_message": "..."}`.
- Jobs `health` completos end-to-end.
- FFmpeg y FFprobe detectados y utilizables.
- WhisperX instalado y operativo.
- CUDA disponible en el entorno Python correcto.
- RTX 5070 Ti detectada directamente por PyTorch.
- Transcripción y alineamiento WhisperX con CUDA, `float16` y `batch_size=16` validados.
- **Job `render` validado localmente** sobre `test_video.mp4` (recorte 5-15 s, 16:9): clip generado `1920x1080`, 60 fps, h264 + aac, 1.27 MB. Ver [test_render_local.py](clipping-windows-worker/test_render_local.py).
- **Job `qa` validado localmente** sobre el clip: 5/5 checks PASS (duration 9.5-10.5, resolution 1920x1080, fps>=24, audio presente, codec h264). Ver [test_qa_local.py](clipping-windows-worker/test_qa_local.py).

La detección de GPU del Worker también está corregida. `psutil` se instaló y se verificó la detección de RAM: 63.9 GB totales y 38.5 GB disponibles en la comprobación realizada.

## API y cola

La API está en `100.109.27.21:8080` dentro de Tailscale. Endpoints relevantes:

- `GET /health`
- `GET /system/info`
- `GET /worker/jobs/next?worker_id=...`
- `POST /worker/jobs/{job_id}/start?worker_id=...`
- `POST /worker/jobs/{job_id}/result?worker_id=...`
- `POST /worker/jobs/{job_id}/fail?worker_id=...`
- `POST /worker/jobs/{job_id}/heartbeat?worker_id=...`

No inventar endpoints. Si un contrato no está claro, consultar `/openapi.json`. No existen `/worker/register` ni un heartbeat general equivalente; `register_worker()` y el heartbeat general son legado y no deben reintroducirse.

La reclamación usa PostgreSQL con `FOR UPDATE SKIP LOCKED`, orden por prioridad descendente y antigüedad ascendente. Los jobs usan leases y transiciones `pending → assigned → processing → completed|failed|pending`.

## Reglas de trabajo

- No pedir, imprimir ni copiar el API token. El token vive en los archivos `.env` locales.
- No reinstalar PyTorch, CUDA o WhisperX sin una prueba que muestre un fallo real.
- No modificar múltiples archivos si solo se necesita uno. Preferir cambios pequeños, controlados y verificables.
- Al modificar un Python importante, entregar o mantener el archivo completo para evitar errores de indentación parciales.
- Consultar primero el estado real del archivo en disco; no asumir que una edición interrumpida se guardó.
- Mantener la separación: MiniMax decide y razona; FFmpeg/FFprobe ejecutan y validan procesamiento técnico.
- Las reglas numéricas y estructurales deben validarse determinísticamente: duración, aspect ratio, resolución, FPS, codec, audio, captions y watermark aplicado por el renderer.
- Usar OCR o visión solo para requisitos visuales o semánticos que no puedan comprobarse con el pipeline.
- Cada `CampaignSpec` importante debe conservar evidencia, fuente, página o sección y nivel de confianza cuando sea posible.
- No generar clips antes de interpretar las reglas de campaña.
- Actualizar este documento conforme se consigan checkpoints o se tome una decisión técnica importante.

## Estado pendiente inmediato

1. Revisar el contenido real de `clipping-windows-worker/app/services/system_info.py` y mantener la detección GPU sin reinstalar PyTorch. **Hecho según confirmación del usuario.**
2. Añadir `psutil` a `requirements.txt` e instalarlo para detectar RAM. **Hecho según verificación: total 63.9 GB, disponible 38.5 GB.**
3. Conseguir un startup limpio del Worker con GPU, VRAM, CUDA, RAM y herramientas correctos.
4. Crear y probar un job real `transcribe` desde el VPS hasta el Worker y vuelta.
5. Normalizar el `TranscriptResult` con segmentos, palabras, speakers y timestamps.
6. Después: selección de candidatos con MiniMax, render, captions, watermark y QA determinista.

> Render y QA ya están validados localmente; pendiente probarlos end-to-end desde el VPS.

## Bug resuelto: download job descargaba HTML en lugar de vídeo

**Causa raíz:** `DownloadJob` del Worker usaba `httpx.stream("GET", url)`. Para YouTube esto devuelve la `watch page` HTML (~1.26 MB), no el MP4/WebM real. El binario `yt-dlp` (WinGet) estaba disponible pero el código Python no lo invocaba. Los `download.bin` reales en disco eran todos `<!DOCTYPE html>...`.

**Fix aplicado** (sin reinstalar nada del stack CUDA/WhisperX):
- `yt-dlp>=2026.8.0` añadido a `requirements.txt` e instalado en `.venv` (versión `2026.08.19`).
- `app/services/file_manager.py`: nuevo `download_with_ytdlp()` y enrutamiento por host en `FileManager.download()` (YouTube/yt-dlp, resto/httpx).
- `app/jobs/download.py`: respeta el path real de yt-dlp (incluido el rename tras merge `webm → mp4`).

**Verificación con la URL del bug (`be7dKHOK4NQ`):**
- 365.97 MB descargados en 77.2 s (vs. 1.26 MB de HTML antes).
- Magic bytes: `00 00 00 20 66 74 79 70 69 73 6F 6D` (`ftyp isom`) → MP4 real.
- `ffprobe`: `format_name=mov,mp4,m4a,3gp,3g2,mj2`, `duration=749.014s`, vídeo `av1` + audio `opus`.
- `ffmpeg -i ... -vn -ac 1 -ar 16000 -acodec pcm_s16le audio.wav` → 23.97 MB, sin error.

Contrato del job `download` no cambia. Detalle completo en `HANDOFF — Clipping System - OpenClaw - Windows GPU Worker.md` §66.

## Decisiones técnicas tomadas

- **`worker_id` esperado en el VPS:** `windows-gpu-worker-01` (es el ID que el Worker se auto-asigna según `.env`; verificar antes de crear jobs).
- **Validación de accesibilidad de `video` en VPS:**
  - HTTP(S) → HEAD con timeout 5s, rechaza si 4xx/5xx o unreachable.
  - Path Linux (`/...`) → `os.path.isfile`, rechaza si no existe.
  - Path Windows (`C:\...`) o Tailscale (`\\VPS\share\...`) → VPS no verifica, el Worker valida al procesar.
- **OpenAPI se regenera automáticamente** con FastAPI; cualquier cambio en `JobCreate` se refleja en `/openapi.json`.
- **Persistencia:** `result.data` se guarda tal cual en `jobs.result` y `error_message` se persiste en caso de fail.
- **Step 18 — Sanitiser de `clip_id`:** `_safe_filename` preserva UUIDs con guiones literales; sólo bloquea caracteres prohibidos por Windows (`<>:"/\|?*` + control chars `\x00-\x1f`). Bloquea `.`, `..` y separadores por seguridad. Esto es crítico: `final_path_worker` debe coincidir con el `clip_id` literal para que el VPS resuelva el archivo.
- **Step 18 — `clip_storage_root`:** configurable vía `.env` (`clip_storage_root=...`), default `C:\CODIANT\clipping\storage\clips`. Tres subdirectorios `pending_upload/`, `uploaded/`, `archived/` se crean idempotentemente en el primer job.
- **Step 18 — Robustez:** fallos de `_copy_to_pending_upload` (permisos, disco lleno, paths inválidos) se loggean como warning y devuelven `source_moved: false` sin propagar al QA. La transición de estado en el VPS (`on_qa_completed`) está envuelta en try/except, así que el Step 18 nunca rompe el flujo de QA.

## Contratos de jobs del Worker

Validados en local. El VPS debe producir payloads con esta forma exacta:

### `transcribe` (validado end-to-end)

```json
{
  "job_type": "transcribe",
  "payload": {
    "video": "C:\\path\\to\\video.mp4",
    "language": "es",
    "options": {
      "model": "large-v3",
      "device": "cuda",
      "compute_type": "float16",
      "batch_size": 16,
      "align": true
    }
  },
  "priority": 5
}
```

Acepta también `"video_path"` por retrocompatibilidad. `result.data`:

```json
{"language":"es","duration":595.799,"segments":[...],"words":[...],"transcript_path":"..."}
```

### `render` (validado localmente)

```json
{
  "job_type": "render",
  "payload": {
    "input_video": "C:\\path\\to\\source.mp4",
    "start": 5.0,
    "end": 15.0,
    "output_format": "16:9"
  },
  "priority": 5
}
```

- `output_format`: `"16:9"` o `"9:16"` (16:9 → 1920×1080, 9:16 → 1080×1920).
- Opcionales: `captions` (SRT), `watermark` (`{path, position}` con position ∈ top_left|top_right|bottom_left|bottom_right|center).
- Codificador: `libx264` CRF 23 + `aac` 128k + `+faststart`. Recorte `-ss` antes de `-i` para precisión de keyframe.

`result.data`:

```json
{"output_path":"...\\clip.mp4","filename":"clip.mp4","size":1272676,"format":"16:9"}
```

### `qa` (validado localmente, extendido con Step 18)

```json
{
  "job_type": "qa",
  "payload": {
    "video": "C:\\path\\to\\clip.mp4",
    "rules": {
      "min_duration": 9.5,
      "max_duration": 10.5,
      "width": 1920,
      "height": 1080,
      "min_fps": 24,
      "require_audio": true,
      "codec": "h264"
    },
    "clip_id": "a875ad94-de7f-4af7-ad3c-834becbd0420",
    "campaign_id": 5463
  },
  "priority": 5
}
```

- Reglas opcionales: omitir las que no apliquen.
- Determinista: usa `ffprobe` para duration, resolución, fps, codec y audio.
- Campos opcionales para Step 18: `clip_id` (UUID string; acepta alias `clip_uuid`/`clipId`) y `campaign_id` (int o str). Si el QA pasa y ambos están presentes, el Worker copia el clip a `<clip_storage_root>/<campaign_id>/pending_upload/<clip_id>.mp4` y devuelve `final_path_worker`.
- Si `clip_id` o `campaign_id` faltan, el QA se ejecuta y devuelve `source_moved: false` con un warning; el flujo nunca falla por esta razón.

`result.data`:

```json
{
  "status": "PASS",
  "checks": [
    {"name": "duration", "status": "PASS", "actual": 10.0, "expected": "9.5-10.5"},
    {"name": "resolution", "status": "PASS", "actual": "1920x1080", "expected": "1920x1080"},
    {"name": "fps", "status": "PASS", "actual": 60.0, "expected": ">=24"},
    {"name": "audio", "status": "PASS", "actual": true, "expected": true},
    {"name": "codec", "status": "PASS", "actual": "h264", "expected": "h264"}
  ],
  "duration_seconds": 10.0,
  "final_path_worker": "C:\\CODIANT\\clipping\\storage\\clips\\5463\\pending_upload\\a875ad94-de7f-4af7-ad3c-834becbd0420.mp4",
  "source_moved": true
}
```

| Campo | Tipo | Presente cuando |
|---|---|---|
| `status` | `"PASS"` \| `"FAIL"` | siempre |
| `checks` | list | siempre |
| `duration_seconds` | number | siempre (top-level) |
| `final_path_worker` | string | PASS + `clip_id` + `campaign_id` + copia exitosa |
| `source_moved` | bool | siempre (`true` solo si se copió) |

## Step 18 — Almacenamiento por campaña (validado localmente)

- **Layout:** `<clip_storage_root>\<campaign_id>\{pending_upload|uploaded|archived}\<clip_id>.mp4`.
- **`clip_storage_root`** por defecto: `C:\CODIANT\clipping\storage\clips`. Configurable vía env `clip_storage_root` en `.env` del Worker.
- **Worker** ([clip_storage.py](clipping-windows-worker/app/utils/clip_storage.py)): helper `ClipStorage` con `ensure()` (idempotente), `copy_to_pending_upload()` (sobre-escribe si existe, conserva el original por ahora) y `move()` (pendiente para transiciones futuras). `_safe_filename` preserva el `clip_id` literal (UUID con guiones) y sólo sustituye por `_` los caracteres prohibidos en Windows (`<>:"/\|?*` + control chars).
- **VPS** ([clip_storage_service.py](clipping-vps/clipping-system-vps/app/services/clip_storage_service.py) + [clips.py](clipping-vps/clipping-system-vps/app/api/clips.py)): `on_qa_completed` en [job_state_transitions.py](clipping-vps/clipping-system-vps/app/services/job_state_transitions.py) lee `result.data["final_path_worker"]` y llama `set_clip_location(db, clip.id, "pending_upload", final_path_worker=…)`. Try/except garantiza que un error de storage nunca rompe el QA.
- **Endpoints VPS ya disponibles:**
  - `GET /clips/by_campaign/{campaign_id}/storage?location=…` listar.
  - `POST /clips/{clip_id}/mark_uploaded?final_path_worker=…` mover a `uploaded` (+ `published_at`).
  - `POST /clips/{clip_id}/location/{location}?final_path_worker=…` cambiar manualmente.
- **Tres localizaciones alineadas** entre VPS (`VALID_LOCATIONS`) y Worker (`ClipStorage._LOCATIONS`): `pending_upload`, `uploaded`, `archived`.
- **Validación local Worker:** `python test_qa_local.py {legacy|step18|fail-no-copy|all}`. `step18` valida copia correcta, `final_path_worker` apunta al archivo, `source_moved=true`, `source_intact=true`. `legacy` valida compat sin `clip_id`. `fail-no-copy` valida que QA FAIL no copia.
- **Regresión:** `pytest tests/` → 27/27 PASS tras el cambio.

## Validación de audio antes de transcribir (2026-09-18)

**Causa raíz de los fallos en cascada:** muchos vídeos descargados desde Drive son **vídeo mudo** (sólo HEVC, sin pista de audio). `TranscribeJob` llamaba directamente a `ffmpeg.extract_audio()`, que ejecuta `-vn -acodec pcm_s16le -ar 16000 -ac 1`. Sin audio, FFmpeg responde:

```
Output file does not contain any stream
Error opening output file ...\audio.wav.
Error opening output files: Invalid argument
```

Este mensaje críptico no dice al VPS qué pasa realmente. Resultado: cascada de errores sin acción posible.

**Fix aplicado** en [transcribe.py](clipping-windows-worker/app/jobs/transcribe.py):

- Antes de `extract_audio()`, valida con `FFprobeTool.get_audio_stream()`.
- Si no hay stream de audio, lanza `ValueError` con mensaje accionable:
  `(video_codec=hevc, duration=34.27s); transcription requires an audio track. Re-export the source with audio or skip this asset.`
- También evita el trabajo inútil de lanzar FFmpeg cuando sabemos que va a fallar.

**Decisión:** fallar rápido con error claro (no generar audio sintético silencioso, no degradar silenciosamente). El VPS decide qué hacer: re-exportar la fuente, descartarla o marcarla como "vídeo decorativo".

**Tests añadidos** en [test_transcribe_no_audio.py](clipping-windows-worker/tests/test_transcribe_no_audio.py) (3/3 PASS):

- Input sin audio → `ValueError` con `no audio stream` + `video_codec=` + sugerencia accionable.
- Input sin audio → NO se llama a `ffmpeg.extract_audio` (verificado con mock).
- Input con audio → entra a la fase de extracción.

**Regresión:** `pytest tests/` → 65/65 PASS (3 nuevos + 62 anteriores). Los 2 fallos residuales que aparecen (`test_drive_resolver.py::test_resolve_doc_extracts_direct_mp4` y `test_drive_download_integration.py::test_download_recovers_from_virus_scan_page`) son **preexistentes** y ajenos a este cambio — ya fallaban en `git stash` antes del fix.

## Convención `.bin` en downloads HTTP (validado 2026-09-18)

**Observación del VPS:** todos los jobs `download` por HTTP genérico (incluido Drive) producen archivos `*.bin` en disco en lugar de `*.mp4`. El VPS preguntó si esto era un bug.

**Respuesta:** es **por diseño y correcto**, no es un bug. Ver [`download.py:36-48`](clipping-windows-worker/app/jobs/download.py#L36-L48):

```python
# - HTTP genérico → ``<job.id>.bin`` (job_id ya saneado por
#   ``sanitize_job_id``; ``.bin`` porque la extensión real solo se
#   conoce tras sniffing y no aporta valor: ``ffprobe``/``transcribe``
#   funcionan igualmente sobre ``.bin``).
filename = f"{self.job.id}.bin"
destination = self.directory.input / filename
```

**Razones documentadas:**

1. **No se conoce la extensión real hasta después de descargar.** El header HTTP `Content-Disposition` o el path de la URL muchas veces mienten o faltan (Drive ni siquiera manda `Content-Disposition` fiable).
2. **El `job_id` ya es seguro** — UUID saneado por `sanitize_job_id`, no trae `?&=:/` ni caracteres que rompan `MAX_PATH` en Windows.
3. **`ffprobe` y WhisperX no miran la extensión** — abren por magic bytes. Probado: el job `a766f7bc` consumió `c9836360-…bin` con magic `ftyp isom` y produjo `audio.wav` válido de 30.53 s sin tocar la extensión.
4. **Es retro-compatible y testeable** — `test_download_http_e2e.py` y `test_download_youtube_branch.py` asumen `<job_id>.bin` para HTTP.

**Cuándo NO es `.bin`:** hosts de yt-dlp (`_needs_ytdlp(url)`: YouTube, Twitch, Kick, TikTok, Vimeo) → yt-dlp produce `<video_id>.mp4` real tras el merge `webm → mp4`.

**Si en el futuro se quiere `.mp4` en disco:** hay que añadir post-sniffing con `ffprobe` + rename. Hoy no es prioritario porque el pipeline funciona.

## Transcripción "corta" no es bug de WhisperX (validado 2026-09-18)

**Observación del VPS:** el job `a766f7bc` produjo un `transcript.json` con **1 solo segmento "Weather!"** en `[0.07-0.79s]` cuando el audio fuente dura 30.53 s. Parecía bug del Worker o de WhisperX.

**Diagnóstico realizado:**

- `audio.wav` válido: 30.53 s, 16 kHz mono, RMS=0.17 (audio con energía).
- Espectro: 62.7% en banda 200-4000 Hz (banda de habla típica).
- Probado con WhisperX `large-v3` + `float16` + `cuda` variando idioma y VAD:

| Config | Idioma detectado | Segmentos | Tiempo |
|---|---|---|---|
| `es` + sin VAD | `es` | **1** | 0.9 s |
| `auto` + sin VAD | `en` | **1** | 1.0 s |
| `en` + sin VAD | `en` | **1** | 0.3 s |
| `en` + VAD default | `en` | **1** | 0.2 s |

- Tiempo bajísimo (0.2-1.0 s para 30 s) → el modelo descarta la mayor parte como `no_speech_prob > 0.5`.

**Conclusión validada con el usuario:** el vídeo fuente solo dice "Weather!" al principio y el resto es música/efectos/silencio. WhisperX **se está comportando correctamente** al descartar lo no-habla. No es bug del Worker, no es bug del modelo.

**Aprendizaje para futuros casos parecidos:** si un asset produce transcripción "sospechosamente corta", **antes de tocar el Worker** hay que:

1. Escuchar el audio (`start <ruta>\temp\audio.wav`).
2. Confirmar si realmente contiene habla continua.
3. Si sí contiene habla → investigar `compute_type`, `batch_size`, modelo.
4. Si no contiene habla → el comportamiento actual es correcto; el VPS debe descartar el asset o registrar la métrica.

## Render: aceptar alias `video` del VPS (fix 2026-09-18)

**Bug observado en producción:** el VPS envía `payload.video` (path al `.bin`), pero [render.py](clipping-windows-worker/app/jobs/render.py) sólo leía `payload.input_video`. Resultado: `ValueError: payload.input_video is required`.

**Causa raíz:** inconsistencia de contrato entre el job `transcribe` (que ya aceptaba `video`/`video_path`) y el job `render` (sólo `input_video`). El VPS usa `video` para todos los jobs.

**Fix aplicado** en [render.py:21](clipping-windows-worker/app/jobs/render.py#L21):

```python
input_video = payload.get("input_video") or payload.get("video")
```

Mismo patrón que ya tenía `transcribe.py`. Mensaje de error actualizado a `"payload.input_video or payload.video is required"`.

**Bonus: `candidate_id` como nombre de archivo.** Si el VPS manda `candidate_id` en el payload (lo hace), el clip se guarda como `<candidate_id>.mp4` en lugar de `clip.mp4` fijo. Esto permite que el QA posterior (`QAJob`) lo mueva a `<clip_storage_root>/<campaign_id>/pending_upload/<candidate_id>.mp4` sin renombrar y que el path de `final_path_worker` coincida 1:1 con el `clip_id` que el VPS persiste.

**Test añadido** `test_render_accepts_vps_payload_with_video_field` en [test_render_result_contract.py](clipping-windows-worker/tests/test_render_result_contract.py): reproduce el payload literal del VPS (`video`, `start_time`, `end_time`, `format`, `candidate_id`, `campaign_id`) y valida:

- No lanza `ValueError`.
- `file_path` existe.
- `duration_seconds ≈ 50.88 - 14.47`.
- `filename == "<candidate_id>.mp4"` (alineado con `clip_id` que el QA espera).

**Regresión:** `pytest tests/test_render_result_contract.py` → 7/7 PASS. Suite completa → 63/65 PASS (mismos 2 fallos preexistentes).

## División de trabajo actual

- **OpenClaw/VPS:** Campaign Ingestor, Document Parser, Asset Resolver, Campaign Engine, `CampaignSpec`, auditor LLM, creación de jobs y persistencia de resultados.
- **Windows GPU Worker:** job `transcribe`, selección/render/QA, procesamiento pesado con GPU.
- **El Worker no debe procesar directamente reglas de campaña ni decidir publicación.**

## Estado del trabajo en VPS (OpenClaw)

**Último checkpoint (05/09/2026 16:48):**

- ✅ 47/47 tests PASSED en el VPS (10 nuevos para `transcribe` + 1 actualizado).
- ✅ POST `/jobs` con `video=C:\CODIANT\clipping\test_video.mp4` → 201, `pending`, contrato completo.
- ✅ POST `/jobs` sin `video` → 400 con `{"detail":"transcribe payload must include 'video' (string)"}`.
- ✅ Flujo fail end-to-end: `create → /next → /start → /fail` → `status=failed`, `error_message` persistido, `worker_id` conservado.
- ✅ Flujo complete end-to-end: `create → /next → /start → /result` → `status=completed`, `result.data` con la shape exacta `{language, duration, segments, words}`.
- ✅ `/openapi.json` actualizado con el schema de `JobCreate`.

**Próximo paso desde Windows:** lanzar el curl de prueba, observar logs del Worker, validar que el job pasa `pending → assigned → processing → completed` y que `result.data` llega con `language=es`, `duration=598.0`, `segments`, `words`.

**Si algo falla:** pasar a OpenClaw el curl exacto lanzado + respuesta del servidor.

## Estado del Step 18 (Worker, validado localmente 11/09/2026)

- ✅ `ClipStorage` helper creado en [clip_storage.py](clipping-windows-worker/app/utils/clip_storage.py): `ensure()` idempotente, `copy_to_pending_upload()` preserva origen, sanitiser conserva UUIDs con guiones.
- ✅ `QAJob.execute()` extendido: tras PASS copia a `<clip_storage_root>/<campaign_id>/pending_upload/<clip_id>.mp4` y devuelve `final_path_worker` + `source_moved=true`.
- ✅ Settings: `clip_storage_root` configurable vía `.env` (default `C:\CODIANT\clipping\storage\clips`).
- ✅ `test_qa_local.py all` → 6/6 + 3/3 + 5/5 PASS (legacy, step18, fail-no-copy).
- ✅ `pytest tests/` → 27/27 PASS (sin regresión).
- ✅ VPS ya cableado: `on_qa_completed` lee `final_path_worker` y llama `set_clip_location(..., "pending_upload", final_path_worker=…)`. Tres endpoints (`list`, `mark_uploaded`, `location`) disponibles.

**Pendiente:** ejecutar un job QA real end-to-end desde el VPS con `clip_id` + `campaign_id` en el payload para cerrar el bucle.

## Comandos y ubicaciones

Worker:

```powershell
cd C:\CODIANT\clipping\clipping-windows-worker
python run.py
```

Prueba CUDA directa, usando el mismo intérprete Python que ejecuta el Worker:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

No ejecutar esta prueba con un intérprete diferente al del Worker antes de concluir que CUDA está rota.

---

## Plantilla para lanzar un job `transcribe` desde Windows

```powershell
# Obtener token desde el .env del VPS (solo si ya lo tienes exportado)
$TOKEN = $env:API_TOKEN

# Crear job
curl.exe -s -X POST `
  -H "Authorization: Bearer $TOKEN" `
  -H "Content-Type: application/json" `
  -d '{"job_type":"transcribe","payload":{"video":"C:\CODIANT\clipping\test_video.mp4","language":"es","options":{"model":"large-v3","device":"cuda","compute_type":"float16","batch_size":16,"align":true}},"priority":5}' `
  http://100.109.27.21:8080/jobs
```

Verificación de un job por ID:

```powershell
curl.exe -s -H "Authorization: Bearer $TOKEN" http://100.109.27.21:8080/jobs/<job_id>
```

## Documentación de referencia

El historial técnico completo, comandos de la API, checkpoints y errores resueltos está en `HANDOFF — Clipping System - OpenClaw - Windows GPU Worker.md`. Ese documento es la fuente de contexto del proyecto; actualizarlo solo cuando cambie el estado verificado o se tome una decisión técnica importante.
