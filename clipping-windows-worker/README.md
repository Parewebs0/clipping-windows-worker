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
| `INSTAGRAM_ACCESS_TOKEN` / `INSTAGRAM_IG_USER_ID` | Graph API (Reels). Otra cuenta: `INSTAGRAM_ACCESS_TOKEN_<KEY>` | — |
| `INSTAGRAM_AUTH` | `facebook` (página vinculada, `graph.facebook.com`) o `instagram` (Instagram Login, `graph.instagram.com`) | `facebook` |
| `INSTAGRAM_APP_ID` / `INSTAGRAM_APP_SECRET` | App de Meta. Hacen falta para refrescar el token en modo `facebook` | — |
| `INSTAGRAM_GRAPH_VERSION` | Versión de Graph API en las URLs de publicación y de refresco `facebook` | `v21.0` |
| `TIKTOK_MODE` | `inbox` (borrador en la app) o `direct` (publica solo, scope `video.publish`). Si el sondeo se agota con el vídeo ya subido, el `publish_id` queda en `<WORKING_DIRECTORY>/tiktok_pending/` y el reintento solo consulta ese id | `inbox` |
| `TIKTOK_ACCESS_TOKEN` | Access token. Si hay client key, secret y refresh, se refresca cuando la caducidad es desconocida, queda menos de 10 min, o TikTok responde 401. El nuevo queda en `<WORKING_DIRECTORY>/oauth_state/` (no en `.env`). Sin material de refresco se usa tal cual | — |
| `TIKTOK_CLIENT_KEY` / `TIKTOK_CLIENT_SECRET` / `TIKTOK_REFRESH_TOKEN` | OAuth para refrescar el access token. Por cuenta: sufijo `_<ACCOUNT>`. TikTok rota el refresh: el mini PC y el worker de Windows no pueden compartir el mismo token; credenciales distintas por worker | — |
| `TIKTOK_PRIVACY` | `SELF_ONLY` / `MUTUAL_FOLLOW_FRIENDS` / `FOLLOWER_OF_CREATOR` / `PUBLIC_TO_EVERYONE`. En `direct` tiene que estar en `privacy_level_options`. Una campaña de pago (`brand_content_toggle`) no puede ir con `SELF_ONLY` | `SELF_ONLY` |

El token de larga duración dura unos 60 días. Antes de subir, si quedan menos de 7 días (o no hay fecha), el worker lo renueva y guarda token + caducidad en `<WORKING_DIRECTORY>/oauth_state/instagram_<cuenta>.json` (modo `0600`). No subas esa carpeta a git. Si cambias el token en `.env`, ese json deja de usarse. En modo `facebook` hacen falta `INSTAGRAM_APP_ID` y `INSTAGRAM_APP_SECRET`; si faltan, no se intenta el refresco y el aviso sale una sola vez. Si el refresco falla por otro motivo, la subida sigue con el token actual y queda un WARNING en el log (sin el token ni el secret).

> **Seguridad:** nunca subas `.env` a Git. Usa `.env.example` como plantilla.

Otras cuentas usan el mismo nombre de variable más `_` y la clave en mayúsculas, leída del entorno del proceso. YouTube, si la cuenta no es `default`, exige las tres: `YOUTUBE_CLIENT_ID_YT2`, `YOUTUBE_CLIENT_SECRET_YT2` y `YOUTUBE_REFRESH_TOKEN_YT2`. Si falta una, el job falla y no rellena con las de `default`, así que no mezcla el client de una cuenta con el refresh token de otra. Instagram y TikTok, si no hay sufijo, siguen usando la credencial `default` campo a campo: `INSTAGRAM_ACCESS_TOKEN_IG2`, `INSTAGRAM_IG_USER_ID_IG2`, `TIKTOK_ACCESS_TOKEN_STUDIO`.

El job publica exactamente `payload.account` (o `default` si el payload no trae cuenta). No recorre cuentas, no elige la primera que esté sana por orden alfabético y no asocia una cuenta a la campaña: eso lo decide quien crea el job. En YouTube, `channels.list` devuelve los canales de ese token y el chequeo enseña el primero (`items[0]`). Eso no elige canal por campaña.

## Comprobar credenciales (sin publicar)

`scripts/check_publish_creds.py` hace una llamada de solo lectura por plataforma. No sube ni publica nada, y no imprime tokens: si falta una variable, el motivo es el nombre (`YOUTUBE_REFRESH_TOKEN missing`).

```bash
python scripts/check_publish_creds.py
python scripts/check_publish_creds.py --platform youtube --account default
python scripts/check_publish_creds.py --platform instagram --account ig2
python scripts/check_publish_creds.py --platform tiktok --account studio
```

`--platform` es `all` (por defecto), `youtube`, `instagram` o `tiktok`. `--account` es `default` si no se indica. Cada línea es `plataforma cuenta OK|FAIL motivo`. El proceso sale con 0 si todas están OK y con 1 si alguna falla.

`default` usa `YOUTUBE_CLIENT_ID` / `YOUTUBE_CLIENT_SECRET` / `YOUTUBE_REFRESH_TOKEN` (y los equivalentes de Instagram y TikTok). Otra cuenta lee solo las variables con sufijo. En YouTube el job hace lo mismo: sin las tres variables con sufijo, falla y no llama a la API. Si falta `YOUTUBE_REFRESH_TOKEN_YT2`, el script también falla sin red. Instagram y TikTok en el job sí caen a `default` cuando no hay sufijo; el script no, para no dar un OK de la cuenta equivocada.

`--account` comprueba una sola cuenta, la que pases. No busca la primera cuenta sana.

YouTube refresca el token (sin pedir scopes de más) y llama a `channels.list(part=snippet, mine=true)`. Imprime el título del primer canal que devuelve el token, si incluye `youtube.upload` y, si está, `yt-analytics.readonly`. Instagram hace `GET /{ig-user-id}?fields=username,account_type` y exige `BUSINESS`, `CREATOR` o `MEDIA_CREATOR`. El host es `graph.facebook.com` salvo `INSTAGRAM_AUTH=instagram` (`graph.instagram.com`). La versión es `INSTAGRAM_GRAPH_VERSION` si tiene forma `v21.0` (si no, `v21.0`). Si existe `<WORKING_DIRECTORY>/oauth_state/instagram_<cuenta>.json` con `access_token`, el chequeo usa ese token (salvo que `seed` indique que el token de entorno cambió) y no refresca ni escribe el fichero. TikTok hace `POST /v2/post/publish/creator_info/query/` e imprime el username y `privacy_level_options`. Con `oauth_state/tiktok_<cuenta>.json`, usa el `access_token` guardado antes que `TIKTOK_ACCESS_TOKEN`. El script no imprime esos tokens.

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

### Render: contrato `render_spec` v2 (issue #4)

El VPS manda en el payload del job `render` (tiempos relativos al clip):

```json
{
  "captions": {"enabled": true, "segments": [{"start": 0.0, "end": 2.1, "text": "...", "words": [{"start": 0.0, "end": 0.3, "word": "..."}]}],
               "brand_dictionary": ["BOXABL"], "style": {"position": "bottom", "max_words": 4}},
  "on_screen_text": {"enabled": true, "items": [{"text": "Link in bio", "start": 0, "end": 3, "position": "top"}]},
  "watermark": {"enabled": true, "url": "https://.../logo.png", "position": "top_right", "width": 220, "start": 0, "end": null}
}
```

- Subtítulos y texto en pantalla se generan en un único `.ass` (filtro `ass`, libass) desde la transcripción ya hecha; el diccionario de marca fija la ortografía.
- El logo se descarga a `jobs/<id>/input/logo.*` (PNG/JPG/WebP; SVG no) y se superpone con `overlay` (ventana `start`/`end` opcional).
- Compatibilidad: `captions.file` (.srt/.ass) y `watermark.file` siguen funcionando.
- El resultado añade `applied{captions,on_screen_text,watermark}` y `probe{width,height,fps,has_audio,duration,codec}` para el verificador post-render del VPS.
- `audio.enhance` (`{engine: none|deepfilter|resemble, strength}`) corre **antes** del loudnorm a -14 LUFS. El defecto `none` no toca el audio. `audio.original_only` tampoco. DeepFilterNet 3 se instala con `requirements-enhance.txt` (sin bajar torch). Resemble Enhance es opcional: su rueda de PyPI fija torch 2.1.1.

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
