# Worker de respaldo CPU (mini PC)

Entra solo cuando el worker GPU de Windows no da señales (poll/lease) en
`WORKER_STALE_SECONDS` (servidor, 600 s por defecto). Requiere en el `.env`
del servidor `FALLBACK_WORKER_IDS=minipc-cpu-worker-01`.

- Solo reclama `download` libres; lo que descarga (transcribe/render/qa) queda
  fijado a él (`affinity_worker_id`). Los trabajos fijados al GPU esperan.
- Whisper `small` int8, sin diarización, 1 trabajo, `nice 10`, 4 CPU / 4 GB.
- Retención: borra medios y clips con más de 24 h (`MEDIA_RETENTION_*`).
- Publicación apagada: no se le pasan tokens de plataformas.

`.env.cpu` (no se versiona): `API_BASE_URL`, `API_TOKEN`, `GOG_ACCOUNT`,
`GOG_KEYRING_PASSWORD`.

## Instagram y TikTok por Buffer

Con `BUFFER_API_KEY` en `.env.cpu`, los jobs `publish` de `instagram` y `tiktok`
van por la API GraphQL de Buffer (`app/services/buffer_publish.py`) en vez de
por los uploaders directos.

Buffer no tiene endpoint de subida: descarga el vídeo desde una URL pública.
El worker copia el clip con un nombre aleatorio a `BUFFER_PUBLIC_DIR`
(montado en el contenedor con la misma ruta), que el servidor `clip-public`
sirve por Tailscale Funnel en `BUFFER_PUBLIC_BASE_URL`. Crea el post con
`mode: shareNow` y consulta su estado cada 30 s hasta que es `sent` con
`externalLink`, que es la URL que se reporta a la API y se envía a Content
Rewards. La copia pública se borra cuando el post queda `sent` o `error`.

Si se agota `BUFFER_PUBLISH_TIMEOUT` (1200 s por defecto), el job falla pero
el id del post queda en `publish_state/<clip>__<plataforma>.buffer.json`:
el reintento sigue consultando ese post y nunca crea otro.

Variables: `BUFFER_API_KEY`, `BUFFER_CHANNEL_INSTAGRAM`, `BUFFER_CHANNEL_TIKTOK`
(por cuenta: `BUFFER_CHANNEL_INSTAGRAM_<CUENTA>`), `BUFFER_PUBLIC_DIR`,
`BUFFER_PUBLIC_BASE_URL` y, opcional, `BUFFER_PUBLISH_TIMEOUT`.

Límites de Buffer: 250 peticiones al día por key (100 cada 15 min) y
publicaciones diarias por canal (`dailyPostingLimits`).
