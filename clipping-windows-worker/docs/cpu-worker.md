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
