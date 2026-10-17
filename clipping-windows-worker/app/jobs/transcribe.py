"""Job de transcripción con WhisperX + GPU."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.jobs.base import BaseJob
from app.tools.ffmpeg import FFmpegTool
from app.tools.ffprobe import FFprobeTool
from app.tools.whisperx import WhisperXTool


class TranscribeJob(BaseJob):
    """Transcribe un vídeo/audio y guarda transcript.json."""

    type = "transcribe"

    def execute(self) -> dict[str, Any]:
        payload = self.job.payload
        # Aceptar `video_path` (contrato interno del Worker) y `video`
        # (contrato del VPS / OpenClaw). Priorizar `video_path` si llega.
        video_path = payload.get("video_path") or payload.get("video")
        language = payload.get("language", "auto")
        diarization = bool(payload.get("diarization", False)) and getattr(self.settings, "diarization_enabled", True)
        # MOLI-54: vocabulario de marca como initial_prompt (ortografía exacta:
        # "boxabl" → "BOXABL") sin condicionar las palabras.
        initial_prompt = payload.get("initial_prompt") or None
        if initial_prompt is not None:
            initial_prompt = str(initial_prompt).strip()[:900] or None

        if not video_path:
            raise ValueError("payload.video_path or payload.video is required")

        # Resolver la ruta: puede ser absoluta o relativa al job.
        source = self._resolve_input(video_path)
        if not source.exists():
            raise FileNotFoundError(f"Input video not found: {source}")

        # Validar que el input tiene pista de audio ANTES de extraer.
        # Si no la tiene, ffmpeg.extract_audio falla con un mensaje críptico
        # ("Output file does not contain any stream") que no deja claro
        # al VPS la causa real. Mejor fallar pronto con un error accionable.
        ffprobe = FFprobeTool(self.settings)
        audio_stream = ffprobe.get_audio_stream(source)
        if audio_stream is None:
            video_stream = ffprobe.get_video_stream(source)
            video_codec = video_stream.get("codec_name") if video_stream else None
            duration = ffprobe.get_duration(source)
            duration_str = f"{duration:.2f}s" if duration is not None else "unknown"
            raise ValueError(
                f"Input has no audio stream "
                f"(video_codec={video_codec}, duration={duration_str}); "
                "transcription requires an audio track. "
                "Re-export the source with audio or skip this asset."
            )

        self.logger.info("extracting audio", video=str(source))
        ffmpeg = FFmpegTool(self.settings)
        work = self._work_dir(payload)
        audio_path = (work or self.directory.temp) / "audio.wav"
        ffmpeg.extract_audio(source, audio_path)

        self.logger.info("starting whisperx", language=language, diarization=diarization,
                         initial_prompt=bool(initial_prompt))
        whisperx = WhisperXTool(self.settings)
        result = whisperx.transcribe(
            audio_path,
            language=None if language == "auto" else language,
            diarization=diarization,
            initial_prompt=initial_prompt,
        )

        # Guardar transcript.json en el directorio de salida.
        transcript_path = (work or self.directory.output) / "transcript.json"
        transcript_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        self.logger.info(
            "transcription completed",
            language=result.get("language"),
            segments=len(result.get("segments", [])),
        )
        return {
            "transcript_path": str(transcript_path),
            "language": result.get("language"),
            "duration": result.get("duration"),
            "segments": result.get("segments", []),
            "words": result.get("words", []),
        }

    def _work_dir(self, payload: dict) -> Path | None:
        """<STORAGE_ROOT>/<campaign_id>/work/<asset_id>/ when configured, else None (job dir)."""
        root = getattr(self.settings, "storage_root", None)
        cid, aid = payload.get("campaign_id"), payload.get("asset_id")
        if not root or cid in (None, "") or not aid:
            return None
        from app.utils.storage_layout import StorageLayout, StoragePathError

        try:
            work = StorageLayout(root).work_dir(cid, aid)
        except StoragePathError:
            return None  # odd ids: keep the job-dir behaviour
        work.mkdir(parents=True, exist_ok=True)
        return work

    def _resolve_input(self, video_path: str) -> Path:
        """Resuelve una ruta de entrada, absoluta o relativa al job."""
        path = Path(video_path)
        if path.is_absolute():
            return path
        # Relativa: buscar en input/ del job y en el directorio raíz.
        candidates = [
            self.directory.input / path,
            self.directory.root / path,
            Path(self.settings.working_directory) / path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]
