"""Wrapper de WhisperX para transcripción con GPU.

Adaptado a la API de whisperx 3.8+ (fork mantenido), donde:
- `load_model()` devuelve un pipeline (`FasterWhisperPipeline`) con método `.transcribe()`.
- El resultado es un `TranscriptionResult` (dict-like).
- `align()` devuelve un `AlignedTranscriptionResult` con palabras por segmento.
- `DiarizationPipeline` vive en `whisperx.diarize`.

whisperx se importa de forma perezosa (lazy) para que el worker pueda
arrancar y reportar health incluso si WhisperX no está instalado.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import Settings
from app.utils.logging import get_logger

logger = get_logger("whisperx")

_MODEL_NAME = "large-v3"
_SAMPLE_RATE = 16000


class WhisperXUnavailableError(RuntimeError):
    """Se lanza cuando WhisperX no está instalado o no se puede importar."""


class WhisperXTool:
    """Encapsula la detección y uso de WhisperX."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None
        self._model_name = None
        self._resolved_device: str | None = None

    @property
    def available(self) -> bool:
        try:
            import whisperx  # noqa: F401
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Preparación
    # ------------------------------------------------------------------
    def _ensure_import(self):
        """Importa whisperx o lanza WhisperXUnavailableError."""
        try:
            import whisperx  # noqa: PLC0415
            return whisperx
        except ImportError as exc:
            raise WhisperXUnavailableError(
                "WhisperX no está instalado. Ejecuta: pip install whisperx"
            ) from exc

    def _resolve_device(self, whisperx) -> str:
        """Resuelve el dispositivo, aplicando CUDA check y fallback configurable."""
        if self._resolved_device is not None:
            return self._resolved_device

        device = self.settings.device
        if device == "cuda":
            if self._cuda_available():
                self._resolved_device = "cuda"
            elif self.settings.allow_cpu_fallback:
                logger.warning("CUDA no disponible, usando CPU (fallback)")
                self._resolved_device = "cpu"
            else:
                raise WhisperXUnavailableError(
                    "CUDA no está disponible y ALLOW_CPU_FALLBACK=false. "
                    "No se procesará el vídeo con CPU."
                )
        else:
            self._resolved_device = "cpu"
        return self._resolved_device

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch  # noqa: PLC0415
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _load_model(self, whisperx, language: str | None):
        """Carga (o reutiliza) el pipeline WhisperX. Devuelve el pipeline."""
        if self._model is not None and self._model_name == language:
            return self._model

        device = self._resolve_device(whisperx)
        compute_type = self.settings.compute_type
        if device == "cpu":
            compute_type = "int8"

        logger.info(
            "loading whisperx model",
            model=_MODEL_NAME,
            device=device,
            compute_type=compute_type,
            language=language,
        )
        self._model = whisperx.load_model(
            _MODEL_NAME,
            device=device,
            compute_type=compute_type,
            language=language,
        )
        self._model_name = language
        return self._model

    # ------------------------------------------------------------------
    # Transcripción
    # ------------------------------------------------------------------
    def transcribe(
        self,
        audio_path: str | Path,
        *,
        language: str | None = None,
        diarization: bool = False,
    ) -> dict[str, Any]:
        """Transcribe un archivo de audio y devuelve un dict estructurado.

        El audio se pre-carga a numpy (16kHz mono) para evitar depender del
        decoding interno de whisperx (torchcodec), que puede estar roto.
        """
        whisperx = self._ensure_import()
        import numpy as np  # noqa: PLC0415

        pipeline = self._load_model(whisperx, language)
        device = self._resolved_device or self.settings.device
        audio = self._load_audio_numpy(audio_path)

        logger.info("transcribing audio", audio=str(audio_path), device=device)
        raw = pipeline.transcribe(audio, batch_size=16, language=language)
        result = dict(raw) if not isinstance(raw, dict) else raw

        # El language detectado se conserva, porque `align` lo pierde.
        detected_language = result.get("language") or language

        # --- Alineación de palabras (tolerante a fallos) ---
        try:
            if result.get("segments"):
                logger.info("aligning word timestamps")
                lang_code = detected_language or "en"
                model_a, metadata = whisperx.load_align_model(
                    language_code=lang_code,
                    device=device,
                )
                aligned = whisperx.align(
                    result["segments"],
                    model_a,
                    metadata,
                    audio,
                    device,
                    return_char_alignments=False,
                )
                result = dict(aligned)
                result["language"] = detected_language
        except Exception as exc:  # noqa: BLE001
            # Si la alineación falla (p. ej. modelo no accesible), se
            # mantienen los segmentos sin word-timestamps.
            logger.warning("alignment failed; usando solo segmentos", error=str(exc))

        # --- Diarización de hablantes (tolerante a fallos) ---
        # WhisperX 3.8: el pipeline recibe el audio (numpy), no el dict del
        # transcript. assign_word_speakers escribe speaker en segmentos y palabras.
        if diarization:
            try:
                from whisperx.diarize import (  # noqa: PLC0415
                    DiarizationPipeline,
                    assign_word_speakers,
                )

                token = getattr(self.settings, "hf_token", None)
                logger.info("running diarization", device=device, token=bool(token))
                # whisperx 3.8.6: DiarizationPipeline(model_name=None, token=None, device=..., cache_dir=None).
                # pyannote 4 Pipeline.from_pretrained takes token=, not use_auth_token=.
                diarize_model = DiarizationPipeline(device=device, token=token)
                diarize_df = diarize_model(audio)
                result = assign_word_speakers(diarize_df, result)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "diarization failed; continúo sin speakers", error=str(exc)
                )

        return self._normalize_result(result)

    # ------------------------------------------------------------------
    # Procesado de resultados
    # ------------------------------------------------------------------
    def _normalize_result(self, result: dict[str, Any]) -> dict[str, Any]:
        """Convierte el resultado de WhisperX a un formato estable."""
        segments: list[dict[str, Any]] = []
        words: list[dict[str, Any]] = []

        for seg in result.get("segments", []):
            segment = {
                "start": float(seg.get("start", 0.0) or 0.0),
                "end": float(seg.get("end", 0.0) or 0.0),
                "text": (seg.get("text") or "").strip(),
                "speaker": seg.get("speaker"),
            }
            segments.append(segment)

            for w in seg.get("words", []) or []:
                word = {
                    "word": w.get("word", ""),
                    "start": float(w.get("start", 0.0) or 0.0),
                    "end": float(w.get("end", 0.0) or 0.0),
                }
                if w.get("speaker"):
                    word["speaker"] = w.get("speaker")
                words.append(word)

        # `duration` no siempre está a nivel top; se calcula de los segmentos.
        duration = result.get("duration")
        if duration is None and segments:
            duration = max(s["end"] for s in segments)

        return {
            "language": result.get("language"),
            "duration": duration,
            "segments": segments,
            "words": words,
        }

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------
    @staticmethod
    def _load_audio_numpy(path: str | Path):
        """Carga un WAV (16-bit, 16kHz) a un array numpy float32 mono."""
        import numpy as np  # noqa: PLC0415
        import wave  # noqa: PLC0415

        with wave.open(str(path), "rb") as w:
            sample_rate = w.getframerate()
            sampler = w.getsampwidth()
            channels = w.getnchannels()
            frames = w.readframes(w.getnframes())

        if sample_rate != _SAMPLE_RATE:
            raise ValueError(
                f"Audio sample rate {sample_rate} != {_SAMPLE_RATE}; "
                "extrae el audio con ffmpeg a 16kHz"
            )
        if sampler != 2:  # PCM 16-bit
            raise ValueError(f"Audio no es PCM 16-bit (sampwidth={sampler})")

        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            data = data.reshape(-1, channels).mean(axis=1)
        return data
