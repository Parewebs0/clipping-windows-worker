"""Job de QA técnico de vídeo con FFprobe."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from app.jobs.base import BaseJob
from app.tools.ffprobe import FFprobeTool
from app.utils.clip_storage import ClipStorage, ClipStorageError


class QAJob(BaseJob):
    """Valida propiedades técnicas de un vídeo contra unas reglas."""

    type = "qa"

    def execute(self) -> dict[str, Any]:
        payload = self.job.payload
        # Aceptamos `file_path` (canónico VPS) y `video` (legacy)
        video = payload.get("file_path") or payload.get("video")
        rules = payload.get("rules") or {}

        if not video:
            raise ValueError("payload.file_path is required")

        source = self._resolve_input(video)
        if not source.exists():
            raise FileNotFoundError(f"Video not found: {source}")

        ffprobe = FFprobeTool(self.settings)
        checks: list[dict[str, Any]] = []

        # --- Duración ---
        duration = ffprobe.get_duration(source)
        min_dur = rules.get("min_duration")
        max_dur = rules.get("max_duration")
        if min_dur is not None or max_dur is not None:
            ok = True
            if min_dur is not None and (duration is None or duration < min_dur):
                ok = False
            if max_dur is not None and (duration is None or duration > max_dur):
                ok = False
            checks.append(
                {
                    "name": "duration",
                    "status": "PASS" if ok else "FAIL",
                    "actual": duration,
                    "expected": f"{min_dur}-{max_dur}" if min_dur is not None or max_dur is not None else None,
                }
            )

        # --- Resolución ---
        resolution = ffprobe.get_resolution(source)
        width = rules.get("width")
        height = rules.get("height")
        if width is not None or height is not None:
            ok = True
            if resolution is None:
                ok = False
            else:
                if width is not None and resolution[0] != width:
                    ok = False
                if height is not None and resolution[1] != height:
                    ok = False
            checks.append(
                {
                    "name": "resolution",
                    "status": "PASS" if ok else "FAIL",
                    "actual": f"{resolution[0]}x{resolution[1]}" if resolution else None,
                    "expected": f"{width}x{height}" if width is not None or height is not None else None,
                }
            )

        # --- FPS ---
        fps = ffprobe.get_fps(source)
        min_fps = rules.get("min_fps")
        if min_fps is not None:
            ok = fps is not None and fps >= min_fps
            checks.append(
                {
                    "name": "fps",
                    "status": "PASS" if ok else "FAIL",
                    "actual": fps,
                    "expected": f">={min_fps}",
                }
            )

        # --- Audio presente ---
        if rules.get("require_audio", False):
            has_audio = ffprobe.get_audio_stream(source) is not None
            checks.append(
                {
                    "name": "audio",
                    "status": "PASS" if has_audio else "FAIL",
                    "actual": has_audio,
                    "expected": True,
                }
            )

        # --- Codec de vídeo ---
        codec = rules.get("codec")
        if codec:
            stream = ffprobe.get_video_stream(source)
            actual_codec = stream.get("codec_name") if stream else None
            ok = actual_codec == codec
            checks.append(
                {
                    "name": "codec",
                    "status": "PASS" if ok else "FAIL",
                    "actual": actual_codec,
                    "expected": codec,
                }
            )

        overall = "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL"
        self.logger.info("qa completed", status=overall, checks=len(checks))

        # --------------------------------------------------------------
        # Step 18 (architecture_flow.md):
        # Si QA = PASS, copiar el clip a
        #   <clip_storage_root>/<campaign_id>/pending_upload/<clip_id>.mp4
        # y reportar el path nuevo al VPS vía `final_path_worker` en el
        # payload del resultado. El VPS lo persiste y marca
        # `clips.location = "pending_upload"` automáticamente desde
        # `on_qa_completed`.
        #
        # Si el payload no incluye `clip_id` o `campaign_id` (jobs QA
        # manuales o tests legacy), el QA sigue siendo válido pero sin
        # storage copy — el clip queda solo en su carpeta de output.
        # --------------------------------------------------------------

        payload_dict = payload if isinstance(payload, dict) else {}
        clip_id = (
            payload_dict.get("clip_id")
            or payload_dict.get("clip_uuid")
            or payload_dict.get("clipId")
        )
        campaign_id = payload_dict.get("campaign_id")

        result: dict[str, Any] = {
            "status": overall,
            "checks": checks,
            "duration_seconds": duration,
        }

        if overall == "PASS" and clip_id and campaign_id is not None:
            final_path = self._copy_to_pending_upload(
                source=source,
                clip_id=str(clip_id),
                campaign_id=campaign_id,
            )
            if final_path is not None:
                result["final_path_worker"] = str(final_path)
                result["source_moved"] = True
            else:
                result["source_moved"] = False
        else:
            # No intentamos storage copy. Lo dejamos explícito.
            result["source_moved"] = False
            if overall == "PASS" and not clip_id:
                self.logger.warning(
                    "qa pass but no clip_id in payload; skipping storage copy"
                )
            if overall == "PASS" and campaign_id is None:
                self.logger.warning(
                    "qa pass but no campaign_id in payload; skipping storage copy"
                )

        return result

    def _resolve_input(self, path: str) -> Path:
        """Resuelve una ruta de entrada, absoluta o relativa al job."""
        p = Path(path)
        if p.is_absolute():
            return p
        candidates = [
            self.directory.input / p,
            self.directory.root / p,
            Path(self.settings.working_directory) / p,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _copy_to_pending_upload(
        self,
        source: Path,
        clip_id: str,
        campaign_id: int | str,
    ) -> Optional[Path]:
        """Copia ``source`` a pending_upload/ del storage por campaña.

        Política de errores (de architecture_flow.md):
          - NO borrar el original. Siempre copy, never move.
          - Si la copia falla, loggear warning y continuar. NO fallar el QA:
            la BD se actualizará sin path nuevo (`location=pending_upload`
            pero `final_path_worker=NULL`).
          - Idempotente: si el destino ya existe, se sobreescribe.
        """
        try:
            storage = ClipStorage.for_settings(self.settings, campaign_id)
            destination = storage.copy_to_pending_upload(
                src=source,
                clip_id=clip_id,
            )
            self.logger.info(
                "clip %s copied to pending_upload (campaign=%s)",
                clip_id, campaign_id,
                destination=str(destination),
            )
            return destination
        except ClipStorageError as exc:
            self.logger.warning(
                "qa: invalid storage args; continuing without copy",
                clip_id=clip_id,
                campaign_id=campaign_id,
                error=str(exc),
            )
            return None
        except FileNotFoundError as exc:
            self.logger.warning(
                "qa: source clip missing for storage copy; continuing without copy",
                clip_id=clip_id,
                error=str(exc),
            )
            return None
        except OSError as exc:
            self.logger.warning(
                "qa: storage copy failed (OS error); continuing without copy",
                clip_id=clip_id,
                campaign_id=campaign_id,
                error=str(exc),
            )
            return None
