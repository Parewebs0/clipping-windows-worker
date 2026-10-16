"""Orquestador de ejecución de jobs.

Ciclo:
    obtener job
    → start
    → heartbeat periódico
    → ejecutar handler
    → resultado / fallo

El heartbeat se ejecuta en segundo plano durante jobs largos.
"""

from __future__ import annotations

import threading
import traceback
from typing import Any

from app.config import Settings
from app.models.job import Job
from app.services.api_client import BaseAPIClient
from app.services.download_errors import PermanentDownloadError, permanent_download_kind
from app.services.job_manager import JobManager
from app.utils.logging import get_logger

logger = get_logger("job_runner")


class JobHeartbeat:
    """Mantiene renovado el lease de un job mientras se ejecuta."""

    def __init__(
        self,
        api_client: BaseAPIClient,
        job_id: str,
        interval: float,
    ) -> None:
        self.api = api_client
        self.job_id = job_id
        self.interval = interval

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Arranca el heartbeat en segundo plano."""

        self._thread = threading.Thread(
            target=self._run,
            name=f"job-heartbeat-{self.job_id}",
            daemon=True,
        )

        self._thread.start()

        logger.debug(
            "job heartbeat started",
            job_id=self.job_id,
            interval=self.interval,
        )

    def stop(self) -> None:
        """Detiene el heartbeat."""

        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(
                timeout=min(
                    self.interval + 2.0,
                    10.0,
                )
            )

        logger.debug(
            "job heartbeat stopped",
            job_id=self.job_id,
        )

    def _run(self) -> None:
        """Loop interno del heartbeat."""

        while not self._stop_event.wait(self.interval):
            try:
                self.api.job_heartbeat(self.job_id)

                logger.debug(
                    "job heartbeat sent",
                    job_id=self.job_id,
                )

            except Exception as exc:  # noqa: BLE001
                # IMPORTANTE:
                # Un fallo temporal del heartbeat NO debe matar
                # WhisperX/FFmpeg.
                logger.warning(
                    "job heartbeat failed",
                    job_id=self.job_id,
                    error=str(exc),
                )


class JobRunner:
    """Ejecuta un job de principio a fin."""

    def __init__(
        self,
        settings: Settings,
        api_client: BaseAPIClient,
        job_manager: JobManager,
    ) -> None:
        self.settings = settings
        self.api = api_client
        self.job_manager = job_manager

    def run(
        self,
        job: Job,
    ) -> dict[str, Any]:
        """Ejecuta un job y devuelve el resultado."""

        logger.info(
            "job received",
            job_id=job.id,
            type=job.type,
        )

        # --------------------------------------------------------------
        # START
        # --------------------------------------------------------------

        started = self.api.start_job(job.id)

        if not started:
            raise RuntimeError(
                f"Could not start job {job.id}"
            )

        logger.info(
            "job started",
            job_id=job.id,
        )

        handler = None

        heartbeat = JobHeartbeat(
            api_client=self.api,
            job_id=job.id,
            interval=self.settings.heartbeat_interval,
        )

        try:
            # ----------------------------------------------------------
            # HEARTBEAT
            # ----------------------------------------------------------

            heartbeat.start()

            # ----------------------------------------------------------
            # HANDLER
            # ----------------------------------------------------------

            handler = self.job_manager.create_handler(job)

            # ----------------------------------------------------------
            # EXECUTE
            # ----------------------------------------------------------

            result = handler.execute()

            logger.info(
                "job execution completed",
                job_id=job.id,
            )

            # ----------------------------------------------------------
            # RESULT
            # ----------------------------------------------------------

            self.api.upload_result(
                job.id,
                result,
            )

            logger.info(
                "job result uploaded",
                job_id=job.id,
            )

            return result

        except Exception as exc:  # noqa: BLE001
            self._handle_failure(
                job,
                exc,
            )
            raise

        finally:
            # Detener heartbeat antes de terminar el job.
            heartbeat.stop()

            # Limpieza local.
            if handler is not None:
                try:
                    handler.cleanup()

                except Exception:  # noqa: BLE001
                    logger.exception(
                        "cleanup failed",
                        job_id=job.id,
                    )

    def _handle_failure(
        self,
        job: Job,
        exc: Exception,
    ) -> None:
        """Reporta el fallo al VPS."""

        error_trace = "".join(
            traceback.format_exception(
                type(exc),
                exc,
                exc.__traceback__,
            )
        )

        logger.error(
            "job failed",
            job_id=job.id,
            error=str(exc),
        )

        logger.debug(
            "job traceback",
            job_id=job.id,
            traceback=error_trace,
        )

        try:
            # The traceback stays in the debug log. The API stores
            # error_message and used to 422 when this was the full stack
            # (often with ANSI color from yt-dlp).
            message = str(exc)
            kind = permanent_download_kind(message)
            permanent = isinstance(exc, PermanentDownloadError) or (
                job.type == "download" and kind is not None
            )
            if permanent and kind and not message.startswith(f"{kind}:"):
                message = f"{kind}: {message}"
            self.api.fail_job(
                job.id,
                message,
                permanent=permanent,
            )

        except Exception as report_exc:  # noqa: BLE001
            logger.error(
                "failed to report job error",
                job_id=job.id,
                error=str(report_exc),
            )