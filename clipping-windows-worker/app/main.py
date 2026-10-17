"""Punto de entrada del Windows GPU Worker.

Lifecycle:
    1. Cargar configuración.
    2. Crear directorios.
    3. Configurar logs.
    4. Comprobar FFmpeg/FFprobe.
    5. Detectar GPU/CUDA.
    6. Comprobar WhisperX.
    7. Conectarse a la API central.
    8. Polling de jobs.
    9. Procesar jobs y mantener su lease mediante heartbeat.
"""

from __future__ import annotations

import os
import time

from app.config import Settings, get_settings
from app.models.job import Job
from app.services.api_client import (
    APIClient,
    BaseAPIClient,
    MockAPIClient,
)
from app.services.job_manager import JobManager
from app.services.job_runner import JobRunner
from app.services.retention import sweep_media
from app.services.storage_cleanup import run_cleanup
from app.services.system_info import SystemInfoService
from app.utils.logging import (
    configure_logging,
    get_logger,
)

logger = get_logger("worker")


class Worker:
    """Orquesta el arranque y el loop principal del worker."""

    def __init__(
        self,
        settings: Settings,
        api_client: BaseAPIClient,
    ) -> None:
        self.settings = settings
        self.api = api_client

        self.system_info = SystemInfoService(
            settings
        )

        self.job_manager = JobManager(
            settings
        )

        self.job_runner = JobRunner(
            settings,
            api_client,
            self.job_manager,
        )

    # ------------------------------------------------------------------
    # Arranque
    # ------------------------------------------------------------------

    def startup(self) -> None:
        """Ejecuta las comprobaciones iniciales."""

        logger.info(
            "Worker starting"
        )

        apply_nice(self.settings.nice_level)

        logger.info(
            "Worker ID",
            worker_id=self.settings.worker_id,
        )

        system = self.system_info.get_system_info()
        gpu = self.system_info.get_gpu_info()
        tools = self.system_info.get_tools_info()

        logger.info(
            "OS",
            os=system.os,
            version=system.os_version,
        )

        logger.info(
            "CPU",
            cpu=system.cpu,
        )

        logger.info(
            "RAM total (GB)",
            ram=system.ram_total_gb,
        )

        if gpu.available:
            logger.info(
                "GPU detected",
                name=gpu.name,
                vram_gb=gpu.vram_total_gb,
            )

            logger.info(
                "CUDA available",
                version=gpu.cuda_version,
            )

        else:
            logger.warning(
                "No GPU/CUDA detected",
                name=gpu.name,
            )

        logger.info(
            "FFmpeg available",
            available=tools.ffmpeg,
        )

        logger.info(
            "FFprobe available",
            available=tools.ffprobe,
        )

        logger.info(
            "WhisperX available",
            available=tools.whisperx,
        )

        logger.info(
            "Worker ready",
            worker_id=self.settings.worker_id,
            capabilities=list(
                self.job_manager.handlers.keys()
            ),
        )

    # ------------------------------------------------------------------
    # Loop principal
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Bucle infinito de polling."""

        logger.info(
            "Waiting for jobs..."
        )
        last_sweep = 0.0
        last_storage_cleanup = 0.0

        while True:
            try:
                if (
                    self.settings.media_retention_enabled
                    and time.monotonic() - last_sweep >= self.settings.retention_sweep_interval
                ):
                    last_sweep = time.monotonic()
                    sweep_media(self.settings)

                if (
                    self.settings.storage_cleanup_enabled
                    and self.settings.storage_root
                    and time.monotonic() - last_storage_cleanup >= self.settings.storage_cleanup_interval
                ):
                    last_storage_cleanup = time.monotonic()
                    run_cleanup(self.settings, self.api)

                job = self.api.get_next_job()

                if job is not None:
                    self._process_job(job)

                else:
                    time.sleep(
                        self.settings.poll_interval
                    )

            except KeyboardInterrupt:
                logger.info(
                    "Worker stopped by user"
                )
                break

            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "worker loop error",
                    error=str(exc),
                )

                time.sleep(
                    self.settings.poll_interval
                )

    def _process_job(
        self,
        job: Job,
    ) -> None:
        """Procesa un job."""

        logger.info(
            "job received",
            job_id=job.id,
            job_type=job.type,
        )

        try:
            result = self.job_runner.run(
                job
            )

            logger.info(
                "job processing finished",
                job_id=job.id,
                result_keys=list(result.keys()),
            )

        except Exception as exc:  # noqa: BLE001
            # JobRunner ya reporta el error al VPS.
            logger.error(
                "job processing failed",
                job_id=job.id,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Cierra recursos del worker."""

        try:
            self.api.close()
        finally:
            logger.info(
                "Worker stopped"
            )


def apply_nice(level: int) -> int:
    """Lower our CPU priority (children such as ffmpeg inherit it)."""
    if level <= 0 or not hasattr(os, "nice"):
        return 0
    try:
        return os.nice(level)
    except OSError as exc:
        logger.warning("nice failed", error=str(exc))
        return 0


def _build_api_client(
    settings: Settings,
) -> BaseAPIClient:
    """Construye cliente real o mock."""

    if settings.api_base_url.lower().startswith(
        "mock"
    ):
        logger.info(
            "Using MockAPIClient (development mode)"
        )

        return MockAPIClient(
            settings
        )

    return APIClient(
        settings
    )


def main() -> None:
    """Función principal invocada desde run.py."""

    settings = get_settings()

    settings.ensure_directories()

    configure_logging(
        settings.log_level
    )

    api_client = _build_api_client(
        settings
    )

    worker = Worker(
        settings,
        api_client,
    )

    try:
        worker.startup()
        worker.run()

    except KeyboardInterrupt:
        pass

    finally:
        worker.shutdown()


if __name__ == "__main__":
    main()