"""Registro de handlers de jobs.

Añadir un nuevo tipo de job solo requiere registrar su clase aquí.
No es necesario modificar el runner ni el worker loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Type

from app.config import Settings
from app.jobs.base import BaseJob
from app.jobs.download import DownloadJob
from app.jobs.health import HealthJob
from app.jobs.publish import PublishJob
from app.jobs.qa import QAJob
from app.jobs.render import RenderJob
from app.jobs.transcribe import TranscribeJob
from app.utils.logging import get_logger

if TYPE_CHECKING:
    from app.models.job import Job

logger = get_logger("job_manager")

JOB_HANDLERS: dict[str, Type[BaseJob]] = {
    "health": HealthJob,
    "download": DownloadJob,
    "transcribe": TranscribeJob,
    "render": RenderJob,
    "qa": QAJob,
    "publish": PublishJob,
}


class JobManager:
    """Resuelve el handler adecuado para cada tipo de job."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.handlers = dict(JOB_HANDLERS)

    def register(self, job_type: str, handler: Type[BaseJob]) -> None:
        self.handlers[job_type] = handler
        logger.info("registered job handler", job_type=job_type)

    def create_handler(self, job: "Job") -> BaseJob:
        handler_cls = self.handlers.get(job.type)
        if handler_cls is None:
            raise ValueError(f"Unknown job type: {job.type}")
        return handler_cls(self.settings, job)

    def supports(self, job_type: str) -> bool:
        return job_type in self.handlers
