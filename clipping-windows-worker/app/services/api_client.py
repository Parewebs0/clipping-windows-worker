"""Cliente para comunicarse con la API central del clipping system."""

from __future__ import annotations

from typing import Any

import httpx
import structlog

from app.config import Settings
from app.models.job import Job
from app.models.worker import Heartbeat, WorkerRegistration
from app.services.download_errors import worker_fail_body

logger = structlog.get_logger()


class BaseAPIClient:
    """Interfaz común para cliente real y mock."""

    def register_worker(self, registration: WorkerRegistration) -> bool:
        raise NotImplementedError

    def heartbeat(self, heartbeat: Heartbeat) -> bool:
        raise NotImplementedError

    def get_next_job(self) -> Job | None:
        raise NotImplementedError

    def start_job(self, job_id: str) -> bool:
        raise NotImplementedError

    def job_heartbeat(self, job_id: str) -> bool:
        raise NotImplementedError

    def upload_result(
        self,
        job_id: str,
        result: dict[str, Any],
    ) -> bool:
        raise NotImplementedError

    def fail_job(
        self,
        job_id: str,
        error: str,
        *,
        permanent: bool = False,
    ) -> bool:
        raise NotImplementedError

    def close(self) -> None:
        pass


class APIClient(BaseAPIClient):
    """Cliente HTTP real contra la API central."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        self.client = httpx.Client(
            base_url=settings.api_base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {settings.api_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=httpx.Timeout(
                connect=10.0,
                read=30.0,
                write=30.0,
                pool=10.0,
            ),
        )

    def register_worker(
        self,
        registration: WorkerRegistration,
    ) -> bool:
        response = self.client.post(
            "/worker/register",
            json=registration.model_dump(mode="json"),
        )
        response.raise_for_status()

        logger.info(
            "worker registered",
            worker_id=self.settings.worker_id,
        )

        return True

    def heartbeat(self, heartbeat: Heartbeat) -> bool:
        response = self.client.post(
            "/worker/heartbeat",
            json=heartbeat.model_dump(mode="json"),
        )
        response.raise_for_status()

        logger.debug(
            "worker heartbeat sent",
            worker_id=self.settings.worker_id,
        )

        return True

    def get_next_job(self) -> Job | None:
        response = self.client.get(
            "/worker/jobs/next",
            params={
                "worker_id": self.settings.worker_id,
            },
        )

        if response.status_code == 204:
            return None

        response.raise_for_status()

        data = response.json()

        if not data:
            return None

        # La API central utiliza "job_type",
        # mientras que el modelo Job del worker utiliza "type".
        if "job_type" in data and "type" not in data:
            data["type"] = data["job_type"]

        return Job.model_validate(data)

    def start_job(self, job_id: str) -> bool:
        response = self.client.post(
            f"/worker/jobs/{job_id}/start",
            params={
                "worker_id": self.settings.worker_id,
            },
        )
        response.raise_for_status()

        logger.info(
            "job started",
            job_id=job_id,
            worker_id=self.settings.worker_id,
        )

        return True

    def job_heartbeat(self, job_id: str) -> bool:
        response = self.client.post(
            f"/worker/jobs/{job_id}/heartbeat",
            params={
                "worker_id": self.settings.worker_id,
            },
        )
        response.raise_for_status()

        logger.debug(
            "job heartbeat sent",
            job_id=job_id,
            worker_id=self.settings.worker_id,
        )

        return True

    def upload_result(
        self,
        job_id: str,
        result: dict[str, Any],
    ) -> bool:
        # El VPS espera el payload envuelto bajo la clave "result"
        # (ver `WorkerResultCreate` en /openapi.json del backend).
        response = self.client.post(
            f"/worker/jobs/{job_id}/result",
            params={
                "worker_id": self.settings.worker_id,
            },
            json={"result": result},
        )
        response.raise_for_status()

        logger.info(
            "job result uploaded",
            job_id=job_id,
            worker_id=self.settings.worker_id,
        )

        return True

    def fail_job(
        self,
        job_id: str,
        error: str,
        *,
        permanent: bool = False,
    ) -> bool:
        response = self.client.post(
            f"/worker/jobs/{job_id}/fail",
            params={
                "worker_id": self.settings.worker_id,
            },
            json=worker_fail_body(error, permanent=permanent),
        )
        response.raise_for_status()

        logger.error(
            "job marked as failed",
            job_id=job_id,
            worker_id=self.settings.worker_id,
        )

        return True

    def close(self) -> None:
        self.client.close()


class MockAPIClient(BaseAPIClient):
    """Cliente mock utilizado para tests locales."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        self.jobs: list[Job] = []
        self.started_jobs: set[str] = set()
        self.completed_jobs: set[str] = set()
        self.failed_jobs: set[str] = set()

        self.results: dict[str, dict[str, Any]] = {}
        self.errors: dict[str, str] = {}
        self.permanent_failures: set[str] = set()

    def register_worker(
        self,
        registration: WorkerRegistration,
    ) -> bool:
        logger.info(
            "mock: worker registered",
            worker_id=self.settings.worker_id,
        )
        return True

    def heartbeat(self, heartbeat: Heartbeat) -> bool:
        logger.debug(
            "mock: worker heartbeat",
            worker_id=self.settings.worker_id,
        )
        return True

    def get_next_job(self) -> Job | None:
        for job in self.jobs:
            if (
                job.id not in self.started_jobs
                and job.id not in self.completed_jobs
                and job.id not in self.failed_jobs
            ):
                return job

        return None

    def start_job(self, job_id: str) -> bool:
        self.started_jobs.add(job_id)

        logger.info(
            "mock: job started",
            job_id=job_id,
        )

        return True

    def job_heartbeat(self, job_id: str) -> bool:
        logger.debug(
            "mock: job heartbeat",
            job_id=job_id,
        )

        return True

    def upload_result(
        self,
        job_id: str,
        result: dict[str, Any],
    ) -> bool:
        self.results[job_id] = result
        self.completed_jobs.add(job_id)

        logger.info(
            "mock: result uploaded",
            job_id=job_id,
        )

        return True

    def fail_job(
        self,
        job_id: str,
        error: str,
        *,
        permanent: bool = False,
    ) -> bool:
        self.errors[job_id] = error
        self.failed_jobs.add(job_id)
        if permanent:
            self.permanent_failures.add(job_id)

        logger.info(
            "mock: job failed",
            job_id=job_id,
            error=error,
        )

        return True

    def enqueue_health(self) -> None:
        self.jobs.append(
            Job(
                id="job_health",
                type="health",
                status="pending",
                priority=5,
                payload={},
                created_at=None,
            )
        )

    def enqueue_download(
        self,
        payload: dict[str, Any],
    ) -> None:
        self.jobs.append(
            Job(
                id=f"job_download_{len(self.jobs) + 1}",
                type="download",
                status="pending",
                priority=5,
                payload=payload,
                created_at=None,
            )
        )

    def enqueue_transcribe(
        self,
        payload: dict[str, Any],
    ) -> None:
        self.jobs.append(
            Job(
                id=f"job_transcribe_{len(self.jobs) + 1}",
                type="transcribe",
                status="pending",
                priority=5,
                payload=payload,
                created_at=None,
            )
        )

    def enqueue_render(
        self,
        payload: dict[str, Any],
    ) -> None:
        self.jobs.append(
            Job(
                id=f"job_render_{len(self.jobs) + 1}",
                type="render",
                status="pending",
                priority=5,
                payload=payload,
                created_at=None,
            )
        )

    def enqueue_qa(
        self,
        payload: dict[str, Any],
    ) -> None:
        self.jobs.append(
            Job(
                id=f"job_qa_{len(self.jobs) + 1}",
                type="qa",
                status="pending",
                priority=5,
                payload=payload,
                created_at=None,
            )
        )

    def close(self) -> None:
        pass