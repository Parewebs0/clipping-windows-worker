"""Modelos del Job y sus estados."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Estados posibles de un job."""

    PENDING = "pending"
    CLAIMED = "claimed"
    DOWNLOADING = "downloading"
    PROCESSING = "processing"
    UPLOADING = "uploading"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Job(BaseModel):
    """Estructura común de un job recibido del servidor."""

    id: str
    type: str
    status: str = JobStatus.PENDING.value
    priority: int = Field(default=5, ge=0, le=100)  # API publish jobs use 10; headroom so a bump never crashes the loop
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None


class JobStatusUpdate(BaseModel):
    """Payload para informar al servidor de un cambio de estado."""

    job_id: str
    status: JobStatus
    message: str | None = None
    progress: float | None = Field(default=None, ge=0.0, le=1.0)
