"""Modelos de resultados devueltos por los jobs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class JobResult(BaseModel):
    """Resultado genérico de un job.

    `data` contiene el payload específico del tipo de job.
    """

    job_id: str
    type: str
    status: str = "completed"
    data: dict[str, Any] = Field(default_factory=dict)


class DownloadResult(BaseModel):
    """Resultado del job de descarga."""

    file_path: str
    filename: str
    size: int
    sha256: str


class TranscriptSegment(BaseModel):
    """Segmento de transcripción con timestamps."""

    start: float
    end: float
    text: str
    speaker: str | None = None


class TranscriptWord(BaseModel):
    """Palabra con timestamps (para subtítulos word-level)."""

    word: str
    start: float
    end: float
    speaker: str | None = None


class TranscriptResult(BaseModel):
    """Resultado completo de la transcripción."""

    language: str | None = None
    duration: float | None = None
    segments: list[TranscriptSegment] = Field(default_factory=list)
    words: list[TranscriptWord] = Field(default_factory=list)


class QACheck(BaseModel):
    """Resultado de una comprobación individual de QA."""

    name: str
    status: str  # PASS | FAIL
    actual: Any = None
    expected: Any = None
    message: str | None = None


class QAResult(BaseModel):
    """Resultado del job de QA técnico."""

    status: str  # PASS | FAIL
    checks: list[QACheck] = Field(default_factory=list)
