"""Tests de los modelos Pydantic."""

from __future__ import annotations

from app.models.job import Job, JobStatus
from app.models.results import QACheck, QAResult, TranscriptResult
from app.models.worker import GPUInfo, Heartbeat, SystemInfo, ToolsInfo


def test_job_defaults() -> None:
    job = Job(id="job_1", type="health")
    assert job.status == "pending"
    assert job.priority == 5
    assert job.payload == {}


def test_job_status_enum() -> None:
    assert JobStatus.PENDING.value == "pending"
    assert JobStatus.COMPLETED.value == "completed"
    assert JobStatus.FAILED.value == "failed"


def test_job_from_dict() -> None:
    data = {
        "id": "job_123",
        "type": "transcribe",
        "status": "pending",
        "priority": 5,
        "payload": {"video_path": "x.mp4"},
    }
    job = Job.model_validate(data)
    assert job.id == "job_123"
    assert job.type == "transcribe"
    assert job.payload["video_path"] == "x.mp4"


def test_gpu_info_defaults() -> None:
    gpu = GPUInfo()
    assert gpu.available is False
    assert gpu.cuda_available is False


def test_system_info() -> None:
    info = SystemInfo(os="Windows", cpu="AMD Ryzen 7 5800X")
    assert info.os == "Windows"
    assert info.cpu == "AMD Ryzen 7 5800X"


def test_tools_info() -> None:
    tools = ToolsInfo(ffmpeg=True, ffprobe=True, whisperx=False)
    assert tools.ffmpeg is True
    assert tools.whisperx is False


def test_heartbeat() -> None:
    hb = Heartbeat(worker_id="w1", current_job="job_1")
    assert hb.status == "online"
    assert hb.current_job == "job_1"


def test_transcript_result() -> None:
    result = TranscriptResult(
        language="es",
        duration=10.0,
        segments=[{"start": 0.0, "end": 5.0, "text": "Hola"}],
        words=[{"word": "Hola", "start": 0.0, "end": 0.5}],
    )
    assert result.language == "es"
    assert result.segments[0].text == "Hola"
    assert result.words[0].word == "Hola"


def test_qa_result() -> None:
    result = QAResult(
        status="PASS",
        checks=[QACheck(name="duration", status="PASS", actual=42.1)],
    )
    assert result.status == "PASS"
    assert result.checks[0].name == "duration"


def test_job_accepts_publish_priority_above_10() -> None:
    # The API queues publish at 10 and once shipped 20; a priority above the
    # old le=10 crashed the worker loop and left jobs stuck in "assigned".
    for p in (10, 11, 20, 100):
        job = Job.model_validate({"id": "j", "type": "publish", "priority": p})
        assert job.priority == p
