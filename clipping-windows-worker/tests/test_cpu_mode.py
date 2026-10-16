"""CPU fallback worker: model selection, diarization switch, nice, retention."""
from __future__ import annotations

import os
import sys
import time
import types
from pathlib import Path

from app.config import Settings
from app.services.retention import sweep_media
from app.tools.whisperx import WhisperXTool


def _cpu(tmp_path: Path, **kw) -> Settings:
    base = dict(
        working_directory=str(tmp_path / "data"),
        clip_storage_root=str(tmp_path / "clips"),
        device="cpu",
        whisper_model="small",
        diarization_enabled=False,
        media_retention_enabled=True,
        media_retention_hours=24,
    )
    base.update(kw)
    return Settings(**base)


def test_defaults_keep_gpu_worker_behaviour(tmp_path):
    s = Settings(working_directory=str(tmp_path))
    assert s.whisper_model == "large-v3"
    assert s.diarization_enabled is True
    assert s.nice_level == 0
    assert s.media_retention_enabled is False


def test_cpu_loads_small_int8(tmp_path, monkeypatch):
    calls = {}
    fake = types.SimpleNamespace(
        load_model=lambda name, **kw: calls.update(name=name, **kw) or object()
    )
    monkeypatch.setitem(sys.modules, "whisperx", fake)
    tool = WhisperXTool(_cpu(tmp_path, compute_type="float16"))
    tool._load_model(fake, "es")
    assert calls["name"] == "small"
    assert calls["device"] == "cpu"
    assert calls["compute_type"] == "int8"


def test_diarization_off_overrides_payload(tmp_path, monkeypatch):
    from app.jobs import transcribe as tmod

    seen = {}

    class FakeTool:
        def __init__(self, settings):
            pass

        def transcribe(self, path, **kw):
            seen.update(kw)
            return {"segments": [], "language": "es"}

    video = tmp_path / "v.mp4"
    video.write_bytes(b"x" * 10)
    monkeypatch.setattr(tmod, "WhisperXTool", FakeTool)

    class FakeProbe:
        def __init__(self, settings):
            pass

        def get_audio_stream(self, path):
            return {"codec_name": "aac"}

    class FakeFfmpeg:
        def __init__(self, settings):
            pass

        def extract_audio(self, src, dst):
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            Path(dst).write_bytes(b"RIFF")

    monkeypatch.setattr(tmod, "FFprobeTool", FakeProbe)
    monkeypatch.setattr(tmod, "FFmpegTool", FakeFfmpeg)
    s = _cpu(tmp_path)
    s.ensure_directories()
    job = types.SimpleNamespace(id="j1", type="transcribe",
                                payload={"video_path": str(video), "diarization": True})
    handler = tmod.TranscribeJob(s, job)
    handler.execute()
    assert seen.get("diarization") is False


def test_retention_deletes_only_old_media(tmp_path):
    s = _cpu(tmp_path)
    s.ensure_directories()
    old = s.downloads_dir / "a1" / "old.mp4"
    new = s.outputs_dir / "new.mp4"
    clip = Path(s.clip_storage_root) / "7" / "pending_upload" / "c.mp4"
    log = s.logs_dir / "worker.log"
    for p in (old, new, clip, log):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    past = time.time() - 25 * 3600
    for p in (old, clip, log):
        os.utime(p, (past, past))
    out = sweep_media(s)
    assert out["removed"] == 2
    assert not old.exists() and not clip.exists()
    assert not old.parent.exists()
    assert new.exists()
    assert log.exists()  # logs are never swept


def test_apply_nice(monkeypatch):
    from app import main
    seen = []
    monkeypatch.setattr(main.os, "nice", lambda n: seen.append(n) or n)
    assert main.apply_nice(0) == 0 and seen == []
    main.apply_nice(10)
    assert seen == [10]
