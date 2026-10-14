"""MOLI-54 — transcribe: initial_prompt goes through pipeline.options (whisperx 3.8.6)."""

import dataclasses
from types import SimpleNamespace

from app.tools.whisperx import WhisperXTool


def _tool(monkeypatch, pipeline):
    class FakeWX:
        @staticmethod
        def load_model(*a, **k):
            return pipeline

        @staticmethod
        def align(*a, **k):
            raise RuntimeError("no align in test")

    import app.tools.whisperx as wx
    monkeypatch.setattr(wx.WhisperXTool, "_ensure_import", lambda self: FakeWX)
    monkeypatch.setattr(wx.WhisperXTool, "_resolve_device", lambda self, wx: "cpu")
    monkeypatch.setattr(wx.WhisperXTool, "_load_audio_numpy", lambda self, p: [0.0] * 16)
    settings = SimpleNamespace(compute_type="int8", device="cpu", allow_cpu_fallback=True)
    return WhisperXTool(settings=settings)


@dataclasses.dataclass
class _Options:
    initial_prompt: str | None = None
    beam_size: int = 5


def test_initial_prompt_is_set_on_options_and_restored(monkeypatch):
    """whisperx 3.8.6 reads initial_prompt from pipeline.options, not transcribe()."""
    captured = {}
    original = _Options()

    class FakePipeline:
        def __init__(self):
            self.options = original

        def transcribe(self, audio, **kwargs):
            captured["kwargs"] = kwargs
            captured["during"] = self.options.initial_prompt
            return {"language": "es", "segments": [{"start": 0.0, "end": 1.0, "text": "hola"}]}

    pipeline = FakePipeline()
    tool = _tool(monkeypatch, pipeline)
    out = tool.transcribe("audio.wav", language="es", initial_prompt="BOXABL CASITA")
    assert out["language"] == "es"
    assert captured["during"] == "BOXABL CASITA"
    assert "initial_prompt" not in captured["kwargs"]
    assert pipeline.options is original
    assert original.initial_prompt is None


def test_initial_prompt_restored_when_transcribe_fails(monkeypatch):
    original = _Options(initial_prompt="keep-me")

    class FakePipeline:
        def __init__(self):
            self.options = original

        def transcribe(self, audio, **kwargs):
            assert self.options.initial_prompt == "BOXABL"
            raise RuntimeError("vad failed")

    pipeline = FakePipeline()
    tool = _tool(monkeypatch, pipeline)
    try:
        tool.transcribe("audio.wav", language="en", initial_prompt="BOXABL")
    except RuntimeError as exc:
        assert str(exc) == "vad failed"
    else:
        raise AssertionError("transcribe should have raised")
    assert pipeline.options is original
    assert original.initial_prompt == "keep-me"


def test_initial_prompt_skipped_when_options_are_missing(monkeypatch):
    """No dataclass options: transcribe still runs, and the kwarg is not passed."""
    captured = {}

    class FakePipeline:
        def transcribe(self, audio, **kwargs):
            captured.update(kwargs)
            return {"language": "en", "segments": []}

    tool = _tool(monkeypatch, FakePipeline())
    out = tool.transcribe("audio.wav", language="en", initial_prompt="BOXABL")
    assert out["language"] == "en"
    assert "initial_prompt" not in captured


def test_transcribe_job_normalises_the_prompt(monkeypatch, tmp_path):
    """The job strips/limits initial_prompt before calling WhisperX."""
    import app.jobs.transcribe as tmod

    seen = {}

    def fake_whisperx(self=None, audio_path=None, **kw):
        seen.update(kw)
        return {"language": "es", "duration": 1.0, "segments": [], "words": []}

    job = SimpleNamespace(id="t1", payload={"video": "x.mp4",
                                            "initial_prompt": "  " + "BOXABL " * 300 + "  "})
    tj = tmod.TranscribeJob.__new__(tmod.TranscribeJob)
    tj.job = job
    tj.logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
    tj.directory = SimpleNamespace(temp=tmp_path, output=tmp_path, input=tmp_path, root=tmp_path)
    monkeypatch.setattr(tmod.TranscribeJob, "_resolve_input", lambda self, p: tmp_path / "x.mp4")
    (tmp_path / "x.mp4").write_bytes(b"fake")  # exists → passes the check
    import app.tools.ffprobe as fp
    import app.tools.ffmpeg as ff
    tj.settings = SimpleNamespace(ffmpeg_path=None, ffprobe_path=None)  # constructs FFprobe/FFmpeg
    monkeypatch.setattr(fp.FFprobeTool, "get_audio_stream", lambda self, src: object())
    monkeypatch.setattr(ff.FFmpegTool, "extract_audio", lambda self, src, dst: dst)
    monkeypatch.setattr(tmod, "WhisperXTool", lambda settings: SimpleNamespace(
        transcribe=lambda *a, **k: fake_whisperx(*a, **k)))
    result = tj.execute()
    assert result["language"] == "es"
    prompt = seen.get("initial_prompt")
    assert prompt is not None
    assert prompt.startswith("BOXABL BOXABL")
    assert len(prompt) <= 900