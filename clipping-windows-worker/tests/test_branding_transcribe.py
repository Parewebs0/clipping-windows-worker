"""MOLI-54 — transcribe: initial_prompt accepted (and tolerated when unsupported)."""

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


def test_initial_prompt_reaches_whisperx(monkeypatch):
    captured = {}

    def fake_pipeline_transcribe(audio, **kwargs):
        captured.update(kwargs)
        return {"language": "es", "segments": [{"start": 0.0, "end": 1.0, "text": "hola"}]}

    class FakePipeline:
        transcribe = staticmethod(fake_pipeline_transcribe)

    tool = _tool(monkeypatch, FakePipeline())
    out = tool.transcribe("audio.wav", language="es", initial_prompt="BOXABL CASITA")
    assert out["language"] == "es"
    assert captured.get("initial_prompt") == "BOXABL CASITA"


def test_initial_prompt_typeerror_fallback(monkeypatch):
    """Older fork without initial_prompt: second attempt without it succeeds."""
    calls = []

    def fake_pipeline_transcribe(audio, batch_size=None, language=None, initial_prompt=None):
        calls.append(initial_prompt)
        if initial_prompt is not None:
            raise TypeError("unexpected keyword argument 'initial_prompt'")
        return {"language": "en", "segments": []}

    class FakePipeline:
        transcribe = staticmethod(fake_pipeline_transcribe)

    tool = _tool(monkeypatch, FakePipeline())
    out = tool.transcribe("audio.wav", language="en", initial_prompt="BOXABL")
    assert out["language"] == "en"
    assert calls == ["BOXABL", None]


def test_transcribe_job_normalises_the_prompt(monkeypatch):
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
    tj.directory = SimpleNamespace(temp=__import__("pathlib").Path("/tmp/t1"),
                                   output=__import__("pathlib").Path("/tmp/t1"),
                                   input=__import__("pathlib").Path("/tmp/t1"),
                                   root=__import__("pathlib").Path("/tmp/t1"))
    monkeypatch.setattr(tmod.TranscribeJob, "_resolve_input",
                        lambda self, p: __import__("pathlib").Path("/tmp/t1/x.mp4"))
    (tj.directory.temp / "x.mp4").write_bytes(b"fake")  # exists → passes the check
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