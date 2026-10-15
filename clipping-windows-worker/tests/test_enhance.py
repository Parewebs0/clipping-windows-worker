"""MOLI-52: speech enhancement before loudnorm. Default engine is none."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from app.jobs.render import RenderJob
from app.tools.enhance import (
    EnhanceError,
    _enhance_resemble,
    atten_lim_db,
    enhance_decision,
    estimate_snr_db,
    parse_enhance,
    read_wav,
    run_enhance,
    write_wav,
)
from app.tools.ffmpeg import parse_loudnorm_json
from tests.test_render_rules_v2 import _job, _logo, _settings, _src


def _integrated_loudness(video: Path) -> tuple[float, float]:
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(video), "-vn",
         "-af", "loudnorm=print_format=json", "-f", "null", "-"],
        capture_output=True, text=True, check=True)
    data = parse_loudnorm_json(out.stderr + out.stdout)
    assert data, out.stderr
    return float(data["input_i"]), float(data["input_tp"])


def _bursts(sample_rate: int = 48000, seconds: float = 4.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Voiced bursts with pauses, plus stationary noise. Returns (clean, noisy)."""
    n = int(sample_rate * seconds)
    t = np.arange(n) / sample_rate
    speech = np.zeros(n, dtype=np.float32)
    period = int(sample_rate * 0.45)
    voiced = int(sample_rate * 0.25)
    for start in range(0, n, period):
        end = min(n, start + voiced)
        window = t[start:end]
        tone = np.zeros(end - start, dtype=np.float32)
        for k, amp in enumerate((0.35, 0.18, 0.09, 0.04), start=1):
            tone += np.float32(amp) * np.sin(2 * np.pi * 140 * k * window).astype(np.float32)
        edge = np.hanning(tone.size).astype(np.float32)
        speech[start:end] = tone * edge
    rng = np.random.default_rng(seed)
    noise = (0.08 * rng.standard_normal(n)).astype(np.float32)
    return speech, speech + noise


def test_parse_enhance_defaults_and_rejects_unknown():
    assert parse_enhance(None) == {"engine": "none", "strength": 1.0}
    assert parse_enhance({}) == {"engine": "none", "strength": 1.0}
    assert parse_enhance({"engine": " DeepFilter ", "strength": "0.5"}) == {
        "engine": "deepfilter", "strength": 0.5,
    }
    assert parse_enhance({"engine": "resemble", "strength": 4})["strength"] == 1.0
    assert parse_enhance({"engine": "none", "strength": -2})["strength"] == 0.0
    with pytest.raises(EnhanceError):
        parse_enhance({"engine": "rnnoise"})
    with pytest.raises(EnhanceError):
        parse_enhance({"strength": "loud"})
    with pytest.raises(EnhanceError):
        parse_enhance("deepfilter")


def test_decision_skips_original_only_even_if_an_engine_is_set():
    cfg = parse_enhance({"engine": "deepfilter", "strength": 1})
    skipped = enhance_decision(cfg, original_only=True, has_audio=True)
    assert skipped["applied"] is False
    assert skipped["reason"] == "audio.original_only"
    assert skipped["engine"] == "deepfilter"
    assert enhance_decision(parse_enhance(None), original_only=False, has_audio=True)["reason"] == "engine_none"
    assert enhance_decision(cfg, original_only=False, has_audio=False)["reason"] == "no_audio"
    assert enhance_decision(
        parse_enhance({"engine": "resemble", "strength": 0}), original_only=False, has_audio=True,
    )["reason"] == "strength_zero"
    assert enhance_decision(cfg, original_only=False, has_audio=True)["reason"] is None


def test_atten_lim_is_unlimited_only_at_full_strength():
    assert atten_lim_db(1.0) is None
    assert atten_lim_db(0.5) == pytest.approx(15.0)


def test_estimate_snr_rises_when_the_noise_floor_drops():
    speech, noisy = _bursts()
    quiet = speech + (noisy - speech) * np.float32(0.1)
    assert estimate_snr_db(quiet, 48000) > estimate_snr_db(noisy, 48000) + 3


def test_resemble_maps_strength_to_lambd_and_keeps_length():
    seen = {}

    def fake(dwav, sr, device, nfe, solver, lambd, tau):
        seen.update(sr=sr, nfe=nfe, lambd=lambd, tau=tau, solver=solver, device=device)
        return dwav * 0.5, sr

    samples = np.ones(1600, dtype=np.float32)
    out, info = _enhance_resemble(samples, 16000, 0.4, enhance_fn=fake)
    assert seen == {
        "sr": 16000, "nfe": 32, "lambd": 0.4, "tau": 0.5, "solver": "midpoint", "device": seen["device"],
    }
    assert out.shape == samples.shape
    assert info["model"] == "resemble-enhance"
    assert info["lambd"] == 0.4


def test_resemble_missing_package_names_the_torch_pin():
    try:
        import resemble_enhance  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("resemble-enhance is installed")
    with pytest.raises(EnhanceError, match="torch==2.1.1"):
        _enhance_resemble(np.zeros(100, dtype=np.float32), 48000, 1.0)


def test_default_engine_does_not_call_the_model(tmp_path, monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("enhance ran")

    monkeypatch.setattr("app.jobs.render.run_enhance", boom)
    src = _src(tmp_path, audio=True, duration=4)
    result = RenderJob(settings=_settings(tmp_path), job=_job("r-off", {
        "input_video": str(src), "format": "9:16", "start_time": 0, "end_time": 3,
    })).execute()
    assert result["applied"]["audio"]["normalized"] is True
    assert result["applied"]["audio"]["enhance"] == {
        "engine": "none", "applied": False, "strength": 1.0, "reason": "engine_none",
    }


def test_original_only_does_not_touch_audio_even_with_deepfilter(tmp_path, monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("enhance ran")

    monkeypatch.setattr("app.jobs.render.run_enhance", boom)
    monkeypatch.setattr("app.tools.ffmpeg.FFmpegTool.extract_audio_segment", boom)
    src = _src(tmp_path, audio=True, duration=6)
    kept = RenderJob(settings=_settings(tmp_path), job=_job("r-oo-enh", {
        "input_video": str(src), "format": "9:16", "start_time": 0, "end_time": 4,
        "audio": {"original_only": True, "enhance": {"engine": "deepfilter", "strength": 1}},
    })).execute()
    meta = kept["applied"]["audio"]
    assert meta["normalized"] is False
    assert meta["reason"] == "audio.original_only"
    assert meta["enhance"]["applied"] is False
    assert meta["enhance"]["engine"] == "deepfilter"
    assert meta["enhance"]["reason"] == "audio.original_only"
    loudness, _peak = _integrated_loudness(Path(kept["file_path"]))
    assert abs(loudness - (-14)) > 5


def test_enhance_runs_before_loudnorm_and_holds_target(tmp_path, monkeypatch):
    calls = {}

    def fake(src_wav, dst_wav, *, engine, strength):
        calls["engine"] = engine
        calls["strength"] = strength
        samples, rate = read_wav(src_wav)
        # Much quieter than the source. Only a later loudnorm lands on -14.
        write_wav(dst_wav, samples * np.float32(0.02), rate)
        return {"engine": engine, "applied": True, "strength": strength, "sample_rate": rate, "device": "fake"}

    monkeypatch.setattr("app.jobs.render.run_enhance", fake)
    src = _src(tmp_path, audio=True, duration=6)
    logo = _logo(tmp_path)
    result = RenderJob(settings=_settings(tmp_path), job=_job("r-enh", {
        "input_video": str(src), "format": "9:16", "start_time": 0.5, "end_time": 4.5,
        "audio": {"enhance": {"engine": "deepfilter", "strength": 0.8}},
        "watermark": {"enabled": True, "file": str(logo), "position": "top_right", "width": 120},
    })).execute()
    assert calls == {"engine": "deepfilter", "strength": 0.8}
    meta = result["applied"]["audio"]
    assert meta["normalized"] is True and meta["filter"] == "loudnorm"
    assert meta["enhance"]["applied"] is True and meta["enhance"]["device"] == "fake"
    assert result["applied"]["watermark"]["applied"] is True
    assert result["probe"]["has_audio"] is True
    loudness, peak = _integrated_loudness(Path(result["file_path"]))
    assert loudness == pytest.approx(-14, abs=1.0)
    assert peak <= -0.5


def test_no_audio_skips_enhance(tmp_path, monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("enhance ran")

    monkeypatch.setattr("app.jobs.render.run_enhance", boom)
    src = _src(tmp_path, audio=False, duration=2)
    result = RenderJob(settings=_settings(tmp_path), job=_job("r-na-enh", {
        "input_video": str(src), "format": "9:16", "start_time": 0, "end_time": 1,
        "audio": {"enhance": {"engine": "deepfilter"}},
    })).execute()
    assert result["applied"]["audio"]["reason"] == "no_audio"
    assert result["applied"]["audio"]["enhance"]["reason"] == "no_audio"
    assert result["applied"]["audio"]["enhance"]["applied"] is False


def test_deepfilter_improves_snr_on_noisy_speech(tmp_path):
    pytest.importorskip("df")
    _speech, noisy = _bursts(seconds=4.0)
    before = estimate_snr_db(noisy, 48000)
    src = tmp_path / "noisy.wav"
    dst = tmp_path / "clean.wav"
    write_wav(src, noisy, 48000)
    info = run_enhance(src, dst, engine="deepfilter", strength=1.0)
    enhanced, rate = read_wav(dst)
    after = estimate_snr_db(enhanced, rate)
    assert info["applied"] is True
    assert info["model"] == "DeepFilterNet3"
    assert after > before + 1.0, f"SNR {before:.2f} -> {after:.2f}"
    # The voiced part should still be present (not silenced).
    assert float(np.max(np.abs(enhanced))) > 0.05
