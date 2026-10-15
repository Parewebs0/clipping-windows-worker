"""Before/after proof for MOLI-52. Same source, same window.

Writes reports/moli-52/{before,after}.mp4 plus before-after.md.
DeepFilterNet runs on the real render path, then loudnorm. `original_only`
is measured on the same window and is not normalized.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.jobs.render import RenderJob  # noqa: E402
from app.models.job import Job  # noqa: E402
from app.tools.enhance import estimate_snr_db, read_wav, run_enhance, write_wav  # noqa: E402
from app.tools.ffmpeg import FFmpegTool, parse_loudnorm_json  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "reports" / "moli-52"
WINDOW = (0.4, 6.4)


def _synthesize(path: Path) -> None:
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$s.Rate = -1; "
        f"$s.SetOutputToWaveFile('{path}'); "
        "$s.Speak('This is a short clip recorded in a noisy room. "
        "Speech enhancement should keep the voice and drop the hiss "
        "before the loudness pass. The same window is rendered twice.'); "
        "$s.Dispose()"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        check=True, capture_output=True,
    )


def _mix(speech_wav: Path, dest: Path) -> None:
    speech, rate = read_wav(speech_wav)
    if rate != 48000:
        from app.tools.enhance import _resample
        speech = _resample(speech, rate, 48000)
        rate = 48000
    rms = float(np.sqrt(np.mean(speech.astype(np.float64) ** 2)) or 1e-6)
    rng = np.random.default_rng(52)
    noise = (rng.standard_normal(speech.size) * rms * 0.55).astype(np.float32)
    write_wav(dest, speech + noise, rate)


def _mux(wav: Path, mp4: Path) -> float:
    samples, rate = read_wav(wav)
    duration = samples.size / rate
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         f"color=c=0x1c1c1c:s=640x360:r=30:d={duration:.3f}",
         "-i", str(wav), "-shortest", "-pix_fmt", "yuv420p",
         "-c:v", "libx264", "-c:a", "aac", "-b:a", "160k", str(mp4)],
        check=True, capture_output=True,
    )
    return duration


def _render(settings: Settings, src: Path, job_id: str, audio: dict | None) -> dict:
    job = MagicMock(spec=Job)
    job.id = job_id
    job.job_type = "render"
    job.payload = {
        "input_video": str(src),
        "format": "9:16",
        "start_time": WINDOW[0],
        "end_time": WINDOW[1],
    }
    if audio:
        job.payload["audio"] = audio
    return RenderJob(settings=settings, job=job).execute()


def _lufs(path: Path) -> tuple[float, float]:
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path), "-vn",
         "-af", "loudnorm=print_format=json", "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    )
    data = parse_loudnorm_json(out.stderr + out.stdout)
    return float(data["input_i"]), float(data["input_tp"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    work = OUT / "_work"
    work.mkdir(exist_ok=True)
    clean = work / "speech.wav"
    noisy = work / "noisy.wav"
    _synthesize(clean)
    _mix(clean, noisy)
    src = work / "source.mp4"
    _mux(noisy, src)

    settings = Settings(
        api_base_url="http://localhost", api_token="proof", worker_id="proof",
        working_directory=str(work / "wd"), output_dir=str(work / "out"),
    )
    segment = work / "window.wav"
    FFmpegTool(settings).extract_audio_segment(
        src, segment, start=WINDOW[0], duration=WINDOW[1] - WINDOW[0], sample_rate=48000,
    )
    enhanced = work / "window_enhanced.wav"
    info = run_enhance(segment, enhanced, engine="deepfilter", strength=1.0)
    noisy_snr = estimate_snr_db(*read_wav(segment))
    enhanced_snr = estimate_snr_db(*read_wav(enhanced))

    before = _render(settings, src, "proof-before", None)
    after = _render(settings, src, "proof-after", {"enhance": {"engine": "deepfilter", "strength": 1}})
    kept = _render(settings, src, "proof-original", {
        "original_only": True,
        "enhance": {"engine": "deepfilter", "strength": 1},
    })
    before_i, before_tp = _lufs(Path(before["file_path"]))
    after_i, after_tp = _lufs(Path(after["file_path"]))
    kept_i, kept_tp = _lufs(Path(kept["file_path"]))

    before_mp4 = OUT / "before.mp4"
    after_mp4 = OUT / "after.mp4"
    before_mp4.write_bytes(Path(before["file_path"]).read_bytes())
    after_mp4.write_bytes(Path(after["file_path"]).read_bytes())
    (OUT / "before-window.wav").write_bytes(segment.read_bytes())
    (OUT / "after-window.wav").write_bytes(enhanced.read_bytes())

    report = {
        "asset": "synthetic speech (Windows SAPI) + stationary noise, muxed to mp4",
        "window_seconds": list(WINDOW),
        "snr_db": {"before_enhance": round(noisy_snr, 2), "after_deepfilter": round(enhanced_snr, 2)},
        "lufs": {
            "before_loudnorm_only": {"integrated": round(before_i, 2), "true_peak": round(before_tp, 2)},
            "after_enhance_then_loudnorm": {"integrated": round(after_i, 2), "true_peak": round(after_tp, 2)},
            "original_only": {"integrated": round(kept_i, 2), "true_peak": round(kept_tp, 2)},
        },
        "enhance": info,
        "applied_after": after["applied"]["audio"],
        "applied_original_only": kept["applied"]["audio"],
    }
    (OUT / "before-after.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "before-after.md").write_text(
        "\n".join([
            "# MOLI-52 before / after",
            "",
            "Same source and the same window "
            f"({WINDOW[0]:.1f}s–{WINDOW[1]:.1f}s).",
            "",
            f"- SNR estimated on the clip window, before loudnorm: "
            f"{noisy_snr:.1f} dB → {enhanced_snr:.1f} dB after DeepFilterNet 3.",
            f"- Loudnorm only (`engine: none`): {before_i:.2f} LUFS, true peak {before_tp:.2f} dBTP.",
            f"- Enhance then loudnorm: {after_i:.2f} LUFS, true peak {after_tp:.2f} dBTP.",
            f"- `original_only` with the engine requested anyway: {kept_i:.2f} LUFS "
            f"(not pulled to -14; enhance applied={kept['applied']['audio']['enhance']['applied']}).",
            "",
            "Files: `before.mp4` (loudnorm only), `after.mp4` (deepfilter then loudnorm), "
            "`before-window.wav`, `after-window.wav`.",
            "",
        ]),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
