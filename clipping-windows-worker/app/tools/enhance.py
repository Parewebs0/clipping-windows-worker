"""Speech enhancement before loudnorm (MOLI-52).

``audio.enhance = {engine: none|deepfilter|resemble, strength}``.

``none`` (the default) does nothing, so a render stays loudnorm-only.
``deepfilter`` runs DeepFilterNet 3 (MIT) and maps ``strength`` onto its
attenuation limit (1 = full suppression). ``resemble`` calls Resemble Enhance
(MIT) and maps ``strength`` onto ``lambd`` (1 = full restoration). The PyPI
release of resemble-enhance pins torch==2.1.1, which fights the worker CUDA
build, so that engine stays optional and errors clearly when it is missing.

``audio.original_only`` skips this step. The caller still loudnorm-normalizes
only when that flag is false.
"""

from __future__ import annotations

import threading
import wave
from pathlib import Path
from typing import Any, Callable

import numpy as np

ENGINES = ("none", "deepfilter", "resemble")
DEFAULT_ENHANCE = {"engine": "none", "strength": 1.0}
# DeepFilterNet3 and our extracted clip audio.
SAMPLE_RATE = 48000

_DF_LOCK = threading.Lock()
_DF_STATE: tuple[Any, Any] | None = None


class EnhanceError(RuntimeError):
    """The requested enhancer cannot run. The render job should fail."""


def parse_enhance(raw: Any) -> dict[str, Any]:
    """Return ``{engine, strength}``. Missing values keep the loudnorm-only default.

    Unknown engines raise. Strength is clamped to 0..1.
    """
    if raw is None:
        return dict(DEFAULT_ENHANCE)
    if not isinstance(raw, dict):
        raise EnhanceError("audio.enhance must be an object")
    engine = str(raw.get("engine") or "none").strip().lower()
    if engine not in ENGINES:
        allowed = ", ".join(ENGINES)
        raise EnhanceError(f"audio.enhance.engine must be one of {allowed}")
    strength = raw.get("strength", DEFAULT_ENHANCE["strength"])
    try:
        strength_f = float(strength)
    except (TypeError, ValueError) as exc:
        raise EnhanceError("audio.enhance.strength must be a number") from exc
    if strength_f != strength_f:  # NaN
        raise EnhanceError("audio.enhance.strength must be a number")
    strength_f = min(1.0, max(0.0, strength_f))
    return {"engine": engine, "strength": strength_f}


def enhance_decision(cfg: dict[str, Any], *, original_only: bool, has_audio: bool) -> dict[str, Any]:
    """Say whether this clip should be enhanced. ``reason is None`` means run it."""
    meta: dict[str, Any] = {
        "engine": cfg["engine"],
        "applied": False,
        "strength": cfg["strength"],
    }
    if original_only:
        meta["reason"] = "audio.original_only"
    elif not has_audio:
        meta["reason"] = "no_audio"
    elif cfg["engine"] == "none":
        meta["reason"] = "engine_none"
    elif cfg["strength"] <= 0:
        meta["reason"] = "strength_zero"
    else:
        meta["reason"] = None
    return meta


def atten_lim_db(strength: float) -> float | None:
    """DeepFilterNet attenuation limit. ``None`` means unlimited suppression."""
    if strength >= 0.999:
        return None
    return max(0.1, float(strength) * 30.0)


def estimate_snr_db(samples: np.ndarray, sample_rate: int, *, frame_ms: float = 32.0) -> float:
    """Noise-floor SNR in dB.

    Quiet frames (20th percentile energy) are the noise floor. Louder frames
    are treated as speech. Needs pauses in the clip; a constant tone has no
    floor to compare against.
    """
    audio = np.asarray(samples, dtype=np.float64).reshape(-1)
    if audio.size == 0 or sample_rate <= 0:
        return 0.0
    audio = audio - np.mean(audio)
    frame = max(1, int(sample_rate * frame_ms / 1000.0))
    count = audio.size // frame
    if count < 4:
        return 0.0
    energy = np.mean(audio[: count * frame].reshape(count, frame) ** 2, axis=1)
    noise = float(np.percentile(energy, 20))
    speech = energy[energy > max(noise * 3.0, 1e-12)]
    if speech.size < 2 or noise <= 1e-12:
        return 60.0 if noise <= 1e-12 else 0.0
    signal = max(float(np.mean(speech) - noise), 1e-12)
    return float(10.0 * np.log10(signal / noise))


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read PCM 16-bit WAV as mono float32 in [-1, 1]."""
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        raise EnhanceError(f"expected 16-bit wav, got sample width {width}")
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32), int(sample_rate)


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> None:
    pcm = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    ints = (pcm * 32767.0).astype(np.int16)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(ints.tobytes())


def _match_length(samples: np.ndarray, length: int) -> np.ndarray:
    if samples.size == length:
        return samples
    if samples.size > length:
        return samples[:length]
    return np.pad(samples, (0, length - samples.size))


def _resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or samples.size == 0:
        return np.asarray(samples, dtype=np.float32)
    dst_len = max(1, int(round(samples.size * dst_rate / src_rate)))
    if samples.size == 1 or dst_len == 1:
        return np.resize(samples, dst_len).astype(np.float32)
    old = np.linspace(0.0, 1.0, num=samples.size, endpoint=False)
    new = np.linspace(0.0, 1.0, num=dst_len, endpoint=False)
    return np.interp(new, old, samples).astype(np.float32)


def _df_model() -> tuple[Any, Any]:
    global _DF_STATE
    with _DF_LOCK:
        if _DF_STATE is None:
            try:
                from df.enhance import init_df
            except ImportError as exc:
                raise EnhanceError(
                    "deepfilternet is not installed. See requirements-enhance.txt. "
                    "Install it with --no-deps if the worker already has numpy 2."
                ) from exc
            # 0.5.6 returns (model, df_state, suffix). Later builds also return the epoch.
            loaded = init_df(
                model_base_dir="DeepFilterNet3",
                log_level="ERROR",
                log_file=None,
            )
            _DF_STATE = (loaded[0], loaded[1])
        return _DF_STATE


def _enhance_deepfilter(samples: np.ndarray, sample_rate: int, strength: float) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from df.enhance import enhance

    model, df_state = _df_model()
    target_rate = int(df_state.sr())
    audio = samples if sample_rate == target_rate else _resample(samples, sample_rate, target_rate)
    tensor = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)
    limit = atten_lim_db(strength)
    enhanced = enhance(model, df_state, tensor, pad=True, atten_lim_db=limit)
    out = enhanced.squeeze(0).detach().cpu().numpy().astype(np.float32)
    if sample_rate != target_rate:
        out = _resample(out, target_rate, sample_rate)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return _match_length(out, samples.size), {
        "device": device,
        "model": "DeepFilterNet3",
        "atten_lim_db": limit,
        "sample_rate": sample_rate,
    }


def _enhance_resemble(
    samples: np.ndarray,
    sample_rate: int,
    strength: float,
    *,
    enhance_fn: Callable[..., Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    if enhance_fn is None:
        try:
            from resemble_enhance.enhancer.inference import enhance as enhance_fn
        except ImportError as exc:
            raise EnhanceError(
                "resemble-enhance is not installed. Its PyPI release pins torch==2.1.1, "
                "which conflicts with the worker CUDA build, so it is optional. "
                "Install it only into an environment that already has a compatible torch."
            ) from exc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wav = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32))
    restored = enhance_fn(
        wav, sample_rate, device, nfe=32, solver="midpoint", lambd=float(strength), tau=0.5,
    )
    out_rate = sample_rate
    if isinstance(restored, tuple):
        restored, out_rate = restored
    out = restored.detach().cpu().numpy().astype(np.float32).reshape(-1)
    if int(out_rate) != sample_rate:
        out = _resample(out, int(out_rate), sample_rate)
    return _match_length(out, samples.size), {
        "device": device,
        "model": "resemble-enhance",
        "lambd": float(strength),
        "nfe": 32,
        "sample_rate": sample_rate,
    }


def run_enhance(input_wav: str | Path, output_wav: str | Path, *, engine: str, strength: float) -> dict[str, Any]:
    """Enhance a PCM wav in place of the clip's audio. Returns applied metadata."""
    samples, sample_rate = read_wav(input_wav)
    if engine == "deepfilter":
        enhanced, info = _enhance_deepfilter(samples, sample_rate, strength)
    elif engine == "resemble":
        enhanced, info = _enhance_resemble(samples, sample_rate, strength)
    else:
        raise EnhanceError("audio.enhance.engine must be one of deepfilter, resemble")
    write_wav(output_wav, enhanced, sample_rate)
    return {"engine": engine, "applied": True, "strength": float(strength), **info}
