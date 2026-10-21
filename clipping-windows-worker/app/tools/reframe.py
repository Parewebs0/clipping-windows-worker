"""Dynamic reframe: face / object tracking, speaker follow, layout switch.

Default ``mode=center`` + ``layout=single`` does not build a crop path. The
renderer keeps the historical FFmpeg center-crop for 9:16 and 16:9.

Tracking samples the clip at 5–10 fps, picks the active subject (diarized
speaker matched by mouth motion, else the largest face, else a tracked
object), then eases the crop with a dead zone, an EMA and a hard per-frame
shift cap. The cap is what removes jerks. Frame 0 is allowed to sit on the
subject; every later frame moves at most ``max_shift_frac * min(crop_w, crop_h)``
pixels.

Layouts (used when mode is face, object or auto, or when layout is forced):

- ``single`` — one window following the subject (FFmpeg ``sendcmd`` + ``crop``)
- ``split`` — two speakers, stacked (9:16, 1:1) or side by side (16:9)
- ``screenshare`` — content on the large pane, face on the small pane
- ``fit`` — full frame letterboxed over a blurred copy of itself

``layout=auto`` holds a choice for ``hold_s`` (default 0.75 s) before switching.

Face mode (T1, 2026-10-21):

- the window follows one main person: the diarized speaker when known, else
  the face track with the most screen time × size, else the face nearest to
  the previous target. When no face is seen the window holds its last spot
  instead of jumping back to the center;
- ``zoom`` (on by default in face mode) tightens the single window when the
  main face is small (wide shot), so the face is about ``target_face_frac`` of
  the crop height, never past ``max_zoom`` nor below ``min_crop_h`` source
  pixels. One zoom per clip, so it never pumps;
- ``mode=face`` never picks ``fit`` (blurred backdrop) on its own, and the
  single window is scaled to the exact output size: no pad bars.
"""

from __future__ import annotations

import math
import os
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from app.utils.logging import get_logger

logger = get_logger("reframe")

OUTPUT_SIZES: dict[str, tuple[int, int]] = {
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
    "16:9": (1920, 1080),
}

SCREEN_LABELS = {"tv", "laptop", "cell phone", "monitor"}
MODES = {"center", "face", "object", "auto"}
LAYOUTS = {"single", "split", "screenshare", "fit", "auto"}

# Full-range BlazeFace (Apache-2.0 model card, MediaPipe). Faces in a wide
# interview are often small; the short-range model misses them.
_FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_full_range/float16/latest/blaze_face_full_range.tflite"
)
_FACE_MODEL_NAME = "blaze_face_full_range.tflite"
_YOLO_MODEL = "yolo11n.pt"

# MediaPipe's Windows wheel runs the detector on CPU. Object tracking goes
# through Ultralytics YOLO + ByteTrack, which uses CUDA when torch can see it.
_GPU_NOTE = (
    "Face detection (MediaPipe) runs on CPU on the Windows wheel. "
    "Object tracking (YOLO11n + ByteTrack) uses CUDA when torch.cuda is available. "
    "The final encode stays libx264 unless the worker is switched to nvenc."
)


@dataclass
class Smoothing:
    ema: float = 0.45
    deadzone_frac: float = 0.03
    max_shift_frac: float = 0.05

    @property
    def as_dict(self) -> dict[str, float]:
        return {
            "ema": self.ema,
            "deadzone_frac": self.deadzone_frac,
            "max_shift_frac": self.max_shift_frac,
        }


@dataclass
class Det:
    t: float
    x: float  # center, source pixels
    y: float
    w: float
    h: float
    score: float = 1.0
    kind: str = "face"
    label: str = "face"
    track_id: int | None = None
    mouth: float = 0.0
    ident: float | None = None  # T1b: cosine vs the campaign subject (None = not measured)
    subject: bool = False       # T1b: the track was recognised as the subject
    subject_score: float = 0.0  # T1b: the track's identity score (mean of its best matches)

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)


@dataclass
class Sample:
    t: float
    faces: list[Det] = field(default_factory=list)
    objects: list[Det] = field(default_factory=list)


@dataclass
class ReframePlan:
    """What ``render_clip`` should do.

    ``legacy`` means "do not touch the filter graph". ``vf`` replaces only the
    crop/scale/pad stage. ``complex_graph`` is a filter_complex body that
    already ends in ``[vmain]``.
    """

    legacy: bool
    vf: list[str] | None
    complex_graph: str | None
    metadata: dict[str, Any]
    command_files: list[Path]


def output_size(fmt: str) -> tuple[int, int]:
    try:
        return OUTPUT_SIZES[fmt]
    except KeyError:
        raise ValueError(f"Unsupported output_format: {fmt}") from None


def square_center_filters() -> list[str]:
    """Centered 1:1 crop. 9:16 and 16:9 keep the historical strings in ffmpeg.py."""
    return [
        "crop=min(iw\\,ih):min(iw\\,ih):(iw-min(iw\\,ih))/2:(ih-min(iw\\,ih))/2",
        "scale=1080:1080:force_original_aspect_ratio=decrease",
        "pad=1080:1080:(ow-iw)/2:(oh-ih)/2",
    ]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def normalize_reframe(raw: dict | None) -> dict[str, Any]:
    """Fill defaults and clamp numbers. Unknown modes fall back to center."""
    src = raw or {}
    mode = str(src.get("mode") or "center").strip().lower()
    layout = str(src.get("layout") or "single").strip().lower()
    if mode not in MODES:
        mode = "center"
    if layout not in LAYOUTS:
        layout = "single"
    smoothing = _parse_smoothing(src.get("smoothing"))
    try:
        sample_fps = float(src.get("sample_fps") or 8)
    except (TypeError, ValueError):
        sample_fps = 8.0
    sample_fps = _clamp(sample_fps, 5.0, 10.0)
    try:
        hold_s = float(src.get("hold_s") if src.get("hold_s") is not None else 0.75)
    except (TypeError, ValueError):
        hold_s = 0.75
    hold_s = _clamp(hold_s, 0.0, 3.0)
    speakers = []
    for seg in src.get("speakers") or []:
        if not isinstance(seg, dict) or not seg.get("speaker"):
            continue
        try:
            speakers.append({
                "start": float(seg.get("start") or 0),
                "end": float(seg.get("end") or 0),
                "speaker": str(seg["speaker"]),
            })
        except (TypeError, ValueError):
            continue
    out: dict[str, Any] = {
        "mode": mode,
        "layout": layout,
        "smoothing": smoothing,
        "sample_fps": sample_fps,
        "hold_s": hold_s,
        "speakers": speakers,
        "zoom": _parse_zoom(src.get("zoom"), default_on=mode == "face"),
    }
    subject = _parse_subject(src.get("subject"))
    if subject:
        out["subject"] = subject
    if "detections" in src:
        out["detections"] = src.get("detections") or []
    return out


def _parse_subject(raw: Any) -> dict[str, Any] | None:
    """``subject: {name, reference_paths[], threshold}``. Needs local reference images."""
    if not isinstance(raw, dict):
        return None
    paths = [str(p) for p in raw.get("reference_paths") or [] if str(p or "").strip()]
    if not paths:
        return None
    try:
        threshold = _clamp(float(raw.get("threshold") or SUBJECT_MATCH), 0.2, 0.9)
    except (TypeError, ValueError):
        threshold = SUBJECT_MATCH
    return {"name": str(raw.get("name") or "subject")[:80], "reference_paths": paths[:8], "threshold": threshold}


# T1b: SFace cosine for "same person" and the share of a track's measured faces
# that must match before the whole track counts as the subject.
SUBJECT_MATCH = 0.363
SUBJECT_TRACK_SHARE = 0.3
# A lookalike can pass the absolute SFace threshold (0.36-0.43 seen against
# Speed's references while Speed himself scores 0.7-0.85), so a track must
# also score close to the best-matching track of the clip.
SUBJECT_RELATIVE = 0.55
SUBJECT_TOP_K = 3
SUBJECT_EVERY_S = 0.25


def label_subject_tracks(samples: list[Sample], threshold: float = SUBJECT_MATCH) -> list[int]:
    """Mark every face of a track whose measured faces match the subject often enough.

    Recognition only sees frontal faces, so one track is judged as a whole: a
    profile in the middle of a recognised track still counts as the subject.
    """
    seen: dict[int, list[float]] = {}
    for sample in samples:
        for face in sample.faces:
            if face.track_id is not None and face.ident is not None:
                seen.setdefault(face.track_id, []).append(face.ident)
    scores = {tid: sum(top) / len(top) for tid, sims in seen.items()
              for top in [sorted(sims, reverse=True)[:SUBJECT_TOP_K]]}
    best = max(scores.values(), default=0.0)
    floor = max(threshold, SUBJECT_RELATIVE * best)
    tracks = sorted(
        tid for tid, sims in seen.items()
        if scores[tid] >= floor
        and sum(1 for v in sims if v >= threshold) >= max(1, round(SUBJECT_TRACK_SHARE * len(sims)))
    )
    for sample in samples:
        for face in sample.faces:
            face.subject = face.track_id in tracks
            face.subject_score = scores.get(face.track_id, 0.0) if face.subject else 0.0
    return tracks


def subject_on_screen(
    samples: list[Sample], origins: list[tuple[int, int]], times: list[float], crop_w: int, crop_h: int,
) -> tuple[float | None, float | None]:
    """(share of samples with the subject anywhere, share with the subject inside the window)."""
    if not samples or not origins:
        return None, None
    present = inside = 0
    for sample in samples:
        faces = [f for f in sample.faces if f.subject]
        if not faces:
            continue
        present += 1
        index = min(range(len(times)), key=lambda i: abs(times[i] - sample.t))
        ox, oy = origins[index]
        if any(ox <= f.x <= ox + crop_w and oy <= f.y <= oy + crop_h for f in faces):
            inside += 1
    return present / len(samples), inside / len(samples)


ZOOM_DEFAULTS = {"enabled": False, "target_face_frac": 0.14, "max_zoom": 1.6, "min_crop_h": 640}


def _parse_zoom(raw: Any, *, default_on: bool) -> dict[str, Any]:
    """``zoom: true|false|{enabled, target_face_frac, max_zoom, min_crop_h}``."""
    out = dict(ZOOM_DEFAULTS, enabled=default_on)
    if isinstance(raw, bool):
        out["enabled"] = raw
        return out
    if not isinstance(raw, dict):
        return out
    if "enabled" in raw:
        out["enabled"] = bool(raw.get("enabled"))
    for key, lo, hi in (("target_face_frac", 0.05, 0.5), ("max_zoom", 1.0, 3.0), ("min_crop_h", 240, 4320)):
        try:
            out[key] = _clamp(float(raw.get(key, out[key])), lo, hi)
        except (TypeError, ValueError):
            pass
    out["min_crop_h"] = int(out["min_crop_h"])
    return out


def _parse_smoothing(raw: Any) -> Smoothing:
    if isinstance(raw, (int, float)):
        return Smoothing(ema=_clamp(float(raw), 0.05, 1.0))
    if not isinstance(raw, dict):
        return Smoothing()
    def num(key: str, default: float, lo: float, hi: float) -> float:
        try:
            return _clamp(float(raw.get(key, default)), lo, hi)
        except (TypeError, ValueError):
            return default
    return Smoothing(
        ema=num("ema", 0.45, 0.05, 1.0),
        deadzone_frac=num("deadzone_frac", 0.03, 0.0, 0.25),
        max_shift_frac=num("max_shift_frac", 0.05, 0.001, 0.5),
    )


def needs_source_size(reframe: dict | None, output_format: str) -> bool:
    """False for the historical center crop, so the default render does not probe."""
    cfg = normalize_reframe(reframe)
    if cfg["mode"] == "center" and cfg["layout"] in ("single", "auto"):
        return False
    if output_format not in OUTPUT_SIZES and cfg["mode"] == "center":
        return False
    return True


def _even(n: float) -> int:
    v = int(math.floor(n))
    if v % 2:
        v -= 1
    return max(2, v)


def _crop_for_slot(src_w: int, src_h: int, slot_w: int, slot_h: int) -> tuple[int, int]:
    """Largest even crop that fits in the source and matches the slot aspect."""
    aspect = slot_w / slot_h
    ch = float(src_h)
    cw = ch * aspect
    if cw > src_w:
        cw = float(src_w)
        ch = cw / aspect
    cw_i, ch_i = _even(cw), _even(ch)
    if cw_i > src_w:
        cw_i = _even(src_w)
    if ch_i > src_h:
        ch_i = _even(src_h)
    return cw_i, ch_i


def _slot_sizes(fmt: str, layout: str) -> list[tuple[str, int, int]]:
    ow, oh = output_size(fmt)
    if layout == "split":
        if fmt == "16:9":
            w = _even(ow / 2)
            return [("ptop", w, oh), ("pbot", w, oh)]
        h = _even(oh / 2)
        return [("ptop", ow, h), ("pbot", ow, h)]
    if layout == "screenshare":
        if fmt == "16:9":
            face_w = _even(ow * 0.32)
            return [("screen", ow - face_w, oh), ("face", face_w, oh)]
        face_h = _even(oh * 0.32)
        return [("screen", ow, oh - face_h), ("face", ow, face_h)]
    if layout == "fit":
        return [("fit", ow, oh)]
    return [("single", ow, oh)]


def _scale_pad(w: int, h: int) -> str:
    return (
        f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2"
    )


def _box_iou(a: Det, b: Det) -> float:
    ax0, ay0 = a.x - a.w / 2, a.y - a.h / 2
    ax1, ay1 = ax0 + a.w, ay0 + a.h
    bx0, by0 = b.x - b.w / 2, b.y - b.h / 2
    bx1, by1 = bx0 + b.w, by0 + b.h
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = a.area + b.area - inter
    if union <= 0:
        return 0.0
    return inter / union


def assign_track_ids(dets: list[Det]) -> list[Det]:
    """Greedy IoU tracker. Detections that already have a track id keep it."""
    by_t: dict[float, list[Det]] = {}
    for det in dets:
        by_t.setdefault(round(det.t, 3), []).append(det)
    next_id = 1 + max((d.track_id or 0) for d in dets) if dets else 1
    active: list[Det] = []
    ordered: list[Det] = []
    for t in sorted(by_t):
        used: set[int] = set()
        frame_out: list[Det] = []
        for det in by_t[t]:
            if det.track_id is not None:
                used.add(det.track_id)
                frame_out.append(det)
                continue
            best_id = None
            best_iou = 0.3
            for prev in active:
                if prev.track_id in used:
                    continue
                score = _box_iou(prev, det)
                if score > best_iou:
                    best_iou = score
                    best_id = prev.track_id
            if best_id is None:
                best_id = next_id
                next_id += 1
            det.track_id = best_id
            used.add(best_id)
            frame_out.append(det)
        active = frame_out
        ordered.extend(frame_out)
    return ordered


def _parse_detections(raw: Iterable[dict]) -> list[Det]:
    dets: list[Det] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            det = Det(
                t=float(item["t"]),
                x=float(item["x"]),
                y=float(item["y"]),
                w=float(item.get("w") or item.get("width") or 1),
                h=float(item.get("h") or item.get("height") or 1),
                score=float(item.get("score") or 1),
                kind=str(item.get("kind") or "face"),
                label=str(item.get("label") or item.get("kind") or "face"),
                mouth=float(item.get("mouth") or 0),
            )
        except (KeyError, TypeError, ValueError):
            continue
        if item.get("track_id") is not None:
            try:
                det.track_id = int(item["track_id"])
            except (TypeError, ValueError):
                det.track_id = None
        if item.get("ident") is not None:
            try:
                det.ident = float(item["ident"])
            except (TypeError, ValueError):
                det.ident = None
        dets.append(det)
    return assign_track_ids(dets)


def _samples_from_dets(dets: list[Det]) -> list[Sample]:
    by_t: dict[float, Sample] = {}
    for det in dets:
        key = round(det.t, 3)
        sample = by_t.setdefault(key, Sample(t=key))
        if det.kind == "object":
            sample.objects.append(det)
        else:
            sample.faces.append(det)
    return [by_t[k] for k in sorted(by_t)]


def speaker_at(t: float, speakers: list[dict]) -> str | None:
    """Clip-relative speaker. The interval that contains ``t`` wins; the last
    one is closed at its end so the final frame still has a speaker."""
    if not speakers:
        return None
    last = max(speakers, key=lambda s: float(s["end"]))
    for seg in speakers:
        start, end = float(seg["start"]), float(seg["end"])
        if start <= t < end:
            return str(seg["speaker"])
        if seg is last and start <= t <= end:
            return str(seg["speaker"])
    return None


def map_speakers(samples: list[Sample], speakers: list[dict]) -> dict[str, int]:
    """Match each diarized speaker to the face track whose mouth moves while they talk."""
    spk_ids: list[str] = []
    for seg in speakers:
        name = str(seg["speaker"])
        if name not in spk_ids:
            spk_ids.append(name)
    track_ids = sorted({
        f.track_id for sample in samples for f in sample.faces if f.track_id is not None
    })
    if not spk_ids or not track_ids:
        return {}
    buckets: dict[str, dict[int, list[float]]] = {
        spk: {tid: [] for tid in track_ids} for spk in spk_ids
    }
    for sample in samples:
        spk = speaker_at(sample.t, speakers)
        if spk not in buckets:
            continue
        for face in sample.faces:
            if face.track_id in buckets[spk]:
                buckets[spk][face.track_id].append(face.mouth)
    means = {
        spk: {tid: (sum(vals) / len(vals) if vals else 0.0) for tid, vals in tracks.items()}
        for spk, tracks in buckets.items()
    }
    mapping: dict[str, int] = {}
    used: set[int] = set()
    pairs: list[tuple[float, str, int]] = []
    for spk, tracks in means.items():
        for tid, score in tracks.items():
            pairs.append((score, spk, tid))
    pairs.sort(key=lambda item: item[0], reverse=True)
    for score, spk, tid in pairs:
        if spk in mapping or tid in used:
            continue
        if score <= 0:
            continue
        mapping[spk] = tid
        used.add(tid)
    if len(mapping) < min(len(spk_ids), len(track_ids)):
        # No mouth signal: left-to-right face order follows speaker order.
        xs: dict[int, list[float]] = {tid: [] for tid in track_ids}
        for sample in samples:
            for face in sample.faces:
                if face.track_id in xs:
                    xs[face.track_id].append(face.x)
        ordered_tracks = sorted(track_ids, key=lambda tid: sum(xs[tid]) / len(xs[tid]))
        for spk, tid in zip(spk_ids, ordered_tracks):
            if spk not in mapping and tid not in mapping.values():
                mapping[spk] = tid
    return mapping


def _interp(targets: list[tuple[float, float, float]], t: float) -> tuple[float, float]:
    if t <= targets[0][0]:
        return targets[0][1], targets[0][2]
    if t >= targets[-1][0]:
        return targets[-1][1], targets[-1][2]
    for left, right in zip(targets, targets[1:]):
        if left[0] <= t <= right[0]:
            span = right[0] - left[0]
            if span <= 0:
                return right[1], right[2]
            u = (t - left[0]) / span
            return left[1] + u * (right[1] - left[1]), left[2] + u * (right[2] - left[2])
    return targets[-1][1], targets[-1][2]


def _clamp_center(cx: float, cy: float, cw: int, ch: int, src_w: int, src_h: int) -> tuple[float, float]:
    half_w = cw / 2
    half_h = ch / 2
    if src_w <= cw:
        cx = src_w / 2
    else:
        cx = _clamp(cx, half_w, src_w - half_w)
    if src_h <= ch:
        cy = src_h / 2
    else:
        cy = _clamp(cy, half_h, src_h - half_h)
    return cx, cy


def _origin(cx: float, cy: float, cw: int, ch: int, src_w: int, src_h: int) -> tuple[int, int]:
    x = int(round(cx - cw / 2))
    y = int(round(cy - ch / 2))
    x = int(_clamp(x, 0, max(0, src_w - cw)))
    y = int(_clamp(y, 0, max(0, src_h - ch)))
    return x, y


def _limit_step(prev: tuple[int, int], x: int, y: int, max_shift: float) -> tuple[int, int]:
    dx = x - prev[0]
    dy = y - prev[1]
    dist = math.hypot(dx, dy)
    if dist <= max_shift or dist == 0:
        return x, y
    scale = max_shift / dist
    nx = prev[0] + int(math.copysign(math.floor(abs(dx * scale)), dx))
    ny = prev[1] + int(math.copysign(math.floor(abs(dy * scale)), dy))
    return nx, ny


def smooth_origins(
    targets: list[tuple[float, float, float]],
    times: list[float],
    *,
    crop_w: int,
    crop_h: int,
    src_w: int,
    src_h: int,
    smoothing: Smoothing,
    cuts: Iterable[float] = (),
) -> tuple[list[tuple[int, int]], float]:
    """Integer crop origins at ``times``. Steps after frame 0 stay within the cap,
    except at ``cuts`` (new shots), where the window jumps to the target."""
    max_shift = smoothing.max_shift_frac * float(min(crop_w, crop_h))
    dead = smoothing.deadzone_frac * float(min(crop_w, crop_h))
    if not targets:
        targets = [(0.0, src_w / 2, src_h / 2)]
    if not times:
        times = [0.0]

    def desired(t: float) -> tuple[float, float]:
        cx, cy = _interp(targets, t)
        return _clamp_center(cx, cy, crop_w, crop_h, src_w, src_h)

    snaps = set(cut_indices(times, cuts))
    cx, cy = desired(times[0])
    origins = [_origin(cx, cy, crop_w, crop_h, src_w, src_h)]
    for i, t in enumerate(times[1:], start=1):
        tx, ty = desired(t)
        if i in snaps:
            cx, cy = tx, ty
            origins.append(_origin(cx, cy, crop_w, crop_h, src_w, src_h))
            continue
        dx, dy = tx - cx, ty - cy
        dist = math.hypot(dx, dy)
        if dist >= dead:
            dx *= smoothing.ema
            dy *= smoothing.ema
            dist = math.hypot(dx, dy)
            if dist > max_shift and dist > 0:
                scale = max_shift / dist
                dx *= scale
                dy *= scale
            cx += dx
            cy += dy
        ox, oy = _origin(cx, cy, crop_w, crop_h, src_w, src_h)
        ox, oy = _limit_step(origins[-1], ox, oy, max_shift)
        origins.append((ox, oy))
        cx = ox + crop_w / 2
        cy = oy + crop_h / 2
    return origins, max_shift


def cut_indices(times: list[float], cuts: Iterable[float]) -> list[int]:
    """First output frame at or after each cut (never frame 0)."""
    out = []
    for cut in cuts:
        index = next((i for i, t in enumerate(times) if t >= cut - 1e-6), None)
        if index:
            out.append(index)
    return out


def _max_step(origins: list[tuple[int, int]], skip: Iterable[int] = ()) -> float:
    skipped = set(skip)
    worst = 0.0
    for i, (left, right) in enumerate(zip(origins, origins[1:]), start=1):
        if i not in skipped:
            worst = max(worst, math.hypot(right[0] - left[0], right[1] - left[1]))
    return worst


def _face_at(sample: Sample, track_id: int | None) -> Det | None:
    if track_id is not None:
        for face in sample.faces:
            if face.track_id == track_id:
                return face
    if not sample.faces:
        return None
    return max(sample.faces, key=lambda face: face.area)


def _center_of(src_w: int, src_h: int) -> tuple[float, float]:
    return src_w / 2, src_h / 2


def classify_sample(
    sample: Sample,
    *,
    src_w: int,
    src_h: int,
    two_speakers: bool,
) -> str:
    frame = float(src_w * src_h) or 1.0
    screens = [
        obj for obj in sample.objects
        if (obj.label or "").lower() in SCREEN_LABELS and obj.area >= 0.12 * frame
    ]
    if screens and sample.faces:
        return "screenshare"
    if len(sample.faces) == 1:
        face = sample.faces[0]
        near_edge = (
            face.x < src_w * 0.22 or face.x > src_w * 0.78
            or face.y < src_h * 0.22 or face.y > src_h * 0.78
        )
        if near_edge and face.area < 0.12 * frame:
            return "screenshare"
    if two_speakers and len(sample.faces) >= 2:
        return "split"
    if not sample.faces and not sample.objects:
        return "fit"
    return "single"


def coalesce_layouts(labels: list[tuple[float, str]], *, hold_s: float, duration: float) -> list[dict]:
    if not labels:
        return [{"start": 0.0, "end": round(duration, 3), "layout": "single"}]
    current = labels[0][1]
    pending: str | None = None
    pending_since = labels[0][0]
    chosen: list[tuple[float, str]] = []
    for t, label in labels:
        if label == current:
            pending = None
        elif pending != label:
            pending = label
            pending_since = t
        elif t - pending_since >= hold_s:
            current = label
            pending = None
        chosen.append((t, current))
    spans: list[dict] = []
    span_start = chosen[0][0]
    span_label = chosen[0][1]
    for (t, label), nxt in zip(chosen, chosen[1:] + [(duration, chosen[-1][1])]):
        next_t, next_label = nxt
        if next_label != span_label:
            spans.append({"start": round(span_start, 3), "end": round(next_t, 3), "layout": span_label})
            span_start = next_t
            span_label = next_label
    spans.append({"start": round(span_start, 3), "end": round(duration, 3), "layout": span_label})
    # The loop above appends the closing span twice when the label never changes
    # on the sentinel. Collapse neighbours.
    merged: list[dict] = []
    for span in spans:
        if span["end"] <= span["start"]:
            continue
        if merged and merged[-1]["layout"] == span["layout"] and abs(merged[-1]["end"] - span["start"]) < 1e-6:
            merged[-1]["end"] = span["end"]
        else:
            merged.append(span)
    return merged or [{"start": 0.0, "end": round(duration, 3), "layout": labels[0][1]}]


def _output_times(duration: float, fps: float) -> list[float]:
    fps = fps if fps > 0 else 30.0
    count = max(1, int(round(duration * fps)))
    return [i / fps for i in range(count)]


def _static_centers(layout: str, src_w: int, src_h: int) -> dict[str, tuple[float, float]]:
    cx, cy = _center_of(src_w, src_h)
    if layout == "split":
        return {"ptop": (src_w * 0.33, cy), "pbot": (src_w * 0.67, cy)}
    if layout == "screenshare":
        return {"screen": (cx, cy), "face": (cx, cy)}
    return {"single": (cx, cy), "fit": (cx, cy)}


def main_track(samples: list[Sample]) -> int | None:
    """Face track with the most screen time weighted by size: the main person."""
    score: dict[int, float] = {}
    for sample in samples:
        for face in sample.faces:
            if face.track_id is not None:
                score[face.track_id] = score.get(face.track_id, 0.0) + face.area
    if not score:
        return None
    return max(score, key=lambda tid: (score[tid], -tid))


def _nearest(faces: list[Det], x: float, y: float) -> Det:
    return min(faces, key=lambda f: (f.x - x) ** 2 + (f.y - y) ** 2)


def _subject_targets(
    samples: list[Sample],
    *,
    speakers: list[dict],
    mapping: dict[str, int],
    src_w: int,
    src_h: int,
    use_objects: bool,
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float, float]], list[tuple[float, float]]]:
    """Targets for the single window, the real faces it followed (t, x, y, h)
    and jumps (t, distance px) where it moved to a face track never seen before
    (a new shot: the caller snaps instead of panning when the jump is large).

    Order of preference per sample: the campaign subject (T1b), the diarized
    speaker's face, the main track, the face nearest to the last target, the largest face. With no
    face the window holds its last target (center before the first face).
    """
    main = main_track(samples)
    targets: list[tuple[float, float, float]] = []
    followed: list[tuple[float, float, float, float]] = []
    jumps: list[tuple[float, float]] = []
    seen: set[int] = set()
    last: tuple[float, float] | None = None
    for sample in samples:
        fresh = {f.track_id for f in sample.faces if f.track_id is not None} - seen
        seen |= fresh
        spk = speaker_at(sample.t, speakers)
        tid = mapping.get(spk) if spk else None
        # T1b: the campaign subject first; without it in shot, whoever speaks.
        subject = [f for f in sample.faces if f.subject]
        face = max(subject, key=lambda f: (f.subject_score, f.area)) if subject else None
        for wanted in (tid, main):
            if face is None and wanted is not None:
                face = next((f for f in sample.faces if f.track_id == wanted), None)
        if face is None and sample.faces:
            face = _nearest(sample.faces, *last) if last else max(sample.faces, key=lambda f: f.area)
        if face is None and use_objects and sample.objects:
            obj = max(sample.objects, key=lambda item: item.score * item.area)
            targets.append((sample.t, obj.x, obj.y))
            last = (obj.x, obj.y)
            continue
        if face is None:
            cx, cy = last or _center_of(src_w, src_h)
            targets.append((sample.t, cx, cy))
            continue
        if last is not None and face.track_id in fresh:
            jumps.append((sample.t, math.hypot(face.x - last[0], face.y - last[1])))
        targets.append((sample.t, face.x, face.y))
        followed.append((sample.t, face.x, face.y, face.h))
        last = (face.x, face.y)
    if not targets:
        targets = [(0.0, *_center_of(src_w, src_h))]
    return targets, followed, jumps


def zoom_for(face_heights: list[float], crop_h: int, zoom: dict[str, Any]) -> float:
    """Zoom so the median face is ``target_face_frac`` of the crop (1.0 = none)."""
    if not zoom.get("enabled") or not face_heights or crop_h <= 0:
        return 1.0
    ordered = sorted(face_heights)
    median = ordered[len(ordered) // 2]
    if median <= 0:
        return 1.0
    want = float(zoom["target_face_frac"]) * crop_h / median
    limit = min(float(zoom["max_zoom"]), crop_h / max(1.0, float(zoom["min_crop_h"])))
    value = min(want, limit)
    return round(value, 3) if value > 1.02 else 1.0


def centered_ratio(
    followed: list[tuple[float, float, float, float]],
    origins: list[tuple[int, int]],
    times: list[float],
    crop_w: int,
    crop_h: int,
    *,
    band: float = 0.25,
) -> float | None:
    """Share of followed faces whose center sits in the middle half of the window."""
    if not followed or not origins:
        return None
    hits = 0
    for t, x, y, _h in followed:
        index = min(range(len(times)), key=lambda i: abs(times[i] - t))
        ox, oy = origins[index]
        if abs(x - (ox + crop_w / 2)) <= band * crop_w and oy <= y <= oy + crop_h:
            hits += 1
    return hits / len(followed)


def _targets_for(
    name: str,
    samples: list[Sample],
    *,
    layout: str,
    speakers: list[dict],
    mapping: dict[str, int],
    src_w: int,
    src_h: int,
    screen_box: Det | None,
) -> list[tuple[float, float, float]]:
    static = _static_centers(layout, src_w, src_h)
    if name == "screen":
        if screen_box is not None:
            return [(0.0, screen_box.x, screen_box.y)]
        return [(0.0, *static["screen"])]
    if name == "fit":
        return [(0.0, *_center_of(src_w, src_h))]
    track_id: int | None = None
    if layout == "split":
        ordered = list(mapping.items())
        if name == "ptop" and ordered:
            track_id = ordered[0][1]
        elif name == "pbot" and len(ordered) > 1:
            track_id = ordered[1][1]
        elif name == "pbot" and ordered:
            track_id = ordered[0][1]
    targets: list[tuple[float, float, float]] = []
    for sample in samples:
        if layout == "single" or name == "face":
            spk = speaker_at(sample.t, speakers)
            tid = mapping.get(spk) if spk else None
            face = _face_at(sample, tid)
        else:
            face = _face_at(sample, track_id)
        if face is None and sample.objects and name == "single":
            obj = max(sample.objects, key=lambda item: item.score * item.area)
            targets.append((sample.t, obj.x, obj.y))
            continue
        if face is None:
            cx, cy = static.get(name, _center_of(src_w, src_h))
            targets.append((sample.t, cx, cy))
        else:
            targets.append((sample.t, face.x, face.y))
    if not targets:
        cx, cy = static.get(name, _center_of(src_w, src_h))
        targets = [(0.0, cx, cy)]
    return targets


def _screen_box(samples: list[Sample], src_w: int, src_h: int) -> Det | None:
    frame = float(src_w * src_h) or 1.0
    best: Det | None = None
    for sample in samples:
        for obj in sample.objects:
            if (obj.label or "").lower() not in SCREEN_LABELS:
                continue
            if obj.area < 0.12 * frame:
                continue
            if best is None or obj.area > best.area:
                best = obj
    return best


def _sendcmd(events: list[tuple[float, str, int, int]]) -> str:
    """One sendcmd line per timestamp. Several crops that move together share it."""
    last: dict[str, tuple[int, int]] = {}
    grouped: dict[float, list[str]] = {}
    order: list[float] = []
    for t, target, x, y in events:
        prev = last.get(target)
        if prev == (x, y):
            continue
        last[target] = (x, y)
        key = round(float(t), 3)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(f"{target} x {int(x)}, {target} y {int(y)}")
    lines = [f"{key:.3f} " + ", ".join(grouped[key]) + ";" for key in order]
    return ("\n".join(lines) + "\n") if lines else ""


def _chain_label(src: str, body: str, out: str) -> str:
    return f"{src}{body}{out}"


def _layout_chain(
    layout: str,
    *,
    fmt: str,
    src_label: str,
    paths: dict[str, dict],
    screen: Det | None,
    src_w: int,
    src_h: int,
    out_label: str,
) -> str:
    ow, oh = output_size(fmt)
    slots = {name: (cw, ch, ow_s, oh_s) for name, cw, ch, ow_s, oh_s in (
        (name, *paths[name]["crop"], * _slot_pair(fmt, layout, name))
        for name in paths if paths[name]["layout"] == layout
    )}
    if layout == "fit":
        return (
            f"{src_label}split=2[bg][fg];"
            f"[bg]scale={ow}:{oh}:force_original_aspect_ratio=increase,crop={ow}:{oh},boxblur=20:2[bgb];"
            f"[fg]scale={ow}:{oh}:force_original_aspect_ratio=decrease[fgf];"
            f"[bgb][fgf]overlay=(W-w)/2:(H-h)/2{out_label}"
        )
    if layout == "single":
        cw, ch, slot_w, slot_h = slots["single"]
        x0, y0 = paths["single"]["origins"][0]
        return _chain_label(
            src_label,
            f"crop@single={cw}:{ch}:{x0}:{y0},scale={slot_w}:{slot_h},setsar=1",
            out_label,
        )
    if layout == "split":
        horizontal = fmt == "16:9"
        parts = []
        outs = []
        for key, tag in (("ptop", "a"), ("pbot", "b")):
            cw, ch, slot_w, slot_h = slots[key]
            x0, y0 = paths[key]["origins"][0]
            branch = f"[{tag}]"
            outs.append(branch)
            parts.append(
                f"{branch}crop@{key}={cw}:{ch}:{x0}:{y0},{_scale_pad(slot_w, slot_h)}[{key}]"
            )
        stack = "hstack" if horizontal else "vstack"
        return (
            f"{src_label}split=2{outs[0]}{outs[1]};"
            + ";".join(parts)
            + f";[ptop][pbot]{stack}{out_label}"
        )
    # screenshare
    face_w_crop, face_h_crop, face_ow, face_oh = slots["face"]
    screen_ow, screen_oh = slots["screen"][2], slots["screen"][3]
    x0, y0 = paths["face"]["origins"][0]
    if screen is not None:
        sw, sh = _crop_for_slot(src_w, src_h, screen_ow, screen_oh)
        sx, sy = _origin(screen.x, screen.y, sw, sh, src_w, src_h)
        screen_body = f"crop@screen={sw}:{sh}:{sx}:{sy},scale={screen_ow}:{screen_oh}:force_original_aspect_ratio=increase,crop={screen_ow}:{screen_oh}[screen]"
    else:
        screen_body = (
            f"scale={screen_ow}:{screen_oh}:force_original_aspect_ratio=increase,"
            f"crop={screen_ow}:{screen_oh}[screen]"
        )
    stack = "hstack" if fmt == "16:9" else "vstack"
    # Screen takes the main pane (top, or left on 16:9); the face takes the other.
    return (
        f"{src_label}split=2[scr][fac];"
        f"[scr]{screen_body};"
        f"[fac]crop@face={face_w_crop}:{face_h_crop}:{x0}:{y0},{_scale_pad(face_ow, face_oh)}[face];"
        f"[screen][face]{stack}{out_label}"
    )


def _slot_pair(fmt: str, layout: str, name: str) -> tuple[int, int]:
    for slot_name, w, h in _slot_sizes(fmt, layout):
        if slot_name == name:
            return w, h
    ow, oh = output_size(fmt)
    return ow, oh


def _enable_expr(spans: list[tuple[float, float]]) -> str:
    parts = [f"between(t,{start:.3f},{end:.3f})" for start, end in spans if end > start]
    return "+".join(parts) if parts else "0"


def _build_graph(
    spans: list[dict],
    paths: dict[str, dict],
    *,
    fmt: str,
    src_w: int,
    src_h: int,
    screen: Det | None,
    duration: float,
) -> tuple[list[str] | None, str | None, list[tuple[float, str, int, int]]]:
    """Return (vf, complex, sendcmd events). One of vf/complex is set."""
    layouts = []
    for span in spans:
        if span["layout"] not in layouts:
            layouts.append(span["layout"])
    events: list[tuple[float, str, int, int]] = []
    for name, path in paths.items():
        target = "crop" if name == "single" and layouts == ["single"] else f"crop@{name}"
        if name in ("fit", "screen"):
            continue
        for t, (x, y) in zip(path["times"], path["origins"]):
            events.append((t, target, x, y))

    if layouts == ["single"]:
        cw, ch = paths["single"]["crop"]
        ow, oh = output_size(fmt)
        x0, y0 = paths["single"]["origins"][0]
        # Exact scale, not decrease+pad: even-pixel rounding of the crop left
        # 1-2 px black lines top/bottom. The aspect error is under 0.3%.
        vf = [
            "sendcmd=f=reframe.cmd",
            f"crop={cw}:{ch}:{x0}:{y0}",
            f"scale={ow}:{oh}",
            "setsar=1",
        ]
        # Untargeted-style commands use the filter name `crop` (one crop only).
        return vf, None, [(t, "crop", x, y) for t, _target, x, y in events]

    if len(layouts) == 1:
        layout = layouts[0]
        graph = _layout_chain(
            layout, fmt=fmt, src_label="[0:v]sendcmd=f=reframe.cmd," if events else "[0:v]",
            paths=paths, screen=screen, src_w=src_w, src_h=src_h, out_label="[vmain]",
        )
        # _layout_chain receives src_label including sendcmd only when we prefix
        # [0:v]. The helper expects src_label like "[0:v]" or "[s0]" and then
        # writes filters. Passing a sendcmd prefix works for single/fit because
        # they start with src_label immediately. For split, src_label is
        # followed by split=. Prefixing sendcmd there is correct:
        # [0:v]sendcmd=f=reframe.cmd,split=2...
        if not events and "sendcmd" in graph:
            graph = graph.replace("sendcmd=f=reframe.cmd,", "")
        return None, graph, events

    # More than one layout over the clip: one chain each, overlay to switch.
    base = spans[0]["layout"]
    pieces = ["[0:v]sendcmd=f=reframe.cmd,split=" + str(len(layouts)) + "".join(
        f"[s{i}]" for i in range(len(layouts))
    )]
    labels = {}
    for index, layout in enumerate(layouts):
        out = f"[{layout}]"
        labels[layout] = out
        pieces.append(_layout_chain(
            layout, fmt=fmt, src_label=f"[s{index}]", paths=paths, screen=screen,
            src_w=src_w, src_h=src_h, out_label=out,
        ))
    label = labels[base]
    step = 0
    for layout in layouts:
        if layout == base:
            continue
        intervals = [(span["start"], span["end"]) for span in spans if span["layout"] == layout]
        enable = _enable_expr(intervals)
        nxt = f"[o{step}]"
        pieces.append(f"{label}{labels[layout]}overlay=enable='{enable}'{nxt}")
        label = nxt
        step += 1
    # Rename the last label to [vmain] by a null scale (setsar keeps the frames).
    if label != "[vmain]":
        pieces.append(f"{label}null[vmain]")
    return None, ";".join(pieces), events


def _path_is_center(origins: list[tuple[int, int]], src_w: int, src_h: int, cw: int, ch: int) -> bool:
    x = int(round((src_w - cw) / 2))
    y = int(round((src_h - ch) / 2))
    x = int(_clamp(x, 0, max(0, src_w - cw)))
    y = int(_clamp(y, 0, max(0, src_h - ch)))
    return all(abs(ox - x) <= 1 and abs(oy - y) <= 1 for ox, oy in origins)


def _in_frame_ratio(
    samples: list[Sample],
    speakers: list[dict],
    mapping: dict[str, int],
    origins: list[tuple[int, int]],
    times: list[float],
    crop_w: int,
    crop_h: int,
) -> float | None:
    if not samples or not origins:
        return None
    hits = 0
    total = 0
    for sample in samples:
        spk = speaker_at(sample.t, speakers)
        face = _face_at(sample, mapping.get(spk) if spk else None)
        if face is None:
            continue
        total += 1
        index = min(range(len(times)), key=lambda i: abs(times[i] - sample.t))
        x, y = origins[index]
        if x <= face.x <= x + crop_w and y <= face.y <= y + crop_h:
            hits += 1
    if total == 0:
        return None
    return hits / total


def _compile(
    samples: list[Sample],
    cfg: dict[str, Any],
    *,
    src_w: int,
    src_h: int,
    fmt: str,
    duration: float,
    fps: float,
) -> dict[str, Any] | None:
    """Return graph inputs, or None when the result is the legacy center crop."""
    smoothing: Smoothing = cfg["smoothing"]
    speakers: list[dict] = cfg["speakers"]
    mapping = map_speakers(samples, speakers)
    two_speakers = len({str(seg["speaker"]) for seg in speakers}) >= 2 and any(
        len(sample.faces) >= 2 for sample in samples
    )
    if cfg["layout"] == "auto":
        labels = [
            (sample.t, classify_sample(sample, src_w=src_w, src_h=src_h, two_speakers=two_speakers))
            for sample in samples
        ] or [(0.0, "single")]
        spans = coalesce_layouts(labels, hold_s=cfg["hold_s"], duration=duration)
    else:
        spans = [{"start": 0.0, "end": round(duration, 3), "layout": cfg["layout"]}]

    if cfg["mode"] == "face" and cfg["layout"] == "auto":
        # Face mode never shrinks the shot into a blurred backdrop on its own.
        merged: list[dict] = []
        for span in spans:
            span = dict(span, layout="single" if span["layout"] == "fit" else span["layout"])
            if merged and merged[-1]["layout"] == span["layout"]:
                merged[-1]["end"] = span["end"]
            else:
                merged.append(span)
        spans = merged

    used = []
    for span in spans:
        if span["layout"] not in used:
            used.append(span["layout"])
    if used == ["single"] and cfg["mode"] == "center":
        return None
    if used == ["fit"] and not samples and cfg["mode"] == "center":
        pass

    screen = _screen_box(samples, src_w, src_h)
    times = _output_times(duration, fps)
    paths: dict[str, dict] = {}
    max_shift = 0.0
    observed = 0.0
    zoom = 1.0
    followed: list[tuple[float, float, float, float]] = []
    cuts: list[float] = []
    for layout in used:
        for name, slot_w, slot_h in _slot_sizes(fmt, layout):
            if name == "fit":
                continue
            cw, ch = _crop_for_slot(src_w, src_h, slot_w, slot_h)
            if name == "single":
                targets, followed, jumps = _subject_targets(
                    samples, speakers=speakers, mapping=mapping, src_w=src_w, src_h=src_h,
                    use_objects=cfg["mode"] != "face",
                )
                zoom = zoom_for([h for *_rest, h in followed], ch, cfg["zoom"])
                if zoom > 1.0:
                    cw, ch = _even(cw / zoom), _even(ch / zoom)
                if cfg["mode"] == "face":
                    # Eyes a bit above the middle: the face sits at ~40% of the window height.
                    targets = [(t, x, y + ch * 0.10) for t, x, y in targets]
                    # New shot far away: jump there instead of panning across.
                    cuts = [t for t, dist in jumps if dist > 0.5 * cw]
            else:
                targets = _targets_for(
                    name, samples, layout=layout, speakers=speakers, mapping=mapping,
                    src_w=src_w, src_h=src_h, screen_box=screen,
                )
            slot_cuts = cuts if name == "single" else []
            origins, cap = smooth_origins(
                targets, times, crop_w=cw, crop_h=ch, src_w=src_w, src_h=src_h, smoothing=smoothing,
                cuts=slot_cuts,
            )
            paths[name] = {
                "layout": layout,
                "crop": (cw, ch),
                "origins": origins,
                "times": times,
                "cap": cap,
            }
            if name in ("single", "face"):
                max_shift = cap
                observed = max(observed, _max_step(origins, cut_indices(times, slot_cuts)))
            elif layout == "split":
                max_shift = max(max_shift, cap)
                observed = max(observed, _max_step(origins))

    # A single window that never leaves the center is the historical crop.
    if used == ["single"] and "single" in paths and fmt in ("9:16", "16:9") and zoom == 1.0:
        cw, ch = paths["single"]["crop"]
        if _path_is_center(paths["single"]["origins"], src_w, src_h, cw, ch):
            return None

    centered = None
    present = on_screen = None
    if "single" in paths:
        cw, ch = paths["single"]["crop"]
        centered = centered_ratio(followed, paths["single"]["origins"], times, cw, ch)
        if cfg.get("subject"):
            present, on_screen = subject_on_screen(samples, paths["single"]["origins"], times, cw, ch)

    ratio = None
    if "single" in paths:
        cw, ch = paths["single"]["crop"]
        ratio = _in_frame_ratio(samples, speakers, mapping, paths["single"]["origins"], times, cw, ch)
    elif "face" in paths:
        cw, ch = paths["face"]["crop"]
        ratio = _in_frame_ratio(samples, speakers, mapping, paths["face"]["origins"], times, cw, ch)

    vf, graph, events = _build_graph(
        spans, paths, fmt=fmt, src_w=src_w, src_h=src_h, screen=screen, duration=duration,
    )
    return {
        "vf": vf,
        "graph": graph,
        "events": events,
        "spans": spans,
        "mapping": {str(k): int(v) for k, v in mapping.items()},
        "max_shift_px": round(max_shift, 3),
        "max_shift_observed_px": round(observed, 3),
        "speaker_in_frame": None if ratio is None else round(ratio, 4),
        "crop": paths.get("single", paths.get("face", {})).get("crop"),
        "zoom": zoom,
        "main_track": main_track(samples),
        "subject_centered": None if centered is None else round(centered, 4),
        "subject_frames": len(followed),
        "cuts": [round(t, 3) for t in cuts],
        "subject_present": present,
        "subject_on_screen": on_screen,
    }


def _gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"available": False, "name": None, "device": "cpu", "note": _GPU_NOTE}
    try:
        import torch
    except ImportError:
        return info
    if torch.cuda.is_available():
        info["available"] = True
        info["device"] = "cuda"
        try:
            info["name"] = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001
            info["name"] = "cuda"
    return info


def _ensure_face_model(model_dir: Path) -> Path | None:
    baked = os.environ.get("REFRAME_FACE_MODEL", "").strip()
    if baked and Path(baked).is_file() and Path(baked).stat().st_size > 0:
        return Path(baked)
    dest = model_dir / _FACE_MODEL_NAME
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    model_dir.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(_FACE_MODEL_URL, dest)  # noqa: S310 — fixed MediaPipe URL
    except Exception as exc:  # noqa: BLE001
        logger.warning("face model download failed", error=str(exc))
        return None
    return dest if dest.is_file() else None


def _open_face_detector(model_dir: Path):
    try:
        import mediapipe as mp  # noqa: F401
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions
    except (ImportError, OSError):
        logger.info("mediapipe is not installed; face tracking unavailable")
        return None
    model = _ensure_face_model(model_dir)
    if model is None:
        return None
    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        min_detection_confidence=0.5,
    )
    try:
        return FaceDetector.create_from_options(options)
    except Exception as exc:  # noqa: BLE001 — e.g. libEGL missing: fall back to center, do not drop the plan
        logger.warning("face detector unavailable", error=str(exc))
        return None


def _detect_faces(detector, rgb, det_w: int, det_h: int, src_w: int, src_h: int, t: float) -> list[Det]:
    import mediapipe as mp
    import numpy as np

    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.frombuffer(rgb, dtype=np.uint8).reshape(det_h, det_w, 3))
    result = detector.detect(image)
    sx = src_w / det_w
    sy = src_h / det_h
    dets: list[Det] = []
    for found in result.detections or []:
        box = found.bounding_box
        score = 0.0
        if found.categories:
            score = float(found.categories[0].score or 0)
        w = float(box.width) * sx
        h = float(box.height) * sy
        x = (float(box.origin_x) + float(box.width) / 2) * sx
        y = (float(box.origin_y) + float(box.height) / 2) * sy
        dets.append(Det(t=t, x=x, y=y, w=w, h=h, score=score, kind="face", label="face"))
    return dets


def _mouth_score(rgb: bytes, det_w: int, det_h: int, det: Det, src_w: int, src_h: int, prev: dict[int, Any]) -> float:
    """Mean abs diff of the lower 40% of the face (mouth band) versus the previous sample."""
    import numpy as np

    frame = np.frombuffer(rgb, dtype=np.uint8).reshape(det_h, det_w, 3)
    sx = det_w / src_w
    sy = det_h / src_h
    x0 = int((det.x - det.w / 2) * sx)
    y0 = int((det.y - det.h / 2) * sy)
    x1 = int((det.x + det.w / 2) * sx)
    y1 = int((det.y + det.h / 2) * sy)
    mouth_y0 = y0 + int((y1 - y0) * 0.55)
    x0, y0 = max(0, x0), max(0, mouth_y0)
    x1, y1 = min(det_w, x1), min(det_h, y1)
    if x1 - x0 < 4 or y1 - y0 < 4 or det.track_id is None:
        return 0.0
    patch = frame[y0:y1, x0:x1].astype("float32")
    # Mean-pool to 16x8 so the score does not depend on face size.
    ys = np.linspace(0, patch.shape[0], 9, dtype=int)
    xs = np.linspace(0, patch.shape[1], 17, dtype=int)
    small = np.zeros((8, 16), dtype="float32")
    for iy in range(8):
        for ix in range(16):
            block = patch[ys[iy]:max(ys[iy] + 1, ys[iy + 1]), xs[ix]:max(xs[ix] + 1, xs[ix + 1])]
            if block.size:
                small[iy, ix] = float(block.mean())
    previous = prev.get(det.track_id)
    prev[det.track_id] = small
    if previous is None or previous.shape != small.shape:
        return 0.0
    return float(np.abs(small - previous).mean()) / 255.0


class _YoloTracker:
    def __init__(self, model_dir: Path, device: str):
        from ultralytics import YOLO

        model_dir.mkdir(parents=True, exist_ok=True)
        self.model = YOLO(str(model_dir / _YOLO_MODEL))
        self.device = device
        self.names: dict[int, str] = {}

    def detect(self, rgb: bytes, det_w: int, det_h: int, src_w: int, src_h: int, t: float) -> list[Det]:
        import numpy as np

        frame = np.frombuffer(rgb, dtype=np.uint8).reshape(det_h, det_w, 3)
        results = self.model.track(
            frame, persist=True, tracker="bytetrack.yaml", verbose=False, device=self.device, conf=0.25,
        )
        if not results:
            return []
        boxes = results[0].boxes
        self.names = results[0].names or self.names
        if boxes is None or boxes.xyxy is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.detach().cpu().numpy()
        conf = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else [0.5] * len(xyxy)
        cls = boxes.cls.detach().cpu().numpy() if boxes.cls is not None else [0] * len(xyxy)
        ids = boxes.id.detach().cpu().numpy() if boxes.id is not None else [None] * len(xyxy)
        sx = src_w / det_w
        sy = src_h / det_h
        dets: list[Det] = []
        for box, score, kind, track in zip(xyxy, conf, cls, ids):
            label = str(self.names.get(int(kind), "object"))
            x0, y0, x1, y1 = [float(v) for v in box]
            det = Det(
                t=t,
                x=(x0 + x1) / 2 * sx,
                y=(y0 + y1) / 2 * sy,
                w=(x1 - x0) * sx,
                h=(y1 - y0) * sy,
                score=float(score),
                kind="face" if label == "person" else "object",
                label=label,
                track_id=None if track is None else int(track),
            )
            # A YOLO person is a body. Keep it as an object unless face mode has
            # nothing else; callers treat kind=face as a face box. Bodies are
            # objects so they do not override a real face box.
            if label == "person":
                det.kind = "object"
                det.label = "person"
            dets.append(det)
        return dets


def _iter_frames(ffmpeg_bin: str, path: Path, start: float, duration: float, fps: float,
                 src_w: int, src_h: int, det_w: int = 640):
    det_w = _even(min(det_w, src_w))
    det_h = _even(src_h * det_w / src_w)
    frame_bytes = det_w * det_h * 3
    cmd = [
        ffmpeg_bin, "-v", "error", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
        "-i", str(path), "-vf", f"fps={fps},scale={det_w}:{det_h}",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    index = 0
    try:
        while True:
            buf = bytearray()
            while len(buf) < frame_bytes:
                chunk = proc.stdout.read(frame_bytes - len(buf))
                if not chunk:
                    break
                buf.extend(chunk)
            if len(buf) < frame_bytes:
                break
            yield index / fps, bytes(buf), det_w, det_h
            index += 1
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)


def _collect(
    *,
    input_path: Path,
    start: float,
    duration: float,
    cfg: dict[str, Any],
    src_w: int,
    src_h: int,
    ffmpeg_bin: str,
    model_dir: Path,
) -> tuple[list[Sample], str, str | None, dict[str, Any]]:
    """Return samples, detector name, error, gpu info."""
    if "detections" in cfg:
        dets = _parse_detections(cfg["detections"])
        return _samples_from_dets(dets), "injected", None, {"available": False, "device": "cpu", "name": None}
    if cfg["mode"] == "center":
        return [], "none", None, {"available": False, "device": "cpu", "name": None}

    gpu = _gpu_info()
    want_faces = cfg["mode"] in ("face", "auto")
    want_objects = cfg["mode"] in ("object", "auto") or cfg["layout"] in ("screenshare", "auto")
    face_detector = _open_face_detector(model_dir) if want_faces else None
    yolo = None
    yolo_error = None
    if want_objects:
        try:
            yolo = _YoloTracker(model_dir, "0" if gpu.get("device") == "cuda" else "cpu")
        except Exception as exc:  # noqa: BLE001
            yolo_error = str(exc)
            logger.warning("yolo unavailable", error=yolo_error)
    if face_detector is None and yolo is None:
        reason = yolo_error or "detector_unavailable"
        return [], "unavailable", reason, gpu

    samples: list[Sample] = []
    mouth_prev: dict[int, Any] = {}
    names = []
    if face_detector is not None:
        names.append("mediapipe")
    if yolo is not None:
        names.append("yolo+bytetrack")
    try:
        pending: list[Det] = []
        for t, rgb, det_w, det_h in _iter_frames(
            ffmpeg_bin, input_path, start, duration, cfg["sample_fps"], src_w, src_h,
        ):
            frame_dets: list[Det] = []
            if face_detector is not None:
                frame_dets.extend(_detect_faces(face_detector, rgb, det_w, det_h, src_w, src_h, t))
            if yolo is not None:
                frame_dets.extend(yolo.detect(rgb, det_w, det_h, src_w, src_h, t))
            pending.extend(frame_dets)
            # Mouth scores need track ids, so score after a per-frame association below.
            sample = Sample(t=round(t, 3))
            for det in frame_dets:
                if det.kind == "face":
                    sample.faces.append(det)
                else:
                    sample.objects.append(det)
            samples.append(sample)
        assign_track_ids([det for sample in samples for det in sample.faces])
        # Second pass for mouth motion. Frames are gone, so mouth stays 0 unless
        # the caller injected it. Real lip motion is computed in the loop when
        # we already know a provisional id. Re-run is not possible without the
        # pixels. Score it inside the loop instead — handled below if we still
        # have rgb. We don't. See _collect_scored.
    except Exception as exc:  # noqa: BLE001
        logger.warning("reframe sampling failed", error=str(exc))
        return [], "unavailable", str(exc), gpu
    finally:
        close = getattr(face_detector, "close", None)
        if close:
            close()
    return samples, "+".join(names), None, gpu


def _collect_scored(**kwargs) -> tuple[list[Sample], str, str | None, dict[str, Any]]:
    """Like ``_collect`` but scores mouth motion while the frame is still in memory."""
    cfg = kwargs["cfg"]
    if "detections" in cfg or cfg["mode"] == "center":
        return _collect(**kwargs)
    gpu = _gpu_info()
    want_faces = cfg["mode"] in ("face", "auto")
    want_objects = cfg["mode"] in ("object", "auto") or cfg["layout"] in ("screenshare", "auto")
    model_dir: Path = kwargs["model_dir"]
    face_detector = _open_face_detector(model_dir) if want_faces else None
    yolo = None
    yolo_error = None
    if want_objects:
        try:
            yolo = _YoloTracker(model_dir, "0" if gpu.get("device") == "cuda" else "cpu")
        except Exception as exc:  # noqa: BLE001
            yolo_error = str(exc)
            logger.warning("yolo unavailable", error=yolo_error)
    if face_detector is None and yolo is None:
        return [], "unavailable", yolo_error or "detector_unavailable", gpu

    names = []
    if face_detector is not None:
        names.append("mediapipe")
    if yolo is not None:
        names.append("yolo+bytetrack")
    samples: list[Sample] = []
    active: list[Det] = []
    next_id = 1
    mouth_prev: dict[int, Any] = {}
    fid, refs = _open_subject(cfg, model_dir) if face_detector is not None else (None, [])
    cfg["_subject_refs"] = len(refs)
    last_id = -1e9
    try:
        for t, rgb, det_w, det_h in _iter_frames(
            kwargs["ffmpeg_bin"], kwargs["input_path"], kwargs["start"], kwargs["duration"],
            cfg["sample_fps"], kwargs["src_w"], kwargs["src_h"], det_w=1280 if refs else 640,
        ):
            faces = _detect_faces(face_detector, rgb, det_w, det_h, kwargs["src_w"], kwargs["src_h"], t) if face_detector else []
            if refs and faces and t - last_id >= SUBJECT_EVERY_S - 1e-6:
                last_id = t
                _identify(fid, refs, rgb, det_w, det_h, faces, kwargs["src_w"], kwargs["src_h"])
            objects = yolo.detect(rgb, det_w, det_h, kwargs["src_w"], kwargs["src_h"], t) if yolo else []
            used: set[int] = set()
            for det in faces:
                best_id = None
                best_iou = 0.3
                for prev in active:
                    if prev.track_id in used:
                        continue
                    score = _box_iou(prev, det)
                    if score > best_iou:
                        best_iou = score
                        best_id = prev.track_id
                if best_id is None:
                    best_id = next_id
                    next_id += 1
                det.track_id = best_id
                used.add(best_id)
                det.mouth = _mouth_score(rgb, det_w, det_h, det, kwargs["src_w"], kwargs["src_h"], mouth_prev)
            active = faces
            sample = Sample(t=round(t, 3), faces=faces, objects=objects)
            samples.append(sample)
    except Exception as exc:  # noqa: BLE001
        logger.warning("reframe sampling failed", error=str(exc))
        return [], "unavailable", str(exc), gpu
    finally:
        close = getattr(face_detector, "close", None)
        if close:
            close()
    return samples, "+".join(names), None, gpu


def _open_subject(cfg: dict[str, Any], model_dir: Path) -> tuple[Any, list[Any]]:
    """Face-id models plus reference embeddings, or (None, []) when unavailable."""
    subject = cfg.get("subject")
    if not subject:
        return None, []
    from app.tools.face_id import open_face_id

    fid = open_face_id(model_dir)
    if fid is None:
        return None, []
    refs = fid.reference_features([Path(p) for p in subject["reference_paths"]])
    return (fid, refs) if refs else (None, [])


def _identify(fid, refs: list[Any], rgb: bytes, det_w: int, det_h: int, faces: list[Det],
              src_w: int, src_h: int) -> None:
    """Set ``ident`` on the MediaPipe faces that YuNet/SFace also see in this frame."""
    import numpy as np

    bgr = np.ascontiguousarray(np.frombuffer(rgb, np.uint8).reshape(det_h, det_w, 3)[:, :, ::-1])
    sx, sy = src_w / det_w, src_h / det_h
    for (x, y, w, h), feature in fid.faces(bgr):
        cx, cy = (x + w / 2) * sx, (y + h / 2) * sy
        hits = [f for f in faces if abs(f.x - cx) <= f.w * 0.6 and abs(f.y - cy) <= f.h * 0.6]
        if not hits:
            continue
        face = min(hits, key=lambda f: (f.x - cx) ** 2 + (f.y - cy) ** 2)
        sim = fid.similarity(feature, refs)
        face.ident = sim if face.ident is None else max(face.ident, sim)


def _base_meta(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": cfg["mode"],
        "layout": cfg["layout"],
        "smoothing": cfg["smoothing"].as_dict,
        "sample_fps": cfg["sample_fps"],
        "legacy": False,
        "fallback": None,
        "detector": "none",
        "gpu": {"available": False, "device": "cpu", "name": None, "note": _GPU_NOTE},
        "timings": {},
        "layouts": [],
        "speaker_map": {},
        "speaker_in_frame": None,
        "max_shift_px": None,
        "max_shift_observed_px": None,
        "zoom": 1.0,
        "main_track": None,
        "subject_centered": None,
        "note": _GPU_NOTE,
    }


def plan_reframe(
    *,
    input_path: Path,
    start: float,
    end: float,
    output_format: str,
    reframe: dict | None,
    src_w: int,
    src_h: int,
    output_fps: float,
    work_dir: Path,
    ffmpeg_bin: str = "ffmpeg",
    model_dir: Path | None = None,
) -> ReframePlan:
    cfg = normalize_reframe(reframe)
    meta = _base_meta(cfg)
    duration = max(0.0, end - start)
    legacy = ReframePlan(True, None, None, meta, [])

    if output_format not in OUTPUT_SIZES:
        meta["legacy"] = True
        meta["fallback"] = "unsupported_format"
        return legacy
    if cfg["mode"] == "center" and cfg["layout"] in ("single", "auto"):
        meta["legacy"] = True
        meta["layouts"] = [{"start": 0.0, "end": round(duration, 3), "layout": "single"}]
        return legacy
    if src_w < 2 or src_h < 2:
        meta["legacy"] = True
        meta["fallback"] = "no_source_size"
        return legacy

    t0 = time.perf_counter()
    samples, detector, error, gpu = _collect_scored(
        input_path=input_path, start=start, duration=duration, cfg=cfg,
        src_w=src_w, src_h=src_h, ffmpeg_bin=ffmpeg_bin,
        model_dir=model_dir or (work_dir / "models"),
    )
    meta["timings"]["detect_s"] = round(time.perf_counter() - t0, 3)
    meta["detector"] = detector
    meta["gpu"] = gpu
    meta["frames_sampled"] = len(samples)
    if error and cfg["layout"] in ("single", "auto") and cfg["mode"] in ("face", "object", "auto"):
        # Missing optional detector must not change today's center crop.
        meta["legacy"] = True
        meta["fallback"] = error
        meta["layouts"] = [{"start": 0.0, "end": round(duration, 3), "layout": "single"}]
        return legacy

    subject_cfg = cfg.get("subject")
    if subject_cfg:
        refs = cfg.get("_subject_refs")
        meta["subject"] = {
            "name": subject_cfg["name"],
            "references": refs,
            "tracks": label_subject_tracks(samples, subject_cfg["threshold"]),
            "present": None,
            "on_screen": None,
            "error": "unavailable" if refs == 0 else None,
        }
    compiled = _compile(
        samples, cfg, src_w=src_w, src_h=src_h, fmt=output_format,
        duration=duration, fps=output_fps or 30.0,
    )
    if compiled is None:
        meta["legacy"] = True
        meta["fallback"] = meta["fallback"] or "center_path"
        meta["layouts"] = [{"start": 0.0, "end": round(duration, 3), "layout": "single"}]
        return legacy

    meta["layouts"] = compiled["spans"]
    meta["speaker_map"] = compiled["mapping"]
    meta["speaker_in_frame"] = compiled["speaker_in_frame"]
    meta["max_shift_px"] = compiled["max_shift_px"]
    meta["max_shift_observed_px"] = compiled["max_shift_observed_px"]
    meta["zoom"] = compiled["zoom"]
    meta["main_track"] = compiled["main_track"]
    meta["subject_centered"] = compiled["subject_centered"]
    meta["subject_frames"] = compiled["subject_frames"]
    meta["cuts"] = compiled["cuts"]
    if subject_cfg and compiled["subject_on_screen"] is not None:
        meta["subject"]["present"] = round(compiled["subject_present"], 4)
        meta["subject"]["on_screen"] = round(compiled["subject_on_screen"], 4)
    if compiled["crop"]:
        meta["crop"] = {"w": compiled["crop"][0], "h": compiled["crop"][1]}

    files: list[Path] = []
    if compiled["events"]:
        work_dir.mkdir(parents=True, exist_ok=True)
        cmd_path = work_dir / "reframe.cmd"
        cmd_path.write_text(_sendcmd(compiled["events"]), encoding="ascii")
        files.append(cmd_path)
    meta["legacy"] = False
    return ReframePlan(False, compiled["vf"], compiled["graph"], meta, files)
