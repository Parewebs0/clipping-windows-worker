"""Screen recording with a webcam inset -> 9:16 with the screen on top and the webcam below.

Campaign 58 (PBInvesting) style: a full-screen trading chart with the
speaker's webcam composited into a corner. The 9:16 centre crop slices the
chart and face mode zooms on the corner, so neither works. This layout keeps
the whole frame (16:9) at full width in the upper half and the webcam window,
cut out exactly, enlarged underneath, over a blurred copy of the source.
Static per shot: no tracking (the inset does not move).

Per shot (cuts from ``reframe.detect_scene_cuts``) a few frames are sampled,
faces are searched for in each corner quadrant (so a small webcam face is big
enough for BlazeFace), the webcam rectangle is found from the straight edges
of the window around the face, and the shot counts as an inset only when most
samples agree on the same rectangle. Anything uncertain falls back to the
historical centred crop, never to something that moves.

Portions adapted from OpenShorts ``camera_inset.py`` and ``ffmpeg_utils.py``
(https://github.com/mutonby/openshorts): the corner/offset/size gates
(``is_cornered`` and its constants), the agree-around-the-median stability
gate (``stable_box``) and the quarter-size blurred backdrop. Used under the
MIT License:

    MIT License

    Copyright (c) 2024 OpenShorts

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from app.utils.logging import get_logger

logger = get_logger("inset_layout")

# --- Gates (OpenShorts camera_inset.py, MIT) -------------------------------
CORNER_MARGIN = 0.20      # a subject this far from an edge still counts as anchored to it
MAX_SUBJECT_HEIGHT = 0.35  # a face taller than this is the shot, not an inset
MIN_OFFSET = 0.18         # horizontal offset of the subject from the centre
MAX_CENTRE_SPREAD = 0.05  # real insets move a few px across samples, presenters hundreds
MIN_INSET_HEIGHT = 0.10
# OpenShorts caps at 0.38; PBInvesting's window is ~0.20, a larger one is a cam shot.
MAX_INSET_HEIGHT = 0.45

# --- Ours -------------------------------------------------------------------
SAMPLE_W = 1920            # detection resolution (corner quadrants are 960x540)
SAMPLES_PER_SHOT = 4
MAX_SAMPLES = 16
MIN_SHOT_S = 1.0           # shorter shots inherit the previous shot's layout
EDGE_DIFF = 24.0           # per-channel step across an edge (0-255)
EDGE_SCORE = 0.5           # share of the edge's length that must show the step
AGREE = 0.6                # share of a shot's samples that must agree on the rectangle
BOX_TOLERANCE = 0.02       # agreement tolerance, fraction of the frame width
MIN_ASPECT, MAX_ASPECT = 1.0, 2.4
FACE_IN_BOX = (0.08, 0.6)  # face width / window width
INNER_MARGIN = 0.03        # trimmed off the window on each side (border glow, rounded corners)

# Output geometry (9:16, 1080x1920). The screen sits in the upper half, the
# webcam right below the middle, so the caption band (70-78 % of the height)
# lands on the speaker's torso, not on the face or the chart.
SCREEN_TOP = 0.125
GAP = 20
MAX_CAM_FRAC = 0.5


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    def scaled(self, k: float) -> "Box":
        return Box(self.x * k, self.y * k, self.w * k, self.h * k)

    def as_list(self) -> list[int]:
        return [int(round(v)) for v in (self.x, self.y, self.w, self.h)]


def is_cornered(face: Box, frame_w: int, frame_h: int, margin: float = CORNER_MARGIN) -> bool:
    """Small, off-centre and near an edge: looks like a webcam inset (OpenShorts)."""
    if face.h > frame_h * MAX_SUBJECT_HEIGHT:
        return False
    off_centre = abs(face.cx / frame_w - 0.5) >= MIN_OFFSET
    near_edge = (face.x <= frame_w * margin or face.x + face.w >= frame_w * (1 - margin)
                 or face.y <= frame_h * margin or face.y + face.h >= frame_h * (1 - margin))
    return off_centre and near_edge


def _edge_scores(img, lines: Iterable[int], lo: int, hi: int, *, vertical: bool) -> list[tuple[int, float]]:
    """Share of the segment [lo, hi) along each line where colour steps across it."""
    import numpy as np

    H, W = img.shape[:2]
    out = []
    for p in lines:
        if vertical:
            if p < 1 or p >= W - 1:
                continue
            a, b = img[max(0, lo):min(H, hi), p - 1], img[max(0, lo):min(H, hi), p + 1]
        else:
            if p < 1 or p >= H - 1:
                continue
            a, b = img[p - 1, max(0, lo):min(W, hi)], img[p + 1, max(0, lo):min(W, hi)]
        if not len(a):
            continue
        step = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=-1)
        out.append((p, float((step > EDGE_DIFF).mean())))
    return out


def _best(scores: list[tuple[int, float]], *, prefer_far: bool) -> int | None:
    good = [(p, s) for p, s in scores if s >= EDGE_SCORE]
    if not good:
        return None
    top = max(s for _p, s in good)
    # Among near-best candidates take the outermost: the window border is the
    # outermost long straight edge around the face; inner ones are furniture.
    near = [p for p, s in good if s >= top - 0.1]
    return min(near) if prefer_far else max(near)


def window_around(img, face: Box) -> Box | None:
    """The webcam window around ``face`` from the straight edges of its border.

    Left/right edges are searched on the face rows first, then top/bottom on the
    columns between them, then left/right again on the window's own rows. A side
    without an edge must be the frame edge the window is pinned to.
    """
    H, W = img.shape[:2]
    fx0, fx1, fy0, fy1 = face.x, face.x + face.w, face.y, face.y + face.h
    reach_x, reach_y = 7 * face.w, 5 * face.h

    def lr(y0: float, y1: float) -> tuple[float | None, float | None]:
        left = _best(_edge_scores(img, range(int(max(1, fx0 - reach_x)), int(fx0 - 0.3 * face.w)), int(y0), int(y1),
                                  vertical=True), prefer_far=True)
        right = _best(_edge_scores(img, range(int(fx1 + 0.3 * face.w), int(min(W - 1, fx1 + reach_x))), int(y0),
                                   int(y1), vertical=True), prefer_far=False)
        return left, right

    left, right = lr(fy0, fy1)
    x0 = left if left is not None else (0 if fx0 < W * CORNER_MARGIN else None)
    x1 = right if right is not None else (W if fx1 > W * (1 - CORNER_MARGIN) else None)
    if x0 is None or x1 is None:
        return None
    pad = 0.1 * (x1 - x0)
    top = _best(_edge_scores(img, range(int(max(1, fy0 - reach_y)), int(fy0 - 0.2 * face.h)), int(x0 + pad),
                             int(x1 - pad), vertical=False), prefer_far=True)
    bottom = _best(_edge_scores(img, range(int(fy1 + 0.3 * face.h), int(min(H - 1, fy1 + reach_y))), int(x0 + pad),
                                int(x1 - pad), vertical=False), prefer_far=False)
    y0 = top if top is not None else (0 if fy0 < H * CORNER_MARGIN else None)
    y1 = bottom if bottom is not None else (H if fy1 > H * (1 - CORNER_MARGIN) else None)
    if y0 is None or y1 is None:
        return None
    vpad = 0.1 * (y1 - y0)
    left2, right2 = lr(y0 + vpad, y1 - vpad)
    if left2 is not None:
        x0 = left2
    if right2 is not None:
        x1 = right2
    box = Box(float(x0), float(y0), float(x1 - x0), float(y1 - y0))
    return box if plausible(box, face, W, H) else None


def plausible(box: Box, face: Box, frame_w: int, frame_h: int) -> bool:
    if box.w <= 0 or box.h <= 0:
        return False
    aspect = box.w / box.h
    if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
        return False
    if not (MIN_INSET_HEIGHT * frame_h <= box.h <= MAX_INSET_HEIGHT * frame_h):
        return False
    if not (box.x <= face.cx <= box.x + box.w and box.y <= face.cy <= box.y + box.h):
        return False
    share = face.w / box.w
    if not (FACE_IN_BOX[0] <= share <= FACE_IN_BOX[1]):
        return False
    # The window itself must sit off-centre, against a side (OpenShorts gates on the box).
    return is_cornered(box, frame_w, frame_h) or is_cornered(face, frame_w, frame_h)


def stable_box(boxes: list[Box | None], frame_w: int) -> Box | None:
    """Median of the samples that agree with it, if enough of them do (OpenShorts' gate, tighter)."""
    found = [b for b in boxes if b is not None]
    if not boxes or not found:
        return None

    def med(vals: list[float]) -> float:
        vals = sorted(vals)
        n = len(vals)
        return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2

    m = Box(med([b.x for b in found]), med([b.y for b in found]), med([b.w for b in found]), med([b.h for b in found]))
    tol = BOX_TOLERANCE * frame_w
    close = [b for b in found if abs(b.x - m.x) <= tol and abs(b.y - m.y) <= tol
             and abs(b.x + b.w - m.x - m.w) <= tol and abs(b.y + b.h - m.y - m.h) <= tol]
    need = max(2 if len(boxes) >= 2 else 1, AGREE * len(boxes))
    if len(close) < need:
        return None
    return Box(med([b.x for b in close]), med([b.y for b in close]), med([b.w for b in close]), med([b.h for b in close]))


def corner_faces(detect: Callable[[Any], list[Box]], img) -> list[Box]:
    """Faces found in the four corner quadrants (BlazeFace misses webcam-sized faces on the full frame)."""
    import numpy as np

    H, W = img.shape[:2]
    out: list[Box] = []
    for x0, y0 in ((0, 0), (W // 2, 0), (0, H // 2), (W // 2, H // 2)):
        crop = np.ascontiguousarray(img[y0:y0 + H // 2, x0:x0 + W // 2])
        for f in detect(crop):
            box = Box(f.x - f.w / 2 + x0, f.y - f.h / 2 + y0, f.w, f.h)  # Det x/y are centres
            if is_cornered(box, W, H):
                out.append(box)
    return out


def sample_times(shots: list[tuple[float, float]]) -> list[list[float]]:
    per = max(1, min(SAMPLES_PER_SHOT, MAX_SAMPLES // max(1, len(shots))))
    out = []
    for a, b in shots:
        span = b - a
        if span < MIN_SHOT_S:
            out.append([])
            continue
        out.append([round(a + span * (i + 0.5) / per, 3) for i in range(per)])
    return out


def grab_frame(ffmpeg_bin: str, path: Path, t: float, src_w: int, src_h: int):
    """One RGB frame at absolute time ``t``, scaled to SAMPLE_W wide (never up)."""
    import numpy as np

    w = min(SAMPLE_W, src_w)
    w -= w % 2
    h = int(round(src_h * w / src_w))
    h -= h % 2
    proc = subprocess.run(
        [ffmpeg_bin, "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
         "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, timeout=60,
    )
    if len(proc.stdout) < w * h * 3:
        return None
    return np.frombuffer(proc.stdout[:w * h * 3], np.uint8).reshape(h, w, 3)


def classify_shots(
    shots: list[tuple[float, float]],
    frames: list[list[Any]],
    detect: Callable[[Any], list[Box]],
    *,
    scale: float,
    frame_w: int,
) -> list[dict[str, Any]]:
    """Per shot: {start, end, inset: Box|None (source px), samples, found}."""
    out: list[dict[str, Any]] = []
    for (a, b), imgs in zip(shots, frames):
        boxes: list[Box | None] = []
        for img in imgs:
            if img is None:
                boxes.append(None)
                continue
            faces = corner_faces(detect, img)
            found = None
            for face in sorted(faces, key=lambda f: -f.w):
                found = window_around(img, face)
                if found:
                    break
            boxes.append(found)
        sample_w = imgs[0].shape[1] if imgs and imgs[0] is not None else frame_w
        box = stable_box(boxes, sample_w) if imgs else None
        out.append({
            "start": round(a, 3), "end": round(b, 3),
            "inset": box.scaled(scale) if box else None,
            "samples": len(imgs), "found": sum(1 for x in boxes if x is not None),
            "short": not imgs,
        })
    # Short shots (no samples) follow the previous shot; leading ones the next.
    for i, shot in enumerate(out):
        if shot["short"]:
            ref = next((out[j] for j in range(i - 1, -1, -1) if not out[j]["short"]), None) or \
                  next((s for s in out[i + 1:] if not s["short"]), None)
            shot["inset"] = ref["inset"] if ref else None
    return out


def clip_window(shots: list[dict[str, Any]], frame_w: int) -> Box | None:
    """One webcam rectangle for the whole clip: the median of the inset shots that agree."""
    boxes = [s["inset"] for s in shots if s["inset"] is not None]
    if not boxes:
        return None
    m = stable_box(boxes, frame_w)
    return m or boxes[0]


def _even(v: float) -> int:
    n = int(round(v))
    return n - (n % 2)


def geometry(src_w: int, src_h: int, cam: Box, out_w: int = 1080, out_h: int = 1920) -> dict[str, Any]:
    """Crop/scale/position numbers for the stacked layout."""
    mx, my = cam.w * INNER_MARGIN, cam.h * INNER_MARGIN
    cx = max(0, _even(cam.x + mx))
    cy = max(0, _even(cam.y + my))
    cw = min(src_w - cx, _even(cam.w - 2 * mx))
    ch = min(src_h - cy, _even(cam.h - 2 * my))
    screen_h = _even(out_w * src_h / src_w)
    cam_h = min(_even(out_w * ch / cw), _even(out_h * MAX_CAM_FRAC))
    screen_y = _even(out_h * SCREEN_TOP)
    cam_y = screen_y + screen_h + GAP
    if cam_y + cam_h > out_h:
        cam_y = out_h - cam_h
        screen_y = max(0, cam_y - GAP - screen_h)
    # Keep the cam crop's aspect equal to its band (no stretch): trim height or width.
    want = out_w / cam_h
    if cw / ch > want:
        new_w = _even(ch * want)
        cx += (cw - new_w) // 2 - ((cw - new_w) // 2) % 2
        cw = new_w
    else:
        new_h = _even(cw / want)
        cy += (ch - new_h) // 2 - ((ch - new_h) // 2) % 2
        ch = new_h
    return {"cam_crop": [cx, cy, cw, ch], "screen": [0, screen_y, out_w, screen_h],
            "cam": [0, cam_y, out_w, cam_h], "out": [out_w, out_h]}


def blurred_backdrop(out_w: int, out_h: int, sigma: float) -> str:
    """Quarter-size blur then upscale (OpenShorts ffmpeg_utils.blurred_backdrop, MIT)."""
    small_w = max(2, out_w // 4 - (out_w // 4) % 2)
    small_h = max(2, out_h // 4 - (out_h // 4) % 2)
    return (f"scale=-2:{small_h},crop=w=min(iw\\,{small_w}):h={small_h},"
            f"gblur=sigma={sigma / 4:g},scale={out_w}:{out_h}")


def inset_chain(src_label: str, geo: dict[str, Any], out_label: str) -> str:
    out_w, out_h = geo["out"]
    cx, cy, cw, ch = geo["cam_crop"]
    _x, sy, sw, sh = geo["screen"]
    _x, cam_y, camw, camh = geo["cam"]
    return (
        f"{src_label}split=3[ibg][iscr][icam];"
        f"[ibg]{blurred_backdrop(out_w, out_h, 24)}[ibgb];"
        f"[iscr]scale={sw}:{sh}:flags=lanczos[iscreen];"
        f"[icam]crop={cw}:{ch}:{cx}:{cy},scale={camw}:{camh}:flags=lanczos[icamv];"
        f"[ibgb][iscreen]overlay=0:{sy}[ibs];"
        f"[ibs][icamv]overlay=0:{cam_y},setsar=1{out_label}"
    )


def center_chain(src_label: str, src_w: int, src_h: int, out_label: str, out_w: int = 1080, out_h: int = 1920) -> str:
    """The historical 9:16 centre crop, as a chain (fixed, never moves)."""
    cw = min(src_w, _even(src_h * out_w / out_h))
    x = _even((src_w - cw) / 2)
    return f"{src_label}crop={cw}:{src_h}:{x}:0,scale={out_w}:{out_h},setsar=1{out_label}"


def build_graph(spans: list[dict[str, Any]], geo: dict[str, Any], src_w: int, src_h: int) -> str:
    """filter_complex ending in [vmain]: inset where spans say so, centre crop elsewhere."""
    inset = [(s["start"], s["end"]) for s in spans if s["layout"] == "inset"]
    if len(inset) == len(spans):
        return inset_chain("[0:v]", geo, "[vmain]")
    enable = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in inset) or "0"
    return (
        "[0:v]split=2[lc][li];"
        + center_chain("[lc]", src_w, src_h, "[lcenter]") + ";"
        + inset_chain("[li]", geo, "[linset]") + ";"
        + f"[lcenter][linset]overlay=enable='{enable}'[vmain]"
    )


def spans_from(shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for s in shots:
        layout = "inset" if s["inset"] is not None else "center"
        if spans and spans[-1]["layout"] == layout:
            spans[-1]["end"] = s["end"]
        else:
            spans.append({"start": s["start"], "end": s["end"], "layout": layout})
    return spans
