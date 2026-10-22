"""Screen recording with a webcam inset -> 9:16 with the screen on top and the webcam below.

A screen recording (chart, slides, IDE...) with the speaker's webcam
composited into a corner. The 9:16 centre crop slices the screen and face
mode zooms on the corner, so neither works. This layout puts the useful part
of the screen at full width on top and the webcam window, cut out exactly,
enlarged underneath, over a blurred copy of the source. Static per shot: no
tracking (the inset does not move).

v2: the screen band is cropped to the content area (a window drawn inside a
decorative frame, letterbox bars and plain margins are dropped; found from
long straight border lines and flat rows/columns, never from a colour), the
webcam thumbnail is cut away or masked from that band so the face is not shown
twice, the webcam band is sized to fill the height left (upscale capped, light
sharpen when enlarged) and fixed text zones are returned for the captions (under
the webcam) and the on-screen texts (above the screen) so text never covers
the face or the chart. A quick probe aborts the detection early when no webcam
window is there.

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

# --- v2: content band, thumbnail mask, text zones ---------------------------
LINE_STEP = 30             # colour step across a border line (0-255, 2 px each side)
LINE_FRAC = 0.4            # a border line spans at least this share of the frame side
LINE_CLUSTER = 4           # border lines closer than this (px) are one border (thickness, glow)
EDGE_SEARCH = 0.45         # borders are searched in the outer 45 % of each side
DETAIL_STEP = 20           # pixel step that counts as detail (text, candles, UI)
OUTSIDE_DETAIL = 0.5       # outside a border: less than this x the detail inside (plain / decorative)
UNIFORM_STD = 6.0          # a row/column this flat in every sample is a bar or plain margin
CONTENT_PAD = 0.004        # extra trim inside a border (glow, rounded corners), fraction of the side
MIN_CONTENT = (0.4, 0.3)   # smaller content boxes (w, h fractions) are distrusted: whole frame
FULL_CONTENT = 0.97        # a box this close to the frame is the frame
SHRINK_MAX = 0.10          # cut the band to drop the thumbnail when that loses at most this share
THUMB_PAD = 0.01           # margin around the thumbnail when masking/cutting, fraction of frame width
PROBES = 4                 # quick probe frames before the full per-shot pass

# Output geometry (9:16, 1080x1920), top to bottom: on-screen text zone,
# screen band (full width), webcam band, caption zone. Spare height is split
# between the two text zones.
TOP_TEXT = 150             # minimum zone above the screen for on-screen texts
CAPTION_ZONE = 210         # minimum zone under the webcam for captions (2 lines)
CAPTION_PAD = 24
OST_LINE = 90              # on-screen text box height (Overlay style) used to centre it
GAP = 0
MAX_SCREEN_FRAC = 0.45
CAM_UPSCALE_MAX = 2.0
CAM_MIN_WIDTH = 0.6        # the webcam crop keeps at least this share of the window width
SHARPEN_ABOVE = 1.15       # light unsharp when the webcam is enlarged more than this
SHARPEN = "unsharp=5:5:0.6:5:5:0.0"


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


def find_window(detect: Callable[[Any], list[Box]], img) -> tuple[Box | None, Box | None]:
    """(webcam window, its face) in one frame, or (None, None)."""
    for face in sorted(corner_faces(detect, img), key=lambda f: -f.w):
        found = window_around(img, face)
        if found:
            return found, face
    return None, None


def _median_box(boxes: list[Box]) -> Box:
    def med(vals: list[float]) -> float:
        vals = sorted(vals)
        return vals[len(vals) // 2]

    return Box(med([b.x for b in boxes]), med([b.y for b in boxes]), med([b.w for b in boxes]), med([b.h for b in boxes]))


def probe_times(duration: float, n: int = PROBES) -> list[float]:
    return [round(duration * (i + 0.5) / n, 3) for i in range(n)]


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
        faces_seen: list[Box] = []
        for img in imgs:
            if img is None:
                boxes.append(None)
                continue
            found, face = find_window(detect, img)
            boxes.append(found)
            if face is not None:
                faces_seen.append(face)
        sample_w = imgs[0].shape[1] if imgs and imgs[0] is not None else frame_w
        box = stable_box(boxes, sample_w) if imgs else None
        out.append({
            "start": round(a, 3), "end": round(b, 3),
            "inset": box.scaled(scale) if box else None,
            "samples": len(imgs), "found": sum(1 for x in boxes if x is not None),
            "short": not imgs,
            "face": _median_box(faces_seen).scaled(scale) if box and faces_seen else None,
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


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def geometry(
    src_w: int,
    src_h: int,
    cam: Box,
    out_w: int = 1080,
    out_h: int = 1920,
    *,
    band: Box | None = None,
    face: Box | None = None,
    mask: Box | None = None,
    fill: str = "000000",
) -> dict[str, Any]:
    """Crop/scale/position numbers for the stacked layout (all source / output px, even).

    ``band`` is the screen area kept on top (default: the whole frame),
    ``mask`` a rectangle of it painted ``fill`` (the webcam thumbnail),
    ``face`` centres the webcam crop when it is narrower than the window.
    """
    mx, my = cam.w * INNER_MARGIN, cam.h * INNER_MARGIN
    ix, iy = max(0, _even(cam.x + mx)), max(0, _even(cam.y + my))
    iw, ih = min(src_w - ix, _even(cam.w - 2 * mx)), min(src_h - iy, _even(cam.h - 2 * my))

    b = band or Box(0, 0, src_w, src_h)
    bx, by = max(0, _even(b.x)), max(0, _even(b.y))
    bw, bh = min(src_w - bx, _even(b.w)), min(src_h - by, _even(b.h))
    sw, sh = out_w, _even(out_w * bh / bw)
    if sh > out_h * MAX_SCREEN_FRAC:
        sh = _even(out_h * MAX_SCREEN_FRAC)
        sw = _even(sh * bw / bh)

    avail = max(2, out_h - TOP_TEXT - sh - GAP - CAPTION_ZONE)
    if out_w / iw > CAM_UPSCALE_MAX:            # small window: do not blow it up to the full width
        camw = _even(iw * CAM_UPSCALE_MAX)
        camh = min(avail, _even(ih * CAM_UPSCALE_MAX))
    else:
        camw = out_w
        camh = min(avail, _even(ih * CAM_UPSCALE_MAX), _even(out_w * ih / (iw * CAM_MIN_WIDTH)))
    camh = max(2, camh)
    want = camw / camh
    if iw / ih > want:                          # window wider than the band: trim the sides
        ch, cw = ih, min(iw, _even(ih * want))
    else:                                       # taller: trim top/bottom
        cw, ch = iw, min(ih, _even(iw / want))
    fcx = face.cx if face else ix + iw / 2
    fcy = face.cy if face else iy + ih / 2
    cx = _even(_clamp(fcx - cw / 2, ix, ix + iw - cw))
    cy = _even(_clamp(fcy - ch * 0.45, iy, iy + ih - ch))
    upscale = camh / ch

    free = max(0, out_h - TOP_TEXT - sh - GAP - camh - CAPTION_ZONE)
    sy = _even(TOP_TEXT + free / 2)
    cam_y = sy + sh + GAP
    cap_top = cam_y + camh
    geo: dict[str, Any] = {
        "screen_crop": [bx, by, bw, bh],
        "screen": [(out_w - sw) // 2 - ((out_w - sw) // 2) % 2, sy, sw, sh],
        "cam_crop": [cx, cy, cw, ch],
        "cam": [(out_w - camw) // 2 - ((out_w - camw) // 2) % 2, cam_y, camw, camh],
        "out": [out_w, out_h],
        "upscale": round(upscale, 3),
        "sharpen": upscale > SHARPEN_ABOVE,
        "mask": None,
        "fill": fill,
    }
    if mask is not None:
        x0, y0 = max(bx, int(mask.x)), max(by, int(mask.y))
        x1, y1 = min(bx + bw, int(round(mask.x + mask.w))), min(by + bh, int(round(mask.y + mask.h)))
        if x1 > x0 and y1 > y0:
            geo["mask"] = [x0 - bx, y0 - by, x1 - x0, y1 - y0]
    geo["text_zones"] = {
        "top": [0, sy],
        "caption": [cap_top, out_h],
        "overlay_margin_v": max(12, (sy - OST_LINE) // 2),
        "caption_margin_v": cap_top + CAPTION_PAD,
    }
    geo["metrics"] = {
        "chart_width_px": sw,
        "chart_height_px": sh,
        "content_px_on_screen_band": round(sw * sh),
        "empty_bg_px_rows": out_h - sh - GAP - camh,
        "cam_upscale": round(upscale, 3),
    }
    return geo


def blurred_backdrop(out_w: int, out_h: int, sigma: float) -> str:
    """Quarter-size blur then upscale (OpenShorts ffmpeg_utils.blurred_backdrop, MIT)."""
    small_w = max(2, out_w // 4 - (out_w // 4) % 2)
    small_h = max(2, out_h // 4 - (out_h // 4) % 2)
    return (f"scale=-2:{small_h},crop=w=min(iw\\,{small_w}):h={small_h},"
            f"gblur=sigma={sigma / 4:g},scale={out_w}:{out_h}")


def inset_chain(src_label: str, geo: dict[str, Any], out_label: str) -> str:
    out_w, out_h = geo["out"]
    cx, cy, cw, ch = geo["cam_crop"]
    sx, sy, sw, sh = geo["screen"]
    camx, cam_y, camw, camh = geo["cam"]
    bx, by, bw, bh = geo.get("screen_crop") or [0, 0, 0, 0]
    screen = f"crop={bw}:{bh}:{bx}:{by}," if bw and bh else ""
    if geo.get("mask"):
        mx, my, mw, mh = geo["mask"]
        screen += f"drawbox=x={mx}:y={my}:w={mw}:h={mh}:color=0x{geo.get('fill') or '000000'}:t=fill,"
    sharpen = f",{SHARPEN}" if geo.get("sharpen") else ""
    return (
        f"{src_label}split=3[ibg][iscr][icam];"
        f"[ibg]{blurred_backdrop(out_w, out_h, 24)}[ibgb];"
        f"[iscr]{screen}scale={sw}:{sh}:flags=lanczos[iscreen];"
        f"[icam]crop={cw}:{ch}:{cx}:{cy},scale={camw}:{camh}:flags=lanczos{sharpen}[icamv];"
        f"[ibgb][iscreen]overlay={sx}:{sy}[ibs];"
        f"[ibs][icamv]overlay={camx}:{cam_y},setsar=1{out_label}"
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


# --- v2: content band -------------------------------------------------------

def _longest_runs(mask) -> Any:
    """Longest run of True per row of a 2D bool array."""
    import numpy as np

    h, w = mask.shape
    padded = np.zeros((h, w + 2), np.int8)
    padded[:, 1:-1] = mask
    d = np.diff(padded, axis=1)
    rs, cs = np.nonzero(d == 1)
    _re, ce = np.nonzero(d == -1)   # row-major order: starts and ends pair up
    out = np.zeros(h, np.int64)
    if len(rs):
        np.maximum.at(out, rs, ce - cs)
    return out


def _blank(mask, box: Box | None, pad: int) -> None:
    if box is None:
        return
    H, W = mask.shape[:2]
    x0, y0 = max(0, int(box.x) - pad), max(0, int(box.y) - pad)
    x1, y1 = min(W, int(box.x + box.w) + pad), min(H, int(box.y + box.h) + pad)
    mask[y0:y1, x0:x1] = False


def line_profiles(img, exclude: Box | None = None) -> tuple[Any, Any]:
    """Longest straight edge per row (horizontal lines, share of W) and per column (share of H)."""
    import numpy as np

    a = img.astype(np.int16)
    H, W = a.shape[:2]
    k = 2
    hy = np.zeros((H, W), bool)
    hy[k:H - k] = np.abs(a[2 * k:] - a[:-2 * k]).max(axis=-1) > LINE_STEP
    vx = np.zeros((H, W), bool)
    vx[:, k:W - k] = np.abs(a[:, 2 * k:] - a[:, :-2 * k]).max(axis=-1) > LINE_STEP
    _blank(hy, exclude, k + 2)
    _blank(vx, exclude, k + 2)
    return _longest_runs(hy) / W, _longest_runs(vx.T) / H


def _detail(img) -> Any:
    import numpy as np

    g = img.astype(np.int16).mean(axis=-1)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:] = np.abs(np.diff(g, axis=1))
    gy[1:] = np.abs(np.diff(g, axis=0))
    return np.maximum(gx, gy) > DETAIL_STEP


def _border(frac, lo: int, hi: int, *, from_start: bool) -> tuple[int, int] | None:
    """(outer, inner) of the outermost border-line cluster in [lo, hi); inner is the first content line."""
    idx = [i for i in range(lo, hi) if frac[i] >= LINE_FRAC]
    if not idx:
        return None
    if not from_start:
        idx = idx[::-1]
    first = last = idx[0]
    for n in idx[1:]:
        if abs(n - last) <= LINE_CLUSTER:
            last = n
        else:
            break
    return (first, last + 1) if from_start else (first + 1, last)


def _density(det, valid, y0: int, y1: int, x0: int, x1: int) -> float | None:
    v = valid[y0:y1, x0:x1]
    n = int(v.sum())
    if n == 0:
        return None
    return float(det[y0:y1, x0:x1][v].mean())


def content_box(frames: list[Any], exclude: Box | None = None) -> Box | None:
    """Useful screen area in sample px, or None for the whole frame.

    A side is cut at the outermost long straight border line (persistent across
    samples) when the strip outside it is plain or decorative (much less detail
    than inside). Sides without such a border lose only flat rows/columns
    (letterbox, plain margins). ``exclude`` (the webcam window) is ignored.
    """
    import numpy as np

    imgs = [f for f in frames if f is not None]
    if not imgs:
        return None
    H, W = imgs[0].shape[:2]
    imgs = [f for f in imgs if f.shape[:2] == (H, W)]
    profs = [line_profiles(f, exclude) for f in imgs]
    rows = np.median(np.stack([p[0] for p in profs]), axis=0)
    cols = np.median(np.stack([p[1] for p in profs]), axis=0)
    det = np.mean(np.stack([_detail(f) for f in imgs]), axis=0)
    valid = np.ones((H, W), bool)
    _blank(valid, exclude, 4)

    def plain_outside(o0: int, o1: int, i0: int, i1: int, *, vertical: bool) -> bool:
        if vertical:
            out = _density(det, valid, 0, H, o0, o1) if o1 > o0 else 0.0
            inside = _density(det, valid, 0, H, i0, i1)
        else:
            out = _density(det, valid, o0, o1, 0, W) if o1 > o0 else 0.0
            inside = _density(det, valid, i0, i1, 0, W)
        if inside is None or inside <= 0:
            return False
        return (out or 0.0) < OUTSIDE_DETAIL * inside

    y0, y1, x0, x1 = 0, H, 0, W
    found = {"top": False, "bottom": False, "left": False, "right": False}
    top = _border(rows, 0, int(H * EDGE_SEARCH), from_start=True)
    if top and plain_outside(0, top[0], top[1], H // 2, vertical=False):
        y0, found["top"] = top[1], True
    bottom = _border(rows, int(H * (1 - EDGE_SEARCH)), H, from_start=False)
    if bottom and plain_outside(bottom[0], H, H // 2, bottom[1], vertical=False):
        y1, found["bottom"] = bottom[1], True
    left = _border(cols, 0, int(W * EDGE_SEARCH), from_start=True)
    if left and plain_outside(0, left[0], left[1], W // 2, vertical=True):
        x0, found["left"] = left[1], True
    right = _border(cols, int(W * (1 - EDGE_SEARCH)), W, from_start=False)
    if right and plain_outside(right[0], W, W // 2, right[1], vertical=True):
        x1, found["right"] = right[1], True

    # Flat rows/columns (bars, plain margins) on the sides without a border.
    grey = [f.astype(np.float32).mean(axis=-1) for f in imgs]
    row_std = np.max(np.stack([g.std(axis=1) for g in grey]), axis=0)
    col_std = np.max(np.stack([g.std(axis=0) for g in grey]), axis=0)
    if not found["top"]:
        while y0 < H // 2 and row_std[y0] < UNIFORM_STD:
            y0 += 1
    if not found["bottom"]:
        while y1 > H // 2 and row_std[y1 - 1] < UNIFORM_STD:
            y1 -= 1
    if not found["left"]:
        while x0 < W // 2 and col_std[x0] < UNIFORM_STD:
            x0 += 1
    if not found["right"]:
        while x1 > W // 2 and col_std[x1 - 1] < UNIFORM_STD:
            x1 -= 1

    px, py = W * CONTENT_PAD, H * CONTENT_PAD
    bx0 = x0 + (px if found["left"] else 0)
    bx1 = x1 - (px if found["right"] else 0)
    by0 = y0 + (py if found["top"] else 0)
    by1 = y1 - (py if found["bottom"] else 0)
    w, h = bx1 - bx0, by1 - by0
    if w < MIN_CONTENT[0] * W or h < MIN_CONTENT[1] * H:
        return None
    if w >= FULL_CONTENT * W and h >= FULL_CONTENT * H:
        return None
    return Box(float(bx0), float(by0), float(w), float(h))


def fill_colour(frames: list[Any], band: Box, exclude: Box | None) -> str:
    """Median colour of the band (webcam excluded) as RRGGBB: the thumbnail mask blends in."""
    import numpy as np

    img = next((f for f in frames if f is not None), None)
    if img is None:
        return "000000"
    x0, y0 = int(band.x), int(band.y)
    x1, y1 = int(band.x + band.w), int(band.y + band.h)
    valid = np.ones(img.shape[:2], bool)
    _blank(valid, exclude, 4)
    px = img[y0:y1, x0:x1][valid[y0:y1, x0:x1]]
    if not len(px):
        return "000000"
    r, g, b = (int(v) for v in np.median(px.reshape(-1, 3), axis=0))
    return f"{r:02x}{g:02x}{b:02x}"


def band_and_mask(band: Box, cam: Box, frame_w: int) -> tuple[Box, Box | None, str]:
    """Drop the webcam thumbnail from the screen band: cut a side when cheap, else mask it.

    Returns (band, mask or None, how) with how in none / cut_<side> / mask.
    """
    pad = THUMB_PAD * frame_w
    tx0, ty0 = cam.x - pad, cam.y - pad
    tx1, ty1 = cam.x + cam.w + pad, cam.y + cam.h + pad
    bx0, by0, bx1, by1 = band.x, band.y, band.x + band.w, band.y + band.h
    ix0, iy0, ix1, iy1 = max(bx0, tx0), max(by0, ty0), min(bx1, tx1), min(by1, ty1)
    if ix1 <= ix0 or iy1 <= iy0:
        return band, None, "none"
    options = []
    if ix1 >= bx1 - 1:
        options.append(((bx1 - ix0) / band.w, "cut_right", Box(bx0, by0, ix0 - bx0, band.h)))
    if ix0 <= bx0 + 1:
        options.append(((ix1 - bx0) / band.w, "cut_left", Box(ix1, by0, bx1 - ix1, band.h)))
    if iy1 >= by1 - 1:
        options.append(((by1 - iy0) / band.h, "cut_bottom", Box(bx0, by0, band.w, iy0 - by0)))
    if iy0 <= by0 + 1:
        options.append(((iy1 - by0) / band.h, "cut_top", Box(bx0, iy1, band.w, by1 - iy1)))
    if options:
        loss, how, cut = min(options, key=lambda o: o[0])
        if loss <= SHRINK_MAX:
            return cut, None, how
    return band, Box(ix0, iy0, ix1 - ix0, iy1 - iy0), "mask"
