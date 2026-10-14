"""B-roll, music ducking, xfade and CC0 SFX for a render that opted in.

Same keys as the VPS ``production_blocks`` payload. Every block defaults off,
and ``audio.original_only`` refuses music even if the payload asks for it.
The untouched ``FFmpegTool.render_clip`` path stays in place for that case.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from app.utils.subprocess import run_command

CC0 = {
    "license": "CC0-1.0",
    "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
    "source": "bundled-cc0",
    "author": "clipping-windows-worker",
}
LIBRARY: dict[str, dict[str, str]] = {
    "bed_soft": {**CC0, "id": "bed_soft", "title": "Soft bed", "kind": "music"},
    "whoosh": {**CC0, "id": "whoosh", "title": "Whoosh", "kind": "sfx"},
    "hit": {**CC0, "id": "hit", "title": "Hit", "kind": "sfx"},
    "pop": {**CC0, "id": "pop", "title": "Pop", "kind": "sfx"},
}
XFADE_EFFECTS = {
    "fade", "wipeleft", "wiperight", "slideleft", "slideright",
    "circleopen", "dissolve", "smoothleft",
}
PEXELS_LICENSE = "Pexels License"
PEXELS_LICENSE_URL = "https://www.pexels.com/license/"
PIXABAY_LICENSE = "Pixabay Content License"
PIXABAY_LICENSE_URL = "https://pixabay.com/service/license-summary/"

# sidechaincompress: first input is ducked, second is the speech key.
DUCK = "threshold=0.05:ratio=8:attack=20:release=250"


def _block(payload: dict[str, Any] | None, name: str) -> dict[str, Any]:
    raw = (payload or {}).get(name)
    return dict(raw) if isinstance(raw, dict) else {}


def _enabled(block: dict[str, Any]) -> bool:
    return block.get("enabled") is True or str(block.get("enabled")).strip().lower() in {"1", "true", "yes"}


def features_requested(payload: dict[str, Any] | None) -> bool:
    return any(_enabled(_block(payload, name)) for name in ("broll", "music", "transitions", "sfx"))


def frame_size(output_format: str) -> tuple[int, int]:
    if output_format == "16:9":
        return 1920, 1080
    if output_format == "9:16":
        return 1080, 1920
    raise ValueError(f"Unsupported output_format: {output_format}")


def base_video_filter(output_format: str, fps: float = 30.0) -> str:
    """Same crop as ``render_clip``, plus a shared timebase so xfade accepts it."""
    if output_format == "9:16":
        scale = "crop=ih*9/16:ih,scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2"
    elif output_format == "16:9":
        scale = "crop=iw:iw*9/16,scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2"
    else:
        raise ValueError(f"Unsupported output_format: {output_format}")
    return f"{scale},setsar=1,fps={fps:.3f},settb=AVTB,format=yuv420p"


def cover_filter(width: int, height: int, fps: float = 30.0) -> str:
    return (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},setsar=1,fps={fps:.3f},settb=AVTB,format=yuv420p"
    )


def pexels_license(video: dict[str, Any], query: str) -> dict[str, Any]:
    user = video.get("user") if isinstance(video.get("user"), dict) else {}
    return {
        "provider": "pexels",
        "id": str(video.get("id") or ""),
        "url": str(video.get("url") or ""),
        "author": str(user.get("name") or ""),
        "query": query,
        "license": PEXELS_LICENSE,
        "license_url": PEXELS_LICENSE_URL,
    }


def pixabay_license(hit: dict[str, Any], query: str) -> dict[str, Any]:
    return {
        "provider": "pixabay",
        "id": str(hit.get("id") or ""),
        "url": str(hit.get("pageURL") or ""),
        "author": str(hit.get("user") or ""),
        "query": query,
        "license": PIXABAY_LICENSE,
        "license_url": PIXABAY_LICENSE_URL,
    }


def choose_pexels_url(video: dict[str, Any]) -> str | None:
    files = [f for f in (video.get("video_files") or []) if isinstance(f, dict) and f.get("link")]
    mp4 = [f for f in files if "mp4" in str(f.get("file_type") or f.get("link"))] or files
    if not mp4:
        return None

    def _rank(item: dict[str, Any]) -> tuple[int, int]:
        try:
            height = int(item.get("height") or 0)
        except (TypeError, ValueError):
            height = 0
        # Prefer a file near 720p. SD is the fallback, 4K is the last resort.
        return (abs(height - 720), -height)

    return str(sorted(mp4, key=_rank)[0]["link"])


def choose_pixabay_url(hit: dict[str, Any]) -> str | None:
    videos = hit.get("videos") if isinstance(hit.get("videos"), dict) else {}
    for key in ("large", "medium", "small", "tiny"):
        url = (videos.get(key) or {}).get("url") if isinstance(videos.get(key), dict) else None
        if url:
            return str(url)
    return None


def search_stock(
    *,
    provider: str,
    query: str,
    pexels_key: str | None,
    pixabay_key: str | None,
    get: Callable[..., Any],
) -> dict[str, Any] | None:
    """One stock hit plus its license. ``get(url, headers, params)`` returns JSON."""
    order = [provider if provider in {"pexels", "pixabay"} else "pexels"]
    for other in ("pexels", "pixabay"):
        if other not in order:
            order.append(other)
    for name in order:
        if name == "pexels" and pexels_key:
            body = get(
                "https://api.pexels.com/videos/search",
                {"Authorization": pexels_key},
                {"query": query, "per_page": 5, "orientation": "portrait"},
            )
            for video in (body or {}).get("videos") or []:
                url = choose_pexels_url(video) if isinstance(video, dict) else None
                if url:
                    record = pexels_license(video, query)
                    record["download_url"] = url
                    if name != provider:
                        record["provider_fallback"] = name
                    return record
        if name == "pixabay" and pixabay_key:
            body = get(
                "https://pixabay.com/api/videos/",
                {},
                {"key": pixabay_key, "q": query, "per_page": 5},
            )
            for hit in (body or {}).get("hits") or []:
                url = choose_pixabay_url(hit) if isinstance(hit, dict) else None
                if url:
                    record = pixabay_license(hit, query)
                    record["download_url"] = url
                    if name != provider:
                        record["provider_fallback"] = name
                    return record
    return None


def _http_get(url: str, headers: dict[str, str], params: dict[str, Any]) -> Any:
    import httpx

    with httpx.Client(timeout=30, follow_redirects=True) as client:
        response = client.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()


def default_stock_search(**kwargs: Any) -> dict[str, Any] | None:
    return search_stock(get=_http_get, **kwargs)


def library_record(name: str) -> dict[str, str] | None:
    item = LIBRARY.get(name)
    return dict(item) if item else None


def ensure_cc0(dest: Path, binary: str = "ffmpeg") -> dict[str, Path]:
    """Write the bundled CC0 beds into ``dest`` if they are not there yet.

    The tones are generated here, so the license really is CC0 and the render
    does not download audio.
    """
    dest.mkdir(parents=True, exist_ok=True)
    paths = {
        "bed_soft": dest / "bed_soft.wav",
        "whoosh": dest / "whoosh.wav",
        "hit": dest / "hit.wav",
        "pop": dest / "pop.wav",
    }
    if all(path.is_file() and path.stat().st_size > 0 for path in paths.values()):
        return paths
    run_command(
        [binary, "-y",
         "-f", "lavfi", "-i", "sine=frequency=220:duration=8:sample_rate=48000",
         "-f", "lavfi", "-i", "sine=frequency=277:duration=8:sample_rate=48000",
         "-f", "lavfi", "-i", "sine=frequency=330:duration=8:sample_rate=48000",
         "-filter_complex",
         "[0:a][1:a][2:a]amix=inputs=3:duration=first:normalize=0,volume=0.25,aformat=sample_rates=48000:channel_layouts=stereo",
         str(paths["bed_soft"])],
        timeout=60,
    )
    run_command(
        [binary, "-y", "-f", "lavfi", "-i", "anoisesrc=color=pink:duration=0.45:sample_rate=48000",
         "-af", "afade=t=in:st=0:d=0.05,afade=t=out:st=0.12:d=0.33,highpass=f=700,aformat=sample_rates=48000:channel_layouts=stereo",
         str(paths["whoosh"])],
        timeout=30,
    )
    run_command(
        [binary, "-y", "-f", "lavfi", "-i", "sine=frequency=140:duration=0.18:sample_rate=48000",
         "-af", "afade=t=out:st=0.02:d=0.16,aformat=sample_rates=48000:channel_layouts=stereo",
         str(paths["hit"])],
        timeout=30,
    )
    run_command(
        [binary, "-y", "-f", "lavfi", "-i", "sine=frequency=880:duration=0.09:sample_rate=48000",
         "-af", "afade=t=out:st=0.01:d=0.08,aformat=sample_rates=48000:channel_layouts=stereo",
         str(paths["pop"])],
        timeout=30,
    )
    return paths


def _local_license(mark: dict[str, Any], path: Path) -> dict[str, Any]:
    raw = mark.get("license") if isinstance(mark.get("license"), dict) else {}
    record = {
        "provider": str(raw.get("provider") or "local"),
        "license": str(raw.get("license") or "unspecified"),
        "file": path.name,
    }
    for key in ("license_url", "author", "url", "id", "query", "source"):
        if raw.get(key):
            record[key] = raw[key]
    return record


def _download(url: str, dest: Path) -> None:
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            size = 0
            with dest.open("wb") as handle:
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > 80 * 1024 * 1024:
                        raise ValueError("stock download is larger than 80MB")
                    handle.write(chunk)


def resolve_broll(
    block: dict[str, Any],
    *,
    work_dir: Path,
    pexels_key: str | None,
    pixabay_key: str | None,
    stock_search: Callable[..., dict[str, Any] | None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Download or accept local files. The license travels with each file."""
    marks = [m for m in (block.get("marks") or []) if isinstance(m, dict)]
    if not marks:
        return [], {"applied": False, "reason": "no_marks", "assets": []}
    search = stock_search or default_stock_search
    assets: list[dict[str, Any]] = []
    resolved: list[dict[str, Any]] = []
    provider = str(block.get("provider") or "pexels")
    for index, mark in enumerate(marks):
        try:
            start = float(mark["start"])
            end = float(mark["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        path: Path | None = None
        license_record: dict[str, Any] | None = None
        local = str(mark.get("file") or "").strip()
        if local and Path(local).is_file():
            path = Path(local)
            license_record = _local_license(mark, path)
        else:
            query = " ".join(str(mark.get("query") or "").split())
            if not query:
                assets.append({"error": "no_query", "start": start})
                continue
            if not pexels_key and not pixabay_key and stock_search is None:
                assets.append({"error": "missing_api_key", "query": query})
                continue
            try:
                hit = search(
                    provider=provider,
                    query=query,
                    pexels_key=pexels_key,
                    pixabay_key=pixabay_key,
                )
            except Exception as exc:  # noqa: BLE001
                assets.append({"error": str(exc), "query": query})
                continue
            if not hit or not hit.get("download_url"):
                assets.append({"error": "no_hit", "query": query})
                continue
            dest = work_dir / f"broll_{index}.mp4"
            try:
                _download(str(hit["download_url"]), dest)
            except Exception as exc:  # noqa: BLE001
                assets.append({"error": str(exc), "query": query, **{k: hit[k] for k in hit if k != "download_url"}})
                continue
            path = dest
            license_record = {k: v for k, v in hit.items() if k != "download_url"}
        placement = str(mark.get("placement") or block.get("placement") or "overlay")
        if placement not in {"overlay", "cut"}:
            placement = "overlay"
        try:
            opacity = float(block.get("opacity") if mark.get("opacity") is None else mark.get("opacity"))
        except (TypeError, ValueError):
            opacity = 1.0
        opacity = min(1.0, max(0.05, opacity))
        item = {
            "path": str(path),
            "start": start,
            "end": end,
            "placement": placement,
            "opacity": opacity,
            "license": license_record,
        }
        resolved.append(item)
        assets.append(license_record or {})
    applied = {
        "applied": bool(resolved),
        "provider": provider,
        "placement": block.get("placement") or "overlay",
        "assets": assets,
    }
    if not resolved:
        applied["reason"] = "missing_api_key" if any(a.get("error") == "missing_api_key" for a in assets) else "unresolved"
    return resolved, applied


def resolve_music(
    block: dict[str, Any],
    *,
    library: dict[str, Path],
    original_only: bool,
    no_added_music: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if original_only or no_added_music:
        reason = "audio.original_only" if original_only else "audio.no_added_music"
        return None, {"applied": False, "reason": reason, "asset": None}
    track = str(block.get("track") or "bed_soft")
    record = library_record(track) or library_record("bed_soft")
    path = library.get(track) or library.get("bed_soft")
    if path is None or record is None or not path.is_file():
        return None, {"applied": False, "reason": "missing_track", "asset": None}
    try:
        volume = float(block.get("volume", 0.18))
    except (TypeError, ValueError):
        volume = 0.18
    ducking = block.get("ducking") is not False
    asset = dict(record)
    if track not in LIBRARY:
        asset["fallback"] = "bed_soft"
    music = {"path": str(path), "volume": min(1.0, max(0.0, volume)), "ducking": ducking, "license": asset}
    return music, {"applied": True, "ducking": "sidechaincompress" if ducking else None, "asset": asset}


def resolve_sfx(
    block: dict[str, Any],
    *,
    library: dict[str, Path],
    duration: float,
    broll_marks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    events = block.get("events") if isinstance(block.get("events"), list) else []
    if not events:
        points = [float(m["start"]) for m in broll_marks] or [0.15]
        events = [{"name": "whoosh", "at_s": point} for point in points[:4]]
    try:
        default_gain = float(block.get("gain", 0.35))
    except (TypeError, ValueError):
        default_gain = 0.35
    resolved: list[dict[str, Any]] = []
    assets: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        name = str(event.get("name") or "whoosh").strip().lower()
        record = library_record(name)
        path = library.get(name)
        if record is None or path is None or not path.is_file():
            assets.append({"error": "unknown_sfx", "name": name})
            continue
        try:
            at_s = float(event.get("at_s") or 0.0)
        except (TypeError, ValueError):
            continue
        if at_s < 0 or at_s >= duration:
            continue
        try:
            gain = float(event.get("gain", default_gain))
        except (TypeError, ValueError):
            gain = default_gain
        item = {"path": str(path), "at_s": at_s, "gain": min(1.0, max(0.0, gain)), "license": dict(record)}
        resolved.append(item)
        assets.append(dict(record) | {"at_s": round(at_s, 3)})
    applied: dict[str, Any] = {"applied": bool(resolved), "library": "cc0", "assets": assets}
    if not resolved:
        applied["reason"] = "no_events"
    return resolved, applied


def _subtitle_filter(path: Path) -> str:
    escaped = str(path.name).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    if path.suffix.lower() in {".ass", ".ssa"}:
        return f"ass={escaped}"
    return f"subtitles={escaped}"


def _enable(start: float, end: float) -> str:
    return f"enable='between(t\\,{start:.3f}\\,{end:.3f})'"


def compile_production(
    *,
    binary: str,
    source: str,
    output_path: str,
    start: float,
    duration: float,
    output_format: str,
    output_fps: float,
    captions_file: str | None,
    watermark: dict[str, Any] | None,
    broll: list[dict[str, Any]],
    music: dict[str, Any] | None,
    sfx: list[dict[str, Any]],
    transitions: dict[str, Any] | None,
    audio_filter: str | None,
    has_speech: bool,
) -> tuple[list[str], str | None]:
    """Full ffmpeg argv and the cwd required by the subtitle filter."""
    width, height = frame_size(output_format)
    chains: list[str] = []
    inputs: list[dict[str, Any]] = []

    def _add(spec: dict[str, Any]) -> int:
        inputs.append(spec)
        return len(inputs)  # source is input 0

    cuts = [item for item in broll if item.get("placement") == "cut"]
    overlays = [] if cuts else [item for item in broll if item.get("placement") != "cut"]
    for item in overlays + cuts:
        item["index"] = _add({"kind": "broll", "path": item["path"], "length": max(0.2, float(item["end"]) - float(item["start"]))})
    if music:
        music["index"] = _add({"kind": "music", "path": music["path"]})
    for item in sfx:
        item["index"] = _add({"kind": "sfx", "path": item["path"]})

    effect = "fade"
    fade = 0.4
    if transitions:
        effect = str(transitions.get("effect") or "fade")
        if effect not in XFADE_EFFECTS:
            effect = "fade"
        try:
            fade = float(transitions.get("duration_s") or 0.4)
        except (TypeError, ValueError):
            fade = 0.4
        fade = min(1.5, max(0.1, fade))
    use_head_tail = bool(transitions) and not cuts and duration > fade * 2 + 0.05
    use_cut_xfade = bool(transitions) and bool(cuts)
    black_index = None
    if use_head_tail:
        black_index = _add({
            "kind": "black",
            "lavfi": f"color=c=black:s={width}x{height}:r={output_fps:.3f}:d={fade:.3f}",
        })
    watermark_index = None
    if watermark and watermark.get("enabled") and watermark.get("file"):
        watermark_index = _add({"kind": "watermark", "path": str(watermark["file"])})

    if cuts:
        segments: list[tuple[str, float]] = []
        cursor = 0.0
        seg_i = 0
        ordered = sorted(cuts, key=lambda item: float(item["start"]))
        for cut in ordered:
            c0 = max(cursor, min(duration, float(cut["start"])))
            c1 = max(c0, min(duration, float(cut["end"])))
            if c0 - cursor >= 0.08:
                label = f"seg{seg_i}"
                seg_i += 1
                chains.append(
                    f"[0:v]trim=start={cursor:.3f}:end={c0:.3f},setpts=PTS-STARTPTS,{base_video_filter(output_format, output_fps)}[{label}]"
                )
                segments.append((label, c0 - cursor))
            label = f"seg{seg_i}"
            seg_i += 1
            length = max(0.08, c1 - c0)
            chains.append(
                f"[{cut['index']}:v]trim=duration={length:.3f},setpts=PTS-STARTPTS,{cover_filter(width, height, output_fps)}[{label}]"
            )
            segments.append((label, length))
            cursor = c1
        if duration - cursor >= 0.08:
            label = f"seg{seg_i}"
            chains.append(
                f"[0:v]trim=start={cursor:.3f}:end={duration:.3f},setpts=PTS-STARTPTS,{base_video_filter(output_format, output_fps)}[{label}]"
            )
            segments.append((label, duration - cursor))
        if not segments:
            video = "v0"
            graph_duration = duration
            chains.append(f"[0:v]{base_video_filter(output_format, output_fps)}[{video}]")
        else:
            video, graph_duration = _join_segments(
                chains, segments, effect=effect if use_cut_xfade else None, fade=fade
            )
    else:
        video = "v0"
        graph_duration = duration
        chains.append(f"[0:v]{base_video_filter(output_format, output_fps)}[{video}]")
        for item in overlays:
            length = max(0.2, float(item["end"]) - float(item["start"]))
            branched = f"br{item['index']}"
            alpha = ""
            if float(item.get("opacity") or 1) < 0.999:
                alpha = f",format=yuva420p,colorchannelmixer=aa={float(item['opacity']):.3f}"
            chains.append(
                f"[{item['index']}:v]trim=duration={length:.3f},"
                f"setpts=PTS-STARTPTS+{float(item['start']):.3f}/TB,"
                f"{cover_filter(width, height, output_fps)}{alpha}[{branched}]"
            )
            nxt = f"ov{item['index']}"
            chains.append(
                f"[{video}][{branched}]overlay=0:0:{_enable(float(item['start']), float(item['end']))}:eof_action=pass[{nxt}]"
            )
            video = nxt

    if use_head_tail and black_index is not None:
        chains.append(
            f"[{black_index}:v]scale={width}:{height},setsar=1,fps={output_fps:.3f},settb=AVTB,format=yuv420p,split=2[bk0][bk1]"
        )
        chains.append(f"[bk0][{video}]xfade=transition={effect}:duration={fade:.3f}:offset=0[vfi]")
        offset = max(0.0, duration - fade)
        chains.append(f"[vfi][bk1]xfade=transition={effect}:duration={fade:.3f}:offset={offset:.3f}[vxf]")
        video = "vxf"
        graph_duration = duration

    cwd: str | None = None
    if captions_file:
        cap = Path(captions_file)
        nxt = "vcap"
        chains.append(f"[{video}]{_subtitle_filter(cap)}[{nxt}]")
        video = nxt
        cwd = str(cap.parent)

    if watermark_index is not None and watermark is not None:
        wm_w = int(watermark.get("width") or 180)
        x, y = _position_xy(
            str(watermark.get("position") or "top_right"),
            int(watermark.get("margin_x") or 40),
            int(watermark.get("margin_y") or 40),
        )
        enable = ""
        if watermark.get("start") is not None or watermark.get("end") is not None:
            a = float(watermark.get("start") or 0.0)
            b = float(watermark["end"]) if watermark.get("end") is not None else duration
            enable = ":" + _enable(a, b)
        nxt = "vwm"
        chains.append(
            f"[{watermark_index}:v]scale={wm_w}:-1[wml];"
            f"[{video}][wml]overlay={x}:{y}{enable}[{nxt}]"
        )
        video = nxt

    audio_label = _audio_chain(
        chains,
        duration=graph_duration,
        audio_filter=audio_filter,
        has_speech=has_speech,
        music=music,
        sfx=sfx,
    )

    command = [binary, "-y", "-ss", f"{start:.3f}", "-i", source]
    for spec in inputs:
        if spec["kind"] == "broll":
            command += ["-t", f"{float(spec['length']):.3f}", "-i", spec["path"]]
        elif spec["kind"] == "music":
            command += ["-stream_loop", "-1", "-i", spec["path"]]
        elif spec["kind"] == "sfx":
            command += ["-i", spec["path"]]
        elif spec["kind"] == "black":
            command += ["-f", "lavfi", "-i", spec["lavfi"]]
        elif spec["kind"] == "watermark":
            command += ["-i", spec["path"]]
    command += ["-filter_complex", ";".join(chains), "-map", f"[{video}]"]
    if audio_label:
        command += ["-map", f"[{audio_label}]"]
    elif has_speech:
        command += ["-map", "0:a?"]
    command += [
        "-t", f"{max(0.1, graph_duration):.3f}",
        "-r", f"{output_fps:.3f}",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        output_path,
    ]
    return command, cwd


def _join_segments(
    chains: list[str],
    segments: list[tuple[str, float]],
    *,
    effect: str | None,
    fade: float,
) -> tuple[str, float]:
    if not segments:
        return "v0", 0.0
    if len(segments) == 1:
        return segments[0][0], segments[0][1]
    if not effect:
        labels = "".join(f"[{name}]" for name, _dur in segments)
        chains.append(f"{labels}concat=n={len(segments)}:v=1:a=0[vcat]")
        return "vcat", sum(dur for _name, dur in segments)
    current, current_dur = segments[0]
    for index, (label, dur) in enumerate(segments[1:], start=1):
        fd = min(fade, current_dur * 0.45, dur * 0.45)
        nxt = f"xv{index}"
        if fd < 0.08:
            chains.append(f"[{current}][{label}]concat=n=2:v=1:a=0[{nxt}]")
            current_dur = current_dur + dur
        else:
            offset = max(0.0, current_dur - fd)
            chains.append(
                f"[{current}][{label}]xfade=transition={effect}:duration={fd:.3f}:offset={offset:.3f}[{nxt}]"
            )
            current_dur = current_dur + dur - fd
        current = nxt
    return current, current_dur


def _audio_chain(
    chains: list[str],
    *,
    duration: float,
    audio_filter: str | None,
    has_speech: bool,
    music: dict[str, Any] | None,
    sfx: list[dict[str, Any]],
) -> str | None:
    pieces: list[str] = []
    speech = None
    if has_speech:
        head = "aformat=sample_rates=48000:channel_layouts=stereo"
        if audio_filter:
            head = f"{head},{audio_filter}"
        chains.append(f"[0:a]{head}[speech]")
        speech = "speech"
    if music and speech and music.get("ducking"):
        chains.append(
            f"[{music['index']}:a]atrim=0:{duration:.3f},asetpts=PTS-STARTPTS,"
            f"volume={float(music['volume']):.3f},aformat=sample_rates=48000:channel_layouts=stereo[music]"
        )
        chains.append("[speech]asplit=2[sc][dry]")
        chains.append(f"[music][sc]sidechaincompress={DUCK}[ducked]")
        chains.append("[dry][ducked]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[am0]")
        current = "am0"
    elif music and speech:
        chains.append(
            f"[{music['index']}:a]atrim=0:{duration:.3f},asetpts=PTS-STARTPTS,"
            f"volume={float(music['volume']):.3f},aformat=sample_rates=48000:channel_layouts=stereo[music]"
        )
        chains.append("[speech][music]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[am0]")
        current = "am0"
    elif music:
        chains.append(
            f"[{music['index']}:a]atrim=0:{duration:.3f},asetpts=PTS-STARTPTS,"
            f"volume={float(music['volume']):.3f},aformat=sample_rates=48000:channel_layouts=stereo[am0]"
        )
        current = "am0"
    else:
        current = speech
    for index, item in enumerate(sfx):
        delay = max(0, int(round(float(item["at_s"]) * 1000)))
        label = f"sfx{index}"
        chains.append(
            f"[{item['index']}:a]adelay=delays={delay}:all=1,volume={float(item['gain']):.3f},"
            f"aformat=sample_rates=48000:channel_layouts=stereo[{label}]"
        )
        pieces.append(label)
    if not pieces:
        return current
    if current is None:
        mix = "".join(f"[{name}]" for name in pieces)
        chains.append(
            f"{mix}amix=inputs={len(pieces)}:duration=longest:dropout_transition=0:normalize=0[aout]"
        )
        return "aout"
    mix_in = f"[{current}]" + "".join(f"[{name}]" for name in pieces)
    chains.append(
        f"{mix_in}amix=inputs={1 + len(pieces)}:duration=first:dropout_transition=0:normalize=0[aout]"
    )
    return "aout"


def _position_xy(position: str, margin_x: int, margin_y: int) -> tuple[str, str]:
    if position == "top_left":
        return f"{margin_x}", f"{margin_y}"
    if position == "bottom_left":
        return f"{margin_x}", f"H-h-{margin_y}"
    if position == "bottom_right":
        return f"W-w-{margin_x}", f"H-h-{margin_y}"
    if position == "center":
        return "(W-w)/2", "(H-h)/2"
    return f"W-w-{margin_x}", f"{margin_y}"


def prepare_production(
    payload: dict[str, Any],
    *,
    duration: float,
    original_only: bool,
    work_dir: Path,
    binary: str,
    pexels_key: str | None = None,
    pixabay_key: str | None = None,
    stock_search: Callable[..., dict[str, Any] | None] | None = None,
    has_speech: bool = True,
) -> dict[str, Any]:
    """Resolve assets and the applied{} report. Does not run the main encode."""
    broll_block = _block(payload, "broll")
    music_block = _block(payload, "music")
    trans_block = _block(payload, "transitions")
    sfx_block = _block(payload, "sfx")
    audio = _block(payload, "audio")
    no_added = bool(audio.get("no_added_music"))
    applied: dict[str, Any] = {}
    broll: list[dict[str, Any]] = []
    if _enabled(broll_block):
        broll, applied["broll"] = resolve_broll(
            broll_block,
            work_dir=work_dir,
            pexels_key=pexels_key,
            pixabay_key=pixabay_key,
            stock_search=stock_search,
        )
    music = None
    if _enabled(music_block):
        library = ensure_cc0(work_dir / "cc0", binary)
        music, applied["music"] = resolve_music(
            music_block,
            library=library,
            original_only=original_only,
            no_added_music=no_added,
        )
        if music and music["ducking"] and not has_speech:
            music["ducking"] = False
            applied["music"]["ducking"] = None
            applied["music"]["duck_reason"] = "no_speech"
        elif music and not music["ducking"]:
            applied["music"]["ducking"] = None
    transitions = None
    if _enabled(trans_block):
        effect = str(trans_block.get("effect") or "fade")
        if effect not in XFADE_EFFECTS:
            effect = "fade"
        try:
            fade = float(trans_block.get("duration_s") or 0.4)
        except (TypeError, ValueError):
            fade = 0.4
        fade = min(1.5, max(0.1, fade))
        if duration <= fade * 2 + 0.05 and not any(item.get("placement") == "cut" for item in broll):
            applied["transitions"] = {"applied": False, "filter": "xfade", "reason": "clip_too_short"}
        else:
            transitions = {"effect": effect, "duration_s": fade}
            applied["transitions"] = {
                "applied": True,
                "filter": "xfade",
                "effect": effect,
                "duration_s": fade,
            }
    sfx: list[dict[str, Any]] = []
    if _enabled(sfx_block):
        library = ensure_cc0(work_dir / "cc0", binary)
        sfx, applied["sfx"] = resolve_sfx(
            sfx_block, library=library, duration=duration, broll_marks=broll,
        )
    use_graph = bool(broll or music or sfx or transitions)
    return {
        "use_graph": use_graph,
        "applied": applied,
        "broll": broll,
        "music": music,
        "sfx": sfx,
        "transitions": transitions,
    }


def production_command(binary: str, **kwargs: Any) -> tuple[list[str], str | None]:
    return compile_production(binary=binary, **kwargs)


def dump_applied(applied: dict[str, Any]) -> str:
    return json.dumps(applied, sort_keys=True)
