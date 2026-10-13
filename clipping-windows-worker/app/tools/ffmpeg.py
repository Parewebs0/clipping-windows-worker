"""Wrapper de FFmpeg para procesamiento y renderizado de vídeo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import Settings
from app.utils.logging import get_logger
from app.utils.subprocess import run_command

logger = get_logger("ffmpeg")

# Publish loudness (MOLI-35). Skipped only when audio.original_only.
LOUDNORM_TARGET_LUFS = -14
LOUDNORM_TRUE_PEAK_DB = -1
LOUDNORM_LRA = 11

# Posiciones de watermark soportadas.
WATERMARK_POSITIONS = {
    "top_left": "x=0:y=0",
    "top_right": "x=W-w:y=0",
    "bottom_left": "x=0:y=H-h",
    "bottom_right": "x=W-w:y=H-h",
    "center": "x=(W-w)/2:y=(H-h)/2",
}


def _finite_number(value: Any) -> str | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if number != number or abs(number) == float("inf"):
        return None
    return f"{number:.2f}"


def loudnorm_filter(measured: dict[str, Any] | None = None) -> str:
    """Single-pass loudnorm, or a linear second pass when `measured` is usable."""
    chain = f"loudnorm=I={LOUDNORM_TARGET_LUFS}:TP={LOUDNORM_TRUE_PEAK_DB}:LRA={LOUDNORM_LRA}"
    if not measured:
        return chain
    fields = {
        "measured_I": _finite_number(measured.get("input_i")),
        "measured_TP": _finite_number(measured.get("input_tp")),
        "measured_LRA": _finite_number(measured.get("input_lra")),
        "measured_thresh": _finite_number(measured.get("input_thresh")),
        "offset": _finite_number(measured.get("target_offset")),
    }
    if any(value is None for value in fields.values()):
        return chain
    measured_opts = ":".join(f"{key}={value}" for key, value in fields.items())
    return f"{chain}:{measured_opts}:linear=true"


def parse_loudnorm_json(stderr: str) -> dict[str, Any] | None:
    marker = stderr.rfind('"input_i"')
    if marker < 0:
        return None
    start = stderr.rfind("{", 0, marker)
    end = stderr.find("}", marker)
    if start < 0 or end < 0:
        return None
    try:
        data = json.loads(stderr[start:end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


class FFmpegTool:
    """Encapsula la detección y uso de ffmpeg."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._binary = settings.ffmpeg_path or "ffmpeg"

    @property
    def available(self) -> bool:
        try:
            run_command([self._binary, "-version"], timeout=10)
            return True
        except Exception:  # noqa: BLE001 - cualquier error = no disponible
            return False

    def extract_audio(
        self,
        input_path: str | Path,
        output_path: str | Path,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> Path:
        """Extrae el audio de un vídeo a WAV mono (ideal para WhisperX)."""
        output = Path(output_path)
        run_command(
            [
                self._binary,
                "-y",
                "-i",
                str(input_path),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(sample_rate),
                "-ac",
                str(channels),
                str(output),
            ],
            timeout=600,
        )
        return output

    def prepare_loudnorm(
        self,
        input_path: str | Path,
        *,
        start: float,
        duration: float,
        original_only: bool,
        has_audio: bool,
    ) -> tuple[dict[str, Any], str | None]:
        """Decide loudnorm for this clip. Returns `(applied.audio, filter or None)`.

        `audio.original_only` leaves the track untouched. No audio stream means
        there is nothing to normalize. Otherwise measure, then linear loudnorm
        to -14 LUFS / -1 dBTP (single-pass if the measure is unusable).
        """
        meta: dict[str, Any] = {
            "normalized": False,
            "original_only": bool(original_only),
            "filter": None,
            "target_lufs": LOUDNORM_TARGET_LUFS,
            "true_peak_db": LOUDNORM_TRUE_PEAK_DB,
        }
        if original_only:
            meta["reason"] = "audio.original_only"
            return meta, None
        if not has_audio:
            meta["reason"] = "no_audio"
            return meta, None
        measured = self._measure_loudnorm(input_path, start=start, duration=duration)
        chain = loudnorm_filter(measured)
        meta["normalized"] = True
        meta["filter"] = "loudnorm"
        meta["lra"] = LOUDNORM_LRA
        meta["mode"] = "linear" if ":linear=true" in chain else "single_pass"
        return meta, chain

    def _measure_loudnorm(self, input_path: str | Path, *, start: float, duration: float) -> dict[str, Any] | None:
        command = [
            self._binary,
            "-hide_banner",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(input_path),
            "-t",
            f"{max(0.0, duration):.3f}",
            "-vn",
            "-af",
            loudnorm_filter() + ":print_format=json",
            "-f",
            "null",
            "-",
        ]
        try:
            result = run_command(command, timeout=600)
        except Exception as exc:  # noqa: BLE001 - unusable measure falls back to single-pass
            logger.warning("loudnorm measure failed", error=str(exc))
            return None
        parsed = parse_loudnorm_json(result.stderr or "")
        if parsed is None:
            logger.warning("loudnorm measure returned no json")
        return parsed

    def render_clip(
        self,
        *,
        input_path: str | Path,
        output_path: str | Path,
        start: float,
        end: float,
        output_format: str = "9:16",
        captions_file: str | Path | None = None,
        watermark: dict[str, Any] | None = None,
        output_fps: float = 30.0,
        audio_filter: str | None = None,
        fontsdir: str | Path | None = None,
    ) -> Path:
        """Renderiza un clip: trim + crop/scale + subtítulos + watermark + encode.

        ``output_format`` soporta "9:16" (1080x1920) y "16:9" (1920x1080).

        ``output_fps`` fuerza el framerate de salida (default 30.0) para que el
        QA no falle por fuentes nativos a 23.976/24/25/29.97. El ajuste se
        aplica como ``-r`` **después** del filtergraph (encoder/muxer), no
        como filtro ``fps=`` (eso duplicaría trabajo con ``scale``/``crop``).
        Para clips cortos de 30-60s este cambio sólo reescribe timestamps;
        si en el futuro hace falta interpolación real de frames, mover a
        ``fps=30:round=near`` dentro del filtergraph.
        """
        output = Path(output_path)
        duration = max(0.0, end - start)

        # --- Filtros de vídeo (crop/scale) ---
        vf_parts: list[str] = []
        if output_format == "9:16":
            vf_parts.append("crop=ih*9/16:ih")
            vf_parts.append("scale=1080:1920:force_original_aspect_ratio=decrease")
            vf_parts.append("pad=1080:1920:(ow-iw)/2:(oh-ih)/2")
        elif output_format == "16:9":
            vf_parts.append("crop=iw:iw*9/16")
            vf_parts.append("scale=1920:1080:force_original_aspect_ratio=decrease")
            vf_parts.append("pad=1920:1080:(ow-iw)/2:(oh-ih)/2")
        else:
            raise ValueError(f"Unsupported output_format: {output_format}")

        # --- Subtítulos (filtro de vídeo adicional) ---
        # Los filtros de FFmpeg (subtitles/ass) no aceptan rutas Windows con
        # drive colon (`C:\...`). Por ello se referencia el archivo por nombre
        # relativo y el proceso arranca con cwd en su carpeta.
        vfilter_cwd: str | None = None
        if captions_file:
            cf = Path(captions_file)
            vf_parts.append(self._subtitle_filter(cf.name, fontsdir=fontsdir))
            vfilter_cwd = str(cf.parent)

        # --- Construcción del comando ---
        # Nota: `-t` se coloca DESPUÉS de todos los inputs para que se aplique
        # al output (no al input). Si va antes de un `-i` extra (watermark),
        # ffmpeg lo interpreta como opción de input y no limita el clip.
        command = [
            self._binary,
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(input_path),
        ]

        has_watermark = bool(watermark and watermark.get("enabled"))

        if has_watermark:
            # El overlay usa 2 entradas (vídeo + logo): requiere -filter_complex.
            wm_file = watermark.get("file")
            if not wm_file:
                raise ValueError("watermark.file is required when watermark.enabled=true")
            command += ["-i", str(wm_file)]

        command += ["-t", f"{duration:.3f}"]

        if has_watermark:
            width = int(watermark.get("width", 180))
            x, y = self._position_xy(
                watermark.get("position", "top_right"),
                int(watermark.get("margin_x", 40)),
                int(watermark.get("margin_y", 40)),
            )
            enable = ""
            wm_start, wm_end = watermark.get("start"), watermark.get("end")
            if wm_start is not None or wm_end is not None:
                a = float(wm_start or 0.0)
                b = float(wm_end) if wm_end is not None else duration
                enable = f":enable='between(t,{a:.3f},{b:.3f})'"
            video_chain = (
                f"[0:v]{','.join(vf_parts)}[vmain];"
                f"[1:v]scale={width}:-1[wm];"
                f"[vmain][wm]overlay={x}:{y}{enable}[vout]"
            )
            if audio_filter:
                command += [
                    "-filter_complex",
                    f"{video_chain};[0:a]{audio_filter}[aout]",
                    "-map",
                    "[vout]",
                    "-map",
                    "[aout]",
                ]
            else:
                command += ["-filter_complex", video_chain]
                # `0:a?`: no fallar si la fuente no tiene audio (el QA lo detecta).
                command += ["-map", "[vout]", "-map", "0:a?"]
        else:
            # Sin watermark: simple filtergraph con -vf (1 entrada / 1 salida).
            command += ["-vf", ",".join(vf_parts)]
            if audio_filter:
                command += ["-af", audio_filter]

        # Forzar framerate de salida. Se coloca DESPUÉS del filtergraph
        # para que se aplique al stream codificado (no al input). Esto
        # evita que clips con fuente 23.976/24/25/29.97 fallen la regla
        # QA de ``min_fps >= 24`` por margen de milésimas.
        if output_fps and output_fps > 0:
            command += ["-r", f"{output_fps:.3f}"]

        command += [
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(output),
        ]

        logger.info(
            "rendering clip",
            input=str(input_path),
            output=str(output),
            start=start,
            end=end,
            format=output_format,
            watermark=has_watermark,
            loudnorm=bool(audio_filter),
            output_fps=output_fps,
        )
        run_command(command, timeout=3600, cwd=vfilter_cwd)
        return output

    def concat_clips(
        self,
        *,
        head: str | Path | None = None,
        body: str | Path,
        tail: str | Path | None = None,
        output_path: str | Path,
        max_side_seconds: float = 10.0,
    ) -> Path:
        """Concatena intro + clip + outro (MOLI-54), re-codificando.

        Cada tarjeta (vídeo o imagen) se recorta a `max_side_seconds`,
        re-escala al tamaño del clip principal y compone una imagen en blanco
        si no tiene pista de audio, para que las 2-3 partes compartan formato
        exacto (`concat` demuxer + re-encode libx264/aac). Sin intro ni outro
        no se re-codifica nada: se devuelve `body` tal cual.
        """
        body = Path(body)
        head = Path(head) if head else None
        tail = Path(tail) if tail else None
        out = Path(output_path)
        if not head and not tail:
            return body
        out.parent.mkdir(parents=True, exist_ok=True)

        side_specs = [("head", head), ("tail", tail)]
        # probe main size (needed to normalise cards to the same geometry)
        w, h = self._video_size(body) or (1080, 1920)
        head_norm = out.with_name("_brand_head.mp4")
        tail_norm = out.with_name("_brand_tail.mp4")
        for side, path in side_specs:
            if path is None or not path.is_file():
                continue
            norm = out.with_name(f"_brand_{side}.mp4")
            self._normalize_brand_card(path, norm, size=(w, h),
                                       max_seconds=max_side_seconds)

        # concat demuxer: file list (head, body, tail) then one encode pass
        listing = out.with_name("_brand_concat.txt")
        body_norm = body  # re-encode everything for uniformity
        order: list[str] = []
        if head and head.is_file():
            order.append(f"file '{head_norm.as_posix()}'")
        order.append(f"file '{body_norm.as_posix()}'")
        if tail and tail.is_file():
            order.append(f"file '{tail_norm.as_posix()}'")
        listing.write_text("\n".join(order) + "\n", encoding="utf-8")

        command = [
            self._binary, "-y",
            "-f", "concat", "-safe", "0", "-i", str(listing),
            "-c:v", "libx264", "-preset", "medium", "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(out),
        ]
        run_command(command, timeout=3600)
        listing.unlink(missing_ok=True)
        for tmp in (head_norm, tail_norm):
            if tmp.exists():
                tmp.unlink()
        return out

    def _video_size(self, path: str | Path) -> tuple[int, int] | None:
        from app.tools.ffprobe import FFprobeTool

        try:
            data = FFprobeTool(self.settings).probe(str(path))
        except Exception:  # noqa: BLE001
            return None
        for st in data.get("streams", []):
            if st.get("width") and st.get("height"):
                return int(st["width"]), int(st["height"])
        return None

    def _normalize_brand_card(
        self,
        src: str | Path,
        dst: str | Path,
        *,
        size: tuple[int | None, int | None],
        max_seconds: float,
    ) -> Path:
        """Vídeo o imagen → MP4 corto con el tamaño del clip principal.

        Imagen: se mantiene fija `max_seconds` con una pista de audio muda
        (`anullsrc`) para que el concat no diverja de las partes con sonido.
        Vídeo: recortado a `max_seconds`; sin audio → pista muda.
        """
        src = Path(src)
        dst = Path(dst)
        w, h = size
        scale = "" if not (w and h) else f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,fps=30,"
        is_video = src.suffix.lower() in (".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v")
        duration_arg = ["-t", f"{max(0.1, float(max_seconds)):.3f}"]
        if is_video:
            video_in = ["-ss", "0", "-i", str(src), *duration_arg]
        else:
            video_in = ["-loop", "1", "-i", str(src), *duration_arg]
        command = [
            self._binary, "-y",
            *video_in,
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-t", f"{max(0.1, float(max_seconds)):.3f}",
            "-vf", scale + "setsar=1",
            "-c:v", "libx264", "-preset", "medium", "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-shortest",
            str(dst),
        ]
        run_command(command, timeout=1200)
        return dst

    @staticmethod
    def _position_xy(position: str, margin_x: int, margin_y: int) -> tuple[str, str]:
        """Devuelve (x, y) para el overlay según la posición y márgenes."""
        if position == "top_left":
            return f"{margin_x}", f"{margin_y}"
        if position == "top_right":
            return f"W-w-{margin_x}", f"{margin_y}"
        if position == "bottom_left":
            return f"{margin_x}", f"H-h-{margin_y}"
        if position == "bottom_right":
            return f"W-w-{margin_x}", f"H-h-{margin_y}"
        if position == "center":
            return "(W-w)/2", "(H-h)/2"
        raise ValueError(f"Unsupported watermark position: {position}")

    @staticmethod
    def _subtitle_filter(path: str | Path, fontsdir: str | Path | None = None) -> str:
        """Elige el filtro de FFmpeg correcto según la extensión del archivo.

        - `.srt`  → filtro `subtitles` (libass lo convierte).
        - `.ass`/`.ssa` → filtro `ass` (formatos estilizados avanzados).

        `fontsdir` (MOLI-54): carpeta local del job con la fuente de marca
        para libass (`fontsdir=...`). Es necesario porque el filtro no acepta
        el `fontsdir` por command-line común.
        """
        ext = Path(path).suffix.lower()
        if ext in (".ass", ".ssa"):
            base = f"ass={FFmpegTool._escape_path(path)}"
        else:
            # Por defecto (incluido .srt): usar el filtro subtitles.
            base = f"subtitles={FFmpegTool._escape_path(path)}"
        if fontsdir and ext in (".ass", ".ssa", ".srt"):
            base += f":fontsdir={FFmpegTool._escape_path(str(fontsdir))}"
        return base

    @staticmethod
    def _escape_path(path: str | Path) -> str:
        """Escapa una ruta para usarla dentro de un filtro de FFmpeg."""
        return str(path).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
