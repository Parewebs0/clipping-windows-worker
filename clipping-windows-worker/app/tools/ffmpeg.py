"""Wrapper de FFmpeg para procesamiento y renderizado de vídeo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import Settings
from app.tools.speech_cleanup import FADE_S
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


def cleanup_concat_graph(
    keeps: list[tuple[float, float]],
    *,
    fade_s: float = FADE_S,
    video: bool = True,
) -> str:
    """trim/atrim + concat, with a 10–20 ms afade on each audio piece.

    Labels the joined streams `[vcat]` (when `video`) and `[acat]`.
    Keep times are seconds relative to the already-seeked clip.
    """
    if not keeps:
        raise ValueError("cleanup concat needs at least one keep range")
    parts: list[str] = []
    vlabels: list[str] = []
    alabels: list[str] = []
    single = len(keeps) == 1
    for index, (start, end) in enumerate(keeps):
        dur = max(0.0, end - start)
        fade = fade_s if dur >= 0.050 and 0.010 <= fade_s <= 0.020 else 0.0
        vlabel = "[vcat]" if single else f"[v{index}]"
        alabel = "[acat]" if single else f"[a{index}]"
        if video:
            parts.append(f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS{vlabel}")
            vlabels.append(vlabel)
        fade_chain = ""
        if fade:
            fade_chain = f",afade=t=in:st=0:d={fade:.3f},afade=t=out:st={max(0.0, dur - fade):.3f}:d={fade:.3f}"
        parts.append(
            f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS,"
            f"aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo{fade_chain}{alabel}"
        )
        alabels.append(alabel)
    if not single:
        interleaved = "".join(
            (vlabels[index] if video else "") + alabels[index] for index in range(len(keeps))
        )
        if video:
            parts.append(f"{interleaved}concat=n={len(keeps)}:v=1:a=1[vcat][acat]")
        else:
            parts.append(f"{interleaved}concat=n={len(keeps)}:v=0:a=1[acat]")
    return ";".join(parts)


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
        keeps: list[tuple[float, float]] | None = None,
    ) -> tuple[dict[str, Any], str | None]:
        """Decide loudnorm for this clip. Returns `(applied.audio, filter or None)`.

        `audio.original_only` leaves the track untouched. No audio stream means
        there is nothing to normalize. Otherwise measure, then linear loudnorm
        to -14 LUFS / -1 dBTP (single-pass if the measure is unusable).
        `keeps` measures the concatenated speech-cleanup timeline instead of
        the whole window, so removed silence does not pull the loudness down.
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
        if keeps:
            measured = self._measure_loudnorm_keeps(input_path, start=start, keeps=keeps)
        else:
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

    def _measure_loudnorm_keeps(
        self,
        input_path: str | Path,
        *,
        start: float,
        keeps: list[tuple[float, float]],
    ) -> dict[str, Any] | None:
        graph = cleanup_concat_graph(keeps, video=False)
        graph = f"{graph};[acat]{loudnorm_filter()}:print_format=json[aout]"
        command = [
            self._binary,
            "-hide_banner",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(input_path),
            "-filter_complex",
            graph,
            "-map",
            "[aout]",
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
        keeps: list[tuple[float, float]] | None = None,
    ) -> Path:
        """Renderiza un clip: trim + crop/scale + subtítulos + watermark + encode.

        `keeps` (clip-relative) switches to trim/atrim + concat. None keeps the
        single `-ss`/`-t` path, which is the default when speech cleanup is off.

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
        if keeps:
            return self._render_kept_segments(
                input_path=input_path,
                output_path=output,
                start=start,
                end=end,
                output_format=output_format,
                captions_file=captions_file,
                watermark=watermark,
                output_fps=output_fps,
                audio_filter=audio_filter,
                keeps=keeps,
            )

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
            vf_parts.append(self._subtitle_filter(cf.name))
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

    def _render_kept_segments(
        self,
        *,
        input_path: str | Path,
        output_path: Path,
        start: float,
        end: float,
        output_format: str,
        captions_file: str | Path | None,
        watermark: dict[str, Any] | None,
        output_fps: float,
        audio_filter: str | None,
        keeps: list[tuple[float, float]],
    ) -> Path:
        """Join keep ranges with trim/atrim + concat + a short audio fade."""
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

        vfilter_cwd: str | None = None
        if captions_file:
            cf = Path(captions_file)
            vf_parts.append(self._subtitle_filter(cf.name))
            vfilter_cwd = str(cf.parent)

        duration = max(0.0, end - start)
        graph = cleanup_concat_graph(keeps, video=True)
        graph = f"{graph};[vcat]{','.join(vf_parts)}[vmain]"

        has_watermark = bool(watermark and watermark.get("enabled"))
        command = [
            self._binary,
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(input_path),
        ]
        if has_watermark:
            wm_file = watermark.get("file")
            if not wm_file:
                raise ValueError("watermark.file is required when watermark.enabled=true")
            command += ["-i", str(wm_file)]
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
                b = float(wm_end) if wm_end is not None else sum(stop - begin for begin, stop in keeps)
                enable = f":enable='between(t,{a:.3f},{b:.3f})'"
            graph += (
                f";[1:v]scale={width}:-1[wm];"
                f"[vmain][wm]overlay={x}:{y}{enable}[vout]"
            )
            vout = "[vout]"
        else:
            vout = "[vmain]"

        if audio_filter:
            graph += f";[acat]{audio_filter}[aout]"
            aout = "[aout]"
        else:
            aout = "[acat]"

        command += ["-filter_complex", graph, "-map", vout, "-map", aout]
        if output_fps and output_fps > 0:
            command += ["-r", f"{output_fps:.3f}"]
        command += [
            "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
            str(output_path),
        ]
        logger.info(
            "rendering cleaned clip",
            input=str(input_path),
            output=str(output_path),
            start=start,
            keeps=len(keeps),
            duration=round(duration, 3),
            watermark=has_watermark,
            loudnorm=bool(audio_filter),
        )
        run_command(command, timeout=3600, cwd=vfilter_cwd)
        return output_path

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
    def _subtitle_filter(path: str | Path) -> str:
        """Elige el filtro de FFmpeg correcto según la extensión del archivo.

        - `.srt`  → filtro `subtitles` (libass lo convierte).
        - `.ass`/`.ssa` → filtro `ass` (formatos estilizados avanzados).
        """
        ext = Path(path).suffix.lower()
        if ext in (".ass", ".ssa"):
            return f"ass={FFmpegTool._escape_path(path)}"
        # Por defecto (incluido .srt): usar el filtro subtitles.
        return f"subtitles={FFmpegTool._escape_path(path)}"

    @staticmethod
    def _escape_path(path: str | Path) -> str:
        """Escapa una ruta para usarla dentro de un filtro de FFmpeg."""
        return str(path).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
