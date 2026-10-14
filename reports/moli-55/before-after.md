# MOLI-55 before / after

Same lavfi source (black 640x360, 440 Hz sine, 5 s) and the same window 0.0–4.0 s.

- before.mp4: production blocks off. The worker takes the existing render_clip path.
- after.mp4: local green plate as overlay from 1.0–3.0 s (CC0-1.0), bed_soft ducked with sidechaincompress, head and tail xfade fade 0.3 s, whoosh at 1.0 s.

UAVG chroma at t=2.0: before 131.38, after 92.38, delta 39.00.

Cuts and overlays are not mixed: if any mark is a cut, overlays are skipped.
A missing Pexels/Pixabay key does not fail the render; that mark stays unresolved and the clip falls back to render_clip when nothing else was resolved.
