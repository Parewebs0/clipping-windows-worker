# MOLI-52 before / after

Same source and the same window (0.4s–6.4s).

- SNR estimated on the clip window, before loudnorm: 7.9 dB → 18.9 dB after DeepFilterNet 3.
- Loudnorm only (`engine: none`): -14.15 LUFS, true peak 1.28 dBTP.
- Enhance then loudnorm: -14.44 LUFS, true peak -0.82 dBTP.
- `original_only` with the engine requested anyway: -19.30 LUFS (not pulled to -14; enhance applied=False).

Files: `before.mp4` (loudnorm only), `after.mp4` (deepfilter then loudnorm), `before-window.wav`, `after-window.wav`.
