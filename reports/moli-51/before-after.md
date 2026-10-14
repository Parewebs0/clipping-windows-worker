# MOLI-51 before / after

Same asset and window: a 11.185 s clip, `start=0`, `end=11.185`, 9:16.

Spoken track (es-ES):

1. "Hoy vamos a hablar de casas." (toma mala)
2. "Hoy vamos a hablar de casas." (se conserva)
3. pausa larga
4. "Eh. O sea. La casa se monta en un dia."

`cleanup` off in `before.mp4` (11.185 s). On in `after.mp4` (`fillers`, `max_pause_ms=500`, `bad_takes`, lista por defecto de `es`).

`applied.cleanup.removed_seconds` = **7.182**:

| kind | seconds | what |
| --- | --- | --- |
| bad_take | 2.681 | primera "Hoy vamos a hablar de casas" |
| pauses | 3.530 | silencios por encima de 500 ms, con 80 ms de margen |
| fillers | 0.971 | "Eh" y "o sea" |

After duration 4.1 s. Subtítulos en el timeline nuevo: la frase que se queda entra en `0:00:00.08` y "la casa se monta" en `0:00:01.94` (justo después del corte). Los fundidos de audio son `afade` de 15 ms.
