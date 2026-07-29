# FMMS G0 exploratory analysis

## Garden smoke 02 — 2026-07-30

Classification: **EXPLORATORY** (`g0_max_views=1`, one timing repetition). This is
not a confirmatory G0 result.

The official 30k Garden checkpoint loaded 3,248,603 vertices and 6,934,075 triangles.
All four render variants completed on one held-out view.

| Variant | PSNR | SSIM | LPIPS-VGG | ΔPSNR vs SSAA4 | Time | Peak increment |
|---|---:|---:|---:|---:|---:|---:|
| ssaa4 | 21.494 | 0.7051 | 0.2705 | — | 67.9 ms | 4.75 GB |
| point1 | 19.935 | 0.5467 | 0.3689 | −1.559 | 50.1 ms | 1.30 GB |
| aa1 | 20.427 | 0.5763 | 0.3567 | −1.067 | 50.2 ms | 1.30 GB |
| aa2 | 20.882 | 0.6215 | 0.3275 | −0.612 | 92.7 ms | 1.30 GB |

Interpretation:

1. Analytic AA has a real effect: AA1 recovers about 0.49 dB over point1 and AA2
   recovers another 0.46 dB.
2. The remaining gap is far outside the preregistered tolerance: AA2 is still down
   0.61 dB, 0.084 SSIM, and up 0.057 LPIPS.
3. Native rasterization is geometry-bound at 6.9M triangles. AA1 is only about 1.35x
   faster than SSAA4 in this exploratory sample; AA2 is slower. Peak incremental
   memory improves by about 3.66x, short of the 4x gate.
4. Do not launch the full three-scene gate or G1 training until the residual gap is
   decomposed into sampling error versus renderer-semantic error.

Next experiment: the preregistered exploratory renderer ladder in
`renderer_ladder_protocol.md`.
