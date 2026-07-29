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

## Garden renderer ladder 01 — 2026-07-30

Classification: **EXPLORATORY MECHANISM DIAGNOSTIC** (one held-out view and one
timing repetition). The protocol in `renderer_ladder_protocol.md` was committed
before this output was observed.

| Variant | PSNR | SSIM | LPIPS-VGG | ΔPSNR vs SSAA4 | Time | Peak increment |
|---|---:|---:|---:|---:|---:|---:|
| splat1 | 19.923 | 0.5880 | 0.3493 | −1.571 | 33.5 ms | 2.13 GB |
| splat2 | 21.285 | 0.6879 | 0.2804 | −0.209 | 46.9 ms | 2.77 GB |
| ssaa4 | 21.494 | 0.7051 | 0.2705 | — | 67.7 ms | 4.75 GB |
| point1 | 19.935 | 0.5467 | 0.3689 | −1.559 | 52.8 ms | 1.30 GB |
| aa1 | 20.427 | 0.5763 | 0.3566 | −1.067 | 50.7 ms | 1.30 GB |
| aa2 | 20.882 | 0.6215 | 0.3275 | −0.612 | 92.8 ms | 1.30 GB |
| aa4 | 20.958 | 0.6304 | 0.3254 | −0.536 | 209.8 ms | 2.43 GB |

Precommitted decision: **renderer-semantic mismatch; stop FMMS native-AA
replacement and do not run G1 in its current form.** AA4 remains 0.536 dB and
0.0549 LPIPS behind SSAA4, exceeding both locked mismatch thresholds (0.20 dB and
0.01 LPIPS). Increasing the hard renderer from AA2 to AA4 recovers only 0.076 dB
while making it 2.26x slower. AA4 is 3.10x slower than SSAA4.

The same-renderer ladder also prevents an over-broad conclusion: sampling still
matters inside the MeshSplatting compositor. Splat2 recovers 1.362 dB over splat1,
and SSAA4 adds a further 0.209 dB. What is rejected is replacing that compositor
with the current opaque hard z-buffer plus coverage AA—not the general value of
footprint filtering.

Consequences:

1. Do not spend the three-scene budget on confirmatory G0: no hard-renderer variant
   from the mechanism ladder is close enough to make a pass plausible.
2. Retain the original MeshSplatting compositor for the next quality-led study.
3. Test the independent appearance-bandwidth hypothesis with a frozen-checkpoint,
   deterministic screen-footprint filter before implementing a learned hierarchy.
