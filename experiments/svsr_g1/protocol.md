# SVSR-G1: frozen-checkpoint projected-footprint filter

Status: **PREREGISTERED — no SVSR-G1 output has been observed**
Date: 2026-07-30
Research plan: SVSR-v10

## Question

Are the cross-scene regressions from fixed per-face texels caused by exposing
within-face detail when a projected texel is subpixel?

## Fixed inputs

Use the exact SH-only and fixed-texel checkpoints and test-camera splits from the
completed nine-scene experiment for:

1. Garden, the positive control;
2. Room, a negative control;
3. Stump, a negative control.

Before the first image is rendered, write a manifest with the resolved dataset and
checkpoint paths, checkpoint SHA-256 hashes, final iteration, test-view names,
original image dimensions, GPU, package versions, Git commit, and filter settings.
The output directory must not exist before the run.

## Compared variants

1. `sh`: the paired SH-only result/checkpoint from the original experiment;
2. `fixed`: the fixed-texel checkpoint rendered without modification;
3. `footprint`: the same fixed checkpoint and renderer, changing only the tensor
   passed as texels.

For a face with residual texels `t`, final-resolution projected area `A`, and order
`L`, define:

```
m = mean(t, over L² cells)
w = clamp(A / L², 0, 1)
t_footprint = m + w * (t - m)
```

`A / L²` is the projected area of one equal-area texel in output-pixel units. The
weight is detached from geometry. No CUDA code, checkpoint tensor, training state,
camera, SH value, opacity, sigma, compositor setting, or output resolution changes.

## Metrics

Compute paired full-test-set PSNR, SSIM, and LPIPS-VGG from linear `[0,1]` tensors.
Also report per-view deltas, projected-texel-area quantiles over rendered faces, and
the fraction of faces in `w=1`, `0<w<1`, and `w=0` regimes. Save diagnostic images
for the five views with largest absolute PSNR changes.

## Precommitted decision

G1 passes only if all conditions hold:

1. Room and Stump each recover at least 50% of the fixed-texel PSNR regression
   relative to SH, with no LPIPS regression versus `fixed`.
2. Garden retains at least 70% of both the fixed-texel PSNR gain and LPIPS gain over
   SH.
3. One formula is used for all scenes; there is no tuned threshold.

G1 fails immediately if either negative control recovers less than 25%, or Garden
retains less than half of either gain. Values between the fail and pass thresholds
allow one projection/correctness audit, followed by one unchanged rerun.

## Interpretation

- **PASS:** implement the minimal two-band representation and multiview sampling
  support in G2.
- **FAIL:** within-face scale aliasing is not the main cause of the texel regressions;
  stop SVSR and reproduce DETRIS exactly.
- **MIXED:** do not tune `w`; inspect only projection units, paired checkpoints, and
  test-view identity.

This gate can support the mechanism but cannot establish an Oral-level contribution.
