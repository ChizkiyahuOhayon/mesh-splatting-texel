# FRT-G1: frozen-base residual texel training

Status: **PREREGISTERED — no FRT-G1 training output has been observed**
Date: 2026-07-30

## Question

Can per-face residual capacity improve held-out quality when it is prevented from
co-adapting the strong final MeshSplatting geometry, opacity, topology, and SH base?

## Fixed inputs

Use the exact final SH-only checkpoints from the completed paired experiment for
Garden and Room. Load their complete held-out splits and final meshes. The output
directory must not exist. Before the first render, record dataset/model paths, base
checkpoint SHA-256, train/test view names, image sizes, tensor counts, GPU/package
versions, and source revision.

## Locked training

- freeze vertices, triangle indices, vertex weights, sigma/opacity, SH DC, and SH rest;
- allocate zero-initialized order-2 residual texels on the final face set;
- assert that a zero-texel render matches the loaded SH base within `1e-7` max error;
- perform exactly 5,000 Adam updates of texels only, learning rate `0.0025`;
- use the original 4x compositor, seed 0, fixed black/white dataset background, and
  `0.8 * L1 + 0.2 * (1 - SSIM)`;
- use training views only for optimization and evaluate the held-out split once at
  the end; no checkpoint selection, regularizer, schedule, or scene-specific change.

A two-update `--frt_smoke` is implementation-only and cannot affect the locked run.
Base tensor identity, shape, and in-place version must remain unchanged through save.

## Evaluation and decision

Evaluate `sh_reference`, trained `fixed`, and its in-memory `zero` and `face_mean`
decompositions on the complete held-out split.

FRT-G1 passes to the nine-scene experiment only if all hold:

1. Garden gains at least `+0.20 dB` PSNR and reduces LPIPS-VGG by at least `0.020`.
2. Room is within `-0.05 dB` PSNR and `+0.002` LPIPS of SH, and materially improves
   at least one metric: `+0.10 dB` PSNR or `-0.010` LPIPS.
3. The evaluated zero-carrier model matches SH within `1e-4` in all aggregate metrics.
4. Both scenes use the identical locked settings above.

FRT-G1 fails immediately if either scene loses more than `0.10 dB` or worsens LPIPS
by more than `0.005`, or if Garden improves by neither `0.05 dB` nor `0.005` LPIPS.
Intermediate outcomes permit one correctness audit and one unchanged rerun, but no
hyperparameter adjustment.

Passing G1 is evidence for the training-path mechanism, not an Oral claim. The final
method must still show a robust nine-scene gain and causal co-adaptation ablations.
