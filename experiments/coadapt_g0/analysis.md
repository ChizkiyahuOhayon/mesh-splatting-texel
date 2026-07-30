---
exp_id: MS-A40-260730-001
date: 2026-07-30
system: MeshSplatting
experiment: COADAPT-G0
classification: exploratory_implementation_smoke
scene: room
views: 1
gpu: NVIDIA A40
source_revision: b6ca8ae0dfd33854ca299be08af1f09e54ffa72b
confirmatory: false
anomaly: false
tags: [coadaptation, texels, checkpoint-decomposition, smoke]
---

# COADAPT-G0 Room smoke 02

## Purpose

Verify that the locked evaluator can load the independent SH and texel checkpoints,
render the unchanged, zero-carrier, and face-mean variants, restore the original
carrier, and serialize metrics before running the full 39-view gate.

## Result

| Variant | PSNR | SSIM | LPIPS-VGG |
|---|---:|---:|---:|
| SH reference | 28.3946 | 0.8571 | 0.2332 |
| Fixed texel | 28.2825 | 0.8622 | 0.2297 |
| Zero texel | 26.0797 | 0.7170 | 0.3619 |
| Face mean | 26.1206 | 0.7082 | 0.3938 |

Relative to the unchanged texel checkpoint, zeroing the carrier loses 2.2028 dB
and worsens LPIPS by 0.13218. Replacing the carrier with its per-face mean loses
2.1618 dB and worsens LPIPS by 0.16414. On this view, the learned carrier strongly
compensates the jointly optimized base; its within-face component is essential.

The fixed checkpoint improves LPIPS over the SH reference on this particular view,
so LPIPS recovery is undefined and stored as `null`. Consequently
`gate_applicable=false`, as required for an exploratory smoke whose selected view
does not reproduce both full-set regressions.

## Interpretation

The implementation smoke passes and provides a strong preliminary co-adaptation
signal. It is not confirmatory evidence and does not authorize training. Run the
unchanged full 39-view Room decomposition next; only its aggregate PSNR and LPIPS
recoveries trigger the preregistered decision.

## Anomaly history

Smoke 01 rendered successfully but the original summary code raised when the
single-view LPIPS did not regress. Revision `b6ca8ae` changed only this exploratory
summary behavior: a non-applicable recovery is serialized as `null`. No renderer,
checkpoint, metric, or decision threshold changed.

## Raw material

Server result directory:
`/home/smbu/dy/nas/meshsplatting_smbu/experiments/coadapt_g0_room_smoke_02/`
