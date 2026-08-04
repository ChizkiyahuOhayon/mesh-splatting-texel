---
exp_id: MS-A40-260805-001
date: 2026-08-05
system: MeshSplatting
experiment: SAC-G0
classification: confirmatory_single_scene_gate
scene: garden
gpu: NVIDIA A40
source_revision: a8eee50f4501e46b18739308971023849fc4a845
confirmatory: true
anomaly: true
tags: [sampling, supersampling, budget, negative-result, unanticipated-positive]
---

# SAC-G0: the reallocation thesis fails, and an unanticipated result appears

## Verdict

**FAIL.** Three locked conditions, one violated:

| Condition | Required | Measured | |
|---|---|---|---|
| stock reproduces garden_baseline | within 0.10 dB | 0.0121 dB | PASS |
| splat2@2 quality cost vs stock@4 | <= 0.35 dB / 0.020 LPIPS | 0.0726 dB / 0.0035 | PASS |
| render speedup, scaling 4 -> 2 | >= 2.5x | **1.53x** | **FAIL** |

The threshold is not relaxed. Note for the record that the preregistration was
internally inconsistent: its justification for the 0.35 dB bound reasoned about
"a saving that buys between 1.5x and 2x primitives", yet the speedup bar was set
at 2.5x. The inconsistency is visible in the committed protocol text and was
flagged before this measurement was taken, but the locked number governs.

## The four cells

| Cell | PSNR | SSIM | LPIPS | ms/view | triangles |
|---|---:|---:|---:|---:|---:|
| stock@4 | 24.7251 | 0.74875 | 0.24823 | 71.08 | 6,947,727 |
| stock@2 | 24.4321 | 0.73436 | 0.25935 | 46.32 | 6,947,727 |
| splat2@2 | 24.6525 | 0.74481 | 0.25170 | 43.81 | 5,699,991 |
| splat2@4 | 24.7827 | 0.75056 | 0.24796 | 66.59 | 5,699,991 |

The matrix separates what the renderer ladder confounded. Rendering the stock
model more cheaply costs 0.293 dB; training at the cheaper rate recovers 0.220
of that, or 75%; the residual cost at matched deployment is 0.073 dB. A model
trained at `scaling 2` still gains 0.130 dB from being rendered at `scaling 4`,
so the sampling rate has a real effect that training does not fully absorb.

## Why the thesis dies

Rendering is heavily primitive-bound at seven million triangles, so cutting
pixel samples fourfold buys only 1.53x. The reallocation argument required the
freed budget to purchase enough capacity to outweigh the sampling cost; 1.53x
does not, and the thesis as formulated is not supported.

## The unanticipated result

`splat2@4` beats `stock@4` on all three metrics — `+0.058 dB`, `+0.0018` SSIM,
`-0.00027` LPIPS — while carrying **18% fewer triangles** (5.70M vs 6.95M) and
rendering 6% faster. Reducing the final-phase supersampling did not merely cost
less; it produced a smaller and slightly better model.

Two cautions, both material:

1. The arms are identical only until iteration 25,000. Pruning and densification
   statistics (`image_size`, `prune_size`, `importance_score`) are measured in
   supersampled pixels, so changing the factor changes which primitives survive.
   The comparison therefore conflates the sampling rate with a primitive-count
   difference and is not a clean sampling ablation.
2. `+0.058 dB` on one scene with one seed is inside plausible seed noise. This
   pipeline has no measured seed variance, so the PSNR difference alone is not
   evidence. The 18% primitive reduction is far outside anything noise would
   produce and is the robust part of the observation.

Correcting the quality for the primitive deficit using MeshSplatting's own
capacity ablation (about `+0.5 dB` per doubling of vertices) suggests training
at `scaling 2` is worth roughly `+0.2 dB` at matched primitive count. That is an
estimate from a scaling law measured elsewhere, not a measurement.

## Status

The sampling-budget reallocation thesis is closed. What remains is an
observation, not a result: it is single-seed, single-scene, and confounded. It
warrants exactly one cheap replication that removes the confound and measures
seed variance; if that does not hold, this axis is exhausted and the project's
goal must be reconsidered rather than a further variant attempted.
