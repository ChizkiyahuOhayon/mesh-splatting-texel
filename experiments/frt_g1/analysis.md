---
exp_id: MS-A40-260730-002
date: 2026-07-30
system: MeshSplatting
experiment: FRT-G1
classification: implementation_smoke
scene: room
updates: 2
gpu: NVIDIA A40
source_revision: 8dd6b427002925a78381fe49eb7fd7f61b52b81b
confirmatory: false
anomaly: false
tags: [frozen-base, texels, integrity, smoke]
---

# FRT-G1 Room smoke 01

## Purpose

Verify causal isolation and checkpoint serialization before the locked Garden/Room
5,000-update training. This smoke is not evaluated on held-out quality and cannot
change the preregistered schedule or thresholds.

## Integrity result

- zero-initialized texel render max error: **0.0**;
- frozen base tensors unchanged: **true**;
- Adam parameter groups: **`["texels"]`**;
- carrier: order 2, 6,458,290 faces, 77,499,480 learned scalars;
- peak allocated GPU memory: **14.51 GB**;
- checkpoint SHA-256:
  `1be2154a3ed5b30bb6c0719622f3e23caf7be29351e0952af9ce4cf09d899932`.

The two optimizer updates ran at about 4.66 iterations/s. The recorded 220.8 s
starts immediately before optimization but ends after serializing and hashing the
large checkpoint on CIFS NAS; it is not two-step compute time.

The two logged losses come from different randomly selected training views. Their
ordering is not a convergence diagnostic and carries no scientific interpretation.

## Decision

**IMPLEMENTATION VALID.** Zero initialization is an exact no-op, every renderer-
defining base tensor remains frozen, and only the texel carrier is optimized. Run
the unchanged, preregistered Garden and Room 5,000-update jobs on separate GPUs.

## Raw material

Server result directory:
`/home/smbu/dy/nas/meshsplatting_smbu/experiments/frt_g1_room_smoke_01/`

---

# FRT-G1 Garden/Room full evaluation — 2026-08-02

## Classification

**CONFIRMATORY TWO-SCENE GATE.** Both models completed the locked 5,000 texel-only
updates with exact zero initialization, unchanged base tensors, and no optimization
parameter other than texels. In evaluation, each zero-carrier model exactly matches
its SH reference in PSNR, SSIM, and LPIPS, closing the integrity chain end to end.

| Scene | ΔPSNR | ΔSSIM | ΔLPIPS | Locked condition |
|---|---:|---:|---:|---|
| Garden | +0.2335 | +0.01672 | −0.03418 | PASS |
| Room | −0.0193 | −0.01252 | +0.01134 | **IMMEDIATE FAIL** |

Garden exceeds its `+0.20 dB` and `−0.020 LPIPS` requirements. Room remains within
the PSNR tolerance but worsens LPIPS by 0.01134, exceeding the preregistered
immediate-fail boundary of `+0.005`; it also fails the required material improvement
of `+0.10 dB` or `−0.010 LPIPS`.

## Locked decision

**FRT-G1 FAIL.** Do not expand to nine scenes and do not tune learning rate,
regularization, iteration count, order, or a Room-specific selector. COADAPT-G0
established that joint training-path co-adaptation is real, but FRT-G1 shows it is
not the sole cause of cross-scene failure: even on an exactly frozen strong base,
the residual carrier improves Garden while overfitting perceptual structure on Room.

Together with the earlier L2/within-face-variance and projected-footprint failures,
this retires the local texel branch as a CVPR-Oral method direction. The next action
is an outer-loop baseline/problem pivot, not a sixth texel patch.

## Raw material

Server result directories were supplied through `$FRT_GARDEN_EVAL` and
`$FRT_ROOM_EVAL`; their expanded paths were not printed in the terminal excerpt.
