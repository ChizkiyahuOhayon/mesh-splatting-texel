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
