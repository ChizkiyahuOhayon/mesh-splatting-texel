---
exp_id: MS-A40-260803-001
date: 2026-08-03
system: MeshSplatting
experiment: XVR-G0
classification: implementation_smoke
scene: garden
train_views: 2
test_views: 1
gpu: NVIDIA A40
source_revision: 7d80535d4915dbf261cce0867f3a5222d27fe2f9
confirmatory: false
anomaly: false
tags: [cross-view, residual, allocation, smoke]
---

# XVR-G0 Garden smoke 01

## Purpose

Verify the cross-view residual accumulator, face eligibility filter, held-out
error-mass measurement, and output contract before the locked Garden/Room gate.
This two-training-view, one-test-view run is an implementation smoke and cannot
trigger the preregistered scientific decision.

## Result

The run completed on all 6,968,004 faces and produced 90,805 eligible faces.
At the top-10% allocation budget:

| Signal | Held-out error capture | Lift over random |
|---|---:|---:|
| persistent error mass (primary) | 37.73% | 3.77x |
| raw error mass | 38.46% | 3.85x |
| projected coverage | 35.74% | 3.57x |
| world-space area | 25.80% | 2.58x |
| maximum blending | 17.79% | 1.78x |

The primary signal is within 1.90% of the raw residual oracle-like control. It
captures 112.02% more held-out error than maximum blending and 46.24% more than
world-space area, but only 5.56% more than projected coverage. The locked gate
requires a 10% relative advantage over the best non-residual control.

## Decision

**IMPLEMENTATION VALID; SCIENTIFIC DECISION UNRESOLVED.** The accumulator and
ranking pipeline produce finite, nontrivial, concentrated signals and the smoke
correctly records `decision.pass = null`. Run the unchanged locked protocol with
16 evenly spaced training views and every held-out view on both Garden and Room.
Do not tune the score or thresholds from this smoke.

## Raw material

Server result directory:
`/home/smbu/dy/nas/meshsplatting_smbu/experiments/xvr_g0_garden_smoke_01/`
