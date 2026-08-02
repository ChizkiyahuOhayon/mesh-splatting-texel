---
exp_id: MS-A40-260803-002
date: 2026-08-03
system: MeshSplatting
experiment: CSU-F0
classification: implementation_smoke
scene: garden
train_views: 1
test_views: 1
selected_parent_faces: 32
gpu: NVIDIA A40
source_revision: 6e5864862331224efb6b1a4b563563b153c638f2
confirmatory: false
anomaly: false
tags: [counterfactual, subdivision, parity, gradients, smoke]
---

# CSU-F0 Garden smoke 01

## Purpose

Verify midpoint topology construction, inherited-parameter integrity, renderer
parity measurement, and new-parameter autograd before the locked two-scene gate.
This one-training-view, one-test-view run cannot trigger the CSU-F0 decision.

## Integrity result

- 32 parents created 94 unique midpoint vertices, consistent with two shared
  selected edges;
- topology counts are exact;
- every original geometry, opacity, DC, and higher-order SH prefix is bitwise
  unchanged;
- all new-parameter gradients are finite and predominantly nonzero;
- the output manifest records the exact source revision.

## Mechanism warning

Global split-versus-unsplit MAE is only `1.2849e-5`, below the locked `1e-4`
limit. Inside the 4,555 pixels dominated by probed parents, however, MAE is
`0.003819`, or 1.91 times the locked `0.002` limit, and the maximum channel error
is `0.5231`.

The combined SH gradient norm is `2.0297e-5`, about 19,308 times the geometry
gradient norm (`1.0512e-9`). This does not invalidate execution, but warns that the
inherited split may alter splat compositing locally and expose primarily appearance
rather than geometry utility.

## Decision

**IMPLEMENTATION VALID; MECHANISM AT RISK; SCIENTIFIC DECISION UNRESOLVED.** The
smoke is nonconfirmatory, so do not change interpolation or thresholds and do not
stop from this result alone. Run the unchanged four-train-view/four-test-view,
512-parent gate on both Garden and Room. CSU-F0 passes only if both scenes satisfy
every preregistered check.

## Raw material

Server result directory:
`/home/smbu/dy/nas/meshsplatting_smbu/experiments/csu_f0_garden_smoke_01/`
