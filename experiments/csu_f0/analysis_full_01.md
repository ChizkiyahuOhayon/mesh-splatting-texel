---
exp_id: MS-A40-260803-003
date: 2026-08-03
system: MeshSplatting
experiment: CSU-F0
classification: confirmatory_two_scene_gate
scenes: [garden, room]
train_views_per_scene: 4
test_views_per_scene: 4
selected_parent_faces_per_scene: 512
gpu: NVIDIA A40
source_revision: 6e5864862331224efb6b1a4b563563b153c638f2
confirmatory: true
anomaly: false
tags: [counterfactual, subdivision, parity, gradients, negative-result]
---

# CSU-F0 Garden/Room full gate 01

## Purpose

Test the preregistered claim that inherited 1-to-4 midpoint subdivision is an
approximately function-preserving parameter expansion with usable gradients on
real scenes. Both Garden and Room must pass every locked check before a diagonal
Hessian utility estimator is implemented.

## Results

| Scene | Global MAE | Limit | Probe-region MAE | Limit | Probe pixels | Result |
|---|---:|---:|---:|---:|---:|---|
| Garden | 0.000110525 | 0.0001 | 0.00420346 | 0.002 | 121,796 | FAIL |
| Room | 0.0000548668 | 0.0001 | 0.00182458 | 0.002 | 172,809 | PASS |

Topology counts, the bitwise-unchanged original-parameter prefix, finite
gradients, and nonzero geometry and appearance gradients passed in both scenes.
Garden exceeded the global-MAE limit by 10.53% and the probe-region limit by
2.10 times. Room passed, although its aggregate probe-region MAE used 91.23% of
the allowed margin and one of its four views exceeded the aggregate threshold.

The combined appearance-gradient norm was `1.6713e-5` in Garden and `6.0006e-5`
in Room. These were respectively about 8,845 and 130,657 times the vertex-gradient
norm. The split therefore exposes gradients, but the immediate signal is dominated
by appearance and does not establish geometry utility.

## Decision

**CSU-F0 FAIL.** The two-scene conjunctive gate fails because Garden violates two
locked parity checks. Do not implement CSU-G0 or the diagonal-Hessian selector.
Do not relax thresholds or report Room alone as validation.

The result isolates a more basic problem: the current connected-mesh midpoint
split preserves topology bookkeeping and inherited parameters, but does not
preserve the renderer's function across scenes. Any subsequent refinement method
must first demonstrate cross-view render continuity independently of its face
selector.

## Raw material

The decision evidence transcribed from the console output is in
`results/garden_room_full_01.json`. The server `results.json` files remain the
authoritative raw records; their directories are the paths stored in `$CSU_GARDEN`
and `$CSU_ROOM` for this run.
