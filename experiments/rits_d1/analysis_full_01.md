---
exp_id: MS-A40-260803-005
date: 2026-08-03
system: MeshSplatting
experiment: RITS-D1
classification: confirmatory_two_scene_gate
scenes: [garden, room]
train_views_per_scene: 4
test_views_per_scene: 4
selected_parent_faces_per_scene: 512
gpu: NVIDIA A40
source_revision: 55e6c6dcadbf3c5a967433930c45424a2ed65501
confirmatory: true
anomaly: false
tags: [rits, prolongation, appearance-donor, parity, positive-result]
---

# RITS-D1 Garden/Room full gate 01

## Purpose

Test whether completing the prolongation operator with parent-domain appearance
interpolation (DONOR_APPEARANCE) restores at least 80% of probe-region parity on
both scenes, under the locked absolute limits, with v1/v4 anchored to the
CSU-F0 and RITS-D0 recorded numbers.

## Results

Probe-region MAE (v1 inherited split; v4 alpha prolongation; v5 completed):

| Scene | v1 | v4 | v5 | v5 reduction vs v1 | Verdict |
|---|---:|---:|---:|---:|---|
| Garden | 4.2035e-3 | 3.7068e-4 | **2.0769e-8** | 99.9995% | PASS |
| Room | 1.8246e-3 | 1.2179e-3 | **2.4305e-11** | ~100% | PASS |

- Both anchors reproduced exactly: v1 matches CSU-F0 and v4 matches RITS-D0 to
  machine precision on both scenes.
- v5 is float-noise-level identity: global MAE 6.17e-10 (Garden) / 4.63e-12
  (Room); p99 absolute error is exactly 0.0 on every view of both scenes; one
  Room view is bitwise 0.0 everywhere.
- Maximum absolute error: Room 4.58e-5; Garden 3.07e-3 concentrated in one view
  (DSC08020), consistent with isolated shared-edge tie-breaking pixels; two
  Garden views max at 1-2e-7.
- 512 trained faces (2048 children) were replaced in a 6.9M/6.5M-face model and
  the rendered function moved by ~1e-8. Exact refinement invariance for
  connected triangle splats is now demonstrated on real scenes, not only in the
  unit-test reference.

## Decision

**RITS-D1 PASS** on both scenes under the locked rule. The mechanism ranking
from D0 stands: window group >> appearance > opacity (negligible), with the
appearance term dominant precisely on large projected triangles (Room).

Per the preregistration chain, this authorizes the differentiability check and
the fixed-budget metric survival experiment (deadline 2026-08-20). No selector
or benchmark claim is authorized by D1 itself: a parity identity is a
prerequisite, not a quality result.

## Raw material

Full server outputs transcribed to `results/{garden,room}_full_01` summaries;
the server `results.json` under `$NAS_ROOT/experiments/rits_d1_{garden,room}_full_01`
are authoritative.
