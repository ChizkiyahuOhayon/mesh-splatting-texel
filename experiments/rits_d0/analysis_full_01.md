---
exp_id: MS-A40-260803-004
date: 2026-08-03
system: MeshSplatting
experiment: RITS-D0
classification: confirmatory_two_scene_gate
scenes: [garden, room]
train_views_per_scene: 4
test_views_per_scene: 4
selected_parent_faces_per_scene: 512
gpu: NVIDIA A40
source_revision: cabb5c43e510526b027b4f976b272e6613ffb5be
confirmatory: true
anomaly: false
tags: [rits, prolongation, window-donor, parity, mechanism-decomposition, negative-result]
---

# RITS-D0 Garden/Room full gate 01

## Purpose

Decompose the CSU-F0 midpoint-split render discontinuity into the preregistered
mechanisms (window domain incl. sub-pixel culling and depth key; face opacity)
and test whether inheriting them restores >= 80% of probe-region parity on both
scenes under the locked absolute limits.

## Results

Probe-region MAE by variant (v1 inherited split, v2 parent window, v3 parent
opacity, v4 both):

| Scene | v1 | v2 | v3 | v4 | v4 reduction vs v1 | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Garden | 4.2035e-3 | 3.7219e-4 | 4.2027e-3 | 3.7068e-4 | **91.2%** | PASS |
| Room | 1.8246e-3 | 1.2182e-3 | 1.8243e-3 | 1.2179e-3 | **33.3%** | FAIL (80% condition) |

Supporting facts:

- Variant 1 reproduced the CSU-F0 parity numbers to machine precision on both
  scenes, so the rebuilt extension left the donor-free path intact.
- Variant 4 satisfies both absolute CSU limits on both scenes (Garden global
  7.33e-6 / probe 3.71e-4; Room global 3.21e-5 / probe 1.22e-3) and beats
  variants 2 and 3 on both scenes.
- Maximum absolute error collapses under the donor window: 0.6708 -> 0.0129 on
  Garden (52x) and 0.3527 -> 0.0276 on Room (12.8x). The catastrophic
  discontinuities are gone in v4; the residual is smooth and low-amplitude.
- The sub-pixel culling diagnosis is confirmed directly: Garden children
  rendered per view rise from 1744/1908/1759/1836 (v1) to 1860/1924/1763/1852
  once the cull defers to the parent.
- Opacity inheritance is empirically negligible: v3 tracks v1 to ~0.02% on both
  scenes. The window group (edges, inradius, culling, depth key) carries
  essentially all of the recovered error.

## Decision

**RITS-D0 FAIL** under the locked conjunctive rule: Room recovers only 33.3% of
its probe-region discrepancy, below the required 80%. No threshold is relaxed
and the scene is not excluded.

## Interpretation (bounded by the preregistration)

The failing condition is the relative one, on the scene whose v1 discrepancy
already satisfied the absolute parity limits (Room passed CSU-F0). The v4
residual on Room is exactly the residual the protocol preregistered as
predicted: midpoint SH values are 3D-affine averages while the renderer
interpolates colors affinely in screen space, so on Room's large projected
triangles the appearance term does not shrink when the window and opacity are
inherited. The protocol's prediction that this residual stays "well below the
CSU parity limits" held on both scenes.

The claimed mechanisms are therefore supported (v4 strictly beats v2/v3 and
passes every absolute check), but they are insufficient on Room because the
operator tested here did not prolong appearance. The prolongation operator of
RESEARCH_PLAN_v13 explicitly includes the parent-barycentric appearance map;
RITS-D0's variant list implemented only its alpha half.

Per the locked rule, no training experiment or selector is authorized. The
topology branch may continue only through a new preregistered gate that
completes the operator (parent-domain appearance interpolation) and is held to
the same 80% standard on both scenes, with exit on failure. That gate is
RITS-D1; its protocol must be committed before its implementation is run.

## Raw material

Full server outputs: `results/garden_full_01.json` and `results/room_full_01.json`
(transcribed from the console output; the server `results.json` files under
`$NAS_ROOT/experiments/rits_d0_{garden,room}_full_01` are authoritative).
