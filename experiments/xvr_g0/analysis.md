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

---

# XVR-G0 Garden/Room full gate — 2026-08-03

## Classification

**CONFIRMATORY TWO-SCENE GATE.** Garden used 16 deterministic training views and
all 24 held-out views; Room used the same 16-view signal and all 39 held-out
views. Both manifests report `confirmatory_settings = true`.

| Scene | Eligible faces | Primary capture | Lift | Raw capture | Best non-residual | Decision |
|---|---:|---:|---:|---:|---:|---|
| Garden | 1,003,391 | 38.689% | 3.869x | 41.950% | 40.009% (coverage) | **FAIL** |
| Room | 680,688 | 42.875% | 4.287x | 45.351% | 42.973% (coverage) | **FAIL** |

The primary persistent-error score is 3.30% below projected coverage and 7.77%
below raw residual on Garden. It is 0.23% below projected coverage and 5.46%
below raw residual on Room. Thus both scenes fail the same locked checks: the
primary neither beats the best non-residual control by 10% nor remains within 5%
of raw residual.

## Interpretation

The strong lift does not validate cross-view persistence: projected coverage alone
captures 40.01% and 42.97% of held-out error because large visible faces carry more
pixel mass. Taking the minimum residual over 16 views discards useful signal whenever
a face is easy in any one view. Raw residual adds only 4.85% and 5.53% relative
capture over coverage, so residual ranking itself contributes limited information
beyond area in these scenes.

## Locked decision

**XVR-G0 FAIL.** Do not implement XVR-G1, alter the minimum aggregation, or relax
the thresholds. Retire predictive residual allocation. The next gate must estimate
whether a topology operation can reduce loss, rather than where loss is currently
large.

## Provenance anomaly

The run command recorded `95c33c5f`, a mistyped short revision, instead of the true
commit `95c33c5e66fadffc87af1d3cfa6d64caa890b889`. This is a metadata-only anomaly:
the server SHA-256 hashes of `xvr_g0_eval.py`, `xvr_score.py`, and
`svsr_metadata.py` exactly match the true commit, and those runtime files are
unchanged from the preceding `7d80535` implementation commit. Raw manifests remain
unaltered; no numerical rerun is required.
