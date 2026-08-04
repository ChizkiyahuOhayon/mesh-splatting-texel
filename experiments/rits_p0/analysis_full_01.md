---
exp_id: MS-A40-260804-003
date: 2026-08-04
system: MeshSplatting
experiment: RITS-P0
classification: confirmatory_two_scene_gate
scenes: [garden, room]
gpu: NVIDIA A40
source_revision: abb6fa9b6261395489312dd56c61006ebab83fa8
confirmatory: true
anomaly: false
tags: [rits, projection, overfitting, negative-result, branch-closed]
---

# RITS-P0: projection does not transfer, and the topology branch closes

## Result

Held-out probe-region MAE, donor-free, against the unsplit render:

| Scene | Inherited | Projected | Change |
|---|---:|---:|---|
| Garden | 4.2035e-3 | 9.7127e-3 | **2.31x worse** |
| Room | 1.8246e-3 | 7.8944e-3 | **4.33x worse** |

Held-out global MAE also worsened, from 1.1053e-4 to 2.6588e-4 on Garden and
from 5.4867e-5 to 2.8676e-4 on Room. Both scenes fail both discrepancy checks.

The run itself is valid: the original parameter prefix is bitwise unchanged and
the topology counts are exact on both scenes, and the measurement is the same
probe used by CSU-F0, RITS-D0, and RITS-D1.

## Interpretation

Fitting the appended parameters to the parent's rendered function on four views
does not project the function; it memorises those views. The midpoint block
carries roughly 1,500 vertices with three position, three DC, and 45
higher-order SH values each — about 7e4 free parameters — and the higher-order
coefficients are view-dependent, so four views cannot constrain view dependence
at all.

RITS-D0 had already shown that the window group, not appearance, dominates the
child-local discrepancy, and that a child's window family cannot reproduce its
parent's. The fit therefore had no way to match the target through the
mechanism responsible for the error, and spent its remaining freedom —
view-dependent appearance — on the four fitted views. The consequence is worse
behaviour everywhere else, which is exactly what the held-out clause was
written to detect.

This is a property of what was asked of the representation, not a tuning
failure. Constraining the fit (DC only, more views, regularisation toward the
inherited values) would reduce the damage, but the quantity it must reproduce
is unreachable in the child-local window family, so the ceiling is set by that
family, not by the optimiser.

## Decision

**RITS-P0 FAIL on both scenes.** Per the preregistered rule, the topology
branch closes. This exit is final and is not revisited by a further variant.

Unlike RITS-T0 attempt 01, which was void because every arm fell below its own
starting checkpoint, this run's platform behaved correctly and its verdict is
evidence.

## What the branch established

- Connected triangle splats admit **exact refinement invariance** under a
  parent-domain prolongation: probe-region MAE 2.08e-8 on Garden and 2.43e-11
  on Room, with p99 error exactly zero on every view (RITS-D1). This result
  stands on its own.
- The discrepancy of the production split decomposes as window group >>
  appearance > opacity, with sub-pixel culling of half-size children a
  previously unreported contributor (RITS-D0).
- Exactness holds only while donors are active, and donors carry no gradient
  for the new degrees of freedom. Turning them off returns the model to the
  abrupt state (RITS-T0 analysis), and fitting the difference away does not
  generalise (this gate). Exact refinement invariance therefore does not, by
  itself, yield an optimisation or quality advantage.
