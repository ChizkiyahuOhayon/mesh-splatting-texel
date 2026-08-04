# RITS-P0: refinement by projection, donor-free

Status: **PREREGISTERED — no RITS-P0 output has been observed**

Date: 2026-08-04

## Question

The abrupt operator initialises midpoint parameters by interpolating the
parent's, which leaves a probe-region render discrepancy of `4.20e-3` on Garden
and `1.82e-3` on Room (CSU-F0, reproduced as RITS-D0 variant 1). RITS-D1 showed
that the parent's rendered function is recoverable to ~1e-8 **while donors are
active**, which is not a deployable state: donors carry no gradient for the new
degrees of freedom, and turning them off returns the model to the abrupt state.

Can the same function instead be reproduced **donor-free**, by projecting the
parent's rendered function onto the child parameter space rather than
interpolating parameters — and does that projection generalise to views it was
not fitted on?

## Why this is well posed

The rendered image is exactly linear in the vertex colours, since per-pixel
alpha and transmittance depend on geometry and opacity only. Fitting midpoint
appearance to a target image is therefore a linear least-squares problem.
Midpoint geometry enters nonlinearly and is optimised jointly by gradient
descent from the interpolated initialisation. D1 supplies the target: the donor
render reproduces the parent render to ~1e-8, so no separate reference is
needed.

## Locked probe

Identical to CSU-F0 and RITS-D0: final SH-only Garden and Room checkpoints,
four evenly spaced name-sorted training views for coverage and fitting, the 512
most-covered faces, four evenly spaced held-out views for evaluation, and the
probe-region mask of parent-dominated pixels.

Procedure per scene:

1. render the four **training** views with donors active and store them as
   targets (this is the parent's function, by D1);
2. install the split with inherited initialisation;
3. optimise **only** the midpoint parameters — midpoint vertices, opacity, DC
   and higher-order SH — with Adam for 1,000 steps at the restore-path learning
   rates, minimising mean squared error between the **donor-free** render and
   the stored target over the four training views; every original parameter
   stays frozen and is asserted bitwise unchanged afterwards;
4. render the four **held-out** views donor-free and compare against the
   unsplit model's render of the same views.

The implementation smoke uses one training view, one held-out view, 32 faces,
and 50 steps on Garden only, and can trigger no decision.

## Locked decision

Per scene, let `M_inherited` be RITS-D0 variant 1's probe-region MAE on the
held-out views and `M_projected` the same quantity after fitting. RITS-P0
passes only if, on **both** Garden and Room:

1. `M_projected <= 0.20 * M_inherited` (at least an 80% reduction);
2. the global held-out MAE also improves versus inherited;
3. every original parameter is bitwise unchanged and the topology counts are
   exact.

Condition 1 is evaluated on held-out views only. A fit that reduces training-view
error without transferring is a memorisation of those views, not a projection of
the function, and fails this gate.

No threshold, view, face count, step count, or learning rate may change after
any P0 output is observed. If P0 fails, the topology branch closes — final — and
the project pivots to the soft-compositor efficiency thesis.

## Reported alongside the decision

Training-view MAE before and after fitting, held-out MAE before and after, the
generalisation gap between them, the loss trace, and the norm of the parameter
change split into geometry and appearance groups. These do not enter the
decision; they characterise what the projection actually moved.
