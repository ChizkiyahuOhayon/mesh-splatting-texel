# XVR-G0: cross-view residual allocation gate

Status: **PREREGISTERED — no XVR-G0 output has been observed**  
Date: 2026-08-02

## Question

Does MeshSplatting's current opacity/size-based densification leave a predictable
quality gap that a cross-view persistent reconstruction-error score can target?

This is a diagnostic gate, not a method result. It changes neither checkpoints nor
training and does not authorize another texel or appearance experiment.

## Locked measurement

Use the final SH-only Garden and Room baselines. Select 16 evenly spaced, name-sorted
training views and every held-out test view. At final output resolution, assign each
covered pixel to its dominant face and compute its RGB L1 residual.

For face `f`, let `e_fv` be its mean pixel residual in visible training view `v`, and
`a_f` its mean covered pixels per visible training view. Faces need at least three
training views and one test-view pixel. The primary score has no tuned exponent:

```
persistent_error_mass_f = min_v(e_fv) * a_f
```

The minimum is a conservative cross-view persistence test; multiplying by `a_f`
converts residual severity to an error-mass lower bound. Compare against raw mean
error mass and the current non-residual signals: maximum blending weight, projected
coverage, and world-space area. For each signal, report the fraction of held-out L1
error mass captured by its top 1%, 5%, and 10% eligible faces.

## Locked decision

The scene passes at top 10% only if all hold:

1. at least 10,000 faces are eligible;
2. held-out error-mass capture lift is at least `1.75x` over random selection;
3. capture is at least 10% relatively higher than the best non-residual control;
4. capture is no more than 5% relatively below raw mean-error mass.

XVR-G0 passes only if **both Garden and Room pass** under identical settings. If it
passes, implement one fixed-budget error-guided subdivision intervention. If either
scene fails, retire residual-guided densification and do not tune the aggregation,
view count, alpha threshold, or gate thresholds.

Runs with fewer than 16 training views or a restricted test split are implementation
smokes only and cannot trigger a decision.
