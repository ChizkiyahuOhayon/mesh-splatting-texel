# CSU-F0: counterfactual midpoint-split feasibility

Status: **PREREGISTERED — no CSU-F0 output has been observed**

Date: 2026-08-03

## Question

Can MeshSplatting's inherited 1-to-4 midpoint subdivision serve as an approximately
function-preserving parameter expansion with usable gradients on real scenes?

This is an integrity and feasibility gate. It neither ranks faces by proposed
utility nor changes a trained checkpoint.

## Locked probe

Use the final SH-only Garden and Room checkpoints. Accumulate rendered pixel
coverage over four evenly spaced, name-sorted training views and select the 512
most-covered faces. Coverage only ensures the probe touches visible faces; it is
not a method result. Replace every selected parent with four midpoint children,
sharing midpoint vertices across selected adjacent faces and inheriting endpoint
geometry, opacity, DC, and higher-order SH by interpolation.

Measure unsplit-versus-split render parity on four evenly spaced held-out views.
Then compute one training-view L1 backward pass and report gradients of only the
new midpoint parameters. Do not update any parameter.

The implementation smoke uses one training view, one held-out view, and 32 faces.
It cannot trigger a decision.

## Locked decision

Each confirmatory scene passes only if all hold:

1. topology counts are exact: each selected parent becomes four children and each
   unique selected edge creates one midpoint;
2. every original geometry, opacity, DC, and higher-order SH value is bitwise
   unchanged after installing the probe;
3. at least 100 held-out pixels are dominated by selected parent faces;
4. mean global split-versus-unsplit render MAE is at most `1e-4`;
5. mean MAE inside selected-parent regions is at most `2e-3`;
6. every new-parameter gradient is finite, and geometry and combined SH gradient
   norms are both nonzero.

CSU-F0 passes only if both Garden and Room pass without changing the thresholds,
view counts, face count, interpolation, or candidate screen. If either fails, do
not implement the diagonal-Hessian utility estimator.
