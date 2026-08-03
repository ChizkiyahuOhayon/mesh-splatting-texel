# RITS-D1: completed prolongation with parent-domain appearance

Status: **PREREGISTERED — no RITS-D1 output has been observed**

Date: 2026-08-03

## Question

RITS-D0 failed its locked gate: inheriting the window group and face opacity
recovered 91.2% of the probe-region discrepancy on Garden but only 33.3% on
Room, while passing every absolute limit and beating both single-mechanism
controls on both scenes. The residual matches the preregistered appearance
term: the renderer interpolates vertex colors affinely in screen space, so
midpoint colors formed by 3D averaging cannot reproduce the parent's color
field under perspective, and the effect grows with projected triangle size —
largest on Room, whose unsplit discrepancy was already within absolute limits.

D1 asks one question: does completing the prolongation operator with the
parent-domain appearance map — the third invariance item already specified in
RESEARCH_PLAN_v13, which D0's variant list did not implement — restore at
least 80% of probe-region parity on **both** scenes?

## Mechanism

For a face with an appearance donor, the per-pixel color barycentrics are
computed in the donor's projected triangle and applied to the donor's corner
vertex colors (and vertex depths), instead of the face's own. Because an
affine function on a triangle is determined by its values at any three
non-collinear points, the child's screen-affine interpolation of the parent's
field evaluated in the parent's frame reproduces the parent's color field
exactly; no per-view midpoint color could do this. Alpha handling is exactly
D0's variant 4. Texel carriers remain face-local; all D1 runs are SH-only.

Prediction: the completed prolongation (v5) leaves only float noise and edge
tie-breaking, so its probe-region MAE should fall well below D0's variant 4 on
both scenes, and its maximum absolute error should be far below v4's.

## Implementation

One new donor mode bit, `DONOR_APPEARANCE = 4`. The preprocess stores the
donor's projected vertices; the render kernel, for donor faces under this bit,
swaps the interpolation frame (uvs, vertex indices) to the donor's. Donor-free
calls and donor_mode values 1–3 must be bit-identical to the D0 revision.

## Locked probe

Identical to CSU-F0/RITS-D0: same checkpoints, view selection, coverage screen,
512 parents, model reload per variant. Variants evaluated per scene:

| Variant | donor_mode | meaning |
|---|---|---|
| v1_inherited | 0 | current inherited split (regression anchor) |
| v4_prolongation | 3 | D0's alpha prolongation (regression anchor) |
| v5_full_prolongation | 7 | alpha + parent-domain appearance |

Variant 1 must reproduce the CSU-F0 parity numbers and variant 4 must
reproduce the RITS-D0 variant-4 numbers, both within 5% relative, on each
scene; otherwise the run is invalid regardless of v5.

## Locked decision

RITS-D1 **passes** only if, without touching any threshold, view, or face
count, on **both** Garden and Room:

1. v5 satisfies the absolute CSU limits (global MAE `<= 1e-4`, probe-region
   MAE `<= 2e-3`);
2. v5 reduces probe-region MAE by at least 80% relative to v1;
3. v5's probe-region MAE is strictly below v4's.

If D1 fails, retire refinement invariance and exit the topology branch — this
exit is final; there is no D2. If D1 passes, the next gate is RITS-G0
(backward implementation and gradient checks) on the completed operator; no
training experiment starts before G0 passes.
