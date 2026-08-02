# RITS-D0: source-of-discontinuity diagnostic for midpoint refinement

Status: **PREREGISTERED — no RITS-D0 output has been observed**

Date: 2026-08-02

## Question

CSU-F0 established that the inherited 1-to-4 midpoint split changes the rendered
function even though every original parameter is bitwise preserved. Which renderer
mechanisms cause the discrepancy, and does removing them restore parity?

This gate evaluates forward renders only. It trains nothing, updates nothing, and
authorizes no selector.

## Mechanism inventory (fixed before implementation)

Reading the production kernels identifies four ways a child face diverges from the
parent it replaces, beyond any parameter change:

1. **Window domain.** Each face's soft value is `phi^sigma` with
   `phi = max_k(d_k(p)) / (-r)`, where `d_k` are the face's own projected edge
   signed distances and `r` its projected inradius. Children recompute four
   different windows whose sum does not reproduce the parent's.
2. **Face opacity.** A face's opacity is the minimum of its three activated vertex
   opacities. Midpoint averaging does not preserve that minimum for any child
   whose minimum vertex is lost.
3. **Sub-pixel culling.** Preprocess culls any face whose projected inradius is
   below one pixel (`dist > -1`) or whose vertex-to-center distance is below one
   pixel. Children have half the parent's linear size, so every parent with
   projected inradius between one and two pixels loses all four children
   entirely. We predict this is a major contributor on Garden, whose outdoor
   geometry projects smaller triangles than Room's.
4. **Depth-key reordering.** The compositing sort key is the face centroid's view
   depth. Child centroids differ from the parent's, so faces from other surfaces
   can interleave differently between the children and the unsplit parent.

The "parent-domain window" condition is therefore defined to inherit items 1, 3,
and 4 together from the parent (window edges and inradius, preprocess culling and
frustum reference point, and depth sort key), while the pixel-inside test and tile
bounding box remain the child's own so the four children exactly partition the
parent's support. The "inherited parent opacity" condition covers item 2 alone.

## Predicted residual of the full prolongation

Even with all four mechanisms inherited, variant 4 is not analytically exact:
vertex colors are interpolated affinely in screen space, while midpoint SH values
are 3D-affine averages. Under perspective, the projected 3D midpoint does not sit
at the screen midpoint of the projected edge, so a small appearance discrepancy
proportional to per-face depth variation remains, plus float tie-breaking on
shared child edges. We predict this residual is well below the CSU parity limits;
if it is not, that is a genuine failure of the refinement-invariance premise, not
an implementation defect to be patched post hoc.

## Implementation: window donors

The rasterizer gains two optional per-call tensors and two flags:

- `window_source` int32 `[F]`: `-1` renders the face exactly as today; a
  non-negative value selects a row of `donor_indices`.
- `donor_indices` int32 `[D, 3]`: vertex indices of donor triangles (the original
  parent corners, which remain in the vertex buffer).
- `donor_window`, `donor_opacity`: booleans applying mechanism groups {1,3,4} and
  {2} respectively to every face with a donor.

When both tensors are absent the kernels must follow the original code path
bitwise. Backward under active donors is not implemented at D0 and must raise.

## Locked probe

Identical to CSU-F0: final SH-only Garden and Room checkpoints; four evenly
spaced, name-sorted training views accumulate rendered coverage; the 512
most-covered faces are split; parity is measured on four evenly spaced held-out
views against the unsplit render, globally and inside the probe-region mask.
The model is reloaded from the checkpoint for every variant.

Variants, evaluated per scene:

| Variant | donor_window | donor_opacity |
|---|---|---|
| 1 current inherited split | off | off |
| 2 parent-domain window | on | off |
| 3 parent face opacity | off | on |
| 4 full prolongation | on | on |

Variant 1 must reproduce CSU-F0's parity numbers on both scenes (regression
check on the eval path, tolerance 5% relative). Reported per variant, per scene:
global MAE, probe-region MAE, per-view MAE, 99th percentile and maximum absolute
channel error, and child-culling counts (faces with `radii == 0` among children).

The implementation smoke uses one training view, one held-out view, and 32 faces
on one scene, and can only validate execution.

## Locked decision

RITS-D0 **passes** only if, without touching any threshold, view, or face count:

1. variant 4 satisfies the original CSU parity limits (global MAE `<= 1e-4`,
   probe-region MAE `<= 2e-3`) on both Garden and Room;
2. variant 4 reduces probe-region MAE by at least 80% relative to variant 1 on
   each scene;
3. variant 4's probe-region MAE is strictly below variants 2 and 3 on each scene;
   otherwise the claimed joint mechanism is unsupported.

If D0 fails, retire refinement invariance and exit the topology branch entirely.
If D0 passes, the next gate is RITS-G0 (backward implementation and gradient
checks); no training experiment starts before G0 passes.
