# GoRFE-Q0 — angular-carrier representation gate

Status: **passed on attempt 04** at source `6f7b03d`; revised after attempts 02
and 03 and frozen before any scene score.  This gate observes no Garden or Room
data.

Attempt 02 showed that the original full-image vertex probe crossed the
rasterizer's discrete visibility boundary: a `2e-4` perturbation changed the
set of covered pixels.  MeshSplatting's native backward, like standard hard
rasterizers, differentiates the fixed-coverage branch and does not define a
gradient for that discrete membership jump.  Revision 1 therefore evaluates
the vertex derivative on the fixed interior window `[30:35, 30:35]`.  The
fixture, coordinate, epsilon, tolerance, coefficients, and all other checks
remain unchanged.  The full-image discrepancy and support change remain in the
result as diagnostics; attempt 02 remains a failure rather than being relabeled.

Attempt 03 confirmed one changed silhouette pixel but still failed on the fixed
interior window.  Code tracing found that barycentric derivatives reached
`dL_dpoints2D` but were never propagated through the perspective projection to
3D vertices when precomputed vertex colors were used.  Revision 2 implements
that missing Jacobian and adds the zero-detail parent's interior vertex gradient
as a decision control.  It does not alter the fixture, window, coordinate,
epsilon, tolerance, coefficients, or any earlier check.

## Object

The optional edge tensor accepts the legacy `[E,3]` P2-DC layout or a
`[E,4,3]` layout whose rows are `(DC, SH1[-1], SH1[0], SH1[1])`.  SH1 factors
use the stock real basis `(-C1*y, C1*z, -C1*x)` at each camera-to-vertex
direction and are barycentrically interpolated before multiplication by the
topology-shared P2 edge basis.

The native backward must include derivatives with respect to coefficients,
barycentric coordinates, and the normalized camera-to-vertex direction.

## Locked fixture and checks

- One triangle at positive depth, one canonical active edge, 64x64 render.
- Two camera positions with the same projection isolate angular dependence.
- Active all-zero `[1,4,3]` output is bitwise identical to the disabled parent.
- Legacy `[1,3]` DC and `[1,4,3]` with only row zero active are bitwise equal.
- A nonzero angular intervention changes the render and changes under camera motion.
- The dense CPU definition is identical on a shared edge under reversed incident-face orientation.
- All 12 coefficient derivatives are finite and match central differences at
  both cameras with maximum relative error at most `5e-3`; scale is clamped to 1.
- On the fixed interior window `[30:35, 30:35]`, vertex `(0,0)` analytic
  derivative is finite and matches central difference within `5e-2`, including
  barycentric, P2-basis, and normalized-direction paths without a visibility
  membership change.
- The zero-detail parent on the same interior window also matches its vertex
  finite difference within `5e-2`, directly testing the repaired perspective
  Jacobian without relying on cancellation from the angular carrier.
- The original full-image vertex comparison and the number of parent-support
  pixels changed by the perturbation are reported as non-decision diagnostics.
- The full CPU unit-test suite passes after rebuilding the extension.

Every Boolean must be true.  Failure stops P2-SH1 and does not invalidate the
sealed P2-DC E0 result.

## Artifact contract

The runner refuses an existing output directory or a dirty tracked checkout,
requires an exclusive physical GPU with at least 40,000 MiB free, builds in a
validated local `/tmp` directory, and writes the build log, full tests, result,
manifest, extension hash, environment, `DONE`, and `SHA256SUMS` to NAS.
Before reserving an attempt suffix it requires the locked environment
`torch 2.7.1+cu126` with CUDA `12.6`; unit discovery fixes the repository root
as its top-level directory so installed packages named `tests` cannot shadow
the project suite.
