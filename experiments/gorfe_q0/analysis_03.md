# GoRFE-Q0 attempt 03 — valid fixed-support failure

Date: 2026-08-11

Source revision: `d59e69a583575a2467e768b598c6a4dd77d93f74`

Artifact directory: `${NAS_ROOT}/experiments/gorfe_q0_03`

## Classification

**VALID FAILED GATE.** The native extension built in the locked environment on
one exclusive NVIDIA A40 and all 200 CPU tests passed.  The CUDA smoke reached
its decision and correctly stopped before manifest, checksum ledger, and
`DONE` because the fixed-support vertex-gradient check was false.

## Result

The run reproduced every representation and coefficient-gradient success from
attempt 02.  It also confirmed the visibility diagnosis:

- the parent support changed by exactly one pixel under the vertex perturbation;
- the original full-image derivative remained analytic `-0.70129317` versus
  finite difference `-403.30887`.

However, the fixed `[30:35, 30:35]` interior window still failed:

- analytic derivative: `-0.01545610`;
- central difference: `-0.71167946`;
- relative error: `0.69622338`.

Thus visibility explained the full-image explosion but not the remaining local
gradient omission.  Attempt 03 remains failed and immutable.

## Root cause and repair contract

The render backward correctly accumulated barycentric-coordinate effects into
the per-vertex screen-space buffer `dL_dpoints2D`.  No kernel consumed that
buffer: the nominal vertex-color backward accepted it but never applied the
perspective projection Jacobian, and that kernel was skipped entirely for the
precomputed vertex colors used by MeshSplatting and the Q0 fixture.  The
analytic value therefore contained the new normalized-direction contribution
but omitted most of the fixed-support barycentric geometry contribution.

Revision 2 adds a dedicated always-executed vertex-geometry kernel implementing
the exact derivative of `ndc2Pix(p_hom.xy / (p_hom.w + 1e-7))` plus the existing
view-depth path.  Gradients are accumulated rather than overwriting other 3D
vertex contributions.  A zero-detail parent finite-difference control is added
on the same interior window.  No epsilon, tolerance, fixture, coefficient, or
earlier decision check changes.  Replay must use a fresh suffix `_04`.
