# GoRFE-Q0 attempt 02 — valid gate failure

Date: 2026-08-11

Source revision: `f7b4fd1fed7f68c21277d1455e927be0b3c01924`

Artifact directory: `${NAS_ROOT}/experiments/gorfe_q0_02`

## Classification

**VALID FAILED GATE.** The locked environment was correct
(`torch 2.7.1+cu126`, CUDA `12.6`) on one NVIDIA A40.  The runner reached the
CUDA smoke and correctly stopped before manifest, checksum ledger, and `DONE`
because one of eleven decision checks was false.

## Result

Ten checks passed, including zero-parent identity, legacy-DC identity,
orientation continuity, angular camera dependence, and all 24 coefficient
gradient comparisons across two cameras.  The coefficient maximum relative
errors were `0.0023586` and `0.00077769`, both below `0.005`.

The full-image vertex check failed:

- analytic derivative: `-0.70129347`;
- central difference: `-403.30887`;
- relative error: `0.99826115`.

The result decision is `fail`.  `result.json` exists; `manifest.json`,
`SHA256SUMS`, and `DONE` do not, as required by the write-once runner.

## Diagnosis and revision

The vertex perturbation changes hard triangle coverage at the silhouette.  A
central difference of the sum over every pixel therefore includes a discrete
pixel-membership jump of order `1/epsilon`; the analytic MeshSplatting
rasterizer differentiates only the current fixed-coverage branch.  At such a
jump the full discrete renderer is not differentiable, so the finite difference
is not a valid reference for the new carrier's local derivative.

Revision 1 keeps the same vertex, epsilon, tolerance, coefficients, and native
backward, but evaluates the vertex derivative over the fixed interior window
`[30:35, 30:35]`.  This still exercises the barycentric, P2-basis, and
normalized-direction paths.  The original full-image comparison and a support
change count remain reported as diagnostics.  Attempt 02 is immutable and is
not retroactively passed.
