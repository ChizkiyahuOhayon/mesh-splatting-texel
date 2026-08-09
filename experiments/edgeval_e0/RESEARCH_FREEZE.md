# EdgeVal research freeze after HARD-G0

Date: 2026-08-08

## Decision

HARD-G0 falsified earlier global hardening: on Garden seed 0, moving
`sigma_until` from 30,000 to 25,000 changed scaling-4 PSNR by -0.8136 dB,
SSIM by -0.0493, and LPIPS by +0.0430. We will not run its A1/bridge follow-ups
or tune the failed endpoint after observing it.

The next hypothesis is representation-level: a connected mesh can expose one
zero-initialized RGB detail row per canonical undirected edge with quadratic P2
basis `4*lambda_i*lambda_j`. Under frozen geometry, opacity, visibility,
primitive order, and base appearance, the rendered pixels are exactly affine in
these rows. Their RGB squared-error change can therefore be evaluated exactly,
including negative changes, rather than approximated by residual magnitude or a
first-order renderer score.

## Staged authorization

1. **E0 — representation gate (implemented, awaiting GPU evidence).** Require a
   clean native build, full mathematical-core tests, bitwise zero-detail render
   identity, a nonzero effect, and a finite CUDA gradient matching central
   difference. No Garden checkpoint is used.
2. **E1 — independent-value gate (not authorized until E0 passes).** On one
   sealed Garden checkpoint, export the exact sparse edge design, accumulate
   float64 fold statistics, and test whether four-fold signed values transfer
   across cameras better than residual, same-view gain, coverage, and random-ID
   controls. Freeze its executable pass predicate before observing scores.
3. **E2 — conditional budget gate (not authorized until E1 passes).** Resolve
   overlapping supports by cross-fitted conditional forward selection. For a
   current set `S` and candidate `e`, jointly ridge-fit `S` and `S+e` only on
   complementary folds and rank their signed exact held-out difference minus
   one standard error. Ordinary OMP is explicitly excluded because it need not
   use the independently measured value and can make the permutation control
   inert.
4. **E3 — equal-budget quality gate (not authorized until E2 passes).** Reset
   selected physical rows to zero, freeze every parent tensor, train only the
   selected rows under one preregistered optimizer, and compare official
   PSNR/SSIM/LPIPS against equal-parameter controls. Only a prospective pass can
   authorize multi-scene, multi-seed full training.

## Fixed E0/E1 mathematics

- Local face-edge order is `(v0,v1)`, `(v1,v2)`, `(v2,v0)`; endpoint pairs are
  canonicalized and globally sorted. Degenerate and non-manifold inputs fail.
- Cameras are sorted by immutable image name and assigned `rank % 4`.
- Every fold must contain at least three contributing pixels for an edge.
- Ridge candidates are `10**s * trace(H)/3`, `s=-6,...,2`; GCV uses
  `n*RSS/(n-df)**2`, and exact ties choose the larger ridge.
- The held-out signed gain is `2*r^T*z-z^T*z`; the value is its fold mean minus
  one sample standard error. Negative values are preserved and invalid designs
  become `NaN`, never zero.

## Claim boundary

The classical P2 edge basis, ridge regression, cross-validation, and greedy
selection are not novelty claims. The proposed research claim is the coupled
object: exact renderer-affine, zero-init, topology-shared radiance enrichment
whose candidate capacity is valued prospectively on held-out camera blocks.
E0 alone supports only implementation correctness, not novelty or quality.
