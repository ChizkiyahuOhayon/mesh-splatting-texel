# EdgeVal-E0: exact connected edge-field representation gate

Status: **COMPLETED — PASS on attempt 02**

Date: 2026-08-08

The sealed result is `analysis_02.md`. Attempt 01 is an excluded pre-build
infrastructure failure recorded in `analysis_01.md`.

## Scope

E0 tests only the load-bearing representation and algebra needed by EdgeVal. It
does not select Garden edges, train a quality arm, or support a paper claim.
Those steps are forbidden until this gate passes.

The frozen-renderer premise is concrete in this repository: base SH color is
evaluated and clamped before triangle interpolation, then the interpolated RGB
is multiplied by the fixed front-to-back `alpha*T` weight. E0 inserts one
three-channel edge residual after base-color evaluation and before composition:

```
psi_01 = 4*wA*wB
psi_12 = 4*wB*wC
psi_20 = 4*wC*wA
```

Both incident faces address the same canonical undirected edge row. With
geometry, opacity, visibility, ordering, and base appearance frozen, the final
pixels are therefore exactly affine in the edge rows. The RGB squared loss is
an exact finite-data quadratic, not a first-order renderer approximation.

## Fixed estimator conventions

These are frozen before any scene score is observed:

- cameras are sorted by immutable `image_name` and assigned as
  `sorted_rank % 4`;
- an edge needs at least three contributing pixels in every fold;
- complementary-fold ridge candidates are
  `10**s * trace(H)/3`, for integer `s` from `-6` through `2`;
- generalized cross-validation is `n*RSS/(n-df)**2`, computed only on the
  complementary folds; exact ties choose the larger ridge;
- `V_e` is the mean signed fold gain minus one sample standard error
  (`beta_se=1`); negative fold gains and negative values are never clamped;
- a zero/degenerate or nonfinite design is invalid and emits `NaN`, never a
  score of zero.

The earlier phrase "run OMP after computing V_e" is not considered an
implementation specification: ordinary OMP can ignore `V_e`, making its
permutation control inert. If E0 and the later independent-value gate pass,
the redundancy stage will use cross-fitted conditional forward selection: at
selected set `S`, jointly ridge-fit `S` and `S+e` on each complementary camera
set, evaluate their exact loss difference on its held-out fold, subtract one
standard error, and choose the largest conditional value with ascending edge
ID ties. This preserves cross-fitting and makes score permutation operative.

## E0 checks

The run must satisfy all of the following:

1. the existing extension rebuilds from the recorded source revision;
2. all CPU mathematical-core tests pass;
3. enabling a zero edge row yields a bitwise-identical rendered tensor;
4. a nonzero edge row changes the rendered RGB;
5. its CUDA backward gradient is finite;
6. its gradient agrees with a central finite difference to relative error at
   most `5e-3` on the fixed synthetic triangle fixture.

The finite-difference bound is an implementation tolerance for float32 CUDA,
not an outcome threshold or a paper metric.

## Stop rule

E0 passes only if all six checks hold. Any build failure, invalid input
acceptance, zero-init mismatch, nonfinite derivative, or derivative mismatch
stops the EdgeVal implementation before real-scene statistic extraction.

## Recorded artifacts

The immutable output directory contains the extension build log, complete unit
test log, smoke output and log, manifest, source revision, GPU identity, PyTorch
and CUDA versions, extension binary path and SHA-256, and SHA-256 hashes of all
other gate artifacts. The runner refuses to overwrite an existing suffix.
