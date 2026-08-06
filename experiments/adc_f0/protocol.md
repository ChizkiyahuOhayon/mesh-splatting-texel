# ADC-F0: is the rasterizer's SH-DC gradient under-report the same for every primitive?

Status: **PREREGISTERED — no ADC-F0 output has been observed**

Date: 2026-08-05

## Question

`experiments/rits_t0/results/g0_diag_garden_01.md` established that MeshSplatting's
rasterizer backward reports SH-DC gradients about `8.44x` smaller than a converged
central difference, on the untouched production path. The project recorded this and
moved on, reasoning that Adam's per-parameter scale invariance absorbs a uniform
rescaling of the gradient.

That reasoning is only valid if the under-report is uniform, and nobody measured
whether it is. `8.44` is not `16`, not `4`, and not any other clean constant that a
missing normalisation would produce, which is weak evidence against uniformity.

> Does the ratio between the finite-difference and analytic SH-DC gradient depend on
> a primitive's projected size, its depth, or how much of the image it wins?

## Why this blocks the density-control programme

The next gate (ADC-G0) replaces MeshSplatting's densification criterion — currently
`max_blending`, a pure visibility score with no error term (`train.py:210-212`,
`scene/triangle_model.py:1001-1006`) — with one that responds to reconstruction
error. Two families are available:

- an **error** criterion accumulated in the forward/backward traversal from the
  photometric residual, which never passes through the analytic gradient;
- a **gradient** criterion in the style of AbsGS, which is built entirely on the
  analytic gradient.

If the under-report is heterogeneous, the gradient family reads a signal with a
per-primitive multiplicative bias, and any criterion built on it silently inherits
that bias — a criterion that thresholds gradient magnitude would be systematically
wrong in whichever regime the bias is strongest. ADC-F0 decides which family is safe
before either is implemented.

## This is a measurement, not a pass/fail gate

Both outcomes advance the plan; they select a route rather than authorise or block
one. Nothing here can "fail" in the sense the previous eleven gates could.

| Outcome | Reading | Consequence |
|---|---|---|
| **HOMOGENEOUS** | a global constant; Adam absorbs it | gradient criteria are safe up to a rescale; ADC-G0 still uses the error criterion first, because it needs no rasterizer change, and keeps the AbsGS-style gradient criterion as a later arm |
| **HETEROGENEOUS** | a per-primitive bias in the published backward | gradient criteria are unsafe until the backward is fixed; the error criterion becomes the only sound route, and the bias itself is a reportable defect of the baseline whose repair is a candidate quality gain |

## Method

One trained Garden checkpoint, evaluated at `scaling 4`, the deployed configuration.

1. Render one training view and read the per-primitive covariates the renderer
   already returns: projected image size and `max_blending` (both face-indexed, via
   the misleadingly named `vertex_index` slice) and vertex depth.
2. Reduce the two face covariates to vertices by taking the maximum over incident
   faces — the same reduction `train.py` uses when it accumulates them.
3. Keep vertices with at least one rendered incident face.
4. Sort by projected size, cut into `STRATA` equal quantile bins, and take
   `PROBES_PER_STRATUM` evenly spaced members of each. Even spacing rather than
   random sampling makes the probe set a deterministic function of the checkpoint,
   so a rerun reproduces it exactly without carrying a seed.
5. One backward pass yields the analytic gradient for every probed vertex at once.
   For each vertex, probe the colour channel with the largest `|gradient|`, which
   keeps the finite-difference signal as far above the float floor as the gradient
   allows.
6. Central-difference that scalar at two step sizes.
7. Record `ratio = fd_fine / analytic` together with every covariate, and repeat the
   whole procedure on a second view.

### Probe loss

Mean squared error against the target, reduced in `float64`.

The rendered image is linear in a vertex's SH-DC coefficient, so a squared loss is
**exactly quadratic** in the probed scalar and its central difference is exact at any
step size. This was learned the expensive way: the project's `L1 + SSIM` training loss
has kinks wherever a pixel residual changes sign, and those kinks made two
finite-difference rungs disagree by 8% on Room (G0-lite failure #4). The `float64`
reduction addresses the other historical failure, where a `float32` mean lost the
loss change entirely below its own ulp.

### Validity guards

Each is checked and recorded, not assumed.

- **Determinism.** The same view is rendered twice and the two images must be
  bitwise identical. A non-deterministic forward pass would make every central
  difference noise, and it is cheaper to prove it once than to explain a strange
  ratio later.
- **Rung agreement.** Because the probe loss is exactly quadratic, the two rungs must
  agree to `RUNG_TOLERANCE`. They will not if the step crossed the CUDA SH clamp at
  `sh2rgb + 0.5 < 0`, which makes the loss piecewise quadratic; such a probe is
  discarded rather than repaired.
- **Survival.** At least `MIN_SURVIVAL_FRACTION` of probes must survive the rung
  check, otherwise the measurement is reported as inconclusive and no reading is
  taken.

## Locked constants

| Name | Value | Why |
|---|---|---|
| `STRATA` | 5 | quintiles; enough resolution to see a monotone trend, few enough that each bin holds a usable sample |
| `PROBES_PER_STRATUM` | 16 | 80 probes per view, ~4 renders each — about 30 s at Garden's 7M faces |
| `RUNGS` | `(0.002, 0.001)` | the project's established pair, reused so ratios stay comparable with `g0_diag_garden_01.md` |
| `RUNG_TOLERANCE` | `1e-3` | exact arithmetic would give 0; this is float headroom, not a fit tolerance |
| `MIN_SURVIVAL_FRACTION` | `0.8` | |
| `HOMOGENEITY_SPREAD` | `1.25` | `max/min` of the per-stratum median ratios. Matches `FD_FIDELITY_TOLERANCE = 0.25` already in use in `rits_t0_train.py`, so the project applies one notion of "the same ratio" throughout |
| `HOMOGENEITY_RHO` | `0.3` | Spearman `|rho|` of ratio against covariate; catches a monotone trend too gentle to move the quintile extremes |

## Reading

**HOMOGENEOUS** requires, for every covariate and on **both** views, a stratum-median
spread `<= HOMOGENEITY_SPREAD` **and** `|rho| <= HOMOGENEITY_RHO`. Anything else reads
**HETEROGENEOUS**, and the covariate and direction of the strongest trend are named
in the record — that is what a later repair would have to explain.

Two views is replication, not power. A disagreement between them is itself a result
and is reported as `VIEW_DEPENDENT` rather than averaged away.

## Recorded

Every probe: vertex index, channel, analytic gradient, both rungs, ratio, rung
disagreement, projected size, depth, `max_blending`, incident-face count. Plus the
per-stratum medians, the Spearman coefficients, the determinism check, the survival
fraction, the checkpoint hash, and the source revision.
