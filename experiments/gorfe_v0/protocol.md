# GoRFE-V0 — exact sparse-design integrity gate

Status: **frozen before implementation**.  This gate is synthetic and must not
read Garden, Room, or any other scene pixels.

## Object

Validate the sufficient statistics used to value one affine GoRFE group.  The
two separately evaluated group dimensions are:

- `P2-DC`: `Q=1` scalar spatial feature with three RGB coefficients;
- `P2-SH1`: `Q=3` angular features with a `Q x 3` RGB coefficient block.

For one camera pixel `p`, canonical group `g`, and all triangle fragments that
refer to that group, define the duplicate-reduced feature

`f_pg = sum_j f_pgj`.

For fold `k`, the required float64 statistics are

`H_gk = sum_p f_pg f_pg^T`,

`b_gk = sum_p f_pg r_p^T`,

`s_gk = sum_p ||r_p||_2^2`, and

`n_gk = number of supported camera pixels`.

The RSS and count include each supported `(camera, pixel, group)` once, after
duplicate reduction.  An exactly zero reduced feature is not support.  Negative
held-out gain is retained without clamping.

## Locked synthetic fixture

- seed `20260811`, eight immutable camera names, four sorted-name round-robin
  folds, three canonical groups, six residual pixels per camera;
- both `Q=1` and `Q=3` use the same camera, pixel, group, face, and depth rows;
- one canonical shared edge receives contributions from both incident faces at
  the same camera pixel, with distinct positive depths `1.25` and `2.50`;
- contributions are deliberately split across input chunks so that correct
  reduction cannot rely on fragment adjacency;
- residuals and features are nontrivial finite float64 values and every group
  has support in every fold.

The independent dense oracle first sums all fragment rows into an explicit
`pixel x group x Q` design, then constructs the RGB block design in Python.
The production path processes one camera at a time and may retain sparse rows
for that camera, but must never materialize a dense pixel-by-group matrix.

## Locked fit and signed-gain convention

For held-out fold `k`, fit on the other three folds:

`H_-k = sum_{j != k} H_gj`, `b_-k = sum_{j != k} b_gj`,

`lambda = 1e-3 * trace(H_-k) / Q`, and

`B_gk = (H_-k + lambda I)^-1 b_-k`.

The exact signed squared-loss gain on fold `k` is

`Delta_gk = 2 <B_gk, b_gk> - sum_c B_gk[:,c]^T H_gk B_gk[:,c]`.

Zero or nonfinite ridge scale is invalid and is a gate failure.  No GCV,
ranking, or candidate selection is part of V0.

## Locked checks

All arithmetic below uses float64.  Tensor agreement uses
`atol=1e-11, rtol=1e-11`; integer counts must be exact.

For both `Q=1` and `Q=3`:

1. sparse and dense fold Gram matrices agree;
2. sparse and dense fold right-hand sides agree;
3. sparse and dense support RSS values agree;
4. sparse and dense support-pixel counts agree exactly;
5. sparse and dense held-out signed gains agree;
6. reverse order and the seed-locked permutation agree with the reference for
   chunk sizes `1`, `2`, `7`, and all rows;
7. the shared-edge duplicate has a nonzero analytic cross term, the correct
   sparse result contains it, and a deliberately incorrect per-fragment Gram
   differs by more than `1e-8`;
8. every output is finite and has the declared shape and dtype.

The production-path result must additionally report input rows, reduced rows,
maximum rows buffered for one camera, an explicit temporary-memory estimate,
and wall time.  For this locked fixture, estimated temporary memory must be at
most 4 MiB and gate wall time at most 10 seconds.  These synthetic ceilings are
integrity checks, not scene-scale performance claims.

Every Boolean must be true.  Failure blocks V1.  A pass authorizes freezing the
V1 executable protocol; it does not establish transfer or image-quality gain.

## Artifact contract

The runner requires the locked `mesh_splatting` environment (`torch
2.7.1+cu126`, CUDA build `12.6`), but V0 executes on CPU and does not reserve a
GPU.  It refuses an existing output directory or dirty tracked checkout and
writes full unit-test output, gate result, manifest, Python environment,
`DONE`, and `SHA256SUMS` to a write-once NAS attempt directory.  The manifest
binds every artifact to the source revision and protocol path.
