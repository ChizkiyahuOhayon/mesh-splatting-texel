# GoRFE-V1 — prospective real-scene value gate

Status: **PREREGISTERED — no Garden or Room target pixel or V1 score has been
read**.  This protocol and `protocol_constants.json` are committed before the
native exporter, scene evaluator, or decision code is implemented.

Protocol revision 1 is a pre-data executability clarification: after independent
review of the initial protocol commit and before any scene implementation or
target access, it fixes the byte serialization of camera identities, the
direction of an odd-length circular permutation, and the ordering of random
hashes.  It changes no scientific object, threshold, candidate, or outcome.

Protocol revision 2 is another pre-data executability clarification.  The SH1
rank threshold was already fixed below as `1e-10 * trace(H)/3`; revision 2 copies
the same value into `protocol_constants.json` so implementation code has one
machine-readable source.  It changes no value, threshold, candidate, or outcome.

V1 asks one narrow question: on a frozen connected MeshSplatting checkpoint,
does an exact renderer-affine value estimated without one camera fold predict
the independent squared-error reduction on that unseen fold better than simpler
residual, same-view, gradient, coverage, and random controls?  V1 is a screen for
prospective capacity allocation.  It is not a training result and cannot by
itself support a claim of better PSNR, SSIM, LPIPS, speed, or mesh quality.

## Sealed inputs and target-access boundary

The two inputs are the final, stock, seed-zero, SH-only SAC-G1 checkpoints for
Garden and Room at iteration 30,000.  Their paths must be supplied explicitly as
`GORFE_V1_GARDEN_MODEL` and `GORFE_V1_ROOM_MODEL`; directory guessing and latest-
iteration lookup are forbidden.  The preparation manifest binds each resolved
`point_cloud_state_dict.pt` SHA-256 and tensor shapes.  A checkpoint is invalid
unless `active_sh_degree=3`, `texel_order=0`, and its activated endpoint sigma is
`1e-4` within the tolerance in `protocol_constants.json`.  Rendering explicitly
sets `scaling=4`, which is not stored in a checkpoint.

Only the official training split is used.  Official test cameras are never
instantiated, decoded, rendered, or scored in V1; their name-list hashes are
checked only from COLMAP metadata and reserved for the later physical method
evaluation.  The locked train/test counts and name-list hashes are in
`protocol_constants.json`.

V1 has two separate invocations and artifacts:

1. `prepare` may read checkpoint tensors, mesh topology, COLMAP camera metadata,
   and renderer-derived design support.  It runs with an image-decoder sentinel
   that makes any target-file decode an error.  It writes a candidate manifest,
   candidate face-edge maps, target-free support statistics, and their hashes,
   then exits.
2. The preparation hashes must be transcribed to a tracked
   `candidate_freeze_<attempt>.json` commit.  A later `evaluate` invocation must
   be given that exact file and refuses any identity mismatch.  Only then may it
   decode official *training* images and form residuals.

There is no command that prepares candidates and evaluates them in one process.
A dirty tracked checkout, reused output directory, missing freeze commit, or
target access before the freeze makes the attempt invalid.

## Representation and exact sparse design

Canonical undirected edges are lexicographically sorted endpoint pairs `(u,v)`,
`u<v`.  Face-local columns are `(v0,v1)`, `(v1,v2)`, `(v2,v0)`.  Repeated-vertex
faces and edges incident to more than two faces are invalid.

For one high-resolution sample `h`, accepted fragment `a`, and local edge `e`,
the spatial basis is `psi_ae=4 lambda_i lambda_j`.  The native forward blending
weight is `alpha_ha T_ha`, where `T_ha` is transmittance before that fragment.
For output pixel `p`, the exporter must produce

```
x_pe^DC  = (1/16) sum_{h in p} sum_{a -> (h,e)} alpha_ha T_ha psi_ae
x_pe^SH1 = (1/16) sum_{h in p} sum_{a -> (h,e)} alpha_ha T_ha psi_ae Ytilde_ha
```

where `Ytilde` is the barycentric interpolation of the stock per-vertex real
degree-one factors `(-C1*y, C1*z, -C1*x)`.  DC has feature dimension `q=1` and
three RGB coefficients.  SH1 excludes DC, has `q=3`, and nine RGB coefficients.

The exporter replays the actual forward order and semantics: identical hard
inside test, depth-sorted point list, `alpha <= 0.999`, `alpha < 1/255` skip,
`T(1-alpha) < 1e-4` termination, screen-space barycentrics, and SH1 convention.
It must not use dominant-face maps, `was_rendered`, or the backward kernel.  The
backward kernel has a different alpha cap and is not an exact forward oracle.

All fragments from both incident faces, all depths, and all 16 high-resolution
samples are summed for the same final-resolution `(camera,pixel,edge)` *before*
an outer product is formed.  COO rows are candidate-filtered on CUDA, reduced a
complete camera at a time in float64, and immediately discarded.  A dense
`pixel x candidate` matrix and persisted raw COO are forbidden.

## Target-free candidate freeze

Let `E` be the number of sorted canonical edges and `i` its zero-based row.  The
candidate priority is the SplitMix64 permutation

```
z = (i + seed + 0x9E3779B97F4A7C15) mod 2^64
z = ((z xor (z >> 30)) * 0xBF58476D1CE4E5B9) mod 2^64
z = ((z xor (z >> 27)) * 0x94D049BB133111EB) mod 2^64
h = z xor (z >> 31)
```

with the unsigned seed in `protocol_constants.json`.  Select the smallest
`min(E, 131072)` values of `h`; SplitMix64 is bijective, and endpoint order is a
defensive tie break.  No rejected candidate is replaced after support is known.

The target-free exporter then determines support separately for DC and SH1.  A
duplicate-reduced feature that is exactly zero is not support.  A group is
eligible only if every one of the four folds has at least 32 supported output
pixels from at least four distinct cameras, all statistics are finite, and each
fold Gram has positive trace.  For SH1, each fold Gram must additionally have
numerical rank three under `lambda_min > 1e-10 * trace(H)/3`.  The eligible masks
are sealed before target decode and cannot later be filtered or refilled using a
residual, gain, loss, or outcome.

Cameras have unique immutable `image_name` values, sorted by UTF-8 bytes; fold is
`sorted_training_rank mod 4`.  Garden fold sizes must be `[41,40,40,40]` and Room
fold sizes `[68,68,68,68]`.  The name-to-fold map and its SHA-256 are sealed in
the preparation manifest.  A name-list hash serializes every sorted,
extension-stripped name as its UTF-8 bytes followed by `0x0a`, including the last
name; this is the encoding used by the four locked dataset hashes.  A fold-map
hash uses UTF-8 canonical JSON with sorted keys, compact separators, and
`ensure_ascii=False`.

## Sufficient statistics and ridge convention

For eligible group `g` and fold `k`, after duplicate reduction,

```
H_gk = sum_p x_pg x_pg^T
b_gk = sum_p x_pg r_p^T
s_gk = sum_p ||r_p||_2^2
n_gk = number of supported output pixels
```

where `r = target - frozen_parent_render`.  Reductions and all following
statistics use float64.  Signed gain is never clamped.

For any training-fold set `A`, sum its four statistics.  Ridge candidates are
`lambda_t = 10^t trace(H_A)/q`, `t=-6,...,2`, and
`B=(H_A+lambda I)^-1 b_A`.  Define

```
df = trace(H_A (H_A + lambda I)^-1)
RSS(lambda) = s_A - 2 <B,b_A> + sum_c B[:,c]^T H_A B[:,c]
GCV(lambda) = n_A RSS(lambda) / (n_A-df)^2.
```

Only finite candidates with positive lambda and `n_A>df` are accepted.  An RSS
below `-1e-10*max(1,s_A)` is invalid; a negative value within that roundoff band
is set to zero.  Exact GCV ties choose the larger lambda.  Any numerical failure
for a group sealed as eligible invalidates the scene; it is not silently dropped
or assigned zero.

## Nested selector and independent outcome

Using a four-fold value as both selector and outcome would be circular.  V1 uses
strict nested leave-one-fold-out evaluation.  For outer fold `k`, let
`T_k={0,1,2,3}\\{k}`.

For every inner fold `j in T_k`, fit by the GCV rule on the other two folds and
evaluate exact signed gain on `j`:

```
d_gj|-(k,j) = 2 <B,b_gj> - sum_c B[:,c]^T H_gj B[:,c].
```

The primary score, which cannot see outer fold `k`, is

```
v_g,-k = mean_j d_gj|-(k,j) - sample_sd_j(d_gj|-(k,j))/sqrt(3).
```

Separately fit on all of `T_k` and reveal the outer outcome only after every
score and selected set for `k` is fixed:

```
y_gk = 2 <B_g,-k,b_gk> - sum_c B_g,-k[:,c]^T H_gk B_g,-k[:,c].
```

Changing outer-fold residuals must be incapable of changing its primary score,
control scores, ridge choices, or selected sets.

## Controls, costs, and deterministic policies

All control scores for outer fold `k` use only `T_k`:

- `raw_residual`: `s_g,T_k`;
- `same_view_gain`: GCV fit and unpenalized gain evaluated on `T_k` itself;
- `rhs_norm`: Frobenius norm `||b_g,T_k||_F`;
- `coverage`: `n_g,T_k`;
- `random_id`: SHA-256 of the domain string
  `GoRFE-V1-random|scene|k|type|u|v`, with `k,u,v` in canonical base-ten ASCII
  and type exactly `DC` or `SH1`, interpreted as an unsigned big-endian integer
  priority; larger integers rank first;
- `permuted_value`: within each type and outer fold, sort by canonical key and
  move the value at sorted position `i` to
  `(i + floor(N_type/2)) mod N_type`.  This preserves the value multiset and
  type/cost distribution while breaking edge identity.

One cost unit is one RGB coefficient row: DC costs one and SH1 costs three.  It
corresponds to three trainable scalars (12 float32 coefficient bytes); topology,
index, alignment, and dispatch bytes are reported separately.  This is an active
parameter budget, not a claim that the current dense Q0 tensor is packed.

Families are `DC`, `SH1`, and their `MIXED` union.  A family is valid only if its
eligible total cost is at least 65,536 units.  Policies sort `score/cost`
descending.  Exact ties use type order `DC < SH1`, then `(u,v)`.  A budget scan
skips an item that does not fit and continues until no remaining item fits.
`random_id` uses its hash priority directly, without another cost division.

## Locked metrics and Boolean decision

The budgets are 4,096 and 16,384 cost units.  For selector/control `a`, outer
fold `k`, and budget `B`, let `S_a,k,B` be its selected groups, `spent` their
cost, and `E_k` the frozen-parent RGB SSE over the entire outer fold.  Define

```
rho_a,k = Spearman(score_a/cost, y_gk/cost)
P_a,k,B = (B/spent) * sum_{g in S_a,k,B} y_gk / E_k
CV4(z)  = mean_k z_k - sample_sd_k(z_k)/2.
```

Ranks use averaged ties.  For `random_id`, `rho` uses the hash priority directly
rather than dividing that priority by cost; its outcome axis remains `y/cost`.
`P` is an independent-candidate portfolio diagnostic;
overlapping groups are not jointly fitted, so it is explicitly not a predicted
joint PSNR change.

A scene/family passes the rank gate only if:

1. `mean_k rho_primary,k >= 0.05` and `CV4(rho_primary)>0`;
2. against every listed control, including random and permuted value,
   `CV4(rho_primary-rho_control)>0`.

At *each* budget, all of the following are required:

1. primary `P` is positive on at least three of four outer folds and
   `CV4(P_primary)>0`;
2. against every control, `CV4(P_primary-P_control)>0`;
3. `mean(P_primary) >= 1.10 * max(0, max_control mean(P_control))`.

The permutation intervention additionally requires:

1. at the small budget, every fold's Jaccard overlap between primary and
   permuted selected sets is at most `0.10`;
2. at both budgets,
   `|mean(P_perm-P_random)| <= 0.5 |mean(P_primary-P_random)|`;
3. the analogous rank inequality holds for mean rho.

A family passes a scene only if eligibility, rank, both budgets, and permutation
checks all pass.  V1 passes only if the intersection of passing families on
Garden and Room is nonempty; Garden and Room cannot pass with different carrier
families.  If several families survive, the sole family advanced to S0 maximizes
the minimum `CV4(P_primary)` over both scenes and budgets.  Exact ties prefer DC,
then SH1, then MIXED.

## Validity and resources

The run requires the locked Torch/CUDA build, one exclusive A40 with at least
40,000 MiB free at start, clean tracked source, native SH evaluation, black
background, no texels, no RITS donors, and the frozen parent checkpoint.  Each
scene runs serially and one camera at a time.

The per-scene limits in `protocol_constants.json` are six hours wall time, 38
GiB PyTorch allocated memory, 42,000 MiB physical GPU usage under one-second
polling, 64 GiB host peak RSS, fewer than `2^31` raw rows for any camera, and
512 MiB persistent artifacts excluding inputs.  The runner records allocated
and reserved CUDA peaks, physical peak, host RSS, per-camera raw/reduced row
counts and duplicate ratio, phase times, and disk bytes.  Exceeding a resource
or platform limit is `invalid`, never a scientific failure.

Attempt directories are write-once.  A completed phase writes its full logs,
environment, source and extension hashes, inputs, diagnostics, result/manifest,
`SHA256SUMS`, and finally `DONE`.  A failed or invalid phase writes `FAILED` or
`INVALID` and never receives `DONE`.

## Required implementation gates

Before real-scene evaluation is authorized, tests must establish:

1. default exporter-off rendering and Q0 behavior are bitwise unchanged;
2. replay transmittance and accepted-fragment count match the saved forward;
3. a synthetic single-triangle DC/SH1 design matches the analytic formula;
4. shared-edge, multi-depth, and 4x subpixel duplicates are reduced before Gram,
   and a deliberately per-fragment Gram fails;
5. sparse design times arbitrary coefficients matches the active Q0 carrier's
   area-downsampled correction, and squared-loss gradient equals `-2b`;
6. invalid shape, dtype, device, ID, scaling, count/write mismatch, and overflow
   are refused;
7. float64 statistics are input-order and chunk invariant;
8. GCV tie, negative-RSS tolerance, nested complement, cost/budget tie, signed
   negative outcome, and within-type permutation semantics are exact;
9. changing outer fold `k` residuals changes only its outcomes, never its scores
   or selected sets;
10. a target-access sentinel blocks image decode during preparation, and any
    official test-camera access fails;
11. checkpoint, camera, candidate, eligible-mask, extension, and artifact hash
    drift is detected.

Every implementation gate is mandatory.  A V1 pass authorizes S0 conditional
selection and a later physical training protocol; it does not authorize a paper
claim until the full unchanged benchmark confirms actual quality and efficiency.
