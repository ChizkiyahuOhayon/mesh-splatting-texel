# GoRFE-D0-A — sealed-state failure-mechanism audit

Status: **EXPLORATORY DIAGNOSTIC; NO PASS/FAIL AND NO TRAINING AUTHORIZATION**

## Question

GoRFE-V1 attempt `_04` is a sealed valid failure.  D0-A asks which part of the
already observed additive portfolio failed:

1. did the primary ordering keep adding groups after its own nested score became
   non-positive, or after cumulative outer-fold utility had peaked;
2. were primary groups more topologically concentrated than the random-ID
   control; and
3. how much did the primary sets overlap the RHS-norm and same-view controls
   that defeated them in V1?

D0-A is descriptive.  Its budgets, summaries, and any apparent mechanism are
post-outcome analyses and cannot rescue V1 or support a paper claim by
themselves.

## Immutable inputs

Only the sealed GoRFE-V1 `_04` preparation and evaluation artifacts identified
in `input_identity.json` may be read.  The audit loads:

- `candidate_state.pt` for the frozen candidate identity and eligibility masks;
- `evaluation_state.pt` for the frozen per-group/per-fold sufficient statistics;
- the sealed V1 scene results for an exact portfolio cross-check.

No dataset path, image, camera, checkpoint, renderer, native extension, CUDA
device, residual replay, or official test view is an input.  The runner hides
CUDA and requires the exact V1 root hashes and completion sentinels before the
audit.

## Descriptive calculations

The V1 nested scores, controls, costs, canonical ties, and signed outer outcomes
are reconstructed without modification.  For each scene, family, outer fold,
and V1 selector, D0-A records additive portfolio values at cost units

`256, 512, 1024, 2048, 4096, 8192, 12288, 16384`.

At each budget it records spent cost, selected count, signed outcome sum,
whole-fold-SSE-normalized value, V1's `budget/spent` normalized value, fractions
of positive/negative outer outcomes, and fractions of positive/non-positive
selector scores.  It also records the best cumulative prefix no larger than
16,384 units, the first cumulative zero crossing after a positive prefix, and
the prefix ending before the selector score first becomes non-positive.

For each selected set, topology is summarized only from canonical endpoints:
the fraction of group pairs sharing at least one vertex and the number of
DC/SH1 group pairs on the identical edge.  Primary/control Jaccards are also
reported.  These are topology and selection proxies, not image-support overlap.

The two original V1 budgets are recomputed and must agree with the sealed
`result.json` portfolios to absolute error at most `1e-12`; otherwise the audit
is invalid.

## Interpretation boundary and next decision

- A cumulative peak before 16,384 or a non-positive-score tail motivates an
  independently specified confidence-stopping hypothesis.
- Excess topological concentration relative to random motivates, but does not
  prove, a redundancy hypothesis.
- Neither observation establishes harmful interaction.  Exact joint utility
  requires cross-group design terms absent from the sealed V1 state.

D0-B residual replay or a conditional selector may be designed only after D0-A
is reviewed.  It requires a separate protocol, must remain train-camera-only,
and cannot retroactively change V1.  S0/T0 and official-test access remain
unauthorized.
