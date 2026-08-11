# GoRFE-V0 attempt 01 — PASS

Date: 2026-08-11

Source revision: `126121dfdeaaebbec6dd978d7f90101425b815ea`

Artifact directory: `${NAS_ROOT}/experiments/gorfe_v0_01`

Environment: Linux 5.15, Python 3.11.15, PyTorch `2.7.1+cu126`, CUDA build
`12.6`. V0 executed on CPU and reserved no GPU.

## Decision

**PASS.** All 27 locked Boolean checks are true. All 212 repository tests
passed in 3.637 seconds. The synthetic gate completed in 0.1185 seconds, both
the result and immutable manifest report `pass`, and `DONE` plus the SHA-256
ledger exist.

## Exact-statistics evidence

The locked tolerance was `atol=rtol=1e-11`.

| group | Gram | RHS | support RSS | held-out gain | order/chunk |
|---|---:|---:|---:|---:|---:|
| P2-DC (`Q=1`) | `5.55e-17` | `2.78e-17` | `2.22e-16` | `5.55e-17` | `2.22e-16` |
| P2-SH1 (`Q=3`) | `1.39e-17` | `1.39e-17` | `2.22e-16` | `4.72e-16` | `2.22e-16` |

Support-pixel counts match the dense oracle exactly. Both group types report
per-fold counts `[4, 4, 6]` across the three canonical groups. Reverse and
seed-locked input permutations, with chunk sizes `1`, `2`, `7`, and all rows,
remain within tolerance. Outputs are finite float64 tensors with the declared
shapes.

The fixture supplied 72 fragment rows across eight cameras and reduced them to
56 unique `(camera, pixel, group)` rows. Maximum camera input/reduced rows were
9/7. The explicit peak temporary tensor-memory estimates were 1,320 bytes for
DC and 2,504 bytes for SH1, far below the locked 4 MiB ceiling.

## Duplicate cross-term manipulation check

Both incident faces contributed to the same canonical edge and pixel at depths
`1.25` and `2.50`. The analytic cross-term norms were `0.06604416` for DC and
`0.03117244` for SH1, and the duplicate-reduced Gram contained both terms.

The deliberately incorrect per-fragment Gram missed `0.06604416` for DC and
`0.02081673` for SH1, both far above the locked `1e-8` failure threshold. This
demonstrates that the gate distinguishes the required duplicate-safe reduction
from the known incorrect implementation.

Multiple held-out fold gains are negative and remain negative in the result;
the implementation does not hide harmful effects by clamping them to zero.

## Artifact integrity

| artifact | SHA-256 |
|---|---|
| `python_env.txt` | `fc77bee1fe40761b54cff209ceb5ca9561d362f795687d7c437a99cddcad2d92` |
| `tests.log` | `1a0704fdd45c77b7191dc3f88a96c5aa4fd477b7a26ee14925a99b57247510f0` |
| `result.json` | `dc08f923572b32a3ccdd995f0817ff277054e2ac9415b66afe9a489fe98021c5` |
| `gate.log` | `dc08f923572b32a3ccdd995f0817ff277054e2ac9415b66afe9a489fe98021c5` |
| `manifest.json` | `afdeee7b8ff1f358db51e4492032eebe2d121484a9984c68c903a6894789d6f6` |
| `DONE` | `37a40f08d8548dba289b9b0bb35bcf63b359f6d37ee86044ebc6b6da080b9ec1` |

The manifest binds the passing result to the exact source revision, protocol,
fixture identity, locked environment, and hashes above.

## Claim boundary and authorization

V0 supports the exact duplicate-safe construction of float64 fold Gram, RHS,
support RSS/counts, and signed held-out gain for the synthetic P2-DC/P2-SH1
design. It does not establish Garden/Room transfer, physical image-quality
gain, scene-scale memory, final speed, or superiority over MeshSplatting.

This pass authorizes freezing the complete V1 executable protocol. V1 must
commit its candidate set, immutable camera folds, support thresholds, ridge/GCV
conventions, controls, cost model, rank/top-budget predicate, and resource
ceilings before any Garden or Room score is read. Physical training remains
blocked until V1 passes on both scenes.
