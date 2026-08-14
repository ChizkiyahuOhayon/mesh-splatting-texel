# GoRFE-D0-B — exact joint-utility replay

Status: **PREREGISTERED EXPLORATORY MECHANISM TEST; NO V1 RESCUE OR TRAINING AUTHORIZATION**

## Question and causal boundary

GoRFE-V1 `_04` is a sealed valid failure.  GoRFE-D0-A `_01` found that the
registered primary selections are much more endpoint-concentrated than the
random-ID control, while a non-positive selector-score tail does not explain
the SH1/MIXED failure.  D0-B asks one narrower question:

> Does independent per-group evaluation overstate the held-out utility obtained
> when the already selected groups are activated simultaneously?

D0-B cannot explain why V1's **additive** large-budget portfolio fell relative
to its small-budget portfolio: cross-group terms are absent from that additive
quantity by definition.  D0-B instead measures whether simultaneous deployment
would incur an additional interaction penalty, and whether that penalty is
larger for the concentrated primary selections than for the locked controls.

D0-B is post-outcome exploratory work.  It cannot change V1, support a
confirmatory paper claim, or authorize S0/T0 by itself.

## Immutable inputs

Only the artifacts and hashes in `input_identity.json` are admissible:

- sealed GoRFE-V1 Prepare `_04`, including its candidate states, frozen wheel,
  native extension, manifests, checksum ledger, and DONE sentinel;
- sealed GoRFE-V1 Evaluate `_04`, including both evaluation states, scene
  results, manifests, checksum ledger, and DONE sentinel;
- sealed GoRFE-D0-A `_01` result, manifest, checksum ledger, and DONE sentinel;
- the exact Garden and Room dataset roots and iteration-30,000 stock checkpoints
  already bound by the V1 candidate manifests.

Every V1 and D0-A ledger row is rehashed before output reservation.  Dataset
metadata, camera identity, target image hashes, checkpoint identity, candidate
state, installed native extension, and per-camera exporter diagnostics must
match V1.  Any drift makes the attempt invalid.

Only official training cameras are read.  Camera folds and ordering remain the
V1 UTF-8 order and `rank mod 4`.  An official-test identity must be rejected
before path lookup, hashing, or decode.  No optimizer step, coefficient refit,
new candidate, new score, or native carrier activation is allowed.

## Frozen decision objects

The constants in `protocol_constants.json` are immutable.  D0-B evaluates:

- scenes: Garden and Room;
- families: SH1 and MIXED; DC is excluded because it failed V1's preregistered
  minimum eligible-cost condition and is not an admissible shared family;
- outer folds: `k = 0,1,2,3`;
- budgets: 4,096 and 16,384 V1 cost units;
- selectors: primary nested value, random-ID, RHS norm, and same-view gain.

The selectors are the unchanged V1 scores, cost normalization, canonical ties,
and skip-if-cost-does-not-fit scan.  Random-ID remains the V1 exception that is
not divided by cost.  The two selected sets per selector/fold/family are frozen
and hashed before the first target path lookup.  D0-B never selects from joint
utility.

For every selected group, the coefficient is exactly the V1 outer coefficient
`B[g,-k]`: GCV fit on the three folds other than `k`.  No joint refit is
performed.  A DC and an SH1 group on the same canonical edge remain distinct
costed groups, while their rendered corrections add at a shared pixel.

## Exact replay quantity

All quantities are evaluated at the final low-resolution output pixels.  The
native exporter and duplicate-safe reducer must first sum every high-resolution
subpixel, depth fragment, and incident-face occurrence sharing the same
`(camera, low-resolution pixel, canonical edge)`.

For held-out fold `k`, let `r[p]` be the frozen parent residual
`target[p] - parent[p]`.  For selected group `g`, let

```
z[p,g] = f[p,g] B[g,-k] in R^3,
```

where `f[p,g]` is the duplicate-reduced DC or SH1 design row.  For selection
`S`, D0-B computes in float64

```
A(S) = sum_g in S { 2 sum_p <z[p,g], r[p]> - sum_p ||z[p,g]||^2 }
J(S) = 2 sum_p <sum_g in S z[p,g], r[p]>
       - sum_p ||sum_g in S z[p,g]||^2
I(S) = A(S) - J(S)
     = 2 sum_p sum_{g<h; g,h in S} <z[p,g], z[p,h]>.
```

`A` is the V1 independent-group additive gain, `J` is the exact joint gain for
simultaneous activation with the unchanged coefficients, and `I` is the signed
interaction penalty.  Positive `I` is destructive redundancy; negative `I` is
synergy.  No term is clamped.

Let `E[k]` be V1's whole-fold parent RGB SSE and `C(S)` the spent cost.  The
reported portfolio-scale values are

```
P_add   = (budget / C(S)) A(S) / E[k]
P_joint = (budget / C(S)) J(S) / E[k]
P_int   = P_add - P_joint.
```

Raw `A`, `J`, `I`, `E`, spent cost, selected count, linear term, diagonal
quadratic term, and joint quadratic term are also retained.

## Identity and numerical gates

Before interpretation, all of the following must pass:

1. state-only `P_add` reproduces the sealed V1 portfolio within absolute
   tolerance `1e-12`;
2. replayed per-group linear and diagonal terms reproduce the same raw `A`
   within `1e-9 * max(1, abs(A))`;
3. `A - J` equals the independently accumulated signed cross term within the
   same relative tolerance;
4. blank, one-group, disjoint-support, destructive-overlap, synergistic-overlap,
   DC+SH1-same-edge, duplicate-row, row-order, and camera-chunk synthetic
   oracles pass;
5. every scalar is finite and every nominally nonnegative squared norm is
   nonnegative up to the registered numerical tolerance.

An identity, arithmetic, resource, or official-split failure makes the attempt
**invalid** and produces no mechanism reading.

## Falsifiable mechanism reading

Controls are `random_id`, `rhs_norm`, and `same_view_gain`.  Define
`CV4(x) = mean(x) - sample_sd(x)/2` across the four outer folds.

For one scene/family, the interaction mechanism reads **supported** only if all
of the following hold:

1. at 16,384 units, primary `P_int > 0` in at least 3/4 folds and
   `CV4(P_int_primary) > 0`;
2. for every control, primary `P_int` exceeds control `P_int` in at least 3/4
   folds and `CV4(P_int_primary - P_int_control) > 0` at 16,384 units;
3. for every control, the primary excess growth from 4,096 to 16,384 has
   positive CV4:

   ```
   CV4((P_int_primary,16384 - P_int_primary,4096)
       - (P_int_control,16384 - P_int_control,4096)) > 0.
   ```

Overall D0-B reads **supported** only if at least one common family is supported
in both Garden and Room.  A family supported in one scene only is recorded as
localized evidence but is insufficient for a shared method.  If there is no
common supported family, the project-level interaction hypothesis is rejected.

A supported reading authorizes only preregistration of a separate conditional
or residualized selector experiment.  It does not establish that such a
selector beats V1 controls.  A rejected reading closes the interaction branch;
the next explanation must concern score calibration or cross-fold
generalization rather than pixel-space redundancy.

## Resource and artifact contract

- Torch `2.7.1+cu126`, CUDA build `12.6`, one exclusive NVIDIA A40, and at
  least 40,000 MiB free before any CUDA initialization;
- scenes run serially, one camera at a time; raw COO rows are never persisted;
- no dense pixel-by-candidate or group-by-group Gram matrix;
- at most 38 GiB allocated CUDA memory, 42,000 MiB physical GPU use, 64 GiB
  host RSS, 6 hours per scene, and 2 GiB persistent artifacts per scene;
- write-once result, manifest, checksum ledger, and DONE sentinel.  A scientific
  rejection is complete and receives DONE; invalid attempts do not.

The result must include selected-set hashes, all fold-level signed quantities,
per-camera replay diagnostics, resource measurements, exact input provenance,
and the complete Boolean reading.  D0-B results are recorded before any V2
protocol is written.
