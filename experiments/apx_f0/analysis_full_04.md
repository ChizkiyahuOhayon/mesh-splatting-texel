# APX-F0 result: FAIL — the gain exists, is concentrated, and cannot be found

Date: 2026-08-06
Protocol: `experiments/apx_f0/protocol.md`, corrected at `1a24bb4`
Records: `apx_f0_{garden,room}_04` (source `65efa8e`)
Checkpoints: `sac_g1_{garden,room}_stock_seed0_train`, iteration 30000
Supersedes `analysis_full_01.md`, which recorded the invalid first run.

## Measured

| | Garden | Room |
|---|---:|---:|
| eligible faces | 1,548,415 / 6,952,816 (**22.27%**) | 1,222,157 / 5,628,158 (**21.72%**) |
| ceiling order 1 | −0.2255 (reaches 21.52%) | −0.7553 (19.91%) |
| ceiling order 2 | −0.2494 (5.74%) | −0.7827 (6.67%) |
| ceiling order 4 | −0.2852 (0.60%) | −0.8098 (1.08%) |
| ceiling adaptive | −0.3342 | −0.9186 |
| **ceiling ORACLE** | **+0.4231** | **+0.3116** |
| **selected top 1%** | **−0.0795** | **−0.2960** |
| selected top 5% | −0.1402 | −0.4954 |
| selected top 10% | −0.1823 | −0.6122 |
| concentration top10% | 0.8626 | 0.9031 |
| `residual_mass` lift | 5.900 | 5.376 |
| best non-residual control | 2.684 | 2.664 |

Locked reading: **ceiling FAIL, both scenes. The gate fails.**

## What the oracle and the selector together establish

The three locked conditions returned `FAIL / PASS / PASS`, and taken at face value that
reads as "the capacity is not there but if it were we could find it." **That reading is
wrong, and the two ceilings added at `65efa8e` are what expose it.**

- The **oracle** — apply the correction only where it helped on held-out pixels — is
  `+0.42` / `+0.31 dB`. So a per-face static correction *does* carry gain, barely above
  the `0.30 dB` bar, and only with information no deployable scheme has.
- The **deployable selector** — `residual_mass`, the signal that scored `lift 5.9`
  against a best control of `2.68` — is **negative at every fraction on both scenes**,
  and monotonically worse the more faces it picks.

The gap is `0.5`-`0.9 dB`, and the sign flips across it.

> **The gain exists, is concentrated, and cannot be found by the signal that appeared
> to find it.** For a deployable scheme, unfindable and absent are the same thing.

### Why `predictability: PASS` was misleading

`concentration` and `predictability` are computed against `gain = clamp_min(0)`, so
they only ever saw the positive tail. `residual_mass` genuinely does rank faces by
positive gain — that lift of `5.9` is real. It also ranks faces by *loss*, and the
losses are larger. Clamping hid exactly the term that decides the question.

**This is the methodological result of the gate.** A ranking statistic computed on a
one-sided quantity can pass convincingly while the two-sided total goes the other way.
Any future gate that scores a signal against a benefit must also score it against the
harm, in the same units, unclamped. Had the oracle/selected pair not been added, this
project would have taken two PASSes into a multi-day training gate that could not have
succeeded.

## Why the naive fit is negative in the first place

`ceiling` at every order is negative and gets worse with capacity
(`−0.2255 → −0.2494 → −0.2852` on Garden, `−0.7553 → −0.7827 → −0.8098` on Room). That
is a one-shot least-squares fit to the training residual failing to transfer: finer
grids mean fewer pixels per cell, noisier fitted means, and more noise injected into
held-out predictions.

**A negative `ceiling` is a fact about that estimator, not about the capacity** — the
true ceiling is `>= 0` by construction because a zero correction is always available.
The locked condition therefore did not test what the protocol asked. It is recorded as
FAIL with the flaw stated rather than reinterpreted after the fact, and the oracle
figure is what actually bounds the question.

## Cross-check against the known texel result

Uniform texels measured Garden `+0.3 dB` (~20 seed SD) and Room `−0.2261 dB`. This gate
puts Garden ahead of Room on every quantity — `−0.23` vs `−0.76` naive, `+0.42` vs
`+0.31` oracle. **The measurement ranks the two scenes the way the trained result
does**, which is the evidence that it is measuring the right thing, and it is why the
negative is trustworthy rather than another model-class error.

The offset — this gate never reproduces Garden's `+0.3` — is expected and is the
design's honest limit: real texels are optimised *jointly* with SH and geometry against
the training objective, so the model can co-adapt and take only the part that
generalises. A post-hoc one-shot fit takes everything, including the view-dependent
part that SH left behind precisely because it is view-dependent.

## Surviving facts

- **~22% of faces carry 4 or more training pixels**; only `0.60%` (Garden) and `1.08%`
  (Room) carry the 64 needed for an order-4 grid. Most of the mesh is sub-pixel, and
  any per-face scheme is bounded by this before anything else.
- `residual_mass` beats every non-residual control by more than 2x on positive-gain
  ranking (`5.90` vs `2.68`, `5.38` vs `2.66`). The signal is informative about where
  appearance error concentrates. It is not informative about where correcting it helps.

## Status

Thirteenth preregistered falsification. **The appearance-capacity axis closes**, joining
topology, sampling budget, and density control. All four closed with a mechanism rather
than with an ambiguous null.
