# ADC-G0 result: FAIL, in the wrong direction — and it closes the branch

Date: 2026-08-06
Protocol: `experiments/adc_g0/protocol.md` (preregistered 2026-08-05, commit `963163b`)
Records: `adc_g0_garden_{stock,rng,multiplicity}_{train,eval}_01`
Training: 5,845 / 6,276 / 6,239 s, one A40 each, Garden, seed 0, `--cleanup_scaling 4`

## Manipulation check: the intervention worked

| | value |
|---|---|
| densification rounds | 16 |
| budget | 20,750,751 net faces |
| spent | 20,754,357 (100.0%) |
| depth-1 faces | 3,526,629 |
| **depth-2 faces** | **678,298 (16.13% of selected)** |

Budget concentration genuinely happened. This is not a null result from an
intervention that failed to intervene — the 0.017% overshoot is the published
`topk(areas)` channel, which is present in both arms and therefore not a disparity.

## Measured

| arm | triangles | @4 PSNR | SSIM | LPIPS | @2 PSNR |
|---|---:|---:|---:|---:|---:|
| `stock` | 6,887,507 | 24.7401 | 0.7483 | 0.2484 | 24.4437 |
| `rng` | 6,912,133 | 24.7157 | 0.7477 | 0.2492 | 24.4246 |
| `multiplicity` | 6,681,744 | **24.5747** | 0.7411 | 0.2564 | 24.2955 |

All arms finished at `sigma = 1e-4`, so deployed hardness is identical.

## Reading against the locked rule

1. **Validity 1 — PASS.** `stock@4` reproduces the recorded `garden_baseline`
   (`24.7372 / 0.7484 / 0.2480`) to `0.0029 dB / 0.0001 / 0.0004`. The platform is
   validated for the fourth consecutive gate.
2. **Validity 2 — FAIL.** `multiplicity` is `-2.99%` on triangle count against a
   locked `2%` band. `rng` is `+0.36%` and passes.
3. **Screen — FAIL, and negative.** `multiplicity@4 - rng@4 = -0.1410 dB` against a
   required `+0.15 dB`, with LPIPS worse (`0.2564` vs `0.2492`).

`rng - stock = -0.0244 dB`, about `-1.6` Garden seed standard deviations (`0.0157`),
with triangle count `+0.36%`. **The `torch.manual_seed(1)` global-reseed defect has no
measurable effect on quality.** It remains a real defect — it resets the ambient torch
stream for every other consumer in the process — but it is not a lever.

## The arithmetic that matters

`multiplicity` loses `0.1654 dB` to `stock`, roughly `10` seed standard deviations, so
this is not noise. It carries `2.99%` fewer triangles. At the `+0.5 dB per doubling`
capacity law that deficit is worth `-0.022 dB`.

> **The primitive deficit explains 13% of the loss. The remaining `0.14 dB` is caused
> by concentration itself.**

The validity-2 failure is therefore not an escape hatch. Repairing the budget match —
for instance by capping depth so children stay above `size_probs_zero` — would recover
at most an eighth of what was lost. The protocol anticipated this temptation and
refused it in advance: *"deeper is a later arm, not a knob to turn if this fails."*

## What this closes, and why it is more than a twelfth miss

XVR-G0 and ADC-G0 are the two halves of density control, and they now explain each
other:

- XVR-G0 established that `max_blending` is a poor error ranker — raw error mass beat
  projected coverage by only `4.85% / 5.53%` against a locked `10%` requirement, and
  `max_blending` was one of its controls.
- ADC-G0 establishes that giving that ranker's *magnitude* any authority costs
  `0.14 dB`.

> **`replacement=False` is not a bug in MeshSplatting. It is accidental
> regularisation.** Discarding score magnitude means trusting a weak ranker only for
> membership, never for degree. Restoring magnitude amplifies the ranker's errors.

**The criterion and the allocator are not independent.** An allocator's capacity to
concentrate is worth having only when the score deserves to be trusted with degree,
and the score has already been retired. Fixing one without the other is not
available, and the other is closed.

A secondary mechanism is visible in the counts: depth-2 faces carry `1/16` the parent
area and fall below the pruning floor, so the arm spent 100% of its budget and still
finished with 3% fewer surviving primitives. Concentration manufactured primitives
that could not survive. This is consistent with the sub-pixel culling behaviour the
RITS work already documented.

## Status

Twelfth preregistered falsification. Density control is closed on both axes, cleanly
and with a mechanism, which makes it one of the few results in this project that
explains its own negative rather than merely reporting one.

Untested and still score-independent: **SGLD noise**, the third MCMC component. It
perturbs primitives by opacity and consults no ranker, so the failure mechanism
established here does not apply to it. It is the only remaining density-control arm
whose premise survives ADC-G0.

`adc_forensics.py` (ADC-F0) remains unrun; it is three minutes and measures a
training-time bias independent of both criterion and allocator.

## Recorded defect in this gate's own code

`install_rng_fix` returns its call-counter state while `install_multiplicity` returns
a list of rounds, so `adc_rounds.json` for the `rng` arm holds `{"calls": 16}` where a
round list belongs. Harmless to the result, cosmetic in the record, and fixed
separately rather than silently.
