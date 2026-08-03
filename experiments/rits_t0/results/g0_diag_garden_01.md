---
exp_id: MS-A40-260804-001
date: 2026-08-04
system: MeshSplatting
experiment: RITS G0-lite gradient diagnostic
classification: diagnostic
scene: garden
view: DSC07957
gpu: NVIDIA A40
source_revision: 547a7428ab9ed375bc69737ee1180d2a903e03bc
confirmatory: false
anomaly: true
tags: [gradient, backward, baseline-property, rasterizer]
---

# The SH-DC gradient scale discrepancy is a baseline rasterizer property

## Result

Median central-difference / analytic ratio on the largest-|gradient| `f_dc`
scalars, one ingredient removed per configuration:

| Configuration | Median ratio |
|---|---:|
| A unsplit model, original vertex, plain render, L1+SSIM, 4x SSAA | 8.44 |
| B as A without the SSIM term | 8.77 |
| C as A at scaling 1 | 9.11 |
| D split model, original vertex | 8.28 |
| E split model, midpoint vertex | 8.65 |
| F as E with the 0.5 donor blend (the G0-lite configuration) | 8.83 |

Every finite difference converged: coarse (0.002) and fine (0.001) steps agree
to better than 0.1% in all 24 probed scalars, so the measurement is sound and
the discrepancy is structural.

## Interpretation

Configuration A is the unmodified production path on an unmodified checkpoint,
and it already shows the full ~8.5x discrepancy. The effect is therefore a
property of MeshSplatting's rasterizer backward, present before any RITS code
runs. It is not caused by midpoint subdivision, window donors, the homotopy
blend, the SSIM term, or supersampling.

Midpoint vertices (E, 8.65) and original vertices in the same split model
(D, 8.28) are statistically indistinguishable, and both match the untouched
baseline (A, 8.44). **The split does not degrade gradient fidelity**, which is
exactly the property the G0-lite precondition exists to establish.

The scatter across probes is about +/-5% within a configuration and about 10%
across configurations, so the discrepancy is close to, but not exactly, a
constant factor. Adam normalizes each parameter by its own gradient magnitude,
which absorbs a near-uniform scale factor; this explains why the baseline
trains normally despite it.

## Consequence for RITS-T0

The original G0-lite item 3 compared midpoint gradients to an absolute finite
difference, which measures the rasterizer rather than the refinement operator.
It is replaced by a relative check: midpoint gradient fidelity must match the
fidelity of original parameters measured in the same run, view, and loss. A
broken split path would move the midpoint ratio away from the baseline ratio;
this diagnostic shows it does not.

All three T0 arms share the same rasterizer, so the discrepancy cancels in the
arm comparison and does not affect the T0 decision rule.

## Not pursued now

Locating the exact source inside the rasterizer backward is deferred. It is not
part of the RITS contribution, does not block T0, and the deadline is close.
It is recorded as a candidate follow-up: a genuine gradient-scale defect in a
published rasterizer would be worth reporting upstream, and worth re-examining
before any claim that depends on absolute gradient magnitudes (for example the
CSU-F0 observation that appearance gradients exceed vertex gradients by ~8,845x).
