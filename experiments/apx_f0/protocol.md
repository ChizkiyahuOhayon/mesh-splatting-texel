# APX-F0: is there appearance capacity left, is it concentrated, and can it be found?

Status: **PREREGISTERED — no APX-F0 output has been observed**

Date: 2026-08-06

## Why appearance, after twelve falsifications

Sorting the twelve closed gates by what they changed produces an asymmetry nobody in
this project has acted on:

| changed | gates | repeatable positives |
|---|---:|---:|
| geometry, allocation, sampling, schedule, criterion | 11 | 0 |
| **appearance representation** (texels) | 1 | **Garden `+0.3 dB`, ~20 seed SD** |

The single appearance intervention produced the only repeatable positive the project
has ever measured. It was abandoned because the 9-scene mean came out at `-0.103 dB`
— but that run gave **every face the same texel order**, so it is evidence about
uniform allocation, not about appearance capacity.

XVR-G0's own retirement note names the hypothesis being tested here:

> A large residual is not necessarily reducible by subdivision. It may come from view
> dependence, visibility, calibration, or **appearance limitations**.

XVR-G0 measured residual against *subdivision* benefit and found it weak. Residual
against *appearance-capacity* benefit has never been measured. They are different
questions, and that sentence points at this one.

### Why this survives the two mechanisms that killed everything else

- **Appearance absorption** killed every geometry-side signal (E9c: regime present at
  `AUC 0.813`, actionable signal at `0.500`). The signal here is the residual that
  *remains after* the appearance model has absorbed what it can, so it cannot be
  absorbed by the mechanism that defeated the others.
- **A well-tuned local optimum** made single-component perturbations neutral to
  negative (SAC-G1 at `0.0104 dB`, ADC-G0 at `-0.141 dB`). Adding appearance capacity
  is not a perturbation around the optimum; it enlarges the model class.

## This gate trains nothing

Signal hypotheses have died five times in this project, so no GPU-hours are spent
before the signal is measured. Everything below is computed from one existing
checkpoint.

## Method

One trained SH-only checkpoint per scene. Training views fit; **held-out views
score**. Fitting and scoring on the same pixels would measure memorisation.

1. Render each view. Assign every covered pixel to its dominant face using
   `rend_ids` and `rend_alpha`, exactly as XVR-G0 did, at the alpha threshold XVR-G0
   locked.
2. Compute each covered pixel's barycentric coordinates from its face's projected
   vertices (`image_2D`). MeshSplatting interpolates appearance in **screen space**,
   so screen-space barycentrics are the model's own parameterisation, not an
   approximation of it.
3. Bin each pixel into one of `k * k` cells of a barycentric partition of its face.
   This is a `k * k`-cell partition of the face in its own coordinates; the deployed
   texel carrier may lay cells out differently, so the quantity measured is the
   capacity of that resolution, and the result is read as an upper bound.
4. For each face and cell, the optimal colour is the mean target colour of the
   **training** pixels in it. One colour per cell, shared across all views — which is
   what a texel carrier is, and which correctly charges view-dependent variation as
   irreducible rather than crediting it as removable.
5. Score that fixed assignment on **held-out** pixels and compare its squared error
   against the current model's rendered prediction on the same pixels.

`k` is reported at `1, 2, 4`. `k = 1` is one colour per face; the current model is
affine in barycentric coordinates (three vertex colours) and therefore sits between
`k = 1` and `k = 2`.

### Eligibility

A face needs at least `MIN_PIXELS_PER_CELL * k * k` training pixels and at least one
held-out pixel. **The eligible fraction is a headline number, not a footnote.** Garden
carries 6.9M triangles over roughly 1M pixels, so the average face is sub-pixel and
cannot support a texel grid at all. If eligibility is small, adaptive appearance can
only ever reach a small part of the model, and that bounds the whole direction
regardless of how well the signal works.

## Locked constants

| Name | Value | Why |
|---|---|---|
| `TRAIN_VIEWS` | 16, evenly spaced, name-sorted | XVR-G0's setting, so the two gates are directly comparable |
| `ALPHA_THRESHOLD` | XVR-G0's locked value | same attribution rule, same threshold |
| `CELL_ORDERS` | `(1, 2, 4)` | |
| `MIN_PIXELS_PER_CELL` | 4 | a cell fitted from fewer than four pixels is memorising |
| `CEILING_DB` | `0.30` | the uniform-texel run measured `+0.3 dB` on Garden; a ceiling below what uniform already achieved leaves nothing for adaptive allocation to win |
| `CONCENTRATION_FRACTION` | `0.10` | |
| `CONCENTRATION_CAPTURE` | `0.50` | |
| `LIFT_MIN` | `1.75` | XVR-G0's locked lift |
| `CONTROL_MARGIN` | `0.10` | XVR-G0's locked margin over the best non-residual control |

## Locked decision

Three questions, all on held-out pixels. **A scene passes only if all three hold, and
the gate passes only if both Garden and Room pass.**

1. **Ceiling.** Best order-4 fit improves held-out covered-pixel PSNR over the current
   model by at least `CEILING_DB`. *If this fails the direction is closed: there is no
   appearance capacity left to allocate, however cleverly.*
2. **Concentration.** The top `CONCENTRATION_FRACTION` of eligible faces by realised
   held-out gain hold at least `CONCENTRATION_CAPTURE` of the total gain. *If gain is
   spread evenly, adaptive allocation cannot beat uniform allocation — and uniform
   allocation already failed the 9-scene mean at `-0.103 dB`. This is the condition
   the whole thesis rests on.*
3. **Predictability.** The cheap signal — the current model's per-face residual mass
   **measured on training views**, because that is all a deployable allocator can see
   — captures the realised **held-out** gain in its top 10% with lift at least
   `LIFT_MIN` over random, and at least `CONTROL_MARGIN` relatively above the best
   non-residual control (`max_blending`, projected coverage, world area). *If this
   fails, the gain exists and is concentrated but cannot be found cheaply, and any
   allocator would need the expensive answer it is trying to predict.*

Thresholds 3 reuses XVR-G0's numbers verbatim so that a pass here against a fail there
is a statement about the two questions differing, not about two different bars.

**Stop otherwise.** A failure closes appearance capacity as an axis, and with density
control and topology already closed, closes mechanism improvement on this baseline.
No re-run with a different `k`, alpha threshold, view count, or eligibility rule.

## Recorded

Per scene: eligible face fraction and absolute count, held-out PSNR of the current
model and of each order on covered pixels, the realised per-face gain distribution,
top-1/5/10% capture for every signal and control, the concentration curve, and the
per-face table needed to reproduce every number without re-rendering.
