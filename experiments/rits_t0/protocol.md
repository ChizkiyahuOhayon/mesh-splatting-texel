# RITS-T0: fixed-budget refinement fine-tuning gate

Status: **PREREGISTERED — no RITS-T0 output has been observed**

Date: 2026-08-03

## Question

RITS-D1 established that the completed prolongation makes midpoint subdivision
an exact identity of the rendered function. T0 asks the decisive causal
question: does introducing new topology **continuously** (exact prolongation
followed by an annealed transition to child-local rendering) let optimization
use the new degrees of freedom better than the abrupt inherited split, at an
identical parameter budget — and does either beat not splitting at all?

This is the branch's survival experiment. Its exit clause is final.

## Arms

All arms start from the final SH-only checkpoint (iteration 30000), use the
same loss, learning rates, view sequence, seed, and 5,000 fine-tuning
iterations, with no densification, pruning, retriangulation, or SH-degree
changes:

1. **unsplit** — fine-tune the checkpoint as loaded.
2. **abrupt** — split the selected faces with inherited midpoint
   initialization (the production operator, D0's variant 1), then fine-tune.
3. **rits** — identical split and initialization, but the training image is
   the homotopy `F_gamma = gamma * F_child + (1 - gamma) * F_donor.detach()`,
   where `F_donor` is the exact-prolongation render (donor mode 7) of the
   current parameters under `no_grad`, and `gamma` ramps linearly from 0 to 1
   over the first 1,000 iterations, after which the donor pass is dropped and
   the arm is identical in cost and semantics to **abrupt**.

The split arms share bitwise-identical topology and initial values; the only
difference is the transition schedule. That isolates continuity from capacity.

## Locked settings

- Face selection: accumulate `triangle_was_rendered` pixel counts over the
  **full training split** on the unsplit checkpoint, then take the top 10% of
  faces. A neutral visibility screen, not a method; identical across arms.
- Optimizer: fresh Adam (eps 1e-15) in every arm with the restore-path
  learning rates — f_dc 0.0016, f_rest 0.00008, vertices 0.0001,
  vertex_weight 0.0 (opacity frozen) — constant over fine-tuning.
- Loss: photometric only, `0.8 * L1 + 0.2 * (1 - SSIM)`. Geometry
  regularizers are deferred to the full-pipeline integration stage; identical
  absence in all arms keeps the comparison internally valid.
- Rendering: `scaling = 4` (the training/eval supersampling), sigma and SH
  degree exactly as loaded, dataset background, no random background.
- View sequence: train.py's pop-random-without-replacement epochs with
  `random.seed(1234)` and `torch.manual_seed(1234)`; one view per iteration;
  the sequence is identical across arms.
- Evaluation: full held-out split with the standard donor-free renderer,
  PSNR, SSIM, and LPIPS-VGG per `svsr_g1_eval._metrics`; the loaded
  checkpoint (0-iteration) is also evaluated once for context.

## G0-lite precondition (rits arm, before its first update)

On the first name-sorted training view at `gamma = 0.5`:

1. gradients of all four parameter tensors are finite;
2. the midpoint-row geometry and appearance gradient group norms are nonzero;
3. a central finite difference **along the unit direction of the midpoint
   `f_dc` gradient block** agrees with the analytic directional derivative
   (the block's gradient norm) within 5% relative error. The step targets a
   loss change of 1e-4 and is capped at 0.05 per element; if the cap makes
   the expected change smaller than 1e-6 the check fails as unresolvable.

If any check fails, no arm trains and the gate stops as an implementation
failure, not a scientific result.

**Amendment 2026-08-04, before any training output was observed.** The
original item 3 prescribed per-scalar central differences (8 scalars, step
1e-3). That procedure is unimplementable in float32: an individual midpoint
SH-DC gradient is ~1e-6, so a 1e-3 step changes the full-image loss by ~1e-9,
below the ~4e-9 spacing of float32 around a loss of ~0.03; the difference
rounds to exactly zero and the relative error is identically 1. The smoke run
failed at this precondition exactly as designed and produced no training or
metric output. The directional probe above aggregates the block's signal so
the expected loss change (~2e-4) is orders of magnitude above float32
resolution, while still validating gradient direction and magnitude. The 5%
tolerance and all training-side settings and decision rules are unchanged.

## Locked decision

Compute per-scene test-set means. RITS-T0 **passes** only if all hold:

1. **vs unsplit**: RITS mean PSNR across Garden and Room is at least
   +0.15 dB above unsplit; neither scene is below unsplit by more than
   0.05 dB; RITS LPIPS improves on both scenes.
2. **vs abrupt (causal)**: RITS PSNR exceeds abrupt on both scenes with a
   two-scene mean margin of at least +0.05 dB, and RITS LPIPS is not worse
   than abrupt on either scene.

If T0 passes, the next step is the nine-scene Mip-NeRF360 integration with the
unchanged paper targets (>= +0.30 dB mean, >= 5% LPIPS reduction, gains on
>= 7/9 scenes, no scene below -0.10 dB, matched budget). If T0 fails, the
topology branch closes — final, no T1 — and the project pivots to the
soft-compositor efficiency thesis.

## Smoke

Garden only: top 1% of faces, 200 iterations, anneal over 40, no decision.
It validates execution, the G0-lite machinery, memory, and timing only.

## Known limitations recorded now

- Non-conforming split boundaries (T-vertices) do not affect the soft
  renderer, whose faces are windowed independently; a conforming closure is
  future work for the exported-asset story.
- The homotopy detaches the donor pass, so early updates follow
  gamma-scaled child-local gradients; this soft-start is part of the claimed
  mechanism, not a bug.
- Fine-tuning from a converged checkpoint is a screen, not the final method;
  a T0 pass authorizes integration into the full training schedule, where
  splits occur during optimization.
