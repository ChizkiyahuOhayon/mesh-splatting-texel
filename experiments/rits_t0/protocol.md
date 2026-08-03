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
3. midpoint parameters carry gradients of the **same fidelity as the
   checkpoint's original parameters**. With the G0-lite loss reduced in
   **float64** (renders stay float32), probe the 4 largest-|gradient| `f_dc`
   scalars of the original block and of the midpoint block at steps 0.002 and
   0.001, and take each block's median central-difference / analytic ratio.
   The check passes when every probe's two rungs agree within 5% (the
   measurement is converged), both medians are positive (sign agreement), and
   the midpoint median is within 25% relative of the original median.

If any check fails, no arm trains and the gate stops as an implementation
failure, not a scientific result.

**Amendment history (both before any training output was observed).**
The original item 3 prescribed per-scalar differences at step 1e-3 with the
float32 training loss; that is unresolvable, because one midpoint SH-DC
gradient is ~1e-6, so the loss change (~1e-9) sits below the ~4e-9 float32
spacing around a loss of ~0.03, and the difference rounds to exactly zero
(smoke 01: relative error identically 1). The first replacement probed along
the block's unit gradient direction, but the measured block norm (8.46e-6)
forced the element-capped step to 0.597, where the objective's curvature
(~1e-4) buries the ~1e-5 first-order signal (smoke 02: fd 7.42e-5 vs
analytic 8.46e-6). Both failures share one root cause: float32 mean-reduction
resolution forcing steps outside the linear regime. Reducing the G0-lite loss
in float64 removes that constraint, so small per-scalar steps become
resolvable (signal ~1e-9 against ~1e-12 reduction accuracy); SH color is
linear in the DC coefficient, so truncation at step 0.002 is benign, and the
two-rung convergence requirement guards against residual nonlinearity. The
donor image is independent of midpoint DC and is cached across evaluations.
The 5% tolerance and all training-side settings and decision rules are
unchanged; a genuine gradient defect still fails this check.

**Amendment 3, 2026-08-04, before any training output was observed.** With a
converged measurement in place, smoke 03 showed the analytic midpoint SH-DC
gradient to be ~8.87x smaller than the finite difference. The diagnostic in
`results/g0_diag_garden_01.md` traced this to the baseline: the unmodified
production path on an unmodified checkpoint (no split, no donors, no blend)
already shows ~8.44x, and midpoint vertices (8.65x) match original vertices
in the same split model (8.28x). The discrepancy is a property of
MeshSplatting's rasterizer backward, not of refinement; Adam's per-parameter
normalization absorbs a near-uniform scale factor, which is why the baseline
trains normally. An absolute finite-difference criterion therefore tests the
rasterizer rather than the refinement operator. Item 3 above is restated as a
relative criterion against original parameters measured in the same run, view,
and loss — the property the precondition actually needs to establish, and one
a broken split path would still fail. All three arms share the rasterizer, so
the discrepancy cancels in the arm comparison; the decision rule and every
training-side setting remain unchanged.

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
