# Project reassessment and next experiment

Date: 2026-08-20.  Evidence basis: `sota/PLAN.md`, `sota/results/`,
`sota/future_modification.md`, and the August GoRFE/D0 records.

## Bottom line

The project is not dead, but it has not yet produced a transferable improvement.
The campaign closed a useful set of local tweaks: hardening schedules, persistent
vertex learning rates, per-face hardening, exact screen-space position gradients,
longer training, alternative densification signals, topology-preserving splits,
and several appearance-capacity changes.  The recurring failure is transfer:
Garden has enough primitive slack to make several arms look good, while Room
removes the apparent gain.  A Garden-only result is therefore not evidence.

The strongest remaining low-complexity direction changes the optimized image,
not the representation or the hardening schedule:

> Render the current mesh with the final opaque coverage in the forward pass,
> but differentiate that RGB loss through the published soft rasterizer.

This is high-risk and not novel merely because it uses a straight-through
estimator.  It becomes paper-shaped only if the full connected-mesh construction
improves all three metrics across scenes with the identical deployed checkpoint
and renderer.

## Brainstorming ledger

The required divergence pass produced these twelve candidates before filtering:

1. Endpoint-hard RGB value with the scheduled soft Jacobian.
2. Apply endpoint supervision only after the restricted-Delaunay topology freezes.
3. Route endpoint supervision to appearance only and keep geometry on the soft loss.
4. Alternate soft-valued and endpoint-valued cameras instead of rendering both.
5. Emit soft and endpoint images in one CUDA traversal.
6. Reweight SH degree against per-face texels at matched checkpoint bytes.
7. Filter projected appearance footprints instead of increasing supersampling.
8. Make the primitive budget depend on scene error per covered pixel.
9. Permit a bounded second opaque depth layer only at thin structures.
10. Replay a short endpoint-only correction stage after ordinary training.
11. Initialize the connected mesh from a denser geometric prior.
12. Learn a residual selector or teacher that decides where capacity is needed.

The simplicity, relevance, transfer, and feasibility filters retain four:

| Rank | Candidate | Why it remains live | Why it is not first/why first |
|---:|---|---|---|
| 1 | Endpoint-hard value / soft Jacobian | Directly attacks the train/deploy mismatch; zero deployment cost; never trained in this repo | First experiment |
| 2 | SH-for-texel exchange | Already implemented and cheap; potentially storage-neutral | Existing Room result says texels alone do not transfer; requires a sweep |
| 3 | Footprint filtering | Targets the weak perceptual metrics and aliasing | More CUDA work and smaller expected PSNR gain |
| 4 | Content-adaptive budget | Directly addresses Garden/Room primitive imbalance | Harder to attribute and likely a systems contribution rather than the core quality idea |

Candidates 9 and 11 change the deliverable or introduce a strong external prior;
candidate 12 violates the requested no-teacher simplicity.  Candidates 2--4 and
10 are fallback variants only after the unmodified endpoint mechanism is tested;
they are not tuning knobs for rescuing a failed premise.

## Selected method: endpoint-forward supervision

For camera `c` and current parameters `theta`, let `S_c(theta)` be the ordinary
scheduled MeshSplatting render.  Let `H_c(theta)` be the same mesh and shader at
the endpoint coverage controls (`sigma=1e-4`, opacity floor `0.9999`, final 4x
supersampling).  The loss image is

```text
E_c(theta) = S_c(theta) + stopgrad(H_c(theta) - S_c(theta)).
```

`E` has the RGB value of `H` and the Jacobian of `S`.  Only the RGB photometric
loss uses `E`; depth, normals, densification, pruning, and statistics retain the
ordinary soft render package.  Evaluation, checkpoint format, primitive count,
and inference are unchanged.

The first implementation intentionally has one boolean flag and no mixture
weight, schedule, extra loss, or learned module.  Rendering the endpoint pass
before the differentiable pass prevents both forward-buffer sets from being live
at the same time.

## Falsification rule

Run one full Room training first with the published settings and compare its
final standard-render PSNR, SSIM, and LPIPS against the recorded matched Room
baseline.  The idea survives only if PSNR and SSIM improve and LPIPS decreases;
primitive count and checkpoint bytes must not increase beyond ordinary run-to-run
effects.  A clear joint regression kills the direction.  A survivor is then run
unchanged on Garden and Bicycle; no seed campaign is needed for this pilot.

The strongest mechanistic objection is biased geometry gradients at hard
visibility boundaries.  A positive final metric result answers the practical
question; if needed for the paper, the hard/soft RGB gap can be logged later as a
mechanism diagnostic.  It is not a precondition for trying the method.

## Reviewer 1

- **Overall assessment:** The project history is unusually informative, but the
  present contribution is still a hypothesis rather than a result.
- **Who would be interested:** Neural rendering, differentiable rasterization,
  explicit radiance fields, and graphics deployment researchers would care if a
  training-only change closes quality without changing the asset.
- **Major strength:** The selected intervention is simple, deployment-neutral,
  and aimed at a measured train/deploy mismatch rather than another schedule.
- **Concern ID R1-M1 -- originality.** Claim pointer: selected method above.
  Evidence pointer: prior-art discussion in `sota/future_modification.md` and the
  IdeaSpark literature cache.  Straight-through gradients, hard opacity, and
  hard-forward surrogate gradients already exist.  **Resolution test:** show
  that prior work did not construct full endpoint coverage/visibility supervision
  for a connected shared-vertex mesh, and demonstrate a transferable result that
  this distinction enables.
- **Recommendation posture:** Promising experiment, not yet an oral-level paper.

## Reviewer 2

- **Overall assessment:** The code path is coherent, but the central estimator is
  biased and may send high-valence shared vertices in harmful directions.
- **Who would be interested:** Optimization and differentiable rendering readers
  would care about separating a renderer's forward value from its backward map.
- **Major strength:** The method preserves the one gradient field known to train
  this representation and changes only the residual that contracts with it.
- **Concern ID R2-M1 -- technical soundness.** Claim pointer: equation for `E`.
  Evidence pointer: exact position-gradient failures in `sota/PLAN.md`.  There is
  no guarantee that the soft Jacobian is useful for the hard residual.
  **Resolution test:** complete Room training without instability and improve all
  three final hard-render metrics; a short toy gradient check is insufficient.
- **Recommendation posture:** Worth the full run because failure is cheap and
  decisive, but do not add rescue components before observing it.

## Reviewer 3

- **Overall assessment:** Empirical transfer, not implementation novelty, is the
  decisive issue.
- **Who would be interested:** Practitioners choosing explicit meshes over splats
  would value a quality gain with identical storage and runtime.
- **Major strength:** The experiment can use existing training and evaluation
  machinery and has a clean matched baseline.
- **Concern ID R3-M1 -- generality.** Claim pointer: project-level goal of
  surpassing MeshSplatting.  Evidence pointer: Garden/Room deltas summarized in
  `sota/future_modification.md`.  Several previous Garden gains disappeared on
  Room.  **Resolution test:** Room first, then unchanged Garden and Bicycle;
  report PSNR, SSIM, LPIPS, primitives, bytes, train time, and render speed.
- **Recommendation posture:** A Room-only pilot is sufficient for go/no-go; a
  paper claim requires the multi-scene result.

## Cross-review synthesis

All three reviews agree that this is the best remaining simple experiment and
that it is not yet evidence of an advance.  The consensus strength is zero
deployment overhead and unusually clean causal attribution.  The consensus risks
are prior-art collision at the generic STE level, biased gradients under hard
visibility, and Garden-specific false positives.  The single most important next
action is therefore the full Room run, not another protocol, proxy, or ablation.

## Execution record

The first two Room launches at revision `e411ef5` were infrastructure failures,
not endpoint-forward results.  The first used the base interpreter and stopped
before training because `rdel` was absent.  The second selected the intended
Torch 2.7.1/CUDA 12.6 environment, loaded the data, and then stopped at iteration
zero because its installed `diff_triangle_rasterization` Python wrapper predated
the checkout's `screen_space_gradients` setting.  No metric or scientific
decision was produced by either attempt.

The launcher now selects the project interpreter itself and binds the installed
rasterizer to a hash of the checkout's Python, C++, CUDA, and GLM sources.  A
missing or mismatched hash triggers a local rebuild before training.  The next
Room launch remains the first scientific endpoint-forward attempt.

The next launch exposed one further pre-training environment mismatch: the
correct cu126 Python still discovered the host CUDA 11.8 compiler because an
absolute Python path does not activate its micromamba prefix.  Wheel construction
stopped before compiling any source.  The launcher now derives CUDA_HOME from
that Python's prefix, puts its `nvcc` first, and verifies the compiler release
equals `torch.version.cuda` before checking or rebuilding either native module.

The first frozen depth-opacity ceiling launch at revision `ff58a58` was also an
infrastructure failure, not a scientific result.  Its launcher invoked
`sota/depth_opacity.py` as a file, so Python used `sota/` rather than the
repository root as its import path and stopped before loading the checkpoint,
data, or CUDA renderer.  The launcher now uses the package entrypoint
`python -m sota.depth_opacity`; the registered experiment and output suffix stay
unchanged because no output directory or metric was produced.
