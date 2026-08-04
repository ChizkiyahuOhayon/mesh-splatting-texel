# SAC-G1: does the splat2 advantage survive without its confound?

Status: **PREREGISTERED — no SAC-G1 output has been observed**

Date: 2026-08-05

## Question

SAC-G0 failed its locked rule, and the reallocation thesis it tested is closed.
It also produced an observation the gate was not designed to test: the arm
trained with a `scaling 2` final phase beat the stock arm on all three metrics
when both were rendered at `scaling 4`, while carrying 18% fewer triangles.

That observation is not evidence. It rests on one scene and one seed, and the
arms differ in more than their sampling rate. SAC-G1 removes the confound,
measures the seed variance the pipeline has never had measured, and asks one
question:

> With the primitive-count confound removed, does training the final phase at
> `scaling 2` produce a better model than the published schedule, by a margin
> larger than seed noise, on more than one scene?

This is a replication of an observation, not a new mechanism. If it fails, this
axis is exhausted and the project's goal is reconsidered rather than an
eleventh variant attempted.

## The confound, and its exact removal

After the training loop, `train.py` renders every training view once more,
accumulates each face's maximum blending weight, and deletes faces scoring at
or below 0.5. That render uses whatever `triangles.scaling` the run ended with.
`max_blending` is a per-pixel maximum, so at `scaling 4` each face is sampled
four times as often and has four times as many chances to exceed the threshold.
The stock arm therefore keeps faces the `splat2` arm deletes, and the entire
6.95M versus 5.70M difference is attributable to this single step.

SAC-G1 evaluates that final pruning at `scaling 4` in **both** arms, via a new
`cleanup_scaling` parameter whose default of 0 means "use the current factor"
and therefore leaves the published pipeline unchanged. Pruning by what a face
contributes at the highest fidelity is also the more defensible rule: it is a
deployment decision, not a training-schedule artefact.

## Arms, scenes, seeds

`safe_state` currently hard-codes seed 0, so the pipeline has never produced
seed variance. A `--seed` argument is added and threaded through.

- Scenes: Garden and Room.
- Arms: `stock` (`--final_scaling 4`, the published pipeline) and `splat2`
  (`--final_scaling 2 --cleanup_scaling 4`).
- Seeds: 0, 1, 2, shared between arms so every comparison is paired.

Twelve trainings in total, roughly 1.5-1.7 hours each; about ten hours of wall
clock across two GPUs. Each trained model is evaluated at `scaling 2` and
`scaling 4` exactly as in SAC-G0.

## Locked decision

Let `d(scene, seed) = PSNR(splat2@4) - PSNR(stock@4)`; the arms share a seed, so
these are paired differences.

1. **Validity.** Per scene, the mean of the three `stock@4` PSNRs is within
   `0.15 dB` of the recorded baseline under this evaluation path: `24.7372` for
   Garden and `28.5142` for Room. A wider tolerance than SAC-G0's `0.10 dB` is
   used because seeds now vary; it is fixed here, before any run.
2. **Effect.** The mean of `d` is positive on **both** scenes, and across all
   six pairs `mean(d) - 2 * SE(d) > 0`, where `SE` is the standard error of the
   paired differences. The effect must exceed twice its own uncertainty.
3. **No perceptual regression.** The mean LPIPS difference
   `LPIPS(splat2@4) - LPIPS(stock@4)` is at most zero on both scenes.

All three must hold. A pass authorises the nine-scene Mip-NeRF360 run and a
quality claim against MeshSplatting; nothing less. A failure closes this axis.

## Reported, not decisive

Final primitive counts per run — with the confound removed these should be
close, and any residual difference is itself informative. Seed standard
deviation per scene and arm, which this project has never measured and which
retrospectively bounds how much of every earlier single-seed result was noise.
The deployment cells `splat2@2` versus `stock@4`: quality difference and
measured render time, which is the Pareto statement independent of the decision.
Training wall clock.

## What a pass would and would not mean

A pass means a smaller or equal model, trained with a cheaper final phase,
matches or beats the published pipeline on two scenes beyond seed noise. It
would be the first time in this project that anything beat MeshSplatting on a
full held-out split. It would still be a schedule finding, not a mechanism, and
the nine-scene table would decide whether it is a paper.
