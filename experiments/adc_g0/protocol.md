# ADC-G0: can the allocator concentrate budget, and does it help?

Status: **PREREGISTERED — no ADC-G0 output has been observed**

Date: 2026-08-05

## Question

MeshSplatting's `add_new_gs` (`scene/triangle_model.py:991`) is a stripped copy of
3DGS-MCMC's. It kept the sampling skeleton and discarded the machinery:

| | 3DGS-MCMC | MeshSplatting |
|---|---|---|
| `probs` | `get_opacity` | `importance_score` (max blending) |
| sampling | `replacement=True`, multiplicity returned as `ratio` | `replacement=False`, and `_update_params_fast` opens with `torch.unique` |
| dead-primitive relocation | yes | absent |
| SGLD noise | yes | absent |
| RNG | ambient | `torch.manual_seed(1)` on every call |

The second row is the subject of this gate. With `replacement=False` followed by
`torch.unique`, a face selected for densification is subdivided exactly once no
matter how it scored:

> **The importance score's magnitude is discarded. Only membership in the selected
> set survives.** A face a hundred times more important than another receives
> exactly the same treatment.

The allocator is therefore structurally unable to concentrate budget, whatever it is
given to rank with.

## Why this is the right thing to test, and the criterion is not

XVR-G0 preregistered and retired residual-guided densification on both scenes: raw
mean-error mass beat projected coverage by only `4.85%` (Garden) and `5.53%` (Room)
against a locked requirement of `10%`, and `max_blending` was itself one of the
controls. That stop rule is honoured here — **ADC-G0 changes no score.** Both arms
rank faces by exactly the `importance_score` the published pipeline uses.

That negative and this hypothesis fit together rather than conflicting. 3DGS-MCMC
achieves its gains using `get_opacity` — a visibility-family score of the same kind
as `max_blending`. Its advantage comes from the machinery, not the criterion. And a
score's magnitude cannot matter to an allocator that discards magnitude, which is a
candidate explanation for why XVR-G0 found score choice nearly irrelevant.

## Arms

Garden, seed 0, three full trainings. Each arm differs from the one above it by
exactly one change, so the effect of each is separately attributable — the design
weakness recorded against HARD-G0 is not repeated.

| Arm | Change | Isolates |
|---|---|---|
| `stock` | none (published pipeline) | reproduction anchor |
| `rng` | `_sample_alives` draws from a call-local `torch.Generator` instead of reseeding the global stream with `torch.manual_seed(1)` | the RNG defect alone |
| `multiplicity` | `rng`, plus sampling **with** replacement and subdividing a face drawn `k` times to depth `min(k, MAX_DEPTH)` | budget concentration |

The reported effect of multiplicity is `multiplicity - rng`, never
`multiplicity - stock`.

### Why the RNG arm exists at all

`torch.manual_seed(1)` inside `_sample_alives` reseeds the **global** torch generator
on every densification call. Two consequences, neither intended by a comment reading
`always same "random" indices`: successive densification rounds draw correlated index
sets, and every other consumer of torch randomness in the process has its stream
reset underneath it. Fixing this is a prerequisite for the multiplicity arm — with
replacement sampling, a correlated draw would concentrate budget on the same faces
round after round. Bundling the two would leave the result uninterpretable, so it
gets its own arm.

## Budget matching

**Primitive count is the confound this project has already been bitten by.** SAC-G0's
apparent win was entirely a pruning artefact once counts were pinned (SAC-G1). The
multiplicity arm therefore spends exactly the budget the baseline would have spent,
distributed differently.

Deepening one face from subdivision depth `d` to `d+1` replaces its `4^d` leaves with
`4^(d+1)`, a net gain of `3 * 4^d` faces. The baseline subdivides `n` distinct faces
once each, so its net cost is `3n`, and that integer is the multiplicity arm's budget.

Allocation draws a single sequence of `n` samples with replacement, then takes the
**longest prefix whose cost does not exceed the budget**. Prefix cost is monotone in
prefix length, so the boundary is found by bisection and is a deterministic function
of the draw. No tuning parameter is introduced.

`MAX_DEPTH = 2`. One face may absorb up to sixteen leaves — enough for concentration
to exist at all, shallow enough that a single densification round cannot produce a
mesh unlike anything the published pipeline creates. Deeper is a later arm, not a
knob to turn if this fails.

### T-junctions

Subdividing a selected face inserts edge midpoints that its unselected neighbours do
not receive, so **the published pipeline already creates T-junctions on every
densification round**, and `run_restricted_delaunay` at `densify_until_iter + 1000`
re-meshes afterwards. Depth-2 subdivision creates more of the same kind of
non-conformity, not a new kind. This is recorded as a known property rather than
claimed to be harmless: the arm's triangle count and post-Delaunay face count are
both reported, and a Delaunay stage that behaves differently between arms would show
up there.

## Locked decision

Evaluation is `sac_eval.py` at `scaling 2` and `scaling 4`, with
`--cleanup_scaling 4` on every arm so the final pruning criterion is the
schedule-independent one SAC-G1 established.

**Validity** — required before any effect is read:

1. `stock` reproduces the recorded `garden_baseline` (`24.7372 / 0.7484 / 0.2480`) to
   within `0.10 dB`, `0.010 SSIM`, `0.010 LPIPS`;
2. final triangle counts of `rng` and `multiplicity` are within `2%` of `stock`'s —
   the budget matching worked, and this is a measurement of the implementation, not
   an assumption about it.

**Screen** — proceed to replication only if:

3. `multiplicity@4` exceeds `rng@4` by at least `0.15 dB`;
4. `multiplicity@4` LPIPS is not worse than `rng@4`.

`0.15 dB` is about `9.5x` Garden's measured held-out PSNR seed standard deviation of
`0.0157 dB` (SAC-G1). One scene and one seed screen; they do not decide.

**Stop** otherwise, and report `rng - stock` regardless of the outcome — the RNG
defect is worth knowing about either way.

If the screen passes, ADC-G1 replicates on Garden and Room across three shared seeds
with paired differences required to exceed twice their own standard error, exactly as
SAC-G1 was run.

## Recorded

Per arm: both scaling cells, triangle and vertex counts, final `sigma`, training wall
clock, the per-1000-iteration test PSNR trace, and — from the allocator itself — the
realised depth histogram per densification round, the budget requested against the
budget spent, and the fraction of faces receiving depth `>= 2`. That histogram is the
manipulation check: if it shows almost no depth-2 faces, the arm did not actually
concentrate anything and a null result says nothing about the hypothesis.
