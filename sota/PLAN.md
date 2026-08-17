# SOTA plan — beat MeshSplatting's Table 1

Started 2026-08-17. One RTX 4090 (24 GB), one scene at a time, roughly ten
30k-iteration runs per day.

## Target

MeshSplatting, Mip-NeRF360 nine-scene mean (their Table 1):

| PSNR | LPIPS | SSIM | \|V\| | train |
|---:|---:|---:|---:|---:|
| 24.78 | 0.310 | 0.728 | 3M | 48 min |

Protocol is `full_eval.py`: outdoor `-i images_4`, indoor `-i images_2 --indoor`.
The claim we want is all three metrics improved at a comparable vertex count.

## Why the previous approach stalled

Fifteen preregistered gates, but only about four end-to-end trainings. Almost all
the compute went into proxy screens — post-hoc fits, finite differences, rank
correlations on frozen checkpoints. Those screens are systematically pessimistic
about a jointly trained model, and this project measured that directly:
`experiments/apx_f0/analysis_full_04.md` records that its screen could not
reproduce the `+0.3 dB` that trained texels actually deliver on Garden, because
"real texels are optimised jointly with SH and geometry and take only what
generalises, while a post-hoc one-shot fit takes everything."

So the instrument used to reject ideas cannot detect the one effect known to be
real. **This plan trains instead of screening.** One change per arm, held-out
PSNR/SSIM/LPIPS as the only currency.

## Where the headroom is

Their own Table 1 puts soft, unconnected Triangle Splatting at **27.16** PSNR and
MeshSplatting at **24.78**. Opaque-only Triangle Splatting† collapses to 21.05,
and connectivity plus restricted Delaunay recovers it to 24.78. The 2.4 dB gap is
the price of the hard connected mesh, and the hardening transition is where it is
paid.

Their Table 4 also shows all three regularizers cost metric quality: removing
`L_n` is `+0.10` PSNR, `L_d` is `+0.05`, `L_z` is `+0.02`, each with an LPIPS
improvement too. They buy geometry, not PSNR.

## The hardening result

The window `phi ** sigma` deposits a closed-form fraction of each triangle's area
(derived and unit-tested in `sota/sigma_schedule.py`, checked against direct
sampling of `phi`):

    coverage(sigma) = 2 / ((sigma + 1) (sigma + 2))

`1/3` at the initial `sigma = 1`, `1` at the opaque endpoint. Under the published
linear-in-sigma anneal:

| iteration | sigma | coverage | vertex lr |
|---:|---:|---:|---:|
| 0 | 1.0 | 0.333 | 100% |
| 20000 | 0.333 | 0.643 | 6.3% |
| 25000 | 0.167 | 0.791 | 2.2% |
| 30000 | 0.0001 | 1.000 | 1.0% |

**31% of the total coverage change lands in the last 5k iterations, where the
vertex learning rate is ~2% of initial.** HARD-G0 measured stock losing
`0.344 dB` from 25k to 30k, on train as well as held-out — a hardening cost, not
overfitting.

A connected mesh cannot absorb this geometrically. Compensating the coverage
growth requires scaling each triangle about its own centroid (to 57.7% linear
size across the full range); shared vertices forbid it. The remaining degree of
freedom is *when* the coverage moves.

`sota/sigma_schedule.py` keeps both endpoints and changes only the path:

| schedule | coverage at 25k | left for the dead-lr tail |
|---|---:|---:|
| `linear` (published) | 0.791 | 0.209 |
| `coverage` | 0.889 | 0.111 |
| `lrmatched` | 0.992 | 0.008 |

`lrmatched` anneals coverage in proportion to the vertex learning-rate budget
still unspent, so the constraint tightens only as fast as the optimiser can
follow. HARD-G0 falsified *early* hardening (`-0.81 dB`) with a linear path and a
dead learning rate, which is a different arm from either of these.

## The baseline was broken before any of this ran

First `base` run on the 4090: Garden peaked at **20.51 dB around iteration 4000** and
then decayed monotonically to **11.38 dB** at 30000 — on the training set as well
(19.81 → 12.36), so it was destruction, not overfitting.

Ruled out by measurement: `fused_ssim` (bit-exact against the reference), the dataset
(byte-identical to the local copy that produced 24.74 on the A40), supervised normals
(`normals_4` is an empty directory the reader creates, so the loss never activates),
the vertex depth loss (clamped at `0.5`, weighted `2.5e-4`, so at most `1e-4` against
an L1 of `0.2`), `--use_sparse_adam` (same default as upstream), pruning (disabling it
still left the loss rising, 0.153 at 4.9k to 0.211 at 6.3k), and every default in
`arguments/__init__.py` (diff against upstream shows additions only).

The cause was **our own commit `6f7b03d`**, made during GoRFE-Q0 on 2026-08-11. Fixing
a finite-difference check there replaced

    dL_dvertices3D[idx].x = transposed_dL_ddepth.x;          // depth term only

with an **unconditionally executed** kernel

    dL_dvertices3D[idx].x += projection_gradient.x + depth_gradient.x;

adding the screen-space term of the vertex position gradient to the *default* path. It
is the more correct derivative, but MeshSplatting's learning rates, densification
thresholds and pruning sizes are all tuned against the published one. The last time
anyone trained a baseline was HARD-G0 at `81d9f99`, which predates it, so the change
sat unmeasured for six days.

It is now behind `--screen_space_gradients`, default off. The exact path is worth
testing as an arm with its own step size — see batch 3.

**The general lesson, and it is the same one this plan is built on:** a change that
passes a gradient check can still destroy training, and only a training run says so.

## Batch 1 — Garden, five arms

| arm | change |
|---|---|
| `base` | stock, at the official resolution |
| `noreg` | `--lambda_normals 0 --lambda_vertex 0` (their Table 4) |
| `lr60k` | `--position_lr_max_steps 60000`, keeps the vertex lr alive |
| `cov` | `--sigma_schedule coverage` |
| `lrm` | `--sigma_schedule lrmatched` |

`sota/batch1.sh garden`, then `python sota/table.py`.

## Results so far, all Garden at the official protocol

Reference: `base` reproduces the baseline at **24.7217 / 0.7616 / 0.2165** in 44 min,
against 24.7467 measured on the A40 and 24.71 in `record1.md`.

| arm | PSNR | vs base | peak | 26k→30k |
|---|---:|---:|---:|---:|
| `noreg` | **24.8063** | **+0.0847** | 25.316 @11k | −0.222 |
| `base` | 24.7217 | — | 25.299 @11k | −0.317 |
| `cov` | 24.5719 | −0.150 | 24.837 @21k | −0.216 |
| `lr60k` | 23.4845 | −1.237 | 24.743 @11k | −0.918 |
| `lrm` | 22.7901 | −1.932 | 23.106 @25k | −0.300 |
| `exact_lr10` | 12.4459 | −12.28 | 16.76 @2k | — |
| `exact_lr100` | 9.5900 | −15.13 | 15.09 @2k | — |
| `exact_lr1000` | 8.9700 | −15.75 | 14.72 @1k | — |

### The hardening cost is real and it is not the regularizers

Between 26k and 30k nothing changes but sigma — densification, pruning, the opacity
anneal and both supersampling switches are all done. Over those 4k iterations `base`
loses `0.317 dB`, `0.0183 SSIM` and gains `0.0149 LPIPS`, accelerating as it goes
(`−0.019, −0.070, −0.086, −0.141`), exactly where the coverage curve is steepest.
From the peak at 11k it is `0.58 dB`. Removing every active regularizer (`noreg`)
leaves the decline in place at `−0.222`, so it is the hardening, not the priors.

### The global schedule axis is closed

Coverage left for the last 5k iterations against final PSNR: `0.209 → 24.72`,
`0.111 → 24.57`, `0.008 → 22.79`. Monotone, and the mechanism is visible in the
peaks: `lrm` did reduce the tail to `−0.300` as designed, but its peak fell from
`25.30` to `23.11`. **The loss is relocated, not removed.** Softness is expressive,
every unit of coverage costs quality whenever it is spent, and the published linear
anneal is already near-optimal in this family because it stays soft as long as it can.
This also explains HARD-G0's `−0.81 dB` early-hardening result.

`lr60k` refutes the premise this plan started from. Keeping the vertex learning rate
alive through hardening does not let the geometry compensate — it costs `1.24 dB` and
triples the tail loss. **The decaying learning rate is not a limitation, it is
stabilisation.**

The closed-form coverage identity is still correct and still the right *analysis* —
it predicts where the decline steepens, and it explains why per-triangle compensation
is impossible on shared vertices. It is not a usable *intervention*.

### The exact vertex gradient is not a training path

Three step sizes, monotone the wrong way: `12.45`, `9.59`, `8.97`. Lowering the step
makes it worse, so the added term is not merely mis-scaled, it points somewhere
harmful. The likely reason is double counting: `BACKWARD::preprocess` already turns
`dL_dnormals`, `dL_doffsets` and `dL_dmean2D` — screen-space quantities derived from
the same projected positions — into vertex gradients, and `dL_dpoints2D` routed
through the projection adds that contribution a second time. Kept behind the flag,
default off.

### Per-face hardening order is closed too, by the same mechanism

Given the *same* total softness the published schedule spends — rates bounded by
`spread` and normalised to mean one, so a face can only buy softness by selling
it — the question was whether spending it unevenly helps. It does not:

| `spread` | final PSNR | peak | 26k | 26k→30k |
|---:|---:|---:|---:|---:|
| 1 (`base`) | **24.7217** | 25.299 @11k | 25.038 | −0.317 |
| 4 (`hard`) | 24.3666 | 25.454 @22k | 25.377 | −1.011 |
| 16 (`hard_wide`) | 23.8581 | 25.373 @22k | 25.360 | −1.502 |

The reallocation is real and the model commits to it hard: the learned rates
saturate at both bounds and are essentially binary, 88% of faces as hard as
allowed and 12% as soft as allowed, at an exactly unit mean. It genuinely helps
mid-training — both arms peak above `base`'s best and lead it by `0.33 dB` at
26k. Then the soft minority has to reach the endpoint, and the tail loss scales
with `spread`, taking more than the reallocation bought.

An earlier *unbudgeted* version answered a different question and answered it
degenerately: 93.6% of faces went softer, the median rate reached 89.5, and the
run held 25.83 dB until the endpoint identity forced every face onto
`sigma_final` in a single step and it collapsed to 21.47. That 25.83 is the soft
model, i.e. the 2.4 dB gap in their Table 1 measured again, not a method result.

### One mechanism closes the whole hardening axis

Four independent interventions, each with a monotone dose response:

| intervention | dose → final PSNR |
|---|---|
| coverage left for the tail | 0.209 / 0.111 / 0.008 → 24.72 / 24.57 / 22.79 |
| vertex learning rate through hardening | 30k / 60k steps → 24.72 / 23.48 |
| exact vertex gradient step size | 1/10, 1/100, 1/1000 → 12.45 / 9.59 / 8.97 |
| per-face reallocation width | spread 1 / 4 / 16 → 24.72 / 24.37 / 23.86 |

> **Softness has a price, it is paid at the moment it is given up, and neither
> when nor on which faces you give it up changes the total.**

Globally and per face, the same statement. The published schedule is already at
the optimum of this family because it keeps the model soft as long as it can and
then pays once.

## What is still open

- **Stretched schedule** (`sota/batch4.sh`). The one variable the four interventions
  above never changed is the number of updates. `lr60k` ruled out a bigger step,
  which is not the same as more steps: if the tail loss is the model failing to
  re-fit rather than coverage costing quality outright, scaling every phase
  boundary together should recover some of it. This is the last hardening question.
- `noreg`'s `+0.085` is small, free, and real, but it buys metrics with geometry —
  their Figure 7a is explicit about that.

If the stretched schedule is also null, the hardening axis is finished and the
direction question has to be reopened. The honest options at that point are
appearance capacity within the hard-opaque regime and primitive count, neither of
which this plan has tested on a correct baseline yet.

Two scenes must agree before any nine-scene run.

## Rules

- One change per arm, always against `base` from the same code revision.
- Every default is the published value; `--sigma_schedule linear` is byte-identical.
- No preregistration, no seed studies, no audit gates. Train, read PSNR, keep what wins.
- Anything that survives two scenes goes to all nine before it is believed.
