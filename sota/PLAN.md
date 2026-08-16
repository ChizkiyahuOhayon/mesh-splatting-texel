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

Batch 2 combines whatever wins and repeats on Room; the nine-scene run comes only
after two scenes agree.

## Rules

- One change per arm, always against `base` from the same code revision.
- Every default is the published value; `--sigma_schedule linear` is byte-identical.
- No preregistration, no seed studies, no audit gates. Train, read PSNR, keep what wins.
- Anything that survives two scenes goes to all nine before it is believed.
