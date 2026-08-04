# SAC-G0: is MeshSplatting's deployment supersampling necessary?

Status: **PREREGISTERED — no SAC-G0 output has been observed**

Date: 2026-08-04

## Question

MeshSplatting renders at four times linear resolution and area-downsamples, so
every deployed pixel costs sixteen samples. The renderer ladder measured what
that last factor buys: on a stock Garden model, dropping from `scaling 4` to
`scaling 2` costs only `0.209 dB`, while `scaling 1 -> 2` was worth `1.362 dB`.
The sampling return is already saturated at four samples per pixel.

That measurement rendered a model **trained** at `scaling 4` more cheaply, so it
confounds the sampling rate with a train/test mismatch. This gate asks the
question that matters for a budget claim:

> If the model is trained at the sampling rate it will be deployed at, how much
> quality does the cheaper rate actually cost?

A small enough cost puts a four-fold deployment sampling saving on the table,
which is the budget the capacity half of the thesis proposes to spend on
primitives. This gate does not test the capacity half and claims nothing about
it.

## Why this is not another mechanism hunt

The nine falsified directions each required discovering a new signal inside
MeshSplatting. This one requires no discovery: it recombines two quantities the
baseline itself has already measured — a saturated sampling return and an
unsaturated capacity return (`+0.46 dB` and `-0.06 LPIPS` from 2M to 5M
vertices in MeshSplatting's own ablation). The risk is correspondingly
different: not that the effect is absent, but that training at the cheaper rate
costs more than rendering at it does.

## Arms

Garden, full 30,000-iteration training, identical in every respect except one
value; same seed, same data, same schedule, same losses, same densification.

| Arm | Supersampling schedule |
|---|---|
| stock | `scaling 1` to 20k, `2` to 25k, **`4`** to 30k (published behaviour) |
| splat2 | `scaling 1` to 20k, `2` to 25k, **`2`** to 30k |

The only code change is that the final factor, hard-coded as `4` in `train.py`,
becomes the parameter `final_scaling` whose default is `4`. The stock arm is
therefore the unmodified pipeline.

## Evaluation

Each trained model is evaluated on the complete held-out split at **both**
`scaling 2` and `scaling 4`, giving four cells. Per cell: PSNR, SSIM,
LPIPS-VGG, and CUDA-event render time per view. Also recorded per arm: training
wall-clock, peak memory, and final vertex and triangle counts.

The cells separate the two effects the ladder confounded. `stock@2` reproduces
the ladder's mismatched measurement; `splat2@2` is the same sampling rate
without the mismatch; `splat2@4` shows whether a model trained cheaply still
benefits from extra samples it never saw.

## Locked decision

1. **Validity.** `stock@4` must land within `0.10 dB` PSNR, `0.010` SSIM and
   `0.010` LPIPS of the recorded `garden_baseline` checkpoint under this same
   evaluation path: `24.7372 dB`, `0.7484`, `0.2480`. If it does not, the
   training platform is wrong and no comparison is read — the failure mode that
   voided RITS-T0.
2. **Quality cost.** `splat2@2` loses at most `0.35 dB` PSNR and at most
   `0.020` LPIPS against `stock@4`.
3. **Real saving.** Rendering the same model at `scaling 2` is at least
   `2.5x` faster than at `scaling 4`, measured by CUDA events over the held-out
   split. The saving must be measured, not assumed from sample counts.

The `0.35 dB` bound is tied to the arithmetic rather than chosen for comfort.
MeshSplatting's ablation implies roughly `+0.5 dB` per doubling of vertices, so
a sampling saving that buys between `1.5x` and `2x` primitives at matched render
cost returns about `+0.3` to `+0.5 dB`. A sampling cost above `0.35 dB` leaves
no net gain and the thesis would be dead on arrival.

All three must hold. A pass authorizes only the next gate — the capacity half,
on Garden and Room — and no quality claim. A failure closes the sampling-budget
thesis; the project then has no remaining direction with measured support and
the goal itself must be reconsidered rather than a tenth variant attempted.

## Recorded but not decisive

Training wall-clock ratio, `splat2@4`, `stock@2`, peak memory, and final
primitive counts. Only 5,000 of 30,000 iterations differ between arms, so a
large training speedup is not expected and is not required: the budget this
thesis proposes to spend is the deployment sampling cost, which every rendered
frame pays.

## Notes on scope

Garden alone cannot support any claim. It is used first because it is the scene
where the ladder was measured and where appearance headroom was repeatedly
observed, so it is the most favourable scene for the sampling half; a failure
here would generalise, a pass would not.
