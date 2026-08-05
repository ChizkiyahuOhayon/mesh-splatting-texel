---
exp_id: MS-A40-260805-002
date: 2026-08-05
system: MeshSplatting
experiment: SAC-G1
classification: confirmatory_two_scene_replication
scenes: [garden, room]
arms: [stock, splat2]
seeds: [0, 1, 2]
gpu: NVIDIA A40
source_revision: 47101877850e8072f2808d684ffa6fdd68b28d07
confirmatory: true
anomaly: false
tags: [sampling, replication, confound-removed, negative-result, seed-variance, axis-closed]
---

# SAC-G1: with the confound removed the effect is gone

## Verdict

**FAIL.** Three of five checks failed; validity passed.

| Check | Result |
|---|---|
| stock reproduces each baseline | PASS (Garden 24.7472 vs 24.7372; Room 28.4666 vs 28.5142) |
| both scenes have paired seeds | PASS |
| mean PSNR difference positive on both scenes | **FAIL** |
| effect exceeds twice its standard error | **FAIL** |
| LPIPS not worse on either scene | **FAIL** |

Paired differences, `splat2@4` minus `stock@4`:

| Scene | seed 0 | seed 1 | seed 2 | mean |
|---|---:|---:|---:|---:|
| Garden | +0.0129 | +0.0072 | -0.0183 | +0.0006 |
| Room | +0.0620 | -0.0591 | -0.0673 | -0.0214 |

Pooled mean `-0.0104 dB`, standard error `0.0198`, lower bound at two standard
errors `-0.0500`. The differences disagree in sign within both scenes.

## The SAC-G0 observation was the confound

With the post-training cleanup pinned to `scaling 4` in both arms, the primitive
counts converge: Garden 6,977,952 versus 7,081,664 (+1.5%), Room 5,638,863
versus 5,671,278 (+0.6%). SAC-G0's 18% gap is gone, and the quality difference
went with it.

That confirms the diagnosis recorded in `experiments/sac_g0/analysis_full_01.md`:
the entire effect came from one step, where faces are deleted unless their
per-pixel maximum blending weight exceeds 0.5, evaluated at whatever
supersampling factor training ended on. It was not a property of training at the
cheaper rate. The cautions recorded with that observation were correct, and the
replication was worth running precisely because they were.

## Seed variance, measured for the first time

`safe_state` hard-coded seed 0, so this pipeline had never produced seed
variance. Standard deviation of held-out PSNR across three seeds:

| Scene | stock | splat2 |
|---|---:|---:|
| Garden | 0.0157 | 0.0151 |
| Room | 0.0861 | 0.1428 |

Garden is remarkably stable and Room is roughly six to nine times noisier. Two
retrospective consequences:

- Garden's repeated texel gain of about `+0.3 dB` is roughly twenty seed
  standard deviations and was never plausibly noise.
- Room conclusions at the `0.1 dB` level were underpowered at one seed. The
  Room texel regression of `0.2261 dB` is about two to three standard
  deviations, so it stands, but only just.

Any future single-seed claim on this pipeline should be read against these
numbers.

## Decision

The sampling axis is closed, as preregistered. No further variant is attempted.

## What the SAC branch leaves behind

- A validated training platform: the stock arm reproduces the recorded baselines
  to 0.010 dB on Garden and 0.048 dB on Room across three seeds.
- A real property of the baseline: its final primitive count is an artefact of
  the training supersampling factor, because the cleanup threshold is applied to
  a per-pixel maximum. Changing the schedule silently changes the model size by
  18% without changing quality.
- The deployment measurement from SAC-G0: training at the deployment rate costs
  `0.073 dB` and renders `1.62x` faster, whereas rendering a stock model cheaply
  costs `0.293 dB`. Two thirds of the apparent supersampling cost is a
  train/test mismatch, not the sampling rate.
- The first seed-variance figures this project has had.
