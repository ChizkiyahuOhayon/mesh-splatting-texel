---
exp_id: MS-A40-260804-002
date: 2026-08-04
system: MeshSplatting
experiment: RITS-T0 attempt 01
classification: confirmatory_two_scene_gate
scenes: [garden, room]
arms: [unsplit, abrupt, rits]
gpu: NVIDIA A40
source_revision: 35aa230781997049dd9b7ecb87292bc598c1c3c1
confirmatory: true
anomaly: true
tags: [rits, training, void-run, negative-result, design-error]
---

# RITS-T0 attempt 01: FAIL by the locked rule, and void as a test

## Recorded verdict

`rits_t0_decide.py` returns `pass: false` with every check false:
mean PSNR gain vs unsplit `-0.0643 dB`; mean margin vs abrupt `-0.2341 dB`.

| Scene | Arm | PSNR | SSIM | LPIPS-VGG |
|---|---|---:|---:|---:|
| Garden | unsplit | 22.0777 | 0.6063 | 0.3926 |
| Garden | abrupt | 22.3033 | 0.6312 | 0.3639 |
| Garden | rits | 22.1132 | 0.6197 | 0.3754 |
| Room | unsplit | 23.0983 | 0.7962 | 0.3827 |
| Room | abrupt | 23.2123 | 0.8020 | 0.3724 |
| Room | rits | 22.9342 | 0.7937 | 0.3829 |

## Why the run is void as evidence about refinement

Every arm ends far below the checkpoint it started from, measured by each run's
own step-0 evaluation on the identical path:

| Scene | Checkpoint | Best arm | Worst arm |
|---|---|---|---|
| Garden | 24.7372 / 0.7484 / 0.2480 | 22.3033 | 22.0777 |
| Room | 28.5142 / 0.8916 / 0.2431 | 23.2123 | 22.9342 |

The unsplit arm changes no topology and still loses 2.66 dB on Garden and
5.42 dB on Room. A procedure that destroys 5 dB cannot resolve the 0.15 dB
effect the gate was designed to detect. This criterion is computed from
step-0 metrics recorded before any arm was compared, so it does not depend on
the arm ranking.

The verdict above stands as recorded and is not revised. What the run does not
provide is evidence for or against refinement continuity.

## Two design errors, both in the protocol author's hands

1. **Invalid fine-tuning configuration.** The arms used the restore-path
   learning rates (`f_dc` 0.0016, `vertices` 1e-4) with fresh Adam moments on
   a converged 30k checkpoint, and a photometric-only loss without the
   production regularizers. The protocol contained no validity precondition
   such as "the unsplit arm must not regress below the loaded checkpoint".

2. **The homotopy is a no-op under Adam.** `F_gamma = gamma * F_child +
   (1 - gamma) * F_donor.detach()` scales the gradient *magnitude* by gamma,
   and Adam normalizes each parameter by its own gradient magnitude, so the
   intended soft start never existed: full-size steps were taken from step 1
   in a direction evaluated on an image that was almost entirely the donor.
   Adam's scale invariance is a first-principles fact; this error was
   identifiable at design time and did not require an experiment.

## The deeper problem the run exposes

At `gamma = 1` the render is the donor-free split model with inherited values,
which is exactly D0's variant 1 — the abrupt state. The exact prolongation
therefore holds only while donors are active, and the homotopy changes the
optimization *path* while both split arms share the same *destination*. Any
benefit is then a trajectory effect, which an "abrupt plus learning-rate
warmup" control could plausibly reproduce. A continuation that only smooths
the path is not a defensible Oral contribution.

## What the data does support

In this damaged regime the abrupt split still beat no split on both scenes
(`+0.226 dB` Garden, `+0.114 dB` Room, with SSIM and LPIPS agreeing). Adding
topology to high-coverage faces helps; what is unsupported is that this
particular continuation improves on the abrupt operator.

## Consequence

Attempt 01 is retained permanently as the record of this configuration. No
re-run inherits its status: any successor is a new preregistered gate, must
declare that its design was chosen after this ranking was observed, and must
carry a validity precondition checkable before the arms are compared.
