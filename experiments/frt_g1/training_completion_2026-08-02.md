---
exp_id: MS-A40-260802-001
date: 2026-08-02
system: MeshSplatting
experiment: FRT-G1
classification: locked_training_complete_metrics_blinded
scenes: [garden, room]
updates_per_scene: 5000
confirmatory: true
anomaly: false
tags: [frozen-base, texels, training-complete, metrics-blinded]
---

# FRT-G1 locked training completion

## Observed completion checks

Both Garden and Room output directories contain `DONE`. Their manifests report
5,000 updates with confirmatory settings. Both integrity files report:

- `zero_init_max_abs = 0.0`;
- `base_tensors_unchanged = true`;
- optimizer parameter groups equal to `["texels"]`.

The shell job table is empty because both processes exited normally. No held-out
quality result was inspected in the supplied material, so FRT-G1 remains blinded
and undecided.

## Next step

Evaluate each final checkpoint exactly once on its complete held-out split using the
locked SH reference and decomposition evaluator. Do not retrain, select an
intermediate checkpoint, or change any setting before the Garden and Room metrics
are both available.

## Raw material

The supplied terminal output referenced the server directories through shell
variables `$FRT_GARDEN` and `$FRT_ROOM`; their expanded paths were not printed.
