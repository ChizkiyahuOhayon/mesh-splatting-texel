# GoRFE-Q0 attempt 01 — excluded environment incident

Date: 2026-08-11

Source revision: `40df5bb16663f28981c997e230650c17f68f9fd0`

Artifact directory: `${NAS_ROOT}/experiments/gorfe_q0_01`

## Classification

**EXCLUDED INFRASTRUCTURE ATTEMPT.** This attempt is not a GoRFE-Q0 result and
contains no scientific measurement. The CUDA smoke and its fixed Boolean
decision were never executed.

## Observed sequence

- Physical GPU 3 was an exclusive NVIDIA A40 with 45,415 MiB free.
- The shell prompt was `(base)`, and the runner used PyTorch `2.1.2+cu118` with
  CUDA `11.8` instead of the locked `mesh_splatting` environment
  (`2.7.1+cu126`, CUDA `12.6`).
- The native wheel compiled and installed successfully in the wrong Python
  environment.
- Test execution stopped with eight errors: five EdgeVal core cases used the
  newer tuple-dimension form of `Tensor.all`, and three legacy tests were
  imported as top-level modules so their `tests.*` imports were shadowed.
- Because the runner uses `set -euo pipefail`, it stopped before
  `gorfe_q0_smoke.py`; no `result.json`, `manifest.json`, `DONE`, or complete
  checksum ledger was produced.

## Repair and replay rule

The repair is source-level and does not change the Q0 scientific predicate:

1. use sequential reductions compatible with both supported Torch APIs;
2. discover tests with the repository root fixed as the top-level directory;
3. verify the exact Torch/CUDA identity before creating an output directory or
   compiling the extension.

Attempt 01 remains immutable. Replay uses a fresh suffix `_02` after activating
`mesh_splatting` and pulling the repair commit.
