# EdgeVal-E0 attempt 01 — infrastructure failure

Date observed: 2026-08-09

Status: **EXCLUDED — native build did not start**

## Identity

- Source revision: `7bd47f31698368d5b3dbb6487104614a9d0dba29`
- Physical GPU: 2, NVIDIA A40, 46,068 MiB total, 45,415 MiB free
- Python environment: PyTorch `2.7.1+cu126`, CUDA runtime `12.6`
- Persistent directory: `${NAS_ROOT}/experiments/edgeval_e0_01`

## Failure

The original runner exported `TMPDIR`, `TMP`, `TEMP`, and
`TORCH_EXTENSIONS_DIR` under `${NAS_ROOT}`. During pip metadata preparation,
setuptools tried to create a temporary file below
`${NAS_ROOT}/tmp/pip-modern-metadata-*/diff_triangle_rasterization.egg-info` and
received:

```text
error: [Errno 1] Operation not permitted: .../diff_triangle_rasterization.egg-info/tmp...
error: metadata-generation-failed
```

The command exited before C++/CUDA compilation, CPU tests, renderer execution,
or result generation. This attempt contains no EdgeVal measurement and cannot
pass or fail E0.

## Repair and replay rule

Persistent logs and sealed results remain on NAS, but all pip/compiler temporary
files now use a unique `mktemp` directory below local `/tmp`; the runner validates
that path and removes only that exact directory on exit. Attempt 01 must remain
untouched. The repaired run uses suffix `02` and a new source revision.
