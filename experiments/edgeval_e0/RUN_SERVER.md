# EdgeVal-E0 server runbook

This gate rebuilds the native renderer, runs the exact mathematical-core tests,
and executes one CUDA forward/backward fixture on exclusive physical GPU 1. It
does not access Garden and does not train a model.

```bash
cd /home/smbu/dy/mesh-splatting-texel
micromamba activate mesh_splatting
git pull --ff-only git@github.com:ChizkiyahuOhayon/mesh-splatting-texel.git main
git rev-parse HEAD
git status --short

export NAS_ROOT=/home/smbu/dy/nas/meshsplatting_smbu
export EDGEVAL_E0_RUN_SUFFIX=01
export EDGEVAL_E0_GPU=1

bash experiments/edgeval_e0/run_gpu1.sh
```

Return the complete sealed evidence:

```bash
OUT="$NAS_ROOT/experiments/edgeval_e0_01"
cat "$OUT/result.json"
cat "$OUT/manifest.json"
cat "$OUT/SHA256SUMS"
```

If the script fails, do not delete or reuse the partial directory. Return the
last 120 lines of `build.log`, `tests.log`, or `smoke.log`, whichever stage
failed, and use a fresh suffix only after the failure is understood.
