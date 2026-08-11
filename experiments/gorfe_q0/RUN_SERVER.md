# GoRFE-Q0 server command — sealed

Attempt `_04` passed at source `6f7b03d6f9a72d75ea143bfd00f23d66e5a3f961`.
Do not rerun or overwrite it. The commands below are retained only as the exact
reproduction record.

Run only from the requested clean source revision and a GPU-exclusive shell.

```bash
cd ~/dy/mesh-splatting-texel
git pull --ff-only fork main

micromamba activate mesh_splatting

export NAS_ROOT=/home/smbu/dy/nas/meshsplatting_smbu
export GORFE_Q0_RUN_SUFFIX=04
export GORFE_Q0_GPU=3
export GORFE_Q0_PYTHON="$(command -v python)"

bash experiments/gorfe_q0/run_gpu2.sh
```

After success:

```bash
OUT="$NAS_ROOT/experiments/gorfe_q0_04"
cat "$OUT/result.json"
cat "$OUT/manifest.json"
cat "$OUT/SHA256SUMS"
```
