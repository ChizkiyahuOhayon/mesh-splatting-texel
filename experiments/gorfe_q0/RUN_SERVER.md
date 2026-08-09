# GoRFE-Q0 server command

Run only from the requested clean source revision and a GPU-exclusive shell.

```bash
cd ~/dy/mesh-splatting-texel
git pull --ff-only fork main

export NAS_ROOT=/home/smbu/dy/nas/meshsplatting_smbu
export GORFE_Q0_RUN_SUFFIX=01
export GORFE_Q0_GPU=2

bash experiments/gorfe_q0/run_gpu2.sh
```

After success:

```bash
OUT="$NAS_ROOT/experiments/gorfe_q0_01"
cat "$OUT/result.json"
cat "$OUT/manifest.json"
cat "$OUT/SHA256SUMS"
```
