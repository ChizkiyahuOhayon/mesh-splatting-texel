# GoRFE-V0 server command

V0 is a synthetic CPU/float64 integrity gate.  It deliberately does not reserve
GPU 3 and does not read a scene or checkpoint.

```bash
cd ~/dy/mesh-splatting-texel
git pull --ff-only fork main

micromamba activate mesh_splatting

export NAS_ROOT=/home/smbu/dy/nas/meshsplatting_smbu
export GORFE_V0_RUN_SUFFIX=01
export GORFE_V0_PYTHON="$(command -v python)"

bash experiments/gorfe_v0/run.sh
```

After success:

```bash
OUT="$NAS_ROOT/experiments/gorfe_v0_01"
cat "$OUT/result.json"
cat "$OUT/manifest.json"
cat "$OUT/SHA256SUMS"
ls -l "$OUT/DONE"
```

Do not advance to V1 unless `result.json` and `manifest.json` both say `pass`
and `DONE` plus `SHA256SUMS` exist.
