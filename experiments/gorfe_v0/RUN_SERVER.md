# GoRFE-V0 server command — sealed

Attempt `_01` passed at source
`126121dfdeaaebbec6dd978d7f90101425b815ea`. Do not rerun or overwrite it. V0
is a synthetic CPU/float64 integrity gate; it reserved no GPU and read no scene
or checkpoint. The commands below are retained only as the exact reproduction
record.

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
