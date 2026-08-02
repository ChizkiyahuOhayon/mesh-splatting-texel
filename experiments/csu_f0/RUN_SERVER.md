# CSU-F0 server runbook

Run from the source checkout containing `csu_f0_eval.py`. Keep output and temporary
files on the NAS.

## Garden implementation smoke

```bash
CSU_SMOKE="$NAS_ROOT/experiments/csu_f0_garden_smoke_01"
test ! -e "$CSU_SMOKE" && echo "smoke output path OK"

TMPDIR="$NAS_ROOT/tmp" TMP="$NAS_ROOT/tmp" TEMP="$NAS_ROOT/tmp" \
TORCH_EXTENSIONS_DIR="$NAS_ROOT/torch_extensions" PYTHONDONTWRITEBYTECODE=1 \
SVSR_SOURCE_REVISION="$SOURCE_REVISION" CUDA_VISIBLE_DEVICES=0 \
python csu_f0_eval.py \
  -s "$GARDEN_DATA" -m "$GARDEN_SH" --eval \
  --csu_scene garden --csu_output "$CSU_SMOKE" --csu_smoke
```

The smoke is valid when `results.json` and `DONE` exist, the topology and prefix
integrity checks are true, all reported values are finite, and `decision.pass` is
null. Only then run the unchanged Garden and Room confirmatory probes.
