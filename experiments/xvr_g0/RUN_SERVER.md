# XVR-G0 server runbook

Run from the source checkout containing `xvr_g0_eval.py`. Keep temporary files on
the NAS because the server root filesystem is full.

## Smoke

```bash
XVR_SMOKE="$NAS_ROOT/experiments/xvr_g0_garden_smoke_01"
test ! -e "$XVR_SMOKE" && echo "smoke output path OK"

TMPDIR="$NAS_ROOT/tmp" TMP="$NAS_ROOT/tmp" TEMP="$NAS_ROOT/tmp" \
TORCH_EXTENSIONS_DIR="$NAS_ROOT/torch_extensions" PYTHONDONTWRITEBYTECODE=1 \
SVSR_SOURCE_REVISION="$SOURCE_REVISION" CUDA_VISIBLE_DEVICES=0 \
python xvr_g0_eval.py \
  -s "$GARDEN_DATA" -m "$GARDEN_SH" --eval \
  --xvr_scene garden --xvr_output "$XVR_SMOKE" \
  --xvr_max_train_views 2 --xvr_max_test_views 1
```

The smoke is valid when `results.json` and `DONE` exist and `decision.pass` is null.

## Locked Garden and Room runs

Use new, non-existing output directories and the exact final SH-only baseline models.

```bash
XVR_GARDEN="$NAS_ROOT/experiments/xvr_g0_garden_full_01"
XVR_ROOM="$NAS_ROOT/experiments/xvr_g0_room_full_01"

TMPDIR="$NAS_ROOT/tmp" TMP="$NAS_ROOT/tmp" TEMP="$NAS_ROOT/tmp" \
TORCH_EXTENSIONS_DIR="$NAS_ROOT/torch_extensions" PYTHONDONTWRITEBYTECODE=1 \
SVSR_SOURCE_REVISION="$SOURCE_REVISION" CUDA_VISIBLE_DEVICES=0 \
python xvr_g0_eval.py \
  -s "$GARDEN_DATA" -m "$GARDEN_SH" --eval \
  --xvr_scene garden --xvr_output "$XVR_GARDEN"

TMPDIR="$NAS_ROOT/tmp" TMP="$NAS_ROOT/tmp" TEMP="$NAS_ROOT/tmp" \
TORCH_EXTENSIONS_DIR="$NAS_ROOT/torch_extensions" PYTHONDONTWRITEBYTECODE=1 \
SVSR_SOURCE_REVISION="$SOURCE_REVISION" CUDA_VISIBLE_DEVICES=1 \
python xvr_g0_eval.py \
  -s "$ROOM_DATA" -m "$ROOM_SH" --eval \
  --xvr_scene room --xvr_output "$XVR_ROOM"
```

Do not launch the locked runs until the smoke completes without an exception.
