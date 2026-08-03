# RITS-D0 server runbook

Run from the source checkout containing `rits_d0_eval.py`. Keep output and
temporary files on the NAS.

## One-time: rebuild the rasterizer extension

RITS-D0 is the first gate that changes the CUDA rasterizer, so the installed
`diff_triangle_rasterization` package must be rebuilt from this source revision
before any run:

```bash
TORCH_EXTENSIONS_DIR="$NAS_ROOT/torch_extensions" TMPDIR="$NAS_ROOT/tmp" \
pip install --no-build-isolation --force-reinstall \
  ./submodules/diff-triangle-mesh-rasterization
python -c "from diff_triangle_rasterization import rasterize_triangles; print('extension OK')"
```

The rebuilt extension must leave donor-free rendering untouched. This is not
assumed: variant 1 of every confirmatory run re-executes the CSU-F0 split on the
original code path and the evaluator fails the gate if its parity numbers drift
more than 5% from the recorded CSU-F0 values.

## Garden implementation smoke

```bash
RITS_SMOKE="$NAS_ROOT/experiments/rits_d0_garden_smoke_01"
test ! -e "$RITS_SMOKE" && echo "smoke output path OK"

TMPDIR="$NAS_ROOT/tmp" TMP="$NAS_ROOT/tmp" TEMP="$NAS_ROOT/tmp" \
TORCH_EXTENSIONS_DIR="$NAS_ROOT/torch_extensions" PYTHONDONTWRITEBYTECODE=1 \
SVSR_SOURCE_REVISION="$SOURCE_REVISION" CUDA_VISIBLE_DEVICES=0 \
python rits_d0_eval.py \
  -s "$GARDEN_DATA" -m "$GARDEN_SH" --eval \
  --rits_scene garden --rits_output "$RITS_SMOKE" --rits_smoke
```

The smoke is valid when `results.json` and `DONE` exist, every variant's
topology and prefix integrity checks are true, all reported values are finite,
`decision.pass` is null, and variant 4's `children_rendered` is nonzero (the
donor path actually rendered children). Only then run the unchanged Garden and
Room confirmatory probes:

```bash
RITS_GARDEN="$NAS_ROOT/experiments/rits_d0_garden_full_01"
TMPDIR="$NAS_ROOT/tmp" TMP="$NAS_ROOT/tmp" TEMP="$NAS_ROOT/tmp" \
TORCH_EXTENSIONS_DIR="$NAS_ROOT/torch_extensions" PYTHONDONTWRITEBYTECODE=1 \
SVSR_SOURCE_REVISION="$SOURCE_REVISION" CUDA_VISIBLE_DEVICES=0 \
python rits_d0_eval.py \
  -s "$GARDEN_DATA" -m "$GARDEN_SH" --eval \
  --rits_scene garden --rits_output "$RITS_GARDEN"

RITS_ROOM="$NAS_ROOT/experiments/rits_d0_room_full_01"
TMPDIR="$NAS_ROOT/tmp" TMP="$NAS_ROOT/tmp" TEMP="$NAS_ROOT/tmp" \
TORCH_EXTENSIONS_DIR="$NAS_ROOT/torch_extensions" PYTHONDONTWRITEBYTECODE=1 \
SVSR_SOURCE_REVISION="$SOURCE_REVISION" CUDA_VISIBLE_DEVICES=0 \
python rits_d0_eval.py \
  -s "$ROOM_DATA" -m "$ROOM_SH" --eval \
  --rits_scene room --rits_output "$RITS_ROOM"
```

RITS-D0 passes only if `decision.scene_pass` is true for **both** scenes. The
locked decision rule is in `experiments/rits_d0/protocol.md`; no threshold,
view, or face count may change after observing any confirmatory output.
