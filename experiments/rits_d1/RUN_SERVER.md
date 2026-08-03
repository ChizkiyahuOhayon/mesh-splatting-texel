# RITS-D1 server runbook

Run from the source checkout containing `rits_d1_eval.py`. Keep output and
temporary files on the NAS. The rasterizer sources changed again
(`DONOR_APPEARANCE`), so the extension must be rebuilt at this revision:

```bash
pip uninstall -y diff_triangle_rasterization
pip install submodules/diff-triangle-mesh-rasterization --no-build-isolation
python -c "from diff_triangle_rasterization import rasterize_triangles; print('extension OK')"
```

Variant 1 and variant 4 re-run the CSU-F0 and RITS-D0 configurations verbatim
inside every confirmatory run and must reproduce their recorded numbers within
5%; the evaluator fails the gate otherwise, so a stale or drifting build cannot
produce a valid pass.

## Garden implementation smoke

```bash
RITS_SMOKE="$NAS_ROOT/experiments/rits_d1_garden_smoke_01"
test ! -e "$RITS_SMOKE" && echo "smoke output path OK"

TMPDIR="$NAS_ROOT/tmp" TMP="$NAS_ROOT/tmp" TEMP="$NAS_ROOT/tmp" \
TORCH_EXTENSIONS_DIR="$NAS_ROOT/torch_extensions" PYTHONDONTWRITEBYTECODE=1 \
SVSR_SOURCE_REVISION="$SOURCE_REVISION" CUDA_VISIBLE_DEVICES=0 \
python rits_d1_eval.py \
  -s "$GARDEN_DATA" -m "$GARDEN_SH" --eval \
  --rits_scene garden --rits_output "$RITS_SMOKE" --rits_smoke
```

Valid when `results.json` and `DONE` exist, every variant's integrity checks
are true, values are finite, and `decision.pass` is null.

## Garden and Room confirmatory runs

```bash
RITS_GARDEN="$NAS_ROOT/experiments/rits_d1_garden_full_01"
TMPDIR="$NAS_ROOT/tmp" TMP="$NAS_ROOT/tmp" TEMP="$NAS_ROOT/tmp" \
TORCH_EXTENSIONS_DIR="$NAS_ROOT/torch_extensions" PYTHONDONTWRITEBYTECODE=1 \
SVSR_SOURCE_REVISION="$SOURCE_REVISION" CUDA_VISIBLE_DEVICES=0 \
python rits_d1_eval.py \
  -s "$GARDEN_DATA" -m "$GARDEN_SH" --eval \
  --rits_scene garden --rits_output "$RITS_GARDEN"

RITS_ROOM="$NAS_ROOT/experiments/rits_d1_room_full_01"
TMPDIR="$NAS_ROOT/tmp" TMP="$NAS_ROOT/tmp" TEMP="$NAS_ROOT/tmp" \
TORCH_EXTENSIONS_DIR="$NAS_ROOT/torch_extensions" PYTHONDONTWRITEBYTECODE=1 \
SVSR_SOURCE_REVISION="$SOURCE_REVISION" CUDA_VISIBLE_DEVICES=0 \
python rits_d1_eval.py \
  -s "$ROOM_DATA" -m "$ROOM_SH" --eval \
  --rits_scene room --rits_output "$RITS_ROOM"
```

RITS-D1 passes only if `decision.scene_pass` is true for **both** scenes. The
locked rule is in `experiments/rits_d1/protocol.md`. On failure the topology
branch exits — final, no D2.
