# RITS-P0 server runbook

Run from the source checkout containing `rits_p0_fit.py`. The rasterizer is
unchanged since RITS-D1 (`55e6c6d`), so no extension rebuild is needed if it was
rebuilt for D1. P0 fits only the parameters a split appends; the base model is
frozen by zeroed gradients and asserted bitwise unchanged at the end.

Output paths are reserved write-once: a failed attempt keeps its directory, and
a retry uses the next suffix.

## Garden implementation smoke

```bash
P0_SMOKE="$NAS_ROOT/experiments/rits_p0_garden_smoke_01"
test ! -e "$P0_SMOKE" && echo "smoke output path OK"

TMPDIR="$NAS_ROOT/tmp" TMP="$NAS_ROOT/tmp" TEMP="$NAS_ROOT/tmp" \
TORCH_EXTENSIONS_DIR="$NAS_ROOT/torch_extensions" PYTHONDONTWRITEBYTECODE=1 \
SVSR_SOURCE_REVISION="$SOURCE_REVISION" CUDA_VISIBLE_DEVICES=0 \
python rits_p0_fit.py \
  -s "$GARDEN_DATA" -m "$GARDEN_SH" --eval \
  --p0_scene garden --p0_output "$P0_SMOKE" --p0_smoke
```

Valid when `results.json` and `DONE` exist, `decision.scene_pass` is null,
`original_parameter_prefix_bitwise_unchanged` and `topology_counts_exact` are
true, the loss trace is finite and decreasing, and both `held_out.inherited`
and `held_out.projected` are populated. The smoke decides nothing.

## Confirmatory runs

```bash
P0_GARDEN="$NAS_ROOT/experiments/rits_p0_garden_full_01"
TMPDIR="$NAS_ROOT/tmp" TMP="$NAS_ROOT/tmp" TEMP="$NAS_ROOT/tmp" \
TORCH_EXTENSIONS_DIR="$NAS_ROOT/torch_extensions" PYTHONDONTWRITEBYTECODE=1 \
SVSR_SOURCE_REVISION="$SOURCE_REVISION" CUDA_VISIBLE_DEVICES=0 \
python rits_p0_fit.py \
  -s "$GARDEN_DATA" -m "$GARDEN_SH" --eval \
  --p0_scene garden --p0_output "$P0_GARDEN"

P0_ROOM="$NAS_ROOT/experiments/rits_p0_room_full_01"
TMPDIR="$NAS_ROOT/tmp" TMP="$NAS_ROOT/tmp" TEMP="$NAS_ROOT/tmp" \
TORCH_EXTENSIONS_DIR="$NAS_ROOT/torch_extensions" PYTHONDONTWRITEBYTECODE=1 \
SVSR_SOURCE_REVISION="$SOURCE_REVISION" CUDA_VISIBLE_DEVICES=0 \
python rits_p0_fit.py \
  -s "$ROOM_DATA" -m "$ROOM_SH" --eval \
  --p0_scene room --p0_output "$P0_ROOM"
```

Each run is 1,000 fitting steps on four views plus a handful of evaluation
renders, so expect roughly 15 minutes per scene.

## Locked decision

```bash
python rits_p0_decide.py --garden "$P0_GARDEN" --room "$P0_ROOM"
```

The rule is in `experiments/rits_p0/protocol.md`: on both scenes the held-out
probe-region MAE must fall to at most 20% of the inherited operator's, the
held-out global MAE must improve, and the base model must be bitwise intact.
On failure the topology branch closes and the project pivots to the
soft-compositor efficiency thesis.
