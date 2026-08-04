# RITS-T0 server runbook

Run from the source checkout containing `rits_t0_train.py`. The rasterizer is
unchanged since RITS-D1 (`55e6c6d`), so if the extension was rebuilt for D1 no
rebuild is needed; otherwise rebuild it first (see `experiments/rits_d1/RUN_SERVER.md`).

Each invocation trains one arm of one scene. Confirmatory settings: top 10% of
faces by full-train-split coverage, 5,000 fine-tuning steps, anneal over the
first 1,000 (rits arm only). Expect roughly 45-60 minutes per arm on the A40;
the rits arm adds ~10% for the donor passes and its G0-lite precondition.

## Garden smoke (rits arm exercises every code path)

```bash
T0_SMOKE="$NAS_ROOT/experiments/rits_t0_garden_rits_smoke_01"
test ! -e "$T0_SMOKE" && echo "smoke output path OK"

TMPDIR="$NAS_ROOT/tmp" TMP="$NAS_ROOT/tmp" TEMP="$NAS_ROOT/tmp" \
TORCH_EXTENSIONS_DIR="$NAS_ROOT/torch_extensions" PYTHONDONTWRITEBYTECODE=1 \
SVSR_SOURCE_REVISION="$SOURCE_REVISION" CUDA_VISIBLE_DEVICES=0 \
python rits_t0_train.py \
  -s "$GARDEN_DATA" -m "$GARDEN_SH" --eval \
  --t0_scene garden --t0_arm rits --t0_output "$T0_SMOKE" --t0_smoke
```

The smoke is valid when `results.json` and `DONE` exist, `g0_lite` reports
nonzero geometry and appearance gradient norms with all finite-difference rows
within tolerance, the loss trace is finite and decreasing in tendency, and
`final_metrics` is populated. The smoke makes no decision.

## Confirmatory runs (6 arms; any order; one GPU each)

```bash
for SCENE in garden room; do
  case $SCENE in
    garden) DATA="$GARDEN_DATA"; SH="$GARDEN_SH";;
    room)   DATA="$ROOM_DATA";   SH="$ROOM_SH";;
  esac
  for ARM in unsplit abrupt rits; do
    OUT="$NAS_ROOT/experiments/rits_t0_${SCENE}_${ARM}_full_01"
    test ! -e "$OUT" || { echo "SKIP existing $OUT"; continue; }
    TMPDIR="$NAS_ROOT/tmp" TMP="$NAS_ROOT/tmp" TEMP="$NAS_ROOT/tmp" \
    TORCH_EXTENSIONS_DIR="$NAS_ROOT/torch_extensions" PYTHONDONTWRITEBYTECODE=1 \
    SVSR_SOURCE_REVISION="$SOURCE_REVISION" CUDA_VISIBLE_DEVICES=0 \
    python rits_t0_train.py \
      -s "$DATA" -m "$SH" --eval \
      --t0_scene "$SCENE" --t0_arm "$ARM" --t0_output "$OUT"
  done
done
```

## Locked decision

```bash
python rits_t0_decide.py \
  --garden_unsplit "$NAS_ROOT/experiments/rits_t0_garden_unsplit_full_01" \
  --garden_abrupt  "$NAS_ROOT/experiments/rits_t0_garden_abrupt_full_01" \
  --garden_rits    "$NAS_ROOT/experiments/rits_t0_garden_rits_full_01" \
  --room_unsplit   "$NAS_ROOT/experiments/rits_t0_room_unsplit_full_01" \
  --room_abrupt    "$NAS_ROOT/experiments/rits_t0_room_abrupt_full_01" \
  --room_rits      "$NAS_ROOT/experiments/rits_t0_room_rits_full_01"
```

The rule is locked in `experiments/rits_t0/protocol.md`. `pass: true` requires
every check; on failure the topology branch closes (final, no T1).

## Retrying an arm

Output paths are reserved write-once. An attempt that fails partway leaves its
reserved directory behind (holding only `t0_manifest.json`, with no
`results.json` and no `DONE`), so a retry must use a new suffix — `_02`, then
`_03` — and the failed directory is left in place as a record of the attempt.
Point the decision command at whichever directory carries `DONE`.
