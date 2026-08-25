#!/usr/bin/env bash
# Run the trained opacity sensitivity study, then export paper-facing images.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/ensure_environment.sh"

NAS_ROOT=${NAS_ROOT:-/home/smbu/dy/nas/meshsplatting_smbu}
DATA_ROOT=${DATA_ROOT:-/home/smbu/dy/nas/dy/mesh-splatting/data/mipnerf360}
RUNS=${RUNS:-$NAS_ROOT/experiments/opacity_floor_01}
FORMAL_ROOT=${FORMAL_ROOT:-$NAS_ROOT/experiments/formal_main_table_01}
OPACITY_ROOT=${OPACITY_ROOT:-$NAS_ROOT/experiments/formal_opacity_ablation_01}
SENSITIVITY_ROOT=${SENSITIVITY_ROOT:-$NAS_ROOT/experiments/opacity_sensitivity_01}
QUALITATIVE_ROOT=${QUALITATIVE_ROOT:-$NAS_ROOT/experiments/qualitative_01}
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU DATA_ROOT RUNS

REPO="$(cd "$HERE/.." && pwd)"
test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" || {
  echo "tracked source changes are present in $REPO" >&2
  exit 1
}
test -f "$FORMAL_ROOT/formal_table.json"
test -f "$OPACITY_ROOT/ablation_table.json"

images_for() {
  case "$1" in
    bicycle|flowers|garden|stump|treehill) echo images_4 ;;
    room|counter|kitchen|bonsai)           echo images_2 ;;
  esac
}

stock_model() {
  case "$1" in
    room)    echo "$NAS_ROOT/experiments/sac_g1_room_stock_seed0_train" ;;
    bicycle) echo "$RUNS/stock__bicycle" ;;
    flowers) echo "$RUNS/stock__flowers" ;;
  esac
}

if [ ! -f "$SENSITIVITY_ROOT/DONE" ]; then
  for FLOOR in 07 09; do
    for SCENE in bicycle room; do
      "$HERE/run.sh" "opacity${FLOOR}" "$SCENE" \
        --final_opacity "0.${FLOOR#0}"
    done
  done

  for SCENE in bicycle room; do
    for FLOOR in 07 09; do
      ARM="ours_opacity${FLOOR}"
      OUTPUT="$SENSITIVITY_ROOT/$SCENE/$ARM"
      [ -f "$OUTPUT/DONE" ] && { echo "== $ARM/$SCENE already evaluated"; continue; }
      "$MESH_SPLATTING_PYTHON" -u -m sota.main_table_eval \
        -s "$DATA_ROOT/$SCENE" -m "$RUNS/opacity${FLOOR}__${SCENE}" \
        -i "$(images_for "$SCENE")" --eval --scene "$SCENE" \
        --arm "$ARM" --iteration 30000 --output "$OUTPUT"
    done
  done

  "$MESH_SPLATTING_PYTHON" -m sota.opacity_sensitivity \
    "$FORMAL_ROOT" "$OPACITY_ROOT" "$SENSITIVITY_ROOT"
fi

if [ ! -f "$QUALITATIVE_ROOT/DONE" ]; then
  for SCENE in bicycle flowers room; do
    for ARM in stock ours_quality; do
      if [ "$ARM" = stock ]; then
        MODEL=$(stock_model "$SCENE")
      else
        MODEL="$RUNS/opacity08__${SCENE}"
      fi
      "$MESH_SPLATTING_PYTHON" -u -m sota.qualitative \
        -s "$DATA_ROOT/$SCENE" -m "$MODEL" -i "$(images_for "$SCENE")" \
        --eval --scene "$SCENE" --arm "$ARM" --iteration 30000 \
        --output "$QUALITATIVE_ROOT/$SCENE/$ARM"
    done
  done
  printf 'complete\n' > "$QUALITATIVE_ROOT/DONE"
fi

echo "Overnight sensitivity and qualitative export complete."
