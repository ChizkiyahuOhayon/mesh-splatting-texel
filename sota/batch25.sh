#!/usr/bin/env bash
# Evaluate the formal nine-scene stock, speed, and quality operating points.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/ensure_environment.sh"

NAS_ROOT=${NAS_ROOT:-/home/smbu/dy/nas/meshsplatting_smbu}
DATA_ROOT=${DATA_ROOT:-/home/smbu/dy/nas/dy/mesh-splatting/data/mipnerf360}
RUNS=${RUNS:-$NAS_ROOT/experiments/opacity_floor_01}
ROOT=${ROOT:-$NAS_ROOT/experiments/formal_main_table_01}
STOCK_GPU=${STOCK_GPU:-0}
METHOD_GPU=${METHOD_GPU:-1}

[ -f "$ROOT/DONE" ] && {
  echo "Formal nine-scene evaluation already complete: $ROOT"
  exit 0
}

REPO="$(cd "$HERE/.." && pwd)"
test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" || {
  echo "tracked source changes are present in $REPO" >&2
  exit 1
}

SCENES=(bicycle flowers garden stump treehill room counter kitchen bonsai)

stock_model() {
  case "$1" in
    garden|room) echo "$NAS_ROOT/experiments/sac_g1_${1}_stock_seed0_train" ;;
    bicycle)     echo "$RUNS/stock__bicycle" ;;
    *)           echo "$RUNS/stock__${1}" ;;
  esac
}

images_for() {
  case "$1" in
    bicycle|flowers|garden|stump|treehill) echo images_4 ;;
    room|counter|kitchen|bonsai)           echo images_2 ;;
  esac
}

evaluate() {
  local scene=$1 arm=$2 model=$3
  local output="$ROOT/$scene/$arm"
  [ -f "$output/DONE" ] && { echo "== $arm/$scene already evaluated"; return; }
  "$MESH_SPLATTING_PYTHON" -u -m sota.main_table_eval \
    -s "$DATA_ROOT/$scene" -m "$model" -i "$(images_for "$scene")" --eval \
    --scene "$scene" --arm "$arm" --iteration 30000 --output "$output"
}

evaluate_stock() {
  export CUDA_VISIBLE_DEVICES=$STOCK_GPU
  for SCENE in "${SCENES[@]}"; do
    evaluate "$SCENE" stock "$(stock_model "$SCENE")"
  done
}

evaluate_method() {
  export CUDA_VISIBLE_DEVICES=$METHOD_GPU
  for SCENE in "${SCENES[@]}"; do
    MODEL="$RUNS/opacity08__${SCENE}"
    evaluate "$SCENE" ours_speed "$MODEL"
    evaluate "$SCENE" ours_quality "$MODEL"
  done
}

if [ "$STOCK_GPU" = "$METHOD_GPU" ]; then
  evaluate_stock
  evaluate_method
else
  evaluate_stock &
  STOCK_PID=$!
  evaluate_method &
  METHOD_PID=$!

  STATUS=0
  wait "$STOCK_PID" || STATUS=1
  wait "$METHOD_PID" || STATUS=1
  test "$STATUS" -eq 0 || exit "$STATUS"
fi

"$MESH_SPLATTING_PYTHON" -m sota.formal_table "$ROOT"
