#!/usr/bin/env bash
# Train and evaluate the frozen SoftTail configuration on Tanks & Temples.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/ensure_environment.sh"

NAS_ROOT=${NAS_ROOT:-/home/smbu/dy/nas/meshsplatting_smbu}
DATA_ROOT=${DATA_ROOT:-/home/smbu/dy/nas/dy/mesh-splatting/data/tandt}
RUNS=${RUNS:-$NAS_ROOT/experiments/softtail_tandt_01}
ROOT=${ROOT:-$NAS_ROOT/experiments/formal_tandt_01}
GPU=${GPU:-0}
SCENES=(train truck)
export CUDA_VISIBLE_DEVICES=$GPU DATA_ROOT RUNS

[ -f "$ROOT/DONE" ] && {
  echo "Formal Tanks & Temples experiment already complete: $ROOT"
  exit 0
}

REPO="$(cd "$HERE/.." && pwd)"
test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" || {
  echo "tracked source changes are present in $REPO" >&2
  exit 1
}

for SCENE in "${SCENES[@]}"; do
  test -d "$DATA_ROOT/$SCENE/images" || {
    echo "missing T&T images: $DATA_ROOT/$SCENE/images" >&2
    exit 1
  }
  test -d "$DATA_ROOT/$SCENE/sparse/0" || {
    echo "missing T&T COLMAP model: $DATA_ROOT/$SCENE/sparse/0" >&2
    exit 1
  }
done

evaluate() {
  local scene=$1 arm=$2 model=$3
  local output="$ROOT/$scene/$arm"
  [ -f "$output/DONE" ] && { echo "== $arm/$scene already evaluated"; return; }
  "$MESH_SPLATTING_PYTHON" -u -m sota.main_table_eval \
    -s "$DATA_ROOT/$scene" -m "$model" -i images --eval \
    --scene "$scene" --arm "$arm" --iteration 30000 --output "$output"
}

for SCENE in "${SCENES[@]}"; do
  "$HERE/run.sh" stock "$SCENE"
  "$HERE/run.sh" opacity08 "$SCENE" --final_opacity 0.8

  evaluate "$SCENE" stock "$RUNS/stock__${SCENE}"
  evaluate "$SCENE" ours_speed "$RUNS/opacity08__${SCENE}"
  evaluate "$SCENE" ours_quality "$RUNS/opacity08__${SCENE}"
done

"$MESH_SPLATTING_PYTHON" -m sota.tandt_table "$ROOT"
