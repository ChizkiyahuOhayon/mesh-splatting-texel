#!/usr/bin/env bash
# Train and evaluate the frozen SoftTail configuration on Deep Blending.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/ensure_environment.sh"

NAS_ROOT=${NAS_ROOT:-/home/smbu/dy/nas/meshsplatting_smbu}
DATA_ROOT=${DATA_ROOT:-/home/smbu/dy/nas/dy/mesh-splatting/data/tandt/db}
RUNS=${RUNS:-$NAS_ROOT/experiments/softtail_deep_blending_01}
ROOT=${ROOT:-$NAS_ROOT/experiments/formal_deep_blending_01}
GPU=${GPU:-0}
SCENES=(drjohnson playroom)
export CUDA_VISIBLE_DEVICES=$GPU DATA_ROOT RUNS

[ -f "$ROOT/DONE" ] && {
  echo "Formal Deep Blending experiment already complete: $ROOT"
  cat "$ROOT/formal_table.json"
  exit 0
}

REPO="$(cd "$HERE/.." && pwd)"
test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" || {
  echo "tracked source changes are present in $REPO" >&2
  exit 1
}

for SCENE in "${SCENES[@]}"; do
  test -d "$DATA_ROOT/$SCENE/images" || {
    echo "missing Deep Blending images: $DATA_ROOT/$SCENE/images" >&2
    exit 1
  }
  test -d "$DATA_ROOT/$SCENE/sparse/0" || {
    echo "missing Deep Blending COLMAP model: $DATA_ROOT/$SCENE/sparse/0" >&2
    exit 1
  }
done

mkdir -p "$ROOT"
REVISION=$(git -C "$REPO" rev-parse HEAD)
if [ -f "$ROOT/source_revision.txt" ]; then
  test "$(tr -d '[:space:]' < "$ROOT/source_revision.txt")" = "$REVISION" || {
    echo "source revision differs from the existing Deep Blending run" >&2
    exit 1
  }
else
  printf '%s\n' "$REVISION" > "$ROOT/source_revision.txt"
fi

evaluate() {
  local scene=$1 arm=$2 model=$3
  local output="$ROOT/$scene/$arm"
  [ -f "$output/DONE" ] && { echo "== $arm/$scene already evaluated"; return; }
  "$MESH_SPLATTING_PYTHON" -u -m sota.main_table_eval \
    -s "$DATA_ROOT/$scene" -m "$model" -i images --eval \
    --scene "$scene" --arm "$arm" --iteration 30000 --output "$output"
}

check_floor() {
  "$MESH_SPLATTING_PYTHON" -c '
import sys
import torch
state = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
actual = float(state.get("opacity_floor", 0.9999))
expected = float(sys.argv[2])
if actual != expected:
    raise SystemExit(f"checkpoint opacity_floor is {actual}, expected {expected}")
' "$1/point_cloud/iteration_30000/point_cloud_state_dict.pt" "$2"
}

for SCENE in "${SCENES[@]}"; do
  "$HERE/run.sh" stock "$SCENE"
  "$HERE/run.sh" opacity08 "$SCENE" --final_opacity 0.8

  for ARM in stock opacity08; do
    MODEL="$RUNS/${ARM}__${SCENE}"
    test "$(tr -d '[:space:]' < "$MODEL/source_revision.txt")" = "$REVISION"
  done
  check_floor "$RUNS/stock__${SCENE}" 0.9999
  check_floor "$RUNS/opacity08__${SCENE}" 0.8

  evaluate "$SCENE" stock "$RUNS/stock__${SCENE}"
  evaluate "$SCENE" ours_speed "$RUNS/opacity08__${SCENE}"
  evaluate "$SCENE" ours_quality "$RUNS/opacity08__${SCENE}"
done

"$MESH_SPLATTING_PYTHON" -m sota.deep_blending_table "$ROOT"
