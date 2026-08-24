#!/usr/bin/env bash
# Train missing matched stock and opacity-0.8 arms for the formal table.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/ensure_environment.sh"

NAS_ROOT=${NAS_ROOT:-/home/smbu/dy/nas/meshsplatting_smbu}
DATA_ROOT=${DATA_ROOT:-/home/smbu/dy/nas/dy/mesh-splatting/data/mipnerf360}
RUNS=${RUNS:-$NAS_ROOT/experiments/opacity_floor_01}
STOCK_GPU=${STOCK_GPU:-0}
METHOD_GPU=${METHOD_GPU:-1}
export DATA_ROOT RUNS

REPO="$(cd "$HERE/.." && pwd)"
test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" || {
  echo "tracked source changes are present in $REPO" >&2
  exit 1
}

SCENES=(bicycle flowers garden stump treehill room counter kitchen bonsai)
MISSING_SCENES=(flowers stump treehill counter kitchen bonsai)

stock_model() {
  case "$1" in
    garden|room) echo "$NAS_ROOT/experiments/sac_g1_${1}_stock_seed0_train" ;;
    bicycle)     echo "$RUNS/stock__bicycle" ;;
    *)           echo "$RUNS/stock__${1}" ;;
  esac
}

for SCENE in "${SCENES[@]}"; do
  test -d "$DATA_ROOT/$SCENE" || {
    echo "missing dataset: $DATA_ROOT/$SCENE" >&2
    exit 1
  }
done

for SCENE in bicycle garden room; do
  STOCK_MODEL=$(stock_model "$SCENE")
  test -f "$STOCK_MODEL/point_cloud/iteration_30000/point_cloud_state_dict.pt" || {
    echo "missing completed stock checkpoint: $STOCK_MODEL" >&2
    exit 1
  }
  test -f "$RUNS/opacity08__${SCENE}/point_cloud/iteration_30000/point_cloud_state_dict.pt" || {
    echo "missing completed method checkpoint: $RUNS/opacity08__${SCENE}" >&2
    exit 1
  }
done

(
  export CUDA_VISIBLE_DEVICES=$STOCK_GPU
  for SCENE in "${MISSING_SCENES[@]}"; do
    "$HERE/run.sh" stock "$SCENE"
  done
) &
STOCK_PID=$!

(
  export CUDA_VISIBLE_DEVICES=$METHOD_GPU
  for SCENE in "${MISSING_SCENES[@]}"; do
    "$HERE/run.sh" opacity08 "$SCENE" --final_opacity 0.8
  done
) &
METHOD_PID=$!

STATUS=0
wait "$STOCK_PID" || STATUS=1
wait "$METHOD_PID" || STATUS=1
test "$STATUS" -eq 0 || exit "$STATUS"

for SCENE in "${SCENES[@]}"; do
  for ARM in stock opacity08; do
    if [ "$ARM" = stock ]; then
      MODEL=$(stock_model "$SCENE")
      EXPECTED=0.9999
    else
      MODEL="$RUNS/opacity08__${SCENE}"
      EXPECTED=0.8
    fi
    CHECKPOINT="$MODEL/point_cloud/iteration_30000/point_cloud_state_dict.pt"
    "$MESH_SPLATTING_PYTHON" -c '
import sys
import torch
state = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
value = float(state.get("opacity_floor", 0.9999))
expected = float(sys.argv[2])
if value != expected:
    raise SystemExit(f"checkpoint opacity_floor is {value}, expected {expected}")
print(sys.argv[1], "opacity_floor:", value)
  ' "$CHECKPOINT" "$EXPECTED"
  done
done

echo "Formal matched stock/opacity-0.8 training complete: $RUNS"
