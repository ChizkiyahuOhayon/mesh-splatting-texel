#!/usr/bin/env bash
# Train the missing opacity-0.8 arms for the formal nine-scene table.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/ensure_environment.sh"

NAS_ROOT=${NAS_ROOT:-/home/smbu/dy/nas/meshsplatting_smbu}
DATA_ROOT=${DATA_ROOT:-/home/smbu/dy/nas/dy/mesh-splatting/data/mipnerf360}
STOCK_ROOT=${STOCK_ROOT:-/home/smbu/dy/nas/dy/mesh-splatting/output/mipnerf360}
RUNS=${RUNS:-$NAS_ROOT/experiments/opacity_floor_01}
export DATA_ROOT RUNS

REPO="$(cd "$HERE/.." && pwd)"
test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" || {
  echo "tracked source changes are present in $REPO" >&2
  exit 1
}

SCENES=(bicycle flowers garden stump treehill room counter kitchen bonsai)
MISSING_METHOD_SCENES=(flowers stump treehill counter kitchen bonsai)

stock_model() {
  case "$1" in
    garden|room) echo "$NAS_ROOT/experiments/sac_g1_${1}_stock_seed0_train" ;;
    bicycle)     echo "$RUNS/stock__bicycle" ;;
    *)           echo "$STOCK_ROOT/$1" ;;
  esac
}

for SCENE in "${SCENES[@]}"; do
  test -d "$DATA_ROOT/$SCENE" || {
    echo "missing dataset: $DATA_ROOT/$SCENE" >&2
    exit 1
  }
  STOCK_MODEL=$(stock_model "$SCENE")
  test -f "$STOCK_MODEL/point_cloud/iteration_30000/point_cloud_state_dict.pt" || {
    echo "missing stock checkpoint: $STOCK_MODEL" >&2
    exit 1
  }
done

for SCENE in bicycle garden room; do
  test -f "$RUNS/opacity08__${SCENE}/point_cloud/iteration_30000/point_cloud_state_dict.pt" || {
    echo "missing completed method checkpoint: $RUNS/opacity08__${SCENE}" >&2
    exit 1
  }
done

for SCENE in "${MISSING_METHOD_SCENES[@]}"; do
  "$HERE/run.sh" opacity08 "$SCENE" --final_opacity 0.8
done

for SCENE in "${SCENES[@]}"; do
  CHECKPOINT="$RUNS/opacity08__${SCENE}/point_cloud/iteration_30000/point_cloud_state_dict.pt"
  "$MESH_SPLATTING_PYTHON" -c '
import sys
import torch
state = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
value = float(state["opacity_floor"])
if value != 0.8:
    raise SystemExit(f"checkpoint opacity_floor is {value}, expected 0.8")
print(sys.argv[1], "opacity_floor:", value)
' "$CHECKPOINT"
done

echo "Formal opacity-0.8 training complete: $RUNS"
