#!/usr/bin/env bash
# Export matched qualitative evidence for the two transfer benchmarks.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/ensure_environment.sh"

NAS_ROOT=${NAS_ROOT:-/home/smbu/dy/nas/meshsplatting_smbu}
TANDT_DATA=${TANDT_DATA:-/home/smbu/dy/nas/dy/mesh-splatting/data/tandt/tandt}
DEEP_DATA=${DEEP_DATA:-/home/smbu/dy/nas/dy/mesh-splatting/data/tandt/db}
TANDT_RUNS=${TANDT_RUNS:-$NAS_ROOT/experiments/softtail_tandt_01}
DEEP_RUNS=${DEEP_RUNS:-$NAS_ROOT/experiments/softtail_deep_blending_01}
ROOT=${ROOT:-$NAS_ROOT/experiments/qualitative_transfer_01}
GPU=${GPU:-0}
SCENES=(train truck drjohnson playroom)
export CUDA_VISIBLE_DEVICES=$GPU

require_dir() {
  [ -d "$1" ] || {
    echo "missing directory: $1" >&2
    exit 1
  }
}

require_file() {
  [ -f "$1" ] || {
    echo "missing file: $1" >&2
    exit 1
  }
}

[ -f "$ROOT/DONE" ] && {
  echo "Transfer qualitative export already complete: $ROOT"
  exit 0
}

REPO="$(cd "$HERE/.." && pwd)"
test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" || {
  echo "tracked source changes are present in $REPO" >&2
  exit 1
}

data_for() {
  case "$1" in
    train|truck)          echo "$TANDT_DATA" ;;
    drjohnson|playroom)  echo "$DEEP_DATA" ;;
  esac
}

runs_for() {
  case "$1" in
    train|truck)          echo "$TANDT_RUNS" ;;
    drjohnson|playroom)  echo "$DEEP_RUNS" ;;
  esac
}

for SCENE in "${SCENES[@]}"; do
  DATA=$(data_for "$SCENE")
  RUNS=$(runs_for "$SCENE")
  require_dir "$DATA/$SCENE/images"
  require_dir "$DATA/$SCENE/sparse/0"

  for ARM in stock ours_quality; do
    if [ "$ARM" = stock ]; then
      MODEL="$RUNS/stock__${SCENE}"
    else
      MODEL="$RUNS/opacity08__${SCENE}"
    fi
    require_file "$MODEL/point_cloud/iteration_30000/point_cloud_state_dict.pt"
    "$MESH_SPLATTING_PYTHON" -u -m sota.qualitative \
      -s "$DATA/$SCENE" -m "$MODEL" -i images --eval \
      --scene "$SCENE" --arm "$ARM" --iteration 30000 \
      --output "$ROOT/$SCENE/$ARM"
  done
done

printf 'complete\n' > "$ROOT/DONE"
echo "Transfer qualitative export complete: $ROOT"
