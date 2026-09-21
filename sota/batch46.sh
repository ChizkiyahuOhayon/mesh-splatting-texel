#!/usr/bin/env bash
# Figure evidence for the full method: qualitative renders and the statistic.
#
# 1. Qualitative. qualitative_01 and qualitative_transfer_01 already hold the
#    ground truth, the stock MeshSplatting renders and the v1 renders for the
#    same test views. This adds the full method beside them, so a figure row is
#    GT / MeshSplatting / SoftTail with pixel-aligned error maps.
#
# 2. The statistic itself. sota/survival_statistic.py re-measures both survival
#    statistics on a pre-cleanup run and samples their joint distribution, which
#    is what the analysis figure plots: the peak and the integral disagree about
#    which faces earn their place, and the figure shows on what kind of face.
#
# Evaluation only, no training. Single GPU, resumable at every step.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/ensure_environment.sh"

NAS_ROOT=${NAS_ROOT:-/home/smbu/dy/nas/meshsplatting_smbu}
M360_DATA=${M360_DATA:-/home/smbu/dy/nas/dy/mesh-splatting/data/mipnerf360}
TANDT_DATA=${TANDT_DATA:-/home/smbu/dy/nas/dy/mesh-splatting/data/tandt/tandt}
DEEP_DATA=${DEEP_DATA:-/home/smbu/dy/nas/dy/mesh-splatting/data/tandt/db}
M360_RUNS=${M360_RUNS:-$NAS_ROOT/experiments/softtail_integrated_01}
TRANSFER_RUNS=${TRANSFER_RUNS:-$NAS_ROOT/experiments/softtail_tandt_db_01}
M360_FIGS=${M360_FIGS:-$NAS_ROOT/experiments/qualitative_01}
TRANSFER_FIGS=${TRANSFER_FIGS:-$NAS_ROOT/experiments/qualitative_transfer_01}
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU

# The scenes whose ground truth and baseline renders are already exported.
M360_SCENES=(bicycle flowers room)
TRANSFER_SCENES=(truck playroom)
# Pre-cleanup runs kept for the statistic figure.
STAT_SCENES=(room bicycle garden)

images_for() {
  case "$1" in
    bicycle|flowers|garden|stump|treehill) echo images_4 ;;
    room|counter|kitchen|bonsai)           echo images_2 ;;
    *)                                     echo images ;;
  esac
}

data_for() {
  case "$1" in
    train|truck)        echo "$TANDT_DATA" ;;
    drjohnson|playroom) echo "$DEEP_DATA" ;;
    *)                  echo "$M360_DATA" ;;
  esac
}

export_full_method() {
  local scene=$1 runs=$2 figs=$3
  local model="$runs/cut__${scene}/oats"
  local output="$figs/$scene/softtail_full"
  test -f "$model/point_cloud/iteration_30000/point_cloud_state_dict.pt" \
    || { echo "missing full-method model for $scene" >&2; exit 1; }
  test -d "$figs/$scene/stock" \
    || { echo "no baseline export to compare against in $figs/$scene" >&2; exit 1; }
  [ -f "$output/DONE" ] && { echo "== qualitative/$scene already exported"; return; }
  "$MESH_SPLATTING_PYTHON" -u -m sota.qualitative \
    -s "$(data_for "$scene")/$scene" -m "$model" -i "$(images_for "$scene")" --eval \
    --scene "$scene" --arm ours_quality --iteration 30000 --output "$output"
}

for SCENE in "${M360_SCENES[@]}"; do
  export_full_method "$SCENE" "$M360_RUNS" "$M360_FIGS"
done

for SCENE in "${TRANSFER_SCENES[@]}"; do
  export_full_method "$SCENE" "$TRANSFER_RUNS" "$TRANSFER_FIGS"
done

for SCENE in "${STAT_SCENES[@]}"; do
  OUT="$M360_RUNS/stat__${SCENE}"
  [ -f "$OUT/statistic.json" ] && { echo "== statistic/$SCENE already measured"; continue; }
  "$MESH_SPLATTING_PYTHON" -u -m sota.survival_statistic \
    -s "$M360_DATA/$SCENE" -m "$M360_RUNS/integrated__${SCENE}" \
    -i "$(images_for "$SCENE")" --eval --out "$OUT"
done

echo "Figure evidence complete:"
echo "  renders   $M360_FIGS/<scene>/softtail_full, $TRANSFER_FIGS/<scene>/softtail_full"
echo "  statistic $M360_RUNS/stat__<scene>/statistic.{npz,json}"
