#!/usr/bin/env bash
# Supplementary qualitative exports: every scene not yet exported, three arms
# (MeshSplatting, + opacity 0.8, SoftTail) on the same test views, into a new
# root. Uses sota/qualitative.py exactly as the paper figure did.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/ensure_environment.sh"
PY=$MESH_SPLATTING_PYTHON
LOCAL=${LOCAL:-/home/smbu/dy/data_local}
E=/home/smbu/dy/nas/meshsplatting_smbu/experiments
OUT=${OUT:-$E/qualitative_02}
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU
cd "$HERE/.."
root_for() { case "$1" in train|truck) echo "$LOCAL/tandt";; drjohnson|playroom) echo "$LOCAL/db";; *) echo "$LOCAL/mipnerf360";; esac; }
images_for() { case "$1" in bicycle|flowers|garden|stump|treehill) echo images_4;; room|counter|kitchen|bonsai) echo images_2;; *) echo images;; esac; }
model_for() {  # model_for <scene> <arm>
  local s=$1
  case "$s:$2" in
    train:stock|truck:stock)         echo "$E/softtail_tandt_01/stock__$s" ;;
    train:v1|truck:v1)               echo "$E/softtail_tandt_01/opacity08__$s" ;;
    drjohnson:stock|playroom:stock)  echo "$E/softtail_deep_blending_01/stock__$s" ;;
    drjohnson:v1|playroom:v1)        echo "$E/softtail_deep_blending_01/opacity08__$s" ;;
    train:full|truck:full|drjohnson:full|playroom:full) echo "$E/softtail_tandt_db_01/cut__$s/oats" ;;
    *:stock) echo "$E/opacity_floor_01/stock__$s" ;;
    *:v1)    echo "$E/opacity_floor_01/opacity08__$s" ;;
    *:full)  echo "$E/softtail_integrated_01/cut__$s/oats" ;;
  esac
}
for SCENE in garden stump treehill counter kitchen bonsai train drjohnson; do
  for ARM in stock v1 full; do
    OUTPUT="$OUT/$SCENE/$([ $ARM = full ] && echo softtail_full || { [ $ARM = v1 ] && echo ours_quality || echo stock; })"
    [ -f "$OUTPUT/DONE" ] && continue
    "$PY" -u -m sota.qualitative -s "$(root_for "$SCENE")/$SCENE" -m "$(model_for "$SCENE" "$ARM")" \
      -i "$(images_for "$SCENE")" --eval --scene "$SCENE" \
      --arm "$([ $ARM = stock ] && echo stock || echo ours_quality)" --iteration 30000 --output "$OUTPUT"
  done
done
touch "$OUT/DONE"
