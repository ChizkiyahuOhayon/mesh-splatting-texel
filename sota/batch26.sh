#!/usr/bin/env bash
# Evaluate the terminal-opacity change without inference modifications.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/ensure_environment.sh"

NAS_ROOT=${NAS_ROOT:-/home/smbu/dy/nas/meshsplatting_smbu}
DATA_ROOT=${DATA_ROOT:-/home/smbu/dy/nas/dy/mesh-splatting/data/mipnerf360}
RUNS=${RUNS:-$NAS_ROOT/experiments/opacity_floor_01}
FORMAL_ROOT=${FORMAL_ROOT:-$NAS_ROOT/experiments/formal_main_table_01}
ROOT=${ROOT:-$NAS_ROOT/experiments/formal_opacity_ablation_01}
GPU=${GPU:-0}

[ -f "$ROOT/DONE" ] && {
  echo "Formal opacity ablation already complete: $ROOT"
  exit 0
}
test -f "$FORMAL_ROOT/formal_table.json"

REPO="$(cd "$HERE/.." && pwd)"
test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" || {
  echo "tracked source changes are present in $REPO" >&2
  exit 1
}

SCENES=(bicycle flowers garden stump treehill room counter kitchen bonsai)

images_for() {
  case "$1" in
    bicycle|flowers|garden|stump|treehill) echo images_4 ;;
    room|counter|kitchen|bonsai)           echo images_2 ;;
  esac
}

export CUDA_VISIBLE_DEVICES=$GPU
for SCENE in "${SCENES[@]}"; do
  OUTPUT="$ROOT/$SCENE/ours_opacity"
  [ -f "$OUTPUT/DONE" ] && { echo "== ours_opacity/$SCENE already evaluated"; continue; }
  "$MESH_SPLATTING_PYTHON" -u -m sota.main_table_eval \
    -s "$DATA_ROOT/$SCENE" -m "$RUNS/opacity08__${SCENE}" \
    -i "$(images_for "$SCENE")" --eval --scene "$SCENE" \
    --arm ours_opacity --iteration 30000 --output "$OUTPUT"
done

"$MESH_SPLATTING_PYTHON" -m sota.opacity_ablation "$FORMAL_ROOT" "$ROOT"
