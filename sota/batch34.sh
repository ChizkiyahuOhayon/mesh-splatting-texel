#!/usr/bin/env bash
# SoftTail v2 negative control: shuffled per-vertex endpoints.
#
# Same configuration as batch33 on Room and Bicycle, except every floor update
# randomly permutes the per-vertex endpoints across vertices. The endpoint
# multiset is preserved; only its assignment to the visibility statistic is
# destroyed. Run only after batch33 wrote gate.json.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/ensure_environment.sh"

NAS_ROOT=${NAS_ROOT:-/home/smbu/dy/nas/meshsplatting_smbu}
DATA_ROOT=${DATA_ROOT:-/home/smbu/dy/nas/dy/mesh-splatting/data/mipnerf360}
GATE_ROOT=${GATE_ROOT:-$NAS_ROOT/experiments/softtail_v2_01}
RUNS=${RUNS:-$NAS_ROOT/experiments/softtail_v2_control_01}
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU DATA_ROOT RUNS

ADAPTIVE_LOW=${ADAPTIVE_LOW:-0.6}
SCENES=(room bicycle)

REPO="$(cd "$HERE/.." && pwd)"
test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" || {
  echo "tracked source changes are present in $REPO" >&2
  exit 1
}
test -f "$GATE_ROOT/gate.json" || { echo "run batch33 first: $GATE_ROOT/gate.json is missing" >&2; exit 1; }

images_for() {
  case "$1" in
    bicycle) echo images_4 ;;
    room)    echo images_2 ;;
  esac
}

mkdir -p "$RUNS"
[ -f "$RUNS/DONE" ] && { echo "SoftTail v2 control already complete: $RUNS"; exit 0; }

for SCENE in "${SCENES[@]}"; do
  "$HERE/run.sh" adaptive_shuffle "$SCENE" \
    --final_opacity 0.8 --adaptive_opacity --adaptive_opacity_low "$ADAPTIVE_LOW" \
    --adaptive_opacity_control shuffle
done

for SCENE in "${SCENES[@]}"; do
  OUTPUT="$RUNS/$SCENE/adaptive_quality"
  [ -f "$OUTPUT/DONE" ] && { echo "== adaptive_quality/$SCENE already evaluated"; continue; }
  "$MESH_SPLATTING_PYTHON" -u -m sota.main_table_eval \
    -s "$DATA_ROOT/$SCENE" -m "$RUNS/adaptive_shuffle__${SCENE}" -i "$(images_for "$SCENE")" --eval \
    --scene "$SCENE" --arm adaptive_quality --iteration 30000 --output "$OUTPUT"
done

"$MESH_SPLATTING_PYTHON" -m sota.v2_gate --control "$GATE_ROOT" "$RUNS"

echo "SoftTail v2 shuffled-endpoint control complete: $RUNS/control.json"
