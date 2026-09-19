#!/usr/bin/env bash
# SoftTail v4 three-scene A/B: the per-vertex opacity field.
#
# The field arm is the frozen v1 configuration plus --opacity_field, so the
# frozen v1 runs in formal_main_table_01 are the matched control and no baseline
# is retrained. Single GPU, resumable at every step.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/ensure_environment.sh"

NAS_ROOT=${NAS_ROOT:-/home/smbu/dy/nas/meshsplatting_smbu}
DATA_ROOT=${DATA_ROOT:-/home/smbu/dy/nas/dy/mesh-splatting/data/mipnerf360}
FORMAL_ROOT=${FORMAL_ROOT:-$NAS_ROOT/experiments/formal_main_table_01}
OPACITY_ROOT=${OPACITY_ROOT:-$NAS_ROOT/experiments/formal_opacity_ablation_01}
RUNS=${RUNS:-$NAS_ROOT/experiments/softtail_v4_02}
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU DATA_ROOT RUNS

SCENES=(room bicycle garden)

REPO="$(cd "$HERE/.." && pwd)"
test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" || {
  echo "tracked source changes are present in $REPO" >&2
  exit 1
}
test -f "$FORMAL_ROOT/formal_table.json"

images_for() {
  case "$1" in
    bicycle|flowers|garden|stump|treehill) echo images_4 ;;
    room|counter|kitchen|bonsai)           echo images_2 ;;
  esac
}

for SCENE in "${SCENES[@]}"; do
  test -d "$DATA_ROOT/$SCENE" || { echo "missing dataset: $DATA_ROOT/$SCENE" >&2; exit 1; }
done

mkdir -p "$RUNS"
[ -f "$RUNS/DONE" ] && { echo "SoftTail v4 gate already complete: $RUNS"; exit 0; }

# 1. Training: one configuration for every scene.
for SCENE in "${SCENES[@]}"; do
  "$HERE/run.sh" field "$SCENE" --final_opacity 0.8 --opacity_field
done

# The field checkpoint is a v1 checkpoint plus one flag: the parameters are the
# same per-vertex opacities, only how a face reads them changes. Evaluation reads
# the flag from the checkpoint, so a missing flag would silently score the min.
for SCENE in "${SCENES[@]}"; do
  "$MESH_SPLATTING_PYTHON" -c '
import sys, torch
state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert float(state["opacity_floor"]) == 0.8, state["opacity_floor"]
assert state.get("opacity_floor_vertex") is None, "field arm must not carry a per-vertex floor"
assert state.get("opacity_field") is True, "checkpoint does not record the opacity field"
print(sys.argv[1], "floor:", float(state["opacity_floor"]), "vertices:", state["vertex_weight"].shape[0])
  ' "$RUNS/field__${SCENE}/point_cloud/iteration_30000/point_cloud_state_dict.pt"
done

# 2. Evaluation with the frozen v1 evaluator and arms.
for SCENE in "${SCENES[@]}"; do
  for ARM in ours_quality ours_speed ours_opacity; do
    OUTPUT="$RUNS/$SCENE/$ARM"
    [ -f "$OUTPUT/DONE" ] && { echo "== $ARM/$SCENE already evaluated"; continue; }
    "$MESH_SPLATTING_PYTHON" -u -m sota.main_table_eval \
      -s "$DATA_ROOT/$SCENE" -m "$RUNS/field__${SCENE}" -i "$(images_for "$SCENE")" --eval \
      --scene "$SCENE" --arm "$ARM" --iteration 30000 --output "$OUTPUT"
  done
done

"$MESH_SPLATTING_PYTHON" -m sota.v3_gate \
  "$FORMAL_ROOT/formal_table.json" "$OPACITY_ROOT/ablation_table.json" "$RUNS" "${SCENES[@]}" \
  --experiment softtail-v4-opacity-field

touch "$RUNS/DONE"
echo "SoftTail v4 three-scene A/B complete: $RUNS/gate.json"
