#!/usr/bin/env bash
# Training-time OATS three-scene A/B: prune and densify by integrated contribution.
#
# The arm is the frozen v1 configuration plus --integrated_importance, so the
# frozen v1 runs in formal_main_table_01 are the matched control and no baseline
# is retrained. The statistic changes which faces the 4k-11k pruning deletes and
# which faces densification splits, never how many. No kernel change: the
# accumulator the OATS cleanup already uses is simply handed to the training
# render. Single GPU, resumable at every step.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/ensure_environment.sh"

NAS_ROOT=${NAS_ROOT:-/home/smbu/dy/nas/meshsplatting_smbu}
DATA_ROOT=${DATA_ROOT:-/home/smbu/dy/nas/dy/mesh-splatting/data/mipnerf360}
FORMAL_ROOT=${FORMAL_ROOT:-$NAS_ROOT/experiments/formal_main_table_01}
OPACITY_ROOT=${OPACITY_ROOT:-$NAS_ROOT/experiments/formal_opacity_ablation_01}
RUNS=${RUNS:-$NAS_ROOT/experiments/softtail_integrated_01}
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
[ -f "$RUNS/DONE" ] && { echo "Training-time OATS gate already complete: $RUNS"; exit 0; }

# 1. Training: one configuration for every scene.
for SCENE in "${SCENES[@]}"; do
  "$HERE/run.sh" integrated "$SCENE" --final_opacity 0.8 --integrated_importance --save_precleanup
done

# The arm differs from v1 only in which faces survived training, so the flag is
# not read at render time. It is asserted anyway: a missing flag means the run
# trained the published rule and the whole comparison would be a null.
for SCENE in "${SCENES[@]}"; do
  "$MESH_SPLATTING_PYTHON" -c '
import sys, torch
state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert float(state["opacity_floor"]) == 0.8, state["opacity_floor"]
assert state.get("opacity_floor_vertex") is None, "this arm must not carry a per-vertex floor"
assert state.get("integrated_importance") is True, "checkpoint did not train the integrated statistic"
assert state.get("elastic_window") is not True, "elastic window must be off in this arm"
print(sys.argv[1], "floor:", float(state["opacity_floor"]), "vertices:", state["vertex_weight"].shape[0])
  ' "$RUNS/integrated__${SCENE}/point_cloud/iteration_30000/point_cloud_state_dict.pt"
done

# 2. Evaluation with the frozen v1 evaluator and arms.
for SCENE in "${SCENES[@]}"; do
  for ARM in ours_quality ours_speed ours_opacity; do
    OUTPUT="$RUNS/$SCENE/$ARM"
    [ -f "$OUTPUT/DONE" ] && { echo "== $ARM/$SCENE already evaluated"; continue; }
    "$MESH_SPLATTING_PYTHON" -u -m sota.main_table_eval \
      -s "$DATA_ROOT/$SCENE" -m "$RUNS/integrated__${SCENE}" -i "$(images_for "$SCENE")" --eval \
      --scene "$SCENE" --arm "$ARM" --iteration 30000 --output "$OUTPUT"
  done
done

"$MESH_SPLATTING_PYTHON" -m sota.v3_gate \
  "$FORMAL_ROOT/formal_table.json" "$OPACITY_ROOT/ablation_table.json" "$RUNS" "${SCENES[@]}" \
  --experiment softtail-integrated-importance

touch "$RUNS/DONE"
echo "Training-time OATS three-scene A/B complete: $RUNS/gate.json"
