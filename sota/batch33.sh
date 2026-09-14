#!/usr/bin/env bash
# SoftTail v2 three-scene gate: visibility-aware terminal opacity (VATO).
#
# 1. Premise check on the frozen v1 checkpoints (no training).
# 2. Train adaptive__{room,bicycle,garden} with one shared configuration.
# 3. Evaluate the adaptive quality/speed/opacity-only arms with the same
#    evaluator as the formal v1 tables and write gate.json against them.
#
# Single GPU, resumable at every step. v1 roots are read only.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/ensure_environment.sh"

NAS_ROOT=${NAS_ROOT:-/home/smbu/dy/nas/meshsplatting_smbu}
DATA_ROOT=${DATA_ROOT:-/home/smbu/dy/nas/dy/mesh-splatting/data/mipnerf360}
V1_RUNS=${V1_RUNS:-$NAS_ROOT/experiments/opacity_floor_01}
FORMAL_ROOT=${FORMAL_ROOT:-$NAS_ROOT/experiments/formal_main_table_01}
OPACITY_ROOT=${OPACITY_ROOT:-$NAS_ROOT/experiments/formal_opacity_ablation_01}
RUNS=${RUNS:-$NAS_ROOT/experiments/softtail_v2_01}
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU DATA_ROOT RUNS

ADAPTIVE_LOW=${ADAPTIVE_LOW:-0.6}
SCENES=(room bicycle garden)

REPO="$(cd "$HERE/.." && pwd)"
test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" || {
  echo "tracked source changes are present in $REPO" >&2
  exit 1
}
test -f "$FORMAL_ROOT/formal_table.json"
test -f "$OPACITY_ROOT/ablation_table.json"

images_for() {
  case "$1" in
    bicycle|flowers|garden|stump|treehill) echo images_4 ;;
    room|counter|kitchen|bonsai)           echo images_2 ;;
  esac
}

stock_model() {
  case "$1" in
    garden|room) echo "$NAS_ROOT/experiments/sac_g1_${1}_stock_seed0_train" ;;
    *)           echo "$V1_RUNS/stock__${1}" ;;
  esac
}

for SCENE in "${SCENES[@]}"; do
  test -d "$DATA_ROOT/$SCENE" || { echo "missing dataset: $DATA_ROOT/$SCENE" >&2; exit 1; }
  test -f "$V1_RUNS/opacity08__${SCENE}/point_cloud/iteration_30000/point_cloud_state_dict.pt" || {
    echo "missing frozen v1 checkpoint for $SCENE" >&2; exit 1; }
done

mkdir -p "$RUNS"
[ -f "$RUNS/DONE" ] && { echo "SoftTail v2 gate already complete: $RUNS"; exit 0; }

# 1. Premise check (frozen v1 checkpoints; nothing is trained).
for SCENE in "${SCENES[@]}"; do
  OUTPUT="$RUNS/premise/$SCENE"
  [ -f "$OUTPUT/DONE" ] && { echo "== premise/$SCENE already done"; continue; }
  "$MESH_SPLATTING_PYTHON" -u -m sota.visibility_diag \
    -s "$DATA_ROOT/$SCENE" -m "$V1_RUNS/opacity08__${SCENE}" \
    -i "$(images_for "$SCENE")" --eval --scene "$SCENE" --iteration 30000 \
    --stock_model "$(stock_model "$SCENE")" --views 3 --output "$OUTPUT"
done
"$MESH_SPLATTING_PYTHON" - "$RUNS/premise" "${SCENES[@]}" <<'PY'
import json, sys
root, scenes = sys.argv[1], sys.argv[2:]
aucs = {s: json.load(open(f"{root}/{s}/diag.json"))["auc_low_dominance_predicts_at_floor"] for s in scenes}
print("premise AUC:", aucs)
weak = [s for s, a in aucs.items() if a < 0.65]
if len(weak) >= 2:
    raise SystemExit(f"premise falsified: AUC < 0.65 on {weak}; the training gate is not launched")
PY

# 2. Training: one configuration for every scene.
for SCENE in "${SCENES[@]}"; do
  "$HERE/run.sh" adaptive "$SCENE" \
    --final_opacity 0.8 --adaptive_opacity --adaptive_opacity_low "$ADAPTIVE_LOW"
done

for SCENE in "${SCENES[@]}"; do
  "$MESH_SPLATTING_PYTHON" -c '
import sys, torch
state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
low = float(sys.argv[2])
assert float(state["opacity_floor"]) == 0.8, state["opacity_floor"]
floors = state["opacity_floor_vertex"]
assert state["adaptive_opacity"]["low"] == low and state["adaptive_opacity"]["high"] == 0.8
assert floors.min() >= low - 1e-6 and floors.max() <= 0.8 + 1e-6, (floors.min(), floors.max())
print(sys.argv[1], "vertex floor range:", float(floors.min()), float(floors.max()),
      "mean:", float(floors.mean()), "dominance:", "visibility_dominance" in state)
  ' "$RUNS/adaptive__${SCENE}/point_cloud/iteration_30000/point_cloud_state_dict.pt" "$ADAPTIVE_LOW"
done

# 3. Evaluation with the formal evaluator and the gate.
for SCENE in "${SCENES[@]}"; do
  for ARM in adaptive_quality adaptive_speed adaptive_opacity; do
    OUTPUT="$RUNS/$SCENE/$ARM"
    [ -f "$OUTPUT/DONE" ] && { echo "== $ARM/$SCENE already evaluated"; continue; }
    "$MESH_SPLATTING_PYTHON" -u -m sota.main_table_eval \
      -s "$DATA_ROOT/$SCENE" -m "$RUNS/adaptive__${SCENE}" -i "$(images_for "$SCENE")" --eval \
      --scene "$SCENE" --arm "$ARM" --iteration 30000 --output "$OUTPUT"
  done
done

"$MESH_SPLATTING_PYTHON" -m sota.v2_gate \
  "$FORMAL_ROOT/formal_table.json" "$OPACITY_ROOT/ablation_table.json" "$RUNS"

echo "SoftTail v2 three-scene gate complete: $RUNS/gate.json"
