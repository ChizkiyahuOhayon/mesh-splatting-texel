#!/usr/bin/env bash
# OATS three-scene A/B: integrated survival versus the v1 cleanup rule.
#
# Each scene trains once in the frozen v1 configuration and saves its state
# just before the final cleanup. Both rules are then applied to that one state
# at the same face budget (sota/survival_cleanup.py), so the paired gate below
# compares two cuts of the same trained model and isolates the rule exactly.
# Single GPU, resumable at every step.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/ensure_environment.sh"

NAS_ROOT=${NAS_ROOT:-/home/smbu/dy/nas/meshsplatting_smbu}
DATA_ROOT=${DATA_ROOT:-/home/smbu/dy/nas/dy/mesh-splatting/data/mipnerf360}
FORMAL_ROOT=${FORMAL_ROOT:-$NAS_ROOT/experiments/formal_main_table_01}
OPACITY_ROOT=${OPACITY_ROOT:-$NAS_ROOT/experiments/formal_opacity_ablation_01}
RUNS=${RUNS:-$NAS_ROOT/experiments/softtail_oats_01}
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU DATA_ROOT RUNS

SCENES=(room bicycle garden)

REPO="$(cd "$HERE/.." && pwd)"
test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" || {
  echo "tracked source changes are present in $REPO" >&2
  exit 1
}

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
[ -f "$RUNS/DONE" ] && { echo "OATS gate already complete: $RUNS"; exit 0; }

# 1. Training: the v1 configuration, plus the pre-cleanup state.
for SCENE in "${SCENES[@]}"; do
  "$HERE/run.sh" base "$SCENE" --final_opacity 0.8 --save_precleanup
done

# 2. Both cleanup rules on each pre-cleanup state, at one face budget.
for SCENE in "${SCENES[@]}"; do
  CUT="$RUNS/cut__${SCENE}"
  [ -f "$CUT/survival.json" ] && { echo "== $SCENE already cut"; continue; }
  rm -rf "$CUT"
  "$MESH_SPLATTING_PYTHON" -u -m sota.survival_cleanup \
    -s "$DATA_ROOT/$SCENE" -m "$RUNS/base__${SCENE}" -i "$(images_for "$SCENE")" --eval \
    --out "$CUT"
done

# 3. Evaluation of both cuts with the frozen v1 evaluator and arms.
for SCENE in "${SCENES[@]}"; do
  for RULE in v1 oats; do
    for ARM in ours_quality ours_speed ours_opacity; do
      OUTPUT="$RUNS/eval_${RULE}/$SCENE/$ARM"
      [ -f "$OUTPUT/DONE" ] && { echo "== $RULE/$ARM/$SCENE already evaluated"; continue; }
      "$MESH_SPLATTING_PYTHON" -u -m sota.main_table_eval \
        -s "$DATA_ROOT/$SCENE" -m "$RUNS/cut__${SCENE}/${RULE}" -i "$(images_for "$SCENE")" --eval \
        --scene "$SCENE" --arm "$ARM" --iteration 30000 --output "$OUTPUT"
    done
  done
done

"$MESH_SPLATTING_PYTHON" -m sota.v3_gate \
  "$FORMAL_ROOT/formal_table.json" "$OPACITY_ROOT/ablation_table.json" "$RUNS/eval_oats" "${SCENES[@]}" \
  --experiment softtail-oats-integrated-survival --paired-reference "$RUNS/eval_v1"
cp "$RUNS/eval_oats/gate.json" "$RUNS/gate.json"

touch "$RUNS/DONE"
echo "OATS three-scene A/B complete: $RUNS/gate.json"
