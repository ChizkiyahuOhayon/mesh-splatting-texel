#!/usr/bin/env bash
# Opaque SoftTail on the ten scenes batch48's gate does not cover: stock
# MeshSplatting (terminal opacity 0.9999) + --integrated_importance + the OATS
# cleanup, scored by the `stock` evaluator arm. Run regardless of the gate so
# the decision can be made on all 13 scenes. Single GPU, resumable.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/ensure_environment.sh"
PY=$MESH_SPLATTING_PYTHON
LOCAL=${LOCAL:-/home/smbu/dy/data_local}
RUNS=${RUNS:-/home/smbu/dy/nas/meshsplatting_smbu/experiments/softtail_opaque_01}
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU RUNS
REPO="$(cd "$HERE/.." && pwd)"
test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" || { echo "dirty tree" >&2; exit 1; }
cd "$REPO"
root_for() { case "$1" in train|truck) echo "$LOCAL/tandt";; drjohnson|playroom) echo "$LOCAL/db";; *) echo "$LOCAL/mipnerf360";; esac; }
images_for() { case "$1" in bicycle|flowers|garden|stump|treehill) echo images_4;; room|counter|kitchen|bonsai) echo images_2;; *) echo images;; esac; }
for SCENE in flowers stump treehill counter kitchen bonsai train truck drjohnson playroom; do
  export DATA_ROOT; DATA_ROOT="$(root_for "$SCENE")"
  "$HERE/run.sh" opaqueint "$SCENE" --integrated_importance --save_precleanup
  CUT="$RUNS/cut__${SCENE}"
  [ -f "$CUT/survival.json" ] || { rm -rf "$CUT"; "$PY" -u -m sota.survival_cleanup -s "$DATA_ROOT/$SCENE" \
    -i "$(images_for "$SCENE")" --eval -m "$RUNS/opaqueint__${SCENE}" --out "$CUT" --quiet; }
  OUTPUT="$CUT/eval_opaque"
  [ -f "$OUTPUT/DONE" ] || { rm -rf "$OUTPUT"; "$PY" -u -m sota.main_table_eval -s "$DATA_ROOT/$SCENE" \
    -i "$(images_for "$SCENE")" --eval -m "$CUT/oats" --scene "$SCENE" --arm stock --iteration 30000 \
    --output "$OUTPUT" --quiet; }
done
touch "$RUNS/ALL_DONE"
