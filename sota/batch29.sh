#!/usr/bin/env bash
# Train and evaluate matched MeshSplatting and SoftTail on DTU geometry.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/ensure_environment.sh"

NAS_ROOT=${NAS_ROOT:-/home/smbu/dy/nas/meshsplatting_smbu}
DATA_ROOT=${DATA_ROOT:-/home/smbu/dy/nas/dy/mesh-splatting/data/dtu}
DTU_EVAL_ROOT=${DTU_EVAL_ROOT:-/home/smbu/dy/nas/dy/mesh-splatting/data/dtu_eval}
RUNS=${RUNS:-$NAS_ROOT/experiments/softtail_dtu_01}
ROOT=${ROOT:-$NAS_ROOT/experiments/formal_dtu_01}
GPU=${GPU:-0}
SCANS=(24 37 40 55 63 65 69 83 97 105 106 110 114 118 122)
export CUDA_VISIBLE_DEVICES=$GPU DATA_ROOT RUNS

[ -f "$ROOT/DONE" ] && {
  echo "Formal DTU experiment already complete: $ROOT"
  exit 0
}

REPO="$(cd "$HERE/.." && pwd)"
test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" || {
  echo "tracked source changes are present in $REPO" >&2
  exit 1
}
mkdir -p "$ROOT"
REVISION=$(git -C "$REPO" rev-parse HEAD)
if [ -f "$ROOT/source_revision.txt" ]; then
  test "$(tr -d '[:space:]' < "$ROOT/source_revision.txt")" = "$REVISION" || {
    echo "source revision differs from the existing DTU experiment" >&2
    exit 1
  }
else
  printf '%s\n' "$REVISION" > "$ROOT/source_revision.txt"
fi

"$MESH_SPLATTING_PYTHON" -c \
  'import cv2, numpy, open3d, scipy, sklearn, skimage, torch, trimesh' || {
    echo "install DTU evaluation dependencies from requirements.txt" >&2
    exit 1
  }

for SCAN in "${SCANS[@]}"; do
  SCAN_DIR="$DATA_ROOT/scan$SCAN"
  test -d "$SCAN_DIR/images" || { echo "missing $SCAN_DIR/images" >&2; exit 1; }
  test -d "$SCAN_DIR/mask" || { echo "missing $SCAN_DIR/mask" >&2; exit 1; }
  test -d "$SCAN_DIR/sparse/0" || { echo "missing $SCAN_DIR/sparse/0" >&2; exit 1; }
  test -f "$SCAN_DIR/cameras.npz" || { echo "missing $SCAN_DIR/cameras.npz" >&2; exit 1; }
  test -f "$DTU_EVAL_ROOT/ObsMask/ObsMask${SCAN}_10.mat" || {
    echo "missing DTU observability mask for scan$SCAN" >&2; exit 1;
  }
  test -f "$DTU_EVAL_ROOT/ObsMask/Plane${SCAN}.mat" || {
    echo "missing DTU ground plane for scan$SCAN" >&2; exit 1;
  }
  printf -v STL "%s/Points/stl/stl%03d_total.ply" "$DTU_EVAL_ROOT" "$SCAN"
  test -f "$STL" || { echo "missing $STL" >&2; exit 1; }
done

evaluate() {
  local scan=$1 arm=$2 run_arm=$3
  local scene="scan$scan"
  local model="$RUNS/${run_arm}__${scene}"
  local checkpoint="$model/point_cloud/iteration_30000"
  local output="$ROOT/$scene/$arm"
  [ -f "$output/DONE" ] && { echo "== $arm/$scene already evaluated"; return; }
  mkdir -p "$output"
  "$MESH_SPLATTING_PYTHON" -u create_ply.py "$checkpoint" \
    --out "$output/mesh.ply" --cpu
  "$MESH_SPLATTING_PYTHON" -u -m sota.dtu_cull \
    --input-mesh "$output/mesh.ply" --scan-dir "$DATA_ROOT/$scene" \
    --output-mesh "$output/culled_mesh.ply"
  "$MESH_SPLATTING_PYTHON" -u eval.py \
    --data "$output/culled_mesh.ply" --scan "$scan" --mode mesh \
    --dataset_dir "$DTU_EVAL_ROOT" --vis_out_dir "$output" --seed 0
  test -s "$output/results.json"
  printf 'complete\n' > "$output/DONE"
}

for SCAN in "${SCANS[@]}"; do
  "$HERE/run.sh" stock "scan$SCAN"
  "$HERE/run.sh" opacity08 "scan$SCAN" --final_opacity 0.8
  evaluate "$SCAN" stock stock
  evaluate "$SCAN" ours_quality opacity08
done

"$MESH_SPLATTING_PYTHON" -m sota.dtu_table "$ROOT" "$RUNS"
