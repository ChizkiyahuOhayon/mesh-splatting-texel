#!/usr/bin/env bash
# Test the published indoor configuration on one DTU baseline checkpoint.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/ensure_environment.sh"

REPO="$(cd "$HERE/.." && pwd)"
test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" || {
  echo "tracked source changes are present in $REPO" >&2
  exit 1
}
cd "$REPO"

NAS_ROOT=${NAS_ROOT:-/home/smbu/dy/nas/meshsplatting_smbu}
DATA_ROOT=${DATA_ROOT:-/home/smbu/dy/nas/dy/mesh-splatting/data/dtu}
DTU_EVAL_ROOT=${DTU_EVAL_ROOT:-/home/smbu/dy/nas/dy/mesh-splatting/data/dtu_eval}
RUNS=${RUNS:-$NAS_ROOT/experiments/dtu_indoor_probe_01/runs}
ROOT=${ROOT:-$NAS_ROOT/experiments/dtu_indoor_probe_01/scan24/stock_indoor}
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU DATA_ROOT RUNS

[ -f "$ROOT/DONE" ] && {
  echo "DTU indoor baseline probe already complete: $ROOT"
  cat "$ROOT/results.json"
  exit 0
}

SCAN_DIR="$DATA_ROOT/scan24"
test -d "$SCAN_DIR/images"
test -d "$SCAN_DIR/mask"
test -d "$SCAN_DIR/sparse/0"
test -f "$SCAN_DIR/cameras.npz"
test -f "$DTU_EVAL_ROOT/ObsMask/ObsMask24_10.mat"
test -f "$DTU_EVAL_ROOT/ObsMask/Plane24.mat"
test -f "$DTU_EVAL_ROOT/Points/stl/stl024_total.ply"

"$MESH_SPLATTING_PYTHON" -c \
  'import cv2, numpy, open3d, scipy, sklearn, skimage, torch, trimesh' || {
    echo "install DTU evaluation dependencies from requirements.txt" >&2
    exit 1
  }

mkdir -p "$ROOT"
REVISION=$(git rev-parse HEAD)
if [ -f "$ROOT/source_revision.txt" ]; then
  test "$(tr -d '[:space:]' < "$ROOT/source_revision.txt")" = "$REVISION" || {
    echo "source revision differs from the existing DTU probe" >&2
    exit 1
  }
else
  printf '%s\n' "$REVISION" > "$ROOT/source_revision.txt"
fi

"$HERE/run.sh" stock_indoor scan24 --indoor

MODEL="$RUNS/stock_indoor__scan24/point_cloud/iteration_30000"
"$MESH_SPLATTING_PYTHON" -u create_ply.py "$MODEL" \
  --out "$ROOT/mesh.ply" --cpu
"$MESH_SPLATTING_PYTHON" -u -m sota.dtu_cull \
  --input-mesh "$ROOT/mesh.ply" --scan-dir "$SCAN_DIR" \
  --output-mesh "$ROOT/culled_mesh.ply"
"$MESH_SPLATTING_PYTHON" -u eval.py \
  --data "$ROOT/culled_mesh.ply" --scan 24 --mode mesh \
  --dataset_dir "$DTU_EVAL_ROOT" --vis_out_dir "$ROOT" --seed 0

test -s "$ROOT/results.json"
printf 'complete\n' > "$ROOT/DONE"
cat "$ROOT/results.json"
