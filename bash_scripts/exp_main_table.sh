#!/usr/bin/env bash
# Main table: baseline (per-vertex SH) vs texel carrier, across Mip-NeRF360 scenes.
# One worker per GPU; pass each GPU its own subset of scenes so the two run in parallel.
#
#   # GPU 0 (5 scenes), GPU 1 (4 scenes) -- roughly balanced by scene size
#   nohup bash bash_scripts/exp_main_table.sh <data_root> output/main 0 \
#        bicycle stump counter kitchen flowers > main_gpu0.log 2>&1 &
#   nohup bash bash_scripts/exp_main_table.sh <data_root> output/main 1 \
#        garden room bonsai treehill          > main_gpu1.log 2>&1 &
#
# <data_root> = /home/smbu/dy/mesh-splatting/data/mipnerf360  (scenes live under it)
# texel order defaults to 2; override with  ORDER=3 bash ...  for an ablation.
set -euo pipefail
DATA=${1:?usage: $0 <data_root> <out_dir> <gpu_id> <scene...>}
OUT=${2:?usage: $0 <data_root> <out_dir> <gpu_id> <scene...>}
GPU=${3:?usage: $0 <data_root> <out_dir> <gpu_id> <scene...>}
shift 3
SCENES="$*"
[ -n "$SCENES" ] || { echo "no scenes given" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="$GPU"
ORDER="${ORDER:-2}"
mkdir -p "$OUT"

# Fail fast on the wrong env (forgotten `micromamba activate mesh_splatting`).
python -c "import diff_triangle_rasterization" 2>/dev/null || {
  echo "ERROR: diff_triangle_rasterization not importable. 'micromamba activate mesh_splatting'?" >&2
  exit 1
}

run () {  # scene  name  extra-train-args...
  local scene=$1 name=$2; shift 2
  local dir="$OUT/${scene}_${name}"
  [ -f "$dir/DONE" ] && { echo "== ${scene}_${name} done, skipping"; return; }
  [ -d "$DATA/$scene/sparse/0" ] || { echo "MISSING scene $DATA/$scene" >&2; exit 1; }
  echo "=================== ${scene}/${name} (GPU $GPU) ==================="
  # DONE only on a genuinely complete run (exit 0 + checkpoint + eval line), else FAILED.
  if python train.py -s "$DATA/$scene" -m "$dir" --eval "$@" 2>&1 | tee "$dir.log" \
     && test -f "$dir/point_cloud/iteration_30000/point_cloud_state_dict.pt" \
     && grep -q "ITER 30000\] Evaluating test" "$dir.log"; then
    touch "$dir/DONE"
  else
    touch "$dir/FAILED"; echo "RUN FAILED: ${scene}_${name}" >&2; exit 1
  fi
}

for scene in $SCENES; do
  run "$scene" baseline
  run "$scene" "texel${ORDER}" --texel_order "$ORDER"
done
echo "MAIN TABLE WORKER DONE (GPU $GPU): $SCENES"
