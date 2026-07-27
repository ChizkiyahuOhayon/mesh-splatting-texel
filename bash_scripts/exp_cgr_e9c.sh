#!/usr/bin/env bash
# E9c (the decisive test): does O_i separate resolved creases from GENUINELY
# under-resolved thin structure, measured cleanly?
#
# Two fixes over E9/E9b:
#   1. Regime: the toy now has thin cylindrical SPOKES (2D-thin tubes), which an
#      opaque connected mesh cannot fully resolve -- so they stay under-resolved
#      (persistently driven) while the cube's creases resolve. (Flat plates were
#      too easy; there was no under-resolved-thin regime to detect.)
#   2. Measurement: cgr_diagnose.py sweeps ALL train views at a frozen checkpoint,
#      accumulating per-vertex sum-of-gradient / sum-of-magnitude / observation
#      count -- removing the EMA-recency and visibility-sparsity artifacts that
#      made nu_i read ~0 in E9b.
#
# Read the self-validating verdict per checkpoint:
#   regime present (nu_i AUC>=0.65)?  no  -> UNINFORMATIVE (thin not under-resolved
#       here, or its residual was absorbed by appearance -> no position-space signal)
#   yes + O_i>=0.8 and beats nu_i/curv -> PASS  (proceed to E10)
#   yes + O_i does not beat them       -> FALSIFIED (coherence adds nothing)
#
#   bash bash_scripts/exp_cgr_e9c.sh <scene_dir> <out_dir> <gpu_id>
set -euo pipefail
SCENE=${1:-data/cgr_toy2}
OUT=${2:-output/cgr_e9c}
GPU=${3:?usage: $0 <scene_dir> <out_dir> <gpu_id>}
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$OUT"
SAVE_ITERS="12000 14000 16000"

python -c "import diff_triangle_rasterization" 2>/dev/null || {
  echo "ERROR: diff_triangle_rasterization not importable. 'micromamba activate mesh_splatting'?" >&2; exit 1; }

# 1) Generate the spoke toy scene (deterministic; ~7 min on CPU) if absent.
if [ ! -f "$SCENE/transforms_train.json" ]; then
  echo "== generating spoke toy scene at $SCENE"
  python "$REPO_ROOT/cgr_toy_scene.py" --out "$SCENE" --n-train 100 --n-test 20 --res 400 --seed 0
fi

# 2) Train normally (NO --cgr_diag), saving mid-training checkpoints where creases
#    are resolved but the thin spokes are still under-resolved.
if [ ! -d "$OUT/point_cloud/iteration_16000" ]; then
  echo "== training (GPU $GPU), checkpoints at: $SAVE_ITERS"
  python train.py -s "$SCENE" -m "$OUT" -w --eval \
    --iterations 16000 --save_iterations $SAVE_ITERS 2>&1 | tee "$OUT/train.log"
fi

# 3) Clean measurement + ROC-AUC at each checkpoint.
for it in $SAVE_ITERS; do
  [ -d "$OUT/point_cloud/iteration_${it}" ] || { echo "  (no checkpoint $it, skipping)"; continue; }
  echo "===== checkpoint iter $it ====="
  DUMP="$OUT/cgr_sweep_${it}.npz"
  python "$REPO_ROOT/cgr_diagnose.py" -s "$SCENE" -w \
    --model "$OUT" --iteration "$it" --min-views 5 --out "$DUMP"
  python "$REPO_ROOT/cgr_auc.py" --dump "$DUMP" --gt "$SCENE/gt_labels.npz" \
    --min-obs 5 --out "$OUT/cgr_e9c_auc_${it}.json"
done
