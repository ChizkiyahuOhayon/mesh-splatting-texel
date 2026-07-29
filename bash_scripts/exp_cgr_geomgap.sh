#!/usr/bin/env bash
# P0 thickness-sweep: does MeshSplatting's GEOMETRY error rise as thin structures
# get thinner, while appearance stays good? (The geometry-accuracy thesis P0.)
#
# The toy has spokes whose radius is GEOMETRICALLY spaced from thick (0.035, ~11px)
# down to near-pixel-scale (0.006, ~1.9px). cgr_geomgap.py reports GT->recon Chamfer
# binned by spoke radius -- the geometry-error-vs-thinness curve.
#
#   flat curve near crease baseline across all radii -> mesh resolves even thin
#       geometry; P0 FAILS (no hidden geometry gap) -> stop the direction.
#   sharp rise below some radius -> the under-resolution regime a fix could target;
#       then a real-data (DTU) confirmation is justified.
#
#   bash bash_scripts/exp_cgr_geomgap.sh <scene_dir> <out_dir> <gpu_id>
set -euo pipefail
SCENE=${1:-data/cgr_toy_sweep}
OUT=${2:-output/cgr_geomgap}
GPU=${3:?usage: $0 <scene_dir> <out_dir> <gpu_id>}
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$OUT"

python -c "import diff_triangle_rasterization" 2>/dev/null || {
  echo "ERROR: diff_triangle_rasterization not importable. 'micromamba activate mesh_splatting'?" >&2; exit 1; }

# 1) Generate the thickness-sweep toy (16 spokes, radius 0.035 -> 0.006).
if [ ! -f "$SCENE/transforms_train.json" ]; then
  echo "== generating thickness-sweep toy at $SCENE"
  python "$REPO_ROOT/cgr_toy_scene.py" --out "$SCENE" --n-train 100 --n-test 20 --res 400 \
    --seed 0 --n-spokes 16 --spoke-r-max 0.035 --spoke-r-min 0.006
fi

# 2) Train normally to a converged mesh.
if [ ! -d "$OUT/point_cloud/iteration_16000" ]; then
  echo "== training (GPU $GPU)"
  python train.py -s "$SCENE" -m "$OUT" -w --eval \
    --iterations 16000 --save_iterations 16000 2>&1 | tee "$OUT/train.log"
fi

# 3) Get the reconstructed mesh (a cgr_diagnose dump carries vertices+faces), then
#    the geometry-error-vs-thinness curve.
RECON="$OUT/cgr_sweep_16000.npz"
[ -f "$RECON" ] || python "$REPO_ROOT/cgr_diagnose.py" -s "$SCENE" -w \
  --model "$OUT" --iteration 16000 --min-views 5 --out "$RECON"

echo "== P0 thickness sweep =="
python "$REPO_ROOT/cgr_geomgap.py" --recon "$RECON" --gt "$SCENE/gt_labels.npz" \
  --out "$OUT/geomgap_sweep.json"
