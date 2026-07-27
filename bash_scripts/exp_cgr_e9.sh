#!/usr/bin/env bash
# E9 (Phase A kill-shot): the CGR separation diagnostic.
#
# Trains MeshSplatting on a controlled toy scene (a cube with sharp CREASES + thin
# FINS) up to a fixed-topology window, observes the per-vertex photometric-gradient
# trajectory, and dumps per-face O_i / nu_i / curvature. The offline ROC-AUC test
# then asks whether the trajectory-coherence signal O_i separates resolved creases
# from under-resolved thin structure where the static competitors tie.
#
# Pre-registered kill-shot (RESEARCH_PLAN_v6 sec.4):
#   O_i AUC >= 0.8  AND  nu_i, curvature ~= 0.5  -> proceed to E10.
#   nu_i or curvature also high                   -> CGR oral claim FALSIFIED here.
#
#   bash bash_scripts/exp_cgr_e9.sh <scene_dir> <out_dir> <gpu_id>
set -euo pipefail
SCENE=${1:-data/cgr_toy}
OUT=${2:-output/cgr_e9}
GPU=${3:?usage: $0 <scene_dir> <out_dir> <gpu_id>}
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$OUT"

python -c "import diff_triangle_rasterization" 2>/dev/null || {
  echo "ERROR: diff_triangle_rasterization not importable. 'micromamba activate mesh_splatting'?" >&2; exit 1; }

# 1) Generate the toy scene (deterministic; ~7 min on CPU) if it is not there yet.
if [ ! -f "$SCENE/transforms_train.json" ]; then
  echo "== generating toy scene at $SCENE"
  python -c "import trimesh" 2>/dev/null || {
    echo "ERROR: trimesh missing (needed for scene gen). 'pip install trimesh'." >&2; exit 1; }
  python "$REPO_ROOT/cgr_toy_scene.py" --out "$SCENE" \
    --n-train 100 --n-test 20 --res 400 --seed 0
fi

# 2) Train with the CGR diagnostic. Dumps the signal at several convergence stages
#    (pre- and post-Delaunay) so we can see WHERE the under-resolved-thin regime
#    exists. Early-stops after the last dump (~16k iters, not the full schedule).
DUMP_ITERS="4000,8000,12000,16000"
LAST_DUMP="$OUT/cgr_diag_${DUMP_ITERS##*,}.npz"
if [ ! -f "$LAST_DUMP" ]; then
  echo "== training with --cgr_diag (GPU $GPU)"
  python train.py -s "$SCENE" -m "$OUT" -w --eval \
    --cgr_diag --cgr_dump_iters "$DUMP_ITERS" --cgr_window 300 --cgr_rho 0.9 2>&1 | tee "$OUT/train.log"
fi
[ -f "$LAST_DUMP" ] || { echo "ERROR: CGR dumps not produced (see $OUT/train.log)" >&2; exit 1; }

# 3) ROC-AUC kill-shot analysis at each convergence stage. The self-validating
#    verdict flags a stage UNINFORMATIVE if the regime is absent (nu_i does not
#    separate), PASS if O_i separates and beats magnitude+curvature, else FALSIFIED.
echo "== E9 ROC-AUC analysis (per convergence stage)"
for it in ${DUMP_ITERS//,/ }; do
  DUMP="$OUT/cgr_diag_${it}.npz"
  [ -f "$DUMP" ] || { echo "  (missing $DUMP, skipping)"; continue; }
  echo "----- iter $it -----"
  python "$REPO_ROOT/cgr_auc.py" \
    --dump "$DUMP" --gt "$SCENE/gt_labels.npz" --out "$OUT/cgr_e9_auc_${it}.json"
done
