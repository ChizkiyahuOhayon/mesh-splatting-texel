#!/bin/bash
# E4: the identification experiment -- the project's decisive test.
#
# Claim under test: the fidelity cost of geometric regularization in MeshSplatting is an
# artifact of *vertex-carried* appearance, not an inherent cost of accurate geometry.
# Because vertices must sit where colour changes, regularizing geometry also smooths
# appearance. Moving appearance onto a per-face carrier should break that coupling.
#
# Prediction:
#   per-vertex SH arm  -> fidelity DECREASES as regularization strengthens
#                         (the paper's own Table 4: removing L_n gives +0.10 PSNR,
#                          removing L_d gives +0.05, i.e. regularization costs fidelity)
#   texel arm          -> that slope FLATTENS
#
# The slope difference is the contribution. A level difference alone is not: "joint
# appearance+geometry optimization helps" is already published (LTM, CVPR 2024, its
# w/o-GO ablation). Without a slope separation there is no delta against LTM.
#
# Two arms, one per GPU, ~1.7 h x 5 runs ~= 8.5 h each:
#   bash bash_scripts/exp_identification.sh <scene> output/ident sh    0
#   bash bash_scripts/exp_identification.sh <scene> output/ident texel 1
set -euo pipefail
SCENE=${1:?usage: $0 <scene_dir> <out_dir> <sh|texel> <gpu_id>}
OUT=${2:?usage: $0 <scene_dir> <out_dir> <sh|texel> <gpu_id>}
ARM=${3:?usage: $0 <scene_dir> <out_dir> <sh|texel> <gpu_id>}
GPU=${4:?usage: $0 <scene_dir> <out_dir> <sh|texel> <gpu_id>}

export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$OUT"

# Fail fast if the CUDA extension is not importable (usually a forgotten
# `micromamba activate mesh_splatting`): the runs would otherwise crash one by one.
python -c "import diff_triangle_rasterization" 2>/dev/null || {
  echo "ERROR: diff_triangle_rasterization not importable. Did you 'micromamba activate mesh_splatting'?" >&2
  exit 1
}

case "$ARM" in
  sh)    EXTRA=""               ;;   # baseline carrier: per-vertex SH only
  texel) EXTRA="--texel_order 2" ;;  # decoupled carrier: + per-face texels
  *) echo "arm must be 'sh' or 'texel'"; exit 1 ;;
esac

# Defaults (arguments/__init__.py): lambda_normals=0.00005,
# lambda_normals_super=0.01, lamba_depth=0.05. All three geometric regularizers are
# scaled by one multiplier so the curve has a single x-axis.
for MULT in 0 0.5 1 2 4; do
  RUN="$OUT/${ARM}_mult${MULT}"
  [ -f "$RUN/DONE" ] && { echo "== $ARM x${MULT} already done, skipping"; continue; }
  LN=$(python3 -c "print(0.00005*$MULT)")
  LNS=$(python3 -c "print(0.01*$MULT)")
  LD=$(python3 -c "print(0.05*$MULT)")
  echo "=========== arm=$ARM  regularization x${MULT}  (GPU $GPU) ==========="
  echo "lambda_normals=$LN lambda_normals_super=$LNS lamba_depth=$LD ${EXTRA}"
  if python train.py -s "$SCENE" -m "$RUN" --eval \
      --lambda_normals "$LN" --lambda_normals_super "$LNS" --lamba_depth "$LD" \
      $EXTRA 2>&1 | tee "$RUN.log" \
     && test -f "$RUN/point_cloud/iteration_30000/point_cloud_state_dict.pt" \
     && grep -q "ITER 30000\] Evaluating test" "$RUN.log"; then
    touch "$RUN/DONE"
  else
    touch "$RUN/FAILED"; echo "RUN FAILED: $RUN" >&2; exit 1
  fi
done

echo
echo "=========== SUMMARY  arm=$ARM ==========="
printf "%-8s %-9s %-9s %-9s %-9s %-9s\n" mult PSNRtest SSIMtest LPIPStest PSNRtrain gap
for MULT in 0 0.5 1 2 4; do
  L="$OUT/${ARM}_mult${MULT}.log"
  [ -f "$L" ] || continue
  TE=$(grep "ITER 30000\] Evaluating test"  "$L" | tail -1)
  TR=$(grep "ITER 30000\] Evaluating train" "$L" | tail -1)
  pt=$(sed -n 's/.*PSNR \([0-9.]*\).*/\1/p' <<<"$TE")
  st=$(sed -n 's/.*SSIM \([0-9.]*\).*/\1/p' <<<"$TE")
  lt=$(sed -n 's/.*LPIPS \([0-9.]*\).*/\1/p' <<<"$TE")
  pr=$(sed -n 's/.*PSNR \([0-9.]*\).*/\1/p' <<<"$TR")
  gp=$(python3 -c "print(f'{float('${pr:-0}')-float('${pt:-0}'):.3f}')" 2>/dev/null || echo "-")
  printf "%-8s %-9.9s %-9.9s %-9.9s %-9.9s %-9s\n" "$MULT" "$pt" "$st" "$lt" "$pr" "$gp"
done
