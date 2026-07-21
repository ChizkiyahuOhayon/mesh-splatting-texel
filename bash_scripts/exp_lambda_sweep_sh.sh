#!/bin/bash
# E3-A: identification experiment, per-vertex-SH arm.
#
# The surviving core claim is that the fidelity cost of geometric regularization is an
# artifact of vertex-carried appearance. Testing it needs TWO curves of fidelity vs
# regularization strength: one for the baseline carrier, one for the decoupled carrier.
#
# This script produces the BASELINE (per-vertex SH) curve. It needs NO code changes --
# only CLI overrides of the regularization weights -- so it can run on an idle GPU
# immediately, in parallel with development of the texel carrier.
#
# Reference (paper Table 4, Mip-NeRF360 average): removing L_n gives +0.10 PSNR and
# removing L_d gives +0.05 PSNR, i.e. regularization *costs* fidelity. We expect a
# negative slope here. The claim under test later is that the texel arm flattens it.
#
# Usage on the server:
#   bash bash_scripts/exp_lambda_sweep_sh.sh /path/to/data/mipnerf360/garden output/lambda_sweep
set -eu
SCENE=${1:?usage: $0 <scene_dir> <out_dir>}
OUT=${2:?usage: $0 <scene_dir> <out_dir>}
mkdir -p "$OUT"

# Defaults from arguments/__init__.py: lambda_normals=0.00005, lambda_normals_super=0.01,
# lamba_depth=0.05. We scale the normal + depth regularizers together as a single
# "regularization strength" knob so the curve has one x-axis.
for MULT in 0 0.5 1 2 4; do
  LN=$(python3 -c "print(0.00005*$MULT)")
  LNS=$(python3 -c "print(0.01*$MULT)")
  LD=$(python3 -c "print(0.05*$MULT)")
  RUN="$OUT/sh_mult${MULT}"
  echo "=================== regularization x${MULT} ==================="
  echo "lambda_normals=$LN lambda_normals_super=$LNS lamba_depth=$LD"
  [ -f "$RUN/DONE" ] && { echo "already done, skipping"; continue; }
  python train.py -s "$SCENE" -m "$RUN" --eval \
      --lambda_normals "$LN" --lambda_normals_super "$LNS" --lamba_depth "$LD" \
      2>&1 | tee "$RUN.log"
  touch "$RUN/DONE"
done

echo
echo "=================== SUMMARY ==================="
for MULT in 0 0.5 1 2 4; do
  echo "--- regularization x${MULT} ---"
  grep -E "Evaluating test|PSNR|SSIM|LPIPS" "$OUT/sh_mult${MULT}.log" 2>/dev/null | tail -4
done
