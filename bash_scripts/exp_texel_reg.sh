#!/bin/bash
# E5: can texel regularization convert the overfitting headroom into test gain?
#
# E4 found the texel carrier overfits hard: train-test PSNR gap ~0.87 at order2/mult1,
# vs ~0.45 for the SH baseline. train PSNR reaches 25.9 while test sits at 25.0. If a
# residual regularizer recovers even half of that gap, test PSNR could reach ~25.4
# (+0.65 over baseline) -- the only lever identified so far that could lift the effect
# size from "incremental" to "interesting".
#
# All runs: garden, order2, default regularization. Only the texel regularizer varies.
# Reference: order2 no texel-reg = 25.034 / 0.7732 / 0.2093 (train 25.905, gap 0.87).
#
#   bash bash_scripts/exp_texel_reg.sh <scene> output/texelreg <gpu>
set -eu
SCENE=${1:?usage: $0 <scene_dir> <out_dir> <gpu_id>}
OUT=${2:?usage: $0 <scene_dir> <out_dir> <gpu_id>}
GPU=${3:?usage: $0 <scene_dir> <out_dir> <gpu_id>}
export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$OUT"

run () {  # name  extra-args...
  local name=$1; shift
  [ -f "$OUT/$name/DONE" ] && { echo "== $name done, skipping"; return; }
  echo "=================== $name (GPU $GPU) ==================="
  python train.py -s "$SCENE" -m "$OUT/$name" --eval --texel_order 2 "$@" \
      2>&1 | tee "$OUT/$name.log"
  touch "$OUT/$name/DONE"
}

run l2_0           # reference: no texel regularization
run l2_1e-4  --texel_l2 1e-4
run l2_1e-3  --texel_l2 1e-3
run l2_1e-2  --texel_l2 1e-2
run tv_1e-3  --texel_tv 1e-3
run tv_1e-2  --texel_tv 1e-2

echo
echo "=========== SUMMARY (test @ 30000) ==========="
printf "%-12s %-8s %-8s %-8s %-8s %-6s\n" run PSNR SSIM LPIPS trainPSNR gap
for name in l2_0 l2_1e-4 l2_1e-3 l2_1e-2 tv_1e-3 tv_1e-2; do
  L="$OUT/$name.log"; [ -f "$L" ] || continue
  TE=$(grep "ITER 30000\] Evaluating test"  "$L" | tail -1)
  TR=$(grep "ITER 30000\] Evaluating train" "$L" | tail -1)
  pt=$(sed -n 's/.*PSNR \([0-9.]*\).*/\1/p' <<<"$TE")
  pr=$(sed -n 's/.*PSNR \([0-9.]*\).*/\1/p' <<<"$TR")
  gp=$(python3 -c "print(f'{float('${pr:-0}')-float('${pt:-0}'):.3f}')" 2>/dev/null||echo -)
  printf "%-12s %-8.6s %-8.6s %-8.6s %-8.6s %-6s\n" "$name" "$pt" \
    "$(sed -n 's/.*SSIM \([0-9.]*\).*/\1/p' <<<"$TE")" \
    "$(sed -n 's/.*LPIPS \([0-9.]*\).*/\1/p' <<<"$TE")" "$pr" "$gp"
done
