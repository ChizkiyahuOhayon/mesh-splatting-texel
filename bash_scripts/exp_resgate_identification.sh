#!/usr/bin/env bash
# E8 (Phase B): the ResidualGate identification / falsification experiment.
#
# Gates the normal-consistency regularizer per face by a signal, and compares the
# signal our theory predicts (gm) against the two negative controls the falsification
# names (raw residual, curvature) plus the ungated baseline. The prediction: gm
# recovers thin-region detail (LPIPS down on high-g_m regions) while raw/curvature do
# not, tying the gain to the appearance-saturated cross-view-consistent residual.
#
#   bash bash_scripts/exp_resgate_identification.sh <scene> output/resgate <gpu>
set -euo pipefail
SCENE=${1:?usage: $0 <scene_dir> <out_dir> <gpu_id>}
OUT=${2:?usage: $0 <scene_dir> <out_dir> <gpu_id>}
GPU=${3:?usage: $0 <scene_dir> <out_dir> <gpu_id> [arms]}
ARMS=${4:-"baseline_probe resgate_gm control_raw control_curvature"}   # which arms this worker runs
ARMS=${ARMS//,/ }
export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$OUT"
python -c "import diff_triangle_rasterization" 2>/dev/null || {
  echo "ERROR: diff_triangle_rasterization not importable. 'micromamba activate mesh_splatting'?" >&2; exit 1; }

run () {  # name  extra-args...
  local name=$1; shift
  local dir="$OUT/$name"
  [ -f "$dir/DONE" ] && { echo "== $name done, skipping"; return; }
  echo "=================== $name (GPU $GPU) ==================="
  if python train.py -s "$SCENE" -m "$dir" --eval "$@" 2>&1 | tee "$dir.log" \
     && test -f "$dir/point_cloud/iteration_30000/point_cloud_state_dict.pt" \
     && grep -q "ITER 30000\] Evaluating test" "$dir.log"; then
    touch "$dir/DONE"
  else
    touch "$dir/FAILED"; echo "RUN FAILED: $name" >&2; exit 1
  fi
}

for arm in $ARMS; do
  case "$arm" in
    baseline)          run baseline ;;                                          # ungated (resgate off)
    baseline_probe)    run baseline_probe    --resgate --resgate_floor 1.0 ;;   # ungated (phi==1) BUT computes+saves g_m -> the reference mask
    resgate_gm)        run resgate_gm        --resgate --resgate_signal gm ;;   # ours
    control_raw)       run control_raw       --resgate --resgate_signal raw ;;  # ctrl 1: no cross-view consistency
    control_curvature) run control_curvature --resgate --resgate_signal curvature ;; # ctrl 2: curvature not residual
    *) echo "unknown arm: $arm" >&2; exit 1 ;;
  esac
done

echo
echo "=========== SUMMARY (test @ 30000) ==========="
printf "%-18s %-8s %-8s %-8s\n" run PSNR SSIM LPIPS
for name in baseline_probe baseline resgate_gm control_raw control_curvature; do
  L="$OUT/$name.log"; [ -f "$L" ] || continue
  TE=$(grep "ITER 30000\] Evaluating test" "$L" | tail -1)
  printf "%-18s %-8.6s %-8.6s %-8.6s\n" "$name" \
    "$(sed -n 's/.*PSNR \([0-9.]*\).*/\1/p' <<<"$TE")" \
    "$(sed -n 's/.*SSIM \([0-9.]*\).*/\1/p' <<<"$TE")" \
    "$(sed -n 's/.*LPIPS \([0-9.]*\).*/\1/p' <<<"$TE")"
done
