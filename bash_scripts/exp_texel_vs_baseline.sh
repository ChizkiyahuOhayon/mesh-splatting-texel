#!/bin/bash
# E3-B: per-face texel carrier vs baseline, end-to-end on the same scene/seed/schedule.
#
# The carrier is allocated zero-initialised right after run_restricted_delaunay()
# (iteration densify_until_iter+1000), so everything before that point is bit-identical
# to the baseline and any difference is attributable to the carrier alone.
# verify_texel.py confirms this: zero texels are an exact no-op (max|diff| = 0).
#
# CONTROL: the baseline run must reproduce the original garden result (PSNR ~24.71,
# see record1.md). If it does not, stop -- the environment is wrong, not the method.
#
# Expectation, from 20+ frozen-geometry configs in maclab/RESULTS.md: the carrier's
# gain is concentrated in the PERCEPTUAL metrics (SSIM/LPIPS), replicated at three
# geometry resolutions. PSNR parity would be a good outcome, not a failure.
set -eu
SCENE=${1:?usage: $0 <scene_dir> <out_dir>}
OUT=${2:?usage: $0 <scene_dir> <out_dir>}
mkdir -p "$OUT"

# Fail fast if the CUDA extension is not importable (usually a forgotten
# `micromamba activate mesh_splatting`): the runs would otherwise crash one by one.
python -c "import diff_triangle_rasterization" 2>/dev/null || {
  echo "ERROR: diff_triangle_rasterization not importable. Did you 'micromamba activate mesh_splatting'?" >&2
  exit 1
}

run () {  # name  extra-args...
  local name=$1; shift
  [ -f "$OUT/$name/DONE" ] && { echo "== $name already done, skipping"; return; }
  echo "=================== $name ==================="
  /usr/bin/time -v python train.py -s "$SCENE" -m "$OUT/$name" --eval "$@" \
      2>&1 | tee "$OUT/$name.log"
  touch "$OUT/$name/DONE"
}

run baseline                              # texel_order defaults to 0 -> original path
run order2   --texel_order 2              # 4 texels/face
run order3   --texel_order 3              # 9 texels/face

echo
echo "=================== SUMMARY ==================="
for n in baseline order2 order3; do
  echo "--- $n ---"
  grep -E "\[texel\] allocated" "$OUT/$n.log" 2>/dev/null || echo "  (no texel carrier)"
  grep -E "after re-delaunay|Evaluating|PSNR|SSIM|LPIPS|Elapsed \(wall clock\)" \
       "$OUT/$n.log" 2>/dev/null | tail -8
done
