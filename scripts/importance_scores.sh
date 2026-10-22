#!/usr/bin/env bash
# D2: equal-budget cuts of each SoftTail pre-cleanup mesh under five importance
# scores (peak, integral, opacity, hits, LightGaussian). Same mesh, same count.
set -uo pipefail
ST=/root/autodl-tmp/st3
cd $ST/code
export MESH_SPLATTING_PYTHON=/root/miniconda3/envs/ms/bin/python
source sota/ensure_environment.sh
PY=$MESH_SPLATTING_PYTHON
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
DATA=/root/autodl-tmp/data/m360 RUNS=$ST/runs OUT=$ST/scores_01
images_for() { case "$1" in bicycle|flowers|garden|stump|treehill) echo images_4;; *) echo images_2;; esac; }
for S in ${SCENES:-garden room bicycle stump}; do
  RUN=$RUNS/softtail__$S
  [ -f "$OUT/$S/scores.json" ] || "$PY" -u -m sota.score_cuts -s "$DATA/$S" -i "$(images_for $S)" --eval \
      -m "$RUN" --out "$OUT/$S" --quiet || { echo "FAIL cut $S"; continue; }
  for B in b10 b25 b50 b100; do
    for SC in integral peak lightgaussian hits opacity; do
      M=$OUT/$S/$B/$SC; O=$M/eval
      [ -f "$O/DONE" ] && continue; rm -rf "$O"
      "$PY" -u -m sota.main_table_eval -s "$DATA/$S" -i "$(images_for $S)" --eval -m "$M" \
        --scene $S --arm ours_quality --iteration 30000 --output "$O" --quiet || echo "FAIL eval $S $B $SC"
      [ -f "$O/DONE" ] && rm -rf "$M/point_cloud"
      echo "D2 $S $B $SC $(grep -o '"psnr": [0-9.]*' $O/result.json 2>/dev/null | head -1) $(date +%T)"
    done
  done
done
echo D2_DONE
