#!/usr/bin/env bash
# Generalization: the integrated survival rule inside Triangle Splatting+.
#
# Two arms per scene, same TS+ code and binary (branch integrated-survival of
# triangle-splatting2; the kernel accumulates sum(alpha*T) in both arms, only
# the flag decides whether it is used):
#   tsplus      published TS+ training, evaluation with TS+'s render.py/metrics.py
#   tsplus_int  + --integrated_importance (budget-matched: the peak rule still
#               sets how many faces go, the integral picks which)
# Scenes: Mip-NeRF 360 (9), Tanks & Temples (2), Deep Blending (2); the three
# gate scenes first. Data is a local mirror of the NAS copies; TS+ writes its
# Metric3D normal maps next to the images there, never into the NAS data.
# Single GPU, resumable per run (DONE marker).
set -uo pipefail

TSPLUS=${TSPLUS:-/home/smbu/dy/triangle-splatting2}
PY=${PY:-/home/smbu/dy/envs/tsplus/bin/python}
DATA=${DATA:-/home/smbu/dy/data_local}
OUT=${OUT:-/home/smbu/dy/nas/meshsplatting_smbu/experiments/tsplus_01}
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$OUT"

SCENES=(bicycle garden room flowers stump treehill counter kitchen bonsai train truck drjohnson playroom)

args_for() {
  case "$1" in
    bicycle|flowers|garden|stump|treehill) echo "-s $DATA/mipnerf360/$1 -i images_4" ;;
    room|counter|kitchen|bonsai)           echo "-s $DATA/mipnerf360/$1 -i images_2 --indoor" ;;
    train|truck)                           echo "-s $DATA/tandt/$1" ;;
    drjohnson|playroom)                    echo "-s $DATA/db/$1 --indoor" ;;
  esac
}

run() {  # run <arm> <scene> [extra train args]
  local arm=$1 scene=$2; shift 2
  local model="$OUT/${arm}__${scene}"
  [ -f "$model/DONE" ] && { echo "== $arm/$scene done"; return 0; }
  rm -rf "$model"; mkdir -p "$model"
  # shellcheck disable=SC2046
  (cd "$TSPLUS" && git rev-parse HEAD 2>/dev/null > "$model/source_revision.txt"
   START=$SECONDS
   "$PY" -u train.py $(args_for "$scene") -m "$model" --eval --quiet --test_iterations -1 "$@" \
     > "$model/train.log" 2>&1 || exit 1
   echo $((SECONDS - START)) > "$model/train_seconds.txt"
   "$PY" -u render.py $(args_for "$scene") -m "$model" --eval --skip_train --iteration 30000 --quiet \
     > "$model/render.log" 2>&1 || exit 1
   "$PY" -u metrics.py -m "$model" > "$model/metrics.log" 2>&1 || exit 1) \
    && test -s "$model/results.json" && echo complete > "$model/DONE" \
    || echo "FAILED $arm/$scene (see $model)"
}

for SCENE in "${SCENES[@]}"; do
  run tsplus "$SCENE"
  run tsplus_int "$SCENE" --integrated_importance
done

"$PY" - "$OUT" "${SCENES[@]}" <<'PY'
import json, sys
from pathlib import Path
out, scenes = Path(sys.argv[1]), sys.argv[2:]
rows = {}
for scene in scenes:
    row = {}
    for arm in ("tsplus", "tsplus_int"):
        path = out / f"{arm}__{scene}" / "results.json"
        if path.is_file():
            metrics = next(iter(json.loads(path.read_text()).values()))
            row[arm] = {k.lower(): v for k, v in metrics.items()}
    rows[scene] = row
(out / "tsplus_table.json").write_text(json.dumps({"experiment": "tsplus-integrated-survival",
                                                  "rows": rows}, indent=2) + "\n")
for scene, row in rows.items():
    if len(row) == 2:
        a, b = row["tsplus"], row["tsplus_int"]
        print(f"{scene:10s} TS+ {a['psnr']:.3f} -> {b['psnr']:.3f} ({b['psnr'] - a['psnr']:+.3f})")
PY
touch "$OUT/TSPLUS_DONE"
