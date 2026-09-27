#!/usr/bin/env bash
# Compact meshes: re-cut every trained Mip-NeRF 360 run offline to 10% and 25%
# of the published face budget, once by the peak rule and once by the integral,
# and score both cuts. Evaluation only; no retraining. Extends batch47 part B
# (which covered 25/50/75% on bicycle, garden, room) to all nine scenes.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/ensure_environment.sh"
PY=$MESH_SPLATTING_PYTHON
LOCAL=${LOCAL:-/home/smbu/dy/data_local}/mipnerf360
E=/home/smbu/dy/nas/meshsplatting_smbu/experiments
RUNS=$E/softtail_integrated_01
OUT=${OUT:-$E/paper_evidence_01/compact}
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU
cd "$HERE/.."
images_for() { case "$1" in bicycle|flowers|garden|stump|treehill) echo images_4;; *) echo images_2;; esac; }
for SCENE in bicycle flowers garden stump treehill room counter kitchen bonsai; do
  V1=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["faces_kept"]["v1"])' "$RUNS/cut__$SCENE/survival.json")
  for FRACTION in 10 25; do
    CUT="$OUT/$SCENE/b$FRACTION"
    if [ ! -d "$CUT/matched_oats/point_cloud" ]; then
      rm -rf "$CUT"
      "$PY" -u -m sota.survival_cleanup -s "$LOCAL/$SCENE" -i "$(images_for "$SCENE")" --eval \
        -m "$RUNS/integrated__$SCENE" --out "$CUT" --budget $((V1 * FRACTION / 100)) --quiet
      rm -rf "$CUT/v1" "$CUT/oats"
    fi
    for RULE in peak integral; do
      MODEL="$CUT/$([ $RULE = peak ] && echo matched || echo matched_oats)"
      OUTPUT="$CUT/eval_$RULE"
      [ -f "$OUTPUT/DONE" ] && continue
      rm -rf "$OUTPUT"
      "$PY" -u -m sota.main_table_eval -s "$LOCAL/$SCENE" -i "$(images_for "$SCENE")" --eval -m "$MODEL" \
        --scene "$SCENE" --arm ours_quality --iteration 30000 --output "$OUTPUT" --quiet
    done
  done
done
"$PY" - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = {}
for scene_dir in sorted(p for p in out.iterdir() if p.is_dir()):
    for cut in sorted(scene_dir.glob("b*")):
        for rule in ("peak", "integral"):
            r = json.loads((cut / f"eval_{rule}" / "result.json").read_text())
            rows.setdefault(scene_dir.name, {}).setdefault(cut.name, {})[rule] = {
                "triangles": r["triangles"], **{k: r["metrics"][k] for k in ("psnr", "ssim", "lpips_vgg")}}
(out / "compact.json").write_text(json.dumps({"experiment": "softtail-compact-budget", "rows": rows}, indent=2) + "\n")
for scene, cuts in rows.items():
    print(scene, " ".join(f"{b}:{c['integral']['psnr'] - c['peak']['psnr']:+.2f}" for b, c in cuts.items()))
PY
touch "$OUT/DONE"
