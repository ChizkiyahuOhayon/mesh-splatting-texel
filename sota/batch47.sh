#!/usr/bin/env bash
# Paper evidence that needs the GPU but no new method: three independent parts,
# each resumable and each writing its own summary JSON.
#
#   A. Opaque deployment. Score every released SoftTail mesh with the `stock`
#      evaluator arm, which clamps every triangle to opacity 0.9999 and renders
#      it exactly like the published opaque baseline. Answers "is the mesh still
#      usable as a plain opaque mesh?" Evaluation only, 13 scenes.
#   B. Face-budget sweep. Re-cut three trained runs offline to 25/50/75% of the
#      v1 face count by both statistics (peak = published, integral = ours) and
#      score each cut: quality against mesh size for both rules. Evaluation only.
#   C. DTU geometry. Train the full method on the 15 DTU scans with the
#      protocol of the matched stock runs in softtail_dtu_01, cut by the
#      integral, export, cull and score with the official evaluator. The stock
#      rows are the archived formal_dtu_02 ones.
#
# Single GPU. Re-running resumes at the first unfinished step.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/ensure_environment.sh"
PY=$MESH_SPLATTING_PYTHON

NAS_ROOT=${NAS_ROOT:-/home/smbu/dy/nas/meshsplatting_smbu}
EXP=$NAS_ROOT/experiments
M360_DATA=${M360_DATA:-/home/smbu/dy/nas/dy/mesh-splatting/data/mipnerf360}
TANDT_DATA=${TANDT_DATA:-/home/smbu/dy/nas/dy/mesh-splatting/data/tandt/tandt}
DB_DATA=${DB_DATA:-/home/smbu/dy/nas/dy/mesh-splatting/data/tandt/db}
DTU_DATA=${DTU_DATA:-/home/smbu/dy/nas/dy/mesh-splatting/data/dtu}
DTU_EVAL_ROOT=${DTU_EVAL_ROOT:-"/home/smbu/dy/nas/dy/mesh-splatting/data/dtu_eval_raw/SampleSet/MVS Data"}
M360_RUNS=$EXP/softtail_integrated_01
TRANSFER_RUNS=$EXP/softtail_tandt_db_01
OUT=${OUT:-$EXP/paper_evidence_01}
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU

REPO="$(cd "$HERE/.." && pwd)"
test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" || {
  echo "tracked source changes are present in $REPO" >&2; exit 1; }
cd "$REPO"
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/source_revision.txt"

data_for() {  # data_for <scene> -> "<dataset dir> <images>"
  case "$1" in
    bicycle|flowers|garden|stump|treehill) echo "$M360_DATA/$1 images_4" ;;
    room|counter|kitchen|bonsai)           echo "$M360_DATA/$1 images_2" ;;
    train|truck)                           echo "$TANDT_DATA/$1 images" ;;
    drjohnson|playroom)                    echo "$DB_DATA/$1 images" ;;
  esac
}

evaluate() {  # evaluate <scene> <model> <arm> <output>
  [ -f "$4/DONE" ] && return 0
  rm -rf "$4"
  read -r data images <<< "$(data_for "$1")"
  "$PY" -u -m sota.main_table_eval -s "$data" -i "$images" --eval -m "$2" \
    --scene "$1" --arm "$3" --iteration 30000 --output "$4" --quiet
}

# ---------------------------------------------------------------- A. opaque
for SCENE in bicycle flowers garden stump treehill room counter kitchen bonsai; do
  evaluate "$SCENE" "$M360_RUNS/cut__$SCENE/oats" stock "$OUT/opaque/$SCENE"
done
for SCENE in train truck drjohnson playroom; do
  evaluate "$SCENE" "$TRANSFER_RUNS/cut__$SCENE/oats" stock "$OUT/opaque/$SCENE"
done
"$PY" - "$OUT/opaque" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = {}
for scene_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    r = json.loads((scene_dir / "result.json").read_text())
    rows[scene_dir.name] = {k: r["metrics"][k] for k in ("psnr", "ssim", "lpips_vgg", "fps")} \
        | {"triangles": r["triangles"], "model": r["model"]}
(root / "opaque.json").write_text(json.dumps({"experiment": "softtail-opaque-deployment",
                                              "arm": "stock", "rows": rows}, indent=2) + "\n")
print(f"opaque deployment: {len(rows)} scenes")
PY

# ---------------------------------------------------------------- B. budget sweep
for SCENE in bicycle garden room; do
  V1=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["faces_kept"]["v1"])' \
       "$M360_RUNS/cut__$SCENE/survival.json")
  read -r data images <<< "$(data_for "$SCENE")"
  for FRACTION in 25 50 75; do
    BUDGET=$((V1 * FRACTION / 100))
    CUT="$OUT/sweep/$SCENE/b$FRACTION"
    if [ ! -d "$CUT/matched_oats/point_cloud" ]; then
      rm -rf "$CUT"
      "$PY" -u -m sota.survival_cleanup -s "$data" -i "$images" --eval \
        -m "$M360_RUNS/integrated__$SCENE" --out "$CUT" --budget "$BUDGET" --quiet
    fi
    evaluate "$SCENE" "$CUT/matched" ours_quality "$CUT/eval_peak"
    evaluate "$SCENE" "$CUT/matched_oats" ours_quality "$CUT/eval_integral"
    # The cut directories hold full model copies; only the scores are kept.
    rm -rf "$CUT/v1" "$CUT/oats"
  done
done
"$PY" - "$OUT/sweep" "$M360_RUNS" <<'PY'
import json, sys
from pathlib import Path
root, runs = Path(sys.argv[1]), Path(sys.argv[2])
curves = {}
for scene_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    scene = scene_dir.name
    points = []
    for cut in sorted(scene_dir.glob("b*")):
        for rule in ("peak", "integral"):
            r = json.loads((cut / f"eval_{rule}" / "result.json").read_text())
            points.append({"fraction": int(cut.name[1:]) / 100, "rule": rule,
                           "triangles": r["triangles"],
                           **{k: r["metrics"][k] for k in ("psnr", "ssim", "lpips_vgg")}})
    # The 100% points are the archived ablation rows: v1 cut and oats cut.
    for rule, name in (("peak", "v1"), ("integral", "main_ours_quality")):
        path = runs / f"cut__{scene}" / ("abl_train_only" if rule == "peak" else name) / "result.json"
        r = json.loads(path.read_text())
        points.append({"fraction": 1.0, "rule": rule, "triangles": r["triangles"],
                       **{k: r["metrics"][k] for k in ("psnr", "ssim", "lpips_vgg")}})
    curves[scene] = sorted(points, key=lambda p: (p["rule"], p["fraction"]))
(root / "sweep.json").write_text(json.dumps({"experiment": "softtail-face-budget-sweep",
                                             "curves": curves}, indent=2) + "\n")
for scene, pts in curves.items():
    by = {(p["rule"], p["fraction"]): p["psnr"] for p in pts}
    print(scene, " ".join(f"{f:.2f}:{by[('integral', f)] - by[('peak', f)]:+.3f}"
                          for f in (0.25, 0.5, 0.75, 1.0)))
PY

# ---------------------------------------------------------------- C. DTU
export DATA_ROOT=$DTU_DATA RUNS=$EXP/softtail_dtu_01
SCANS=(24 37 40 55 63 65 69 83 97 105 106 110 114 118 122)
for SCAN in "${SCANS[@]}"; do
  SCENE=scan$SCAN
  "$HERE/run.sh" integrated "$SCENE" --final_opacity 0.8 --integrated_importance --save_precleanup
  RUN="$RUNS/integrated__$SCENE"
  CUT="$OUT/dtu/$SCENE/cut"
  [ -f "$CUT/survival.json" ] || { rm -rf "$CUT"; \
    "$PY" -u -m sota.survival_cleanup -s "$DTU_DATA/$SCENE" -r 2 -m "$RUN" --out "$CUT" --quiet; }
  OUTPUT="$OUT/dtu/$SCENE/softtail"
  if [ ! -f "$OUTPUT/DONE" ]; then
    mkdir -p "$OUTPUT"
    "$PY" -u create_ply.py "$CUT/oats/point_cloud/iteration_30000" --out "$OUTPUT/mesh.ply" --cpu
    "$PY" -u -m sota.dtu_cull --input-mesh "$OUTPUT/mesh.ply" --scan-dir "$DTU_DATA/$SCENE" \
      --output-mesh "$OUTPUT/culled_mesh.ply"
    "$PY" -u eval.py --data "$OUTPUT/culled_mesh.ply" --scan "$SCAN" --mode mesh \
      --dataset_dir "$DTU_EVAL_ROOT" --vis_out_dir "$OUTPUT" --seed 0
    test -s "$OUTPUT/results.json"
    echo complete > "$OUTPUT/DONE"
  fi
done
"$PY" - "$OUT/dtu" "$EXP/formal_dtu_02" "${SCANS[@]}" <<'PY'
import json, sys
from pathlib import Path
root, stock_root, scans = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
keys = ("accuracy", "completeness", "chamfer")
def read(path):
    r = json.loads(path.read_text())
    return {k: float(r[k] if k in r else r["mean_d2s" if k == "accuracy" else
                     "mean_s2d" if k == "completeness" else "overall"]) for k in keys}
rows = {f"scan{s}": {"stock": read(stock_root / f"scan{s}" / "stock" / "results.json"),
                     "softtail": read(root / f"scan{s}" / "softtail" / "results.json")}
        for s in scans}
means = {arm: {k: sum(r[arm][k] for r in rows.values()) / len(rows) for k in keys}
         for arm in ("stock", "softtail")}
(root / "dtu.json").write_text(json.dumps({"experiment": "softtail-dtu-geometry",
                                           "rows": rows, "means": means}, indent=2) + "\n")
print(json.dumps(means, indent=2))
PY

touch "$OUT/DONE"
echo "paper evidence complete: $OUT"
