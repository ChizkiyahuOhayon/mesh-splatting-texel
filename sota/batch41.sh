#!/usr/bin/env bash
# Equal-budget recheck of the training-time OATS arm (batch40 / softtail_integrated_01).
#
# That arm passed its gate at +0.110 dB, but it also finished with 2.6-22% more
# faces than the v1 control: ranking by the integral changes which faces the
# 4k-11k passes delete, and the survivors are split differently, so the mesh
# grows at a different rate. A gain bought with extra faces is not a gain.
#
# The runs saved their pre-cleanup state, so the question is answered without
# retraining: cut each one with the *published* cleanup rule down to exactly the
# face count v1 finished with, and score that. The `oats` copy is scored too,
# which is the full method (integral in training and in the cleanup).
#
# Single GPU, resumable at every step.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/ensure_environment.sh"

NAS_ROOT=${NAS_ROOT:-/home/smbu/dy/nas/meshsplatting_smbu}
DATA_ROOT=${DATA_ROOT:-/home/smbu/dy/nas/dy/mesh-splatting/data/mipnerf360}
FORMAL_ROOT=${FORMAL_ROOT:-$NAS_ROOT/experiments/formal_main_table_01}
RUNS=${RUNS:-$NAS_ROOT/experiments/softtail_integrated_01}
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU

SCENES=(room bicycle garden)
test -f "$FORMAL_ROOT/formal_table.json"

images_for() {
  case "$1" in
    bicycle|flowers|garden|stump|treehill) echo images_4 ;;
    room|counter|kitchen|bonsai)           echo images_2 ;;
  esac
}

# The control's face count is the budget; it comes from the frozen table, never
# from a number typed here.
budget_for() {
  "$MESH_SPLATTING_PYTHON" -c '
import json, sys
table = json.load(open(sys.argv[1]))
print(table["rows"][sys.argv[2]]["ours_quality"]["triangles"])
  ' "$FORMAL_ROOT/formal_table.json" "$1"
}

for SCENE in "${SCENES[@]}"; do
  test -d "$RUNS/integrated__${SCENE}/point_cloud/iteration_precleanup" \
    || { echo "missing pre-cleanup state for $SCENE" >&2; exit 1; }
done

for SCENE in "${SCENES[@]}"; do
  BUDGET="$(budget_for "$SCENE")"
  CUT="$RUNS/cut__${SCENE}"
  IMAGES="$(images_for "$SCENE")"

  if [ ! -f "$CUT/survival.json" ]; then
    rm -rf "$CUT"
    echo "== cutting $SCENE to $BUDGET faces"
    "$MESH_SPLATTING_PYTHON" -u -m sota.survival_cleanup \
      -s "$DATA_ROOT/$SCENE" -m "$RUNS/integrated__${SCENE}" -i "$IMAGES" --eval \
      --out "$CUT" --budget "$BUDGET"
  fi

  # matched: the published rule at the control's budget, which isolates the
  # training-time statistic. oats: the integral in the cleanup as well.
  for COPY in matched oats; do
    OUTPUT="$CUT/eval_${COPY}"
    [ -f "$OUTPUT/DONE" ] && { echo "== $COPY/$SCENE already evaluated"; continue; }
    "$MESH_SPLATTING_PYTHON" -u -m sota.main_table_eval \
      -s "$DATA_ROOT/$SCENE" -m "$CUT/$COPY" -i "$IMAGES" --eval \
      --scene "$SCENE" --arm ours_quality --iteration 30000 --output "$OUTPUT"
  done
done

"$MESH_SPLATTING_PYTHON" - "$FORMAL_ROOT/formal_table.json" "$RUNS" "${SCENES[@]}" <<'PY'
import json, sys
from pathlib import Path

table = json.load(open(sys.argv[1]))["rows"]
runs = Path(sys.argv[2])
scenes = sys.argv[3:]

rows, totals = [], {"matched": [], "oats": [], "trained": []}
for scene in scenes:
    v1 = table[scene]["ours_quality"]
    row = {"scene": scene, "v1_psnr": v1["psnr"], "v1_faces": v1["triangles"]}
    for copy in ("matched", "oats"):
        result = json.loads((runs / f"cut__{scene}" / f"eval_{copy}" / "result.json").read_text())
        row[f"{copy}_psnr"] = result["metrics"]["psnr"]
        row[f"{copy}_faces"] = result["triangles"]
        totals[copy].append(result["metrics"]["psnr"] - v1["psnr"])
    trained = json.loads((runs / scene / "ours_quality" / "result.json").read_text())
    row["trained_psnr"] = trained["metrics"]["psnr"]
    row["trained_faces"] = trained["triangles"]
    totals["trained"].append(trained["metrics"]["psnr"] - v1["psnr"])
    rows.append(row)

report = {
    "experiment": "softtail-integrated-importance-equal-budget",
    "rows": rows,
    "mean_psnr_gain_db": {k: sum(v) / len(v) for k, v in totals.items()},
    "psnr_wins": {k: sum(1 for d in v if d > 0) for k, v in totals.items()},
}
(runs / "equal_budget.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report["mean_psnr_gain_db"], indent=2))
print(json.dumps(report["psnr_wins"], indent=2))
for row in rows:
    print(f"{row['scene']:8s} v1 {row['v1_psnr']:.4f}/{row['v1_faces']} "
          f"matched {row['matched_psnr']:.4f}/{row['matched_faces']} "
          f"oats {row['oats_psnr']:.4f}/{row['oats_faces']} "
          f"trained {row['trained_psnr']:.4f}/{row['trained_faces']}")
PY

touch "$RUNS/EQUAL_BUDGET_DONE"
echo "Equal-budget recheck complete: $RUNS/equal_budget.json"
