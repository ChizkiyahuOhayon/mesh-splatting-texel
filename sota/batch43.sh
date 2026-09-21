#!/usr/bin/env bash
# Nine-scene equal-budget recheck of the main table.
#
# The main table carries 2.6-33% more faces than the v1 control, so the gain and
# the mesh size are confounded on every row. On the three gate scenes the extra
# faces were worth 0.0036 dB of 0.1100 (3%); this establishes the same number on
# all nine, without retraining: cut each saved pre-cleanup state with the
# *integral* down to exactly the face count v1 finished with (matched_oats,
# i.e. the full method sized like the control) and score it.
#
# Writes into cutb__<scene>, leaving the main table's cut__<scene> untouched.
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

SCENES=(bicycle flowers garden stump treehill room counter kitchen bonsai)
test -f "$FORMAL_ROOT/formal_table.json"
[ -f "$RUNS/EQUAL_BUDGET_NINE_DONE" ] && { echo "Nine-scene equal-budget recheck already complete"; exit 0; }

images_for() {
  case "$1" in
    bicycle|flowers|garden|stump|treehill) echo images_4 ;;
    room|counter|kitchen|bonsai)           echo images_2 ;;
  esac
}

budget_for() {
  "$MESH_SPLATTING_PYTHON" -c '
import json, sys
table = json.load(open(sys.argv[1]))
print(table["rows"][sys.argv[2]]["ours_quality"]["triangles"])
  ' "$FORMAL_ROOT/formal_table.json" "$1"
}

for SCENE in "${SCENES[@]}"; do
  BUDGET="$(budget_for "$SCENE")"
  CUT="$RUNS/cutb__${SCENE}"
  IMAGES="$(images_for "$SCENE")"

  if [ ! -f "$CUT/survival.json" ]; then
    rm -rf "$CUT"
    echo "== cutting $SCENE to $BUDGET faces"
    "$MESH_SPLATTING_PYTHON" -u -m sota.survival_cleanup \
      -s "$DATA_ROOT/$SCENE" -m "$RUNS/integrated__${SCENE}" -i "$IMAGES" --eval \
      --out "$CUT" --budget "$BUDGET"
  fi

  OUTPUT="$CUT/eval_matched_oats"
  [ -f "$OUTPUT/DONE" ] && { echo "== matched_oats/$SCENE already evaluated"; continue; }
  "$MESH_SPLATTING_PYTHON" -u -m sota.main_table_eval \
    -s "$DATA_ROOT/$SCENE" -m "$CUT/matched_oats" -i "$IMAGES" --eval \
    --scene "$SCENE" --arm ours_quality --iteration 30000 --output "$OUTPUT"
done

"$MESH_SPLATTING_PYTHON" - "$FORMAL_ROOT/formal_table.json" "$RUNS" "${SCENES[@]}" <<'PY'
import json, sys
from pathlib import Path

reference = json.load(open(sys.argv[1]))["rows"]
runs = Path(sys.argv[2])
scenes = sys.argv[3:]

rows, matched_gain, free_gain = [], [], []
for scene in scenes:
    v1 = reference[scene]["ours_quality"]
    matched = json.loads((runs / f"cutb__{scene}" / "eval_matched_oats" / "result.json").read_text())
    free = json.loads((runs / f"cut__{scene}" / "main_ours_quality" / "result.json").read_text())
    rows.append({
        "scene": scene,
        "v1_psnr": v1["psnr"], "v1_faces": v1["triangles"],
        "matched_psnr": matched["metrics"]["psnr"], "matched_faces": matched["triangles"],
        "matched_ssim": matched["metrics"]["ssim"], "matched_lpips_vgg": matched["metrics"]["lpips_vgg"],
        "free_psnr": free["metrics"]["psnr"], "free_faces": free["triangles"],
    })
    matched_gain.append(matched["metrics"]["psnr"] - v1["psnr"])
    free_gain.append(free["metrics"]["psnr"] - v1["psnr"])

report = {
    "experiment": "softtail-nine-scene-equal-budget",
    "rows": rows,
    "mean_psnr_gain_db": {
        "matched_to_v1_budget": sum(matched_gain) / len(matched_gain),
        "own_budget": sum(free_gain) / len(free_gain),
    },
    "psnr_wins": {
        "matched_to_v1_budget": sum(1 for d in matched_gain if d > 0),
        "own_budget": sum(1 for d in free_gain if d > 0),
    },
    "scenes": len(scenes),
}
report["capacity_share"] = (
    (report["mean_psnr_gain_db"]["own_budget"] - report["mean_psnr_gain_db"]["matched_to_v1_budget"])
    / report["mean_psnr_gain_db"]["own_budget"])
(runs / "equal_budget_nine.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

print(json.dumps(report["mean_psnr_gain_db"], indent=2))
print(json.dumps(report["psnr_wins"], indent=2))
print(f"capacity share of the gain: {report['capacity_share']:.1%}")
for row in rows:
    print(f"{row['scene']:9s} v1 {row['v1_psnr']:.4f}/{row['v1_faces']} "
          f"matched {row['matched_psnr']:.4f}/{row['matched_faces']} "
          f"own {row['free_psnr']:.4f}/{row['free_faces']}")
PY

touch "$RUNS/EQUAL_BUDGET_NINE_DONE"
echo "Nine-scene equal-budget recheck complete: $RUNS/equal_budget_nine.json"
