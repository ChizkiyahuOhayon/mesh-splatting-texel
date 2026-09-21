#!/usr/bin/env bash
# Nine-scene component ablation: where does the integral have to be applied?
#
# The method uses one statistic in two places. The main table runs both at once,
# so it cannot say which half carries the gain. This script scores the third
# corner of the square on all nine scenes:
#
#   train    cleanup   arm                       source
#   peak     peak      v1 (control)              formal_main_table_01 (archived)
#   integral peak      training-time OATS only   cut__<scene>/v1   <- evaluated here
#   peak     integral  cleanup-time OATS only    softtail_v2_oats_01 (3 scenes)
#   integral integral  full method               cut__<scene>/main_ours_quality
#
# The training-only arm needs no training: survival_cleanup already wrote the
# published peak-rule cut of every integral-trained run as cut__<scene>/v1, at
# the same face budget as the integral cut beside it. This is therefore an
# evaluation-only pass over nine checkpoints.
#
# Single GPU, resumable at every step.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/ensure_environment.sh"

NAS_ROOT=${NAS_ROOT:-/home/smbu/dy/nas/meshsplatting_smbu}
DATA_ROOT=${DATA_ROOT:-/home/smbu/dy/nas/dy/mesh-splatting/data/mipnerf360}
FORMAL_ROOT=${FORMAL_ROOT:-$NAS_ROOT/experiments/formal_main_table_01}
OATS_ROOT=${OATS_ROOT:-$NAS_ROOT/experiments/softtail_v2_oats_01}
RUNS=${RUNS:-$NAS_ROOT/experiments/softtail_integrated_01}
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU

SCENES=(bicycle flowers garden stump treehill room counter kitchen bonsai)

test -f "$FORMAL_ROOT/formal_table.json"
[ -f "$RUNS/ABLATION_DONE" ] && { echo "Nine-scene ablation already complete"; exit 0; }

images_for() {
  case "$1" in
    bicycle|flowers|garden|stump|treehill) echo images_4 ;;
    room|counter|kitchen|bonsai)           echo images_2 ;;
  esac
}

for SCENE in "${SCENES[@]}"; do
  CUT="$RUNS/cut__${SCENE}"
  test -f "$CUT/v1/point_cloud/iteration_30000/point_cloud_state_dict.pt" \
    || { echo "missing peak-rule cut for $SCENE" >&2; exit 1; }

  OUTPUT="$CUT/abl_train_only"
  [ -f "$OUTPUT/DONE" ] && { echo "== train_only/$SCENE already evaluated"; continue; }
  "$MESH_SPLATTING_PYTHON" -u -m sota.main_table_eval \
    -s "$DATA_ROOT/$SCENE" -m "$CUT/v1" -i "$(images_for "$SCENE")" --eval \
    --scene "$SCENE" --arm ours_quality --iteration 30000 --output "$OUTPUT"
done

"$MESH_SPLATTING_PYTHON" - "$FORMAL_ROOT/formal_table.json" "$OATS_ROOT/gate.json" "$RUNS" "${SCENES[@]}" <<'PY'
import json, sys
from pathlib import Path

reference = json.load(open(sys.argv[1]))["rows"]
oats_gate = json.load(open(sys.argv[2]))
runs = Path(sys.argv[3])
scenes = sys.argv[4:]
metrics = ("psnr", "ssim", "lpips_vgg")

def read(path):
    result = json.loads(Path(path).read_text())
    return {m: result["metrics"][m] for m in metrics} | {"triangles": result["triangles"]}

rows = {}
for scene in scenes:
    cut = runs / f"cut__{scene}"
    v1 = reference[scene]["ours_quality"]
    rows[scene] = {
        "v1": {m: v1[m] for m in metrics} | {"triangles": v1["triangles"]},
        "train_only": read(cut / "abl_train_only" / "result.json"),
        "full": read(cut / "main_ours_quality" / "result.json"),
    }

arms = ("v1", "train_only", "full")
means = {arm: {m: sum(rows[s][arm][m] for s in scenes) / len(scenes) for m in metrics}
         for arm in arms}
report = {
    "experiment": "softtail-nine-scene-ablation",
    "scenes": list(scenes),
    "rows": rows,
    "means": means,
    "psnr_gain_vs_v1": {arm: means[arm]["psnr"] - means["v1"]["psnr"] for arm in arms},
    "psnr_wins_vs_v1": {
        arm: sum(1 for s in scenes if rows[s][arm]["psnr"] > rows[s]["v1"]["psnr"])
        for arm in arms},
}

# The fourth corner, measured earlier on room/bicycle/garden: the same integral
# applied only in the cleanup, plus the shuffle control that re-picks the same
# number of faces at random.
cleanup = oats_gate["per_scene"]
cleanup_scenes = sorted(cleanup)
report["cleanup_only_three_scene"] = {
    "scenes": cleanup_scenes,
    "means": {arm: {m: sum(cleanup[s][arm][m] for s in cleanup_scenes) / len(cleanup_scenes)
                    for m in metrics}
              for arm in ("frozen_v1", "peak", "integrated", "shuffle")},
}
three = report["cleanup_only_three_scene"]["means"]
report["cleanup_only_three_scene"]["psnr_gain_vs_peak"] = {
    arm: three[arm]["psnr"] - three["peak"]["psnr"] for arm in ("integrated", "shuffle")}

(runs / "ablation_nine.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

for arm in arms:
    print(f"{arm:11s} PSNR {means[arm]['psnr']:.4f} "
          f"({report['psnr_gain_vs_v1'][arm]:+.4f}) "
          f"SSIM {means[arm]['ssim']:.4f} LPIPS {means[arm]['lpips_vgg']:.4f} "
          f"wins {report['psnr_wins_vs_v1'][arm]}/{len(scenes)}")
print("cleanup-only (3 scenes): "
      f"integral {report['cleanup_only_three_scene']['psnr_gain_vs_peak']['integrated']:+.4f}, "
      f"shuffle {report['cleanup_only_three_scene']['psnr_gain_vs_peak']['shuffle']:+.4f}")
PY

touch "$RUNS/ABLATION_DONE"
echo "Nine-scene ablation complete: $RUNS/ablation_nine.json"
