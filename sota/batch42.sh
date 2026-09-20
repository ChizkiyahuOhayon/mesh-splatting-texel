#!/usr/bin/env bash
# Nine-scene Mip-NeRF360 main table for the full method.
#
# The method is one statistic applied in two places: the integral S_f = sum(alpha * T)
# ranks faces in the 4k-11k pruning and densification passes (--integrated_importance,
# +0.1064 dB at an equal budget), and again in the final cleanup (OATS, sota/survival_cleanup.py).
# Together they measured +0.1388 dB on room, bicycle and garden.
#
# room, bicycle and garden are already trained and cut in this root, so this
# script trains the remaining six scenes and evaluates all nine uniformly.
# Control: the frozen v1 rows in formal_main_table_01. Single GPU, resumable at
# every step - re-running it picks up wherever it stopped.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/ensure_environment.sh"

NAS_ROOT=${NAS_ROOT:-/home/smbu/dy/nas/meshsplatting_smbu}
DATA_ROOT=${DATA_ROOT:-/home/smbu/dy/nas/dy/mesh-splatting/data/mipnerf360}
FORMAL_ROOT=${FORMAL_ROOT:-$NAS_ROOT/experiments/formal_main_table_01}
RUNS=${RUNS:-$NAS_ROOT/experiments/softtail_integrated_01}
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU DATA_ROOT RUNS

# The published nine, in the order the frozen table lists them.
SCENES=(bicycle flowers garden stump treehill room counter kitchen bonsai)
ARMS=(ours_quality ours_speed)

REPO="$(cd "$HERE/.." && pwd)"
test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" || {
  echo "tracked source changes are present in $REPO" >&2
  exit 1
}
test -f "$FORMAL_ROOT/formal_table.json"

images_for() {
  case "$1" in
    bicycle|flowers|garden|stump|treehill) echo images_4 ;;
    room|counter|kitchen|bonsai)           echo images_2 ;;
  esac
}

for SCENE in "${SCENES[@]}"; do
  test -d "$DATA_ROOT/$SCENE" || { echo "missing dataset: $DATA_ROOT/$SCENE" >&2; exit 1; }
done

[ -f "$RUNS/MAIN_TABLE_DONE" ] && { echo "Nine-scene main table already complete"; exit 0; }

for SCENE in "${SCENES[@]}"; do
  # 1. Train. run.sh skips a scene that already carries DONE, which is how the
  #    three gate scenes are reused instead of retrained.
  "$HERE/run.sh" integrated "$SCENE" --final_opacity 0.8 --integrated_importance --save_precleanup

  "$MESH_SPLATTING_PYTHON" -c '
import sys, torch
state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert float(state["opacity_floor"]) == 0.8, state["opacity_floor"]
assert state.get("integrated_importance") is True, "checkpoint did not train the integrated statistic"
assert state.get("elastic_window") is not True, "elastic window must be off"
assert state.get("opacity_field") is not True, "opacity field must be off"
  ' "$RUNS/integrated__${SCENE}/point_cloud/iteration_30000/point_cloud_state_dict.pt"

  # 2. Cleanup by integrated survival, offline on the saved pre-cleanup state.
  #    Same face budget as the published rule; only the survivors differ.
  CUT="$RUNS/cut__${SCENE}"
  if [ ! -f "$CUT/survival.json" ]; then
    rm -rf "$CUT"
    "$MESH_SPLATTING_PYTHON" -u -m sota.survival_cleanup \
      -s "$DATA_ROOT/$SCENE" -m "$RUNS/integrated__${SCENE}" -i "$(images_for "$SCENE")" --eval \
      --out "$CUT"
  fi

  # 3. Score the full method: the oats copy, under the frozen evaluator.
  for ARM in "${ARMS[@]}"; do
    OUTPUT="$CUT/main_${ARM}"
    [ -f "$OUTPUT/DONE" ] && { echo "== $ARM/$SCENE already evaluated"; continue; }
    "$MESH_SPLATTING_PYTHON" -u -m sota.main_table_eval \
      -s "$DATA_ROOT/$SCENE" -m "$CUT/oats" -i "$(images_for "$SCENE")" --eval \
      --scene "$SCENE" --arm "$ARM" --iteration 30000 --output "$OUTPUT"
  done
done

"$MESH_SPLATTING_PYTHON" - "$FORMAL_ROOT/formal_table.json" "$RUNS" "${SCENES[@]}" <<'PY'
import json, sys
from pathlib import Path

reference = json.load(open(sys.argv[1]))["rows"]
runs = Path(sys.argv[2])
scenes = sys.argv[3:]
metrics = ("psnr", "ssim", "lpips_vgg", "l1", "fps")

rows = {}
for scene in scenes:
    row = {}
    for arm in ("ours_quality", "ours_speed"):
        result = json.loads((runs / f"cut__{scene}" / f"main_{arm}" / "result.json").read_text())
        row[arm] = {m: result["metrics"][m] for m in metrics} | {
            "triangles": result["triangles"],
            "vertices": result["vertices"],
            "checkpoint_bytes": result["checkpoint_bytes"],
        }
    rows[scene] = row

def mean(rows_, arm, metric, table=None):
    values = []
    for scene in scenes:
        source = table[scene][arm] if table else rows_[scene][arm]
        values.append(source[metric])
    return sum(values) / len(values)

summary = {
    "experiment": "softtail-nine-scene-main-table",
    "scenes": list(scenes),
    "reference": sys.argv[1],
    "rows": rows,
    "means": {
        arm: {m: mean(rows, arm, m) for m in metrics}
        for arm in ("ours_quality", "ours_speed")
    },
    "reference_means": {
        arm: {m: mean(None, arm, m, reference) for m in metrics}
        for arm in ("stock", "ours_quality", "ours_speed")
    },
}
summary["deltas_vs_v1"] = {
    arm: {m: summary["means"][arm][m] - summary["reference_means"][arm][m] for m in metrics}
    for arm in ("ours_quality", "ours_speed")
}
summary["psnr_wins_vs_v1"] = {
    arm: sum(1 for s in scenes if rows[s][arm]["psnr"] > reference[s][arm]["psnr"])
    for arm in ("ours_quality", "ours_speed")
}
(runs / "main_table.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

for arm in ("ours_quality", "ours_speed"):
    mine, theirs = summary["means"][arm], summary["reference_means"][arm]
    print(f"{arm}: PSNR {theirs['psnr']:.4f} -> {mine['psnr']:.4f} "
          f"({summary['deltas_vs_v1'][arm]['psnr']:+.4f}), "
          f"SSIM {mine['ssim']:.4f}, LPIPS {mine['lpips_vgg']:.4f}, "
          f"wins {summary['psnr_wins_vs_v1'][arm]}/{len(scenes)}")
print(f"stock MeshSplatting mean PSNR {summary['reference_means']['stock']['psnr']:.4f}")
PY

touch "$RUNS/MAIN_TABLE_DONE"
echo "Nine-scene main table complete: $RUNS/main_table.json"
