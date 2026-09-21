#!/usr/bin/env bash
# Tanks & Temples and Deep Blending for the full method.
#
# Same method and the same steps as the nine-scene table (sota/batch42.sh): the
# integral ranks faces in the 4k-11k passes and again in the final cleanup. The
# control is the frozen v1 run for each dataset, whose rows are already archived
# in formal_tandt_01 and formal_deep_blending_01.
#
# run.sh carries the per-scene protocol, including the T&T primitive caps
# (train 2.5M, truck 2.0M) and the indoor override for Deep Blending. Both
# datasets are evaluated with "-i images" (no pyramid level), exactly as the v1
# runs in batch28/batch31 were.
#
# Single GPU, resumable at every step.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/ensure_environment.sh"

NAS_ROOT=${NAS_ROOT:-/home/smbu/dy/nas/meshsplatting_smbu}
DATA_BASE=${DATA_BASE:-/home/smbu/dy/nas/dy/mesh-splatting/data/tandt}
RUNS=${RUNS:-$NAS_ROOT/experiments/softtail_tandt_db_01}
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU RUNS

SCENES=(train truck drjohnson playroom)
ARMS=(ours_quality ours_speed)

REPO="$(cd "$HERE/.." && pwd)"
test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" || {
  echo "tracked source changes are present in $REPO" >&2
  exit 1
}

# The two datasets sit in sibling directories under the same root.
data_root_for() {
  case "$1" in
    train|truck)            echo "$DATA_BASE/tandt" ;;
    drjohnson|playroom)     echo "$DATA_BASE/db" ;;
  esac
}

for SCENE in "${SCENES[@]}"; do
  test -d "$(data_root_for "$SCENE")/$SCENE" \
    || { echo "missing dataset: $(data_root_for "$SCENE")/$SCENE" >&2; exit 1; }
done

mkdir -p "$RUNS"
[ -f "$RUNS/DONE" ] && { echo "T&T + Deep Blending already complete"; exit 0; }

for SCENE in "${SCENES[@]}"; do
  DATA_ROOT="$(data_root_for "$SCENE")" "$HERE/run.sh" integrated "$SCENE" \
    --final_opacity 0.8 --integrated_importance --save_precleanup

  "$MESH_SPLATTING_PYTHON" -c '
import sys, torch
state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert float(state["opacity_floor"]) == 0.8, state["opacity_floor"]
assert state.get("integrated_importance") is True, "checkpoint did not train the integrated statistic"
assert state.get("elastic_window") is not True, "elastic window must be off"
assert state.get("opacity_field") is not True, "opacity field must be off"
  ' "$RUNS/integrated__${SCENE}/point_cloud/iteration_30000/point_cloud_state_dict.pt"

  CUT="$RUNS/cut__${SCENE}"
  if [ ! -f "$CUT/survival.json" ]; then
    rm -rf "$CUT"
    "$MESH_SPLATTING_PYTHON" -u -m sota.survival_cleanup \
      -s "$(data_root_for "$SCENE")/$SCENE" -m "$RUNS/integrated__${SCENE}" -i images --eval \
      --out "$CUT"
  fi

  for ARM in "${ARMS[@]}"; do
    OUTPUT="$CUT/main_${ARM}"
    [ -f "$OUTPUT/DONE" ] && { echo "== $ARM/$SCENE already evaluated"; continue; }
    "$MESH_SPLATTING_PYTHON" -u -m sota.main_table_eval \
      -s "$(data_root_for "$SCENE")/$SCENE" -m "$CUT/oats" -i images --eval \
      --scene "$SCENE" --arm "$ARM" --iteration 30000 --output "$OUTPUT"
  done
done

"$MESH_SPLATTING_PYTHON" - "$NAS_ROOT" "$RUNS" <<'PY'
import json, sys
from pathlib import Path

nas, runs = Path(sys.argv[1]), Path(sys.argv[2])
groups = {
    "tanks_and_temples": (["train", "truck"],
                          nas / "experiments/formal_tandt_01/formal_table.json"),
    "deep_blending": (["drjohnson", "playroom"],
                      nas / "experiments/formal_deep_blending_01/formal_table.json"),
}
metrics = ("psnr", "ssim", "lpips_vgg", "l1", "fps")

summary = {"experiment": "softtail-tandt-deep-blending", "groups": {}}
for name, (scenes, reference_path) in groups.items():
    reference = json.loads(reference_path.read_text())["rows"] if reference_path.exists() else None
    rows = {}
    for scene in scenes:
        rows[scene] = {}
        for arm in ("ours_quality", "ours_speed"):
            result = json.loads((runs / f"cut__{scene}" / f"main_{arm}" / "result.json").read_text())
            rows[scene][arm] = {m: result["metrics"][m] for m in metrics} | {
                "triangles": result["triangles"]}
    block = {"scenes": scenes, "rows": rows,
             "means": {arm: {m: sum(rows[s][arm][m] for s in scenes) / len(scenes) for m in metrics}
                       for arm in ("ours_quality", "ours_speed")}}
    if reference:
        block["reference_means"] = {
            arm: {m: sum(reference[s][arm][m] for s in scenes) / len(scenes) for m in metrics}
            for arm in ("stock", "ours_quality", "ours_speed")}
        block["deltas_vs_v1"] = {
            arm: {m: block["means"][arm][m] - block["reference_means"][arm][m] for m in metrics}
            for arm in ("ours_quality", "ours_speed")}
        block["psnr_wins_vs_v1"] = {
            arm: sum(1 for s in scenes
                     if rows[s][arm]["psnr"] > reference[s][arm]["psnr"])
            for arm in ("ours_quality", "ours_speed")}
    summary["groups"][name] = block

(runs / "tandt_db_table.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
for name, block in summary["groups"].items():
    mine = block["means"]["ours_quality"]
    line = f"{name}: PSNR {mine['psnr']:.4f} SSIM {mine['ssim']:.4f} LPIPS {mine['lpips_vgg']:.4f}"
    if "deltas_vs_v1" in block:
        line += (f" (v1 {block['reference_means']['ours_quality']['psnr']:.4f}, "
                 f"{block['deltas_vs_v1']['ours_quality']['psnr']:+.4f}, "
                 f"wins {block['psnr_wins_vs_v1']['ours_quality']}/{len(block['scenes'])})")
    print(line)
PY

touch "$RUNS/DONE"
echo "T&T + Deep Blending complete: $RUNS/tandt_db_table.json"
