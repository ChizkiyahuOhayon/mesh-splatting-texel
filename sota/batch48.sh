#!/usr/bin/env bash
# Opaque SoftTail gate: the integrated survival statistic alone, at the
# published terminal opacity (0.9999), so the output stays a fully opaque mesh
# that any rasterizer can draw.
#
# Arm: stock MeshSplatting + --integrated_importance, then the OATS cleanup.
# Control: the frozen stock rows of formal_main_table_01 (same code, same
# evaluator arm `stock`, i.e. opacity 0.9999, 4x, no tail absorption).
#
# Pre-set decision: extend to all 13 scenes only if the mean PSNR gain over the
# control is >= +0.10 dB with 3/3 wins; otherwise report nothing beyond this
# gate. Single GPU, resumable.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/ensure_environment.sh"
PY=$MESH_SPLATTING_PYTHON

NAS_ROOT=${NAS_ROOT:-/home/smbu/dy/nas/meshsplatting_smbu}
DATA_ROOT=${DATA_ROOT:-/home/smbu/dy/nas/dy/mesh-splatting/data/mipnerf360}
FORMAL_ROOT=${FORMAL_ROOT:-$NAS_ROOT/experiments/formal_main_table_01}
RUNS=${RUNS:-$NAS_ROOT/experiments/softtail_opaque_01}
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU DATA_ROOT RUNS

SCENES=(room bicycle garden)

REPO="$(cd "$HERE/.." && pwd)"
test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" || {
  echo "tracked source changes are present in $REPO" >&2; exit 1; }
test -f "$FORMAL_ROOT/formal_table.json"
cd "$REPO"

images_for() {
  case "$1" in
    bicycle|flowers|garden|stump|treehill) echo images_4 ;;
    room|counter|kitchen|bonsai)           echo images_2 ;;
  esac
}

[ -f "$RUNS/GATE_DONE" ] && { echo "opaque gate already complete"; exit 0; }

for SCENE in "${SCENES[@]}"; do
  "$HERE/run.sh" opaqueint "$SCENE" --integrated_importance --save_precleanup
  "$PY" -c '
import sys, torch
state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert float(state["opacity_floor"]) == 0.9999, state["opacity_floor"]
assert state.get("integrated_importance") is True
' "$RUNS/opaqueint__${SCENE}/point_cloud/iteration_30000/point_cloud_state_dict.pt"

  CUT="$RUNS/cut__${SCENE}"
  [ -f "$CUT/survival.json" ] || { rm -rf "$CUT"; \
    "$PY" -u -m sota.survival_cleanup -s "$DATA_ROOT/$SCENE" -i "$(images_for "$SCENE")" --eval \
      -m "$RUNS/opaqueint__${SCENE}" --out "$CUT" --quiet; }
  OUTPUT="$CUT/eval_opaque"
  if [ ! -f "$OUTPUT/DONE" ]; then
    rm -rf "$OUTPUT"
    "$PY" -u -m sota.main_table_eval -s "$DATA_ROOT/$SCENE" -i "$(images_for "$SCENE")" --eval \
      -m "$CUT/oats" --scene "$SCENE" --arm stock --iteration 30000 --output "$OUTPUT" --quiet
  fi
done

"$PY" - "$FORMAL_ROOT/formal_table.json" "$RUNS" "${SCENES[@]}" <<'PY'
import json, sys
from pathlib import Path
reference = json.load(open(sys.argv[1]))["rows"]
runs, scenes = Path(sys.argv[2]), sys.argv[3:]
metrics = ("psnr", "ssim", "lpips_vgg", "fps")
rows = {}
for scene in scenes:
    r = json.loads((runs / f"cut__{scene}" / "eval_opaque" / "result.json").read_text())
    rows[scene] = {"opaque_softtail": {m: r["metrics"][m] for m in metrics} | {"triangles": r["triangles"]},
                   "stock": {m: reference[scene]["stock"][m] for m in metrics}
                            | {"triangles": reference[scene]["stock"]["triangles"]}}
gain = {s: rows[s]["opaque_softtail"]["psnr"] - rows[s]["stock"]["psnr"] for s in scenes}
mean_gain = sum(gain.values()) / len(gain)
wins = sum(g > 0 for g in gain.values())
passed = mean_gain >= 0.10 and wins == len(scenes)
report = {"experiment": "softtail-opaque-gate", "rows": rows, "psnr_gain": gain,
          "mean_psnr_gain": mean_gain, "wins": wins, "rule": ">= +0.10 dB and 3/3 wins",
          "passed": passed}
(runs / "gate.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"psnr_gain": gain, "mean": mean_gain, "wins": wins, "passed": passed}, indent=2))
PY
touch "$RUNS/GATE_DONE"
