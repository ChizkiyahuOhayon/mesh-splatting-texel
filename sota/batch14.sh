#!/usr/bin/env bash
# Third-scene transfer screen: terminal opacity floor 0.8 on Bicycle.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/ensure_environment.sh"

DATA_ROOT=${DATA_ROOT:-/home/smbu/dy/mesh-splatting/data/mipnerf360}
RUNS=${RUNS:-/home/smbu/dy/nas/meshsplatting_smbu/experiments/opacity_floor_01}
export DATA_ROOT RUNS

"$HERE/run.sh" opacity08 bicycle --final_opacity 0.8

CHECKPOINT="$RUNS/opacity08__bicycle/point_cloud/iteration_30000/point_cloud_state_dict.pt"
"$MESH_SPLATTING_PYTHON" -c '
import sys
import torch
state = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
value = float(state["opacity_floor"])
if value != 0.8:
    raise SystemExit(f"checkpoint opacity_floor is {value}, expected 0.8")
print("checkpoint opacity_floor:", value)
' "$CHECKPOINT"
