#!/usr/bin/env bash
# Frozen Room checkpoint: absorb the skipped tail into its terminal surface.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/ensure_environment.sh"

NAS_ROOT=${NAS_ROOT:-/home/smbu/dy/nas/meshsplatting_smbu}
DATA_ROOT=${DATA_ROOT:-/home/smbu/dy/mesh-splatting/data/mipnerf360}
MODEL=${MODEL:-$NAS_ROOT/experiments/opacity_floor_01/opacity08__room}
OUTPUT=${OUTPUT:-$NAS_ROOT/experiments/tail_absorption_01/room}
[ ! -e "$OUTPUT" ] || { echo "output already exists: $OUTPUT" >&2; exit 1; }

cd "$HERE/.."
"$MESH_SPLATTING_PYTHON" -u -m sota.tail_culling \
  -s "$DATA_ROOT/room" -m "$MODEL" -i images_2 --eval --absorb-tail \
  --scene room --iteration 30000 --output "$OUTPUT"
