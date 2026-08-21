#!/usr/bin/env bash
# Transfer the frozen quality-constrained supersampling rule to two scenes.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/ensure_environment.sh"

NAS_ROOT=${NAS_ROOT:-/home/smbu/dy/nas/meshsplatting_smbu}
DATA_ROOT=${DATA_ROOT:-/home/smbu/dy/mesh-splatting/data/mipnerf360}
ROOT=${ROOT:-$NAS_ROOT/experiments/adaptive_supersampling_01}
[ ! -e "$ROOT" ] || { echo "output already exists: $ROOT" >&2; exit 1; }

cd "$HERE/.."
for SCENE in garden bicycle; do
  "$MESH_SPLATTING_PYTHON" -u -m sota.supersampling \
    -s "$DATA_ROOT/$SCENE" \
    -m "$NAS_ROOT/experiments/opacity_floor_01/opacity08__${SCENE}" \
    -i images_2 --eval --scene "$SCENE" --iteration 30000 \
    --output "$ROOT/$SCENE"
done
