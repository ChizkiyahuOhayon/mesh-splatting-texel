#!/usr/bin/env bash
# Matched three-scene main table: stock versus one frozen method configuration.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/ensure_environment.sh"

NAS_ROOT=${NAS_ROOT:-/home/smbu/dy/nas/meshsplatting_smbu}
DATA_ROOT=${DATA_ROOT:-/home/smbu/dy/mesh-splatting/data/mipnerf360}
ROOT=${ROOT:-$NAS_ROOT/experiments/main_table_01}
METHOD_UPSAMPLE=${METHOD_UPSAMPLE:-3}
METHOD_THRESHOLD=${METHOD_THRESHOLD:-0.01}
[ ! -e "$ROOT" ] || { echo "output already exists: $ROOT" >&2; exit 1; }

cd "$HERE/.."
for SCENE in garden room bicycle; do
  case "$SCENE" in
    garden|bicycle) IMAGES=images_4 ;;
    room)           IMAGES=images_2 ;;
  esac
  STOCK_MODEL="$NAS_ROOT/experiments/sac_g1_${SCENE}_stock_seed0_train"
  if [ "$SCENE" = bicycle ]; then
    STOCK_MODEL="$NAS_ROOT/experiments/opacity_floor_01/stock__bicycle"
  fi
  OUR_MODEL="$NAS_ROOT/experiments/opacity_floor_01/opacity08__${SCENE}"

  "$MESH_SPLATTING_PYTHON" -u -m sota.main_table_eval \
    -s "$DATA_ROOT/$SCENE" -m "$STOCK_MODEL" -i "$IMAGES" --eval \
    --scene "$SCENE" --arm stock --iteration 30000 \
    --output "$ROOT/$SCENE/stock"

  "$MESH_SPLATTING_PYTHON" -u -m sota.main_table_eval \
    -s "$DATA_ROOT/$SCENE" -m "$OUR_MODEL" -i "$IMAGES" --eval \
    --scene "$SCENE" --arm ours --iteration 30000 \
    --method-upsample "$METHOD_UPSAMPLE" \
    --method-threshold "$METHOD_THRESHOLD" \
    --output "$ROOT/$SCENE/ours"
done

"$MESH_SPLATTING_PYTHON" -m sota.main_table "$ROOT"
