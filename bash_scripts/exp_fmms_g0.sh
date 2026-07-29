#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 8 ]]; then
  echo "usage: $0 OUT GPU GARDEN_DATA GARDEN_MODEL ROOM_DATA ROOM_MODEL STUMP_DATA STUMP_MODEL" >&2
  exit 2
fi

OUT=$1
GPU=$2
GARDEN_DATA=$3
GARDEN_MODEL=$4
ROOM_DATA=$5
ROOM_MODEL=$6
STUMP_DATA=$7
STUMP_MODEL=$8

export CUDA_VISIBLE_DEVICES="$GPU"

python -c "import diff_triangle_rasterization, nvdiffrast.torch, torch; assert torch.cuda.is_available()"

mkdir -p "$OUT"

python g0_native_eval.py -s "$GARDEN_DATA" -m "$GARDEN_MODEL" --eval \
  --g0_scene garden --g0_output "$OUT/garden"
python g0_native_eval.py -s "$ROOM_DATA" -m "$ROOM_MODEL" --eval \
  --g0_scene room --g0_output "$OUT/room"
python g0_native_eval.py -s "$STUMP_DATA" -m "$STUMP_MODEL" --eval \
  --g0_scene stump --g0_output "$OUT/stump"

python g0_decide.py "$OUT/garden" "$OUT/room" "$OUT/stump" \
  --output "$OUT/decision.json"
