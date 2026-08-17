#!/usr/bin/env bash
# Batch 8 -- does trading triangles for per-face detail transfer to a second scene?
#
#   sota/batch8.sh room
#
# On Garden the trade works: at the same primitive cap, adding order-2 texels is
# +0.424 dB over the plain capped run at 4% fewer triangles, which puts a mesh
# 33.8% smaller than the baseline 0.154 dB *above* it, and 0.183 dB above the
# baseline's own storage curve.
#
#     base    6,659,278 triangles   727.5 MB   24.7217 dB
#     cap28   4,594,522 triangles   504.4 MB   24.4517 dB   (on the curve)
#     tex2    4,408,996 triangles   704.5 MB   24.8755 dB   (above it)
#
# Transfer is the question that matters, because it is what killed the schedule
# arms: `noreg` gave +0.221 on Garden and −0.005 on Room, and `combo` gave +0.618
# on Garden and +0.012 on Room. Both pairs are run here so the second scene has
# its own on-curve reference rather than borrowing Garden's.
set -euo pipefail

SCENE=${1:-room}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAP=(--max_points 2800000)

"$HERE/run.sh" cap28 "$SCENE" "${CAP[@]}"
"$HERE/run.sh" tex2  "$SCENE" "${CAP[@]}" --texel_order 2

python -m sota.frontier "$SCENE"
