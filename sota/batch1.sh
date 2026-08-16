#!/usr/bin/env bash
# Batch 1 -- one scene, five arms, run back to back on a single GPU.
#
#   sota/batch1.sh garden
#
# Every arm differs from `base` in exactly one thing, so whatever moves is
# attributable. The two hardening arms reach the same final sigma as `base` on
# the same iteration; only the path differs (sota/sigma_schedule.py).
set -euo pipefail

SCENE=${1:-garden}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$HERE/run.sh" base   "$SCENE"
"$HERE/run.sh" noreg  "$SCENE" --lambda_normals 0 --lambda_vertex 0
"$HERE/run.sh" lr60k  "$SCENE" --position_lr_max_steps 60000
"$HERE/run.sh" cov    "$SCENE" --sigma_schedule coverage
"$HERE/run.sh" lrm    "$SCENE" --sigma_schedule lrmatched

python "$HERE/table.py"
