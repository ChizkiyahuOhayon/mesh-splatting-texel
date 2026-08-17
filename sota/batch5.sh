#!/usr/bin/env bash
# Batch 5 -- the baseline's own quality/size curve.
#
#   sota/batch5.sh garden
#
# Every arm so far has been compared at whatever primitive count it happened to
# land on, and the counts differ by up to 36%. Correcting for that with the
# +0.5 dB per doubling from their own ablation turns three arms positive, which
# is an extrapolation and exactly what a reviewer would refuse to accept.
#
# So measure it instead. `max_points` caps the vertex set during densification,
# so sweeping it walks the baseline along its own frontier and every arm can then
# be read against a curve from this codebase, this scene, this protocol.
#
# `base` already supplies the uncapped point: 3,113,190 vertices / 6,659,278
# triangles at 24.7217 dB.
set -euo pipefail

SCENE=${1:-garden}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$HERE/run.sh" cap28 "$SCENE" --max_points 2800000
"$HERE/run.sh" cap24 "$SCENE" --max_points 2400000
"$HERE/run.sh" cap20 "$SCENE" --max_points 2000000

python "$HERE/table.py"
