#!/usr/bin/env bash
# Storage-paid spatial-detail pilot. Room is the transfer gate: replace degree-3
# vertex SH with degree-2 SH and spend the saved bytes on order-2 face texels.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENE=${1:-room}

source "$HERE/ensure_environment.sh"
"$HERE/run.sh" tex2_sh2 "$SCENE" \
  --max_points 2800000 --texel_order 2 --sh_degree 2
