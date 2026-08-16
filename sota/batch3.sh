#!/usr/bin/env bash
# Batch 3 -- the exact vertex position gradient, at step sizes that suit it.
#
#   sota/batch3.sh garden
#
# The released backward writes the depth term over the spherical-harmonic
# position gradient and never propagates dL_dpoints2D through the perspective
# projection. Restoring both is the exact derivative, and at the published step
# size it destroys the run (Garden decays to 11.4 dB; see sota/PLAN.md). The
# question this batch asks is whether it is a better gradient once the step size
# is matched to it, so the sweep is over `lr_triangles_points_init` alone.
set -euo pipefail

SCENE=${1:-garden}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXACT=(--screen_space_gradients)

"$HERE/run.sh" exact_lr10   "$SCENE" "${EXACT[@]}" --lr_triangles_points_init 0.00015
"$HERE/run.sh" exact_lr100  "$SCENE" "${EXACT[@]}" --lr_triangles_points_init 0.000015
"$HERE/run.sh" exact_lr1000 "$SCENE" "${EXACT[@]}" --lr_triangles_points_init 0.0000015

python "$HERE/table.py"
