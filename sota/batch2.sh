#!/usr/bin/env bash
# Batch 2 -- the learned hardening order, and the learning-rate arm it needs.
#
#   sota/batch2.sh garden
#
# `hard` is the method: each face adds its own hardness to the scheduled floor
# (sota/hardness.py). `hard_lr60k` pairs it with a vertex learning rate that has
# not yet collapsed, because a face that hardens early is only useful if the
# geometry can still respond when it does. The two learning-rate arms bracket
# how sensitive the carrier is to its own step size.
set -euo pipefail

SCENE=${1:-garden}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$HERE/run.sh" hard       "$SCENE" --face_hardness
"$HERE/run.sh" hard_lr60k "$SCENE" --face_hardness --position_lr_max_steps 60000
"$HERE/run.sh" hard_slow  "$SCENE" --face_hardness --face_hardness_lr 0.002
"$HERE/run.sh" hard_fast  "$SCENE" --face_hardness --face_hardness_lr 0.05

python "$HERE/table.py"
