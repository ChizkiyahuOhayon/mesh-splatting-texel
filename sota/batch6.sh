#!/usr/bin/env bash
# Batch 6 -- do the two positive arms compose, and does it survive a second scene?
#
#   sota/batch6.sh
#
# `noreg` and `long50k` are the only arms above the baseline's own quality/size
# curve, at +0.221 and +0.242 capacity-matched. They act on unrelated things --
# one drops the normal and vertex-depth priors, the other scales every schedule
# boundary by 5/3 -- so the first question is whether their gains add.
#
# Nothing goes to nine scenes until two scenes agree, so Room follows immediately
# with its own baseline. Room takes the indoor protocol, which sota/run.sh
# applies from the scene name.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STRETCH=(--iterations 50000 --sigma_until 50000 --position_lr_max_steps 50000
         --densify_until_iter 16666 --final_opacity_iter 40000
         --start_upsampling 33333 --test_iterations 50000)
NOREG=(--lambda_normals 0 --lambda_vertex 0)

"$HERE/run.sh" combo garden "${STRETCH[@]}" "${NOREG[@]}"

"$HERE/run.sh" base   room
"$HERE/run.sh" noreg  room "${NOREG[@]}"
"$HERE/run.sh" combo  room "${STRETCH[@]}" "${NOREG[@]}"

python -m sota.frontier garden
python -m sota.frontier room
