#!/usr/bin/env bash
# Batch 2 -- do faces want different hardening clocks?
#
#   sota/batch2.sh garden
#
# Each face scales the published schedule's remaining distance to the endpoint
# (sota/hardness.py), starting at rate 1 -- the schedule itself -- so the run
# begins identical to `base` and any movement is the loss asking for it. Batch 1
# closed the *global* schedule; this asks whether faces want different clocks.
#
# The rates are held to a fixed budget -- bounded by `spread` and normalised to
# mean one -- because a free rate does not reorder hardening, it postpones all of
# it: measured on Garden, 93.6% of faces went softer, the median rate reached
# 89.5, and the run sat at 25.83 dB one iteration before the endpoint forced it
# to 21.47. `spread` is therefore the knob that matters, and 1.0 would be `base`.
set -euo pipefail

SCENE=${1:-garden}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$HERE/run.sh" hard      "$SCENE" --face_hardness
"$HERE/run.sh" hard_wide "$SCENE" --face_hardness --face_hardness_spread 16
"$HERE/run.sh" hard_slow "$SCENE" --face_hardness --face_hardness_lr 0.002

python "$HERE/table.py"
