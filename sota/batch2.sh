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
# The only free choice is the step size, so that is what the sweep covers.
set -euo pipefail

SCENE=${1:-garden}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$HERE/run.sh" hard      "$SCENE" --face_hardness
"$HERE/run.sh" hard_slow "$SCENE" --face_hardness --face_hardness_lr 0.002
"$HERE/run.sh" hard_fast "$SCENE" --face_hardness --face_hardness_lr 0.05

python "$HERE/table.py"
