#!/usr/bin/env bash
# Batch 4 -- is the hardening loss about coverage spent, or iterations available?
#
#   sota/batch4.sh garden
#
# Batch 1 showed the loss cannot be moved: spend the coverage earlier and the
# peak drops by what the tail saves. That is consistent with two different
# stories. Either every unit of coverage costs quality outright, in which case a
# longer run changes nothing, or the model simply never gets enough updates at
# each hardness to re-fit, in which case stretching the same schedule over more
# iterations recovers some of it. `lr60k` argued against the second by giving the
# geometry a larger step instead of more steps, and made things much worse -- but
# a larger step and more steps are not the same intervention.
#
# Every schedule point scales together so the phases keep their proportions;
# only the number of updates changes. One exception is out of reach without a new
# parameter: train.py switches to the final supersampling factor at a fixed
# `start_upsampling + 5000`, so the last phase grows in relative length rather
# than staying at a sixth of the run.
set -euo pipefail

SCENE=${1:-garden}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

stretched () {  # factor -> the whole published schedule, scaled
  local n=$1
  echo "--iterations $((30000 * n / 3))
        --sigma_until $((30000 * n / 3))
        --position_lr_max_steps $((30000 * n / 3))
        --densify_until_iter $((10000 * n / 3))
        --final_opacity_iter $((24000 * n / 3))
        --start_upsampling $((20000 * n / 3))
        --test_iterations $((30000 * n / 3))"
}

"$HERE/run.sh" long50k "$SCENE" $(stretched 5)
"$HERE/run.sh" long90k "$SCENE" $(stretched 9)

python "$HERE/table.py"
