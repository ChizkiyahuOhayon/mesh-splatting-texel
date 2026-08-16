#!/usr/bin/env bash
# The full Mip-NeRF360 benchmark for one arm, in the order full_eval.py uses.
#
#   sota/bench.sh base
#   sota/bench.sh hard --face_hardness
#
# About twelve hours on one 4090. Runs are skipped if already DONE, so an
# interrupted sweep resumes where it stopped.
set -euo pipefail

ARM=${1:?usage: sota/bench.sh <arm> [train args...]}
shift
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for scene in bicycle flowers garden stump treehill room counter kitchen bonsai; do
  "$HERE/run.sh" "$ARM" "$scene" "$@"
done

python "$HERE/table.py"
