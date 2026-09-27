#!/usr/bin/env bash
# GPU0 queue after queue01: TS+ generalization, opaque SoftTail on the other
# ten scenes, then supplementary qualitative exports.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E=/home/smbu/dy/nas/meshsplatting_smbu/experiments
until grep -q QUEUE01_DONE "$E/paper_evidence_01/batch47.log" 2>/dev/null; do sleep 120; done
until grep -q MIRROR_DONE /home/smbu/dy/mirror.log 2>/dev/null; do sleep 60; done
mkdir -p "$E/tsplus_01" "$E/qualitative_02"
GPU=0 bash "$HERE/batch49.sh" > "$E/tsplus_01/batch49.log" 2>&1
GPU=0 bash "$HERE/batch50.sh" > "$E/softtail_opaque_01/batch50.log" 2>&1
GPU=0 bash "$HERE/batch51.sh" > "$E/qualitative_02/batch51.log" 2>&1
echo QUEUE02_DONE > "$E/queue02.done"
