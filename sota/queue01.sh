#!/usr/bin/env bash
# GPU0 queue: pause batch47 once its evaluation-only parts (A, B) are done, run
# the opaque gate (batch48) ahead of the long DTU part, then resume batch47.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP=/home/smbu/dy/nas/meshsplatting_smbu/experiments
LOG=$EXP/paper_evidence_01
until [ -f "$LOG/sweep/sweep.json" ]; do
  pgrep -f "bash sota/batch47.sh" >/dev/null || break
  sleep 60
done
for pid in $(pgrep -f "bash sota/batch47.sh"); do
  kill -- -"$(ps -o pgid= -p "$pid" | tr -d ' ')" 2>/dev/null
done
sleep 30
mkdir -p "$EXP/softtail_opaque_01"
GPU=0 bash "$HERE/batch48.sh" > "$EXP/softtail_opaque_01/batch48.log" 2>&1
GPU=0 bash "$HERE/batch47.sh" >> "$LOG/batch47.log" 2>&1
echo QUEUE01_DONE >> "$LOG/batch47.log"
