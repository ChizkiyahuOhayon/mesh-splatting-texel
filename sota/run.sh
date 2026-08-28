#!/usr/bin/env bash
# Train one arm on one scene and record its final test metrics.
#
#   sota/run.sh <arm> <scene> [extra train.py args...]
#
# The scene decides the settings fixed by the published protocol: image pyramid,
# indoor override, and the T&T scene-specific primitive cap. Everything else an
# experiment wants to change is passed through verbatim, so an arm is fully
# described by its name plus the arguments after it.
set -euo pipefail

ARM=${1:?usage: sota/run.sh <arm> <scene> [train args...]}
SCENE=${2:?usage: sota/run.sh <arm> <scene> [train args...]}
shift 2

DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/data/m360}
RUNS=${RUNS:-/root/autodl-tmp/runs}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_PYTHON=${MESH_SPLATTING_PYTHON:-python}

case "$SCENE" in
  bicycle|flowers|garden|stump|treehill) PROTOCOL=(-i images_4) ;;
  room|counter|kitchen|bonsai)           PROTOCOL=(-i images_2 --indoor) ;;
  train)                                 PROTOCOL=(--max_points 2500000) ;;
  truck)                                 PROTOCOL=(--max_points 2000000) ;;
  *) echo "unknown supported scene '$SCENE'" >&2; exit 1 ;;
esac

OUT="$RUNS/${ARM}__${SCENE}"
[ -e "$OUT/DONE" ] && { echo "== $ARM/$SCENE already done"; exit 0; }

# Refuse duplicate writers to the same arm/scene while allowing independent
# scenes to use different GPUs.
mkdir -p "$RUNS"
LOCK="$RUNS/.${ARM}__${SCENE}.lock"
exec 9> "$LOCK"
flock -n 9 || { echo "== another run owns $ARM/$SCENE; refusing" >&2; exit 1; }

rm -rf "$OUT"; mkdir -p "$OUT"

cd "$REPO"
git rev-parse HEAD > "$OUT/source_revision.txt"
printf '%s\n' "$*" > "$OUT/args.txt"

START=$SECONDS
set +e
# -u so the evaluation lines reach the log as they happen rather than sitting in
# a block buffer that is lost if the run is interrupted. expandable_segments
# keeps the allocator from fragmenting at the supersampling change, which is
# where a 24 GB card runs out.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$TRAIN_PYTHON" -u train.py -s "$DATA_ROOT/$SCENE" -m "$OUT" --eval "${PROTOCOL[@]}" "$@" \
  > "$OUT/train.log" 2>&1
STATUS=$?
set -e
ELAPSED=$((SECONDS - START))

# One line per evaluated iteration, e.g.
#   [ITER 30000] Evaluating test: L1 ... PSNR ... SSIM ... LPIPS ... FPS ...
grep -E "^\[ITER .*\] Evaluating" "$OUT/train.log" > "$OUT/metrics.txt" || true

if [ "$STATUS" -eq 0 ] && grep -q "Evaluating test" "$OUT/metrics.txt"; then
  echo "$ELAPSED" > "$OUT/DONE"
  echo "== $ARM/$SCENE OK in ${ELAPSED}s"
  grep "Evaluating test" "$OUT/metrics.txt" | tail -1
else
  touch "$OUT/FAILED"
  echo "== $ARM/$SCENE FAILED (exit $STATUS) after ${ELAPSED}s" >&2
  tail -25 "$OUT/train.log" >&2
  exit 1
fi
