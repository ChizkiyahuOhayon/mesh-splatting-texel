#!/usr/bin/env bash
# Train one arm on one scene and record its final test metrics.
#
#   sota/run.sh <arm> <scene> [extra train.py args...]
#
# The scene decides the two settings the official protocol ties to it: the image
# pyramid level and the indoor hyperparameter override (full_eval.py). Everything
# else an experiment wants to change is passed through verbatim, so an arm is
# fully described by its name plus the arguments after it.
set -euo pipefail

ARM=${1:?usage: sota/run.sh <arm> <scene> [train args...]}
SCENE=${2:?usage: sota/run.sh <arm> <scene> [train args...]}
shift 2

DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/data/m360}
RUNS=${RUNS:-/root/autodl-tmp/runs}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case "$SCENE" in
  bicycle|flowers|garden|stump|treehill) PROTOCOL=(-i images_4) ;;
  room|counter|kitchen|bonsai)           PROTOCOL=(-i images_2 --indoor) ;;
  *) echo "unknown scene '$SCENE' (not a Mip-NeRF360 scene)" >&2; exit 1 ;;
esac

OUT="$RUNS/${ARM}__${SCENE}"
[ -e "$OUT/DONE" ] && { echo "== $ARM/$SCENE already done"; exit 0; }
rm -rf "$OUT"; mkdir -p "$OUT"

cd "$REPO"
printf '%s\n' "$*" > "$OUT/args.txt"

START=$SECONDS
set +e
python train.py -s "$DATA_ROOT/$SCENE" -m "$OUT" --eval "${PROTOCOL[@]}" "$@" \
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
  tail -1 "$OUT/metrics.txt"
else
  touch "$OUT/FAILED"
  echo "== $ARM/$SCENE FAILED (exit $STATUS) after ${ELAPSED}s" >&2
  tail -25 "$OUT/train.log" >&2
  exit 1
fi
