#!/bin/bash
set -euo pipefail

OUT=${1:-output/geogauge_g0}
DEVICE=${2:-cpu}
PYTHON=${PYTHON:-python3}
mkdir -p "$OUT"

"$PYTHON" verify_geogauge_reference.py
"$PYTHON" geogauge_eval.py --device "$DEVICE" --out "$OUT/geogauge_g0.json"
