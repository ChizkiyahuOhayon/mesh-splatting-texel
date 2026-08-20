#!/usr/bin/env bash
# Endpoint-forward pilot. Room runs first because Garden-only gains repeatedly
# failed to transfer; Garden and Bicycle are launched only after inspecting Room.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENE=${1:-room}

source "$HERE/ensure_environment.sh"
"$HERE/run.sh" endpoint "$SCENE" --endpoint_supervision
