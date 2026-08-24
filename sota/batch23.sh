#!/usr/bin/env bash
# Balanced main table: factor four with a more aggressive absorbed tail.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAS_ROOT=${NAS_ROOT:-/home/smbu/dy/nas/meshsplatting_smbu}
export ROOT=${ROOT:-$NAS_ROOT/experiments/main_table_balanced_01}
export METHOD_UPSAMPLE=4
export METHOD_THRESHOLD=0.03
exec "$HERE/batch19.sh"
