#!/usr/bin/env bash
# Quality-first main table: restore four spatial samples per axis globally.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAS_ROOT=${NAS_ROOT:-/home/smbu/dy/nas/meshsplatting_smbu}
export ROOT=${ROOT:-$NAS_ROOT/experiments/main_table_quality_01}
export METHOD_UPSAMPLE=4
exec "$HERE/batch19.sh"
