#!/usr/bin/env bash
# Re-evaluate the frozen main table at each scene's training resolution.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAS_ROOT=${NAS_ROOT:-/home/smbu/dy/nas/meshsplatting_smbu}
export ROOT=${ROOT:-$NAS_ROOT/experiments/main_table_02}
exec "$HERE/batch19.sh"
