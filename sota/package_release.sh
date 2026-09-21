#!/usr/bin/env bash
# Package the released SoftTail meshes, one archive per scene, with hashes.
#
# Each archive holds exactly what sota/main_table_eval.py needs to score a
# scene: point_cloud/iteration_30000/point_cloud_state_dict.pt beside the run's
# cfg_args and cameras.json. Machine-readable results are not packaged here -
# they live in results/formal/ inside the repository.
#
#   bash sota/package_release.sh [destination]
#
# Then upload the destination directory (about 8.9 GB) to the release host,
# e.g. with rclone once a remote named "gdrive" is configured:
#
#   rclone config                       # one-off, needs a browser
#   rclone copy <destination> gdrive:"SoftTail Release/checkpoints" -P
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NAS_ROOT=${NAS_ROOT:-/home/smbu/dy/nas/meshsplatting_smbu}
M360_RUNS=${M360_RUNS:-$NAS_ROOT/experiments/softtail_integrated_01}
TRANSFER_RUNS=${TRANSFER_RUNS:-$NAS_ROOT/experiments/softtail_tandt_db_01}
DEST=${1:-$NAS_ROOT/release/softtail_checkpoints}
ITERATION=${ITERATION:-30000}

M360_SCENES=(bicycle flowers garden stump treehill room counter kitchen bonsai)
TRANSFER_SCENES=(train truck drjohnson playroom)

mkdir -p "$DEST"

package() {
  local scene=$1 runs=$2
  local model="$runs/cut__${scene}/oats"
  local archive="$DEST/softtail_${scene}.tar"
  test -f "$model/point_cloud/iteration_${ITERATION}/point_cloud_state_dict.pt" \
    || { echo "missing released mesh for $scene" >&2; exit 1; }
  [ -f "$archive" ] && { echo "== $scene already packaged"; return; }
  # Named after the scene inside the archive, so it untars into its own model
  # directory rather than a path that says "oats".
  local staging
  staging="$(mktemp -d)"
  trap 'rm -rf "$staging"' RETURN
  mkdir -p "$staging/softtail_${scene}/point_cloud"
  cp "$model/cfg_args" "$model/cameras.json" "$staging/softtail_${scene}/"
  cp -r "$model/point_cloud/iteration_${ITERATION}" \
        "$staging/softtail_${scene}/point_cloud/"
  tar -cf "$archive" -C "$staging" "softtail_${scene}"
  echo "== packaged $scene ($(du -h "$archive" | cut -f1))"
}

for SCENE in "${M360_SCENES[@]}"; do package "$SCENE" "$M360_RUNS"; done
for SCENE in "${TRANSFER_SCENES[@]}"; do package "$SCENE" "$TRANSFER_RUNS"; done

(cd "$DEST" && sha256sum softtail_*.tar > SHA256SUMS)
echo "release ready: $DEST ($(du -sh "$DEST" | cut -f1))"
echo "verify with: (cd $DEST && sha256sum -c SHA256SUMS)"
