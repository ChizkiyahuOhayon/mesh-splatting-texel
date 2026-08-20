#!/usr/bin/env bash
# Select the project interpreter and keep its native extensions in sync with
# this checkout. This file is sourced by experiment launchers.
set -euo pipefail

SOTA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOTA_REPO="$(cd "$SOTA_DIR/.." && pwd)"
DEFAULT_PYTHON=/home/smbu/micromamba/envs/mesh_splatting/bin/python

if [ -n "${MESH_SPLATTING_PYTHON:-}" ]; then
  SOTA_PYTHON=$MESH_SPLATTING_PYTHON
elif [ -x "$DEFAULT_PYTHON" ]; then
  SOTA_PYTHON=$DEFAULT_PYTHON
else
  SOTA_PYTHON=$(command -v python)
fi

[ -x "$SOTA_PYTHON" ] || {
  echo "mesh-splatting Python is not executable: $SOTA_PYTHON" >&2
  return 1
}
export MESH_SPLATTING_PYTHON=$SOTA_PYTHON

if ! "$SOTA_PYTHON" -c 'import rdel' >/dev/null 2>&1; then
  echo "== installing rdel into $SOTA_PYTHON"
  "$SOTA_PYTHON" -m pip install --no-build-isolation --no-deps --no-cache-dir \
    "$SOTA_REPO/submodules/effrdel"
fi

RASTERIZER_SOURCE="$SOTA_REPO/submodules/diff-triangle-mesh-rasterization"
RASTERIZER_TREE=$(
  git -C "$SOTA_REPO" rev-parse HEAD:submodules/diff-triangle-mesh-rasterization
)
RASTERIZER_MARKER=$(
  "$SOTA_PYTHON" -c \
    'import sysconfig; print(sysconfig.get_paths()["purelib"] + "/diff_triangle_rasterization/.mesh_splatting_source")'
)

native_contract() {
  "$SOTA_PYTHON" -c '
import inspect
import diff_triangle_rasterization as rasterizer
assert "screen_space_gradients" in rasterizer.TriangleRasterizationSettings._fields
assert "sigma_face" in inspect.signature(rasterizer.TriangleRasterizer.forward).parameters
assert hasattr(rasterizer._C, "rasterize_triangles")
' >/dev/null 2>&1
}

if [ ! -f "$RASTERIZER_MARKER" ] || \
   [ "$(tr -d '[:space:]' < "$RASTERIZER_MARKER")" != "$RASTERIZER_TREE" ] || \
   ! native_contract; then
  echo "== rebuilding diff_triangle_rasterization from the current checkout"
  "$SOTA_PYTHON" -m pip install --no-build-isolation --no-deps --no-cache-dir \
    --force-reinstall "$RASTERIZER_SOURCE"
  native_contract
  printf '%s\n' "$RASTERIZER_TREE" > "$RASTERIZER_MARKER"
fi

"$SOTA_PYTHON" -c '
import diff_triangle_rasterization as rasterizer
import rdel
import torch
print("python:", torch.__version__, "cuda:", torch.version.cuda)
print("rasterizer:", rasterizer.__file__)
print("native:", rasterizer._C.__file__)
'
