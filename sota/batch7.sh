#!/usr/bin/env bash
# Batch 7 -- is the primitive count set by appearance bandwidth?
#
#   sota/batch7.sh garden
#
# At the opaque endpoint a pixel is covered by exactly one triangle and its
# colour is the screen-space barycentric blend of three vertex spherical
# harmonics: Gouraud shading. A large triangle can therefore only carry a linear
# colour ramp, so the spatial frequency of appearance is welded to tessellation
# density. Garden ends with 6.66M triangles over roughly 1M pixels -- about 6.6
# triangles per pixel, which no geometric argument asks for.
#
# If that count is paying for appearance rather than for geometry, then giving
# each face its own detail should let a much smaller mesh match the baseline.
# The comparison is against two measured points on this scene:
#
#     base    6,659,278 triangles   24.7217 dB
#     cap28   4,594,522 triangles   24.4517 dB   (on the baseline's own curve)
#
# Each arm here starts from cap28's cap, so a win is a mesh 31% smaller than the
# baseline that reaches or beats it.
#
# The last arm pays for the texels by dropping the vertex spherical harmonics
# from degree 3 to degree 2, trading view-dependent per-vertex capacity for
# view-independent per-face capacity at a lower total parameter count.
set -euo pipefail

SCENE=${1:-garden}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAP=(--max_points 2800000)

"$HERE/run.sh" tex2     "$SCENE" "${CAP[@]}" --texel_order 2
"$HERE/run.sh" tex3     "$SCENE" "${CAP[@]}" --texel_order 3
"$HERE/run.sh" tex2_sh2 "$SCENE" "${CAP[@]}" --texel_order 2 --sh_degree 2

python -m sota.frontier "$SCENE"
