"""Parameter budget per arm: what the quality actually costs to store.

    python -m sota.budget [scene]

MeshSplatting sells compactness, so a quality gain only counts if it survives on
the storage axis too. Counts what the renderer needs -- vertex positions, the
surviving vertex spherical harmonics, vertex opacity, face indices, and any
per-face texels -- at four bytes each.
"""
import glob
import math
import pathlib
import re
import sys
import torch

SCENE = sys.argv[1] if len(sys.argv) > 1 else "garden"
rows = []
for run in sorted(glob.glob(f"/root/autodl-tmp/runs/*__{SCENE}")):
    arm = run.split("/")[-1].split("__")[0]
    saves = sorted(glob.glob(run + "/point_cloud/iteration_*/point_cloud_state_dict.pt"),
                   key=lambda p: int(p.split("iteration_")[1].split("/")[0]))
    metrics = pathlib.Path(run + "/metrics.txt")
    if not saves or not metrics.exists():
        continue
    lines = [l for l in metrics.open() if "Evaluating test" in l]
    if not lines:
        continue
    state = torch.load(saves[-1], map_location="cpu")
    psnr = float(re.search(r"PSNR (\S+)", lines[-1]).group(1))
    vertices = state["triangles_points"].shape[0]
    faces = state["_triangle_indices"].shape[0]
    # Appearance and geometry the renderer needs: vertex position, vertex SH of
    # whatever degree survived, vertex opacity, face indices, and any texels.
    sh = state["features_dc"].numel() + state["features_rest"].numel()
    floats = vertices * 3 + sh + vertices
    texels = state["texels"].numel() if state.get("texel_order", 0) else 0
    floats += texels
    ints = faces * 3
    rows.append((arm, psnr, vertices, faces, floats, texels, ints))

base = next(r for r in rows if r[0] == "base")
fmt = "{:<10}{:>9}{:>12}{:>12}{:>11}{:>10}{:>9}{:>9}"
print(f"scene: {SCENE}")
print(fmt.format("arm", "PSNR", "vertices", "triangles", "MB", "texelMB", "dMB", "dPSNR"))
for arm, psnr, v, f, floats, texels, ints in sorted(rows, key=lambda r: -r[1]):
    mb = (floats * 4 + ints * 4) / 1e6
    base_mb = (base[4] * 4 + base[6] * 4) / 1e6
    print(fmt.format(arm, "%.4f" % psnr, "{:,}".format(v), "{:,}".format(f),
                     "%.1f" % mb, "%.1f" % (texels * 4 / 1e6),
                     "%+.1f%%" % (100 * (mb / base_mb - 1)),
                     "%+.3f" % (psnr - base[1])))
