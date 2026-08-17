"""Quality against model size for every finished arm.

    python -m sota.frontier [runs_dir]

Arms land on different primitive counts -- up to 36% apart in batch 4 -- so the
raw delta against `base` confounds the change under test with model size. The
last column subtracts a capacity term at MeshSplatting's own +0.5 dB per
doubling of primitives, which is an extrapolation borrowed from their ablation
and should be replaced by the measured curve from sota/batch5.sh as soon as it
exists.
"""

import sys
import glob
import math
import re
import torch

RUNS = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/runs"

rows = []
for run in sorted(glob.glob(RUNS + "/*__garden")):
    arm = run.split("/")[-1].split("__")[0]
    saves = sorted(glob.glob(run + "/point_cloud/iteration_*/point_cloud_state_dict.pt"),
                   key=lambda p: int(p.split("iteration_")[1].split("/")[0]))
    if not saves:
        continue
    state = torch.load(saves[-1], map_location="cpu")
    lines = [l for l in open(run + "/metrics.txt") if "Evaluating test" in l]
    if not lines:
        continue
    psnr = float(re.search(r"PSNR (\S+)", lines[-1]).group(1))
    rows.append((arm, psnr, state["triangles_points"].shape[0],
                 state["_triangle_indices"].shape[0]))

base = next(r for r in rows if r[0] == "base")
header = "{:<13}{:>9}{:>8}{:>13}{:>9}{:>9}"
print(header.format("arm", "PSNR", "dPSNR", "triangles", "size", "matched"))
for arm, psnr, _, faces in sorted(rows, key=lambda r: -r[1]):
    ratio = faces / base[3]
    # Their own ablation: about +0.5 dB per doubling of primitives.
    matched = psnr - base[1] - 0.5 * math.log2(ratio)
    print(header.format(arm, "%.4f" % psnr, "%+.3f" % (psnr - base[1]),
                        "{:,}".format(faces), "%+.1f%%" % (100 * (ratio - 1)),
                        "%+.3f" % matched))
