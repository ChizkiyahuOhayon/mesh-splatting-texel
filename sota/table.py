"""Collect finished runs into one comparison table.

    python sota/table.py [runs_dir]

Prints per-scene metrics for every arm, then a mean row over the scenes that
arm has finished, and the delta against the `base` arm on the scenes both have.
Only scenes shared by both arms are compared, so a partially finished sweep
still reads correctly instead of averaging over different scene sets.
"""

import re
import sys
from pathlib import Path

# MeshSplatting Table 1, nine-scene Mip-NeRF360 mean.
PAPER = {"psnr": 24.78, "ssim": 0.728, "lpips": 0.310}
SCENES = ("bicycle", "flowers", "garden", "stump", "treehill",
          "room", "counter", "kitchen", "bonsai")
LINE = re.compile(
    r"\[ITER (\d+)\] Evaluating test: L1 (\S+) PSNR (\S+) SSIM (\S+) LPIPS (\S+)"
)


def read(run):
    """Final-iteration test metrics of one run, or None if it never finished."""
    if not (run / "DONE").exists():
        return None
    best = None
    for line in (run / "metrics.txt").read_text().splitlines():
        found = LINE.search(line)
        if found:
            iteration, _, psnr, ssim, lpips = found.groups()
            if best is None or int(iteration) >= best[0]:
                best = (int(iteration), float(psnr), float(ssim), float(lpips))
    if best is None:
        return None
    return {"psnr": best[1], "ssim": best[2], "lpips": best[3],
            "minutes": int((run / "DONE").read_text().strip() or 0) / 60}


def collect(runs_dir):
    arms = {}
    for run in sorted(runs_dir.glob("*__*")):
        arm, _, scene = run.name.partition("__")
        row = read(run)
        if row is not None:
            arms.setdefault(arm, {})[scene] = row
    return arms


def mean(rows, key):
    return sum(row[key] for row in rows) / len(rows)


def main(runs_dir):
    arms = collect(runs_dir)
    if not arms:
        print(f"no finished runs under {runs_dir}")
        return
    base = arms.get("base", {})

    for arm in sorted(arms, key=lambda name: (name != "base", name)):
        scenes = arms[arm]
        print(f"\n=== {arm} ({len(scenes)}/9 scenes) ===")
        print(f"{'scene':<10}{'PSNR':>9}{'SSIM':>9}{'LPIPS':>9}{'min':>7}")
        for scene in SCENES:
            if scene in scenes:
                row = scenes[scene]
                print(f"{scene:<10}{row['psnr']:>9.4f}{row['ssim']:>9.4f}"
                      f"{row['lpips']:>9.4f}{row['minutes']:>7.0f}")
        rows = list(scenes.values())
        print(f"{'MEAN':<10}{mean(rows,'psnr'):>9.4f}{mean(rows,'ssim'):>9.4f}"
              f"{mean(rows,'lpips'):>9.4f}{mean(rows,'minutes'):>7.0f}")

        shared = sorted(set(scenes) & set(base)) if arm != "base" else []
        if shared:
            here = [scenes[s] for s in shared]
            there = [base[s] for s in shared]
            print(f"{'vs base':<10}"
                  f"{mean(here,'psnr')-mean(there,'psnr'):>+9.4f}"
                  f"{mean(here,'ssim')-mean(there,'ssim'):>+9.4f}"
                  f"{mean(here,'lpips')-mean(there,'lpips'):>+9.4f}"
                  f"   on {len(shared)}: {' '.join(shared)}")
        if len(scenes) == len(SCENES):
            print(f"{'vs paper':<10}"
                  f"{mean(rows,'psnr')-PAPER['psnr']:>+9.4f}"
                  f"{mean(rows,'ssim')-PAPER['ssim']:>+9.4f}"
                  f"{mean(rows,'lpips')-PAPER['lpips']:>+9.4f}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/runs"))
