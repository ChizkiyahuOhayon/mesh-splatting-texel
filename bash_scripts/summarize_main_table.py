#!/usr/bin/env python3
"""Aggregate the main-table runs into the paper's Table 1.

Reads every <out_dir>/<scene>_<arm>.log, extracts the final (ITER 30000) test metrics,
and prints a per-scene table plus the cross-scene average for each arm and the delta.
Only scenes present for BOTH arms enter the average, so a half-finished sweep still
summarises cleanly.

    python bash_scripts/summarize_main_table.py output/main [--order 2]
"""
import argparse
import glob
import os
import re
import sys

TEST_RE = re.compile(
    r"ITER 30000\] Evaluating test:.*?PSNR\s+([0-9.]+)\s+SSIM\s+([0-9.]+)\s+LPIPS\s+([0-9.]+)")
TRAIN_RE = re.compile(
    r"ITER 30000\] Evaluating train:.*?PSNR\s+([0-9.]+)")


def parse_log(path):
    """Return (psnr, ssim, lpips, train_psnr) from the last matching lines, or None."""
    try:
        txt = open(path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return None
    te = TEST_RE.findall(txt)
    tr = TRAIN_RE.findall(txt)
    if not te:
        return None
    p, s, l = te[-1]
    trp = float(tr[-1]) if tr else float("nan")
    return float(p), float(s), float(l), trp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--order", type=int, default=2)
    args = ap.parse_args()
    arm_b, arm_t = "baseline", f"texel{args.order}"

    scenes = sorted({
        os.path.basename(p).rsplit("_", 1)[0]
        for p in glob.glob(os.path.join(args.out_dir, f"*_{arm_b}.log"))
    })
    if not scenes:
        sys.exit(f"no *_{arm_b}.log under {args.out_dir}")

    rows, acc = [], {arm_b: [], arm_t: []}
    for sc in scenes:
        b = parse_log(os.path.join(args.out_dir, f"{sc}_{arm_b}.log"))
        t = parse_log(os.path.join(args.out_dir, f"{sc}_{arm_t}.log"))
        rows.append((sc, b, t))
        if b and t:
            acc[arm_b].append(b)
            acc[arm_t].append(t)

    def fmt(v, f="{:.3f}"):
        return f.format(v) if v is not None else "—"

    print(f"# Main table ({arm_b} vs {arm_t}) — test @ 30000\n")
    print("| scene | PSNR_b | PSNR_t | ΔPSNR | SSIM_b | SSIM_t | LPIPS_b | LPIPS_t | ΔLPIPS% |")
    print("|---|---|---|---|---|---|---|---|---|")
    for sc, b, t in rows:
        if b and t:
            dpsnr = f"{t[0]-b[0]:+.3f}"
            dlp = f"{(t[2]-b[2])/b[2]*100:+.1f}"
            print(f"| {sc} | {fmt(b[0])} | {fmt(t[0])} | {dpsnr} | {fmt(b[1],'{:.4f}')} | "
                  f"{fmt(t[1],'{:.4f}')} | {fmt(b[2],'{:.4f}')} | {fmt(t[2],'{:.4f}')} | {dlp} |")
        else:
            miss = "baseline" if not b else arm_t
            print(f"| {sc} | {'(pending: '+miss+')':^63} |")

    n = len(acc[arm_b])
    if n:
        def avg(arm, i):
            return sum(r[i] for r in acc[arm]) / n
        pb, pt = avg(arm_b, 0), avg(arm_t, 0)
        sb, st = avg(arm_b, 1), avg(arm_t, 1)
        lb, lt = avg(arm_b, 2), avg(arm_t, 2)
        print(f"| **AVG ({n} scenes)** | **{pb:.3f}** | **{pt:.3f}** | **{pt-pb:+.3f}** | "
              f"**{sb:.4f}** | **{st:.4f}** | **{lb:.4f}** | **{lt:.4f}** | "
              f"**{(lt-lb)/lb*100:+.1f}** |")
        print(f"\nComplete scene-pairs: {n}/{len(scenes)}")
    else:
        print("\n(no scene has both arms finished yet)")


if __name__ == "__main__":
    main()
