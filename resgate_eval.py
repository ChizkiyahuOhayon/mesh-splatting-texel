"""Stratified thin-region evaluation for ResidualGate (Phase B, the decisive measurement).

Full-image PSNR/LPIPS average the claimed thin-region effect away (E8 was uninformative
for exactly this reason). This script measures each arm's fidelity SEPARATELY on
high-g_m (under-resolved / thin) vs low-g_m (planar) image regions, using a SHARED
reference mask so the arms are comparable.

Reference mask: from a `baseline_probe` run (`--resgate --resgate_floor 1.0`, i.e. g_m
is computed but the gate weight is identically 1, so training == ungated baseline). Its
saved per-face g_m defines, per test view, which PIXELS are under-resolved in the
baseline. ALL arms are then scored on the SAME per-view high-g_m pixel set.

    python resgate_eval.py -s <scene> --eval \
        --arms output/resgate2/baseline_probe output/resgate2/resgate_gm \
               output/resgate2/control_raw output/resgate2/control_curvature \
        --ref output/resgate2/baseline_probe --quantile 0.9

Reports, per arm: PSNR / SSIM / LPIPS on {high-g_m region, low-g_m region, full image}.
Prediction (RESEARCH_PLAN_v5): resgate_gm improves the HIGH-g_m region over baseline_probe
AND over control_raw / control_curvature; the controls do not.
"""
import os
import sys
import json
from argparse import ArgumentParser

import numpy as np
import torch


def load_model(model_dir, dataset, TriangleModel, device="cuda"):
    tri = TriangleModel(dataset.sh_degree)
    pc = os.path.join(model_dir, "point_cloud", "iteration_30000")
    tri.load_parameters(pc, device=device)
    return tri


def per_view_gm_map(ref_tri, cam, render, pipe, bg, H0, W0):
    """Render the reference model at this camera and project its per-face g_m to a
    per-pixel [H0,W0] map (nearest, at the eval resolution)."""
    pkg = render(cam, ref_tri, pipe, bg)
    fid = torch.nn.functional.interpolate(
        pkg["rend_ids"].unsqueeze(0), size=(H0, W0), mode="nearest").squeeze(0).squeeze(0)
    Fn = ref_tri._triangle_indices.shape[0]
    cov = (fid >= 0) & (fid < Fn)
    fid_l = fid.long().clamp_(0, Fn - 1)
    gm = ref_tri._g_m if ref_tri._g_m.shape[0] == Fn else torch.zeros(Fn, device=fid.device)
    m = torch.where(cov, gm[fid_l], torch.zeros_like(fid))
    return m  # [H0,W0]


def masked_psnr(a, b, mask):
    if mask.sum() < 10:
        return float("nan")
    mse = (((a - b) ** 2).mean(0)[mask]).mean()
    return float(-10.0 * torch.log10(mse.clamp_min(1e-10)))


def main():
    parser = ArgumentParser()
    from arguments import ModelParams, PipelineParams
    lp = ModelParams(parser, sentinel=True); pp = PipelineParams(parser)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--ref", required=True, help="dir whose saved g_m defines the mask (baseline_probe)")
    parser.add_argument("--quantile", type=float, default=0.9, help="high-g_m = top (1-q) of reference g_m pixels")
    parser.add_argument("--out", default="resgate_eval.json")
    args = parser.parse_args(sys.argv[1:])
    dataset, pipe = lp.extract(args), pp.extract(args)

    from scene import Scene
    from scene.triangle_model import TriangleModel
    from triangle_renderer import render
    import lpips as lpips_mod

    bg = torch.zeros(3, device="cuda")
    lpips_fn = lpips_mod.LPIPS(net="vgg").cuda()

    # test cameras: build Scene with a THROWAWAY model (Scene re-inits the model it is
    # given when load_iteration is None; we must not let it clobber a loaded arm).
    throwaway = TriangleModel(dataset.sh_degree)
    scene = Scene(dataset, throwaway, 0.0, 0.0, shuffle=False)
    test_cams = scene.getTestCameras()

    # reference model (its saved g_m defines the shared masks); loaded SEPARATELY.
    ref_tri = load_model(args.ref, dataset, TriangleModel)
    assert getattr(ref_tri, "_g_m_ready", False) and ref_tri._g_m.numel() > 0, \
        f"reference {args.ref} has no saved g_m — run it as baseline_probe (--resgate --resgate_floor 1.0)"
    print(f"{len(test_cams)} test views; reference g_m from {args.ref}")

    # precompute the shared high-g_m masks (from the reference), per test view
    masks = []
    with torch.no_grad():
        for cam in test_cams:
            gt = cam.original_image.cuda()
            H0, W0 = gt.shape[1], gt.shape[2]
            gmap = per_view_gm_map(ref_tri, cam, render, pipe, bg, H0, W0)
            thr = torch.quantile(gmap[gmap > 0], args.quantile) if (gmap > 0).any() else torch.tensor(1e9, device=gmap.device)
            masks.append((gmap >= thr) & (gmap > 0))     # high-g_m pixels
    hi_frac = float(torch.stack([m.float().mean() for m in masks]).mean())
    print(f"high-g_m region = {hi_frac*100:.1f}% of pixels (quantile {args.quantile})")

    results = {}
    for arm_dir in args.arms:
        tri = load_model(arm_dir, dataset, TriangleModel)
        hi_p, lo_p, full_p, hi_lp, full_lp = [], [], [], [], []
        with torch.no_grad():
            for cam, hi in zip(test_cams, masks):
                gt = cam.original_image.cuda()
                img = render(cam, tri, pipe, bg)["render"].clamp(0, 1)
                lo = ~hi
                hi_p.append(masked_psnr(img, gt, hi)); lo_p.append(masked_psnr(img, gt, lo))
                full_p.append(masked_psnr(img, gt, torch.ones_like(hi)))
                # LPIPS on the high-g_m region: mask by zeroing the complement (approx)
                hi3 = hi.unsqueeze(0).float()
                hi_lp.append(lpips_fn(((img*hi3)*2-1).unsqueeze(0), ((gt*hi3)*2-1).unsqueeze(0)).item())
                full_lp.append(lpips_fn((img*2-1).unsqueeze(0), (gt*2-1).unsqueeze(0)).item())
        name = os.path.basename(arm_dir.rstrip("/"))
        results[name] = dict(
            psnr_high=float(np.nanmean(hi_p)), psnr_low=float(np.nanmean(lo_p)),
            psnr_full=float(np.nanmean(full_p)),
            lpips_high=float(np.nanmean(hi_lp)), lpips_full=float(np.nanmean(full_lp)))
        print(f"  {name}: high-g PSNR {results[name]['psnr_high']:.3f} | "
              f"low-g PSNR {results[name]['psnr_low']:.3f} | high-g LPIPS {results[name]['lpips_high']:.4f}")

    json.dump({"quantile": args.quantile, "high_g_frac": hi_frac, "arms": results},
              open(args.out, "w"), indent=2)
    print("\n== STRATIFIED SUMMARY (high-g_m region is where the claim lives) ==")
    print(f"{'arm':<20} {'PSNR_high':>10} {'PSNR_low':>10} {'LPIPS_high':>11}")
    for n, r in results.items():
        print(f"{n:<20} {r['psnr_high']:>10.3f} {r['psnr_low']:>10.3f} {r['lpips_high']:>11.4f}")
    print(f"\nwrote {args.out}")
    print("READ: resgate_gm should beat baseline_probe AND both controls on PSNR_high / LPIPS_high; "
          "if it does not, the g_m gating does not recover thin geometry -> clean falsification.")


if __name__ == "__main__":
    main()
