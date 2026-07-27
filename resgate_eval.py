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


def compute_ref_gm(ref_tri, train_cams, render, pipe, bg, alpha_thr=0.5, min_views=3, norm_q=0.95):
    """Recompute the reference model's per-face g_m from the training views (same logic
    as training). Done at eval time so it does not depend on g_m being saved in the
    checkpoint, and uses the identical appearance-saturated cross-view-min definition."""
    ref_tri._g_m = torch.empty(0)          # force a clean re-accumulation at current F
    for cam in train_cams:
        with torch.no_grad():
            gt = cam.original_image.cuda()
            H0, W0 = gt.shape[1], gt.shape[2]
            pkg = render(cam, ref_tri, pipe, bg)
            img = pkg["render"].clamp(0, 1)
            Fn = ref_tri._triangle_indices.shape[0]
            fid = torch.nn.functional.interpolate(
                pkg["rend_ids"].unsqueeze(0), size=(H0, W0), mode="nearest").squeeze(0).squeeze(0)
            alpha = torch.nn.functional.interpolate(
                pkg["rend_alpha"].unsqueeze(0), size=(H0, W0), mode="bilinear",
                align_corners=False).squeeze(0).squeeze(0)
            cov = (fid >= 0) & (fid < Fn) & (alpha > alpha_thr)
            fid_l = fid.long().clamp_(0, Fn - 1)
            res = (img - gt).abs().mean(0)
            covf = fid_l[cov].reshape(-1)
            s = torch.zeros(Fn, device=gt.device).scatter_add_(0, covf, res[cov].reshape(-1))
            n = torch.zeros(Fn, device=gt.device).scatter_add_(0, covf, torch.ones_like(res[cov].reshape(-1)))
            face_res = torch.where(n > 0, s / n.clamp_min(1), torch.full((Fn,), float("inf"), device=gt.device))
            ref_tri.resgate_accumulate(face_res)
    ref_tri.resgate_refresh("gm", norm_q, ref_tri.vertices, ema=0.0, min_views=min_views)


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
    lp = ModelParams(parser); pp = PipelineParams(parser)   # NOT sentinel: we need real defaults
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--ref", required=True, help="dir whose saved g_m defines the mask (baseline_probe)")
    parser.add_argument("--quantile", type=float, default=0.9, help="high-g_m = top (1-q) of reference g_m pixels")
    parser.add_argument("--disagree", action="store_true",
                        help="also test the regime where g_m and curvature DISAGREE (the clean test of whether g_m is special)")
    parser.add_argument("--gm-arm", default=None, help="arm dir trained with signal=gm (for --disagree)")
    parser.add_argument("--curv-arm", default=None, help="arm dir trained with signal=curvature (for --disagree)")
    parser.add_argument("--out", default="resgate_eval.json")
    args = parser.parse_args(sys.argv[1:])
    dataset, pipe = lp.extract(args), pp.extract(args)
    if getattr(dataset, "resolution", None) is None:
        dataset.resolution = -1   # default: the training resolution heuristic

    from scene import Scene
    from scene.triangle_model import TriangleModel
    from triangle_renderer import render
    import lpips as lpips_mod

    bg = torch.zeros(3, device="cuda")
    lpips_fn = lpips_mod.LPIPS(net="vgg").cuda()

    # test cameras: build Scene with a THROWAWAY model (Scene re-inits the model it is
    # given when load_iteration is None; we must not let it clobber a loaded arm).
    # Scene writes input.ply into model_path, so give it a scratch dir (we passed no -m).
    if not getattr(dataset, "model_path", None):
        dataset.model_path = os.path.join("/tmp", "resgate_eval_scene")
        os.makedirs(dataset.model_path, exist_ok=True)
    throwaway = TriangleModel(dataset.sh_degree)
    # init_opacity>0, set_sigma>0: create_from_pcd does inverse_exponential_activation =
    # log(set_sigma), so set_sigma=0 -> math domain error. The throwaway's init values are
    # irrelevant (only its cameras are used; every evaluated model is load_parameters'd).
    scene = Scene(dataset, throwaway, 0.1, 1.0, shuffle=False)
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()

    # reference model: loaded SEPARATELY, and its per-face g_m is RECOMPUTED here from the
    # training views (does not depend on g_m being saved in the checkpoint).
    ref_tri = load_model(args.ref, dataset, TriangleModel)
    print(f"{len(test_cams)} test / {len(train_cams)} train views; "
          f"recomputing reference g_m from {args.ref} ...")
    compute_ref_gm(ref_tri, train_cams, render, pipe, bg)
    print(f"reference g_m: {int((ref_tri._g_m > 0).sum())} / {ref_tri._g_m.numel()} faces active")

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

    # ---- disagreement diagnostic: where g_m and curvature DISAGREE ----
    if args.disagree:
        gm_arm = args.gm_arm or next((a for a in args.arms if "gm" in os.path.basename(a)), None)
        cv_arm = args.curv_arm or next((a for a in args.arms if "curv" in os.path.basename(a)), None)
        assert gm_arm and cv_arm, "need --gm-arm and --curv-arm (or arms named *gm* / *curv*)"
        gm_n = ref_tri._g_m.clamp(0, 1)                                   # already [0,1]
        cv = ref_tri._per_face_curvature(ref_tri.vertices)
        cv_n = (cv / torch.quantile(cv[cv > 0], 0.95).clamp_min(1e-8)).clamp(0, 1)
        q_hi, q_lo = 0.85, 0.5
        setA = (cv_n > q_hi) & (gm_n < q_lo)     # high curvature, LOW residual = resolved sharp edges
        setB = (gm_n > q_hi) & (cv_n < q_lo)     # high residual, LOW curvature = under-resolved, not sharp
        print(f"\n-- disagreement sets: A(curv-hi,gm-lo)={int(setA.sum())} faces  "
              f"B(gm-hi,curv-lo)={int(setB.sum())} faces --")
        Fn = ref_tri._triangle_indices.shape[0]
        gm_t = load_model(gm_arm, dataset, TriangleModel)
        cv_t = load_model(cv_arm, dataset, TriangleModel)
        dis = {"A_curvHi_gmLo": {}, "B_gmHi_curvLo": {}}
        with torch.no_grad():
            for setname, fset in (("A_curvHi_gmLo", setA), ("B_gmHi_curvLo", setB)):
                gm_p, cv_p, base_p = [], [], []
                for cam in test_cams:
                    gt = cam.original_image.cuda(); H0, W0 = gt.shape[1], gt.shape[2]
                    fid = torch.nn.functional.interpolate(
                        render(cam, ref_tri, pipe, bg)["rend_ids"].unsqueeze(0),
                        size=(H0, W0), mode="nearest").squeeze(0).squeeze(0)
                    cov = (fid >= 0) & (fid < Fn)
                    pm = torch.zeros_like(fid, dtype=torch.bool)
                    pm[cov] = fset[fid[cov].long()]
                    gm_p.append(masked_psnr(render(cam, gm_t, pipe, bg)["render"].clamp(0, 1), gt, pm))
                    cv_p.append(masked_psnr(render(cam, cv_t, pipe, bg)["render"].clamp(0, 1), gt, pm))
                    base_p.append(masked_psnr(render(cam, ref_tri, pipe, bg)["render"].clamp(0, 1), gt, pm))
                dis[setname] = {"psnr_gm": float(np.nanmean(gm_p)),
                                "psnr_curv": float(np.nanmean(cv_p)),
                                "psnr_baseline": float(np.nanmean(base_p)),
                                "n_faces": int(fset.sum())}
        results["_disagreement"] = dis
        print("\n== DISAGREEMENT DIAGNOSTIC (does g_m beat curvature where they disagree?) ==")
        for k, d in dis.items():
            better = "gm WINS" if d["psnr_gm"] > d["psnr_curv"] + 0.05 else \
                     ("curv wins" if d["psnr_curv"] > d["psnr_gm"] + 0.05 else "tie (<0.05)")
            print(f"  {k} ({d['n_faces']} faces): PSNR gm {d['psnr_gm']:.3f} | curv {d['psnr_curv']:.3f} | "
                  f"baseline {d['psnr_baseline']:.3f}  -> {better}")

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
