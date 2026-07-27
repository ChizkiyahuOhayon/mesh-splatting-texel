"""Geometry-vs-appearance gap, stratified by regime (the geometry-accuracy thesis).

Foundational claim P0 of the geometry-accuracy direction: an opaque connected mesh
reaches high PSNR partly by letting per-vertex SH appearance absorb the photometric
residual that thin geometry cannot represent -- so the rendered image looks right
while the GEOMETRY is wrong on thin structures. If true, thin-region geometry error
is large while thin-region appearance error stays small; the size of that gap is the
headroom a fix could recover.

This script measures the GEOMETRY side (cheap, no rendering, no retraining): the
one-sided Chamfer distance from the ground-truth surface to the reconstructed
surface, stratified by GT regime (crease / thin / flat). It reads the reconstructed
mesh straight out of a cgr_diagnose dump (or any npz with `vertices`,`faces`) and the
GT mesh + per-face labels from the toy scene's gt_labels.npz. Pure numpy + scipy.

Prediction: GT->recon distance is much larger on thin (spoke) faces than on crease /
flat faces -- the mesh did not reconstruct the thin geometry even though it renders
it well. (The appearance side -- that thin-region PSNR stays high -- is a separate,
render-based pass; overall PSNR from training already indicates it.)
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.spatial import cKDTree

LABEL_NAMES = {0: "flat", 1: "crease", 2: "thin"}


def sample_surface(vertices, faces, n_points, rng, labels=None):
    """Area-weighted uniform sample of ``n_points`` on the mesh surface.

    Returns (points [n,3], point_label [n] or None). Barycentric sampling within
    each chosen face; face-selection probability is proportional to face area.
    """
    v = vertices[faces]                       # (F,3,3)
    e1, e2 = v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]
    area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    total = area.sum()
    if total <= 0:
        raise SystemExit("degenerate mesh (zero total area)")
    fidx = rng.choice(len(faces), size=n_points, p=area / total)
    u = rng.random(n_points)
    w = rng.random(n_points)
    over = u + w > 1.0                        # fold back into the triangle
    u[over], w[over] = 1.0 - u[over], 1.0 - w[over]
    pts = v[fidx, 0] + u[:, None] * e1[fidx] + w[:, None] * e2[fidx]
    plab = labels[fidx] if labels is not None else None
    return pts, plab


def main():
    ap = argparse.ArgumentParser(description="Stratified geometry gap (GT->recon Chamfer)")
    ap.add_argument("--recon", required=True,
                    help="npz with reconstructed `vertices` and `faces` (e.g. a cgr_diagnose dump)")
    ap.add_argument("--gt", required=True, help="gt_labels.npz from cgr_toy_scene.py")
    ap.add_argument("--n-gt", type=int, default=200000, help="GT surface samples")
    ap.add_argument("--n-recon", type=int, default=400000, help="recon surface samples (the search set)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    recon = np.load(args.recon)
    gt = np.load(args.gt)
    rv, rf = recon["vertices"].astype(np.float64), recon["faces"].astype(np.int64)
    gv, gf = gt["vertices"].astype(np.float64), gt["faces"].astype(np.int64)
    glab = gt["face_label"]

    # scene scale for a relative report: the GT bounding-box diagonal
    diag = float(np.linalg.norm(gv.max(0) - gv.min(0)))

    gt_pts, gt_plab = sample_surface(gv, gf, args.n_gt, rng, labels=glab)
    recon_pts, _ = sample_surface(rv, rf, args.n_recon, rng)

    # one-sided Chamfer GT -> recon: for each GT point, distance to the nearest
    # reconstructed surface point ("did the mesh grow this GT geometry?").
    tree = cKDTree(recon_pts)
    dist, _ = tree.query(gt_pts, k=1)

    print(f"recon: {len(rv)} verts / {len(rf)} faces | GT: {len(gv)} verts / {len(gf)} faces")
    print(f"scene diag = {diag:.3f}; GT->recon Chamfer by regime "
          f"(distance, and as % of diag):\n")
    print(f"  {'regime':8s}  {'n_pts':>8s}  {'median':>10s}  {'mean':>10s}  {'p90':>10s}  {'%diag(med)':>10s}")
    print("  " + "-" * 66)
    stats = {}
    for k, name in LABEL_NAMES.items():
        m = gt_plab == k
        if m.sum() == 0:
            continue
        d = dist[m]
        med, mean, p90 = float(np.median(d)), float(d.mean()), float(np.percentile(d, 90))
        stats[name] = {"n": int(m.sum()), "median": med, "mean": mean, "p90": p90,
                       "median_pct_diag": 100 * med / diag}
        print(f"  {name:8s}  {int(m.sum()):>8d}  {med:>10.5f}  {mean:>10.5f}  {p90:>10.5f}"
              f"  {100*med/diag:>9.3f}%")

    # the headline ratio: thin geometry error vs crease geometry error
    if "thin" in stats and "crease" in stats:
        ratio = stats["thin"]["median"] / max(stats["crease"]["median"], 1e-12)
        print(f"\n  thin/crease median-Chamfer ratio = {ratio:.2f}x")
        print("  P0 (appearance hides bad thin geometry) is SUPPORTED if thin geometry error"
              " >> crease; the ratio is the recoverable-geometry headroom.")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"scene_diag": diag, "by_regime": stats}, f, indent=2)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
