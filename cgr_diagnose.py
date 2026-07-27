"""CGR E9 clean measurement — frozen-geometry, full-view-sweep coherence.

The training-hook diagnostic (--cgr_diag) was swamped by two artifacts: a short EMA
memory (rho=0.9 sees only ~10 recent single-view steps) and per-face visibility
sparsity (millions of faces, one view per step), so nu_i read ~0 for most faces and
nothing separated. This script removes both by measuring at a FROZEN checkpoint over
a full sweep of all training views, accumulating per vertex:

    mu_sum  += g_v            (vector sum of the photometric position gradient)
    nu_sum  += ||g_v||        (scalar sum of its magnitude)
    obs     += 1[||g_v|| > 0] (how many views actually drove this vertex)

and derives, dropping vertices observed in fewer than --min-views views:

    O_i  = ||mu_sum|| / (nu_sum + eps)   in [0,1]   -- multi-view coherence
    nu_i = nu_sum / obs                              -- mean per-view drive

O_i is LOW where the views' gradients cancel (a resolved crease at a multi-view
fixed point) and HIGH where they persist one-signed (an under-resolved thin
structure still being driven). nu_i is the regime-existence proxy: if the thin
region is genuinely under-resolved, it is driven harder than the crease region, so
nu_i separates them; if nu_i does NOT separate, the residual has been absorbed
elsewhere (e.g. by the SH appearance) and there is no position-space signal to read.

The photometric loss here is L1 only (no SSIM window) so each vertex's gradient is
attributed to its own pixels. Reads the uncontaminated gradient at the rasterizer
autograd boundary (no smoothness term is in this graph). Output matches cgr_auc.py.
"""

import os
import sys
import json
from argparse import ArgumentParser

import numpy as np
import torch


def load_model(model_dir, iteration, dataset, TriangleModel, device="cuda"):
    tri = TriangleModel(dataset.sh_degree)
    pc = os.path.join(model_dir, "point_cloud", f"iteration_{iteration}")
    assert os.path.isdir(pc), f"checkpoint not found: {pc}"
    tri.load_parameters(pc, device=device)
    return tri


def main():
    parser = ArgumentParser(description="CGR E9 frozen full-view-sweep coherence measurement")
    from arguments import ModelParams, PipelineParams
    lp = ModelParams(parser); pp = PipelineParams(parser)   # NOT sentinel: real defaults
    parser.add_argument("--model", required=True, help="training output dir (contains point_cloud/)")
    parser.add_argument("--iteration", type=int, required=True, help="checkpoint iteration to load")
    parser.add_argument("--min-views", type=int, default=5, help="drop vertices observed in fewer views")
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--out", required=True, help="output .npz (per-face signals for cgr_auc.py)")
    args = parser.parse_args(sys.argv[1:])
    dataset, pipe = lp.extract(args), pp.extract(args)
    if getattr(dataset, "resolution", None) is None:
        dataset.resolution = -1

    from scene import Scene
    from scene.triangle_model import TriangleModel
    from triangle_renderer import render
    from scene.cgr import per_face_curvature
    from utils.loss_utils import l1_loss

    bg = torch.zeros(3, device="cuda")

    # cameras via a throwaway model (Scene re-inits the model it is handed; we load the
    # real one separately). Scene writes into model_path, so give it a scratch dir.
    if not getattr(dataset, "model_path", None):
        dataset.model_path = os.path.join("/tmp", "cgr_diagnose_scene")
        os.makedirs(dataset.model_path, exist_ok=True)
    throwaway = TriangleModel(dataset.sh_degree)
    scene = Scene(dataset, throwaway, 0.1, 1.0, shuffle=False)
    train_cams = scene.getTrainCameras()

    tri = load_model(args.model, args.iteration, dataset, TriangleModel)
    verts = tri.get_vertices
    V = verts.shape[0]
    print(f"loaded {args.model} @ iter {args.iteration}: {V} vertices, "
          f"{tri._triangle_indices.shape[0]} faces; sweeping {len(train_cams)} train views")

    mu_sum = torch.zeros(V, 3, device="cuda")
    nu_sum = torch.zeros(V, device="cuda")
    obs = torch.zeros(V, device="cuda")

    for cam in train_cams:
        gt = cam.original_image.cuda()
        image = render(cam, tri, pipe, bg)["render"]
        loss_image = l1_loss(image, gt)          # pure photometric, per-vertex attribution
        (g,) = torch.autograd.grad(loss_image, tri.vertices, retain_graph=False)
        with torch.no_grad():
            gm = g.norm(dim=1)
            mu_sum += g
            nu_sum += gm
            obs += (gm > 0).float()

    with torch.no_grad():
        valid = obs >= args.min_views
        O_vert = mu_sum.norm(dim=1) / (nu_sum + args.eps)         # coherence in [0,1]
        nu_vert = nu_sum / obs.clamp_min(1.0)                     # mean per-view drive
        tri_idx = tri._triangle_indices.long()

        # per-face: mean over its 3 vertices; a face is valid iff all 3 are well-observed
        face_valid = valid[tri_idx].all(dim=1)
        O_face = O_vert[tri_idx].mean(dim=1)
        nu_face = nu_vert[tri_idx].mean(dim=1)
        obs_face = obs[tri_idx].min(dim=1).values
        curv_face = per_face_curvature(verts.detach(), tri_idx)
        centroid = verts.detach()[tri_idx].mean(dim=1)

        n_valid = int(face_valid.sum())
        print(f"vertices observed >= {args.min_views} views: {int(valid.sum())}/{V} "
              f"({100*valid.float().mean():.1f}%); fully-observed faces: {n_valid}/{len(tri_idx)}")
        print(f"nu_i (per-view drive) over valid faces: "
              f"median={float(nu_face[face_valid].median()) if n_valid else float('nan'):.6g}  "
              f"max={float(nu_face[face_valid].max()) if n_valid else float('nan'):.6g}")

        np.savez(
            args.out,
            steps=len(train_cams), rho=-1.0,             # rho=-1 marks a full-sweep (non-EMA) dump
            iteration=args.iteration, min_views=args.min_views,
            face_coherence=O_face.cpu().numpy(),
            face_magnitude=nu_face.cpu().numpy(),
            face_curvature=curv_face.cpu().numpy(),
            face_centroid=centroid.cpu().numpy(),
            face_obs=obs_face.cpu().numpy(),
            face_valid=face_valid.cpu().numpy(),
            vertices=verts.detach().cpu().numpy(),
            faces=tri_idx.cpu().numpy(),
        )
    print(f"wrote {args.out}")
    print("next: python cgr_auc.py --dump <this> --gt <scene>/gt_labels.npz --min-obs "
          f"{args.min_views}")


if __name__ == "__main__":
    main()
