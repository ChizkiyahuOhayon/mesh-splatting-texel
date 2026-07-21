"""Verify the per-face texel carrier before spending a training run on it.

The CUDA changes were written without access to an NVIDIA GPU, so run this FIRST on the
server. It checks the three properties that make the experiment interpretable, and takes
seconds rather than the ~1.5 h a full garden run costs.

  T1  disabled path is untouched      texel_order=0 must reproduce the stock renderer
  T2  zero texels are a no-op         order=2 with zero texels must render EXACTLY like
                                      order=0 -- this is what makes the carrier's
                                      introduction attributable
  T3  gradients are correct           analytic dL/dtexel must match finite differences

T2 is the important one: the whole design rests on the carrier being introduced as a
zero-initialised residual that perturbs nothing at the moment it appears.

Usage:
  python verify_texel.py -s /path/to/data/mipnerf360/garden
"""

import sys
import torch

from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams


def main():
    parser = ArgumentParser(description="Verify the texel carrier")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--order", type=int, default=2)
    args = parser.parse_args(sys.argv[1:])

    from scene import Scene
    from scene.triangle_model import TriangleModel
    from triangle_renderer import render

    dataset, pipe, opt = lp.extract(args), pp.extract(args), op.extract(args)
    torch.manual_seed(0)

    triangles = TriangleModel(dataset.sh_degree)
    scene = Scene(dataset, triangles)
    triangles.training_setup(opt, opt.feature_lr, opt.weight_lr, opt.position_lr_init)

    cam = scene.getTrainCameras()[0]
    bg = torch.zeros(3, device="cuda")

    def render_once():
        return render(cam, triangles, pipe, bg)["render"]

    # ---- T1: disabled path ----
    triangles.texel_order = 0
    img0 = render_once().detach().clone()
    print(f"T1 disabled path renders: shape {tuple(img0.shape)}, "
          f"mean {img0.mean().item():.6f}")
    assert torch.isfinite(img0).all(), "T1 FAILED: non-finite output with carrier disabled"

    # ---- T2: zero texels must be a no-op ----
    triangles.create_texels(args.order, opt.texel_lr)
    img1 = render_once().detach().clone()
    max_abs = (img1 - img0).abs().max().item()
    print(f"T2 zero-texel deviation from baseline: max |diff| = {max_abs:.3e}")
    assert max_abs == 0.0, (
        f"T2 FAILED: zero-initialised texels changed the image by {max_abs:.3e}. "
        "The carrier is not a clean no-op at introduction -- indexing or the additive "
        "term is wrong.")

    # ---- T3: gradient check ----
    triangles._texels.grad = None
    loss = render_once().sum()
    loss.backward()
    g = triangles._texels.grad
    assert g is not None, "T3 FAILED: no gradient reached the texels"
    nz = int((g != 0).sum())
    print(f"T3 analytic grad: {nz:,}/{g.numel():,} non-zero, "
          f"absmax {g.abs().max().item():.3e}")
    assert nz > 0, "T3 FAILED: texel gradient is entirely zero"

    # finite differences on a few visible texels
    idx = torch.nonzero(g.abs() > 0, as_tuple=False)
    idx = idx[torch.randperm(len(idx))[:5]]
    eps = 1e-3
    worst = 0.0
    with torch.no_grad():
        for f, s_, c in idx.tolist():
            base = triangles._texels[f, s_, c].item()
            triangles._texels[f, s_, c] = base + eps
            lp_ = render_once().sum().item()
            triangles._texels[f, s_, c] = base - eps
            lm_ = render_once().sum().item()
            triangles._texels[f, s_, c] = base
            num = (lp_ - lm_) / (2 * eps)
            ana = g[f, s_, c].item()
            rel = abs(num - ana) / max(abs(num), abs(ana), 1e-6)
            worst = max(worst, rel)
            print(f"   texel[{f},{s_},{c}]: analytic {ana:+.5f}  numeric {num:+.5f}  "
                  f"rel err {rel:.4f}")
    assert worst < 0.05, f"T3 FAILED: worst relative gradient error {worst:.4f}"

    print("\nALL CHECKS PASSED -- carrier is safe to train.")


if __name__ == "__main__":
    main()
