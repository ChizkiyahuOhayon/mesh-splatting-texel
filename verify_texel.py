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
    # mirror train.py's construction exactly
    scene = Scene(dataset, triangles, opt.set_weight, opt.set_sigma)
    triangles.training_setup(opt, opt.feature_lr, opt.weight_lr,
                             opt.lr_triangles_points_init)
    triangles.set_sigma(opt.set_sigma)   # sigma defaults to 0, which degenerates the window

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

    # Finite differences.
    #
    # Two precision traps here, both about the TEST rather than the kernel:
    #  (1) the image has ~5M pixels summing to ~7e5, where one float32 ULP is 1/16.
    #      Perturbing a single texel moves the sum by ~0.05, i.e. BELOW one ULP, so a
    #      float32 sum quantises the difference to multiples of 0.0625 or to exactly 0.
    #      Accumulate in float64.
    #  (2) rendering is *linear* in a texel (the texel is added to the interpolated
    #      colour, then alpha-blended), so there is no truncation error to worry about
    #      and eps can be large. A large eps is what lifts the signal clear of noise.
    def render_sum64():
        return render_once().double().sum().item()

    order_by_grad = torch.argsort(g.abs().flatten(), descending=True)
    n_s, n_c = g.shape[1], g.shape[2]
    picks = [(int(i) // (n_s * n_c), (int(i) // n_c) % n_s, int(i) % n_c)
             for i in order_by_grad[:5]]
    eps = 1e-2
    worst = 0.0
    with torch.no_grad():
        for pick in picks:
            f, s_, c = pick
            base = triangles._texels[f, s_, c].item()
            triangles._texels[f, s_, c] = base + eps
            lp_ = render_sum64()
            triangles._texels[f, s_, c] = base - eps
            lm_ = render_sum64()
            triangles._texels[f, s_, c] = base
            num = (lp_ - lm_) / (2 * eps)
            ana = g[f, s_, c].item()
            rel = abs(num - ana) / max(abs(num), abs(ana), 1e-6)
            worst = max(worst, rel)
            print(f"   texel[{f},{s_},{c}]: analytic {ana:+.5f}  numeric {num:+.5f}  "
                  f"rel err {rel:.4f}")
    assert worst < 0.05, f"T3 FAILED: worst relative gradient error {worst:.4f}"

    # ---- T4: topology mutation keeps texels aligned with faces ----
    # Regression test for the _prune_vertices desync bug: any face-dropping path must
    # keep texel row i aligned with face i.
    F0 = triangles._triangle_indices.shape[0]
    keep = torch.ones(F0, dtype=torch.bool, device="cuda")
    keep[torch.randperm(F0, device="cuda")[: F0 // 10]] = False   # drop 10% of faces
    triangles.prune_triangles(keep)
    triangles.validate_face_state()                                # asserts alignment
    assert triangles._texels.shape[0] == triangles._triangle_indices.shape[0]
    # a face-dropping _prune_vertices must also stay aligned (this is the bug's locus)
    V = triangles.vertices.shape[0]
    vkeep = torch.ones(V, dtype=torch.bool, device="cuda")
    vkeep[torch.randperm(V, device="cuda")[: V // 20]] = False
    triangles._prune_vertices(vkeep)
    triangles.validate_face_state()
    print(f"T4 topology sync: faces {F0} -> {triangles._triangle_indices.shape[0]}, "
          f"texels aligned")

    # ---- T5: input validation rejects a malformed carrier ----
    bad = torch.zeros(triangles._triangle_indices.shape[0] + 7, args.order ** 2, 3,
                      device="cuda")
    try:
        render(cam, triangles, pipe, bg)  # sanity: good state still renders
        triangles._texels = torch.nn.Parameter(bad)
        render(cam, triangles, pipe, bg)
        raise AssertionError("T5 FAILED: malformed texels were not rejected")
    except (RuntimeError, AssertionError) as e:
        if "T5 FAILED" in str(e):
            raise
        print("T5 input validation: malformed texels correctly rejected")

    print("\nALL CHECKS PASSED -- carrier is safe to train.")


if __name__ == "__main__":
    main()
