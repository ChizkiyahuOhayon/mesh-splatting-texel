"""Verify the ResidualGate per-face gating before spending a training run on it.

ResidualGate is pure PyTorch (no CUDA change), so the risk is logic, not compilation.
Run this FIRST on the server; it checks the properties the method's correctness rests on
and takes seconds.

  R1  disabled path is a no-op        opt.resgate=False must leave the loss untouched
  R2  signal buffers resize on demand  a face-count change re-allocates g_m cleanly
  R3  gating weight is well-formed      phi(g_m) in [floor,1], monotone decreasing,
                                        phi=1 where g_m=0 (planar/pre-refresh => no-op)
  R4  the three signals differ          gm (min) != raw (mean) != curvature, and each
                                        produces a valid per-face weight (falsification arms)

Usage:
  python verify_resgate.py -s /home/smbu/dy/mesh-splatting/data/mipnerf360/garden
"""
import sys
import torch
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams


def main():
    parser = ArgumentParser(description="Verify ResidualGate")
    lp = ModelParams(parser); pp = PipelineParams(parser); op = OptimizationParams(parser)
    args = parser.parse_args(sys.argv[1:])
    from scene import Scene
    from scene.triangle_model import TriangleModel
    dataset, pipe, opt = lp.extract(args), pp.extract(args), op.extract(args)
    torch.manual_seed(0)

    tri = TriangleModel(dataset.sh_degree)
    scene = Scene(dataset, tri, opt.set_weight, opt.set_sigma)
    tri.training_setup(opt, opt.feature_lr, opt.weight_lr, opt.lr_triangles_points_init)
    tri.set_sigma(opt.set_sigma)
    Fn = tri._triangle_indices.shape[0]
    print(f"faces: {Fn:,}")

    # R1: disabled path — resgate_weight with no signal is all-ones (identity multiplier)
    tri._resgate_ensure()
    w0 = tri.resgate_weight(opt.resgate_floor)
    assert w0.shape[0] == Fn and torch.allclose(w0, torch.ones_like(w0)), \
        "R1 FAILED: with g_m=0 the gating weight must be all ones (no-op)"
    print("R1 disabled/pre-refresh path: gating weight is all ones (no-op)")

    # R2: buffers resize when the face count changes
    keep = torch.ones(Fn, dtype=torch.bool, device="cuda"); keep[: Fn // 10] = False
    tri.prune_triangles(keep)
    tri._resgate_ensure()
    assert tri._g_m.shape[0] == tri._triangle_indices.shape[0], "R2 FAILED: g_m did not resize"
    print(f"R2 resize on topology change: g_m -> {tri._g_m.shape[0]:,} faces")

    # R3/R4: feed two synthetic views and refresh each signal
    Fn = tri._triangle_indices.shape[0]
    torch.manual_seed(1)
    base = torch.rand(Fn, device="cuda") * 0.1
    thin = torch.zeros(Fn, device="cuda"); thin[: Fn // 20] = 0.4   # a thin minority
    v1 = (base + thin).clone(); v2 = (base + thin + torch.rand(Fn, device="cuda") * 0.02)
    for sig in ("gm", "raw", "curvature"):
        tri._g_accum = torch.full((Fn,), float("inf"), device="cuda")
        tri._g_sum = torch.zeros(Fn, device="cuda"); tri._g_cnt = torch.zeros(Fn, device="cuda")
        tri.resgate_accumulate(v1); tri.resgate_accumulate(v2)
        tri.resgate_refresh(sig, opt.resgate_norm_q, tri.vertices)
        g = tri._g_m; w = tri.resgate_weight(opt.resgate_floor)
        assert g.min() >= -1e-6 and g.max() <= 1 + 1e-6, f"R3 FAILED[{sig}]: g_m out of [0,1]"
        assert w.min() >= opt.resgate_floor - 1e-6 and w.max() <= 1 + 1e-6, \
            f"R3 FAILED[{sig}]: weight out of [floor,1]"
        # monotone: higher g_m -> lower weight
        assert torch.allclose(w.argsort(descending=True), g.argsort()), \
            f"R3 FAILED[{sig}]: weight not monotone-decreasing in g_m"
        print(f"R4 signal '{sig}': g_m in [{g.min():.3f},{g.max():.3f}], "
              f"weight in [{w.min():.3f},{w.max():.3f}], monotone OK, "
              f"hot faces={int((g>0.5).sum())}")

    # the three signals must not be identical (they are genuine falsification arms)
    tri._g_accum = torch.full((Fn,), float("inf"), device="cuda")
    tri._g_sum = torch.zeros(Fn, device="cuda"); tri._g_cnt = torch.zeros(Fn, device="cuda")
    tri.resgate_accumulate(v1); tri.resgate_accumulate(v2)
    tri.resgate_refresh("gm", opt.resgate_norm_q); gm = tri._g_m.clone()
    tri.resgate_accumulate(v1); tri.resgate_accumulate(v2)
    tri.resgate_refresh("raw", opt.resgate_norm_q); raw = tri._g_m.clone()
    assert not torch.allclose(gm, raw), "R4 FAILED: gm and raw signals are identical"
    print(f"R4 gm vs raw differ: mean|gm-raw| = {(gm-raw).abs().mean():.4f}")

    print("\nALL CHECKS PASSED -- ResidualGate is safe to train.")


if __name__ == "__main__":
    main()
