"""Paper figure renders that need a GPU. Nothing trains.

compact   Render test views of one compact cut (sota/survival_cleanup --budget):
          the same pre-cleanup mesh cut to the same face count by the peak rule
          (matched/) and by the integral (matched_oats/), with the evaluator's
          SoftTail-Quality settings, next to the ground truth.

              python -m sota.figure_cuts compact -s <scene> -i images_4 --eval \
                  -m <cut>/matched --cut <cut> --views auto --out <dir>

contrib   Where the faces the two rules disagree on deliver light. On the saved
          pre-cleanup mesh, every face is classed as kept by both rules, only by
          the peak rule, or only by the integral, at the published budget. The
          mesh is split into independent faces (each face gets its own copy of
          its three vertices, so geometry, opacity and compositing are unchanged)
          and rendered with an indicator color per class. A pixel's value is then
          the summed blending weight sum_f w_f(p) of that class, the same quantity
          S_f integrates per face.

              python -m sota.figure_cuts contrib -s <scene> -i images_4 --eval \
                  -m <run> --view DSC08140 --out <dir>
"""

import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from scene.triangle_model import TriangleModel
from sota.survival import budget_matched_keep, v1_keep
from sota.survival_cleanup import survival_scores
from triangle_renderer import render
from utils.general_utils import safe_state
from utils.image_utils import psnr

QUALITY = {"threshold": 1e-2, "absorb": True, "upsample": 4}


def save_png(tensor, path):
    array = tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    Image.fromarray((array * 255 + 0.5).astype(np.uint8)).save(path)


def load(dataset, iteration):
    triangles = TriangleModel(dataset.sh_degree)
    scene = Scene(dataset, triangles, init_opacity=None, set_sigma=None,
                  load_iteration=iteration, shuffle=False)
    triangles.scaling = 4
    return scene, triangles


def quality_render(view, triangles, pipeline, background):
    return render(view, triangles, pipeline, background,
                  transmittance_threshold_override=QUALITY["threshold"],
                  absorb_transmittance_tail=QUALITY["absorb"],
                  upsample_override=QUALITY["upsample"])["render"].clamp(0, 1)


def compact(dataset, pipeline, args):
    cut = Path(args.cut)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_view = {}
    for rule in ("peak", "integral"):
        rows = json.loads((cut / f"eval_{rule}" / "result.json").read_text())["metrics"]["per_view"]
        per_view[rule] = {r["view"]: r["psnr"] for r in rows}
    if args.views == "auto":
        # The view with the median integral-minus-peak gap: typical, not the best case.
        gaps = sorted((per_view["integral"][v] - per_view["peak"][v], v) for v in per_view["peak"])
        views = [gaps[len(gaps) // 2][1]]
    else:
        views = args.views.split(",")
    background = torch.zeros(3, device="cuda")
    report = {"cut": str(cut), "views": views, "psnr": {}}
    for rule, sub in (("peak", "matched"), ("integral", "matched_oats")):
        dataset.model_path = str(cut / sub)
        scene, triangles = load(dataset, 30000)
        triangles.opacity_floor = 0.8
        cams = {v.image_name: v for v in scene.getTestCameras()}
        with torch.no_grad():
            for name in views:
                view = cams[name]
                image = quality_render(view, triangles, pipeline, background)
                save_png(image, out / f"{name}_{rule}.png")
                if rule == "peak":
                    save_png(view.original_image[:3], out / f"{name}_gt.png")
                target = view.original_image[:3].cuda()
                report["psnr"].setdefault(name, {})[rule] = float(psnr(image, target).mean())
        report.setdefault("triangles", {})[rule] = int(triangles.get_triangle_indices.shape[0])
        del scene, triangles
        torch.cuda.empty_cache()
    (out / "compact_render.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


def contrib(dataset, pipeline, args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_dir = Path(dataset.model_path)
    meta = json.loads((run_dir / "point_cloud" / "iteration_precleanup" / "cleanup.json").read_text())
    scene, triangles = load(dataset, "precleanup")
    peak, integral = survival_scores(scene, triangles, pipeline, meta["cleanup_scaling"])
    keep_peak = v1_keep(peak)
    keep_integral = budget_matched_keep(integral, int(keep_peak.sum()))
    cls = torch.zeros_like(peak, dtype=torch.long)          # 0: removed by both
    cls[keep_peak & keep_integral] = 1
    cls[keep_peak & ~keep_integral] = 2                     # kept only by the peak rule
    cls[keep_integral & ~keep_peak] = 3                     # kept only by the integral

    # Split the mesh into independent faces so a color can be constant per face.
    faces = triangles.get_triangle_indices.long()
    corners = faces.reshape(-1)
    triangles.vertices = triangles.vertices[corners].contiguous()
    triangles.vertex_weight = triangles.vertex_weight[corners].contiguous()
    triangles._features_dc = triangles._features_dc[corners].contiguous()
    triangles._features_rest = triangles._features_rest[corners].contiguous()
    triangles._triangle_indices = torch.arange(corners.numel(), device="cuda",
                                               dtype=torch.int32).reshape(-1, 3)
    if getattr(triangles, "opacity_floor_vertex", None) is not None:
        triangles.opacity_floor_vertex = triangles.opacity_floor_vertex[corners].contiguous()

    view = next(v for v in scene.getTrainCameras() + scene.getTestCameras() if v.image_name == args.view)
    background = torch.zeros(3, device="cuda")
    indicator = torch.eye(4, 3, device="cuda")  # rows: class 0..3 -> unit color or zero
    maps = {}
    with torch.no_grad():
        save_png(render(view, triangles, pipeline, background)["render"], out / "rgb.png")
        for name, c in (("both", 1), ("peak_only", 2), ("integral_only", 3)):
            colors = torch.zeros(faces.shape[0], 3, device="cuda")
            colors[cls == c] = 1.0
            image = render(view, triangles, pipeline, background,
                           override_color=colors.repeat_interleave(3, dim=0))["render"][0]
            maps[name] = image.cpu().numpy().astype(np.float32)
    np.savez_compressed(out / "contrib.npz", **maps)
    report = {
        "run": str(run_dir), "view": args.view, "faces": int(faces.shape[0]),
        "kept": int(keep_peak.sum()),
        "swapped_each_way": int((keep_integral & ~keep_peak).sum()),
        "pixel_weight_sum": {k: float(v.sum()) for k, v in maps.items()},
        "S_f_sum": {
            "peak_only": float(integral[cls == 2].sum()),
            "integral_only": float(integral[cls == 3].sum()),
        },
    }
    (out / "contrib.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("mode", choices=["compact", "contrib"])
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--cut")
    parser.add_argument("--views", default="auto")
    parser.add_argument("--view")
    parser.add_argument("--out", required=True)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    safe_state(args.quiet)
    (compact if args.mode == "compact" else contrib)(model.extract(args), pipeline.extract(args), args)
