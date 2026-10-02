"""Render where the two survival statistics live on a trained mesh.

Loads a run's pre-cleanup mesh, measures the peak P_f and the integral S_f of
every face over all training views (the same pass as sota.survival_cleanup),
and renders one test view four ways:

  rgb.png    the mesh as trained
  peak.png   per-face P_f as a heat map
  integral.png  per-face log S_f as a heat map
  swap.png   faces kept only by the peak rule (red), only by the integral
             (blue), by both (grey), at the published face budget

Colors are pushed through the rasterizer as per-vertex colors (a vertex takes
the maximum over its incident faces), so the maps are composited exactly like
the RGB render. Nothing trains.

    python -m sota.survival_maps -s <scene> -m <run> -i images_4 --eval \
        --view DSC08140 --out <dir>
"""

import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from scene.triangle_model import TriangleModel
from sota.survival import budget_matched_keep, v1_keep
from sota.survival_cleanup import survival_scores
from triangle_renderer import render
from utils.general_utils import safe_state


def to_image(tensor):
    array = tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((array * 255 + 0.5).astype(np.uint8))


def vertex_values(faces, face_values, count):
    """Maximum of the incident faces' values at every vertex."""
    out = torch.full((count,), float(face_values.min()), device=face_values.device)
    for corner in range(3):
        out.scatter_reduce_(0, faces[:, corner], face_values, reduce="amax")
    return out


def heat(values, low, high, name):
    scaled = ((values - low) / max(high - low, 1e-12)).clamp(0, 1).cpu().numpy()
    return torch.tensor(colormaps[name](scaled)[:, :3], dtype=torch.float32, device="cuda")


def run(dataset, pipeline, args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_dir = Path(dataset.model_path)
    meta = json.loads((run_dir / "point_cloud" / "iteration_precleanup" / "cleanup.json").read_text())

    triangles = TriangleModel(dataset.sh_degree)
    scene = Scene(dataset, triangles, init_opacity=None, set_sigma=None,
                  load_iteration="precleanup", shuffle=False)
    peak, integral = survival_scores(scene, triangles, pipeline, meta["cleanup_scaling"])
    keep_peak = v1_keep(peak)
    keep_integral = budget_matched_keep(integral, int(keep_peak.sum()))

    faces = triangles.get_triangle_indices.long()
    count = triangles.vertices.shape[0]
    views = scene.getTestCameras() if args.view_set == "test" else scene.getTrainCameras()
    view = next(v for v in views if v.image_name == args.view)
    background = torch.ones(3, device="cuda") if args.white else torch.zeros(3, device="cuda")

    log_integral = torch.log10(integral.clamp_min(1e-3))
    lo, hi = torch.quantile(log_integral[keep_peak | keep_integral].float()[::17],
                            torch.tensor([0.02, 0.98], device="cuda")).tolist()
    # A vertex shows a swapped face if any incident face is swapped (blue over red).
    rank = torch.zeros(faces.shape[0], device="cuda")
    rank[keep_peak & ~keep_integral] = 1
    rank[keep_integral & ~keep_peak] = 2
    vertex_rank = vertex_values(faces, rank, count).long()
    vertex_swap = torch.tensor([[0.80, 0.80, 0.80], [0.714, 0.263, 0.259], [0.059, 0.302, 0.573]],
                               device="cuda")[vertex_rank]

    maps = {
        "rgb": None,
        "peak": heat(vertex_values(faces, peak, count), 0.0, 1.0, "magma"),
        "integral": heat(vertex_values(faces, log_integral, count), lo, hi, "magma"),
        "swap": vertex_swap,
    }
    triangles.scaling = 4
    with torch.no_grad():
        for name, colors in maps.items():
            package = render(view, triangles, pipeline, background, override_color=colors)
            to_image(package["render"]).save(out / f"{name}.png")
        # Per-pixel maps: every pixel shows the value of the face that dominates it
        # (the rasterizer's id buffer), so faces are not blended into their neighbours.
        ids = package["rend_ids"][0].long()
        hit = ids >= 0
        face = ids.clamp_min(0)
        fill = torch.ones((*ids.shape, 3), device="cuda")
        cls = torch.zeros(faces.shape[0], dtype=torch.long, device="cuda")
        cls[keep_peak & keep_integral] = 1
        cls[keep_peak & ~keep_integral] = 2
        cls[keep_integral & ~keep_peak] = 3
        palette = torch.tensor([[0.35, 0.35, 0.35], [0.82, 0.82, 0.82],
                                [0.714, 0.263, 0.259], [0.059, 0.302, 0.573]], device="cuda")
        per_pixel = {
            "peak_px": heat(peak[face], 0.0, 1.0, "magma"),
            "integral_px": heat(log_integral[face], lo, hi, "magma"),
            "swap_px": palette[cls[face]].reshape(-1, 3),
        }
        for name, colors in per_pixel.items():
            image = fill.clone()
            image[hit] = colors.reshape(*ids.shape, 3)[hit]
            to_image(image.permute(2, 0, 1)).save(out / f"{name}.png")
        np.save(out / "pixel_class.npy", cls[face].masked_fill(~hit, -1).to(torch.int8).cpu().numpy())
    report = {
        "run": str(run_dir), "view": args.view, "faces": int(faces.shape[0]),
        "kept": int(keep_peak.sum()),
        "swapped_each_way": int((keep_integral & ~keep_peak).sum()),
        "log10_integral_range": [lo, hi],
    }
    (out / "survival_maps.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--view", required=True)
    parser.add_argument("--view_set", default="test", choices=["test", "train"])
    parser.add_argument("--white", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    safe_state(args.quiet)
    run(model.extract(args), pipeline.extract(args), args)
