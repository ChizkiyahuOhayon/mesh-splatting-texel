"""Render one test view of a trained mesh for paper figures.

Writes, for the chosen view, the RGB render under a given evaluator arm, the
world-space normal map, and a wireframe overlay of the triangles visible in a
crop (a face is drawn when its centroid depth matches the rendered median depth
at the centroid pixel). Everything is read from the checkpoint; nothing trains.

    python -m sota.figure_renders -s <scene> -m <model> -i images_4 --eval \
        --view 010_DSC8760 --arm ours_quality --crop 300 400 256 --out <dir>
"""

import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from scene.triangle_model import TriangleModel
from sota.main_table_eval import SETTINGS
from triangle_renderer import render
from utils.general_utils import safe_state


def to_image(tensor):
    array = tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((array * 255 + 0.5).astype(np.uint8))


def wireframe(view, triangles, depth, crop, scale, rgb):
    """Edges of the faces visible inside ``crop`` drawn over the upscaled RGB crop."""
    row, col, size = crop
    vertices = triangles.vertices.detach()
    faces = triangles._triangle_indices.long()
    ones = torch.ones_like(vertices[:, :1])
    clip = torch.cat([vertices, ones], 1) @ view.full_proj_transform
    ndc = clip[:, :2] / clip[:, 3:4]
    width, height = view.image_width, view.image_height
    pixel = torch.stack(((ndc[:, 0] + 1) * width / 2 - 0.5, (ndc[:, 1] + 1) * height / 2 - 0.5), 1)
    camera = torch.cat([vertices, ones], 1) @ view.world_view_transform
    z = camera[:, 2]

    centroid = pixel[faces].mean(1)
    centroid_z = z[faces].mean(1)
    inside = ((centroid[:, 0] >= col) & (centroid[:, 0] < col + size)
              & (centroid[:, 1] >= row) & (centroid[:, 1] < row + size) & (centroid_z > 0))
    idx = inside.nonzero().squeeze(1)
    cx = centroid[idx, 0].round().long().clamp(0, width - 1)
    cy = centroid[idx, 1].round().long().clamp(0, height - 1)
    surface = depth[0, cy, cx]
    visible = idx[(centroid_z[idx] - surface).abs() <= 0.01 * surface.clamp_min(1e-6)]

    canvas = rgb.crop((col, row, col + size, row + size)).resize((size * scale, size * scale),
                                                                  Image.LANCZOS)
    draw = ImageDraw.Draw(canvas)
    corners = ((pixel[faces[visible]] - torch.tensor([col, row], device=pixel.device)) * scale)
    for tri in corners.cpu().numpy():
        draw.line([tuple(tri[0]), tuple(tri[1]), tuple(tri[2]), tuple(tri[0])],
                  fill=(255, 255, 255), width=1)
    return canvas, int(visible.numel())


def run(dataset, pipeline, args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    triangles = TriangleModel(dataset.sh_degree)
    scene = Scene(args=dataset, triangles=triangles, init_opacity=None, set_sigma=None,
                  load_iteration=args.iteration, shuffle=False)
    triangles.scaling = 4
    setting = SETTINGS[args.arm]
    triangles.opacity_floor = setting["opacity_floor"]
    views = sorted(scene.getTestCameras(), key=lambda v: v.image_name)
    view = next(v for v in views if v.image_name == args.view.split("_", 1)[-1]
                or v.image_name == args.view)
    background = torch.zeros(3, device="cuda")
    with torch.no_grad():
        package = render(view, triangles, pipeline, background,
                         upsample_override=setting["upsample"],
                         transmittance_threshold_override=setting["threshold"],
                         absorb_transmittance_tail=setting["absorb_tail"])
    rgb = to_image(package["render"])
    rgb.save(out / "rgb.png")
    normal = package["rend_normal"] if "rend_normal" in package else package["render_normal_full"]
    normal = torch.nn.functional.interpolate(normal[None], size=package["render"].shape[1:],
                                             mode="area")[0]
    to_image(normal * 0.5 + 0.5).save(out / "normal.png")
    report = {"view": view.image_name, "arm": args.arm, "faces": int(triangles._triangle_indices.shape[0])}
    if args.crop:
        depth = package.get("surf_depth", package.get("rend_depth"))
        depth = torch.nn.functional.interpolate(depth[None], size=package["render"].shape[1:],
                                                mode="nearest")[0]
        canvas, drawn = wireframe(view, triangles, depth, args.crop, args.scale, rgb)
        canvas.save(out / "wire.png")
        report["crop"], report["faces_drawn"] = args.crop, drawn
    (out / "figure.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--view", required=True)
    parser.add_argument("--arm", choices=tuple(SETTINGS), required=True)
    parser.add_argument("--crop", type=int, nargs=3, metavar=("ROW", "COL", "SIZE"))
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--out", required=True)
    parsed = get_combined_args(parser)
    safe_state(True)
    run(model.extract(parsed), pipeline.extract(parsed), parsed)
