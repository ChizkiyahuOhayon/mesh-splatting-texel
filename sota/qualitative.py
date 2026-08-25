"""Export deterministic test renders and error maps for paper figures."""

import json
import subprocess
from argparse import ArgumentParser
from pathlib import Path

import torch
import torchvision

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from scene.triangle_model import TriangleModel
from sota.main_table_eval import SETTINGS
from triangle_renderer import render
from utils.general_utils import safe_state


ARMS = ("stock", "ours_quality")
ERROR_SCALE = 4.0


def run(dataset, pipeline, args):
    output = Path(args.output).resolve()
    if (output / "DONE").is_file():
        print(f"Qualitative export already complete: {output}")
        return
    checkpoint = (
        Path(dataset.model_path) / "point_cloud" /
        f"iteration_{args.iteration}" / "point_cloud_state_dict.pt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    setting = SETTINGS[args.arm]
    triangles = TriangleModel(dataset.sh_degree)
    scene = Scene(
        args=dataset,
        triangles=triangles,
        init_opacity=None,
        set_sigma=None,
        load_iteration=args.iteration,
        shuffle=False,
    )
    triangles.scaling = 4
    triangles.opacity_floor = setting["opacity_floor"]
    pipeline.texel_footprint_filter = False
    views = sorted(scene.getTestCameras(), key=lambda view: view.image_name)
    if not views:
        raise RuntimeError("qualitative export requires the official test split")

    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )
    render_dir = output / "renders"
    target_dir = output.parent / "targets"
    error_dir = output / "errors_x4"
    for directory in (render_dir, target_dir, error_dir):
        directory.mkdir(parents=True, exist_ok=True)

    names = []
    with torch.no_grad():
        for index, view in enumerate(views):
            prediction = render(
                view,
                triangles,
                pipeline,
                background,
                transmittance_threshold_override=setting["threshold"],
                absorb_transmittance_tail=setting["absorb_tail"],
                upsample_override=setting["upsample"],
            )["render"].clamp(0.0, 1.0)
            target = view.original_image[:3].to(prediction.device).clamp(0.0, 1.0)
            error = (prediction - target).abs().mean(dim=0, keepdim=True)
            name = f"{index:03d}_{Path(view.image_name).stem}.png"
            torchvision.utils.save_image(prediction, render_dir / name)
            torchvision.utils.save_image(target, target_dir / name)
            torchvision.utils.save_image(
                (error * ERROR_SCALE).clamp(0.0, 1.0), error_dir / name
            )
            names.append(view.image_name)

    manifest = {
        "experiment": "formal-qualitative-export-v1",
        "scene": args.scene,
        "arm": args.arm,
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "checkpoint": str(checkpoint.resolve()),
        "settings": setting,
        "error_scale": ERROR_SCALE,
        "view_names": names,
    }
    with open(output / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
    (output / "DONE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--quiet", action="store_true")
    parsed = get_combined_args(parser)
    if not parsed.eval:
        parser.error("--eval is required")
    safe_state(parsed.quiet)
    run(model.extract(parsed), pipeline.extract(parsed), parsed)
