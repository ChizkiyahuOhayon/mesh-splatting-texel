"""Evaluate the preregistered SVSR-G1 footprint filter on one scene."""

import copy
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

import torch
import torchvision
from argparse import ArgumentParser

from arguments import ModelParams, PipelineParams, get_combined_args
from lpipsPyTorch.modules.lpips import LPIPS
from scene import Scene
from scene.triangle_model import TriangleModel
from triangle_renderer import render
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import ssim
from utils.system_utils import searchForMaxIteration


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint(model_path, requested_iteration):
    point_cloud = Path(model_path) / "point_cloud"
    iteration = requested_iteration
    if iteration < 0:
        iteration = searchForMaxIteration(str(point_cloud))
    path = point_cloud / f"iteration_{iteration}" / "point_cloud_state_dict.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return iteration, path


def _load_scene(dataset, model_path, iteration):
    scene_args = copy.copy(dataset)
    scene_args.model_path = str(Path(model_path).resolve())
    triangles = TriangleModel(scene_args.sh_degree)
    triangles.scaling = 4
    scene = Scene(
        args=scene_args,
        triangles=triangles,
        init_opacity=None,
        set_sigma=None,
        load_iteration=iteration,
        shuffle=False,
    )
    views = sorted(scene.getTestCameras(), key=lambda view: view.image_name)
    return scene, triangles, views


def _metrics(prediction, target, lpips_metric):
    prediction = prediction.clamp(0.0, 1.0)[None]
    target = target.clamp(0.0, 1.0)[None]
    return {
        "psnr": float(psnr(prediction, target).mean()),
        "ssim": float(ssim(prediction, target)),
        "lpips_vgg": float(lpips_metric(prediction, target).mean()),
    }


def _summarize(rows, variant):
    selected = [row for row in rows if row["variant"] == variant]
    return {
        key: float(statistics.fmean(row[key] for row in selected))
        for key in ("psnr", "ssim", "lpips_vgg")
    }


def _save(image, output_root, variant, index):
    directory = output_root / variant / "renders"
    directory.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(image.clamp(0.0, 1.0), directory / f"{index:05d}.png")


def run(dataset, pipeline, args):
    output_root = Path(args.svsr_output).resolve()
    if output_root.exists():
        raise RuntimeError(f"SVSR output path must not exist: {output_root}")

    texel_iteration, texel_checkpoint = _checkpoint(dataset.model_path, args.iteration)
    sh_iteration, sh_checkpoint = _checkpoint(args.svsr_sh_model, args.svsr_sh_iteration)

    sh_scene, sh_triangles, sh_views = _load_scene(
        dataset, args.svsr_sh_model, sh_iteration)
    if not sh_views:
        raise RuntimeError("SVSR-G1 requires a non-empty held-out test split.")
    if sh_triangles.texel_order != 0:
        raise RuntimeError("--svsr_sh_model must be an SH-only checkpoint.")
    if args.svsr_max_views > 0:
        sh_views = sh_views[:args.svsr_max_views]

    output_root.mkdir(parents=True)
    manifest = {
        "protocol": "experiments/svsr_g1/protocol.md",
        "scene": args.svsr_scene,
        "dataset_path": str(Path(dataset.source_path).resolve()),
        "sh_model": str(Path(args.svsr_sh_model).resolve()),
        "sh_iteration": sh_iteration,
        "sh_checkpoint_sha256": _sha256(sh_checkpoint),
        "texel_model": str(Path(dataset.model_path).resolve()),
        "texel_iteration": texel_iteration,
        "texel_checkpoint_sha256": _sha256(texel_checkpoint),
        "test_views": [view.image_name for view in sh_views],
        "image_sizes": sorted({
            (int(view.image_height), int(view.image_width)) for view in sh_views
        }),
        "filter": "mean + clamp(projected_area / texel_count, 0, 1) * detail",
        "max_views": args.svsr_max_views,
        "confirmatory_settings": args.svsr_max_views == 0,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        "torch": torch.__version__,
    }
    with open(output_root / "svsr_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    background_values = [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0]
    background = torch.tensor(background_values, dtype=torch.float32, device="cuda")
    lpips_metric = LPIPS("vgg", "0.1").cuda().eval()
    rows = []

    pipeline.texel_footprint_filter = False
    with torch.no_grad():
        for index, view in enumerate(sh_views):
            prediction = render(view, sh_triangles, pipeline, background)["render"]
            gt = view.original_image[:3]
            rows.append({"view": view.image_name, "variant": "sh",
                         **_metrics(prediction, gt, lpips_metric)})
            _save(prediction, output_root, "sh", index)
            _save(gt, output_root, "gt", index)

    del sh_scene, sh_triangles, sh_views
    torch.cuda.empty_cache()

    texel_scene, texel_triangles, texel_views = _load_scene(
        dataset, dataset.model_path, texel_iteration)
    if texel_triangles.texel_order <= 0:
        raise RuntimeError("The evaluated model does not contain residual texels.")
    if args.svsr_max_views > 0:
        texel_views = texel_views[:args.svsr_max_views]
    if [view.image_name for view in texel_views] != manifest["test_views"]:
        raise RuntimeError("SH and texel checkpoints resolve to different test views.")

    footprint_rows = []
    with torch.no_grad():
        for index, view in enumerate(texel_views):
            gt = view.original_image[:3]
            pipeline.texel_footprint_filter = False
            fixed = render(view, texel_triangles, pipeline, background)
            pipeline.texel_footprint_filter = True
            footprint = render(view, texel_triangles, pipeline, background)

            for variant, package in (("fixed", fixed), ("footprint", footprint)):
                prediction = package["render"]
                rows.append({"view": view.image_name, "variant": variant,
                             **_metrics(prediction, gt, lpips_metric)})
                _save(prediction, output_root, variant, index)

            rendered = footprint["triangle_was_rendered"] > 0
            weights = footprint["texel_footprint_weights"][rendered]
            footprint_rows.append({
                "view": view.image_name,
                "rendered_faces": int(weights.numel()),
                "q10": float(torch.quantile(weights, 0.10)),
                "q50": float(torch.quantile(weights, 0.50)),
                "q90": float(torch.quantile(weights, 0.90)),
                "fraction_full_detail": float((weights >= 1.0).float().mean()),
                "fraction_partial_detail": float(((weights > 0.0) & (weights < 1.0)).float().mean()),
                "fraction_zero_detail": float((weights <= 0.0).float().mean()),
            })

    summary = {variant: _summarize(rows, variant) for variant in ("sh", "fixed", "footprint")}
    summary["footprint"]["delta_vs_fixed"] = {
        key: summary["footprint"][key] - summary["fixed"][key]
        for key in ("psnr", "ssim", "lpips_vgg")
    }
    results = {
        "scene": args.svsr_scene,
        "summary": summary,
        "per_view": rows,
        "footprint_statistics": footprint_rows,
    }
    with open(output_root / "results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(json.dumps({"results": str(output_root / "results.json"), "summary": summary}, indent=2))


if __name__ == "__main__":
    parser = ArgumentParser(description="SVSR-G1 frozen-checkpoint evaluation")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--svsr_scene", required=True, choices=("garden", "room", "stump"))
    parser.add_argument("--svsr_sh_model", required=True)
    parser.add_argument("--svsr_sh_iteration", default=-1, type=int)
    parser.add_argument("--svsr_output", required=True)
    parser.add_argument("--svsr_max_views", default=0, type=int,
                        help="Exploratory smoke only; 0 evaluates the full test split.")
    parsed = get_combined_args(parser)
    if not parsed.model_path or not parsed.source_path:
        parser.error("Both --model_path/-m and --source_path/-s are required.")
    if not parsed.eval:
        parser.error("SVSR-G1 requires --eval so held-out views are loaded.")
    if parsed.svsr_max_views < 0:
        parser.error("--svsr_max_views must be non-negative.")
    safe_state(parsed.quiet)
    run(model.extract(parsed), pipeline.extract(parsed), parsed)
