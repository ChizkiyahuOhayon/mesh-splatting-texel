"""Run one scene of the preregistered FMMS G0 native-AA evaluation."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess

import torch
import torch.nn.functional as F
import torchvision
from argparse import ArgumentParser

from arguments import ModelParams, PipelineParams, get_combined_args
from fmms_native_renderer import create_raster_context, render_native
from lpipsPyTorch.modules.lpips import LPIPS
from scene import Scene
from scene.triangle_model import TriangleModel
from triangle_renderer import render as render_baseline
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import ssim
from utils.system_utils import searchForMaxIteration


VARIANTS = ("ssaa4", "point1", "aa1", "aa2")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _git_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _render_variant(variant, view, triangles, pipeline, background, context):
    if variant == "ssaa4":
        previous_scale = triangles.scaling
        triangles.scaling = 4
        try:
            package = render_baseline(view, triangles, pipeline, background)
            return {"render": package["render"], "rend_alpha": package["rend_alpha"]}
        finally:
            triangles.scaling = previous_scale
    if variant == "point1":
        return render_native(view, triangles, background, context, scale=1, antialias=False)
    if variant == "aa1":
        return render_native(view, triangles, background, context, scale=1, antialias=True)
    if variant == "aa2":
        return render_native(view, triangles, background, context, scale=2, antialias=True)
    raise ValueError(f"Unknown G0 variant: {variant}")


def _silhouette_band(alpha):
    alpha = alpha[None]
    local_max = F.max_pool2d(alpha, kernel_size=3, stride=1, padding=1)
    local_min = -F.max_pool2d(-alpha, kernel_size=3, stride=1, padding=1)
    return (local_max - local_min > 0.01)[0]


def _image_metrics(prediction, target, lpips_metric):
    prediction = prediction.clamp(0.0, 1.0)[None]
    target = target.clamp(0.0, 1.0)[None]
    return {
        "psnr": float(psnr(prediction, target).mean()),
        "ssim": float(ssim(prediction, target)),
        "lpips_vgg": float(lpips_metric(prediction, target).mean()),
    }


def _summary(rows):
    keys = ("psnr", "ssim", "lpips_vgg", "mae_vs_ssaa4", "p95_vs_ssaa4", "silhouette_mae_vs_ssaa4")
    summary = {}
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        summary[variant] = {
            key: float(statistics.fmean(row[key] for row in selected))
            for key in keys if all(row[key] is not None for row in selected)
        }
    reference = summary["ssaa4"]
    for variant in VARIANTS[1:]:
        summary[variant]["delta_vs_ssaa4"] = {
            "psnr": summary[variant]["psnr"] - reference["psnr"],
            "ssim": summary[variant]["ssim"] - reference["ssim"],
            "lpips_vgg": summary[variant]["lpips_vgg"] - reference["lpips_vgg"],
        }
    return summary


def _benchmark(variant, views, triangles, pipeline, background, context, warmup, repeats):
    with torch.no_grad():
        for view in views:
            for _ in range(warmup):
                output = _render_variant(variant, view, triangles, pipeline, background, context)
                del output
        torch.cuda.synchronize()

        samples_ms = []
        for view in views:
            for _ in range(repeats):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                output = _render_variant(variant, view, triangles, pipeline, background, context)
                end.record()
                end.synchronize()
                samples_ms.append(float(start.elapsed_time(end)))
                del output

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base_bytes = torch.cuda.memory_allocated()
        output = _render_variant(variant, views[0], triangles, pipeline, background, context)
        torch.cuda.synchronize()
        peak_increment = max(0, torch.cuda.max_memory_allocated() - base_bytes)
        del output

    ordered = sorted(samples_ms)
    return {
        "samples_ms": samples_ms,
        "median_ms": float(statistics.median(samples_ms)),
        "q1_ms": float(ordered[len(ordered) // 4]),
        "q3_ms": float(ordered[(3 * len(ordered)) // 4]),
        "peak_increment_bytes": int(peak_increment),
    }


def run(dataset, pipeline, args):
    output_root = Path(args.g0_output).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"G0 output directory must be empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    point_cloud_root = Path(dataset.model_path) / "point_cloud"
    iteration = args.iteration if args.iteration >= 0 else searchForMaxIteration(str(point_cloud_root))
    checkpoint = point_cloud_root / f"iteration_{iteration}" / "point_cloud_state_dict.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    triangles = TriangleModel(dataset.sh_degree)
    scene = Scene(
        args=dataset,
        triangles=triangles,
        init_opacity=None,
        set_sigma=None,
        load_iteration=iteration,
        shuffle=False,
    )
    views = sorted(scene.getTestCameras(), key=lambda view: view.image_name)
    if not views:
        raise RuntimeError("G0 requires a non-empty test split; pass --eval and the correct dataset.")
    if args.g0_max_views > 0:
        views = views[:args.g0_max_views]
    if triangles.texel_order != 0:
        raise RuntimeError("G0 requires an unmodified baseline checkpoint (texel_order=0).")

    manifest = {
        "protocol": "experiments/fmms_g0/protocol.md",
        "scene": args.g0_scene,
        "dataset_path": str(Path(dataset.source_path).resolve()),
        "model_path": str(Path(dataset.model_path).resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "iteration": iteration,
        "git_commit": _git_commit(),
        "variants": list(VARIANTS),
        "warmup": args.g0_warmup,
        "timing_repeats": args.g0_timing_repeats,
        "timing_views": args.g0_timing_views,
        "max_views": args.g0_max_views,
        "test_views": [view.image_name for view in views],
        "image_sizes": sorted({
            (int(view.image_height), int(view.image_width)) for view in views
        }),
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        "torch": torch.__version__,
        "nvdiffrast": _package_version("nvdiffrast"),
        "confirmatory_settings": (
            args.g0_warmup == 5 and args.g0_timing_repeats == 20
            and args.g0_timing_views == 5 and args.g0_max_views == 0
        ),
    }
    with open(output_root / "g0_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    context = create_raster_context()
    background_values = [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0]
    background = torch.tensor(background_values, dtype=torch.float32, device="cuda")
    lpips_metric = LPIPS("vgg", "0.1").cuda().eval()

    rows = []
    with torch.no_grad():
        for view_index, view in enumerate(views):
            gt = view.original_image[:3]
            rendered = {
                variant: _render_variant(variant, view, triangles, pipeline, background, context)
                for variant in VARIANTS
            }
            reference = rendered["ssaa4"]["render"].clamp(0.0, 1.0)
            band = _silhouette_band(rendered["ssaa4"]["rend_alpha"])

            if view_index == 0:
                native = rendered["aa1"]["render"].clamp(0.0, 1.0)
                direct_error = torch.mean(torch.abs(native - reference))
                flip_y_error = torch.mean(torch.abs(native.flip(-2) - reference))
                flip_x_error = torch.mean(torch.abs(native.flip(-1) - reference))
                if min(flip_y_error, flip_x_error) < 0.5 * direct_error:
                    raise RuntimeError(
                        "Native projection orientation sanity check failed; refusing to score flipped output."
                    )

            for variant, output in rendered.items():
                prediction = output["render"].clamp(0.0, 1.0)
                metrics = _image_metrics(prediction, gt, lpips_metric)
                if variant == "ssaa4":
                    mae = p95 = silhouette_mae = None
                else:
                    absolute = torch.abs(prediction - reference)
                    mae = float(absolute.mean())
                    p95 = float(torch.quantile(absolute, 0.95))
                    silhouette_mae = float(absolute[:, band[0]].mean()) if band.any() else 0.0
                rows.append({
                    "view": view.image_name,
                    "variant": variant,
                    **metrics,
                    "mae_vs_ssaa4": mae,
                    "p95_vs_ssaa4": p95,
                    "silhouette_mae_vs_ssaa4": silhouette_mae,
                })
                render_dir = output_root / variant / "renders"
                render_dir.mkdir(parents=True, exist_ok=True)
                torchvision.utils.save_image(prediction, render_dir / f"{view_index:05d}.png")

            gt_dir = output_root / "gt"
            gt_dir.mkdir(parents=True, exist_ok=True)
            torchvision.utils.save_image(gt, gt_dir / f"{view_index:05d}.png")

    results = {"scene": args.g0_scene, "per_view": rows, "summary": _summary(rows)}
    with open(output_root / "results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    timing_views = views[: min(args.g0_timing_views, len(views))]
    timing = {
        "scene": args.g0_scene,
        "view_names": [view.image_name for view in timing_views],
        "variants": {
            variant: _benchmark(
                variant, timing_views, triangles, pipeline, background, context,
                args.g0_warmup, args.g0_timing_repeats,
            )
            for variant in VARIANTS
        },
    }
    with open(output_root / "timing.json", "w", encoding="utf-8") as handle:
        json.dump(timing, handle, indent=2)

    print(json.dumps({"results": str(output_root / "results.json"), "summary": results["summary"]}, indent=2))


if __name__ == "__main__":
    parser = ArgumentParser(description="FMMS G0 frozen-checkpoint evaluation")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--g0_scene", required=True, choices=("garden", "room", "stump"))
    parser.add_argument("--g0_output", required=True)
    parser.add_argument("--g0_warmup", default=5, type=int)
    parser.add_argument("--g0_timing_repeats", default=20, type=int)
    parser.add_argument("--g0_timing_views", default=5, type=int)
    parser.add_argument("--g0_max_views", default=0, type=int,
                        help="Exploratory smoke test only; 0 uses the full test split.")
    parsed = get_combined_args(parser)
    if not parsed.model_path or not parsed.source_path:
        parser.error("Both --model_path/-m and --source_path/-s are required.")
    if not parsed.eval:
        parser.error("G0 requires --eval so the held-out test split is loaded.")
    if (parsed.g0_warmup < 0 or parsed.g0_timing_repeats < 1
            or parsed.g0_timing_views < 1 or parsed.g0_max_views < 0):
        parser.error("G0 timing counts must be positive (warmup may be zero).")
    safe_state(parsed.quiet)
    run(model.extract(parsed), pipeline.extract(parsed), parsed)
