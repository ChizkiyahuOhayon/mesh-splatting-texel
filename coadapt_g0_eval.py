"""Evaluate the preregistered COADAPT-G0 checkpoint decomposition on Room."""

import copy
import json
from argparse import ArgumentParser
from pathlib import Path

import torch

from arguments import ModelParams, PipelineParams, get_combined_args
from coadapt_decompose import recovery_fraction, texel_variants
from lpipsPyTorch.modules.lpips import LPIPS
from scene import Scene
from scene.triangle_model import TriangleModel
from svsr_g1_eval import _checkpoint, _metrics, _save, _sha256, _summarize
from svsr_metadata import reserve_output_directory, source_revision
from triangle_renderer import render
from utils.general_utils import safe_state


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


def run(dataset, pipeline, args):
    output_root = reserve_output_directory(Path(args.coadapt_output).resolve())
    texel_iteration, texel_checkpoint = _checkpoint(dataset.model_path, args.iteration)
    sh_iteration, sh_checkpoint = _checkpoint(args.coadapt_sh_model, args.coadapt_sh_iteration)

    sh_scene, sh_triangles, sh_views = _load_scene(
        dataset, args.coadapt_sh_model, sh_iteration)
    if not sh_views:
        raise RuntimeError("COADAPT-G0 requires a non-empty held-out test split.")
    if sh_triangles.texel_order != 0:
        raise RuntimeError("--coadapt_sh_model must be an SH-only checkpoint.")
    if args.coadapt_max_views > 0:
        sh_views = sh_views[:args.coadapt_max_views]

    manifest = {
        "protocol": "experiments/coadapt_g0/protocol.md",
        "scene": "room",
        "dataset_path": str(Path(dataset.source_path).resolve()),
        "sh_model": str(Path(args.coadapt_sh_model).resolve()),
        "sh_iteration": sh_iteration,
        "sh_checkpoint_sha256": _sha256(sh_checkpoint),
        "texel_model": str(Path(dataset.model_path).resolve()),
        "texel_iteration": texel_iteration,
        "texel_checkpoint_sha256": _sha256(texel_checkpoint),
        "test_views": [view.image_name for view in sh_views],
        "image_sizes": sorted({
            (int(view.image_height), int(view.image_width)) for view in sh_views
        }),
        "variants": ["sh_reference", "fixed", "zero", "face_mean"],
        "max_views": args.coadapt_max_views,
        "confirmatory_settings": args.coadapt_max_views == 0,
        "git_commit": source_revision(),
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        "torch": torch.__version__,
    }
    with open(output_root / "coadapt_manifest.json", "w", encoding="utf-8") as handle:
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
            rows.append({"view": view.image_name, "variant": "sh_reference",
                         **_metrics(prediction, gt, lpips_metric)})
            _save(prediction, output_root, "sh_reference", index)
            _save(gt, output_root, "gt", index)

    del sh_scene, sh_triangles, sh_views
    torch.cuda.empty_cache()

    texel_scene, texel_triangles, texel_views = _load_scene(
        dataset, dataset.model_path, texel_iteration)
    if texel_triangles.texel_order <= 0:
        raise RuntimeError("The evaluated model does not contain residual texels.")
    if args.coadapt_max_views > 0:
        texel_views = texel_views[:args.coadapt_max_views]
    if [view.image_name for view in texel_views] != manifest["test_views"]:
        raise RuntimeError("SH and texel checkpoints resolve to different test views.")

    original = texel_triangles.get_texels.detach().clone()
    variants = {"fixed": original, **texel_variants(original)}
    try:
        with torch.no_grad():
            for variant, carrier in variants.items():
                texel_triangles.get_texels.copy_(carrier)
                for index, view in enumerate(texel_views):
                    prediction = render(view, texel_triangles, pipeline, background)["render"]
                    gt = view.original_image[:3]
                    rows.append({"view": view.image_name, "variant": variant,
                                 **_metrics(prediction, gt, lpips_metric)})
                    _save(prediction, output_root, variant, index)
    finally:
        with torch.no_grad():
            texel_triangles.get_texels.copy_(original)

    names = ("sh_reference", "fixed", "zero", "face_mean")
    summary = {name: _summarize(rows, name) for name in names}
    recovery = {
        name: {
            "psnr": recovery_fraction(
                summary["sh_reference"]["psnr"], summary["fixed"]["psnr"],
                summary[name]["psnr"], True),
            "lpips_vgg": recovery_fraction(
                summary["sh_reference"]["lpips_vgg"], summary["fixed"]["lpips_vgg"],
                summary[name]["lpips_vgg"], False),
        }
        for name in ("zero", "face_mean")
    }
    results = {
        "scene": "room",
        "summary": summary,
        "recovery_fraction": recovery,
        "per_view": rows,
    }
    with open(output_root / "results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(json.dumps({"results": str(output_root / "results.json"),
                      "summary": summary, "recovery_fraction": recovery}, indent=2))


if __name__ == "__main__":
    parser = ArgumentParser(description="COADAPT-G0 frozen-checkpoint decomposition")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--coadapt_sh_model", required=True)
    parser.add_argument("--coadapt_sh_iteration", default=-1, type=int)
    parser.add_argument("--coadapt_output", required=True)
    parser.add_argument("--coadapt_max_views", default=0, type=int,
                        help="Exploratory smoke only; 0 evaluates the full test split.")
    parsed = get_combined_args(parser)
    if not parsed.model_path or not parsed.source_path:
        parser.error("Both --model_path/-m and --source_path/-s are required.")
    if not parsed.eval:
        parser.error("COADAPT-G0 requires --eval so held-out views are loaded.")
    if parsed.coadapt_max_views < 0:
        parser.error("--coadapt_max_views must be non-negative.")
    safe_state(parsed.quiet)
    run(model.extract(parsed), pipeline.extract(parsed), parsed)
