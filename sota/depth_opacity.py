"""Frozen-checkpoint ceiling for deeper alpha compositing.

A fixed subset of training views selects one global opacity scale.  The official
test split is rendered only after that choice is frozen.  This is a diagnostic
ceiling, not a trained method.
"""

import hashlib
import json
import statistics
import subprocess
import time
from argparse import ArgumentParser
from pathlib import Path

import lpips
import torch

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from scene.triangle_model import TriangleModel
from sota.depth_opacity_core import (
    PSNR_GATE_DB,
    SCALES,
    SELECTION_VIEWS,
    choose_scale,
    evenly_spaced_indices,
)
from triangle_renderer import render
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim


FINAL_OPACITY_FLOOR = 0.9999


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(values):
    return float(statistics.fmean(values))


def _render_psnr(views, triangles, pipeline, background, scale):
    values = []
    with torch.no_grad():
        for view in views:
            prediction = render(
                view,
                triangles,
                pipeline,
                background,
                opacity_scale_override=scale,
            )["render"].clamp(0.0, 1.0)
            target = view.original_image[:3].to(prediction.device).clamp(0.0, 1.0)
            values.append(float(psnr(prediction, target).mean()))
    return _mean(values)


def _evaluate(views, triangles, pipeline, background, scale, lpips_metric):
    rows = []
    with torch.no_grad():
        for view in views:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            prediction = render(
                view,
                triangles,
                pipeline,
                background,
                opacity_scale_override=scale,
            )["render"].clamp(0.0, 1.0)
            end.record()
            torch.cuda.synchronize()
            target = view.original_image[:3].to(prediction.device).clamp(0.0, 1.0)
            rows.append({
                "view": view.image_name,
                "l1": float(l1_loss(prediction, target).mean()),
                "psnr": float(psnr(prediction, target).mean()),
                "ssim": float(ssim(prediction, target).mean()),
                "lpips_vgg": float(lpips_metric(prediction, target).mean()),
                "render_ms": float(start.elapsed_time(end)),
            })
    return {
        key: _mean(row[key] for row in rows)
        for key in ("l1", "psnr", "ssim", "lpips_vgg", "render_ms")
    } | {
        "fps": 1000.0 / _mean(row["render_ms"] for row in rows),
        "views": len(rows),
        "per_view": rows,
    }


def run(dataset, pipeline, args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    checkpoint = (
        Path(dataset.model_path) / "point_cloud" /
        f"iteration_{args.iteration}" / "point_cloud_state_dict.pt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

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
    triangles.opacity_floor = FINAL_OPACITY_FLOOR
    if triangles.texel_order != 0:
        raise RuntimeError("depth-opacity ceiling requires the SH-only baseline")

    train = sorted(scene.getTrainCameras(), key=lambda view: view.image_name)
    test = sorted(scene.getTestCameras(), key=lambda view: view.image_name)
    selection = [train[index] for index in evenly_spaced_indices(len(train), SELECTION_VIEWS)]
    if not test:
        raise RuntimeError("depth-opacity ceiling requires the official test split")

    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )
    pipeline.texel_footprint_filter = False
    started = time.perf_counter()
    with torch.no_grad():
        ordinary = render(selection[0], triangles, pipeline, background)["render"]
        scale_one = render(
            selection[0],
            triangles,
            pipeline,
            background,
            opacity_scale_override=1.0,
        )["render"]
    if not torch.equal(ordinary, scale_one):
        raise RuntimeError("opacity scale 1.0 changed the parent render")
    del ordinary, scale_one
    selection_psnr = {
        scale: _render_psnr(selection, triangles, pipeline, background, scale)
        for scale in SCALES
    }
    selected_scale = choose_scale(selection_psnr)

    lpips_metric = lpips.LPIPS(net="vgg").cuda().eval()
    baseline = _evaluate(test, triangles, pipeline, background, 1.0, lpips_metric)
    selected = (baseline if selected_scale == 1.0 else
                _evaluate(test, triangles, pipeline, background, selected_scale, lpips_metric))
    psnr_gain = selected["psnr"] - baseline["psnr"]
    decision = "continue" if psnr_gain >= PSNR_GATE_DB else "stop"

    source_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    manifest = {
        "experiment": "depth-opacity-ceiling-v0",
        "scene": args.scene,
        "source_revision": source_revision,
        "dataset": str(Path(dataset.source_path).resolve()),
        "model": str(Path(dataset.model_path).resolve()),
        "iteration": args.iteration,
        "checkpoint_sha256": _sha256(checkpoint),
        "scale_grid": list(SCALES),
        "selection_view_names": [view.image_name for view in selection],
        "test_view_names": [view.image_name for view in test],
        "selection_rule": "maximum mean train-subset PSNR; exact tie prefers larger scale",
        "gate_psnr_db": PSNR_GATE_DB,
        "scale_one_bitwise_parent": True,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
    }
    result = {
        "decision": decision,
        "selected_scale": selected_scale,
        "selection_mean_psnr": {str(key): value for key, value in selection_psnr.items()},
        "baseline_test": baseline,
        "selected_test": selected,
        "test_psnr_gain_db": psnr_gain,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output.mkdir(parents=True)
    for name, payload in (("manifest.json", manifest), ("result.json", result)):
        with open(output / name, "x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
    checksums = "".join(
        f"{_sha256(output / name)}  {name}\n"
        for name in ("manifest.json", "result.json")
    )
    (output / "SHA256SUMS").write_text(checksums, encoding="utf-8")
    (output / "DONE").write_text("complete\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **result}, indent=2))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scene", default="room")
    parser.add_argument("--quiet", action="store_true")
    parsed = get_combined_args(parser)
    if not parsed.eval:
        parser.error("--eval is required")
    safe_state(parsed.quiet)
    run(model.extract(parsed), pipeline.extract(parsed), parsed)
