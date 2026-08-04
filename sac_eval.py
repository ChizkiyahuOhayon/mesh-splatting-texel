"""SAC-G0: evaluate one trained arm at both supersampling rates, with timing.

MeshSplatting always deploys at four times linear resolution, so every rendered
pixel costs sixteen samples. Evaluating each arm at `scaling 2` and `scaling 4`
separates what the renderer ladder confounded: the cost of the sampling rate
itself from the cost of rendering a model at a rate it was not trained for.
"""

import json
import time
from argparse import ArgumentParser
from pathlib import Path

import torch

from arguments import ModelParams, PipelineParams, get_combined_args
from lpipsPyTorch.modules.lpips import LPIPS
from sac_decide import decide
from scene import Scene
from scene.triangle_model import TriangleModel
from svsr_g1_eval import _checkpoint, _metrics, _sha256
from svsr_metadata import reserve_output_directory, source_revision
from triangle_renderer import render
from utils.general_utils import safe_state


EVALUATED_SCALINGS = (2, 4)
TIMING_WARMUP_VIEWS = 3


def _evaluate_at(scaling, views, triangles, pipeline, background, lpips_metric):
    """Metrics and CUDA-event render time for one supersampling rate."""
    triangles.scaling = scaling
    rows = []
    elapsed_ms = 0.0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        for index, view in enumerate(views):
            timed = index >= TIMING_WARMUP_VIEWS
            if timed:
                start.record()
            prediction = render(view, triangles, pipeline, background)["render"]
            if timed:
                end.record()
                torch.cuda.synchronize()
                elapsed_ms += start.elapsed_time(end)
            target = view.original_image[:3].to(prediction.device)
            rows.append(
                {"view": view.image_name, **_metrics(prediction, target, lpips_metric)}
            )
    timed_views = max(1, len(views) - TIMING_WARMUP_VIEWS)
    return {
        "scaling": scaling,
        **{
            key: sum(row[key] for row in rows) / len(rows)
            for key in ("psnr", "ssim", "lpips_vgg")
        },
        "render_ms_per_view": elapsed_ms / timed_views,
        "timed_views": timed_views,
        "per_view": rows,
    }


def run(dataset, pipeline, args):
    output_root = reserve_output_directory(Path(args.sac_output).resolve())
    iteration, checkpoint = _checkpoint(dataset.model_path, args.iteration)

    triangles = TriangleModel(dataset.sh_degree)
    scene = Scene(
        args=dataset,
        triangles=triangles,
        init_opacity=None,
        set_sigma=None,
        load_iteration=iteration,
        shuffle=False,
    )
    if triangles.texel_order != 0:
        raise RuntimeError("SAC-G0 evaluates SH-only checkpoints.")
    views = sorted(scene.getTestCameras(), key=lambda view: view.image_name)
    if len(views) <= TIMING_WARMUP_VIEWS:
        raise RuntimeError(
            f"need more than {TIMING_WARMUP_VIEWS} held-out views for timing"
        )
    pipeline.texel_footprint_filter = False
    background_values = [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0]
    background = torch.tensor(background_values, dtype=torch.float32, device="cuda")
    lpips_metric = LPIPS("vgg", "0.1").cuda().eval()

    manifest = {
        "protocol": "experiments/sac_g0/protocol.md",
        "arm": args.sac_arm,
        "scene": args.sac_scene,
        "seed": args.sac_seed,
        "dataset_path": str(Path(dataset.source_path).resolve()),
        "model_path": str(Path(dataset.model_path).resolve()),
        "iteration": iteration,
        "checkpoint_sha256": _sha256(checkpoint),
        "source_revision": source_revision(),
        "test_views": len(views),
        "evaluated_scalings": list(EVALUATED_SCALINGS),
        "timing_warmup_views": TIMING_WARMUP_VIEWS,
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        "torch": torch.__version__,
    }
    with open(output_root / "sac_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    cells = {
        f"scaling_{scaling}": _evaluate_at(
            scaling, views, triangles, pipeline, background, lpips_metric
        )
        for scaling in EVALUATED_SCALINGS
    }
    torch.cuda.synchronize()
    results = {
        "arm": args.sac_arm,
        "scene": args.sac_scene,
        "seed": args.sac_seed,
        "cells": cells,
        "primitives": {
            "vertices": int(triangles.vertices.shape[0]),
            "triangles": int(triangles._triangle_indices.shape[0]),
        },
        "training_seconds": args.sac_training_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    with open(output_root / "results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, allow_nan=False)
    (output_root / "DONE").write_text("complete\n", encoding="utf-8")
    print(json.dumps({"results": str(output_root / "results.json"), **results}, indent=2))


if __name__ == "__main__":
    parser = ArgumentParser(description="SAC-G0 two-rate evaluation of one arm")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--sac_arm", required=True, choices=("stock", "splat2"))
    parser.add_argument("--sac_scene", required=True)
    parser.add_argument("--sac_seed", type=int, default=0)
    parser.add_argument("--sac_output", required=True)
    parser.add_argument(
        "--sac_training_seconds",
        type=float,
        default=None,
        help="wall-clock of the training run that produced this model, for the record",
    )
    parsed = get_combined_args(parser)
    if not parsed.model_path or not parsed.source_path:
        parser.error("Both --model_path/-m and --source_path/-s are required.")
    if not parsed.eval:
        parser.error("SAC-G0 requires --eval so the held-out split is fixed.")
    safe_state(parsed.quiet)
    run(model.extract(parsed), pipeline.extract(parsed), parsed)
