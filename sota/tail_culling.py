"""Select and test one inference-time transmittance cutoff on frozen Room."""

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
from sota.depth_opacity_core import evenly_spaced_indices
from sota.tail_culling_core import (
    DEFAULT_THRESHOLD,
    MINIMUM_FPS_MULTIPLIER,
    SELECTION_PSNR_TOLERANCE_DB,
    SELECTION_VIEWS,
    TEST_PSNR_TOLERANCE_DB,
    THRESHOLDS,
    choose_threshold,
    passes_test_gate,
)
from triangle_renderer import render
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim


EXPECTED_OPACITY_FLOOR = 0.8


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(values):
    return float(statistics.fmean(values))


def _render_timed(view, triangles, pipeline, background, threshold):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    prediction = render(
        view,
        triangles,
        pipeline,
        background,
        transmittance_threshold_override=threshold,
    )["render"].clamp(0.0, 1.0)
    end.record()
    torch.cuda.synchronize()
    return prediction, float(start.elapsed_time(end))


def _measure_selection(views, triangles, pipeline, background, threshold):
    with torch.no_grad():
        render(
            views[0], triangles, pipeline, background,
            transmittance_threshold_override=threshold,
        )
        rows = []
        for view in views:
            prediction, render_ms = _render_timed(
                view, triangles, pipeline, background, threshold
            )
            target = view.original_image[:3].to(prediction.device).clamp(0.0, 1.0)
            rows.append({
                "view": view.image_name,
                "psnr": float(psnr(prediction, target).mean()),
                "render_ms": render_ms,
            })
    return {
        "psnr": _mean(row["psnr"] for row in rows),
        "render_ms": _mean(row["render_ms"] for row in rows),
        "fps": 1000.0 / _mean(row["render_ms"] for row in rows),
        "per_view": rows,
    }


def _evaluate(views, triangles, pipeline, background, threshold, lpips_metric):
    rows = []
    with torch.no_grad():
        render(
            views[0], triangles, pipeline, background,
            transmittance_threshold_override=threshold,
        )
        for view in views:
            prediction, render_ms = _render_timed(
                view, triangles, pipeline, background, threshold
            )
            target = view.original_image[:3].to(prediction.device).clamp(0.0, 1.0)
            rows.append({
                "view": view.image_name,
                "l1": float(l1_loss(prediction, target).mean()),
                "psnr": float(psnr(prediction, target).mean()),
                "ssim": float(ssim(prediction, target).mean()),
                "lpips_vgg": float(lpips_metric(prediction, target).mean()),
                "render_ms": render_ms,
            })
    mean_render_ms = _mean(row["render_ms"] for row in rows)
    return {
        key: _mean(row[key] for row in rows)
        for key in ("l1", "psnr", "ssim", "lpips_vgg")
    } | {
        "render_ms": mean_render_ms,
        "fps": 1000.0 / mean_render_ms,
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
    if float(triangles.opacity_floor) != EXPECTED_OPACITY_FLOOR:
        raise RuntimeError(
            f"checkpoint opacity_floor is {triangles.opacity_floor}, expected 0.8"
        )
    if triangles.texel_order != 0:
        raise RuntimeError("tail culling requires the SH-only opacity-0.8 checkpoint")

    train = sorted(scene.getTrainCameras(), key=lambda view: view.image_name)
    test = sorted(scene.getTestCameras(), key=lambda view: view.image_name)
    selection = [train[index] for index in evenly_spaced_indices(len(train), SELECTION_VIEWS)]
    if not test:
        raise RuntimeError("tail culling requires the official test split")

    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )
    pipeline.texel_footprint_filter = False
    started = time.perf_counter()
    with torch.no_grad():
        ordinary = render(selection[0], triangles, pipeline, background)
        explicit = render(
            selection[0], triangles, pipeline, background,
            transmittance_threshold_override=DEFAULT_THRESHOLD,
        )
    default_is_bitwise = all(
        torch.equal(ordinary[key], explicit[key]) for key in ("render", "full_image")
    )
    if not default_is_bitwise:
        raise RuntimeError("explicit 1e-4 changed the default parent render")
    del ordinary, explicit

    selection_measurements = {
        threshold: _measure_selection(
            selection, triangles, pipeline, background, threshold
        )
        for threshold in THRESHOLDS
    }
    selected_threshold = choose_threshold({
        threshold: {
            "psnr": row["psnr"],
            "render_ms": row["render_ms"],
        }
        for threshold, row in selection_measurements.items()
    })

    lpips_metric = lpips.LPIPS(net="vgg").cuda().eval()
    baseline = _evaluate(
        test, triangles, pipeline, background, DEFAULT_THRESHOLD, lpips_metric
    )
    selected = (
        baseline if selected_threshold == DEFAULT_THRESHOLD else
        _evaluate(
            test, triangles, pipeline, background, selected_threshold, lpips_metric
        )
    )
    decision = "continue" if passes_test_gate(baseline, selected) else "stop"
    source_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    manifest = {
        "experiment": "transmittance-tail-culling-v0",
        "scene": args.scene,
        "source_revision": source_revision,
        "dataset": str(Path(dataset.source_path).resolve()),
        "model": str(Path(dataset.model_path).resolve()),
        "iteration": args.iteration,
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_opacity_floor": float(triangles.opacity_floor),
        "threshold_grid": list(THRESHOLDS),
        "selection_view_names": [view.image_name for view in selection],
        "test_view_names": [view.image_name for view in test],
        "selection_rule": (
            "minimum mean render_ms within 0.02 dB of default train-subset PSNR; "
            "exact timing tie prefers smaller threshold"
        ),
        "selection_psnr_tolerance_db": SELECTION_PSNR_TOLERANCE_DB,
        "test_psnr_tolerance_db": TEST_PSNR_TOLERANCE_DB,
        "minimum_fps_multiplier": MINIMUM_FPS_MULTIPLIER,
        "default_threshold_bitwise_parent": default_is_bitwise,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
    }
    result = {
        "decision": decision,
        "selected_threshold": selected_threshold,
        "selection": {
            str(threshold): row for threshold, row in selection_measurements.items()
        },
        "baseline_test": baseline,
        "selected_test": selected,
        "test_psnr_delta_db": selected["psnr"] - baseline["psnr"],
        "test_fps_multiplier": selected["fps"] / baseline["fps"],
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
    print(json.dumps({
        "output": str(output),
        "decision": decision,
        "selected_threshold": selected_threshold,
        "selection": {
            str(threshold): {
                key: value for key, value in row.items() if key != "per_view"
            }
            for threshold, row in selection_measurements.items()
        },
        "baseline_test": {
            key: value for key, value in baseline.items() if key != "per_view"
        },
        "selected_test": {
            key: value for key, value in selected.items() if key != "per_view"
        },
        "test_psnr_delta_db": result["test_psnr_delta_db"],
        "test_fps_multiplier": result["test_fps_multiplier"],
        "elapsed_seconds": result["elapsed_seconds"],
    }, indent=2))


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
