"""Screen reduced supersampling with the frozen absorbed-tail renderer."""

import json
import subprocess
import time
from argparse import ArgumentParser
from pathlib import Path

import lpips
import torch

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from scene.triangle_model import TriangleModel
from sota.depth_opacity_core import SELECTION_VIEWS, evenly_spaced_indices
from sota.supersampling_core import (
    FACTORS,
    MINIMUM_FPS_MULTIPLIER,
    PSNR_TOLERANCE_DB,
    choose_factor,
    passes_test_gate,
)
from sota.tail_culling import _evaluate, _measure_selection, _sha256
from sota.tail_culling_core import DEFAULT_THRESHOLD
from utils.general_utils import safe_state


ABSORBED_THRESHOLD = 1e-2
EXPECTED_OPACITY_FLOOR = 0.8


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
        raise RuntimeError("supersampling screen requires the SH-only checkpoint")

    train = sorted(scene.getTrainCameras(), key=lambda view: view.image_name)
    test = sorted(scene.getTestCameras(), key=lambda view: view.image_name)
    selection = [train[index] for index in evenly_spaced_indices(len(train), SELECTION_VIEWS)]
    if not test:
        raise RuntimeError("supersampling screen requires the official test split")
    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )
    pipeline.texel_footprint_filter = False
    started = time.perf_counter()

    baseline_selection = _measure_selection(
        selection,
        triangles,
        pipeline,
        background,
        DEFAULT_THRESHOLD,
        False,
        4,
    )
    candidates = {
        factor: _measure_selection(
            selection,
            triangles,
            pipeline,
            background,
            ABSORBED_THRESHOLD,
            True,
            factor,
        )
        for factor in FACTORS
    }
    selected_factor = choose_factor(
        baseline_selection["psnr"],
        {
            factor: {"psnr": row["psnr"], "render_ms": row["render_ms"]}
            for factor, row in candidates.items()
        },
    )

    lpips_metric = lpips.LPIPS(net="vgg").cuda().eval()
    baseline_test = _evaluate(
        test,
        triangles,
        pipeline,
        background,
        DEFAULT_THRESHOLD,
        False,
        lpips_metric,
        4,
    )
    selected_test = _evaluate(
        test,
        triangles,
        pipeline,
        background,
        ABSORBED_THRESHOLD,
        True,
        lpips_metric,
        selected_factor,
    )
    decision = (
        "continue" if passes_test_gate(baseline_test, selected_test) else "stop"
    )
    source_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    manifest = {
        "experiment": "reduced-supersampling-v0",
        "scene": args.scene,
        "source_revision": source_revision,
        "dataset": str(Path(dataset.source_path).resolve()),
        "model": str(Path(dataset.model_path).resolve()),
        "iteration": args.iteration,
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_opacity_floor": float(triangles.opacity_floor),
        "factor_grid": list(FACTORS),
        "absorbed_threshold": ABSORBED_THRESHOLD,
        "selection_view_names": [view.image_name for view in selection],
        "test_view_names": [view.image_name for view in test],
        "selection_psnr_tolerance_db": PSNR_TOLERANCE_DB,
        "test_psnr_tolerance_db": PSNR_TOLERANCE_DB,
        "minimum_fps_multiplier": MINIMUM_FPS_MULTIPLIER,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
    }
    result = {
        "decision": decision,
        "selected_factor": selected_factor,
        "baseline_selection": baseline_selection,
        "candidate_selection": {
            str(factor): row for factor, row in candidates.items()
        },
        "baseline_test": baseline_test,
        "selected_test": selected_test,
        "test_psnr_delta_db": selected_test["psnr"] - baseline_test["psnr"],
        "test_fps_multiplier": selected_test["fps"] / baseline_test["fps"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    output.mkdir(parents=True)
    for name, payload in (("manifest.json", manifest), ("result.json", result)):
        with open(output / name, "x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{_sha256(output / name)}  {name}\n"
            for name in ("manifest.json", "result.json")
        ),
        encoding="utf-8",
    )
    (output / "DONE").write_text("complete\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "decision": decision,
        "selected_factor": selected_factor,
        "baseline_selection": {
            key: value for key, value in baseline_selection.items()
            if key != "per_view"
        },
        "candidate_selection": {
            str(factor): {
                key: value for key, value in row.items() if key != "per_view"
            }
            for factor, row in candidates.items()
        },
        "baseline_test": {
            key: value for key, value in baseline_test.items() if key != "per_view"
        },
        "selected_test": {
            key: value for key, value in selected_test.items() if key != "per_view"
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
