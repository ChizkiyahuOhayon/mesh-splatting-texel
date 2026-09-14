"""Evaluate one frozen stock or method checkpoint for the main table."""

import json
import subprocess
from argparse import ArgumentParser
from pathlib import Path

import lpips
import torch

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from scene.triangle_model import TriangleModel
from sota.tail_culling import _evaluate
from sota.visibility import check_adaptive_contract
from utils.general_utils import safe_state


SETTINGS = {
    "stock": {
        "opacity_floor": 0.9999,
        "threshold": 1e-4,
        "absorb_tail": False,
        "upsample": 4,
    },
    "ours": {
        "opacity_floor": 0.8,
        "threshold": 1e-2,
        "absorb_tail": True,
        "upsample": 3,
    },
    "ours_speed": {
        "opacity_floor": 0.8,
        "threshold": 1e-2,
        "absorb_tail": True,
        "upsample": 3,
    },
    "ours_quality": {
        "opacity_floor": 0.8,
        "threshold": 1e-2,
        "absorb_tail": True,
        "upsample": 4,
    },
    "ours_opacity": {
        "opacity_floor": 0.8,
        "threshold": 1e-4,
        "absorb_tail": False,
        "upsample": 4,
    },
    "ours_opacity07": {
        "opacity_floor": 0.7,
        "threshold": 1e-4,
        "absorb_tail": False,
        "upsample": 4,
    },
    "ours_opacity09": {
        "opacity_floor": 0.9,
        "threshold": 1e-4,
        "absorb_tail": False,
        "upsample": 4,
    },
    # SoftTail v2: visibility-aware per-vertex terminal opacity. The checkpoint
    # carries its own per-vertex floor (global reference 0.8); the renderer
    # settings mirror the v1 quality, speed, and opacity-only arms exactly.
    "adaptive_quality": {
        "opacity_floor": 0.8,
        "threshold": 1e-2,
        "absorb_tail": True,
        "upsample": 4,
        "adaptive": True,
    },
    "adaptive_speed": {
        "opacity_floor": 0.8,
        "threshold": 1e-2,
        "absorb_tail": True,
        "upsample": 3,
        "adaptive": True,
    },
    "adaptive_opacity": {
        "opacity_floor": 0.8,
        "threshold": 1e-4,
        "absorb_tail": False,
        "upsample": 4,
        "adaptive": True,
    },
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

    setting = dict(SETTINGS[args.arm])
    if args.arm == "ours":
        setting["upsample"] = args.method_upsample
        setting["threshold"] = args.method_threshold
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
    expected_floor = setting["opacity_floor"]
    if args.arm != "stock" and float(triangles.opacity_floor) != expected_floor:
        raise RuntimeError(
            f"method checkpoint opacity_floor is {triangles.opacity_floor}, "
            f"expected {expected_floor}"
        )
    # An adaptive arm needs the per-vertex floor and a v1 arm must not silently
    # render one; both directions are refused loudly.
    check_adaptive_contract(
        torch.load(checkpoint, map_location="cpu", weights_only=False),
        expect_adaptive=bool(setting.get("adaptive", False)),
        expected_high=expected_floor,
    )
    # Garden/Room stock checkpoints predate persistence of this scalar.  Their
    # endpoint is the published opaque baseline used in the matched runs.
    triangles.opacity_floor = setting["opacity_floor"]
    if triangles.texel_order != 0:
        raise RuntimeError("main-table comparison requires SH-only checkpoints")

    test = sorted(scene.getTestCameras(), key=lambda view: view.image_name)
    if not test:
        raise RuntimeError("main-table comparison requires the official test split")
    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )
    pipeline.texel_footprint_filter = False
    lpips_metric = lpips.LPIPS(net="vgg").cuda().eval()
    metrics = _evaluate(
        test,
        triangles,
        pipeline,
        background,
        setting["threshold"],
        setting["absorb_tail"],
        lpips_metric,
        setting["upsample"],
    )
    result = {
        "experiment": "matched-main-table-v0",
        "scene": args.scene,
        "arm": args.arm,
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "model": str(Path(dataset.model_path).resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "triangles": int(triangles.get_triangle_indices.shape[0]),
        "vertices": int(triangles.get_vertices.shape[0]),
        "opacity_floor": setting["opacity_floor"],
        "adaptive_opacity": (
            dict(triangles.adaptive_opacity) | {
                "vertex_floor_min": float(triangles.opacity_floor_vertex.min()),
                "vertex_floor_max": float(triangles.opacity_floor_vertex.max()),
                "vertex_floor_mean": float(triangles.opacity_floor_vertex.mean()),
            } if triangles.opacity_floor_vertex is not None else None
        ),
        "upsample": setting["upsample"],
        "transmittance_threshold": setting["threshold"],
        "absorb_transmittance_tail": setting["absorb_tail"],
        "metrics": metrics,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
    }
    output.mkdir(parents=True)
    with open(output / "result.json", "x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    (output / "DONE").write_text("complete\n", encoding="utf-8")
    print(json.dumps({
        key: value for key, value in result.items()
        if key != "metrics"
    } | {
        "metrics": {
            key: value for key, value in metrics.items() if key != "per_view"
        }
    }, indent=2))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--arm", choices=tuple(SETTINGS), required=True)
    parser.add_argument("--method-upsample", type=int, choices=(1, 2, 3, 4), default=3)
    parser.add_argument("--method-threshold", type=float, default=1e-2)
    parser.add_argument("--quiet", action="store_true")
    parsed = get_combined_args(parser)
    if not parsed.eval:
        parser.error("--eval is required")
    safe_state(parsed.quiet)
    run(model.extract(parsed), pipeline.extract(parsed), parsed)
