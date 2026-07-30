"""Train the preregistered FRT-G1 texel residual on a frozen final SH checkpoint."""

import copy
import json
import random
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from frt_freeze import assert_base_unchanged, freeze_base_tensors
from scene import Scene
from scene.triangle_model import TriangleModel
from svsr_g1_eval import _checkpoint, _sha256
from svsr_metadata import reserve_output_directory, source_revision
from triangle_renderer import render
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim


LOCKED_STEPS = 5_000
SMOKE_STEPS = 2
TEXEL_ORDER = 2
TEXEL_LR = 0.0025
LAMBDA_DSSIM = 0.2


def _load_base(dataset, iteration):
    scene_args = copy.copy(dataset)
    triangles = TriangleModel(scene_args.sh_degree)
    triangles.scaling = 4
    scene = Scene(
        args=scene_args,
        triangles=triangles,
        init_opacity=None,
        set_sigma=None,
        load_iteration=iteration,
        shuffle=True,
    )
    return scene, triangles


def run(dataset, pipeline, args):
    base_model = str(Path(dataset.model_path).resolve())
    base_iteration, base_checkpoint = _checkpoint(base_model, args.iteration)
    scene, triangles = _load_base(dataset, base_iteration)
    if triangles.texel_order != 0:
        raise RuntimeError("FRT-G1 base checkpoint must be SH-only.")
    if not scene.getTrainCameras() or not scene.getTestCameras():
        raise RuntimeError("FRT-G1 requires non-empty train and held-out splits.")

    output_root = reserve_output_directory(Path(args.frt_output).resolve())
    steps = SMOKE_STEPS if args.frt_smoke else LOCKED_STEPS
    train_views = sorted(scene.getTrainCameras(), key=lambda view: view.image_name)
    test_views = sorted(scene.getTestCameras(), key=lambda view: view.image_name)
    manifest = {
        "protocol": "experiments/frt_g1/protocol.md",
        "scene": args.frt_scene,
        "dataset_path": str(Path(dataset.source_path).resolve()),
        "base_model": base_model,
        "base_iteration": base_iteration,
        "base_checkpoint_sha256": _sha256(base_checkpoint),
        "output_model": str(output_root),
        "train_views": [view.image_name for view in train_views],
        "test_views": [view.image_name for view in test_views],
        "image_sizes": sorted({
            (int(view.image_height), int(view.image_width))
            for view in train_views + test_views
        }),
        "vertices": int(triangles.vertices.shape[0]),
        "faces": int(triangles._triangle_indices.shape[0]),
        "texel_order": TEXEL_ORDER,
        "texel_lr": TEXEL_LR,
        "updates": steps,
        "lambda_dssim": LAMBDA_DSSIM,
        "sampling_seed": 0,
        "scaling": 4,
        "smoke": args.frt_smoke,
        "confirmatory_settings": not args.frt_smoke,
        "git_commit": source_revision(),
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        "torch": torch.__version__,
    }
    with open(output_root / "frt_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    saved_args = vars(args).copy()
    saved_args["model_path"] = str(output_root)
    saved_args["source_path"] = str(Path(dataset.source_path).resolve())
    with open(output_root / "cfg_args", "w", encoding="utf-8") as handle:
        handle.write(str(Namespace(**saved_args)))

    background_values = [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0]
    background = torch.tensor(background_values, dtype=torch.float32, device="cuda")
    pipeline.texel_footprint_filter = False
    base_fingerprint = freeze_base_tensors(triangles)
    triangles.optimizer = None

    parity_view = test_views[0]
    with torch.no_grad():
        base_render = render(parity_view, triangles, pipeline, background)["render"]
    triangles.create_texels(TEXEL_ORDER, TEXEL_LR)
    with torch.no_grad():
        zero_render = render(parity_view, triangles, pipeline, background)["render"]
    zero_init_max_abs = float((base_render - zero_render).abs().max())
    if zero_init_max_abs > 1e-7:
        raise RuntimeError(
            f"zero-initialized texels changed the base render by {zero_init_max_abs}"
        )
    del base_render, zero_render

    random.seed(0)
    viewpoint_stack = []
    losses = []
    started = time.perf_counter()
    progress = tqdm(range(1, steps + 1), desc=f"FRT-G1 {args.frt_scene}")
    for step in progress:
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        view = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack) - 1))
        prediction = render(view, triangles, pipeline, background)["render"]
        target = view.original_image.cuda()
        pixel_l1 = l1_loss(prediction, target)
        loss = ((1.0 - LAMBDA_DSSIM) * pixel_l1
                + LAMBDA_DSSIM * (1.0 - ssim(prediction, target)))
        loss.backward()
        triangles.texel_optimizer.step()
        triangles.texel_optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % 100 == 0 or step == steps:
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "l1": float(pixel_l1.detach()),
            }
            losses.append(row)
            progress.set_postfix(loss=f"{row['loss']:.5f}")

    assert_base_unchanged(triangles, base_fingerprint)
    scene.model_path = str(output_root)
    checkpoint = scene.save(steps)
    integrity = {
        "zero_init_max_abs": zero_init_max_abs,
        "base_tensors_unchanged": True,
        "optimizer_parameter_groups": [
            group.get("name") for group in triangles.texel_optimizer.param_groups
        ],
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "checkpoint": checkpoint,
        "checkpoint_sha256": _sha256(checkpoint),
        "loss_trace": losses,
    }
    with open(output_root / "training.json", "w", encoding="utf-8") as handle:
        json.dump(integrity, handle, indent=2)
    (output_root / "DONE").touch()
    print(json.dumps({"output": str(output_root), **integrity}, indent=2))


if __name__ == "__main__":
    parser = ArgumentParser(description="FRT-G1 frozen-base residual training")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--frt_scene", required=True, choices=("garden", "room"))
    parser.add_argument("--frt_output", required=True)
    parser.add_argument("--frt_smoke", action="store_true")
    parsed = get_combined_args(parser)
    if not parsed.model_path or not parsed.source_path:
        parser.error("Both --model_path/-m and --source_path/-s are required.")
    if not parsed.eval:
        parser.error("FRT-G1 requires --eval so the held-out split is fixed.")
    safe_state(parsed.quiet)
    run(model.extract(parsed), pipeline.extract(parsed), parsed)
