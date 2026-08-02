"""Probe whether inherited midpoint subdivision is a valid CSU parameter expansion."""

import json
import time
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn.functional as F

from arguments import ModelParams, PipelineParams, get_combined_args
from csu_split import install_midpoint_probe
from scene import Scene
from scene.triangle_model import TriangleModel
from svsr_g1_eval import _checkpoint, _sha256
from svsr_metadata import reserve_output_directory, source_revision
from triangle_renderer import render
from utils.general_utils import safe_state


LOCKED_TRAIN_VIEWS = 4
LOCKED_TEST_VIEWS = 4
LOCKED_FACE_COUNT = 512
SMOKE_TRAIN_VIEWS = 1
SMOKE_TEST_VIEWS = 1
SMOKE_FACE_COUNT = 32
GLOBAL_MAE_LIMIT = 1e-4
REGION_MAE_LIMIT = 2e-3
MIN_REGION_PIXELS = 100


def _select_evenly(views, count):
    ordered = sorted(views, key=lambda view: view.image_name)
    if len(ordered) < count:
        raise RuntimeError(f"requested {count} views, but only {len(ordered)} are available")
    if count == 1:
        return [ordered[len(ordered) // 2]]
    indices = torch.linspace(0, len(ordered) - 1, count).round().long().tolist()
    return [ordered[index] for index in indices]


def _dominant_parent_mask(package, selected_faces, image_shape):
    face_ids = F.interpolate(
        package["rend_ids"].unsqueeze(0), size=image_shape, mode="nearest"
    ).squeeze(0).squeeze(0).long()
    return torch.isin(face_ids, selected_faces)


def _gradient_statistics(gradient):
    if gradient is None:
        return {
            "finite": False,
            "l2": 0.0,
            "max_abs": 0.0,
            "nonzero_fraction": 0.0,
        }
    finite = bool(torch.isfinite(gradient).all())
    if not finite:
        return {
            "finite": False,
            "l2": None,
            "max_abs": None,
            "nonzero_fraction": None,
        }
    absolute = gradient.detach().abs()
    return {
        "finite": finite,
        "l2": float(torch.linalg.vector_norm(gradient.detach().float())),
        "max_abs": float(absolute.max()),
        "nonzero_fraction": float((absolute > 0).float().mean()),
    }


def run(dataset, pipeline, args):
    output_root = reserve_output_directory(Path(args.csu_output).resolve())
    iteration, checkpoint = _checkpoint(dataset.model_path, args.iteration)
    train_count = SMOKE_TRAIN_VIEWS if args.csu_smoke else LOCKED_TRAIN_VIEWS
    test_count = SMOKE_TEST_VIEWS if args.csu_smoke else LOCKED_TEST_VIEWS
    face_limit = SMOKE_FACE_COUNT if args.csu_smoke else LOCKED_FACE_COUNT

    triangles = TriangleModel(dataset.sh_degree)
    triangles.scaling = 4
    scene = Scene(
        args=dataset,
        triangles=triangles,
        init_opacity=None,
        set_sigma=None,
        load_iteration=iteration,
        shuffle=False,
    )
    if triangles.texel_order != 0:
        raise RuntimeError("CSU-F0 requires an unmodified SH-only checkpoint.")
    train_views = _select_evenly(scene.getTrainCameras(), train_count)
    test_views = _select_evenly(scene.getTestCameras(), test_count)
    pipeline.texel_footprint_filter = False
    background_values = [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0]
    background = torch.tensor(background_values, dtype=torch.float32, device="cuda")

    manifest = {
        "protocol": "experiments/csu_f0/protocol.md",
        "scene": args.csu_scene,
        "dataset_path": str(Path(dataset.source_path).resolve()),
        "model_path": str(Path(dataset.model_path).resolve()),
        "iteration": iteration,
        "checkpoint_sha256": _sha256(checkpoint),
        "source_revision": source_revision(),
        "train_views": [view.image_name for view in train_views],
        "test_views": [view.image_name for view in test_views],
        "candidate_screen": "summed_projected_pixel_coverage",
        "selected_parent_faces": face_limit,
        "smoke": args.csu_smoke,
        "confirmatory_settings": not args.csu_smoke,
        "limits": {
            "global_mae": GLOBAL_MAE_LIMIT,
            "probe_region_mae": REGION_MAE_LIMIT,
            "probe_region_pixels": MIN_REGION_PIXELS,
        },
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        "torch": torch.__version__,
    }
    with open(output_root / "csu_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    face_count = triangles._triangle_indices.shape[0]
    coverage = torch.zeros(face_count, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        for view in train_views:
            package = render(view, triangles, pipeline, background)
            coverage += package["triangle_was_rendered"].float()
    visible_faces = int((coverage > 0).sum())
    if visible_faces < face_limit:
        raise RuntimeError(
            f"only {visible_faces} faces are visible, fewer than the locked {face_limit}"
        )
    selected_faces = torch.topk(coverage, face_limit, sorted=False).indices

    baseline_frames = []
    with torch.no_grad():
        for view in test_views:
            package = render(view, triangles, pipeline, background)
            prediction = package["render"].detach()
            mask = _dominant_parent_mask(
                package, selected_faces, prediction.shape[-2:]
            )
            baseline_frames.append((view, prediction.cpu(), mask.cpu()))

    probe = install_midpoint_probe(triangles, selected_faces)
    global_error_sum = 0.0
    global_pixels = 0
    region_error_sum = 0.0
    region_pixels = 0
    per_view = []
    with torch.no_grad():
        for view, baseline, region_mask in baseline_frames:
            split = render(view, triangles, pipeline, background)["render"].detach().cpu()
            pixel_error = (split - baseline).abs().mean(dim=0)
            view_region_pixels = int(region_mask.sum())
            view_region_mae = (
                float(pixel_error[region_mask].mean()) if view_region_pixels else None
            )
            per_view.append(
                {
                    "view": view.image_name,
                    "global_mae": float(pixel_error.mean()),
                    "max_abs": float((split - baseline).abs().max()),
                    "probe_region_pixels": view_region_pixels,
                    "probe_region_mae": view_region_mae,
                }
            )
            global_error_sum += float(pixel_error.sum())
            global_pixels += pixel_error.numel()
            if view_region_pixels:
                region_error_sum += float(pixel_error[region_mask].sum())
                region_pixels += view_region_pixels

    global_mae = global_error_sum / global_pixels
    region_mae = region_error_sum / region_pixels if region_pixels else None
    gradient_view = train_views[0]
    prediction = render(gradient_view, triangles, pipeline, background)["render"]
    target = gradient_view.original_image[:3].to(prediction.device)
    gradient_loss = (prediction - target).abs().mean()
    parameter_names = list(probe["parameters"])
    gradients = torch.autograd.grad(
        gradient_loss,
        tuple(probe["parameters"][name] for name in parameter_names),
        allow_unused=True,
    )
    gradient_stats = {
        name: _gradient_statistics(gradient)
        for name, gradient in zip(parameter_names, gradients)
    }
    gradients_finite = all(row["finite"] for row in gradient_stats.values())
    appearance_l2 = (
        (
            gradient_stats["features_dc"]["l2"] ** 2
            + gradient_stats["features_rest"]["l2"] ** 2
        ) ** 0.5
        if gradients_finite
        else None
    )

    checks = {
        "topology_counts_exact": probe["topology_valid"],
        "original_parameter_prefix_bitwise_unchanged": probe["prefix_unchanged"],
        "probe_region_pixels_at_least_100": region_pixels >= MIN_REGION_PIXELS,
        "global_mae_at_most_1e_4": global_mae <= GLOBAL_MAE_LIMIT,
        "probe_region_mae_at_most_2e_3": (
            region_mae is not None and region_mae <= REGION_MAE_LIMIT
        ),
        "all_new_gradients_finite": gradients_finite,
        "new_geometry_gradient_nonzero": (
            gradients_finite and gradient_stats["vertices"]["l2"] > 0.0
        ),
        "new_appearance_gradient_nonzero": (
            gradients_finite and appearance_l2 > 0.0
        ),
    }
    decision = (
        {"pass": None, "reason": "implementation smoke; no gate decision"}
        if args.csu_smoke
        else {"pass": all(checks.values()), "checks": checks}
    )
    torch.cuda.synchronize()
    results = {
        "scene": args.csu_scene,
        "base_vertices": probe["base_vertex_count"],
        "base_faces": probe["base_face_count"],
        "selected_parent_faces": int(probe["selected_faces"].numel()),
        "unique_midpoint_vertices": probe["unique_edge_count"],
        "split_vertices": probe["split_vertex_count"],
        "split_faces": probe["split_face_count"],
        "visible_candidate_faces": visible_faces,
        "parity": {
            "global_mae": global_mae,
            "probe_region_pixels": region_pixels,
            "probe_region_mae": region_mae,
            "per_view": per_view,
        },
        "gradient_view": gradient_view.image_name,
        "gradient_loss": float(gradient_loss.detach()),
        "new_parameter_gradients": gradient_stats,
        "combined_appearance_gradient_l2": appearance_l2,
        "integrity_checks": checks,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "decision": decision,
    }
    with open(output_root / "results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, allow_nan=False)
    (output_root / "DONE").write_text("complete\n", encoding="utf-8")
    print(json.dumps({"results": str(output_root / "results.json"), **results}, indent=2))


if __name__ == "__main__":
    parser = ArgumentParser(description="CSU-F0 midpoint-split feasibility probe")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--csu_scene", required=True, choices=("garden", "room"))
    parser.add_argument("--csu_output", required=True)
    parser.add_argument("--csu_smoke", action="store_true")
    parsed = get_combined_args(parser)
    if not parsed.model_path or not parsed.source_path:
        parser.error("Both --model_path/-m and --source_path/-s are required.")
    if not parsed.eval:
        parser.error("CSU-F0 requires --eval so held-out views are loaded.")
    safe_state(parsed.quiet)
    run(model.extract(parsed), pipeline.extract(parsed), parsed)
