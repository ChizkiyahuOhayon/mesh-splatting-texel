"""Evaluate whether cross-view residuals can guide fixed-budget mesh refinement."""

import hashlib
import json
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn.functional as F

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from scene.triangle_model import TriangleModel
from svsr_metadata import reserve_output_directory, source_revision
from triangle_renderer import render
from utils.general_utils import safe_state
from utils.system_utils import searchForMaxIteration
from xvr_score import persistent_error_mass, scene_gate, top_fraction_capture


FRACTIONS = (("top_1pct", 0.01), ("top_5pct", 0.05), ("top_10pct", 0.10))


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_evenly(views, maximum):
    views = sorted(views, key=lambda view: view.image_name)
    if maximum == 0 or maximum >= len(views):
        return views
    if maximum < 1:
        raise ValueError("maximum must be non-negative")
    if maximum == 1:
        return [views[len(views) // 2]]
    indices = torch.linspace(0, len(views) - 1, maximum).round().long().tolist()
    return [views[index] for index in indices]


def _empty_stats(face_count, device):
    return {
        "min_view_error": torch.full((face_count,), float("inf"), device=device),
        "sum_view_error": torch.zeros(face_count, device=device),
        "view_count": torch.zeros(face_count, device=device),
        "pixel_error_sum": torch.zeros(face_count, device=device),
        "pixel_count": torch.zeros(face_count, device=device),
        "max_blending": torch.zeros(face_count, device=device),
    }


def _accumulate(stats, package, prediction, target, alpha_threshold):
    face_count = stats["view_count"].numel()
    if package["max_blending"].numel() != face_count:
        raise RuntimeError("renderer max_blending is not aligned with the face set")
    height, width = target.shape[-2:]
    face_ids = F.interpolate(
        package["rend_ids"].unsqueeze(0), size=(height, width), mode="nearest"
    ).squeeze(0).squeeze(0)
    alpha = package["rend_alpha"].squeeze(0)
    covered = (face_ids >= 0) & (face_ids < face_count) & (alpha > alpha_threshold)
    safe_ids = face_ids.long().clamp_(0, face_count - 1)
    residual = (prediction - target).abs().mean(0)

    covered_ids = safe_ids[covered].reshape(-1)
    covered_residual = residual[covered].reshape(-1)
    view_error_sum = torch.zeros(face_count, device=residual.device).scatter_add_(
        0, covered_ids, covered_residual
    )
    view_pixel_count = torch.zeros(face_count, device=residual.device).scatter_add_(
        0, covered_ids, torch.ones_like(covered_residual)
    )
    seen = view_pixel_count > 0
    view_mean = view_error_sum / view_pixel_count.clamp_min(1)

    stats["min_view_error"] = torch.minimum(
        stats["min_view_error"],
        torch.where(seen, view_mean, torch.full_like(view_mean, float("inf"))),
    )
    stats["sum_view_error"] += torch.where(seen, view_mean, torch.zeros_like(view_mean))
    stats["view_count"] += seen
    stats["pixel_error_sum"] += view_error_sum
    stats["pixel_count"] += view_pixel_count
    stats["max_blending"] = torch.maximum(stats["max_blending"], package["max_blending"])


def _score(train_stats, test_stats, world_area, minimum_views):
    eligible = (
        (train_stats["view_count"] >= minimum_views)
        & (test_stats["pixel_count"] > 0)
        & torch.isfinite(train_stats["min_view_error"])
    )
    mean_visible_pixels = train_stats["pixel_count"] / train_stats["view_count"].clamp_min(1)
    raw_mean = train_stats["sum_view_error"] / train_stats["view_count"].clamp_min(1)
    scores = {
        "persistent_error_mass": persistent_error_mass(
            train_stats["min_view_error"], train_stats["pixel_count"], train_stats["view_count"]
        ),
        "raw_error_mass": raw_mean * mean_visible_pixels,
        "max_blending": train_stats["max_blending"],
        "projected_coverage": mean_visible_pixels,
        "world_area": world_area,
    }

    target_mass = test_stats["pixel_error_sum"]
    signal_metrics = {
        name: {
            label: top_fraction_capture(value, target_mass, eligible, fraction)
            for label, fraction in FRACTIONS
        }
        for name, value in scores.items()
    }
    return eligible, signal_metrics


def run(dataset, pipeline, args):
    output_root = reserve_output_directory(Path(args.xvr_output).resolve())
    point_cloud_root = Path(dataset.model_path) / "point_cloud"
    iteration = args.iteration
    if iteration < 0:
        iteration = searchForMaxIteration(str(point_cloud_root))
    checkpoint = point_cloud_root / f"iteration_{iteration}" / "point_cloud_state_dict.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

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
        raise RuntimeError("XVR-G0 requires an unmodified SH-only baseline checkpoint.")

    train_views = _select_evenly(scene.getTrainCameras(), args.xvr_max_train_views)
    test_views = _select_evenly(scene.getTestCameras(), args.xvr_max_test_views)
    if not train_views or not test_views:
        raise RuntimeError("XVR-G0 requires non-empty train and held-out test splits.")

    confirmatory = args.xvr_max_train_views == 16 and args.xvr_max_test_views == 0
    minimum_views = 3 if len(train_views) >= 3 else len(train_views)
    manifest = {
        "protocol": "experiments/xvr_g0/protocol.md",
        "scene": args.xvr_scene,
        "dataset_path": str(Path(dataset.source_path).resolve()),
        "model_path": str(Path(dataset.model_path).resolve()),
        "iteration": iteration,
        "checkpoint_sha256": _sha256(checkpoint),
        "source_revision": source_revision(),
        "train_views": [view.image_name for view in train_views],
        "test_views": [view.image_name for view in test_views],
        "minimum_train_views_per_face": minimum_views,
        "alpha_threshold": 0.5,
        "confirmatory_settings": confirmatory,
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        "torch": torch.__version__,
    }
    with open(output_root / "xvr_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    pipeline.texel_footprint_filter = False
    background_values = [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0]
    background = torch.tensor(background_values, dtype=torch.float32, device="cuda")
    face_count = triangles._triangle_indices.shape[0]
    train_stats = _empty_stats(face_count, triangles.vertices.device)
    test_stats = _empty_stats(face_count, triangles.vertices.device)

    with torch.no_grad():
        for views, stats in ((train_views, train_stats), (test_views, test_stats)):
            for view in views:
                package = render(view, triangles, pipeline, background)
                prediction = package["render"].clamp(0.0, 1.0)
                target = view.original_image[:3].to(prediction.device)
                _accumulate(stats, package, prediction, target, alpha_threshold=0.5)

        eligible, signal_metrics = _score(
            train_stats, test_stats, triangles.triangle_areas(), minimum_views
        )

    decision = scene_gate(signal_metrics) if confirmatory else {
        "pass": None,
        "reason": "exploratory settings; no gate decision",
    }
    results = {
        "scene": args.xvr_scene,
        "face_count": int(face_count),
        "eligible_faces": int(eligible.sum()),
        "signals": signal_metrics,
        "decision": decision,
    }
    with open(output_root / "results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    (output_root / "DONE").write_text("complete\n", encoding="utf-8")
    print(json.dumps({"results": str(output_root / "results.json"), **results}, indent=2))


if __name__ == "__main__":
    parser = ArgumentParser(description="XVR-G0 cross-view refinement diagnostic")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--xvr_scene", required=True, choices=("garden", "room"))
    parser.add_argument("--xvr_output", required=True)
    parser.add_argument("--xvr_max_train_views", default=16, type=int)
    parser.add_argument("--xvr_max_test_views", default=0, type=int)
    parsed = get_combined_args(parser)
    if not parsed.model_path or not parsed.source_path:
        parser.error("Both --model_path/-m and --source_path/-s are required.")
    if not parsed.eval:
        parser.error("XVR-G0 requires --eval so held-out views are loaded.")
    if parsed.xvr_max_train_views < 1 or parsed.xvr_max_test_views < 0:
        parser.error("train max must be positive and test max must be non-negative.")
    safe_state(parsed.quiet)
    run(model.extract(parsed), pipeline.extract(parsed), parsed)
