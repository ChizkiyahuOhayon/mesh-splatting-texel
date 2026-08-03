"""RITS-D0: decompose the midpoint-split render discontinuity by mechanism."""

import json
import math
import time
from argparse import ArgumentParser
from pathlib import Path

import torch

from arguments import ModelParams, PipelineParams, get_combined_args
from csu_f0_eval import _dominant_parent_mask, _select_evenly
from rits_prolongation import install_prolongation_probe
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
REQUIRED_REGION_REDUCTION = 0.80
VARIANT1_TOLERANCE = 0.05

# CSU-F0 confirmatory parity (experiments/csu_f0/analysis_full_01.md): variant 1
# re-runs that gate's split verbatim and must reproduce these numbers.
CSU_F0_REFERENCE = {
    "garden": {"global_mae": 1.10525e-4, "probe_region_mae": 4.20346e-3},
    "room": {"global_mae": 5.48668e-5, "probe_region_mae": 1.82458e-3},
}

# Variant -> (donor_window, donor_opacity), as preregistered in
# experiments/rits_d0/protocol.md.
VARIANTS = {
    "v1_inherited": (False, False),
    "v2_parent_window": (True, False),
    "v3_parent_opacity": (False, True),
    "v4_prolongation": (True, True),
}

MODEL_STATE = (
    "vertices",
    "vertex_weight",
    "_features_dc",
    "_features_rest",
    "_triangle_indices",
    "image_size",
    "importance_score",
    "pixel_count",
)


def _snapshot(model):
    return {name: getattr(model, name) for name in MODEL_STATE}


def _restore(model, state):
    for name, value in state.items():
        setattr(model, name, value)


def _percentile_99(errors):
    flat = torch.cat([error.flatten() for error in errors])
    k = max(1, math.ceil(0.99 * flat.numel()))
    return float(flat.kthvalue(k).values)


def _evaluate_variant(model, pipeline, background, baseline_frames, probe):
    donors = probe["window_donors"]
    child_start = probe["child_face_start"]
    per_view = []
    error_tensors = []
    global_error_sum = 0.0
    global_pixels = 0
    region_error_sum = 0.0
    region_pixels = 0
    with torch.no_grad():
        for view, baseline, region_mask, _ in baseline_frames:
            package = render(view, model, pipeline, background, window_donors=donors)
            split = package["render"].detach().cpu()
            error = (split - baseline).abs()
            pixel_error = error.mean(dim=0)
            children = package["radii"][child_start:]
            view_region_pixels = int(region_mask.sum())
            per_view.append(
                {
                    "view": view.image_name,
                    "global_mae": float(pixel_error.mean()),
                    "probe_region_pixels": view_region_pixels,
                    "probe_region_mae": (
                        float(pixel_error[region_mask].mean()) if view_region_pixels else None
                    ),
                    "p99_abs_error": _percentile_99([error]),
                    "max_abs_error": float(error.max()),
                    "children_total": int(children.numel()),
                    "children_rendered": int((children > 0).sum()),
                }
            )
            error_tensors.append(error)
            global_error_sum += float(pixel_error.sum())
            global_pixels += pixel_error.numel()
            if view_region_pixels:
                region_error_sum += float(pixel_error[region_mask].sum())
                region_pixels += view_region_pixels
    return {
        "global_mae": global_error_sum / global_pixels,
        "probe_region_pixels": region_pixels,
        "probe_region_mae": (
            region_error_sum / region_pixels if region_pixels else None
        ),
        "p99_abs_error": _percentile_99(error_tensors),
        "max_abs_error": max(row["max_abs_error"] for row in per_view),
        "per_view": per_view,
    }


def _within(value, reference, tolerance):
    return abs(value - reference) <= tolerance * reference


def _decide(scene, variants, region_pixels):
    reference = CSU_F0_REFERENCE[scene]
    v1, v2, v3, v4 = (
        variants["v1_inherited"],
        variants["v2_parent_window"],
        variants["v3_parent_opacity"],
        variants["v4_prolongation"],
    )
    if any(row["probe_region_mae"] is None for row in (v1, v2, v3, v4)):
        return {
            "scene_pass": False,
            "checks": {"probe_region_pixels_at_least_100": False},
        }
    checks = {
        "probe_region_pixels_at_least_100": region_pixels >= MIN_REGION_PIXELS,
        "variant1_reproduces_csu_f0": (
            _within(v1["global_mae"], reference["global_mae"], VARIANT1_TOLERANCE)
            and _within(
                v1["probe_region_mae"], reference["probe_region_mae"], VARIANT1_TOLERANCE
            )
        ),
        "variant4_global_mae_at_most_1e_4": v4["global_mae"] <= GLOBAL_MAE_LIMIT,
        "variant4_probe_region_mae_at_most_2e_3": v4["probe_region_mae"] <= REGION_MAE_LIMIT,
        "variant4_reduces_probe_mae_by_80_percent": (
            v4["probe_region_mae"]
            <= (1.0 - REQUIRED_REGION_REDUCTION) * v1["probe_region_mae"]
        ),
        "variant4_beats_variant2_and_variant3": (
            v4["probe_region_mae"] < v2["probe_region_mae"]
            and v4["probe_region_mae"] < v3["probe_region_mae"]
        ),
    }
    return {"scene_pass": all(checks.values()), "checks": checks}


def run(dataset, pipeline, args):
    output_root = reserve_output_directory(Path(args.rits_output).resolve())
    iteration, checkpoint = _checkpoint(dataset.model_path, args.iteration)
    train_count = SMOKE_TRAIN_VIEWS if args.rits_smoke else LOCKED_TRAIN_VIEWS
    test_count = SMOKE_TEST_VIEWS if args.rits_smoke else LOCKED_TEST_VIEWS
    face_limit = SMOKE_FACE_COUNT if args.rits_smoke else LOCKED_FACE_COUNT

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
        raise RuntimeError("RITS-D0 requires an unmodified SH-only checkpoint.")
    train_views = _select_evenly(scene.getTrainCameras(), train_count)
    test_views = _select_evenly(scene.getTestCameras(), test_count)
    pipeline.texel_footprint_filter = False
    background_values = [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0]
    background = torch.tensor(background_values, dtype=torch.float32, device="cuda")

    manifest = {
        "protocol": "experiments/rits_d0/protocol.md",
        "scene": args.rits_scene,
        "dataset_path": str(Path(dataset.source_path).resolve()),
        "model_path": str(Path(dataset.model_path).resolve()),
        "iteration": iteration,
        "checkpoint_sha256": _sha256(checkpoint),
        "source_revision": source_revision(),
        "train_views": [view.image_name for view in train_views],
        "test_views": [view.image_name for view in test_views],
        "candidate_screen": "summed_projected_pixel_coverage",
        "selected_parent_faces": face_limit,
        "variants": {name: {"donor_window": w, "donor_opacity": o} for name, (w, o) in VARIANTS.items()},
        "smoke": args.rits_smoke,
        "confirmatory_settings": not args.rits_smoke,
        "limits": {
            "global_mae": GLOBAL_MAE_LIMIT,
            "probe_region_mae": REGION_MAE_LIMIT,
            "probe_region_pixels": MIN_REGION_PIXELS,
            "required_region_reduction": REQUIRED_REGION_REDUCTION,
            "variant1_relative_tolerance": VARIANT1_TOLERANCE,
        },
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        "torch": torch.__version__,
    }
    with open(output_root / "rits_manifest.json", "w", encoding="utf-8") as handle:
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
            mask = _dominant_parent_mask(package, selected_faces, prediction.shape[-2:])
            parents_rendered = int((package["radii"][selected_faces] > 0).sum())
            baseline_frames.append(
                (view, prediction.cpu(), mask.cpu(), parents_rendered)
            )

    base_state = _snapshot(triangles)
    variants = {}
    integrity = {}
    for name, (donor_window, donor_opacity) in VARIANTS.items():
        probe = install_prolongation_probe(
            triangles, selected_faces, donor_window, donor_opacity
        )
        integrity[name] = {
            "topology_counts_exact": probe["topology_valid"],
            "original_parameter_prefix_bitwise_unchanged": probe["prefix_unchanged"],
        }
        variants[name] = _evaluate_variant(
            triangles, pipeline, background, baseline_frames, probe
        )
        _restore(triangles, base_state)

    region_pixels = variants["v1_inherited"]["probe_region_pixels"]
    decision = (
        {"pass": None, "reason": "implementation smoke; no gate decision"}
        if args.rits_smoke
        else _decide(args.rits_scene, variants, region_pixels)
    )
    torch.cuda.synchronize()
    results = {
        "scene": args.rits_scene,
        "visible_candidate_faces": visible_faces,
        "selected_parent_faces": face_limit,
        "baseline_parents_rendered_per_view": {
            view.image_name: parents for view, _, _, parents in baseline_frames
        },
        "integrity": integrity,
        "variants": variants,
        "decision": decision,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    with open(output_root / "results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, allow_nan=False)
    (output_root / "DONE").write_text("complete\n", encoding="utf-8")
    print(json.dumps({"results": str(output_root / "results.json"), **results}, indent=2))


if __name__ == "__main__":
    parser = ArgumentParser(description="RITS-D0 source-of-discontinuity diagnostic")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--rits_scene", required=True, choices=("garden", "room"))
    parser.add_argument("--rits_output", required=True)
    parser.add_argument("--rits_smoke", action="store_true")
    parsed = get_combined_args(parser)
    if not parsed.model_path or not parsed.source_path:
        parser.error("Both --model_path/-m and --source_path/-s are required.")
    if not parsed.eval:
        parser.error("RITS-D0 requires --eval so held-out views are loaded.")
    safe_state(parsed.quiet)
    run(model.extract(parsed), pipeline.extract(parsed), parsed)
