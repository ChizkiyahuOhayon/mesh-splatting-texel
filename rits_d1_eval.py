"""RITS-D1: does completing the prolongation with parent-domain appearance
restore refinement invariance on both scenes?"""

import json
import time
from argparse import ArgumentParser
from pathlib import Path

import torch

from arguments import ModelParams, PipelineParams, get_combined_args
from csu_f0_eval import _dominant_parent_mask, _select_evenly
from rits_d0_eval import (
    CSU_F0_REFERENCE,
    GLOBAL_MAE_LIMIT,
    LOCKED_FACE_COUNT,
    LOCKED_TEST_VIEWS,
    LOCKED_TRAIN_VIEWS,
    MIN_REGION_PIXELS,
    REGION_MAE_LIMIT,
    REQUIRED_REGION_REDUCTION,
    SMOKE_FACE_COUNT,
    SMOKE_TEST_VIEWS,
    SMOKE_TRAIN_VIEWS,
    _evaluate_variant,
    _restore,
    _snapshot,
    _within,
)
from rits_prolongation import install_prolongation_probe
from scene import Scene
from scene.triangle_model import TriangleModel
from svsr_g1_eval import _checkpoint, _sha256
from svsr_metadata import reserve_output_directory, source_revision
from triangle_renderer import render
from utils.general_utils import safe_state


ANCHOR_TOLERANCE = 0.05

# RITS-D0 confirmatory variant-4 parity (experiments/rits_d0/analysis_full_01.md):
# v4 re-runs that configuration verbatim and must reproduce these numbers.
RITS_D0_V4_REFERENCE = {
    "garden": {"global_mae": 7.3331611727195356e-6, "probe_region_mae": 3.7067671503988273e-4},
    "room": {"global_mae": 3.214550297183123e-5, "probe_region_mae": 1.2178946797153183e-3},
}

# Variant -> (donor_window, donor_opacity, donor_appearance), as preregistered
# in experiments/rits_d1/protocol.md.
VARIANTS = {
    "v1_inherited": (False, False, False),
    "v4_prolongation": (True, True, False),
    "v5_full_prolongation": (True, True, True),
}


def _decide(scene, variants, region_pixels):
    v1, v4, v5 = (
        variants["v1_inherited"],
        variants["v4_prolongation"],
        variants["v5_full_prolongation"],
    )
    if any(row["probe_region_mae"] is None for row in (v1, v4, v5)):
        return {
            "scene_pass": False,
            "checks": {"probe_region_pixels_at_least_100": False},
        }
    csu = CSU_F0_REFERENCE[scene]
    d0v4 = RITS_D0_V4_REFERENCE[scene]
    checks = {
        "probe_region_pixels_at_least_100": region_pixels >= MIN_REGION_PIXELS,
        "variant1_reproduces_csu_f0": (
            _within(v1["global_mae"], csu["global_mae"], ANCHOR_TOLERANCE)
            and _within(v1["probe_region_mae"], csu["probe_region_mae"], ANCHOR_TOLERANCE)
        ),
        "variant4_reproduces_rits_d0": (
            _within(v4["global_mae"], d0v4["global_mae"], ANCHOR_TOLERANCE)
            and _within(v4["probe_region_mae"], d0v4["probe_region_mae"], ANCHOR_TOLERANCE)
        ),
        "variant5_global_mae_at_most_1e_4": v5["global_mae"] <= GLOBAL_MAE_LIMIT,
        "variant5_probe_region_mae_at_most_2e_3": v5["probe_region_mae"] <= REGION_MAE_LIMIT,
        "variant5_reduces_probe_mae_by_80_percent": (
            v5["probe_region_mae"]
            <= (1.0 - REQUIRED_REGION_REDUCTION) * v1["probe_region_mae"]
        ),
        "variant5_beats_variant4": v5["probe_region_mae"] < v4["probe_region_mae"],
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
        raise RuntimeError("RITS-D1 requires an unmodified SH-only checkpoint.")
    train_views = _select_evenly(scene.getTrainCameras(), train_count)
    test_views = _select_evenly(scene.getTestCameras(), test_count)
    pipeline.texel_footprint_filter = False
    background_values = [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0]
    background = torch.tensor(background_values, dtype=torch.float32, device="cuda")

    manifest = {
        "protocol": "experiments/rits_d1/protocol.md",
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
        "variants": {
            name: {"donor_window": w, "donor_opacity": o, "donor_appearance": a}
            for name, (w, o, a) in VARIANTS.items()
        },
        "smoke": args.rits_smoke,
        "confirmatory_settings": not args.rits_smoke,
        "limits": {
            "global_mae": GLOBAL_MAE_LIMIT,
            "probe_region_mae": REGION_MAE_LIMIT,
            "probe_region_pixels": MIN_REGION_PIXELS,
            "required_region_reduction": REQUIRED_REGION_REDUCTION,
            "anchor_relative_tolerance": ANCHOR_TOLERANCE,
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
            baseline_frames.append((view, prediction.cpu(), mask.cpu(), parents_rendered))

    base_state = _snapshot(triangles)
    variants = {}
    integrity = {}
    for name, (donor_window, donor_opacity, donor_appearance) in VARIANTS.items():
        probe = install_prolongation_probe(
            triangles, selected_faces, donor_window, donor_opacity, donor_appearance
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
    parser = ArgumentParser(description="RITS-D1 completed-prolongation gate")
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
        parser.error("RITS-D1 requires --eval so held-out views are loaded.")
    safe_state(parsed.quiet)
    run(model.extract(parsed), pipeline.extract(parsed), parsed)
