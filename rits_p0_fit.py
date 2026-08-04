"""RITS-P0: project the parent's rendered function onto the child parameters.

The abrupt operator interpolates a parent's values onto its midpoints, which
leaves a render discrepancy the optimizer must repair. RITS-D1 showed the
parent's function is recoverable to ~1e-8 while window donors are active, but
that state carries no gradient for the new degrees of freedom. This gate asks
whether the same function survives **donor-free** when the midpoint parameters
are fitted to it instead of inherited from it, and whether that fit transfers
to views it never saw.
"""

import json
import time
from argparse import ArgumentParser
from pathlib import Path

import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from csu_f0_eval import _dominant_parent_mask, _select_evenly
from rits_p0_decide import REQUIRED_REDUCTION, decide
from rits_prolongation import (
    SPLIT_PARAMETERS,
    install_trainable_split,
    original_prefix_unchanged,
    zero_original_gradients,
)
from scene import Scene
from scene.triangle_model import TriangleModel
from svsr_g1_eval import _checkpoint, _sha256
from svsr_metadata import reserve_output_directory, source_revision
from triangle_renderer import render
from utils.general_utils import safe_state


LOCKED_TRAIN_VIEWS = 4
LOCKED_TEST_VIEWS = 4
LOCKED_FACE_COUNT = 512
LOCKED_STEPS = 1_000
SMOKE_TRAIN_VIEWS = 1
SMOKE_TEST_VIEWS = 1
SMOKE_FACE_COUNT = 32
SMOKE_STEPS = 50
GEOMETRY_PARAMETERS = ("vertices",)
APPEARANCE_PARAMETERS = ("_features_dc", "_features_rest")


def _discrepancy(views_with_targets, triangles, pipeline, background):
    """Donor-free render error against stored per-view targets and masks."""
    global_error = 0.0
    global_pixels = 0
    region_error = 0.0
    region_pixels = 0
    per_view = []
    with torch.no_grad():
        for view, target, mask in views_with_targets:
            prediction = render(view, triangles, pipeline, background)["render"].cpu()
            pixel_error = (prediction - target).abs().mean(dim=0)
            view_region_pixels = int(mask.sum()) if mask is not None else 0
            per_view.append(
                {
                    "view": view.image_name,
                    "global_mae": float(pixel_error.mean()),
                    "probe_region_pixels": view_region_pixels,
                    "probe_region_mae": (
                        float(pixel_error[mask].mean()) if view_region_pixels else None
                    ),
                    "max_abs_error": float((prediction - target).abs().max()),
                }
            )
            global_error += float(pixel_error.sum())
            global_pixels += pixel_error.numel()
            if view_region_pixels:
                region_error += float(pixel_error[mask].sum())
                region_pixels += view_region_pixels
    return {
        "global_mae": global_error / global_pixels,
        "probe_region_pixels": region_pixels,
        "probe_region_mae": region_error / region_pixels if region_pixels else None,
        "per_view": per_view,
    }


def _group_change(triangles, initial, names, base_vertices):
    total = 0.0
    for name in names:
        moved = getattr(triangles, name)[base_vertices:].detach() - initial[name]
        total += float(moved.pow(2).sum())
    return total ** 0.5


def run(dataset, pipeline, args):
    output_root = reserve_output_directory(Path(args.p0_output).resolve())
    iteration, checkpoint = _checkpoint(dataset.model_path, args.iteration)
    train_count = SMOKE_TRAIN_VIEWS if args.p0_smoke else LOCKED_TRAIN_VIEWS
    test_count = SMOKE_TEST_VIEWS if args.p0_smoke else LOCKED_TEST_VIEWS
    face_limit = SMOKE_FACE_COUNT if args.p0_smoke else LOCKED_FACE_COUNT
    steps = SMOKE_STEPS if args.p0_smoke else LOCKED_STEPS

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
        raise RuntimeError("RITS-P0 requires an unmodified SH-only checkpoint.")
    train_views = _select_evenly(scene.getTrainCameras(), train_count)
    test_views = _select_evenly(scene.getTestCameras(), test_count)
    pipeline.texel_footprint_filter = False
    background_values = [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0]
    background = torch.tensor(background_values, dtype=torch.float32, device="cuda")

    manifest = {
        "protocol": "experiments/rits_p0/protocol.md",
        "scene": args.p0_scene,
        "dataset_path": str(Path(dataset.source_path).resolve()),
        "model_path": str(Path(dataset.model_path).resolve()),
        "iteration": iteration,
        "checkpoint_sha256": _sha256(checkpoint),
        "source_revision": source_revision(),
        "train_views": [view.image_name for view in train_views],
        "test_views": [view.image_name for view in test_views],
        "candidate_screen": "summed_projected_pixel_coverage",
        "selected_parent_faces": face_limit,
        "fit_steps": steps,
        "fit_objective": "mse_vs_stored_donor_render",
        "fit_schedule": "one training view per step, fixed cyclic order",
        "opacity_learning_rate": 0.0,
        "required_probe_region_reduction": REQUIRED_REDUCTION,
        "smoke": args.p0_smoke,
        "confirmatory_settings": not args.p0_smoke,
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        "torch": torch.__version__,
    }
    with open(output_root / "p0_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    face_count = triangles._triangle_indices.shape[0]
    coverage = torch.zeros(face_count, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        for view in train_views:
            coverage += render(view, triangles, pipeline, background)[
                "triangle_was_rendered"
            ].float()
    visible_faces = int((coverage > 0).sum())
    if visible_faces < face_limit:
        raise RuntimeError(
            f"only {visible_faces} faces are visible, fewer than the locked {face_limit}"
        )
    selected_faces = torch.topk(coverage, face_limit, sorted=False).indices

    # The unsplit model is the reference the projection must reproduce; the
    # probe mask marks the pixels its selected parents dominate.
    held_out = []
    with torch.no_grad():
        for view in test_views:
            package = render(view, triangles, pipeline, background)
            prediction = package["render"].detach()
            mask = _dominant_parent_mask(package, selected_faces, prediction.shape[-2:])
            held_out.append((view, prediction.cpu(), mask.cpu()))
    fitted_reference = []
    with torch.no_grad():
        for view in train_views:
            prediction = render(view, triangles, pipeline, background)["render"].detach()
            fitted_reference.append((view, prediction.cpu(), None))

    originals = {
        name: getattr(triangles, name).detach().clone() for name in SPLIT_PARAMETERS
    }
    split = install_trainable_split(triangles, selected_faces)
    base_vertices = split["base_vertex_count"]
    initial_midpoint = {
        name: getattr(triangles, name)[base_vertices:].detach().clone()
        for name in GEOMETRY_PARAMETERS + APPEARANCE_PARAMETERS
    }

    # The target is the parent's function, frozen at initialisation: RITS-D1
    # showed the donor render reproduces it to ~1e-8. It must not be refreshed
    # during fitting, because moving midpoints would drag the target with them.
    targets = []
    with torch.no_grad():
        for view in train_views:
            targets.append(
                render(
                    view, triangles, pipeline, background,
                    window_donors=split["window_donors"],
                )["render"].detach().clone()
            )

    inherited_held_out = _discrepancy(held_out, triangles, pipeline, background)
    inherited_train = _discrepancy(fitted_reference, triangles, pipeline, background)

    losses = []
    progress = tqdm(range(1, steps + 1), desc=f"RITS-P0 {args.p0_scene}")
    for step in progress:
        view = train_views[(step - 1) % len(train_views)]
        target = targets[(step - 1) % len(targets)]
        prediction = render(view, triangles, pipeline, background)["render"]
        loss = (prediction - target).pow(2).mean()
        loss.backward()
        zero_original_gradients(triangles, base_vertices)
        triangles.optimizer.step()
        triangles.optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % 100 == 0 or step == steps:
            losses.append({"step": step, "mse": float(loss.detach())})
            progress.set_postfix(mse=f"{float(loss.detach()):.3e}")

    projected_held_out = _discrepancy(held_out, triangles, pipeline, background)
    projected_train = _discrepancy(fitted_reference, triangles, pipeline, background)

    checks = decide(inherited_held_out, projected_held_out)
    checks["original_parameter_prefix_bitwise_unchanged"] = original_prefix_unchanged(
        triangles, originals
    )
    checks["topology_counts_exact"] = (
        triangles._triangle_indices.shape[0] == split["split_face_count"]
        and triangles.vertices.shape[0] == split["split_vertex_count"]
    )
    decision = (
        {"scene_pass": None, "reason": "implementation smoke; no gate decision"}
        if args.p0_smoke
        else {"scene_pass": all(checks.values()), "checks": checks}
    )
    torch.cuda.synchronize()
    results = {
        "scene": args.p0_scene,
        "visible_candidate_faces": visible_faces,
        "selected_parent_faces": face_limit,
        "split": {
            key: split[key]
            for key in (
                "base_vertex_count",
                "base_face_count",
                "split_vertex_count",
                "split_face_count",
                "unique_edge_count",
                "child_face_start",
            )
        },
        "held_out": {"inherited": inherited_held_out, "projected": projected_held_out},
        "fitted_views": {"inherited": inherited_train, "projected": projected_train},
        "generalisation_gap": (
            projected_held_out["probe_region_mae"] - projected_train["global_mae"]
            if projected_held_out["probe_region_mae"] is not None
            else None
        ),
        "parameter_change": {
            "geometry_l2": _group_change(
                triangles, initial_midpoint, GEOMETRY_PARAMETERS, base_vertices
            ),
            "appearance_l2": _group_change(
                triangles, initial_midpoint, APPEARANCE_PARAMETERS, base_vertices
            ),
        },
        "loss_trace": losses,
        "decision": decision,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    with open(output_root / "results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, allow_nan=False)
    (output_root / "DONE").write_text("complete\n", encoding="utf-8")
    print(json.dumps({"results": str(output_root / "results.json"), **results}, indent=2))


if __name__ == "__main__":
    parser = ArgumentParser(description="RITS-P0 projection gate")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--p0_scene", required=True, choices=("garden", "room"))
    parser.add_argument("--p0_output", required=True)
    parser.add_argument("--p0_smoke", action="store_true")
    parsed = get_combined_args(parser)
    if not parsed.model_path or not parsed.source_path:
        parser.error("Both --model_path/-m and --source_path/-s are required.")
    if not parsed.eval:
        parser.error("RITS-P0 requires --eval so held-out views are loaded.")
    safe_state(parsed.quiet)
    run(model.extract(parsed), pipeline.extract(parsed), parsed)
