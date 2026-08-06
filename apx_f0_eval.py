"""APX-F0: how much appearance capacity is left, and can it be found cheaply?

Trains nothing. One checkpoint, three passes over the training views and one over the
held-out views, and the three locked questions are answered:

  ceiling         can a higher-capacity appearance model remove held-out error at all
  concentration   is that gain concentrated, or would uniform allocation do as well
  predictability  does a signal an allocator can actually see find the faces

Cell colours are fitted on training views and scored on held-out views; fitting and
scoring the same pixels would measure memorisation. The protocol is
experiments/apx_f0/protocol.md and the rule is apx_f0_decide.py.
"""

import json
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn.functional as F

from apx_cells import (
    CELL_ORDERS,
    MIN_PIXELS_PER_CELL,
    barycentric,
    cell_index,
    gain_to_db,
    squared_error,
)
from apx_f0_decide import NON_RESIDUAL_CONTROLS, PRIMARY_SIGNAL, scene_checks
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from scene.triangle_model import TriangleModel
from svsr_g1_eval import _checkpoint, _sha256
from svsr_metadata import reserve_output_directory, source_revision
from triangle_renderer import render
from utils.general_utils import safe_state
from xvr_g0_eval import _select_evenly
from xvr_score import top_fraction_capture


ALPHA_THRESHOLD = 0.5  # XVR-G0's locked value; the same attribution rule and threshold
TRAIN_VIEWS = 16
CAPTURE_FRACTIONS = (0.01, 0.05, 0.10)


def _covered(package, target):
    """Dominant-face id, pixel coordinate and colours for every covered pixel."""
    height, width = target.shape[-2:]
    face_ids = F.interpolate(
        package["rend_ids"].unsqueeze(0), size=(height, width), mode="nearest"
    ).squeeze(0).squeeze(0)
    alpha = F.interpolate(
        package["rend_alpha"].unsqueeze(0), size=(height, width), mode="nearest"
    ).squeeze(0).squeeze(0)
    face_count = package["max_blending"].numel()
    mask = (face_ids >= 0) & (face_ids < face_count) & (alpha > ALPHA_THRESHOLD)

    rows, columns = torch.meshgrid(
        torch.arange(height, device=target.device, dtype=torch.float32),
        torch.arange(width, device=target.device, dtype=torch.float32),
        indexing="ij",
    )
    return {
        "faces": face_ids[mask].long(),
        "points": torch.stack([columns[mask], rows[mask]], dim=1),
        "target": target[:, mask].transpose(0, 1).contiguous(),
        "prediction": package["render"][:, mask].transpose(0, 1).contiguous(),
    }


def _cells(pixels, triangles, image_2d, order):
    """Which cell of each face's `order * order` grid every covered pixel falls in."""
    corners = image_2d[triangles._triangle_indices[pixels["faces"]].long()]
    return cell_index(barycentric(pixels["points"], corners), order)


def _pass(views, triangles, pipeline, background, visit):
    """Render each view once and hand the covered pixels to `visit`."""
    with torch.no_grad():
        for view in views:
            package = render(view, triangles, pipeline, background)
            target = view.original_image[:3].to(package["render"].device)
            visit(package, _covered(package, target))


def _train_pixel_counts(views, triangles, pipeline, background, face_count):
    counts = torch.zeros(face_count, device="cuda")
    residual = torch.zeros(face_count, device="cuda")
    blending = torch.zeros(face_count, device="cuda")

    def visit(package, pixels):
        counts.scatter_add_(0, pixels["faces"], torch.ones_like(pixels["faces"], dtype=counts.dtype))
        residual.scatter_add_(
            0, pixels["faces"], (pixels["prediction"] - pixels["target"]).abs().mean(1)
        )
        blending.copy_(torch.maximum(blending, package["max_blending"]))

    _pass(views, triangles, pipeline, background, visit)
    return counts, residual, blending


def _fit(views, triangles, pipeline, background, compact, order):
    """Fit one **residual correction** per cell from the candidate faces' training pixels.

    The deployed texel carrier is additive on top of SH -- `forward.cu:775-792` does
    `interp += texels[...]`, with SH carrying view dependence and texels carrying high
    spatial frequency. Fitting the target instead of the residual would measure
    *replacing* the appearance with a static per-cell colour, which discards view
    dependence and loses several dB for that reason alone. That is a correct
    measurement of a model class nobody proposed, and it is what the first run of this
    gate did.
    """
    bin_count = compact["count"] * order * order
    totals = torch.zeros(bin_count, 3, device="cuda")
    weights = torch.zeros(bin_count, device="cuda")

    def visit(package, pixels):
        keep = compact["index"][pixels["faces"]] >= 0
        if not bool(keep.any()):
            return
        kept = {k: v[keep] for k, v in pixels.items()}
        local = compact["index"][kept["faces"]]
        bins = local * (order * order) + _cells(
            kept, triangles, package["image_2D"], order
        )
        totals.index_add_(0, bins, kept["target"] - kept["prediction"])
        weights.index_add_(0, bins, torch.ones_like(bins, dtype=weights.dtype))

    _pass(views, triangles, pipeline, background, visit)
    return totals / weights.clamp_min(1.0).unsqueeze(1), weights == 0


def _score(views, triangles, pipeline, background, compact, fits):
    """Held-out squared error of the current model and of each fitted order."""
    faces = compact["count"]
    current = torch.zeros(faces, device="cuda")
    fitted = {order: torch.zeros(faces, device="cuda") for order in CELL_ORDERS}
    pixels_seen = torch.zeros(faces, device="cuda")

    def visit(package, pixels):
        keep = compact["index"][pixels["faces"]] >= 0
        if not bool(keep.any()):
            return
        kept = {k: v[keep] for k, v in pixels.items()}
        local = compact["index"][kept["faces"]]
        pixels_seen.scatter_add_(0, local, torch.ones_like(local, dtype=current.dtype))
        current.scatter_add_(
            0, local, ((kept["prediction"] - kept["target"]) ** 2).sum(1)
        )
        # The order-1 correction, used where a finer cell received no training pixel.
        coarse = fits[1][0][local]
        for order in CELL_ORDERS:
            corrections, empty = fits[order]
            bins = local * (order * order) + _cells(
                kept, triangles, package["image_2D"], order
            )
            # The model class is SH plus a per-cell correction, so the correction is
            # added to the current prediction rather than replacing it.
            wanted = kept["target"] - kept["prediction"]
            fitted[order].scatter_add_(
                0, local, squared_error(bins, wanted, corrections, empty, coarse)
            )

    _pass(views, triangles, pipeline, background, visit)
    return current, fitted, pixels_seen


def measure(triangles, pipeline, background, train_views, test_views):
    face_count = triangles._triangle_indices.shape[0]
    counts, residual, blending = _train_pixel_counts(
        train_views, triangles, pipeline, background, face_count
    )

    # A cell fitted from fewer than MIN_PIXELS_PER_CELL pixels is memorising, so each
    # order carries its own pixel requirement. The candidate set is the union -- every
    # face that can support the coarsest grid -- and per-order eligibility is applied
    # afterwards. Using the strictest order for all of them, as the first run did,
    # discarded every face that could have supported order 1 or 2.
    needed = {order: MIN_PIXELS_PER_CELL * order * order for order in CELL_ORDERS}
    candidate_ids = (counts >= min(needed.values())).nonzero(as_tuple=True)[0]
    index = torch.full((face_count,), -1, dtype=torch.long, device="cuda")
    index[candidate_ids] = torch.arange(candidate_ids.numel(), device="cuda")
    compact = {"index": index, "ids": candidate_ids, "count": int(candidate_ids.numel())}
    if compact["count"] == 0:
        raise RuntimeError(
            f"no face carries {min(needed.values())} training pixels; "
            "the mesh is too fine for any texel grid"
        )

    fits = {
        order: _fit(train_views, triangles, pipeline, background, compact, order)
        for order in CELL_ORDERS
    }
    current, fitted, seen = _score(
        test_views, triangles, pipeline, background, compact, fits
    )

    compact_counts = counts[compact["ids"]]
    eligible = {
        order: (seen > 0) & (compact_counts >= needed[order]) for order in CELL_ORDERS
    }
    ceiling = {
        str(order): gain_to_db(
            float(current[mask].sum()), float(fitted[order][mask].sum())
        )
        for order, mask in eligible.items()
    }

    # What an adaptive scheme could actually deploy: each face takes the finest grid
    # its own pixel count supports. Reported alongside the locked per-order ceilings
    # because it is the quantity the thesis is really about, and it is measured over
    # every candidate face rather than only those that can hold the finest grid.
    adaptive = fitted[min(CELL_ORDERS)].clone()
    for order in sorted(CELL_ORDERS):
        adaptive = torch.where(compact_counts >= needed[order], fitted[order], adaptive)
    scored = seen > 0

    decision_mask = eligible[max(CELL_ORDERS)]
    gain = (current - fitted[max(CELL_ORDERS)]).clamp_min(0.0)
    signals = {
        PRIMARY_SIGNAL: residual[compact["ids"]],
        "max_blending": blending[compact["ids"]],
        "projected_coverage": counts[compact["ids"]],
        "world_area": triangles.triangle_areas().squeeze()[compact["ids"]],
    }
    return {
        "faces_total": face_count,
        "faces_with_enough_pixels": compact["count"],
        "faces_scored": int(scored.sum()),
        # Headline, not a footnote: at 7M triangles over ~1M pixels the average face is
        # sub-pixel, and a per-face scheme can only ever reach the eligible part.
        "eligible_fraction": compact["count"] / face_count,
        "eligible_fraction_per_order": {
            str(order): int(mask.sum()) / face_count for order, mask in eligible.items()
        },
        "held_out_pixels": int(seen.sum()),
        "ceiling_db": ceiling,
        "ceiling_db_adaptive": gain_to_db(
            float(current[scored].sum()), float(adaptive[scored].sum())
        ),
        "concentration": {
            f"top_{int(fraction * 100)}pct": top_fraction_capture(
                gain, gain, decision_mask, fraction
            )["capture"]
            for fraction in CAPTURE_FRACTIONS
        },
        "signals": {
            name: {
                f"top_{int(fraction * 100)}pct": top_fraction_capture(
                    score, gain, decision_mask, fraction
                )
                for fraction in CAPTURE_FRACTIONS
            }
            for name, score in signals.items()
        },
    }


def run(dataset, pipeline, args):
    output_root = reserve_output_directory(Path(args.apx_output).resolve())
    iteration, checkpoint = _checkpoint(dataset.model_path, args.iteration)

    triangles = TriangleModel(dataset.sh_degree)
    triangles.scaling = 4
    scene = Scene(
        args=dataset, triangles=triangles, init_opacity=None, set_sigma=None,
        load_iteration=iteration, shuffle=False,
    )
    if triangles.texel_order != 0:
        raise RuntimeError("APX-F0 measures the SH-only baseline's remaining capacity.")
    pipeline.texel_footprint_filter = False
    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32, device="cuda",
    )
    train_views = _select_evenly(
        sorted(scene.getTrainCameras(), key=lambda v: v.image_name), TRAIN_VIEWS
    )
    test_views = sorted(scene.getTestCameras(), key=lambda v: v.image_name)

    scene_result = measure(triangles, pipeline, background, train_views, test_views)
    results = {
        "protocol": "experiments/apx_f0/protocol.md",
        "scene": args.apx_scene,
        "model_path": str(Path(dataset.model_path).resolve()),
        "iteration": iteration,
        "checkpoint_sha256": _sha256(checkpoint),
        "source_revision": source_revision(),
        "train_views": len(train_views),
        "test_views": len(test_views),
        "constants": {
            "alpha_threshold": ALPHA_THRESHOLD,
            "cell_orders": list(CELL_ORDERS),
            "min_pixels_per_cell": MIN_PIXELS_PER_CELL,
        },
        **scene_result,
        "checks": scene_checks(scene_result),
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
    }
    with open(output_root / "results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, allow_nan=False)
    (output_root / "DONE").write_text("complete\n", encoding="utf-8")

    print(f"\nAPX-F0 {args.apx_scene}")
    print(f"  eligible faces {scene_result['faces_with_enough_pixels']:,}"
          f" / {scene_result['faces_total']:,}"
          f"  ({100 * scene_result['eligible_fraction']:.2f}%)")
    for order, value in scene_result["ceiling_db"].items():
        share = scene_result["eligible_fraction_per_order"][order]
        print(f"  ceiling order {order}: {value:+.4f} dB   (reaches {100 * share:.2f}% of faces)")
    print(f"  ceiling adaptive: {scene_result['ceiling_db_adaptive']:+.4f} dB"
          f"   (each face takes the finest grid it supports)")
    print(f"  concentration top10% {scene_result['concentration']['top_10pct']:.4f}")
    for name in (PRIMARY_SIGNAL, *NON_RESIDUAL_CONTROLS):
        row = scene_result["signals"][name]["top_10pct"]
        print(f"  {name:>20}  capture {row['capture']:.4f}  lift {row['lift']:.3f}")
    for condition, check in results["checks"].items():
        print(f"  {condition:>15}: {'PASS' if check['pass'] else 'FAIL'}")
    print(f"  results: {output_root / 'results.json'}")


if __name__ == "__main__":
    parser = ArgumentParser(description="APX-F0 appearance-capacity forensics")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--apx_scene", required=True)
    parser.add_argument("--apx_output", required=True)
    parsed = get_combined_args(parser)
    if not parsed.model_path or not parsed.source_path:
        parser.error("Both --model_path/-m and --source_path/-s are required.")
    if not parsed.eval:
        parser.error("APX-F0 requires --eval so the held-out split is fixed.")
    safe_state(parsed.quiet)
    run(model.extract(parsed), pipeline.extract(parsed), parsed)
