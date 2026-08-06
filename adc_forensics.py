"""ADC-F0: is the rasterizer's SH-DC gradient under-report the same for every primitive?

The published backward reports SH-DC gradients about 8.44x smaller than a converged
central difference (experiments/rits_t0/results/g0_diag_garden_01.md). The project
assumed Adam's per-parameter scale invariance absorbs that, which holds only if the
under-report is uniform -- and 8.44 is not a clean constant. This measures whether the
ratio depends on a primitive's projected size, its depth, or its share of the image.

Diagnostic only: it trains nothing and writes no checkpoint. The protocol is
experiments/adc_f0/protocol.md; the reading lives in adc_f0_decide.py and decides
whether the coming densification criterion may be built on analytic gradients at all.
"""

import json
from argparse import ArgumentParser
from pathlib import Path

import torch

from adc_f0_decide import COVARIATES, read, spearman, spread, stratum_medians
from adc_probe import (
    PROBES_PER_STRATUM,
    RUNG_TOLERANCE,
    RUNGS,
    STRATA,
    central_differences,
    face_max_per_vertex,
    rung_disagreement,
    stratified_probe_set,
)
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from scene.triangle_model import TriangleModel
from svsr_g1_eval import _checkpoint, _sha256
from svsr_metadata import reserve_output_directory, source_revision
from triangle_renderer import render
from utils.general_utils import safe_state


STRATIFY_BY = "projected_size"


def _covariates(package, triangles):
    """Per-vertex covariates, and the mask of vertices this view actually rendered.

    `scaling` and `max_blending` come back face-indexed despite the renderer slicing
    them with a variable named `vertex_index`; that name is the face count.
    """
    faces = triangles._triangle_indices
    vertex_count = triangles.vertices.shape[0]
    covariates = {
        "projected_size": face_max_per_vertex(
            package["scaling"].detach().float(), faces, vertex_count
        ),
        "depth": package["vertex_depth_out"].detach().float(),
        "max_blending": face_max_per_vertex(
            package["max_blending"].detach().float(), faces, vertex_count
        ),
    }
    rendered = face_max_per_vertex(
        (package["triangle_was_rendered"] > 0).float(), faces, vertex_count
    )
    return covariates, rendered > 0


def _probe_loss(view, triangles, pipeline, background):
    """Mean squared error against the target, reduced in float64.

    The image is linear in a vertex's SH-DC coefficient, so this is exactly quadratic
    in the probed scalar and its central difference is exact at any step size. The
    training loss is not: `L1 + SSIM` has kinks wherever a pixel residual changes
    sign, and those kinks once made two rungs disagree by 8% on Room. The float64
    reduction answers the other historical failure, where a float32 mean lost the
    loss change below its own ulp.
    """
    target = view.original_image[:3].cuda().double()

    def loss():
        image = render(view, triangles, pipeline, background)["render"].double()
        return ((image - target) ** 2).mean()

    return loss


def _probe(parameter, gradient, vertex, loss_fn):
    """Central-difference / analytic ratio for one vertex's strongest colour channel.

    Probing the largest-|gradient| channel keeps the loss change as far above the
    float floor as the gradient allows; a uniformly chosen channel is often invisible
    in any single view.
    """
    channel = int(gradient[vertex, 0].abs().argmax())
    analytic = float(gradient[vertex, 0, channel])
    original = float(parameter[vertex, 0, channel])

    def set_value(value):
        parameter[vertex, 0, channel] = value

    with torch.no_grad():
        estimates = central_differences(set_value, loss_fn, original)
    coarse, fine = estimates
    return {
        "vertex": int(vertex),
        "channel": channel,
        "analytic": analytic,
        "fd_coarse": coarse,
        "fd_fine": fine,
        "rung_disagreement": rung_disagreement(estimates),
        "ratio": fine / analytic if analytic != 0.0 else None,
    }


def _measure_view(view, triangles, pipeline, background):
    """Every probe for one view, with covariates attached and validity checked."""
    loss_fn = _probe_loss(view, triangles, pipeline, background)
    with torch.no_grad():
        package = render(view, triangles, pipeline, background)
        # A non-deterministic forward would make every central difference noise; it
        # is cheaper to prove it once than to explain a strange ratio later.
        deterministic = bool(
            torch.equal(
                package["render"].clone(),
                render(view, triangles, pipeline, background)["render"],
            )
        )
    covariates, eligible_mask = _covariates(package, triangles)
    eligible = eligible_mask.nonzero(as_tuple=True)[0]
    if eligible.numel() < STRATA * PROBES_PER_STRATUM:
        raise RuntimeError(
            f"{view.image_name}: only {eligible.numel()} rendered vertices, "
            f"need {STRATA * PROBES_PER_STRATUM}"
        )

    # One backward yields the analytic gradient for every probed vertex at once.
    triangles._features_dc.grad = None
    loss_fn().backward()
    gradient = triangles._features_dc.grad.detach()

    probes = []
    for vertex in stratified_probe_set(covariates[STRATIFY_BY], eligible).tolist():
        row = _probe(triangles._features_dc.data, gradient, vertex, loss_fn)
        row.update({name: float(values[vertex]) for name, values in covariates.items()})
        probes.append(row)
    triangles._features_dc.grad = None

    kept = [
        row
        for row in probes
        if row["ratio"] is not None and row["rung_disagreement"] <= RUNG_TOLERANCE
    ]
    ratios = [row["ratio"] for row in kept]
    return {
        "view": view.image_name,
        "deterministic_forward": deterministic,
        "probes": probes,
        "probes_attempted": len(probes),
        "probes_kept": len(kept),
        "survival_fraction": len(kept) / len(probes) if probes else 0.0,
        "median_ratio": float(torch.tensor(ratios, dtype=torch.float64).median())
        if ratios
        else None,
        "stratum_medians": {
            covariate: stratum_medians([row[covariate] for row in kept], ratios)
            for covariate in COVARIATES
        },
        "spearman": {
            covariate: spearman([row[covariate] for row in kept], ratios)
            for covariate in COVARIATES
        },
    }


def _report(results):
    print(f"\nADC-F0 reading: {results['reading']}")
    for view in results["views"]:
        median = view["median_ratio"]
        print(
            f"  {view['view']}: median ratio {median:.4f}"
            if median is not None
            else f"  {view['view']}: median ratio n/a"
        )
        print(
            f"    kept {view['probes_kept']}/{view['probes_attempted']}"
            f"  deterministic={view['deterministic_forward']}"
        )
        for covariate in COVARIATES:
            medians = view["stratum_medians"][covariate]
            ratio_spread = spread(medians)
            quintiles = ", ".join(f"{value:.3f}" for value in medians)
            print(
                f"    {covariate:>16}  spread "
                f"{'n/a' if ratio_spread is None else f'{ratio_spread:.3f}'}"
                f"  rho {view['spearman'][covariate]:+.3f}  [{quintiles}]"
            )
    if "strongest_trend" in results:
        trend = results["strongest_trend"]
        print(
            f"  strongest trend: {trend['covariate']} on {trend['view']}"
            f" (rho {trend['spearman']:+.3f})"
        )
    if "reason" in results:
        print(f"  reason: {results['reason']}")


def run(dataset, pipeline, args):
    output_root = reserve_output_directory(Path(args.adc_output).resolve())
    iteration, checkpoint = _checkpoint(dataset.model_path, args.iteration)

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
        raise RuntimeError("ADC-F0 measures SH-only checkpoints.")
    pipeline.texel_footprint_filter = False
    background_values = [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0]
    background = torch.tensor(background_values, dtype=torch.float32, device="cuda")

    # Loading leaves the appearance detached; the probe needs a trainable leaf.
    triangles._features_dc = triangles._features_dc.detach().requires_grad_(True)
    views = sorted(scene.getTrainCameras(), key=lambda view: view.image_name)
    measured = [
        _measure_view(view, triangles, pipeline, background)
        for view in views[: args.adc_views]
    ]

    results = {
        "protocol": "experiments/adc_f0/protocol.md",
        "scene": args.adc_scene,
        "dataset_path": str(Path(dataset.source_path).resolve()),
        "model_path": str(Path(dataset.model_path).resolve()),
        "iteration": iteration,
        "checkpoint_sha256": _sha256(checkpoint),
        "source_revision": source_revision(),
        "scaling": 4,
        "constants": {
            "strata": STRATA,
            "probes_per_stratum": PROBES_PER_STRATUM,
            "rungs": list(RUNGS),
            "rung_tolerance": RUNG_TOLERANCE,
            "stratify_by": STRATIFY_BY,
        },
        "views": measured,
        **read(measured),
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        "torch": torch.__version__,
    }
    with open(output_root / "results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, allow_nan=False)
    (output_root / "DONE").write_text("complete\n", encoding="utf-8")
    _report(results)
    print(f"  results: {output_root / 'results.json'}")


if __name__ == "__main__":
    parser = ArgumentParser(description="ADC-F0 SH-DC gradient homogeneity forensics")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--adc_scene", required=True)
    parser.add_argument("--adc_output", required=True)
    parser.add_argument("--adc_views", type=int, default=2,
                        help="training views to measure; two is replication, not power")
    parsed = get_combined_args(parser)
    if not parsed.model_path or not parsed.source_path:
        parser.error("Both --model_path/-m and --source_path/-s are required.")
    safe_state(parsed.quiet)
    run(model.extract(parsed), pipeline.extract(parsed), parsed)
