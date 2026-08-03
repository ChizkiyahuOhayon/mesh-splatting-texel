"""RITS-T0: fixed-budget refinement fine-tuning, one arm per invocation."""

import json
import random
import time
from argparse import ArgumentParser
from pathlib import Path

import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from lpipsPyTorch.modules.lpips import LPIPS
from rits_prolongation import FINETUNE_LRS, fd_ratio_probe, install_trainable_split
from scene import Scene
from scene.triangle_model import TriangleModel
from svsr_g1_eval import _checkpoint, _metrics, _sha256
from svsr_metadata import reserve_output_directory, source_revision
from triangle_renderer import render
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim


LOCKED_STEPS = 5_000
LOCKED_ANNEAL_STEPS = 1_000
LOCKED_SELECT_FRACTION = 0.10
SMOKE_STEPS = 200
SMOKE_ANNEAL_STEPS = 40
SMOKE_SELECT_FRACTION = 0.01
LAMBDA_DSSIM = 0.2
SEED = 1234
FD_PROBES = 4
FD_RUNGS = (0.002, 0.001)
FD_CONVERGENCE_TOLERANCE = 0.05
FD_FIDELITY_TOLERANCE = 0.25
ARMS = ("unsplit", "abrupt", "rits")

PARAMETER_NAMES = ("vertices", "vertex_weight", "_features_dc", "_features_rest")


def _photometric_loss(prediction, target):
    return (1.0 - LAMBDA_DSSIM) * l1_loss(prediction, target) + LAMBDA_DSSIM * (
        1.0 - ssim(prediction, target)
    )


def _gamma(step, anneal_steps):
    return min(1.0, step / anneal_steps)


def _blended_render(view, triangles, pipeline, background, donors, gamma):
    """The homotopy F_gamma; the donor pass carries no gradient by design."""
    child = render(view, triangles, pipeline, background)["render"]
    if gamma >= 1.0 or donors is None:
        return child
    with torch.no_grad():
        donor = render(view, triangles, pipeline, background, window_donors=donors)[
            "render"
        ]
    return gamma * child + (1.0 - gamma) * donor


def _evaluate(views, triangles, pipeline, background, lpips_metric):
    rows = []
    with torch.no_grad():
        for view in views:
            prediction = render(view, triangles, pipeline, background)["render"]
            target = view.original_image[:3].to(prediction.device)
            rows.append(
                {"view": view.image_name, **_metrics(prediction, target, lpips_metric)}
            )
    means = {
        key: sum(row[key] for row in rows) / len(rows)
        for key in ("psnr", "ssim", "lpips_vgg")
    }
    return {"mean": means, "per_view": rows}


def _select_faces(triangles, pipeline, background, train_views, fraction):
    face_count = triangles._triangle_indices.shape[0]
    coverage = torch.zeros(face_count, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        for view in train_views:
            package = render(view, triangles, pipeline, background)
            coverage += package["triangle_was_rendered"].float()
    count = max(1, int(face_count * fraction))
    visible = int((coverage > 0).sum())
    if visible < count:
        raise RuntimeError(f"only {visible} visible faces for a selection of {count}")
    return torch.topk(coverage, count, sorted=False).indices, count


def _g0_lite(triangles, pipeline, background, view, split):
    """Locked precondition of the rits arm; raises on failure.

    The loss here is reduced in float64 (renders stay float32) so per-scalar
    steps of 0.002 produce differences far above the reduction accuracy; the
    float32 training loss cannot resolve them (protocol amendment history).
    The donor image is independent of midpoint parameters, so it is rendered
    once and reused across every evaluation.
    """
    base_vertices = split["base_vertex_count"]
    target = view.original_image.cuda().double()
    with torch.no_grad():
        donor_image = render(
            view, triangles, pipeline, background, window_donors=split["window_donors"]
        )["render"]

    def blended_loss():
        child = render(view, triangles, pipeline, background)["render"]
        image = 0.5 * child + 0.5 * donor_image
        return _photometric_loss(image.double(), target)

    blended_loss().backward()
    grads = {name: getattr(triangles, name).grad for name in PARAMETER_NAMES}
    if not all(grad is not None and torch.isfinite(grad).all() for grad in grads.values()):
        raise RuntimeError("G0-lite: non-finite or missing gradients")
    geometry_norm = float(grads["vertices"][base_vertices:].norm())
    appearance_norm = float(
        (
            grads["_features_dc"][base_vertices:].norm() ** 2
            + grads["_features_rest"][base_vertices:].norm() ** 2
        )
        ** 0.5
    )
    if geometry_norm == 0.0 or appearance_norm == 0.0:
        raise RuntimeError(
            f"G0-lite: zero midpoint gradient group (geometry {geometry_norm}, "
            f"appearance {appearance_norm})"
        )

    # Relative fidelity: the split's new parameters must carry gradients as
    # faithful as the checkpoint's own. The absolute ratio is a rasterizer
    # property (~8.5x even on an untouched checkpoint), so only the agreement
    # between the two blocks is evidence about refinement.
    dc_data = triangles._features_dc.data
    dc_grad = triangles._features_dc.grad.detach()
    blocks = {
        "original": (dc_data[:base_vertices], dc_grad[:base_vertices]),
        "midpoint": (dc_data[base_vertices:], dc_grad[base_vertices:]),
    }
    fidelity = {
        name: fd_ratio_probe(
            parameter.reshape(-1), gradient.reshape(-1), blended_loss, FD_PROBES, FD_RUNGS
        )
        for name, (parameter, gradient) in blocks.items()
    }
    original_ratio = fidelity["original"]["median_ratio"]
    midpoint_ratio = fidelity["midpoint"]["median_ratio"]
    unconverged = {
        name: row["max_rung_disagreement"]
        for name, row in fidelity.items()
        if row["max_rung_disagreement"] > FD_CONVERGENCE_TOLERANCE
    }
    if unconverged:
        raise RuntimeError(f"G0-lite: finite differences did not converge: {unconverged}")
    if original_ratio is None or midpoint_ratio is None:
        raise RuntimeError("G0-lite: a probed block has no nonzero analytic gradient")
    if original_ratio <= 0.0 or midpoint_ratio <= 0.0:
        raise RuntimeError(
            f"G0-lite: gradient sign disagreement (original {original_ratio:.4f}, "
            f"midpoint {midpoint_ratio:.4f})"
        )
    relative = abs(midpoint_ratio / original_ratio - 1.0)
    if relative > FD_FIDELITY_TOLERANCE:
        raise RuntimeError(
            "G0-lite: midpoint gradient fidelity departs from the checkpoint's "
            f"by {relative:.4f} (original {original_ratio:.4f}, "
            f"midpoint {midpoint_ratio:.4f})"
        )
    for name in PARAMETER_NAMES:
        getattr(triangles, name).grad = None
    return {
        "geometry_gradient_norm": geometry_norm,
        "appearance_gradient_norm": appearance_norm,
        "fidelity": fidelity,
        "relative_fidelity_gap": relative,
    }


def run(dataset, pipeline, args):
    output_root = reserve_output_directory(Path(args.t0_output).resolve())
    iteration, checkpoint = _checkpoint(dataset.model_path, args.iteration)
    steps = SMOKE_STEPS if args.t0_smoke else LOCKED_STEPS
    anneal_steps = SMOKE_ANNEAL_STEPS if args.t0_smoke else LOCKED_ANNEAL_STEPS
    fraction = SMOKE_SELECT_FRACTION if args.t0_smoke else LOCKED_SELECT_FRACTION

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
        raise RuntimeError("RITS-T0 requires an SH-only checkpoint.")
    train_views = sorted(scene.getTrainCameras(), key=lambda view: view.image_name)
    test_views = sorted(scene.getTestCameras(), key=lambda view: view.image_name)
    pipeline.texel_footprint_filter = False
    background_values = [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0]
    background = torch.tensor(background_values, dtype=torch.float32, device="cuda")
    lpips_metric = LPIPS("vgg", "0.1").cuda().eval()

    manifest = {
        "protocol": "experiments/rits_t0/protocol.md",
        "scene": args.t0_scene,
        "arm": args.t0_arm,
        "dataset_path": str(Path(dataset.source_path).resolve()),
        "model_path": str(Path(dataset.model_path).resolve()),
        "iteration": iteration,
        "checkpoint_sha256": _sha256(checkpoint),
        "source_revision": source_revision(),
        "steps": steps,
        "anneal_steps": anneal_steps,
        "select_fraction": fraction,
        "lambda_dssim": LAMBDA_DSSIM,
        "learning_rates": FINETUNE_LRS,
        "seed": SEED,
        "scaling": 4,
        "train_views": len(train_views),
        "test_views": len(test_views),
        "smoke": args.t0_smoke,
        "confirmatory_settings": not args.t0_smoke,
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        "torch": torch.__version__,
    }
    with open(output_root / "t0_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    checkpoint_metrics = _evaluate(
        test_views, triangles, pipeline, background, lpips_metric
    )

    donors = None
    split_summary = None
    g0_lite = None
    if args.t0_arm != "unsplit":
        selected_faces, count = _select_faces(
            triangles, pipeline, background, train_views, fraction
        )
        split = install_trainable_split(triangles, selected_faces)
        split_summary = {
            key: split[key]
            for key in (
                "child_face_start",
                "base_vertex_count",
                "base_face_count",
                "split_vertex_count",
                "split_face_count",
                "unique_edge_count",
            )
        }
        split_summary["selected_faces"] = count
        if args.t0_arm == "rits":
            donors = split["window_donors"]
            g0_lite = _g0_lite(triangles, pipeline, background, train_views[0], split)

    torch.manual_seed(SEED)
    random.seed(SEED)
    viewpoint_stack = []
    losses = []
    progress = tqdm(range(1, steps + 1), desc=f"RITS-T0 {args.t0_scene}/{args.t0_arm}")
    for step in progress:
        if not viewpoint_stack:
            viewpoint_stack = list(train_views)
        view = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack) - 1))
        gamma = _gamma(step, anneal_steps) if args.t0_arm == "rits" else 1.0
        image = _blended_render(view, triangles, pipeline, background, donors, gamma)
        loss = _photometric_loss(image, view.original_image.cuda())
        loss.backward()
        triangles.optimizer.step()
        triangles.optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % 100 == 0 or step == steps:
            losses.append({"step": step, "gamma": gamma, "loss": float(loss.detach())})
            progress.set_postfix(loss=f"{float(loss.detach()):.5f}", gamma=f"{gamma:.2f}")

    final_metrics = _evaluate(test_views, triangles, pipeline, background, lpips_metric)
    scene.model_path = str(output_root)
    saved_checkpoint = scene.save(steps)
    torch.cuda.synchronize()
    results = {
        "scene": args.t0_scene,
        "arm": args.t0_arm,
        "steps": steps,
        "split": split_summary,
        "g0_lite": g0_lite,
        "checkpoint_metrics": checkpoint_metrics,
        "final_metrics": final_metrics,
        "loss_trace": losses,
        "saved_checkpoint": str(saved_checkpoint),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    with open(output_root / "results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, allow_nan=False)
    (output_root / "DONE").write_text("complete\n", encoding="utf-8")
    print(json.dumps({"results": str(output_root / "results.json"), **results}, indent=2))


if __name__ == "__main__":
    parser = ArgumentParser(description="RITS-T0 refinement fine-tuning arm")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--t0_scene", required=True, choices=("garden", "room"))
    parser.add_argument("--t0_arm", required=True, choices=ARMS)
    parser.add_argument("--t0_output", required=True)
    parser.add_argument("--t0_smoke", action="store_true")
    parsed = get_combined_args(parser)
    if not parsed.model_path or not parsed.source_path:
        parser.error("Both --model_path/-m and --source_path/-s are required.")
    if not parsed.eval:
        parser.error("RITS-T0 requires --eval so the held-out split is fixed.")
    safe_state(parsed.quiet)
    run(model.extract(parsed), pipeline.extract(parsed), parsed)
