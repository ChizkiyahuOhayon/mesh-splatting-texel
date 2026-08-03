"""Locate the RITS G0-lite gradient discrepancy by peeling one ingredient at a time.

Diagnostic only: it trains nothing, decides nothing, and writes no checkpoint.
Smoke 03 showed a converged finite difference (coarse and fine agreed to 0.04%)
that disagreed with the analytic midpoint SH-DC gradient by ~8.87x. The
configurations below differ by exactly one ingredient each, so the first one
whose ratio departs from 1 names the cause:

  A baseline    unsplit model, original vertex, plain render, L1+SSIM, 4x
  B no_ssim     A with the SSIM term removed
  C no_ssaa     A at scaling 1 (no supersampling / area downsample)
  D split_orig  split model, original vertex, otherwise A
  E split_mid   split model, midpoint vertex, otherwise A
  F g0_lite     E with the 0.5 donor blend (what the gate actually evaluates)

If A already deviates, the production backward is systematically scaled and the
gate's premise -- not the split -- is what must change.
"""

import json
from argparse import ArgumentParser
from pathlib import Path

import torch

from arguments import ModelParams, PipelineParams, get_combined_args
from rits_prolongation import fd_probe_indices, install_trainable_split
from rits_t0_train import FD_RUNGS, LAMBDA_DSSIM, _photometric_loss, _select_faces
from scene import Scene
from scene.triangle_model import TriangleModel
from svsr_g1_eval import _checkpoint
from svsr_metadata import reserve_output_directory, source_revision
from triangle_renderer import render
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss


DIAG_FACE_FRACTION = 0.01
PROBES = 4


def _loss_factory(view, triangles, pipeline, background, use_ssim, donor_image):
    target = view.original_image.cuda().double()

    def loss():
        image = render(view, triangles, pipeline, background)["render"]
        if donor_image is not None:
            image = 0.5 * image + 0.5 * donor_image
        image = image.double()
        if use_ssim:
            return _photometric_loss(image, target)
        return l1_loss(image, target)

    return loss


def _probe(triangles, loss_fn, vertex_slice):
    """Analytic vs central-difference gradient on the top-|grad| f_dc scalars."""
    triangles._features_dc.grad = None
    loss_fn().backward()
    block_gradient = triangles._features_dc.grad[vertex_slice].detach()
    indices = fd_probe_indices(block_gradient, PROBES)
    flat_gradient = block_gradient.flatten()
    rows = []
    with torch.no_grad():
        flat = triangles._features_dc.data[vertex_slice].reshape(-1)
        for index in indices.tolist():
            original = float(flat[index])
            analytic = float(flat_gradient[index])
            estimates = []
            for step in FD_RUNGS:
                samples = []
                for sign in (1.0, -1.0):
                    flat[index] = original + sign * step
                    samples.append(float(loss_fn()))
                flat[index] = original
                estimates.append((samples[0] - samples[1]) / (2.0 * step))
            rows.append(
                {
                    "index": index,
                    "analytic": analytic,
                    "fd_coarse": estimates[0],
                    "fd_fine": estimates[1],
                    "ratio_fd_over_analytic": (
                        estimates[1] / analytic if analytic != 0.0 else None
                    ),
                }
            )
    triangles._features_dc.grad = None
    ratios = [row["ratio_fd_over_analytic"] for row in rows if row["ratio_fd_over_analytic"]]
    return {
        "rows": rows,
        "median_ratio": float(torch.tensor(ratios).median()) if ratios else None,
    }


def run(dataset, pipeline, args):
    output_root = reserve_output_directory(Path(args.diag_output).resolve())
    iteration, _ = _checkpoint(dataset.model_path, args.iteration)
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
    pipeline.texel_footprint_filter = False
    background_values = [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0]
    background = torch.tensor(background_values, dtype=torch.float32, device="cuda")
    train_views = sorted(scene.getTrainCameras(), key=lambda view: view.image_name)
    view = train_views[0]

    # The unsplit model needs trainable leaves; loading leaves them detached.
    triangles._features_dc = triangles._features_dc.detach().requires_grad_(True)
    results = {}

    results["A_baseline"] = _probe(
        triangles,
        _loss_factory(view, triangles, pipeline, background, True, None),
        slice(None),
    )
    results["B_no_ssim"] = _probe(
        triangles,
        _loss_factory(view, triangles, pipeline, background, False, None),
        slice(None),
    )
    triangles.scaling = 1
    results["C_no_ssaa"] = _probe(
        triangles,
        _loss_factory(view, triangles, pipeline, background, True, None),
        slice(None),
    )
    triangles.scaling = 4

    selected_faces, _ = _select_faces(
        triangles, pipeline, background, train_views[:4], DIAG_FACE_FRACTION
    )
    split = install_trainable_split(triangles, selected_faces)
    base = split["base_vertex_count"]
    plain = _loss_factory(view, triangles, pipeline, background, True, None)
    results["D_split_orig"] = _probe(triangles, plain, slice(0, base))
    results["E_split_mid"] = _probe(triangles, plain, slice(base, None))

    with torch.no_grad():
        donor_image = render(
            view, triangles, pipeline, background, window_donors=split["window_donors"]
        )["render"]
    results["F_g0_lite"] = _probe(
        triangles,
        _loss_factory(view, triangles, pipeline, background, True, donor_image),
        slice(base, None),
    )

    summary = {
        "scene": args.diag_scene,
        "view": view.image_name,
        "source_revision": source_revision(),
        "lambda_dssim": LAMBDA_DSSIM,
        "fd_rungs": list(FD_RUNGS),
        "split": {key: split[key] for key in ("base_vertex_count", "split_vertex_count")},
        "median_ratio_fd_over_analytic": {
            name: row["median_ratio"] for name, row in results.items()
        },
        "configurations": results,
    }
    with open(output_root / "results.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    (output_root / "DONE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = ArgumentParser(description="RITS G0-lite gradient discrepancy diagnostic")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--diag_scene", required=True, choices=("garden", "room"))
    parser.add_argument("--diag_output", required=True)
    parsed = get_combined_args(parser)
    if not parsed.model_path or not parsed.source_path:
        parser.error("Both --model_path/-m and --source_path/-s are required.")
    safe_state(parsed.quiet)
    run(model.extract(parsed), pipeline.extract(parsed), parsed)
