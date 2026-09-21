"""Apply the v1 and the OATS cleanup rules to one pre-cleanup model.

The final cleanup runs after the last optimizer step, so applying it offline to
the state saved by ``--save_precleanup`` is exactly what training would have
done. Both rules see the same trained model and the same renders, and OATS
keeps as many faces as v1, so the two output models differ only in *which*
faces survived. See handover/OATS_PLAN.md and sota/survival.py.

``--budget N`` writes two more copies at an explicit face count instead of the
rule's own: ``<out>/matched`` (the *published* peak rule) and
``<out>/matched_oats`` (the integral). A run whose training kept more faces than
its control can then be scored at the control's size, which separates a better
choice of survivors from a larger mesh. ``matched_oats`` is the full method at
the control's budget; ``matched`` isolates the training-time statistic alone.

Writes two evaluable model directories (``<out>/v1`` and ``<out>/oats``, each
with ``point_cloud/iteration_<final>/``) and ``<out>/survival.json``. The v1
copy must reproduce the run's own saved checkpoint face for face; this is
checked, not assumed.

Usage:
    python -m sota.survival_cleanup -s <scene> -m <run> -i images_4 --eval --out <dir>
"""

import json
import shutil
from argparse import ArgumentParser
from pathlib import Path

import torch

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from scene.triangle_model import TriangleModel
from sota.survival import budget_matched_keep, rule_disagreement, v1_keep
from triangle_renderer import render
from utils.general_utils import safe_state


def _load(dataset):
    triangles = TriangleModel(dataset.sh_degree)
    scene = Scene(dataset, triangles, init_opacity=None, set_sigma=None,
                  load_iteration="precleanup", shuffle=False)
    return scene, triangles


@torch.no_grad()
def survival_scores(scene, triangles, pipeline, cleanup_scaling):
    """Per-face max and integral of alpha*T over every training view."""
    triangles.scaling = cleanup_scaling
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    faces = triangles.get_triangle_indices.shape[0]
    peak = torch.zeros(faces, dtype=torch.float32, device="cuda")
    integral = torch.zeros(faces, dtype=torch.float32, device="cuda")
    for camera in scene.getTrainCameras():
        package = render(camera, triangles, pipeline, background, integrated_blending=integral)
        peak = torch.maximum(peak, package["max_blending"].detach())
    return peak, integral


def _prune_and_save(triangles, keep, destination, iteration, extra=None):
    """The same two calls train.py makes after its cleanup mask."""
    triangles.prune_triangles(keep)
    used = torch.zeros(triangles.vertices.shape[0], dtype=torch.bool, device=keep.device)
    used[triangles._triangle_indices.flatten().long()] = True
    triangles._prune_vertices(used)
    path = destination / "point_cloud" / f"iteration_{iteration}"
    triangles.save_parameters(str(path))
    if extra:
        state_path = path / "point_cloud_state_dict.pt"
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        state.update(extra)
        torch.save(state, state_path)
    return int(triangles.get_triangle_indices.shape[0])


def run(dataset, pipeline, out, budget=None):
    run_dir = Path(dataset.model_path)
    precleanup = run_dir / "point_cloud" / "iteration_precleanup"
    meta = json.loads((precleanup / "cleanup.json").read_text(encoding="utf-8"))
    iteration = meta["final_iteration"]

    scene, triangles = _load(dataset)
    peak, integral = survival_scores(scene, triangles, pipeline, meta["cleanup_scaling"])
    keep_v1 = v1_keep(peak)
    keep_oats, stats = budget_matched_keep(integral, int(keep_v1.sum()), return_stats=True)

    arms = ["v1", "oats"] + (["matched", "matched_oats"] if budget else [])
    out.mkdir(parents=True, exist_ok=False)
    for arm in arms:
        (out / arm).mkdir()
        for name in ("cfg_args", "cameras.json"):
            # Contents only: the NAS refuses to set timestamps (copy2 fails).
            shutil.copyfile(run_dir / name, out / arm / name)

    faces_v1 = _prune_and_save(triangles, keep_v1, out / "v1", iteration)
    faces_v1_indices = triangles._triangle_indices.detach().cpu()
    _, triangles = _load(dataset)
    faces_oats = _prune_and_save(
        triangles, keep_oats, out / "oats", iteration,
        extra={"face_survival_score": integral[keep_oats].cpu()})

    # Neutral point: the v1 rule applied here must be the cleanup training ran.
    trained = torch.load(run_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud_state_dict.pt",
                         map_location="cpu", weights_only=False)
    faces_trained = int(trained["_triangle_indices"].shape[0])
    if not torch.equal(trained["_triangle_indices"].cpu().int(), faces_v1_indices.int()):
        raise RuntimeError(
            f"offline v1 cleanup kept {faces_v1} faces, training kept {faces_trained}, "
            "or the same count with different faces")

    faces_matched = faces_matched_oats = None
    if budget:
        # Same statistic as the published cleanup, cut to the given count: only
        # the mesh size differs from the v1 copy above.
        keep_matched = budget_matched_keep(peak, int(budget))
        _, triangles = _load(dataset)
        faces_matched = _prune_and_save(triangles, keep_matched, out / "matched", iteration)
        # The integral at the same count: the full method, sized like the control.
        keep_matched_oats = budget_matched_keep(integral, int(budget))
        _, triangles = _load(dataset)
        faces_matched_oats = _prune_and_save(
            triangles, keep_matched_oats, out / "matched_oats", iteration,
            extra={"face_survival_score": integral[keep_matched_oats].cpu()})

    report = {
        "run": str(run_dir),
        "cleanup_scaling": meta["cleanup_scaling"],
        "faces_before": int(peak.numel()),
        "faces_kept": ({"v1": faces_v1, "oats": faces_oats, "trained": faces_trained}
                       | ({"matched": faces_matched, "matched_oats": faces_matched_oats}
                          if faces_matched is not None else {})),
        "disagreement": rule_disagreement(keep_v1, keep_oats),
        "kept_with_zero_score": stats["kept_with_zero_score"],
        "integral": {"v1_kept_mean": float(integral[keep_v1].mean()),
                     "oats_kept_mean": float(integral[keep_oats].mean())},
    }
    (out / "survival.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--out", required=True)
    parser.add_argument("--budget", type=int, default=None,
                        help="also write <out>/matched and <out>/matched_oats at this face count")
    parser.add_argument("--quiet", action="store_true")
    parsed = get_combined_args(parser)
    safe_state(parsed.quiet)
    # get_combined_args drops every argument whose value is None, so an
    # omitted --budget leaves no attribute at all.
    run(model.extract(parsed), pipeline.extract(parsed), Path(parsed.out),
        getattr(parsed, "budget", None))
