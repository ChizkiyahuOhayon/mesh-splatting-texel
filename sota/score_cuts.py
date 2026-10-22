"""Cut one pre-cleanup model to a face budget under several importance scores.

The compact frontier (scripts/compact.sh) compares the peak rule with the
integral at equal size. This tool adds the importance scores used by Gaussian
pruning methods, all computed in one pass over the training views of the same
trained model, and writes one evaluable model per score and budget:

    peak         max over views and pixels of alpha*T  (MeshSplatting, RadSplat)
    integral     sum over views and pixels of alpha*T  (ours)
    opacity      face opacity                          (3DGS opacity pruning)
    hits         pixels with alpha >= 1/255, all views (visibility count)
    lightgaussian hits * opacity * min(area / area_q90, 1) ** 0.1  (LightGaussian,
                 volume replaced by triangle area)

Budgets are fractions of the face count the published peak rule keeps, so every
score at one budget yields a mesh of exactly the same size; only which faces
survive differs. Output: ``<out>/b<pct>/<score>/point_cloud/iteration_<final>/``
and ``<out>/scores.json``.

Usage:
    python -m sota.score_cuts -s <scene> -m <run> -i images_4 --eval --out <dir> \
        --fractions 10 25 50 100
"""

import json
import shutil
from argparse import ArgumentParser
from pathlib import Path

import torch

from arguments import ModelParams, PipelineParams, get_combined_args
from sota.survival import budget_matched_keep, v1_keep
from sota.survival_cleanup import _load, _prune_and_save
from triangle_renderer import render
from utils.general_utils import safe_state

SCORES = ("peak", "integral", "opacity", "hits", "lightgaussian")


@torch.no_grad()
def importance_scores(scene, triangles, pipeline, cleanup_scaling):
    triangles.scaling = cleanup_scaling
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    faces = triangles.get_triangle_indices.shape[0]
    peak = torch.zeros(faces, dtype=torch.float32, device="cuda")
    integral = torch.zeros(faces, dtype=torch.float32, device="cuda")
    hits = torch.zeros(faces, dtype=torch.float64, device="cuda")
    for camera in scene.getTrainCameras():
        package = render(camera, triangles, pipeline, background, integrated_blending=integral)
        peak = torch.maximum(peak, package["max_blending"].detach())
        hits += package["triangle_was_rendered"][:faces].double()
    opacity = triangles.face_opacity().detach().float()
    corners = triangles.vertices[triangles.get_triangle_indices.long()]
    area = 0.5 * torch.linalg.cross(corners[:, 1] - corners[:, 0],
                                    corners[:, 2] - corners[:, 0]).norm(dim=-1)
    area_q90 = torch.quantile(area.float(), 0.9)
    lightgaussian = hits.float() * opacity * torch.clamp(area / area_q90, 0, 1) ** 0.1
    return {"peak": peak, "integral": integral, "opacity": opacity,
            "hits": hits.float(), "lightgaussian": lightgaussian}


def run(dataset, pipeline, out, fractions):
    run_dir = Path(dataset.model_path)
    meta = json.loads((run_dir / "point_cloud" / "iteration_precleanup" / "cleanup.json")
                      .read_text(encoding="utf-8"))
    iteration = meta["final_iteration"]
    scene, triangles = _load(dataset)
    scores = importance_scores(scene, triangles, pipeline, meta["cleanup_scaling"])
    published = int(v1_keep(scores["peak"]).sum())

    out.mkdir(parents=True, exist_ok=True)
    report = {"run": str(run_dir), "faces_before": int(scores["peak"].numel()),
              "published_faces": published, "cuts": {}}
    for fraction in fractions:
        budget = published * fraction // 100
        keeps = {name: budget_matched_keep(scores[name], budget) for name in SCORES}
        report["cuts"][f"b{fraction}"] = {
            "budget": budget,
            # Share of the integral's survivors each score also keeps.
            "overlap_with_integral": {name: float((keeps[name] & keeps["integral"]).sum()) / max(budget, 1)
                                      for name in SCORES},
        }
        for name in SCORES:
            destination = out / f"b{fraction}" / name
            if (destination / "point_cloud").is_dir():
                continue
            destination.mkdir(parents=True, exist_ok=True)
            for item in ("cfg_args", "cameras.json"):
                shutil.copyfile(run_dir / item, destination / item)
            _, triangles = _load(dataset)
            kept = _prune_and_save(triangles, keeps[name], destination, iteration)
            if kept != budget:
                raise RuntimeError(f"{name} b{fraction}: kept {kept}, budget {budget}")
    (out / "scores.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fractions", type=int, nargs="+", default=[10, 25, 50, 100])
    parser.add_argument("--quiet", action="store_true")
    parsed = get_combined_args(parser)
    safe_state(parsed.quiet)
    run(model.extract(parsed), pipeline.extract(parsed), Path(parsed.out), parsed.fractions)
