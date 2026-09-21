"""Dump the two survival statistics per face, for the analysis figure.

The method rests on one empirical claim: on a connected mesh the published peak
statistic ``max_blending`` and the integral ``S_f = sum(alpha * T)`` rank faces
differently, and the faces they disagree about are not a rounding error. This
script measures both on a pre-cleanup run - the same pass
``sota/survival_cleanup.py`` performs - and writes a sample of the joint
distribution, so the figure is drawn from measurements rather than from an
illustration.

Writes ``<out>/statistic.npz`` (sampled peak, integral, and both keep masks) and
``<out>/statistic.json`` (population-level summary over *all* faces).

Usage:
    python -m sota.survival_statistic -s <scene> -m <run> -i images_4 --eval --out <dir>
"""

import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from arguments import ModelParams, PipelineParams, get_combined_args
from sota.survival import budget_matched_keep, rule_disagreement, v1_keep
from sota.survival_cleanup import _load, survival_scores
from utils.general_utils import safe_state

SAMPLE = 200_000


def run(dataset, pipeline, out, seed=0):
    run_dir = Path(dataset.model_path)
    meta = json.loads((run_dir / "point_cloud" / "iteration_precleanup" / "cleanup.json")
                      .read_text(encoding="utf-8"))

    scene, triangles = _load(dataset)
    peak, integral = survival_scores(scene, triangles, pipeline, meta["cleanup_scaling"])
    keep_v1 = v1_keep(peak)
    keep_oats = budget_matched_keep(integral, int(keep_v1.sum()))

    faces = int(peak.numel())
    generator = torch.Generator(device="cuda").manual_seed(seed)
    index = (torch.randperm(faces, generator=generator, device="cuda")[:SAMPLE]
             if faces > SAMPLE else torch.arange(faces, device="cuda"))

    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "statistic.npz",
        peak=peak[index].cpu().numpy(),
        integral=integral[index].cpu().numpy(),
        keep_v1=keep_v1[index].cpu().numpy(),
        keep_oats=keep_oats[index].cpu().numpy(),
        faces=np.int64(faces),
        sampled=np.int64(int(index.numel())),
    )

    # The disagreement is the point of the figure, so describe both sides of it
    # over the whole population, not over the sample.
    only_v1 = keep_v1 & ~keep_oats
    only_oats = keep_oats & ~keep_v1
    finite = peak > 0

    def describe(mask):
        if not bool(mask.any()):
            return None
        return {
            "faces": int(mask.sum()),
            "peak_mean": float(peak[mask].mean()),
            "integral_mean": float(integral[mask].mean()),
            "integral_median": float(integral[mask].median()),
        }

    report = {
        "run": str(run_dir),
        "faces": faces,
        "faces_kept": {"v1": int(keep_v1.sum()), "oats": int(keep_oats.sum())},
        "disagreement": rule_disagreement(keep_v1, keep_oats),
        "kept_only_by_peak": describe(only_v1),
        "kept_only_by_integral": describe(only_oats),
        "kept_by_both": describe(keep_v1 & keep_oats),
        "spearman_peak_vs_integral": float(_spearman(peak[finite], integral[finite])),
        "sampled": int(index.numel()),
    }
    (out / "statistic.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def _spearman(a, b):
    """Rank correlation of the two statistics over the faces that were seen."""
    def ranks(values):
        order = torch.argsort(values)
        result = torch.empty_like(values)
        result[order] = torch.arange(values.numel(), dtype=values.dtype, device=values.device)
        return result

    ra, rb = ranks(a.double()), ranks(b.double())
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    return (ra @ rb) / (ra.norm() * rb.norm())


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--out", required=True)
    parser.add_argument("--quiet", action="store_true")
    parsed = get_combined_args(parser)
    safe_state(parsed.quiet)
    run(model.extract(parsed), pipeline.extract(parsed), Path(parsed.out))
