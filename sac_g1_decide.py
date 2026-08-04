"""The locked SAC-G1 decision: a paired replication across scenes and seeds.

Both arms of a pair share a seed, so their difference is paired and its
uncertainty is the standard error of those differences — the quantity this
pipeline has never measured and against which every earlier single-seed result
should be read.
"""

import json
import statistics
from argparse import ArgumentParser
from pathlib import Path


# Recorded baselines under this evaluation path (RITS-T0 step-0 evaluations).
BASELINES = {"garden": 24.7372, "room": 28.5142}
VALIDITY_TOLERANCE = 0.15
SIGMA_MULTIPLE = 2.0


def _spread(values):
    """Sample standard deviation, or None when one run cannot show scatter."""
    return statistics.stdev(values) if len(values) >= 2 else None


def _standard_error(values):
    """Infinite for a single value, so an unreplicated effect can never pass."""
    if len(values) < 2:
        return float("inf")
    return statistics.stdev(values) / len(values) ** 0.5


def _arm_values(runs, scene, arm, key, cell="scaling_4"):
    """One scene and arm's per-seed values of a metric, or of a primitive count."""
    return [
        run["primitives"][key] if key == "triangles" else run["cells"][cell][key]
        for (run_scene, run_arm, _), run in runs.items()
        if run_scene == scene and run_arm == arm
    ]


def _paired(runs, scene, metric, cell="scaling_4"):
    """Per-seed splat2-minus-stock differences for one scene and metric."""
    seeds = sorted(
        {seed for (run_scene, _, seed) in runs if run_scene == scene},
        key=int,
    )
    differences = []
    for seed in seeds:
        try:
            stock = runs[(scene, "stock", seed)]["cells"][cell][metric]
            splat2 = runs[(scene, "splat2", seed)]["cells"][cell][metric]
        except KeyError:
            continue
        differences.append(splat2 - stock)
    return differences


def decide(runs):
    """`runs` maps (scene, arm, seed) to that run's results.json contents."""
    scenes = sorted({scene for (scene, _, _) in runs})
    psnr_differences = {scene: _paired(runs, scene, "psnr") for scene in scenes}
    lpips_differences = {
        scene: _paired(runs, scene, "lpips_vgg") for scene in scenes
    }
    stock_means = {
        scene: statistics.fmean(_arm_values(runs, scene, "stock", "psnr"))
        for scene in scenes
    }
    pooled = [value for scene in scenes for value in psnr_differences[scene]]
    mean_difference = statistics.fmean(pooled) if pooled else 0.0
    lower_bound = mean_difference - SIGMA_MULTIPLE * _standard_error(pooled)

    checks = {
        "stock_reproduces_each_baseline": all(
            scene in BASELINES
            and abs(stock_means[scene] - BASELINES[scene]) <= VALIDITY_TOLERANCE
            for scene in scenes
        ),
        "both_scenes_have_paired_seeds": all(
            len(psnr_differences[scene]) >= 2 for scene in scenes
        ),
        "mean_psnr_difference_positive_on_both_scenes": all(
            psnr_differences[scene] and statistics.fmean(psnr_differences[scene]) > 0
            for scene in scenes
        ),
        "effect_exceeds_twice_its_standard_error": lower_bound > 0,
        "lpips_not_worse_on_either_scene": all(
            lpips_differences[scene]
            and statistics.fmean(lpips_differences[scene]) <= 0
            for scene in scenes
        ),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "mean_psnr_difference": mean_difference,
        "standard_error": _standard_error(pooled),
        "lower_bound_at_2_se": lower_bound,
        "per_scene": {
            scene: {
                "stock_mean_psnr": stock_means[scene],
                "baseline_psnr": BASELINES.get(scene),
                "psnr_differences": psnr_differences[scene],
                "mean_psnr_difference": (
                    statistics.fmean(psnr_differences[scene])
                    if psnr_differences[scene]
                    else None
                ),
                "seed_standard_deviation": {
                    arm: _spread(_arm_values(runs, scene, arm, "psnr"))
                    for arm in ("stock", "splat2")
                },
                "mean_lpips_difference": (
                    statistics.fmean(lpips_differences[scene])
                    if lpips_differences[scene]
                    else None
                ),
                "mean_triangles": {
                    arm: statistics.fmean(_arm_values(runs, scene, arm, "triangles"))
                    for arm in ("stock", "splat2")
                },
            }
            for scene in scenes
        },
    }


def _load(directories):
    runs = {}
    for directory in directories:
        with open(Path(directory) / "results.json", encoding="utf-8") as handle:
            results = json.load(handle)
        key = (results["scene"], results["arm"], results["seed"])
        if key in runs:
            raise ValueError(f"duplicate run for {key}: {directory}")
        runs[key] = results
    return runs


if __name__ == "__main__":
    parser = ArgumentParser(description="SAC-G1 paired replication decision")
    parser.add_argument("results", nargs="+", help="one evaluation output directory per run")
    args = parser.parse_args()
    print(json.dumps(decide(_load(args.results)), indent=2))
