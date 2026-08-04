"""The locked SAC-G0 decision, separable from the CUDA evaluation run."""

import json
from argparse import ArgumentParser
from pathlib import Path


# The garden_baseline checkpoint under this evaluation path, as recorded by the
# RITS-T0 step-0 evaluations (experiments/rits_t0/analysis_full_01.md).
BASELINE = {"psnr": 24.7372, "ssim": 0.7484, "lpips_vgg": 0.2480}
VALIDITY_TOLERANCE = {"psnr": 0.10, "ssim": 0.010, "lpips_vgg": 0.010}
MAX_PSNR_COST = 0.35
MAX_LPIPS_COST = 0.020
MIN_RENDER_SPEEDUP = 2.5


def decide(stock, splat2):
    """`stock` and `splat2` are one arm's results.json contents."""
    stock_at_4 = stock["cells"]["scaling_4"]
    splat2_at_2 = splat2["cells"]["scaling_2"]
    render_speedup = (
        stock["cells"]["scaling_4"]["render_ms_per_view"]
        / stock["cells"]["scaling_2"]["render_ms_per_view"]
    )
    checks = {
        "stock_reproduces_the_baseline_checkpoint": all(
            abs(stock_at_4[key] - BASELINE[key]) <= tolerance
            for key, tolerance in VALIDITY_TOLERANCE.items()
        ),
        "psnr_cost_at_most_0p35": stock_at_4["psnr"] - splat2_at_2["psnr"] <= MAX_PSNR_COST,
        "lpips_cost_at_most_0p020": (
            splat2_at_2["lpips_vgg"] - stock_at_4["lpips_vgg"] <= MAX_LPIPS_COST
        ),
        "render_speedup_at_least_2p5x": render_speedup >= MIN_RENDER_SPEEDUP,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "psnr_cost": stock_at_4["psnr"] - splat2_at_2["psnr"],
        "lpips_cost": splat2_at_2["lpips_vgg"] - stock_at_4["lpips_vgg"],
        "render_speedup": render_speedup,
        "cells": {
            "stock@4": stock_at_4,
            "stock@2": stock["cells"]["scaling_2"],
            "splat2@2": splat2_at_2,
            "splat2@4": splat2["cells"]["scaling_4"],
        },
        "training_wall_clock_ratio": (
            splat2["training_seconds"] / stock["training_seconds"]
            if stock.get("training_seconds") and splat2.get("training_seconds")
            else None
        ),
        "final_primitives": {
            "stock": stock["primitives"],
            "splat2": splat2["primitives"],
        },
    }


def _load(directory):
    with open(Path(directory) / "results.json", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    parser = ArgumentParser(description="SAC-G0 locked decision")
    parser.add_argument("--stock", required=True)
    parser.add_argument("--splat2", required=True)
    args = parser.parse_args()
    print(json.dumps(decide(_load(args.stock), _load(args.splat2)), indent=2))
