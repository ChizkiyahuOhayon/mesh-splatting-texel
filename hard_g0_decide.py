#!/usr/bin/env python3
"""Apply the preregistered HARD-G0 decision rule."""

import argparse
import json
import math
from pathlib import Path


TARGET_SIGMA = 1e-4
SIGMA_REL_TOL = 1e-5
SIGMA_ABS_TOL = 1e-8
MIN_PSNR_GAIN = 0.10


def _scaling4(record, expected_arm):
    required = ("scene", "seed", "arm", "sigma", "cells")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"{expected_arm} record is missing keys: {missing}")

    cell = record["cells"].get("scaling_4")
    if cell is None:
        raise ValueError(f"{expected_arm} record has no scaling_4 cell")

    values = {
        "sigma": float(record["sigma"]),
        "psnr": float(cell["psnr"]),
        "ssim": float(cell["ssim"]),
        "lpips_vgg": float(cell["lpips_vgg"]),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f"{expected_arm} record contains a non-finite value")
    return values


def decide(stock, early):
    """Return the locked HARD-G0 decision for two SAC evaluation records."""
    stock4 = _scaling4(stock, "stock")
    early4 = _scaling4(early, "early")

    psnr_gain = early4["psnr"] - stock4["psnr"]
    ssim_delta = early4["ssim"] - stock4["ssim"]
    lpips_delta = early4["lpips_vgg"] - stock4["lpips_vgg"]

    checks = {
        "record_identity": (
            stock["scene"] == early["scene"] == "garden"
            and int(stock["seed"]) == int(early["seed"]) == 0
            and stock["arm"] == "stock"
            and early["arm"] == "early"
        ),
        "stock_endpoint_sigma_is_1e-4": math.isclose(
            stock4["sigma"], TARGET_SIGMA,
            rel_tol=SIGMA_REL_TOL, abs_tol=SIGMA_ABS_TOL,
        ),
        "early_endpoint_sigma_is_1e-4": math.isclose(
            early4["sigma"], TARGET_SIGMA,
            rel_tol=SIGMA_REL_TOL, abs_tol=SIGMA_ABS_TOL,
        ),
        "early_psnr_gain_at_least_0p10_db": psnr_gain >= MIN_PSNR_GAIN,
        "early_lpips_is_nonworse": lpips_delta <= 0.0,
    }

    return {
        "experiment": "HARD-G0",
        "decision": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "deltas_early_minus_stock_at_scaling_4": {
            "psnr_db": psnr_gain,
            "ssim": ssim_delta,
            "lpips": lpips_delta,
        },
        "stock_at_scaling_4": stock4,
        "early_at_scaling_4": early4,
    }


def _load_results(path_string):
    path = Path(path_string)
    if path.is_dir():
        path = path / "results.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock", required=True,
                        help="Stock eval directory or results.json")
    parser.add_argument("--early", required=True,
                        help="Early eval directory or results.json")
    parser.add_argument("--output", required=True,
                        help="New output JSON; existing files are never overwritten")
    args = parser.parse_args()

    result = decide(_load_results(args.stock), _load_results(args.early))
    serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()
