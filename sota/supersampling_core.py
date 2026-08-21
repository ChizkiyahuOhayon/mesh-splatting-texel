"""Pure selection rule for the frozen supersampling experiment."""

import math


FACTORS = (4, 3, 2, 1)
PSNR_TOLERANCE_DB = 0.05
MINIMUM_FPS_MULTIPLIER = 1.5


def choose_factor(baseline_psnr, measurements):
    if not math.isfinite(float(baseline_psnr)):
        raise ValueError("baseline_psnr must be finite")
    if set(measurements) != set(FACTORS):
        raise ValueError("measurements do not match the locked factor grid")
    for row in measurements.values():
        if (set(row) != {"psnr", "render_ms"}
                or not all(math.isfinite(float(value)) for value in row.values())
                or float(row["render_ms"]) <= 0.0):
            raise ValueError("each measurement needs finite psnr and positive render_ms")
    eligible = [
        factor for factor in FACTORS
        if float(measurements[factor]["psnr"])
        >= float(baseline_psnr) - PSNR_TOLERANCE_DB
    ]
    if not eligible:
        raise ValueError("no supersampling factor meets the quality tolerance")
    return min(
        eligible,
        key=lambda factor: (float(measurements[factor]["render_ms"]), -factor),
    )


def passes_test_gate(baseline, selected):
    return (
        float(selected["psnr"]) >= float(baseline["psnr"]) - PSNR_TOLERANCE_DB
        and float(selected["fps"]) >= float(baseline["fps"]) * MINIMUM_FPS_MULTIPLIER
    )
