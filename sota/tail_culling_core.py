"""Pure selection rule for the frozen transmittance-tail experiment."""

import math


THRESHOLDS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
DEFAULT_THRESHOLD = 1e-4
SELECTION_VIEWS = 32
SELECTION_PSNR_TOLERANCE_DB = 0.02
TEST_PSNR_TOLERANCE_DB = 0.03
MINIMUM_FPS_MULTIPLIER = 1.25


def choose_threshold(measurements):
    """Choose the fastest threshold inside the locked train-PSNR tolerance."""
    if set(measurements) != set(THRESHOLDS):
        raise ValueError("measurements do not match the locked threshold grid")
    for row in measurements.values():
        if (set(row) != {"psnr", "render_ms"}
                or not all(math.isfinite(float(value)) for value in row.values())
                or float(row["render_ms"]) <= 0.0):
            raise ValueError("each measurement needs finite psnr and positive render_ms")
    minimum_psnr = (
        float(measurements[DEFAULT_THRESHOLD]["psnr"])
        - SELECTION_PSNR_TOLERANCE_DB
    )
    eligible = [
        threshold for threshold in THRESHOLDS
        if float(measurements[threshold]["psnr"]) >= minimum_psnr
    ]
    return min(
        eligible,
        key=lambda threshold: (float(measurements[threshold]["render_ms"]), threshold),
    )


def passes_test_gate(baseline, selected):
    return (
        float(selected["psnr"]) >= float(baseline["psnr"]) - TEST_PSNR_TOLERANCE_DB
        and float(selected["fps"]) >= float(baseline["fps"]) * MINIMUM_FPS_MULTIPLIER
    )
