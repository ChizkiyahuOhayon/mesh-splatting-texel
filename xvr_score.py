"""Pure scoring utilities for the XVR-G0 densification diagnostic."""

import math

import torch


PRIMARY_SIGNAL = "persistent_error_mass"
RAW_CONTROL = "raw_error_mass"
NON_RESIDUAL_CONTROLS = ("max_blending", "projected_coverage", "world_area")


def persistent_error_mass(min_view_error, pixel_count, view_count):
    """Conservative per-face error mass: persistent error times mean visible area."""
    mean_visible_pixels = pixel_count / view_count.clamp_min(1)
    return min_view_error * mean_visible_pixels


def top_fraction_capture(scores, target_mass, eligible, fraction):
    """Measure how much target error mass is captured by top-scoring faces."""
    if scores.ndim != 1 or target_mass.shape != scores.shape or eligible.shape != scores.shape:
        raise ValueError("scores, target_mass, and eligible must be aligned 1-D tensors")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")

    valid = eligible & torch.isfinite(scores) & torch.isfinite(target_mass) & (target_mass >= 0)
    indices = torch.nonzero(valid, as_tuple=True)[0]
    if indices.numel() == 0:
        raise ValueError("no eligible faces")

    selected_count = max(1, int(math.ceil(float(indices.numel()) * fraction)))
    local = torch.topk(scores[indices], selected_count, largest=True, sorted=False).indices
    selected = indices[local]
    total_mass = target_mass[indices].sum()
    if total_mass <= 0:
        raise ValueError("eligible target error mass must be positive")

    actual_fraction = selected_count / float(indices.numel())
    capture = target_mass[selected].sum() / total_mass
    return {
        "eligible_faces": int(indices.numel()),
        "selected_faces": int(selected_count),
        "actual_fraction": actual_fraction,
        "capture": float(capture),
        "lift": float(capture) / actual_fraction,
    }


def scene_gate(signal_metrics):
    """Apply the preregistered XVR-G0 scene-level decision at the top 10%."""
    primary = signal_metrics[PRIMARY_SIGNAL]["top_10pct"]
    raw = signal_metrics[RAW_CONTROL]["top_10pct"]
    best_non_residual = max(
        signal_metrics[name]["top_10pct"]["capture"]
        for name in NON_RESIDUAL_CONTROLS
    )

    checks = {
        "eligible_faces_at_least_10000": primary["eligible_faces"] >= 10_000,
        "capture_lift_at_least_1_75x": primary["lift"] >= 1.75,
        "beats_best_non_residual_by_10pct": primary["capture"] >= 1.10 * best_non_residual,
        "within_5pct_of_raw_residual": primary["capture"] >= 0.95 * raw["capture"],
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "primary_top_10pct": primary,
        "raw_capture": raw["capture"],
        "best_non_residual_capture": best_non_residual,
    }
