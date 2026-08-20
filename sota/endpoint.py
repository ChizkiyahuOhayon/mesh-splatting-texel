"""Endpoint-valued supervision with the baseline soft rasterizer gradient."""

import torch


def endpoint_image(soft: torch.Tensor, hard: torch.Tensor) -> torch.Tensor:
    """Return ``hard`` in the forward pass and differentiate through ``soft``."""
    if soft.shape != hard.shape:
        raise ValueError(f"render shapes differ: {tuple(soft.shape)} vs {tuple(hard.shape)}")
    return soft + (hard - soft).detach()


def opacity_at_floor(raw: torch.Tensor, floor: float) -> torch.Tensor:
    """Evaluate the model's opacity parameterization at a chosen floor."""
    if not 0.0 <= floor < 1.0:
        raise ValueError("opacity floor must be in [0, 1)")
    return floor + (1.0 - floor) * torch.sigmoid(raw)
