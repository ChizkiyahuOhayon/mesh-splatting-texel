"""Deterministic numerical checks for ``deform_reference.py``."""

from __future__ import annotations

import math

import torch

from deform_reference import (canonical_grid, control_polygon, deformable_alpha,
                              physical_control_polygon)


def check_zero_equivalence():
    points, _ = canonical_grid(51, dtype=torch.float64)
    sigma, opacity = 0.3, 0.9
    got = deformable_alpha(points, torch.zeros(3, dtype=torch.float64), sigma, opacity)
    u, v = points[:, 0], points[:, 1]
    inside = (u >= 0) & (v >= 0) & (u + v <= 1)
    distance = torch.minimum(torch.minimum(u, v), (1 - u - v) / math.sqrt(2.0))
    distance = torch.where(distance.abs() < 1e-10, torch.zeros_like(distance), distance)
    inradius = 1.0 / (2.0 + math.sqrt(2.0))
    expected = opacity * (distance / inradius).clamp(0, 1).pow(sigma) * inside
    expected = expected.clamp_max(0.99)
    error = float((got - expected).abs().max())
    assert error < 1e-10, error
    return error


def check_displacement_gradient():
    points = torch.tensor([[0.42, 0.08], [0.52, 0.12], [0.58, 0.18]], dtype=torch.float64)
    weights = torch.tensor([0.7, -0.2, 0.4], dtype=torch.float64)
    d = torch.tensor([0.03, -0.02, 0.01], dtype=torch.float64, requires_grad=True)

    def objective(x):
        return (deformable_alpha(points, x, sigma=0.4, opacity=0.8) * weights).sum()

    objective(d).backward()
    analytic = d.grad.detach().clone()
    eps = 1e-6
    numeric = torch.stack([(objective(d.detach() + eps * torch.eye(3, dtype=d.dtype)[i])
                            - objective(d.detach() - eps * torch.eye(3, dtype=d.dtype)[i]))
                           / (2 * eps) for i in range(3)])
    error = float((analytic - numeric).abs().max())
    assert error < 2e-6, (analytic, numeric, error)
    return error


def check_physical_lift():
    triangle = torch.tensor([[0.2, -0.1, 1.0], [1.4, 0.2, 1.3],
                             [-0.3, 1.1, 0.7]], dtype=torch.float64)
    d = torch.tensor([0.1, -0.08, 0.04], dtype=torch.float64)
    uv = control_polygon(d)
    xyz = physical_control_polygon(triangle, d)
    expected = ((1 - uv[:, :1] - uv[:, 1:]) * triangle[0]
                + uv[:, :1] * triangle[1] + uv[:, 1:] * triangle[2])
    error = float((xyz - expected).abs().max())
    assert error < 1e-12, error
    return error


if __name__ == "__main__":
    print(f"d=0 parity max error: {check_zero_equivalence():.3e}")
    print(f"d finite-difference max error: {check_displacement_gradient():.3e}")
    print(f"physical lift max error: {check_physical_lift():.3e}")
    print("PASS")
