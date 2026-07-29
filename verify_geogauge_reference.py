"""Deterministic numerical checks for the GeoGauge exact reference."""

from __future__ import annotations

import torch

from geogauge_reference import (DTYPE, camera_grid, exact_jacobians,
                                marginalized_information, perturb_and_refit,
                                pixel_grid, point_grid, render, sh_basis)
from utils.sh_utils import C0


def _fixture():
    points = point_grid()
    cameras, directions = camera_grid(0.25)
    pixels = pixel_grid()
    geometry = torch.linspace(0.82, 1.18, len(points), dtype=DTYPE)
    appearance = torch.zeros(len(points), sh_basis(directions, 1).shape[1], dtype=DTYPE)
    appearance[:, 0] = torch.linspace(0.35, 0.65, len(points), dtype=DTYPE) / C0
    appearance[:, 1:] = 0.015
    return points, cameras, pixels, geometry, appearance


def check_jacobians():
    points, cameras, pixels, geometry, appearance = _fixture()
    jg, ja = exact_jacobians(geometry, appearance, cameras, pixels, points, 1)
    scale = jg.shape[0] ** 0.5
    eps = 1e-6
    geometry_numeric = []
    for i in range(len(geometry)):
        step = torch.zeros_like(geometry)
        step[i] = eps
        geometry_numeric.append((render(geometry + step, appearance, cameras, pixels,
                                        points, 1)
                                 - render(geometry - step, appearance, cameras, pixels,
                                          points, 1)) / (2 * eps * scale))
    geometry_numeric = torch.stack(geometry_numeric, dim=1)

    appearance_numeric = []
    for flat_i in (0, appearance.numel() // 2, appearance.numel() - 1):
        step = torch.zeros_like(appearance).reshape(-1)
        step[flat_i] = eps
        step = step.reshape_as(appearance)
        appearance_numeric.append((render(geometry, appearance + step, cameras, pixels,
                                          points, 1)
                                   - render(geometry, appearance - step, cameras, pixels,
                                            points, 1)) / (2 * eps * scale))
    appearance_numeric = torch.stack(appearance_numeric, dim=1)
    selected = ja[:, [0, appearance.numel() // 2, appearance.numel() - 1]]
    error_g = float((jg - geometry_numeric).abs().max())
    error_a = float((selected - appearance_numeric).abs().max())
    assert error_g < 1e-8, error_g
    assert error_a < 1e-9, error_a
    return error_g, error_a


def check_schur():
    points, cameras, pixels, geometry, appearance = _fixture()
    jg, ja = exact_jacobians(geometry, appearance, cameras, pixels, points, 1)
    schur, values = marginalized_information(jg, ja, 1e-4)
    symmetry = float((schur - schur.T).abs().max())
    minimum_eigenvalue = float(torch.linalg.eigvalsh(schur).min())
    ratios = values["identifiable_fraction"]
    assert symmetry < 1e-12, symmetry
    assert minimum_eigenvalue > -1e-10, minimum_eigenvalue
    assert bool(((ratios >= -1e-10) & (ratios <= 1.0 + 1e-10)).all()), ratios
    return symmetry, minimum_eigenvalue


def check_refit_ordering():
    points, cameras, pixels, geometry, appearance = _fixture()
    perturbation = torch.linspace(-1.0, 1.0, len(geometry), dtype=DTYPE)
    perturbation = 0.03 * perturbation / torch.linalg.vector_norm(perturbation)
    low_capacity = perturb_and_refit(geometry, appearance[:, :1], perturbation,
                                     cameras, pixels, points, 0)
    high_appearance = torch.zeros(len(points), 16, dtype=DTYPE)
    high_appearance[:, :appearance.shape[1]] = appearance
    high_capacity = perturb_and_refit(geometry, high_appearance, perturbation,
                                      cameras, pixels, points, 3)
    assert float(high_capacity) <= float(low_capacity) + 1e-8, (low_capacity, high_capacity)
    return float(low_capacity), float(high_capacity)


if __name__ == "__main__":
    error_g, error_a = check_jacobians()
    symmetry, minimum_eigenvalue = check_schur()
    low, high = check_refit_ordering()
    print(f"geometry finite-difference max error: {error_g:.3e}")
    print(f"appearance finite-difference max error: {error_a:.3e}")
    print(f"Schur symmetry error / min eigenvalue: {symmetry:.3e} / {minimum_eigenvalue:.3e}")
    print(f"unrecoverable fraction degree0 / degree3: {low:.4f} / {high:.4f}")
    print("PASS")
