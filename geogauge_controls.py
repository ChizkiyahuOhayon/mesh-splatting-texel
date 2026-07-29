"""Controlled fits for the GeoGauge G0 exact-reference gate."""

from __future__ import annotations

import math

from dataclasses import dataclass

import numpy as np
import torch

from geogauge_reference import (DTYPE, _render_terms, camera_grid, exact_jacobians,
                                marginalized_information, perturb_and_refit,
                                pixel_grid, point_grid, refit_appearance, render,
                                sh_basis)
from utils.sh_utils import C0


BASELINES = {"narrow": 0.08, "medium": 0.28, "wide": 0.55}
CAPACITIES = {"rgb": 0, "sh1": 1, "sh3": 3}
REGIMES = ("lambertian", "view-dependent")
DAMPINGS = (1e-6, 1e-4, 1e-2)
CENTRAL_DAMPING = 1e-4


@dataclass(frozen=True)
class FitConfig:
    steps: int = 220
    geometry_lr: float = 0.025
    appearance_lr: float = 0.04
    appearance_l2: float = 1e-5
    init_noise: float = 0.12


def _ground_truth(seed, points, cameras, regime):
    x, y = points.unbind(dim=1)
    geometry = 0.95 + 0.18 * torch.exp(-3.5 * (x.square() + y.square()))
    geometry += 0.045 * torch.sin(3.0 * x - 2.0 * y)

    base = 0.5 + 0.11 * torch.sin(3.5 * x + 1.7 * y + 0.3 * seed)
    appearance = torch.zeros(len(points), 16, dtype=DTYPE, device=points.device)
    appearance[:, 0] = base / C0
    if regime == "view-dependent":
        generator = torch.Generator(device=points.device).manual_seed(1000 + seed)
        random_coefficients = torch.randn(len(points), 15, generator=generator,
                                          dtype=DTYPE, device=points.device)
        band_scale = torch.tensor([0.070] * 3 + [0.045] * 5 + [0.030] * 7,
                                  dtype=DTYPE, device=points.device)
        appearance[:, 1:] = random_coefficients * band_scale
    elif regime != "lambertian":
        raise ValueError(f"unknown regime: {regime}")
    target = render(geometry, appearance, cameras, pixel_grid(device=points.device),
                    points, 3)
    return geometry, appearance, target


def _local_diagnostics(geometry, appearance, target, cameras, pixels, points, degree):
    image, weights, denominator, _, _, _ = _render_terms(
        geometry, appearance, cameras, pixels, points, degree)
    residual_sq = (image.reshape(-1) - target).square().reshape_as(image)
    contribution = weights / denominator[..., None]
    local_residual = torch.einsum("vpi,vp->i", contribution, residual_sq) \
        / contribution.sum(dim=(0, 1)).clamp_min(1e-15)
    coverage = contribution.mean(dim=(0, 1))
    return image.reshape(-1), local_residual, coverage


def fit_case(seed, baseline_name, capacity_name, regime, config=FitConfig(), device="cpu"):
    torch.manual_seed(seed)
    device = torch.device(device)
    points = point_grid(device=device)
    pixels = pixel_grid(device=device)
    baseline = BASELINES[baseline_name]
    degree = CAPACITIES[capacity_name]
    cameras, directions = camera_grid(baseline, device=device)
    geometry_gt, _, target = _ground_truth(seed, points, cameras, regime)

    generator = torch.Generator(device=device).manual_seed(seed)
    initial_geometry = geometry_gt + config.init_noise * torch.randn(
        len(points), generator=generator, dtype=DTYPE, device=device)
    n_basis = sh_basis(directions, degree).shape[1]
    appearance_prior = torch.zeros(len(points), n_basis, dtype=DTYPE, device=device)
    initial_appearance = refit_appearance(
        initial_geometry, target, appearance_prior, cameras, pixels, points, degree,
        damping=1e-3)

    geometry = torch.nn.Parameter(initial_geometry.clone())
    appearance = torch.nn.Parameter(initial_appearance.clone())
    optimizer = torch.optim.Adam([
        {"params": [geometry], "lr": config.geometry_lr},
        {"params": [appearance], "lr": config.appearance_lr},
    ])
    for _ in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = render(geometry, appearance, cameras, pixels, points, degree)
        loss = (prediction - target).square().mean()
        if appearance.shape[1] > 1:
            loss = loss + config.appearance_l2 * appearance[:, 1:].square().mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            geometry.clamp_(0.55, 1.45)

    geometry = geometry.detach()
    appearance = appearance.detach()
    prediction, local_residual, coverage = _local_diagnostics(
        geometry, appearance, target, cameras, pixels, points, degree)
    jg, ja = exact_jacobians(geometry, appearance, cameras, pixels, points, degree)
    appearance_info = ja.square().sum(dim=0).reshape(len(points), -1).sum(dim=1)

    damping_values = {}
    schur_values = {}
    for damping in DAMPINGS:
        schur, values = marginalized_information(jg, ja, damping)
        key = f"{damping:.0e}"
        schur_values[key] = schur
        damping_values[key] = {name: value.detach().cpu().tolist()
                               for name, value in values.items()}

    perturb_generator = torch.Generator(device=device).manual_seed(9000 + seed)
    perturbation = torch.randn(len(points), generator=perturb_generator,
                               dtype=DTYPE, device=device)
    perturbation = 0.03 * perturbation / torch.linalg.vector_norm(perturbation)
    oracle = perturb_and_refit(geometry, appearance, perturbation, cameras, pixels,
                               points, degree, damping=CENTRAL_DAMPING)
    f_gg = jg.T @ jg
    perturb_predictions = {}
    raw_quadratic = perturbation @ f_gg @ perturbation
    for key, schur in schur_values.items():
        conditional_quadratic = perturbation @ schur @ perturbation
        perturb_predictions[key] = float(torch.sqrt(
            conditional_quadratic.clamp_min(0) / raw_quadratic.clamp_min(1e-15)))

    error = (geometry - geometry_gt).abs()
    return {
        "seed": seed,
        "baseline": baseline_name,
        "baseline_value": baseline,
        "capacity": capacity_name,
        "degree": degree,
        "regime": regime,
        "steps": config.steps,
        "initial_geometry_error_mean": float((initial_geometry - geometry_gt).abs().mean()),
        "geometry_error_mean": float(error.mean()),
        "final_mse": float((prediction - target).square().mean()),
        "oracle_unrecoverable_fraction": float(oracle),
        "perturb_prediction": perturb_predictions,
        "local": {
            "geometry": geometry.cpu().tolist(),
            "geometry_gt": geometry_gt.cpu().tolist(),
            "geometry_error": error.cpu().tolist(),
            "residual": local_residual.cpu().tolist(),
            "coverage": coverage.cpu().tolist(),
            "appearance_info": appearance_info.cpu().tolist(),
            "information": damping_values,
        },
    }
