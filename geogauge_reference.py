"""Exact small-system reference for appearance-marginalized geometry information.

The renderer is deliberately independent of the production CUDA path.  A 3x3
height field is rendered as normalized 2D Gaussian splats from a grid of camera
positions.  Height controls parallax (geometry), while per-point real-SH-like
coefficients control view-dependent intensity (appearance).
"""

from __future__ import annotations

import math

import torch

from utils.sh_utils import C0, C1, C2, C3


DTYPE = torch.float64


def point_grid(side=3, *, device="cpu", dtype=DTYPE):
    axis = torch.linspace(-0.55, 0.55, side, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(-1, 2)


def camera_grid(baseline, side=4, *, device="cpu", dtype=DTYPE):
    axis = torch.linspace(-baseline, baseline, side, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    xy = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
    directions = torch.cat((xy, torch.ones(len(xy), 1, device=device, dtype=dtype)), dim=1)
    return xy, directions / torch.linalg.vector_norm(directions, dim=1, keepdim=True)


def pixel_grid(res=8, *, device="cpu", dtype=DTYPE):
    axis = torch.linspace(-0.9, 0.9, res, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(-1, 2)


def sh_basis(directions, degree):
    """Real SH polynomial basis through degree 0, 1, or 3.

    This uses the same constants and ordering as the production renderer.
    """
    if degree not in (0, 1, 3):
        raise ValueError("degree must be 0, 1, or 3")
    x, y, z = directions.unbind(dim=1)
    columns = [C0 * torch.ones_like(x)]
    if degree >= 1:
        columns += [-C1 * y, C1 * z, -C1 * x]
    if degree >= 2:
        columns += [C2[0] * x * y, C2[1] * y * z,
                    C2[2] * (2 * z.square() - x.square() - y.square()),
                    C2[3] * x * z, C2[4] * (x.square() - y.square())]
    if degree >= 3:
        columns += [C3[0] * y * (3 * x.square() - y.square()), C3[1] * x * y * z,
                    C3[2] * y * (4 * z.square() - x.square() - y.square()),
                    C3[3] * z * (2 * z.square() - 3 * x.square() - 3 * y.square()),
                    C3[4] * x * (4 * z.square() - x.square() - y.square()),
                    C3[5] * z * (x.square() - y.square()),
                    C3[6] * x * (x.square() - 3 * y.square())]
    return torch.stack(columns, dim=1)


def _render_terms(geometry, appearance, cameras, pixels, points, degree,
                  sigma=0.24, background_weight=0.25, background=0.5):
    basis = sh_basis(torch.cat((cameras, torch.ones(len(cameras), 1,
                                device=cameras.device, dtype=cameras.dtype)), dim=1)
                     .div(torch.sqrt(1.0 + cameras.square().sum(dim=1, keepdim=True))), degree)
    if appearance.shape != (len(points), basis.shape[1]):
        raise ValueError(f"appearance must have shape {(len(points), basis.shape[1])}")

    centers = points[None] - cameras[:, None] * geometry[None, :, None]
    delta = pixels[None, :, None] - centers[:, None]
    weights = torch.exp(-0.5 * delta.square().sum(dim=-1) / (sigma * sigma))
    denominator = background_weight + weights.sum(dim=-1)
    colors = torch.einsum("vk,ik->vi", basis, appearance)
    numerator = background_weight * background + torch.einsum("vpi,vi->vp", weights, colors)
    image = numerator / denominator
    return image, weights, denominator, colors, delta, basis


def render(geometry, appearance, cameras, pixels, points, degree, **kwargs):
    return _render_terms(geometry, appearance, cameras, pixels, points, degree,
                         **kwargs)[0].reshape(-1)


def design_matrix(geometry, cameras, pixels, points, degree, **kwargs):
    n_basis = sh_basis(torch.cat((cameras, torch.ones(len(cameras), 1,
                                  device=cameras.device, dtype=cameras.dtype)), dim=1)
                       .div(torch.sqrt(1.0 + cameras.square().sum(dim=1, keepdim=True))),
                       degree).shape[1]
    appearance = torch.zeros(len(points), n_basis, device=geometry.device, dtype=geometry.dtype)
    _, weights, denominator, _, _, basis = _render_terms(
        geometry, appearance, cameras, pixels, points, degree, **kwargs)
    normalized = weights / denominator[..., None]
    return torch.einsum("vpi,vk->vpik", normalized, basis).reshape(
        len(cameras) * len(pixels), len(points) * n_basis)


def exact_jacobians(geometry, appearance, cameras, pixels, points, degree,
                    sigma=0.24, background_weight=0.25, background=0.5):
    """Return analytic image Jacobians with respect to height and appearance."""
    image, weights, denominator, colors, delta, _ = _render_terms(
        geometry, appearance, cameras, pixels, points, degree, sigma=sigma,
        background_weight=background_weight, background=background)
    center_derivative = -cameras[:, None, :]
    weight_derivative = weights * torch.einsum(
        "vpid,vid->vpi", delta, center_derivative) / (sigma * sigma)
    geometry_jacobian = (weight_derivative / denominator[..., None]
                         * (colors[:, None] - image[..., None])).reshape(-1, len(points))
    appearance_jacobian = design_matrix(
        geometry, cameras, pixels, points, degree, sigma=sigma,
        background_weight=background_weight, background=background)
    scale = math.sqrt(geometry_jacobian.shape[0])
    return geometry_jacobian / scale, appearance_jacobian / scale


def marginalized_information(geometry_jacobian, appearance_jacobian, damping):
    """Schur complement of the appearance block and per-geometry diagnostics."""
    f_gg = geometry_jacobian.T @ geometry_jacobian
    f_ga = geometry_jacobian.T @ appearance_jacobian
    f_aa = appearance_jacobian.T @ appearance_jacobian
    eye = torch.eye(len(f_aa), device=f_aa.device, dtype=f_aa.dtype)
    schur = f_gg - f_ga @ torch.linalg.solve(f_aa + damping * eye, f_ga.T)
    schur = 0.5 * (schur + schur.T)
    raw = torch.diagonal(f_gg)
    conditional = torch.diagonal(schur)
    ratio = conditional / raw.clamp_min(1e-15)
    return schur, {"raw_info": raw, "conditional_info": conditional,
                   "identifiable_fraction": ratio}


def refit_appearance(geometry, target, prior, cameras, pixels, points, degree,
                     damping=1e-4, background_weight=0.25, background=0.5, sigma=0.24):
    """Ridge least-squares appearance fit for fixed geometry."""
    matrix = design_matrix(geometry, cameras, pixels, points, degree, sigma=sigma,
                           background_weight=background_weight, background=background)
    zero = torch.zeros_like(prior)
    background_image = render(geometry, zero, cameras, pixels, points, degree,
                              sigma=sigma, background_weight=background_weight,
                              background=background)
    flat_prior = prior.reshape(-1)
    lhs = matrix.T @ matrix + damping * torch.eye(
        matrix.shape[1], device=matrix.device, dtype=matrix.dtype)
    rhs = matrix.T @ (target - background_image) + damping * flat_prior
    return torch.linalg.solve(lhs, rhs).reshape_as(prior)


def perturb_and_refit(geometry, appearance, perturbation, cameras, pixels, points,
                      degree, damping=1e-4, **kwargs):
    """Fraction of a finite geometry-induced image change appearance cannot recover."""
    baseline = render(geometry, appearance, cameras, pixels, points, degree, **kwargs)
    perturbed = geometry + perturbation
    raw = render(perturbed, appearance, cameras, pixels, points, degree, **kwargs)
    fitted = refit_appearance(perturbed, baseline, appearance, cameras, pixels, points,
                              degree, damping=damping, **kwargs)
    recovered = render(perturbed, fitted, cameras, pixels, points, degree, **kwargs)
    return (torch.linalg.vector_norm(recovered - baseline)
            / torch.linalg.vector_norm(raw - baseline).clamp_min(1e-15))
