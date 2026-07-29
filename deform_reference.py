"""Small PyTorch reference for a K=1 deformable triangle boundary.

This follows the DETRIS barycentric parameterization, but deliberately stays
outside the production renderer.  A triangle has one scalar displacement per
edge.  Each scalar moves the edge midpoint along its outward unit normal,
forming the ordered polygon ``v0,c01,v1,c12,v2,c20``.
"""

from __future__ import annotations

import math

import torch


def canonical_grid(res: int, extent=(-0.25, 1.25), *, device="cpu", dtype=torch.float32):
    """Return flattened barycentric query points and the square image shape."""
    lo, hi = extent
    axis = torch.linspace(lo, hi, res, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(-1, 2), (res, res)


def control_polygon(displacements: torch.Tensor) -> torch.Tensor:
    """Build K=1 boundary polygons from ``[...,3]`` edge displacements."""
    if displacements.shape[-1] != 3:
        raise ValueError("displacements must have shape [...,3]")
    d = displacements
    base_shape = d.shape[:-1]
    vertices = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                            device=d.device, dtype=d.dtype)
    outward = torch.tensor([[0.0, -1.0], [1.0, 1.0], [-1.0, 0.0]],
                           device=d.device, dtype=d.dtype)
    outward[1] /= math.sqrt(2.0)
    vertices = vertices.expand(*base_shape, 3, 2)
    controls = torch.stack((
        0.5 * (vertices[..., 0, :] + vertices[..., 1, :]) + d[..., 0, None] * outward[0],
        0.5 * (vertices[..., 1, :] + vertices[..., 2, :]) + d[..., 1, None] * outward[1],
        0.5 * (vertices[..., 2, :] + vertices[..., 0, :]) + d[..., 2, None] * outward[2],
    ), dim=-2)
    return torch.stack((vertices[..., 0, :], controls[..., 0, :],
                        vertices[..., 1, :], controls[..., 1, :],
                        vertices[..., 2, :], controls[..., 2, :]), dim=-2)


def lift_barycentric(uv: torch.Tensor, triangle_xyz: torch.Tensor) -> torch.Tensor:
    """Lift barycentric ``(u,v)`` points to the physical plane of a 3D triangle."""
    weights = torch.stack((1.0 - uv[..., 0] - uv[..., 1], uv[..., 0], uv[..., 1]), dim=-1)
    return torch.matmul(weights, triangle_xyz)


def physical_control_polygon(triangle_xyz: torch.Tensor,
                             displacements: torch.Tensor) -> torch.Tensor:
    """Return the six K=1 boundary vertices as real coplanar 3D points."""
    return lift_barycentric(control_polygon(displacements), triangle_xyz)


def _winding_inside(polygon: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Non-zero winding classification; shape ``[batch,n_points]``."""
    a = polygon
    b = torch.roll(polygon, shifts=-1, dims=-2)
    x = points[None, :, None, 0]
    y = points[None, :, None, 1]
    ax, ay = a[:, None, :, 0], a[:, None, :, 1]
    bx, by = b[:, None, :, 0], b[:, None, :, 1]
    cross = (ax - x) * (by - y) - (ay - y) * (bx - x)
    up = (ay <= y) & (y < by) & (cross > 0)
    down = (by <= y) & (y < ay) & (cross < 0)
    return (up.sum(dim=-1) - down.sum(dim=-1)) != 0


def deformable_alpha(points: torch.Tensor, displacements: torch.Tensor,
                     sigma=0.15, opacity=0.95) -> torch.Tensor:
    """Render DETRIS-style bounded opacity on barycentric query ``points``.

    The hard winding test matches the paper.  Interior opacity uses the nearest
    segment distance divided by ``2*area/perimeter``, followed by the TS
    power-law window.  Corner smoothing is intentionally omitted for the K=1
    identification gate.
    """
    single = displacements.ndim == 1
    d = displacements[None] if single else displacements
    polygon = control_polygon(d)
    nxt = torch.roll(polygon, shifts=-1, dims=1)
    edge = nxt - polygon

    delta = points[None, :, None, :] - polygon[:, None, :, :]
    denom = edge.square().sum(dim=-1).clamp_min(1e-12)
    t = (delta * edge[:, None, :, :]).sum(dim=-1) / denom[:, None, :]
    closest = polygon[:, None, :, :] + t.clamp(0.0, 1.0)[..., None] * edge[:, None, :, :]
    distance_sq = (points[None, :, None, :] - closest).square().sum(dim=-1)
    distance = distance_sq.clamp_min(1e-24).sqrt()
    distance = torch.where(distance_sq < 1e-20, torch.zeros_like(distance), distance)
    boundary_distance = distance.min(dim=-1).values

    cross = polygon[..., 0] * nxt[..., 1] - polygon[..., 1] * nxt[..., 0]
    area = 0.5 * cross.sum(dim=-1).abs()
    perimeter = torch.linalg.vector_norm(edge, dim=-1).sum(dim=-1)
    radius = (2.0 * area / perimeter.clamp_min(1e-12)).clamp_min(1e-8)

    sigma = torch.as_tensor(sigma, device=d.device, dtype=d.dtype).reshape(-1)
    opacity = torch.as_tensor(opacity, device=d.device, dtype=d.dtype).reshape(-1)
    if sigma.numel() == 1:
        sigma = sigma.expand(len(d))
    if opacity.numel() == 1:
        opacity = opacity.expand(len(d))
    phi = (boundary_distance / radius[:, None]).clamp(0.0, 1.0)
    powered = phi.clamp_min(1e-12).pow(sigma[:, None])
    powered = torch.where(phi > 0, powered, torch.zeros_like(powered))
    alpha = opacity[:, None] * powered
    alpha = alpha * _winding_inside(polygon, points).to(alpha.dtype)
    alpha = alpha.clamp_max(0.99)
    return alpha[0] if single else alpha
