"""Projected-footprint filtering for per-face residual texels."""

import torch


def projected_texel_weights(image_xy, triangles, texel_order):
    """Return one detail weight per face from final-resolution pixel coordinates."""
    if texel_order <= 0:
        raise ValueError("texel_order must be positive")
    indices = triangles.long()
    p0 = image_xy[indices[:, 0]]
    p1 = image_xy[indices[:, 1]]
    p2 = image_xy[indices[:, 2]]
    edge1 = p1 - p0
    edge2 = p2 - p0
    twice_area = torch.abs(edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0])
    area_per_texel = 0.5 * twice_area / float(texel_order * texel_order)
    return torch.nan_to_num(area_per_texel, nan=0.0, posinf=1.0).clamp_(0.0, 1.0)


def filter_texel_detail(texels, weights):
    """Keep each face's mean residual and footprint-filter only its detail."""
    if texels.ndim != 3 or weights.ndim != 1 or texels.shape[0] != weights.shape[0]:
        raise ValueError("expected texels [F,L^2,C] and weights [F]")
    face_mean = texels.mean(dim=1, keepdim=True)
    return face_mean + weights[:, None, None] * (texels - face_mean)

