"""Per-face barycentric cell fitting, free of the CUDA extension.

The question APX-F0 asks is how much held-out error a higher-capacity appearance
model could remove, so the arithmetic that answers it has to be checkable against
cases whose answer is known by hand. None of it renders.

MeshSplatting interpolates appearance in *screen space*, so screen-space barycentrics
are the model's own parameterisation of a face rather than an approximation of it.
The protocol is experiments/apx_f0/protocol.md.
"""

import math

import torch


CELL_ORDERS = (1, 2, 4)
MIN_PIXELS_PER_CELL = 4


def barycentric(points, corners):
    """Barycentric coordinates of 2-D `points` inside their `corners` triangles.

    `points` is [N, 2] and `corners` is [N, 3, 2]. Degenerate triangles -- a face
    projecting to zero screen area -- return the centroid, which puts their pixels in
    one cell rather than producing a division by zero.
    """
    origin, first, second = corners[:, 0], corners[:, 1], corners[:, 2]
    edge_a, edge_b = first - origin, second - origin
    offset = points - origin
    determinant = edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]
    safe = determinant.abs() > 1e-12
    inverse = torch.where(safe, 1.0 / torch.where(safe, determinant, torch.ones_like(determinant)),
                          torch.zeros_like(determinant))
    weight_a = (offset[:, 0] * edge_b[:, 1] - offset[:, 1] * edge_b[:, 0]) * inverse
    weight_b = (edge_a[:, 0] * offset[:, 1] - edge_a[:, 1] * offset[:, 0]) * inverse
    third = 1.0 / 3.0
    weight_a = torch.where(safe, weight_a, torch.full_like(weight_a, third))
    weight_b = torch.where(safe, weight_b, torch.full_like(weight_b, third))
    return torch.stack([1.0 - weight_a - weight_b, weight_a, weight_b], dim=1)


def cell_index(weights, order):
    """Which of `order * order` barycentric cells each pixel falls in.

    Two of the three coordinates determine the cell; the grid is the square binning
    of those two, which partitions the face into `order * order` regions of equal
    parameter area. Coordinates are clamped, so a pixel whose dominant face is
    slightly off -- rasterisation and attribution disagree at silhouettes -- lands in
    an edge cell instead of out of range.
    """
    if order < 1:
        raise ValueError("order must be at least 1")
    scaled = (weights[:, 1:] * order).floor().to(torch.int64).clamp_(0, order - 1)
    return scaled[:, 0] * order + scaled[:, 1]


def fit_cell_colours(bins, colours, bin_count):
    """Mean colour of each bin, and how many pixels fell in it.

    A bin with no pixels gets zero and is reported as empty; scoring must fall back
    for those rather than treating black as a fitted colour.
    """
    totals = torch.zeros(bin_count, colours.shape[1], dtype=colours.dtype, device=colours.device)
    totals.index_add_(0, bins, colours)
    counts = torch.zeros(bin_count, dtype=colours.dtype, device=colours.device)
    counts.index_add_(0, bins, torch.ones_like(bins, dtype=colours.dtype))
    return totals / counts.clamp_min(1.0).unsqueeze(1), counts


def squared_error(bins, colours, fitted, empty, fallback_colours):
    """Per-pixel squared error of a fitted assignment, summed over channels.

    A pixel whose cell received no training pixels -- it is visible only in held-out
    views -- is scored against `fallback_colours`, the coarser fit for its face,
    rather than against an unfitted cell. Scoring it against an empty cell would
    charge the model class with an error it was never given the chance to avoid, and
    inflating the fitted error is how a ceiling measurement talks itself into a
    negative result.
    """
    predicted = torch.where(empty[bins].unsqueeze(1), fallback_colours, fitted[bins])
    return ((predicted - colours) ** 2).sum(dim=1)


def gain_to_db(current_error, fitted_error):
    """PSNR improvement in dB between two squared-error totals over the same pixels.

    Both are sums over the identical pixel set in [0, 1] colour, so the pixel count
    cancels out of the ratio. Callers must not pass totals from different pixel sets;
    there is no way for this function to detect that.
    """
    if current_error <= 0 or fitted_error <= 0:
        return 0.0
    return float(10.0 * math.log10(current_error / fitted_error))
