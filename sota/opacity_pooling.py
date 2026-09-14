"""Softmin routing of the shared per-vertex opacity gradient.

A connected mesh makes opacity a *shared* parameter: a face's opacity is the
minimum over its three vertices, so the exact gradient reaches the argmin vertex
alone (``backward.cu``) and the other two learn nothing from that face. On a mesh
with F faces and V vertices a vertex is the argmin of only about F/V of the
faces it belongs to, so most (face, vertex) pairs carry no opacity gradient at
all, and the routing is self-reinforcing: the vertex that receives the downward
pressure stays the minimum.

This module owns the scalar schedule that controls the fix. The rasterizer splits
a face's opacity gradient over its three vertices with softmin weights

    w_v = exp(-beta * (o_v - o_min)) / sum_u exp(-beta * (o_u - o_min))

which sum to one, so the total gradient mass is redistributed and never
amplified. ``beta -> inf`` recovers the published argmin routing exactly, and
``beta <= 0`` selects it directly.

``beta`` is not a free constant. It is expressed in units of the *available*
opacity range ``1 - floor``, which the training schedule moves, so the amount of
smoothing stays invariant to where the floor currently sits:

    beta(t) = k(t) / (1 - floor(t))

``k`` is dimensionless: ``k = 1`` gives a vertex at the top of the range a
weight ratio of ``exp(-1) ~ 0.37`` against the minimum, and ``k = 100`` is hard
for any spread the range permits. ``k`` follows a geometric ramp over the same
iteration window the opacity floor already uses, and the routing becomes exactly
hard once that window closes, so the final phase of training optimises the true
objective.
"""

import math


def softmin_weights(opacities, beta):
    """Weights that split one face's opacity gradient over its three vertices.

    ``opacities`` is the face's three per-vertex opacities. Returns three weights
    summing to 1. ``beta <= 0`` returns the published one-hot argmin routing,
    with ties broken by the first minimal vertex exactly as the kernel's
    strictly-less-than comparison does.
    """
    if len(opacities) != 3:
        raise ValueError("a face has exactly three vertices")
    lowest = min(range(3), key=lambda i: opacities[i])
    if beta <= 0.0:
        return [1.0 if i == lowest else 0.0 for i in range(3)]
    # Shift by the minimum so the largest exponent is exactly 1, matching the
    # kernel and keeping every term finite for any beta.
    shifted = [math.exp(-beta * (o - opacities[lowest])) for o in opacities]
    total = sum(shifted)
    return [s / total for s in shifted]


def pool_beta(iteration, floor, start_iter, end_iter, k_start, k_end):
    """The softmin temperature to use at ``iteration``.

    Returns ``0.0`` (hard argmin routing) outside ``[start_iter, end_iter]``, so
    the run is byte-identical to the published method before the ramp opens and
    optimises the true objective after it closes.
    """
    if not 0.0 <= floor < 1.0:
        raise ValueError("opacity floor must lie in [0, 1)")
    if not 0 < k_start <= k_end:
        raise ValueError("k must be positive and non-decreasing")
    if end_iter <= start_iter:
        raise ValueError("the ramp must span at least one iteration")
    if iteration < start_iter or iteration > end_iter:
        return 0.0
    progress = (iteration - start_iter) / (end_iter - start_iter)
    k = k_start * (k_end / k_start) ** progress
    return k / (1.0 - floor)
