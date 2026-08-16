"""Window-hardening schedules expressed in coverage rather than in sigma.

MeshSplatting renders a triangle through the soft window `phi ** sigma`, where
`phi` is the distance to the nearest edge normalised by its largest value inside
the triangle. The level set `{phi >= u}` is the triangle offset inward by
`u * inradius`, i.e. a similar triangle of area `A * (1 - u) ** 2`, so the
fraction of the triangle's area the window actually deposits is

    coverage(sigma) = (1 / A) * integral_T phi ** sigma dA
                    = integral_0^1 (1 - u) ** 2 d(u ** sigma)
                    = 2 / ((sigma + 1) * (sigma + 2)) ,

which is `1 / 3` at the initial `sigma = 1` and `1` at the opaque endpoint. This
is exact, affine invariant (so it holds for the projected triangle too), and it
is the quantity the rest of the model has to adapt to: annealing sigma is really
annealing coverage from a third of each triangle to all of it.

The published schedule is linear in sigma, which is strongly nonlinear in
coverage -- it defers most of the change to the end of training, where the
vertex learning rate has decayed to a few percent of its initial value and the
geometry can no longer respond. The two schedules here move that change to where
the optimiser can still follow it, at an identical endpoint.
"""

import numpy as np


SCHEDULES = ("linear", "coverage", "lrmatched")


def coverage(sigma):
    """Fraction of a triangle's area deposited by the window `phi ** sigma`."""
    return 2.0 / ((sigma + 1.0) * (sigma + 2.0))


def sigma_at(target_coverage):
    """Inverse of `coverage`: the sigma whose window deposits this fraction.

    Solves `s ** 2 + 3 s + 2 - 2 / c = 0` on the positive branch. Coverage below
    `1 / 3` corresponds to `sigma > 1`, which no schedule here asks for, so the
    argument is clamped to the attainable `(0, 1]`.
    """
    target = min(max(float(target_coverage), 1e-12), 1.0)
    return 0.5 * (np.sqrt(1.0 + 8.0 / target) - 3.0)


def _progress(iteration, start, until):
    if iteration <= start:
        return 0.0
    if until <= start:
        return 1.0
    return min((iteration - start) / (until - start), 1.0)


def _lr_mass(fraction, decay):
    """Share of the total learning-rate budget spent by `fraction` of training.

    The vertex learning rate decays log-linearly, `lr(t) = lr_init * decay ** t`
    for `t` in `[0, 1]`, so the budget spent is the normalised integral below.
    A schedule that follows this curve hardens in proportion to how much the
    geometry can still move, and its endpoint is reached with the last update
    that could have used it.
    """
    if np.isclose(decay, 1.0):
        return fraction
    return (1.0 - decay ** fraction) / (1.0 - decay)


def schedule(name, iteration, initial_sigma, final_sigma, start, until, decay=0.01):
    """Sigma at `iteration` under one of `SCHEDULES`.

    Every schedule holds `initial_sigma` until `start` and reaches exactly
    `final_sigma` at `until`, so arms differ only in the path between them.
    """
    fraction = _progress(iteration, start, until)
    if fraction >= 1.0:
        return final_sigma
    if name == "linear":
        return initial_sigma - (initial_sigma - final_sigma) * fraction
    if name == "coverage":
        share = fraction
    elif name == "lrmatched":
        share = _lr_mass(fraction, decay)
    else:
        raise ValueError(f"unknown sigma schedule {name!r}; expected one of {SCHEDULES}")
    low, high = coverage(initial_sigma), coverage(final_sigma)
    return float(sigma_at(low + (high - low) * share))
