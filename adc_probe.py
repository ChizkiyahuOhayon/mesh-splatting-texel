"""Probing primitives for the ADC forensics, kept free of the CUDA extension.

Nothing here renders. Separating these out means the finite difference can be
exercised against a closed-form quadratic, where the correct answer is known exactly,
instead of only ever being observed on a GPU where a wrong answer looks like a
property of the rasterizer -- which is precisely the confusion ADC-F0 exists to
resolve.
"""

import torch


STRATA = 5
PROBES_PER_STRATUM = 16
RUNGS = (0.002, 0.001)
# Exact arithmetic on an exactly quadratic loss would give 0; this is float headroom.
RUNG_TOLERANCE = 1e-3


def face_max_per_vertex(face_values, triangle_indices, vertex_count):
    """Reduce a face-indexed quantity onto vertices by maximum over incident faces.

    This is the reduction train.py already uses when it accumulates `image_size` and
    `importance_score`, so a covariate measured here means the same thing it means to
    the densification criterion being investigated.
    """
    reduced = torch.zeros(vertex_count, dtype=face_values.dtype, device=face_values.device)
    reduced.scatter_reduce_(
        0,
        triangle_indices.reshape(-1).long(),
        face_values.repeat_interleave(3),
        reduce="amax",
        include_self=True,
    )
    return reduced


def stratified_probe_set(values, eligible, strata=STRATA, per_stratum=PROBES_PER_STRATUM):
    """Evenly spaced members of each equal-count quantile bin of `values`.

    Even spacing rather than random sampling makes the probe set a deterministic
    function of the checkpoint, so a rerun reproduces it exactly without carrying a
    seed -- and every stratum is guaranteed a sample, which is the whole point of
    stratifying.
    """
    pool = eligible[torch.argsort(values[eligible])]
    picks = []
    for chunk in torch.chunk(pool, strata):
        if chunk.numel() == 0:
            continue
        take = min(per_stratum, chunk.numel())
        positions = torch.linspace(0, chunk.numel() - 1, take).round().long()
        picks.append(chunk[positions.to(chunk.device)])
    return torch.cat(picks)


def central_differences(set_value, loss_fn, original, rungs=RUNGS):
    """Central-difference estimates of dL/dx at each step size, one per rung.

    `set_value` writes the probed scalar and `loss_fn` returns the loss at the current
    value; the original is always restored, including when a rung raises. Two rungs
    are taken so each measurement carries its own convergence evidence rather than
    being trusted.
    """
    estimates = []
    try:
        for step in rungs:
            samples = []
            for sign in (1.0, -1.0):
                set_value(original + sign * step)
                samples.append(float(loss_fn()))
            estimates.append((samples[0] - samples[1]) / (2.0 * step))
    finally:
        set_value(original)
    return estimates


def rung_disagreement(estimates):
    """Relative gap between the coarsest and finest rung."""
    coarse, fine = estimates[0], estimates[-1]
    return abs(fine - coarse) / max(abs(fine), abs(coarse), 1e-30)
