"""ADC-G0 arms, installed onto a TriangleModel without editing the published path.

Two installers, each replacing exactly one bound method, so an arm that is not
installed runs code identical to the baseline's:

  install_rng_fix          `_sample_alives` draws from a call-local generator
  install_multiplicity     the above, plus replacement sampling and depth allocation

The published `add_new_gs` samples with `replacement=False` and `_update_params_fast`
opens with `torch.unique`, so a selected face is subdivided exactly once regardless of
its score. `install_multiplicity` lets a face drawn `k` times reach depth `k`, paying
for it out of the same budget. The protocol is experiments/adc_g0/protocol.md.
"""

import torch

from adc_allocator import (
    MAX_DEPTH,
    allocate_depths,
    call_generator,
    deepen_cost,
    depth_histogram,
    leaf_cost,
)


def eligible_probs(triangles):
    """The published selection distribution, unchanged.

    ADC-G0 deliberately alters no score: XVR-G0 preregistered and retired
    residual-guided densification on both scenes, and `max_blending` was one of its
    controls. This gate changes how budget is spent, not what it is spent on.
    """
    probs = triangles.importance_score.squeeze().clone()
    areas = triangles.triangle_areas().squeeze()
    probs = torch.where(areas < triangles.size_probs_zero, torch.zeros_like(probs), probs)
    return torch.where(
        triangles.image_size < triangles.size_probs_zero_image_space,
        torch.zeros_like(probs),
        probs,
    )


def _remap_after_prune(indices, removed):
    """Where surviving faces land once `removed` faces are dropped.

    `prune_triangles` masks with a boolean, which preserves order, so a survivor
    moves down by the number of removed faces below it.
    """
    return (torch.cumsum((~removed).to(torch.int64), 0) - 1)[indices]


def _subdivide(triangles, indices, iteration):
    """One round of the published midpoint subdivision.

    Returns the indices the children occupy afterwards and the removal mask, so a
    caller can both deepen those children again and follow other faces across the
    prune. The three calls are exactly what `add_new_gs` does; only the bookkeeping
    around them is new.
    """
    indices = torch.unique(indices)
    before = triangles._triangle_indices.shape[0]
    triangles.densification_postfix(*triangles._update_params_fast(indices, iteration))
    after = triangles._triangle_indices.shape[0]

    device = triangles._triangle_indices.device
    removed = torch.zeros(after, dtype=torch.bool, device=device)
    removed[indices.to(device)] = True
    triangles.prune_triangles(~removed)

    # Every removed face lies below `before`, so the children all shift by the same
    # amount and stay contiguous.
    shift = int(indices.numel())
    return torch.arange(before - shift, after - shift, device=device), removed


def install_rng_fix(triangles, seed):
    """Draw from a call-local generator instead of reseeding the global stream.

    The published `_sample_alives` calls `torch.manual_seed(1)` on every densification
    round. Successive rounds therefore draw correlated index sets, and every other
    consumer of torch randomness in the process has its stream reset underneath it.
    """
    state = {"calls": 0}

    def _sample_alives(probs, num, alive_indices=None):
        generator = call_generator(probs.device, seed, state["calls"])
        state["calls"] += 1
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        sampled = torch.multinomial(probs, num, replacement=False, generator=generator)
        return sampled if alive_indices is None else alive_indices[sampled]

    triangles._sample_alives = _sample_alives
    return state


def install_multiplicity(triangles, seed, max_depth=MAX_DEPTH):
    """Let a face drawn `k` times be subdivided to depth `k`, at matched budget.

    Depth is applied one level at a time using the published subdivision, deepest
    first: the faces needing depth two are subdivided, their children are followed
    across the prune, and the second round takes those children together with the
    faces needing depth one. Faces the published allocator would have added
    unconditionally -- the largest by area -- keep their depth-one treatment in both
    arms, so that channel is not part of what is being tested.
    """
    install_rng_fix(triangles, seed)
    rounds = []

    def add_new_gs(iteration, cap_max, splitt_large_triangles):
        current = triangles.vertices.shape[0]
        target = min(cap_max, int(triangles.add_percentage * current))
        requested = max(0, target - current)
        if requested <= 0:
            return 0

        probs = eligible_probs(triangles)
        budget = deepen_cost(0) * requested
        generator = call_generator(probs.device, seed, len(rounds))
        depths = allocate_depths(probs, budget, generator, max_depth)

        areas = triangles.triangle_areas().squeeze()
        largest = torch.topk(areas, min(splitt_large_triangles, areas.numel()), sorted=False)
        depths = depths.clone()
        depths[largest.indices] = torch.clamp(depths[largest.indices], min=1)

        rounds.append(
            {
                "iteration": int(iteration),
                "requested_faces": int(requested),
                "budget": int(budget),
                "spent": leaf_cost(depths),
                "faces_before": int(triangles._triangle_indices.shape[0]),
                **depth_histogram(depths, max_depth),
            }
        )
        if int(depths.max()) == 0:
            return 0

        # Deepest level first, carrying the shallower selection across each prune.
        pending = (depths >= 1).nonzero(as_tuple=True)[0]
        for level in range(max_depth, 1, -1):
            deep = (depths >= level).nonzero(as_tuple=True)[0]
            if deep.numel() == 0:
                continue
            children, removed = _subdivide(triangles, deep, iteration)
            shallow = pending[~torch.isin(pending, deep)]
            pending = torch.cat([_remap_after_prune(shallow, removed), children])
        if pending.numel():
            _subdivide(triangles, pending, iteration)
        return 0

    triangles.add_new_gs = add_new_gs
    return rounds


INSTALLERS = {
    "stock": lambda triangles, seed: None,
    "rng": install_rng_fix,
    "multiplicity": install_multiplicity,
}


def install_arm(triangles, arm, seed):
    """Install one ADC-G0 arm; `stock` leaves the published pipeline untouched."""
    if arm not in INSTALLERS:
        raise ValueError(f"unknown ADC-G0 arm {arm!r}; expected one of {sorted(INSTALLERS)}")
    return INSTALLERS[arm](triangles, seed)
