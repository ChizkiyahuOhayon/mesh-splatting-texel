"""Budget-matched multiplicity allocation for triangle densification.

MeshSplatting draws densification candidates with `replacement=False` and then opens
`_update_params_fast` with `torch.unique`, so a face selected for subdivision is
subdivided exactly once however it scored: the importance score's magnitude is
discarded and only membership in the selected set survives. This module lets a face
drawn `k` times be subdivided to depth `k`, while spending exactly the budget the
baseline would have spent.

Pure tensor code, no CUDA extension, so the allocation rule can be tested against
hand-checkable cases. The protocol is experiments/adc_g0/protocol.md.
"""

import torch


MAX_DEPTH = 2

# Deepening one face from depth d to d+1 replaces its 4^d leaves with 4^(d+1).
BRANCHING = 4


def deepen_cost(depth):
    """Net faces gained by taking one face from `depth` to `depth + 1`."""
    return (BRANCHING - 1) * BRANCHING**depth


def leaf_cost(depths):
    """Net faces gained by subdividing each face to its assigned depth."""
    return int((BRANCHING ** depths.to(torch.int64) - 1).sum())


def depths_from_prefix(samples, face_count, max_depth=MAX_DEPTH):
    """Per-face subdivision depth implied by a prefix of the draw sequence."""
    if samples.numel() == 0:
        return torch.zeros(face_count, dtype=torch.int64, device=samples.device)
    return torch.bincount(samples, minlength=face_count).clamp(max=max_depth).to(torch.int64)


def allocate_depths(probs, budget, generator=None, max_depth=MAX_DEPTH):
    """Assign each face a subdivision depth, spending at most `budget` new faces.

    A single sequence of draws is taken with replacement and the longest prefix whose
    cost fits the budget is kept. Prefix cost is monotone non-decreasing in prefix
    length -- extending a prefix can only deepen a face or leave it alone, and
    deepening never reduces cost -- so the boundary is exact by bisection and is a
    deterministic function of the draw. Nothing here is tuned.

    The draw length is `budget // deepen_cost(0)`, the number of faces the baseline
    would have subdivided. That is sufficient: if every draw lands on a distinct face
    the prefix costs exactly `budget`, which is the baseline's own allocation.
    """
    face_count = probs.numel()
    draws = budget // deepen_cost(0)
    if draws <= 0 or face_count == 0 or float(probs.sum()) <= 0.0:
        return torch.zeros(face_count, dtype=torch.int64, device=probs.device)

    samples = torch.multinomial(probs, draws, replacement=True, generator=generator)
    low, high = 0, draws
    while low < high:
        middle = (low + high + 1) // 2
        if leaf_cost(depths_from_prefix(samples[:middle], face_count, max_depth)) <= budget:
            low = middle
        else:
            high = middle - 1
    return depths_from_prefix(samples[:low], face_count, max_depth)


def depth_histogram(depths, max_depth=MAX_DEPTH):
    """Faces at each depth, as the manipulation check the protocol requires.

    A run whose histogram shows almost nothing above depth 1 did not concentrate any
    budget, and a null result from it says nothing about the hypothesis.
    """
    counts = torch.bincount(depths.to(torch.int64), minlength=max_depth + 1)
    return {f"depth_{depth}": int(counts[depth]) for depth in range(max_depth + 1)}


def call_generator(device, seed, call_index):
    """A per-call generator that leaves the ambient torch stream untouched.

    The published `_sample_alives` calls `torch.manual_seed(1)` on every densification
    round, which reseeds the *global* generator. Successive rounds therefore draw
    correlated index sets, and every other consumer of torch randomness in the process
    has its stream reset underneath it. Mixing the call index in keeps the run
    reproducible from `seed` alone while making the rounds independent.
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(seed * 1_000_003 + call_index)
    return generator
