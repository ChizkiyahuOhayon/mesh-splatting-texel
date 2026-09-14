"""Visibility-aware terminal opacity (VATO, SoftTail v2).

SoftTail v1 ends the opacity-floor schedule at one global value. The floor is a
lower bound, so the whole v1 effect is carried by the vertices the optimizer
wants below it. This module makes that endpoint local: each vertex's terminal
floor is a bounded function of a measured visibility-ambiguity statistic.

Statistic. For every training render the rasterizer already reports, per face,
how many supersampled samples reach it with alpha >= 1/255
(``triangle_was_rendered``) and, per sample, which face the transmittance
crosses 0.5 on (``rend_ids``; background is -1). Their per-face counts are
``hit_f`` and ``dom_f``. Pooled to vertices by incidence and smoothed by an
exponential moving average, the surface-dominance ratio

    d_v = D_v / H_v            in [0, 1]

is the fraction of the samples reaching the faces around ``v`` on which those
faces are the resolved surface. Resolved surfaces give one; silhouettes, thin
structures and layered fragments give less. It has no photometric term, so
appearance cannot absorb it, and it costs one ``bincount`` per iteration.

Endpoint. ``tau_v = low + (high - low) * d_v`` with ``high`` the v1 global
endpoint, so the constant family is the strict subfamily ``low == high`` and a
run reduces to v1 exactly when every vertex is resolved. The existing linear
ramp then runs per vertex, ``m_v(t) = init + (tau_v - init) * a(t)``.

Everything here is pure torch and CPU-testable; the model and training loop
only call into it.
"""

import torch


def _validate_unit_interval(name, value, upper_inclusive=False):
    value = float(value)
    top_ok = value <= 1.0 if upper_inclusive else value < 1.0
    if not (0.0 <= value and top_ok):
        raise ValueError(f"{name} must be in [0, 1{']' if upper_inclusive else ')'}, got {value}")
    return value


def face_visibility_counts(rend_ids, was_rendered, n_faces):
    """Per-face (dominant, hit) sample counts for one render.

    ``rend_ids`` is the rasterizer's per-sample resolved-face id (float tensor,
    any leading shape, -1 for background); ``was_rendered`` is its per-face hit
    count. Both are at the supersampled resolution.
    """
    n_faces = int(n_faces)
    if was_rendered.numel() < n_faces:
        raise ValueError(
            f"was_rendered has {was_rendered.numel()} rows, expected at least {n_faces} faces")
    ids = rend_ids.reshape(-1)
    ids = ids[ids >= 0].long()
    if ids.numel() and int(ids.max()) >= n_faces:
        raise ValueError(f"face id {int(ids.max())} is outside the {n_faces} faces")
    dominant = torch.bincount(ids, minlength=n_faces).to(torch.float32)
    hits = was_rendered.reshape(-1)[:n_faces].to(torch.float32)
    return dominant, hits


def pool_to_vertices(faces, dominant, hits, n_vertices):
    """Sum per-face counts onto the three corners of every face."""
    n_vertices = int(n_vertices)
    if faces.dim() != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must be [F, 3], got {tuple(faces.shape)}")
    if dominant.shape[0] != faces.shape[0] or hits.shape[0] != faces.shape[0]:
        raise ValueError(
            f"counts have {dominant.shape[0]}/{hits.shape[0]} rows for {faces.shape[0]} faces")
    corners = faces.reshape(-1).long()
    if corners.numel() and int(corners.max()) >= n_vertices:
        raise ValueError(f"face corner {int(corners.max())} exceeds {n_vertices} vertices")
    dom_v = torch.zeros(n_vertices, dtype=torch.float32, device=dominant.device)
    hit_v = torch.zeros(n_vertices, dtype=torch.float32, device=hits.device)
    dom_v.index_add_(0, corners, dominant.repeat_interleave(3))
    hit_v.index_add_(0, corners, hits.repeat_interleave(3))
    return dom_v, hit_v


class VisibilityTracker:
    """Per-vertex EMA of pooled dominant/hit counts.

    The vertex set is fixed between the restricted-Delaunay rebuild and the
    final cleanup, which is where this is used; ``prune`` handles the cleanup.
    """

    def __init__(self, n_vertices, rho=0.995, device="cuda"):
        rho = float(rho)
        if not 0.0 <= rho < 1.0:
            raise ValueError(f"rho must be in [0, 1), got {rho}")
        self.rho = rho
        self.dominant = torch.zeros(int(n_vertices), dtype=torch.float32, device=device)
        self.hits = torch.zeros(int(n_vertices), dtype=torch.float32, device=device)
        self.steps = 0

    @property
    def n_vertices(self):
        return self.dominant.shape[0]

    def update(self, rend_ids, was_rendered, faces):
        n_faces = faces.shape[0]
        dominant, hits = face_visibility_counts(rend_ids, was_rendered, n_faces)
        corners = faces.reshape(-1)
        if corners.numel() and int(corners.max()) >= self.n_vertices:
            raise ValueError(
                f"faces reference {int(corners.max()) + 1} vertices, tracker has {self.n_vertices}")
        dom_v, hit_v = pool_to_vertices(faces, dominant, hits, self.n_vertices)
        self.dominant.mul_(self.rho).add_(dom_v, alpha=1.0 - self.rho)
        self.hits.mul_(self.rho).add_(hit_v, alpha=1.0 - self.rho)
        self.steps += 1

    def ratio(self):
        """Surface-dominance ratio in [0, 1]; a never-hit vertex is neutral (1)."""
        seen = self.hits > 0
        ratio = torch.ones_like(self.hits)
        ratio[seen] = (self.dominant[seen] / self.hits[seen]).clamp(0.0, 1.0)
        return ratio

    def prune(self, mask):
        if mask.dtype != torch.bool or mask.numel() != self.n_vertices:
            raise ValueError(
                f"prune mask must be a bool tensor over {self.n_vertices} vertices")
        self.dominant = self.dominant[mask]
        self.hits = self.hits[mask]


def endpoint_from_ambiguity(ratio, low, high):
    """Bounded per-vertex terminal floor; ``low == high`` is the v1 constant."""
    low = _validate_unit_interval("adaptive_opacity_low", low)
    high = _validate_unit_interval("final_opacity", high)
    if low > high:
        raise ValueError(f"adaptive_opacity_low {low} exceeds final_opacity {high}")
    return low + (high - low) * ratio.clamp(0.0, 1.0)


def adaptive_floor_schedule(endpoint, init_opacity, progress):
    """The v1 linear ramp applied per vertex: init -> endpoint over progress in [0, 1]."""
    a = min(1.0, max(0.0, float(progress)))
    floor = init_opacity + (endpoint - init_opacity) * a
    return torch.minimum(floor, endpoint)


def shuffle_control(endpoint, seed=1234):
    """Negative control: the same endpoint multiset, randomly assigned.

    Uses a private CPU generator so the training RNG stream is untouched.
    """
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutation = torch.randperm(endpoint.numel(), generator=generator)
    return endpoint.reshape(-1)[permutation.to(endpoint.device)].reshape(endpoint.shape)


def _check_floor(raw, floor):
    if torch.is_tensor(floor):
        if raw.dim() != 2 or floor.shape != raw.shape:
            raise ValueError(
                f"per-vertex floor {tuple(floor.shape)} does not match raw {tuple(raw.shape)}; "
                "pass index= for a subset")
    return floor


def activation_with_floor(raw, floor):
    """``floor + (1 - floor) * sigmoid(raw)``; bitwise the v1 expression for a scalar."""
    floor = _check_floor(raw, floor)
    return floor + (1.0 - floor) * torch.sigmoid(raw)


def inverse_activation_with_floor(realized, floor, eps=1e-6):
    """Inverse of ``activation_with_floor`` on [floor, 1); bitwise v1 for a scalar."""
    floor = _check_floor(realized, floor)
    if torch.is_tensor(floor):
        clamped = torch.maximum(realized, floor + eps).clamp(max=1.0 - eps)
    else:
        clamped = realized.clamp(floor + eps, 1.0 - eps)
    z = (clamped - floor) / (1.0 - floor + eps)
    return torch.log(z / (1 - z))  # utils.general_utils.inverse_sigmoid, verbatim


def reparameterize_logits(realized, new_floor, eps=1e-6):
    """Logits that realize the same opacities under ``new_floor`` (clamped up to it)."""
    return inverse_activation_with_floor(realized.detach(), new_floor, eps)


def check_adaptive_contract(state, expect_adaptive, expected_high):
    """Refuse a checkpoint that is not what the evaluation arm assumes.

    ``state`` is the saved dict (or any mapping with the same keys). v1 arms
    must see no per-vertex floor; adaptive arms must see one within
    ``[low, high]`` with ``high`` equal to the arm's global endpoint.
    """
    has_vertex_floor = "opacity_floor_vertex" in state
    if not expect_adaptive:
        if has_vertex_floor or "adaptive_opacity" in state:
            raise RuntimeError("checkpoint carries an adaptive per-vertex floor; use an adaptive arm")
        return
    if not has_vertex_floor or "adaptive_opacity" not in state:
        raise RuntimeError("adaptive arm requires an adaptive checkpoint (opacity_floor_vertex)")
    meta = state["adaptive_opacity"]
    low, high = float(meta["low"]), float(meta["high"])
    floors = state["opacity_floor_vertex"]
    if float(state["opacity_floor"]) != float(expected_high) or high != float(expected_high):
        raise RuntimeError(
            f"adaptive checkpoint endpoint {state['opacity_floor']}/{high}, expected {expected_high}")
    lo, hi = float(floors.min()), float(floors.max())
    if lo < low - 1e-6 or hi > high + 1e-6:
        raise RuntimeError(f"per-vertex floors [{lo}, {hi}] outside the declared range [{low}, {high}]")


def _average_ranks(values):
    """Ranks starting at 1, ties given their average rank (for AUC / Spearman)."""
    values = values.reshape(-1).to(torch.float64)
    order = torch.argsort(values)
    sorted_values = values[order]
    n = values.numel()
    ranks = torch.empty(n, dtype=torch.float64, device=values.device)
    # Positions where a new tie group starts.
    starts = torch.ones(n, dtype=torch.bool, device=values.device)
    starts[1:] = sorted_values[1:] != sorted_values[:-1]
    group = torch.cumsum(starts, 0) - 1
    positions = torch.arange(1, n + 1, dtype=torch.float64, device=values.device)
    n_groups = int(group[-1]) + 1 if n else 0
    sums = torch.zeros(n_groups, dtype=torch.float64, device=values.device).index_add_(0, group, positions)
    counts = torch.zeros(n_groups, dtype=torch.float64, device=values.device).index_add_(
        0, group, torch.ones_like(positions))
    ranks[order] = (sums / counts)[group]
    return ranks


def roc_auc(scores, positive):
    """Probability that a positive scores higher than a negative (Mann-Whitney).

    Returns 0.5 when either class is empty.
    """
    positive = positive.reshape(-1).bool()
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = _average_ranks(scores)
    rank_sum = float(ranks[positive].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def spearman(x, y):
    """Spearman rank correlation of two equally long vectors (0 if degenerate)."""
    x = x.reshape(-1)
    y = y.reshape(-1)
    if x.numel() != y.numel() or x.numel() < 2:
        raise ValueError("spearman needs two equally long vectors with at least two entries")
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denominator = float(torch.sqrt((rx * rx).sum() * (ry * ry).sum()))
    if denominator == 0.0:
        return 0.0
    return float((rx * ry).sum()) / denominator
