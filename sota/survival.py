"""Opacity-aware topology survival (OATS): the final-cleanup rule.

v1 keeps a face if its blending weight ``alpha * T`` ever exceeded 0.5 in one
pixel of one training view: a single-sample order statistic. OATS scores each
face by the weight it integrated over every view, ``S_f = sum alpha * T``, and
keeps exactly as many faces as the v1 rule would, so the two rules differ in
*which* faces survive and never in how many. See handover/OATS_PLAN.md.
"""

import torch

V1_THRESHOLD = 0.5


def v1_keep(max_blending):
    """The published cleanup: ``train.py`` prunes ``importance_score <= 0.5``."""
    return max_blending > V1_THRESHOLD


def budget_matched_keep(scores, budget, return_stats=False):
    """Keep the ``budget`` faces with the largest score.

    Ties break by face index, lowest first, so the result is deterministic.
    Fed ``max_blending`` with the v1 budget, it reproduces ``v1_keep`` exactly.
    """
    faces = scores.numel()
    if not 0 <= budget <= faces:
        raise ValueError(f"budget must be in [0, {faces}], got {budget}")
    # A stable descending sort keeps equal scores in index order.
    order = torch.sort(scores, descending=True, stable=True).indices
    keep = torch.zeros(faces, dtype=torch.bool, device=scores.device)
    keep[order[:budget]] = True
    if not return_stats:
        return keep
    return keep, {
        "budget": int(budget),
        # Faces that never contributed survive only when the budget forces it.
        "kept_with_zero_score": int((keep & (scores <= 0)).sum()),
    }


def budget_matched_delete(peak_delete, integrated):
    """Training-time OATS: the same substitution inside the pruning loop.

    ``train.py`` deletes a face whose peak ``max_blending`` sits at or below a
    rising threshold. The integral lives on a different scale, so the threshold
    cannot be carried over; instead the integral re-picks the faces, as many as
    the peak rule would have deleted. Returns the peak mask untouched when the
    face count and the mask disagree, which only a desync could cause.
    """
    if peak_delete.shape != integrated.shape:
        raise ValueError(
            f"mask and scores differ in shape: {tuple(peak_delete.shape)} vs {tuple(integrated.shape)}")
    cut = int(peak_delete.sum())
    delete = torch.zeros_like(peak_delete)
    if cut > 0:
        # Stable ascending sort: ties break by face index, so faces that never
        # contributed (score 0) go first, exactly as under the peak rule.
        order = torch.sort(integrated, stable=True).indices
        delete[order[:cut]] = True
    return delete


def rule_disagreement(keep_a, keep_b):
    """Symmetric difference and Jaccard overlap of two kept sets."""
    if keep_a.shape != keep_b.shape:
        raise ValueError(f"kept sets differ in shape: {tuple(keep_a.shape)} vs {tuple(keep_b.shape)}")
    differ = int((keep_a ^ keep_b).sum())
    union = int((keep_a | keep_b).sum())
    return {
        "symmetric_difference": differ,
        "fraction": differ / max(1, keep_a.numel()),
        "jaccard": 1.0 if union == 0 else int((keep_a & keep_b).sum()) / union,
    }


def survival_contract(state, expect_survival, n_faces=None):
    """Refuse a checkpoint trained under the other cleanup rule.

    A survival checkpoint carries ``face_survival_score`` [F]; a v1 checkpoint
    must not, or a v1 arm would silently score a survival-pruned mesh.
    """
    score = state.get("face_survival_score")
    if not expect_survival:
        if score is not None:
            raise ValueError("v1 arm was given a survival-pruned checkpoint")
        return
    if score is None:
        raise ValueError("survival arm needs face_survival_score in the checkpoint")
    if n_faces is not None and score.numel() != n_faces:
        raise ValueError(
            f"face_survival_score has {score.numel()} entries for {n_faces} faces")
