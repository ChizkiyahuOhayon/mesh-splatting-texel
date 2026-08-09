"""Exact mathematical core for EdgeVal's frozen-renderer mechanism gate.

This module deliberately contains no renderer approximation.  The CUDA carrier will
export per-fold normal-equation statistics for the exact affine edge design; the
functions below define how those statistics are constructed, validated, and scored.
"""

from dataclasses import dataclass
from typing import Sequence

import torch


RIDGE_EXPONENTS = tuple(range(-6, 3))
DEFAULT_FOLDS = 4
DEFAULT_BETA_SE = 1.0


@dataclass(frozen=True)
class EdgeTopology:
    edge_vertices: torch.Tensor
    face_edges: torch.Tensor
    edge_face_count: torch.Tensor


@dataclass(frozen=True)
class CrossFitValue:
    valid: torch.Tensor
    ridge: torch.Tensor
    coefficients: torch.Tensor
    fold_gain: torch.Tensor
    mean_gain: torch.Tensor
    standard_error: torch.Tensor
    value: torch.Tensor


def _require_shape(tensor, suffix, name):
    if tensor.ndim < len(suffix) or tuple(tensor.shape[-len(suffix):]) != tuple(suffix):
        raise ValueError(f"{name} must end in shape {tuple(suffix)}, got {tuple(tensor.shape)}")


def build_edge_topology(faces, *, reject_nonmanifold=True):
    """Build deterministic global undirected edges and face-local edge rows.

    Local columns are (v0,v1), (v1,v2), and (v2,v0), matching ``edge_basis``.
    Edge rows are sorted lexicographically by their canonical endpoint pair.
    """
    if not torch.is_tensor(faces):
        faces = torch.as_tensor(faces)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must be [F, 3], got {tuple(faces.shape)}")
    if faces.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"faces must use int32 or int64 indices, got {faces.dtype}")
    faces = faces.to(dtype=torch.int64)
    if faces.numel() == 0:
        empty_edges = torch.empty((0, 2), dtype=torch.int64, device=faces.device)
        empty_face_edges = torch.empty((0, 3), dtype=torch.int64, device=faces.device)
        empty_counts = torch.empty((0,), dtype=torch.int64, device=faces.device)
        return EdgeTopology(empty_edges, empty_face_edges, empty_counts)
    if bool((faces < 0).any()):
        raise ValueError("faces contain a negative vertex index")
    degenerate = (
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 2] == faces[:, 0])
    )
    if bool(degenerate.any()):
        first = int(torch.nonzero(degenerate, as_tuple=False)[0, 0])
        raise ValueError(f"face {first} repeats a vertex")

    local = torch.stack(
        (faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]), dim=1
    )
    canonical = torch.sort(local, dim=-1).values
    vertex_count = int(faces.max()) + 1
    max_int64 = torch.iinfo(torch.int64).max
    if vertex_count and vertex_count > max_int64 // vertex_count:
        raise OverflowError("vertex index range is too large for deterministic edge keys")
    keys = canonical[..., 0] * vertex_count + canonical[..., 1]
    unique_keys, inverse, counts = torch.unique(
        keys.reshape(-1), sorted=True, return_inverse=True, return_counts=True
    )
    if reject_nonmanifold and bool((counts > 2).any()):
        row = int(torch.nonzero(counts > 2, as_tuple=False)[0, 0])
        u = int(unique_keys[row] // vertex_count)
        v = int(unique_keys[row] % vertex_count)
        raise ValueError(f"non-manifold edge ({u}, {v}) has {int(counts[row])} incident faces")

    edge_vertices = torch.stack(
        (unique_keys // vertex_count, unique_keys % vertex_count), dim=1
    )
    return EdgeTopology(edge_vertices, inverse.reshape(-1, 3), counts)


def edge_basis(barycentric):
    """Evaluate the three quadratic P2 edge functions in face-local edge order."""
    if not torch.is_tensor(barycentric):
        barycentric = torch.as_tensor(barycentric)
    _require_shape(barycentric, (3,), "barycentric")
    a, b, c = barycentric.unbind(dim=-1)
    return 4.0 * torch.stack((a * b, b * c, c * a), dim=-1)


def deterministic_camera_folds(camera_names, fold_count=DEFAULT_FOLDS):
    """Map unique camera identities to sorted-name round-robin folds."""
    if fold_count < 2:
        raise ValueError(f"fold_count must be at least 2, got {fold_count}")
    names = [str(name) for name in camera_names]
    if len(names) != len(set(names)):
        raise ValueError("camera identities must be unique")
    ordered = sorted(names)
    return {name: rank % fold_count for rank, name in enumerate(ordered)}


def exact_squared_loss_gain(residual, correction):
    """Return ||r||² - ||r-z||² without clamping negative gains."""
    if residual.shape != correction.shape:
        raise ValueError(
            f"residual and correction must have the same shape, got "
            f"{tuple(residual.shape)} and {tuple(correction.shape)}"
        )
    return 2.0 * (residual * correction).sum() - correction.square().sum()


def _validate_fold_statistics(fold_gram, fold_rhs, fold_rss, fold_observations):
    _require_shape(fold_gram, (3, 3), "fold_gram")
    _require_shape(fold_rhs, (3,), "fold_rhs")
    if fold_rhs.shape[:-1] != fold_gram.shape[:-2]:
        raise ValueError("fold_rhs batch and fold axes must match fold_gram")
    if fold_rss.shape != fold_gram.shape[:-2]:
        raise ValueError("fold_rss shape must match fold_gram batch and fold axes")
    if fold_observations.shape != fold_rss.shape:
        raise ValueError("fold_observations shape must match fold_rss")
    if fold_gram.ndim < 3:
        raise ValueError("fold statistics need a fold axis")
    if fold_gram.shape[-3] < 2:
        raise ValueError("at least two folds are required")
    if not all(value.is_floating_point() for value in (fold_gram, fold_rhs, fold_rss)):
        raise TypeError("fold_gram, fold_rhs, and fold_rss must be floating point")
    devices = {fold_gram.device, fold_rhs.device, fold_rss.device, fold_observations.device}
    if len(devices) != 1:
        raise ValueError("all fold statistics must be on the same device")
    if bool((fold_rss < 0).any()):
        raise ValueError("fold_rss must be nonnegative")
    if bool((fold_observations < 0).any()):
        raise ValueError("fold_observations must be nonnegative")


def _ridge_candidates(gram, exponents):
    scale = gram.diagonal(dim1=-2, dim2=-1).sum(-1) / 3.0
    powers = torch.tensor(
        [10.0**int(exp) for exp in exponents], dtype=gram.dtype, device=gram.device
    )
    return scale.unsqueeze(-1) * powers


def _gcv_select(gram, rhs, rss, observations, exponents):
    """Select ridge per batch row; exact ties choose the larger ridge."""
    ridge = _ridge_candidates(gram, exponents)
    valid = (
        torch.isfinite(gram).all(dim=(-2, -1))
        & torch.isfinite(rhs).all(dim=-1)
        & torch.isfinite(rss)
        & (gram.diagonal(dim1=-2, dim2=-1).sum(-1) > 0)
        & (observations > 3)
    )
    eye = torch.eye(3, dtype=gram.dtype, device=gram.device)
    systems = gram.unsqueeze(-3) + ridge[..., None, None] * eye
    safe_systems = torch.where(valid[..., None, None, None], systems, eye)
    safe_rhs = torch.where(valid[..., None], rhs, torch.zeros_like(rhs))
    coefficients = torch.linalg.solve(safe_systems, safe_rhs[..., None, :, None]).squeeze(-1)

    linear = (coefficients * rhs.unsqueeze(-2)).sum(-1)
    quadratic = torch.einsum("...li,...ij,...lj->...l", coefficients, gram, coefficients)
    fitted_rss = (rss.unsqueeze(-1) - 2.0 * linear + quadratic).clamp_min(0.0)
    inv_systems = torch.linalg.inv(safe_systems)
    effective_df = torch.einsum("...ij,...lji->...l", gram, inv_systems)
    denominator = observations.unsqueeze(-1) - effective_df
    scores = observations.unsqueeze(-1) * fitted_rss / denominator.square()
    scores = torch.where(
        valid.unsqueeze(-1) & (denominator > 0) & torch.isfinite(scores),
        scores,
        torch.full_like(scores, float("inf")),
    )

    reverse_index = torch.argmin(torch.flip(scores, dims=(-1,)), dim=-1)
    selected_index = scores.shape[-1] - 1 - reverse_index
    gather_index = selected_index[..., None]
    selected_ridge = torch.gather(ridge, -1, gather_index).squeeze(-1)
    selected_coefficients = torch.gather(
        coefficients, -2, gather_index[..., None].expand(*gather_index.shape[:-1], 1, 3)
    ).squeeze(-2)
    selected_score = torch.gather(scores, -1, gather_index).squeeze(-1)
    valid = valid & torch.isfinite(selected_score)
    selected_ridge = torch.where(valid, selected_ridge, torch.full_like(selected_ridge, float("nan")))
    selected_coefficients = torch.where(
        valid.unsqueeze(-1),
        selected_coefficients,
        torch.full_like(selected_coefficients, float("nan")),
    )
    return valid, selected_ridge, selected_coefficients


def crossfit_edge_value(
    fold_gram,
    fold_rhs,
    fold_rss,
    fold_observations,
    *,
    beta_se=DEFAULT_BETA_SE,
    ridge_exponents: Sequence[int] = RIDGE_EXPONENTS,
):
    """Compute exact complementary-fold ridge fits and conservative signed value.

    Inputs have shape ``[..., K, 3, 3]``, ``[..., K, 3]``, and ``[..., K]``.
    A row with a degenerate complementary design is returned as invalid with NaNs;
    it is never silently assigned zero value.
    """
    _validate_fold_statistics(fold_gram, fold_rhs, fold_rss, fold_observations)
    if not ridge_exponents:
        raise ValueError("ridge_exponents must not be empty")
    if not torch.isfinite(torch.as_tensor(beta_se)) or beta_se < 0:
        raise ValueError(f"beta_se must be finite and nonnegative, got {beta_se}")

    dtype = torch.promote_types(fold_gram.dtype, fold_rhs.dtype)
    if dtype not in (torch.float32, torch.float64):
        dtype = torch.float64
    gram = fold_gram.to(dtype=dtype)
    rhs = fold_rhs.to(dtype=dtype)
    rss = fold_rss.to(dtype=dtype)
    observations = fold_observations.to(dtype=dtype)

    total_gram = gram.sum(dim=-3, keepdim=True)
    total_rhs = rhs.sum(dim=-2, keepdim=True)
    total_rss = rss.sum(dim=-1, keepdim=True)
    total_observations = observations.sum(dim=-1, keepdim=True)
    train_gram = total_gram - gram
    train_rhs = total_rhs - rhs
    train_rss = total_rss - rss
    train_observations = total_observations - observations

    fold_valid, ridge, coefficients = _gcv_select(
        train_gram, train_rhs, train_rss, train_observations, ridge_exponents
    )
    linear = (rhs * coefficients).sum(-1)
    quadratic = torch.einsum("...ki,...kij,...kj->...k", coefficients, gram, coefficients)
    fold_gain = 2.0 * linear - quadratic
    valid = fold_valid.all(dim=-1) & torch.isfinite(fold_gain).all(dim=-1)
    mean_gain = fold_gain.mean(dim=-1)
    standard_error = fold_gain.std(dim=-1, unbiased=True) / (fold_gain.shape[-1] ** 0.5)
    value = mean_gain - float(beta_se) * standard_error

    nan = torch.full_like(mean_gain, float("nan"))
    fold_gain = torch.where(valid.unsqueeze(-1), fold_gain, torch.full_like(fold_gain, float("nan")))
    mean_gain = torch.where(valid, mean_gain, nan)
    standard_error = torch.where(valid, standard_error, nan)
    value = torch.where(valid, value, nan)
    return CrossFitValue(valid, ridge, coefficients, fold_gain, mean_gain, standard_error, value)
