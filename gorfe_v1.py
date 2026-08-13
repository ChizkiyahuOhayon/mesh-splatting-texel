"""Locked mathematical core for the GoRFE-V1 prospective value gate.

This module has no renderer or dataset dependency.  It turns already reduced
float64 fold statistics into strictly nested scores and applies the policies in
``experiments/gorfe_v1/protocol.md``.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import torch


_CONSTANTS_PATH = (
    Path(__file__).resolve().parent / "experiments" / "gorfe_v1" / "protocol_constants.json"
)
with _CONSTANTS_PATH.open(encoding="utf-8") as _handle:
    PROTOCOL_CONSTANTS = json.load(_handle)

FOLD_COUNT = int(PROTOCOL_CONSTANTS["camera_folds"])
RIDGE_EXPONENTS = tuple(int(value) for value in PROTOCOL_CONSTANTS["gcv_ridge_exponents"])
NEGATIVE_RSS_RELATIVE_TOLERANCE = float(
    PROTOCOL_CONSTANTS["gcv_negative_rss_relative_tolerance"]
)
BUDGETS = tuple(int(value) for value in PROTOCOL_CONSTANTS["budgets_cost_units"])
FAMILY_MINIMUM_COST = int(PROTOCOL_CONSTANTS["family_minimum_cost_units"])
RANK_MEAN_RHO_MIN = float(PROTOCOL_CONSTANTS["rank_mean_rho_min"])
TOP_BUDGET_CONTROL_MARGIN = float(PROTOCOL_CONSTANTS["top_budget_control_margin"])
PERMUTATION_COLLAPSE_FRACTION = float(
    PROTOCOL_CONSTANTS["permutation_collapse_fraction"]
)
PERMUTATION_SMALL_BUDGET_JACCARD_MAX = float(
    PROTOCOL_CONSTANTS["permutation_small_budget_jaccard_max"]
)

TYPE_NAMES = ("DC", "SH1")
TYPE_COSTS = torch.tensor(
    [
        int(PROTOCOL_CONSTANTS["cost_units"]["dc"]),
        int(PROTOCOL_CONSTANTS["cost_units"]["sh1"]),
    ],
    dtype=torch.int64,
)
FAMILY_NAMES = ("DC", "SH1", "MIXED")
CONTROL_NAMES = (
    "raw_residual",
    "same_view_gain",
    "rhs_norm",
    "coverage",
    "random_id",
    "permuted_value",
)
SELECTOR_NAMES = ("primary",) + CONTROL_NAMES


@dataclass(frozen=True)
class GCVFit:
    ridge: torch.Tensor
    coefficients: torch.Tensor
    fitted_rss: torch.Tensor
    effective_df: torch.Tensor
    gcv: torch.Tensor


@dataclass(frozen=True)
class NestedScores:
    primary: torch.Tensor
    outcome: torch.Tensor
    raw_residual: torch.Tensor
    same_view_gain: torch.Tensor
    rhs_norm: torch.Tensor
    coverage: torch.Tensor
    inner_ridge: torch.Tensor
    outer_ridge: torch.Tensor
    outer_coefficients: torch.Tensor


@dataclass(frozen=True)
class BudgetSelection:
    indices: torch.Tensor
    spent: int


def _require_shape(value, shape, name):
    if not torch.is_tensor(value) or value.shape != shape:
        raise ValueError(f"{name} must have shape {list(shape)}")


def _require_float64(value, name):
    if value.dtype != torch.float64:
        raise TypeError(f"{name} must use float64")


def _validate_gcv_inputs(gram, rhs, rss, observations):
    if not torch.is_tensor(gram) or gram.ndim != 3 or gram.shape[-1] != gram.shape[-2]:
        raise ValueError("gram must have shape [groups, Q, Q]")
    groups, feature_dim, _ = gram.shape
    if feature_dim not in (1, 3):
        raise ValueError("GoRFE-V1 feature dimension must be one or three")
    _require_shape(rhs, (groups, feature_dim, 3), "rhs")
    _require_shape(rss, (groups,), "rss")
    _require_shape(observations, (groups,), "observations")
    for value, name in ((gram, "gram"), (rhs, "rhs"), (rss, "rss")):
        _require_float64(value, name)
    if observations.dtype not in (torch.int32, torch.int64):
        raise TypeError("observations must use int32 or int64")
    devices = {gram.device, rhs.device, rss.device, observations.device}
    if len(devices) != 1:
        raise ValueError("all GCV inputs must be on the same device")
    if not all(bool(torch.isfinite(value).all()) for value in (gram, rhs, rss)):
        raise ValueError("GCV inputs must be finite")
    if bool((rss < 0).any()):
        raise ValueError("rss must be nonnegative")
    if bool((observations < 0).any()):
        raise ValueError("observations must be nonnegative")
    return groups, feature_dim


def gcv_fit(gram, rhs, rss, observations):
    """Fit all groups with the locked ridge grid and exact larger-ridge tie rule."""
    groups, feature_dim = _validate_gcv_inputs(gram, rhs, rss, observations)
    trace = gram.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    base_valid = torch.isfinite(trace) & (trace > 0)

    eye = torch.eye(feature_dim, dtype=torch.float64, device=gram.device)
    safe_gram = torch.where(base_valid[:, None, None], gram, eye)
    safe_rhs = torch.where(base_valid[:, None, None], rhs, torch.zeros_like(rhs))
    safe_trace = torch.where(base_valid, trace, torch.ones_like(trace))
    powers = torch.tensor(
        [10.0**value for value in RIDGE_EXPONENTS],
        dtype=torch.float64,
        device=gram.device,
    )
    ridge = (safe_trace / feature_dim)[:, None] * powers[None, :]
    systems = safe_gram[:, None] + ridge[:, :, None, None] * eye

    # One solve supplies both B=A^-1 b and trace(A^-1 H), without an inverse.
    right_hand_side = torch.cat((safe_rhs, safe_gram), dim=-1)
    right_hand_side = right_hand_side[:, None].expand(-1, len(RIDGE_EXPONENTS), -1, -1)
    solution, info = torch.linalg.solve_ex(systems, right_hand_side, check_errors=False)
    coefficients = solution[..., :3]
    solved_gram = solution[..., 3:]
    effective_df = solved_gram.diagonal(dim1=-2, dim2=-1).sum(dim=-1)

    linear = torch.einsum("glqc,gqc->gl", coefficients, rhs)
    quadratic = torch.einsum("glqc,gqr,glrc->gl", coefficients, gram, coefficients)
    fitted_rss = rss[:, None] - 2.0 * linear + quadratic
    negative_limit = -NEGATIVE_RSS_RELATIVE_TOLERANCE * torch.maximum(
        torch.ones_like(rss), rss
    )
    candidate_valid = (
        base_valid[:, None]
        & (info == 0)
        & torch.isfinite(ridge)
        & (ridge > 0)
        & torch.isfinite(effective_df)
        & (observations[:, None].to(torch.float64) > effective_df)
        & torch.isfinite(fitted_rss)
        & (fitted_rss >= negative_limit[:, None])
    )
    fitted_rss = fitted_rss.clamp_min(0.0)
    denominator = observations[:, None].to(torch.float64) - effective_df
    gcv = observations[:, None].to(torch.float64) * fitted_rss / denominator.square()
    candidate_valid &= torch.isfinite(gcv)
    scores = torch.where(candidate_valid, gcv, torch.full_like(gcv, float("inf")))

    # Exponents and hence ridges are ascending.  Reversing before argmin makes
    # an exact tie choose the larger ridge without a tolerance-defined tie.
    reverse_index = torch.argmin(torch.flip(scores, dims=(-1,)), dim=-1)
    selected_index = scores.shape[-1] - 1 - reverse_index
    row = torch.arange(groups, device=gram.device)
    selected_valid = candidate_valid[row, selected_index]
    if not bool(selected_valid.all()):
        first = int(torch.nonzero(~selected_valid, as_tuple=False)[0, 0])
        raise ValueError(f"group {first} has no valid GCV ridge candidate")

    return GCVFit(
        ridge=ridge[row, selected_index],
        coefficients=coefficients[row, selected_index],
        fitted_rss=fitted_rss[row, selected_index],
        effective_df=effective_df[row, selected_index],
        gcv=gcv[row, selected_index],
    )


def signed_gain(coefficients, gram, rhs):
    """Exact unpenalized RGB squared-error reduction; negative values survive."""
    if coefficients.shape != rhs.shape:
        raise ValueError("coefficients and rhs must have the same shape")
    groups, feature_dim, channels = coefficients.shape
    if channels != 3 or gram.shape != (groups, feature_dim, feature_dim):
        raise ValueError("signed-gain tensors have incompatible shapes")
    for value, name in ((coefficients, "coefficients"), (gram, "gram"), (rhs, "rhs")):
        _require_float64(value, name)
    linear = (coefficients * rhs).sum(dim=(-2, -1))
    quadratic = torch.einsum("gqc,gqr,grc->g", coefficients, gram, coefficients)
    return 2.0 * linear - quadratic


def _validate_fold_statistics(statistics):
    gram = statistics.gram
    rhs = statistics.rhs
    rss = statistics.support_rss
    observations = statistics.support_pixels
    if not torch.is_tensor(gram) or gram.ndim != 4 or gram.shape[-1] != gram.shape[-2]:
        raise ValueError("fold gram must have shape [groups, 4, Q, Q]")
    groups, folds, feature_dim, _ = gram.shape
    if folds != FOLD_COUNT:
        raise ValueError(f"GoRFE-V1 requires exactly {FOLD_COUNT} folds")
    _require_shape(rhs, (groups, folds, feature_dim, 3), "fold rhs")
    _require_shape(rss, (groups, folds), "fold support_rss")
    _require_shape(observations, (groups, folds), "fold support_pixels")
    _validate_gcv_inputs(
        gram[:, 0], rhs[:, 0], rss[:, 0], observations[:, 0]
    )
    if any(value.device != gram.device for value in (rhs, rss, observations)):
        raise ValueError("all fold statistics must be on the same device")
    if not all(bool(torch.isfinite(value).all()) for value in (gram, rhs, rss)):
        raise ValueError("fold statistics must be finite")
    if bool((rss < 0).any()) or bool((observations < 0).any()):
        raise ValueError("fold RSS and support counts must be nonnegative")
    return gram, rhs, rss, observations


def _sum_folds(value, fold_ids):
    index = torch.tensor(fold_ids, dtype=torch.int64, device=value.device)
    return value.index_select(1, index).sum(dim=1)


def nested_scores(statistics):
    """Compute strict 3-inner/1-outer scores for all four held-out folds."""
    gram, rhs, rss, observations = _validate_fold_statistics(statistics)
    primary_columns = []
    outcome_columns = []
    raw_columns = []
    same_columns = []
    rhs_norm_columns = []
    coverage_columns = []
    inner_ridge_columns = []
    outer_ridge_columns = []
    outer_coefficient_columns = []

    for outer in range(FOLD_COUNT):
        training_folds = [fold for fold in range(FOLD_COUNT) if fold != outer]
        inner_gains = []
        inner_ridges = []
        for heldout in training_folds:
            fitting_folds = [fold for fold in training_folds if fold != heldout]
            fit = gcv_fit(
                _sum_folds(gram, fitting_folds),
                _sum_folds(rhs, fitting_folds),
                _sum_folds(rss, fitting_folds),
                _sum_folds(observations, fitting_folds),
            )
            inner_gains.append(
                signed_gain(fit.coefficients, gram[:, heldout], rhs[:, heldout])
            )
            inner_ridges.append(fit.ridge)
        inner_gain = torch.stack(inner_gains, dim=1)
        primary_columns.append(
            inner_gain.mean(dim=1)
            - inner_gain.std(dim=1, unbiased=True) / (len(training_folds) ** 0.5)
        )
        inner_ridge_columns.append(torch.stack(inner_ridges, dim=1))

        train_gram = _sum_folds(gram, training_folds)
        train_rhs = _sum_folds(rhs, training_folds)
        train_rss = _sum_folds(rss, training_folds)
        train_observations = _sum_folds(observations, training_folds)
        fit = gcv_fit(train_gram, train_rhs, train_rss, train_observations)
        outcome_columns.append(
            signed_gain(fit.coefficients, gram[:, outer], rhs[:, outer])
        )
        raw_columns.append(train_rss)
        same_columns.append(signed_gain(fit.coefficients, train_gram, train_rhs))
        rhs_norm_columns.append(torch.linalg.vector_norm(train_rhs, dim=(-2, -1)))
        coverage_columns.append(train_observations.to(torch.float64))
        outer_ridge_columns.append(fit.ridge)
        outer_coefficient_columns.append(fit.coefficients)

    return NestedScores(
        primary=torch.stack(primary_columns, dim=1),
        outcome=torch.stack(outcome_columns, dim=1),
        raw_residual=torch.stack(raw_columns, dim=1),
        same_view_gain=torch.stack(same_columns, dim=1),
        rhs_norm=torch.stack(rhs_norm_columns, dim=1),
        coverage=torch.stack(coverage_columns, dim=1),
        inner_ridge=torch.stack(inner_ridge_columns, dim=1),
        outer_ridge=torch.stack(outer_ridge_columns, dim=1),
        outer_coefficients=torch.stack(outer_coefficient_columns, dim=1),
    )


def _validate_identities(type_ids, endpoints, groups):
    _require_shape(type_ids, (groups,), "type_ids")
    _require_shape(endpoints, (groups, 2), "endpoints")
    if type_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("type_ids must use int32 or int64")
    if endpoints.dtype not in (torch.int32, torch.int64):
        raise TypeError("endpoints must use int32 or int64")
    if type_ids.device != endpoints.device:
        raise ValueError("identity tensors must be on the same device")
    if bool((type_ids < 0).any()) or bool((type_ids >= len(TYPE_NAMES)).any()):
        raise ValueError("type_ids must encode DC=0 or SH1=1")
    if bool((endpoints[:, 0] < 0).any()) or bool((endpoints[:, 0] >= endpoints[:, 1]).any()):
        raise ValueError("endpoints must be canonical nonnegative pairs u < v")


def _canonical_order(type_ids, endpoints):
    order = torch.arange(type_ids.numel(), dtype=torch.int64, device=type_ids.device)
    for values in (endpoints[:, 1], endpoints[:, 0], type_ids):
        order = order[torch.argsort(values[order], stable=True)]
    return order


def within_type_half_shift(values, type_ids, endpoints):
    """Move sorted-position ``i`` to ``i + floor(N/2)`` within each type."""
    if not torch.is_tensor(values) or values.ndim not in (1, 2):
        raise ValueError("values must have shape [groups] or [groups, folds]")
    groups = values.shape[0]
    _require_float64(values, "values")
    _validate_identities(type_ids, endpoints, groups)
    if values.device != type_ids.device:
        raise ValueError("values and identities must be on the same device")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("values must be finite")
    output = torch.empty_like(values)
    for type_id in range(len(TYPE_NAMES)):
        members = torch.nonzero(type_ids == type_id, as_tuple=False).flatten()
        if not members.numel():
            continue
        local_endpoints = endpoints[members]
        order = torch.arange(members.numel(), dtype=torch.int64, device=members.device)
        order = order[torch.argsort(local_endpoints[order, 1], stable=True)]
        order = order[torch.argsort(local_endpoints[order, 0], stable=True)]
        source = members[order]
        destination = torch.roll(source, shifts=-(source.numel() // 2))
        output[destination] = values[source]
    return output


def random_id_priorities(scene, type_ids, endpoints):
    """Ordinal float64 encoding of the locked SHA-256 unsigned priorities."""
    if scene not in PROTOCOL_CONSTANTS["scene_names"]:
        raise ValueError(f"unknown locked scene {scene!r}")
    groups = type_ids.numel()
    _validate_identities(type_ids, endpoints, groups)
    # Hashes and Python integers are CPU objects; identity metadata is tiny
    # compared with renderer statistics and is copied explicitly for this step.
    cpu_types = type_ids.detach().cpu().tolist()
    cpu_endpoints = endpoints.detach().cpu().tolist()
    columns = []
    for outer in range(FOLD_COUNT):
        priorities = []
        for type_id, (u, v) in zip(cpu_types, cpu_endpoints):
            domain = f"GoRFE-V1-random|{scene}|{outer}|{TYPE_NAMES[type_id]}|{u}|{v}"
            priorities.append(int.from_bytes(sha256(domain.encode("utf-8")).digest(), "big"))
        ordered = sorted(range(groups), key=lambda index: priorities[index])
        ranks = [0.0] * groups
        start = 0
        while start < groups:
            stop = start + 1
            while stop < groups and priorities[ordered[stop]] == priorities[ordered[start]]:
                stop += 1
            average = 0.5 * ((start + 1) + stop)
            for position in range(start, stop):
                ranks[ordered[position]] = average
            start = stop
        columns.append(torch.tensor(ranks, dtype=torch.float64, device=type_ids.device))
    return torch.stack(columns, dim=1)


def control_scores(nested, scene, type_ids, endpoints):
    groups = nested.primary.shape[0]
    for name in (
        "primary",
        "raw_residual",
        "same_view_gain",
        "rhs_norm",
        "coverage",
    ):
        value = getattr(nested, name)
        _require_shape(value, (groups, FOLD_COUNT), name)
        _require_float64(value, name)
        if value.device != nested.primary.device:
            raise ValueError("all nested policy scores must share one device")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("nested policy scores must be finite")
    _validate_identities(type_ids, endpoints, groups)
    permuted = within_type_half_shift(nested.primary, type_ids, endpoints)
    return {
        "primary": nested.primary,
        "raw_residual": nested.raw_residual,
        "same_view_gain": nested.same_view_gain,
        "rhs_norm": nested.rhs_norm,
        "coverage": nested.coverage,
        "random_id": random_id_priorities(scene, type_ids, endpoints),
        "permuted_value": permuted,
    }


def select_budget(scores, costs, type_ids, endpoints, budget, *, divide_by_cost=True):
    """Apply the locked greedy scan, including skip-and-continue leftovers."""
    if not torch.is_tensor(scores) or scores.ndim != 1:
        raise ValueError("scores must have shape [groups]")
    groups = scores.numel()
    _require_float64(scores, "scores")
    _require_shape(costs, (groups,), "costs")
    _validate_identities(type_ids, endpoints, groups)
    if costs.dtype not in (torch.int32, torch.int64):
        raise TypeError("costs must use int32 or int64")
    if any(value.device != scores.device for value in (costs, type_ids, endpoints)):
        raise ValueError("scores, costs, and identities must be on the same device")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("scores must be finite")
    locked_costs = TYPE_COSTS.to(device=costs.device)[type_ids.to(torch.int64)]
    if not torch.equal(costs.to(torch.int64), locked_costs):
        raise ValueError("costs do not match the locked DC/SH1 costs")
    if not isinstance(budget, int) or budget < 0:
        raise ValueError("budget must be a nonnegative integer")

    priority = scores / costs.to(torch.float64) if divide_by_cost else scores
    order = _canonical_order(type_ids, endpoints)
    order = order[torch.argsort(priority[order], descending=True, stable=True)]
    return _scan_budget(order, costs, budget)


def _scan_budget(order, costs, budget, *, cpu_order=None, cpu_costs=None):
    selected = []
    spent = 0
    if cpu_costs is None:
        cpu_costs = costs.detach().cpu().tolist()
    if cpu_order is None:
        cpu_order = order.detach().cpu().tolist()
    for index in cpu_order:
        cost = cpu_costs[index]
        if spent + cost <= budget:
            selected.append(index)
            spent += cost
            if spent == budget:
                break
    return BudgetSelection(
        indices=torch.tensor(selected, dtype=torch.int64, device=costs.device),
        spent=spent,
    )


def average_ranks(values):
    if not torch.is_tensor(values) or values.ndim != 1:
        raise ValueError("rank input must be a vector")
    _require_float64(values, "rank input")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("rank input must be finite")
    order = torch.argsort(values, stable=True)
    ordered = values[order]
    _, counts = torch.unique_consecutive(ordered, return_counts=True)
    ends = counts.cumsum(0).to(torch.float64)
    starts = ends - counts.to(torch.float64) + 1.0
    ordered_ranks = torch.repeat_interleave(0.5 * (starts + ends), counts)
    ranks = torch.empty_like(values)
    ranks[order] = ordered_ranks
    return ranks


def spearman(x, y):
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.float64)
    if not torch.is_tensor(y):
        y = torch.as_tensor(y, dtype=torch.float64)
    if x.device != y.device or x.shape != y.shape:
        raise ValueError("Spearman inputs must have the same shape and device")
    a = average_ranks(x)
    b = average_ranks(y)
    a = a - a.mean()
    b = b - b.mean()
    scale = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    return float((a @ b) / scale) if float(scale) > 0 else 0.0


def cv4(values):
    values = torch.as_tensor(values, dtype=torch.float64)
    if values.shape != (FOLD_COUNT,) or not bool(torch.isfinite(values).all()):
        raise ValueError(f"CV4 needs exactly {FOLD_COUNT} finite values")
    return float(values.mean() - values.std(unbiased=True) / 2.0)


def _family_mask(type_ids, family):
    if family == "DC":
        return type_ids == 0
    if family == "SH1":
        return type_ids == 1
    if family == "MIXED":
        return torch.ones_like(type_ids, dtype=torch.bool)
    raise ValueError(f"unknown family {family!r}")


def _reveal_outcome(nested, groups, outer_sse, device):
    outcome = nested.outcome
    _require_shape(outcome, (groups, FOLD_COUNT), "outcome")
    _require_float64(outcome, "outcome")
    if outcome.device != device or not bool(torch.isfinite(outcome).all()):
        raise ValueError("outcome must be finite and share the identity device")
    outer_sse = torch.as_tensor(outer_sse, dtype=torch.float64, device=device)
    if outer_sse.shape != (FOLD_COUNT,) or not bool(torch.isfinite(outer_sse).all()):
        raise ValueError("outer_sse must contain four finite fold totals")
    if bool((outer_sse <= 0).any()):
        raise ValueError("outer_sse must be positive")
    return outcome, outer_sse


def evaluate_family(nested, scene, type_ids, endpoints, outer_sse, family):
    """Build all locked rank, portfolio, and permutation metrics for one family."""
    groups = nested.primary.shape[0]
    _validate_identities(type_ids, endpoints, groups)
    all_scores = control_scores(nested, scene, type_ids, endpoints)
    mask = _family_mask(type_ids, family)
    selected_rows = torch.nonzero(mask, as_tuple=False).flatten()
    family_types = type_ids[selected_rows]
    family_endpoints = endpoints[selected_rows]
    family_costs = TYPE_COSTS.to(type_ids.device)[family_types.to(torch.int64)]
    eligible_cost = int(family_costs.sum())
    if eligible_cost == 0:
        # The empty selection table is already fixed.  Only now may target-side
        # outcomes and the whole-fold SSE be inspected.
        _reveal_outcome(nested, groups, outer_sse, type_ids.device)
        zeros = {name: [0.0] * FOLD_COUNT for name in SELECTOR_NAMES}
        return {
            "family": family,
            "eligible_cost_units": 0,
            "metrics_applicable": False,
            "rho": {name: list(values) for name, values in zeros.items()},
            "portfolio": {
                str(budget): {name: list(values) for name, values in zeros.items()}
                for budget in BUDGETS
            },
            "small_budget_jaccard_primary_permuted": [1.0] * FOLD_COUNT,
        }
    scores = {name: value[selected_rows] for name, value in all_scores.items()}
    rho = {name: [] for name in SELECTOR_NAMES}
    portfolio = {str(budget): {name: [] for name in SELECTOR_NAMES} for budget in BUDGETS}
    canonical_order = _canonical_order(family_types, family_endpoints)
    cpu_costs = family_costs.detach().cpu().tolist()
    selections = {outer: {name: {} for name in SELECTOR_NAMES} for outer in range(FOLD_COUNT)}

    # Freeze every selected set using score and identity only.  Outcome is not
    # touched until this complete table exists, making the reveal order literal.
    for outer in range(FOLD_COUNT):
        for name in SELECTOR_NAMES:
            divide = name != "random_id"
            rank_score = (
                scores[name][:, outer] / family_costs.to(torch.float64)
                if divide
                else scores[name][:, outer]
            )
            policy_order = canonical_order[
                torch.argsort(rank_score[canonical_order], descending=True, stable=True)
            ]
            cpu_order = policy_order.detach().cpu().tolist()
            for budget in BUDGETS:
                selection = _scan_budget(
                    policy_order,
                    family_costs,
                    budget,
                    cpu_order=cpu_order,
                    cpu_costs=cpu_costs,
                )
                selections[outer][name][budget] = selection

    outcome, outer_sse = _reveal_outcome(
        nested, groups, outer_sse, type_ids.device
    )
    outcome = outcome[selected_rows]
    primary_sets = {}
    permuted_sets = {}
    for outer in range(FOLD_COUNT):
        outcome_per_cost = outcome[:, outer] / family_costs.to(torch.float64)
        for name in SELECTOR_NAMES:
            divide = name != "random_id"
            rank_score = (
                scores[name][:, outer] / family_costs.to(torch.float64)
                if divide
                else scores[name][:, outer]
            )
            rho[name].append(spearman(rank_score, outcome_per_cost))
            for budget in BUDGETS:
                selection = selections[outer][name][budget]
                value = (
                    budget
                    / selection.spent
                    * float(outcome[selection.indices, outer].sum())
                    / float(outer_sse[outer])
                )
                portfolio[str(budget)][name].append(value)
                if budget == BUDGETS[0] and name == "primary":
                    primary_sets[outer] = set(selection.indices.detach().cpu().tolist())
                if budget == BUDGETS[0] and name == "permuted_value":
                    permuted_sets[outer] = set(selection.indices.detach().cpu().tolist())

    jaccard = []
    for outer in range(FOLD_COUNT):
        union = primary_sets[outer] | permuted_sets[outer]
        jaccard.append(
            len(primary_sets[outer] & permuted_sets[outer]) / len(union) if union else 1.0
        )
    return {
        "family": family,
        "eligible_cost_units": eligible_cost,
        "metrics_applicable": True,
        "rho": rho,
        "portfolio": portfolio,
        "small_budget_jaccard_primary_permuted": jaccard,
    }


def _finite_four(report, name):
    values = torch.as_tensor(report, dtype=torch.float64)
    if values.shape != (FOLD_COUNT,) or not bool(torch.isfinite(values).all()):
        raise ValueError(f"{name} must contain four finite values")
    return values


def decide_scene_family(metrics):
    rho = {
        name: _finite_four(metrics["rho"][name], f"rho.{name}")
        for name in SELECTOR_NAMES
    }
    portfolio = {
        budget: {
            name: _finite_four(
                metrics["portfolio"][str(budget)][name], f"portfolio.{budget}.{name}"
            )
            for name in SELECTOR_NAMES
        }
        for budget in BUDGETS
    }
    jaccard = _finite_four(
        metrics["small_budget_jaccard_primary_permuted"], "small-budget jaccard"
    )

    checks = {
        "eligible_cost": int(metrics["eligible_cost_units"]) >= FAMILY_MINIMUM_COST,
        "rank_mean": float(rho["primary"].mean()) >= RANK_MEAN_RHO_MIN,
        "rank_cv4": cv4(rho["primary"]) > 0,
    }
    for control in CONTROL_NAMES:
        checks[f"rank_beats_{control}"] = cv4(rho["primary"] - rho[control]) > 0

    for budget in BUDGETS:
        primary = portfolio[budget]["primary"]
        prefix = f"budget_{budget}"
        checks[f"{prefix}_three_positive"] = int((primary > 0).sum()) >= 3
        checks[f"{prefix}_cv4_positive"] = cv4(primary) > 0
        for control in CONTROL_NAMES:
            checks[f"{prefix}_beats_{control}"] = (
                cv4(primary - portfolio[budget][control]) > 0
            )
        largest_control_mean = max(
            float(portfolio[budget][control].mean()) for control in CONTROL_NAMES
        )
        checks[f"{prefix}_mean_margin"] = float(primary.mean()) >= (
            TOP_BUDGET_CONTROL_MARGIN * max(0.0, largest_control_mean)
        )

    checks["permutation_small_budget_overlap"] = bool(
        (jaccard <= PERMUTATION_SMALL_BUDGET_JACCARD_MAX).all()
    )
    for budget in BUDGETS:
        primary_gap = float(
            (portfolio[budget]["primary"] - portfolio[budget]["random_id"]).mean()
        )
        permuted_gap = float(
            (portfolio[budget]["permuted_value"] - portfolio[budget]["random_id"]).mean()
        )
        checks[f"budget_{budget}_permutation_collapses"] = abs(permuted_gap) <= (
            PERMUTATION_COLLAPSE_FRACTION * abs(primary_gap)
        )
    primary_rank_gap = float((rho["primary"] - rho["random_id"]).mean())
    permuted_rank_gap = float((rho["permuted_value"] - rho["random_id"]).mean())
    checks["rank_permutation_collapses"] = abs(permuted_rank_gap) <= (
        PERMUTATION_COLLAPSE_FRACTION * abs(primary_rank_gap)
    )
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "minimum_primary_portfolio_cv4": min(
            cv4(portfolio[budget]["primary"]) for budget in BUDGETS
        ),
    }


def decide_overall(scene_family_metrics):
    expected_scenes = set(PROTOCOL_CONSTANTS["scene_names"])
    if set(scene_family_metrics) != expected_scenes:
        raise ValueError(f"expected exactly the locked scenes {sorted(expected_scenes)}")
    if any(set(families) != set(FAMILY_NAMES) for families in scene_family_metrics.values()):
        raise ValueError(f"every scene must report exactly {list(FAMILY_NAMES)}")
    per_scene = {
        scene: {
            family: decide_scene_family(scene_family_metrics[scene][family])
            for family in FAMILY_NAMES
        }
        for scene in PROTOCOL_CONSTANTS["scene_names"]
    }
    survivors = [
        family
        for family in FAMILY_NAMES
        if all(per_scene[scene][family]["pass"] for scene in per_scene)
    ]
    advanced = None
    if survivors:
        advanced = max(
            survivors,
            key=lambda family: (
                min(
                    per_scene[scene][family]["minimum_primary_portfolio_cv4"]
                    for scene in per_scene
                ),
                -FAMILY_NAMES.index(family),
            ),
        )
    return {
        "decision": "pass" if advanced is not None else "fail",
        "advanced_family": advanced,
        "passing_intersection": survivors,
        "per_scene": per_scene,
    }
