"""State-only diagnostics for the sealed GoRFE-V1 failure."""

from __future__ import annotations

import math

import torch

from gorfe_v1 import (
    FAMILY_NAMES,
    FOLD_COUNT,
    SELECTOR_NAMES,
    TYPE_COSTS,
    control_scores,
    select_budget,
)


DIAGNOSTIC_BUDGETS = (256, 512, 1024, 2048, 4096, 8192, 12288, 16384)


def _family_mask(type_ids, family):
    if family == "DC":
        return type_ids == 0
    if family == "SH1":
        return type_ids == 1
    if family == "MIXED":
        return torch.ones_like(type_ids, dtype=torch.bool)
    raise ValueError(f"unknown family {family!r}")


def _scan(order, costs, budget):
    selected = []
    spent = 0
    for index in order.detach().cpu().tolist():
        cost = int(costs[index])
        if spent + cost <= budget:
            selected.append(index)
            spent += cost
            if spent == budget:
                break
    return torch.tensor(selected, dtype=torch.int64), spent


def topology_summary(endpoints):
    """Count endpoint-identical and shared-vertex pairs without O(N^2) storage."""
    if endpoints.ndim != 2 or endpoints.shape[1] != 2 or endpoints.dtype != torch.int64:
        raise ValueError("selected endpoints must be int64 [N,2]")
    count = int(endpoints.shape[0])
    possible = count * (count - 1) // 2
    if not count:
        return {
            "selected_groups": 0,
            "possible_group_pairs": 0,
            "same_endpoint_pairs": 0,
            "shared_vertex_pairs": 0,
            "shared_vertex_pair_fraction": 0.0,
        }
    _, endpoint_counts = torch.unique(endpoints, dim=0, return_counts=True)
    same_endpoint = int(((endpoint_counts * (endpoint_counts - 1)) // 2).sum())
    _, vertex_counts = torch.unique(endpoints.reshape(-1), return_counts=True)
    vertex_pair_occurrences = int(((vertex_counts * (vertex_counts - 1)) // 2).sum())
    # Identical endpoint pairs are counted at both vertices; every other pair
    # of distinct canonical edges can share at most one vertex.
    shared_vertex = vertex_pair_occurrences - same_endpoint
    return {
        "selected_groups": count,
        "possible_group_pairs": possible,
        "same_endpoint_pairs": same_endpoint,
        "shared_vertex_pairs": shared_vertex,
        "shared_vertex_pair_fraction": shared_vertex / possible if possible else 0.0,
    }


def _selection_metrics(indices, spent, budget, scores, outcomes, costs, endpoints, outer_sse):
    if not indices.numel():
        return {
            "budget": budget,
            "spent": 0,
            "selected_groups": 0,
            "signed_gain_sum": 0.0,
            "normalized_additive_gain": 0.0,
            "v1_portfolio_value": 0.0,
            "positive_outcome_fraction": 0.0,
            "negative_outcome_fraction": 0.0,
            "positive_score_fraction": 0.0,
            "nonpositive_score_fraction": 0.0,
            "minimum_score": None,
            "topology": topology_summary(endpoints[:0]),
        }
    selected_outcomes = outcomes[indices]
    selected_scores = scores[indices]
    signed = float(selected_outcomes.sum())
    normalized = signed / outer_sse
    return {
        "budget": budget,
        "spent": spent,
        "selected_groups": int(indices.numel()),
        "signed_gain_sum": signed,
        "normalized_additive_gain": normalized,
        "v1_portfolio_value": budget / spent * normalized,
        "positive_outcome_fraction": float((selected_outcomes > 0).to(torch.float64).mean()),
        "negative_outcome_fraction": float((selected_outcomes < 0).to(torch.float64).mean()),
        "positive_score_fraction": float((selected_scores > 0).to(torch.float64).mean()),
        "nonpositive_score_fraction": float((selected_scores <= 0).to(torch.float64).mean()),
        "minimum_score": float(selected_scores.min()),
        "topology": topology_summary(endpoints[indices]),
    }


def _prefix_diagnostic(order, scores, outcomes, costs, outer_sse, maximum_budget):
    indices, spent = _scan(order, costs, maximum_budget)
    if not indices.numel():
        return {
            "maximum_budget": maximum_budget,
            "spent": spent,
            "peak_spent": 0,
            "peak_selected_groups": 0,
            "peak_normalized_additive_gain": 0.0,
            "end_normalized_additive_gain": 0.0,
            "peak_to_end_drop": 0.0,
            "first_zero_crossing_spent": None,
            "positive_score_prefix_spent": 0,
            "positive_score_prefix_groups": 0,
            "positive_score_prefix_normalized_additive_gain": 0.0,
        }
    chosen_outcomes = outcomes[indices]
    chosen_costs = costs[indices]
    cumulative_gain = chosen_outcomes.cumsum(0) / outer_sse
    cumulative_spent = chosen_costs.cumsum(0)
    with_empty = torch.cat((torch.zeros(1, dtype=torch.float64), cumulative_gain))
    peak_count = int(torch.argmax(with_empty))
    peak_value = float(with_empty[peak_count])
    peak_spent = int(cumulative_spent[peak_count - 1]) if peak_count else 0

    positive_seen = False
    zero_crossing = None
    for value, cost in zip(cumulative_gain.tolist(), cumulative_spent.tolist()):
        positive_seen = positive_seen or value > 0
        if positive_seen and value <= 0:
            zero_crossing = int(cost)
            break

    positive_count = 0
    for index in order.detach().cpu().tolist():
        value = float(scores[index])
        if value <= 0:
            break
        positive_count += 1
    if positive_count:
        positive_indices = order[:positive_count]
        positive_spent = int(costs[positive_indices].sum())
        positive_gain = float(outcomes[positive_indices].sum()) / outer_sse
    else:
        positive_spent = 0
        positive_gain = 0.0
    end_value = float(cumulative_gain[-1])
    return {
        "maximum_budget": maximum_budget,
        "spent": spent,
        "peak_spent": peak_spent,
        "peak_selected_groups": peak_count,
        "peak_normalized_additive_gain": peak_value,
        "end_normalized_additive_gain": end_value,
        "peak_to_end_drop": peak_value - end_value,
        "first_zero_crossing_spent": zero_crossing,
        "positive_score_prefix_spent": positive_spent,
        "positive_score_prefix_groups": positive_count,
        "positive_score_prefix_normalized_additive_gain": positive_gain,
    }


def _jaccard(first, second):
    a = set(first.detach().cpu().tolist())
    b = set(second.detach().cpu().tolist())
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _fold_audit(scores, outcomes, costs, type_ids, endpoints, outer_sse, budgets):
    selector_rows = {}
    selections = {}
    total_cost = int(costs.sum())
    for name in SELECTOR_NAMES:
        divide = name != "random_id"
        order = select_budget(
            scores[name], costs, type_ids, endpoints, total_cost, divide_by_cost=divide
        ).indices
        by_budget = {}
        selection_rows = {}
        for budget in budgets:
            indices, spent = _scan(order, costs, budget)
            selection_rows[budget] = indices
            by_budget[str(budget)] = _selection_metrics(
                indices,
                spent,
                budget,
                scores[name],
                outcomes,
                costs,
                endpoints,
                outer_sse,
            )
        selections[name] = selection_rows
        selector_rows[name] = {
            "curve": by_budget,
            "prefix_through_largest_budget": _prefix_diagnostic(
                order, scores[name], outcomes, costs, outer_sse, max(budgets)
            ),
        }
    overlaps = {
        control: {
            str(budget): _jaccard(
                selections["primary"][budget], selections[control][budget]
            )
            for budget in budgets
        }
        for control in SELECTOR_NAMES
        if control != "primary"
    }
    return {"selectors": selector_rows, "primary_control_jaccard": overlaps}


def _summary(folds, small_budget, large_budget):
    primary = [fold["selectors"]["primary"] for fold in folds]
    random = [fold["selectors"]["random_id"] for fold in folds]
    return {
        "large_gain_below_small_folds": sum(
            row["curve"][str(large_budget)]["normalized_additive_gain"]
            < row["curve"][str(small_budget)]["normalized_additive_gain"]
            for row in primary
        ),
        "peak_before_large_budget_folds": sum(
            row["prefix_through_largest_budget"]["peak_spent"] < large_budget
            for row in primary
        ),
        "nonpositive_primary_score_at_large_budget_folds": sum(
            row["curve"][str(large_budget)]["nonpositive_score_fraction"] > 0
            for row in primary
        ),
        "primary_shared_vertex_pair_fraction_mean": {
            str(budget): sum(
                row["curve"][str(budget)]["topology"]["shared_vertex_pair_fraction"]
                for row in primary
            )
            / FOLD_COUNT
            for budget in (small_budget, large_budget)
        },
        "random_shared_vertex_pair_fraction_mean": {
            str(budget): sum(
                row["curve"][str(budget)]["topology"]["shared_vertex_pair_fraction"]
                for row in random
            )
            / FOLD_COUNT
            for budget in (small_budget, large_budget)
        },
    }


def audit_policy_state(*, scene, nested, type_ids, endpoints, outer_sse, budgets=None):
    budgets = tuple(DIAGNOSTIC_BUDGETS if budgets is None else budgets)
    if not budgets or any(not isinstance(value, int) or value <= 0 for value in budgets):
        raise ValueError("diagnostic budgets must be positive integers")
    if tuple(sorted(set(budgets))) != budgets:
        raise ValueError("diagnostic budgets must be strictly increasing")
    if 4096 not in budgets or 16384 not in budgets:
        raise ValueError("diagnostic budgets must retain both sealed V1 budgets")
    groups = int(type_ids.numel())
    if endpoints.shape != (groups, 2) or endpoints.dtype != torch.int64:
        raise ValueError("endpoints must be int64 [groups,2]")
    outcome = nested.outcome
    if outcome.shape != (groups, FOLD_COUNT) or outcome.dtype != torch.float64:
        raise ValueError("nested outcome must be float64 [groups,4]")
    if not bool(torch.isfinite(outcome).all()):
        raise ValueError("nested outcome must be finite")
    outer_sse = torch.as_tensor(outer_sse, dtype=torch.float64)
    if outer_sse.shape != (FOLD_COUNT,) or not bool((outer_sse > 0).all()):
        raise ValueError("outer_sse must contain four positive values")
    all_scores = control_scores(nested, scene, type_ids, endpoints)

    families = {}
    for family in FAMILY_NAMES:
        rows = torch.nonzero(_family_mask(type_ids, family), as_tuple=False).flatten()
        if not rows.numel():
            families[family] = {"applicable": False, "folds": [], "summary": None}
            continue
        family_types = type_ids[rows]
        family_endpoints = endpoints[rows]
        costs = TYPE_COSTS[family_types]
        scores = {name: value[rows] for name, value in all_scores.items()}
        folds = [
            _fold_audit(
                {name: value[:, fold] for name, value in scores.items()},
                outcome[rows, fold],
                costs,
                family_types,
                family_endpoints,
                float(outer_sse[fold]),
                budgets,
            )
            for fold in range(FOLD_COUNT)
        ]
        families[family] = {
            "applicable": True,
            "eligible_groups": int(rows.numel()),
            "eligible_cost_units": int(costs.sum()),
            "folds": folds,
            "summary": _summary(folds, 4096, 16384),
        }
    return {
        "scene": scene,
        "diagnostic_budgets": list(budgets),
        "families": families,
        "interpretation": "descriptive_only_no_pass_fail",
    }


def crosscheck_v1_portfolios(audit, sealed_result, tolerance=1e-12):
    maximum = 0.0
    checked = 0
    for family in FAMILY_NAMES:
        if not audit["families"][family]["applicable"]:
            continue
        sealed = sealed_result["families"][family]["metrics"]["portfolio"]
        for fold, row in enumerate(audit["families"][family]["folds"]):
            for selector in SELECTOR_NAMES:
                for budget in (4096, 16384):
                    observed = row["selectors"][selector]["curve"][str(budget)][
                        "v1_portfolio_value"
                    ]
                    expected = float(sealed[str(budget)][selector][fold])
                    error = abs(observed - expected)
                    if not math.isfinite(error) or error > tolerance:
                        raise ValueError(
                            f"sealed V1 portfolio mismatch for {family}/{fold}/{selector}/{budget}"
                        )
                    maximum = max(maximum, error)
                    checked += 1
    return {"checked_values": checked, "maximum_absolute_error": maximum, "tolerance": tolerance}
