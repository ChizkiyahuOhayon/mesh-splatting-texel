"""Run and score the preregistered GeoGauge G0 control grid."""

from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np
from scipy.stats import spearmanr

from geogauge_controls import (BASELINES, CAPACITIES, CENTRAL_DAMPING, DAMPINGS,
                               REGIMES, FitConfig, fit_case)


def _rho(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return 0.0
    value = float(spearmanr(x, y).statistic)
    return value if np.isfinite(value) else 0.0


def _flatten_local(cases, field):
    return np.concatenate([np.asarray(case["local"][field], dtype=np.float64)
                           for case in cases])


def _information(cases, damping, field):
    key = f"{damping:.0e}"
    return np.concatenate([np.asarray(case["local"]["information"][key][field],
                                      dtype=np.float64) for case in cases])


def _matched_ordering_accuracy(cases, damping):
    key = f"{damping:.0e}"
    correct = total = 0
    for seed in sorted({case["seed"] for case in cases}):
        for regime in sorted({case["regime"] for case in cases}):
            group = [case for case in cases if case["seed"] == seed
                     and case["regime"] == regime]
            for left, right in itertools.combinations(group, 2):
                observed = left["oracle_unrecoverable_fraction"] \
                    - right["oracle_unrecoverable_fraction"]
                predicted = left["perturb_prediction"][key] \
                    - right["perturb_prediction"][key]
                if abs(observed) < 1e-8:
                    continue
                total += 1
                correct += int(observed * predicted > 0)
    return correct / total if total else 0.0, total


def _cross_seed_variance(cases, damping):
    key = f"{damping:.0e}"
    grouped = {}
    for case in cases:
        condition = (case["baseline"], case["capacity"], case["regime"])
        grouped.setdefault(condition, []).append(case)
    information, variance = [], []
    for group in grouped.values():
        if len(group) < 2:
            continue
        geometry = np.asarray([case["local"]["geometry"] for case in group])
        info = np.asarray([case["local"]["information"][key]["conditional_info"]
                           for case in group])
        information.extend(info.mean(axis=0))
        variance.extend(geometry.var(axis=0, ddof=1))
    return _rho(information, variance) if variance else 0.0


def score(cases):
    geometry_error = _flatten_local(cases, "geometry_error")
    raw_info = _information(cases, CENTRAL_DAMPING, "raw_info")
    appearance_info = _flatten_local(cases, "appearance_info")
    residual = _flatten_local(cases, "residual")
    coverage = _flatten_local(cases, "coverage")
    parallax = np.concatenate([
        np.full(len(case["local"]["geometry_error"]), case["baseline_value"])
        for case in cases])
    baseline_correlations = {
        "raw_geometry_info": _rho(raw_info, geometry_error),
        "total_parameter_info": _rho(raw_info + appearance_info, geometry_error),
        "residual": _rho(residual, geometry_error),
        "coverage": _rho(coverage, geometry_error),
        "camera_baseline": _rho(parallax, geometry_error),
    }
    damping_results = {}
    for damping in DAMPINGS:
        conditional = _information(cases, damping, "conditional_info")
        fraction = _information(cases, damping, "identifiable_fraction")
        key = f"{damping:.0e}"
        damping_results[key] = {
            "conditional_info_vs_geometry_error_rho": _rho(conditional, geometry_error),
            "identifiable_fraction_vs_geometry_error_rho": _rho(fraction, geometry_error),
            "cross_seed_information_vs_geometry_variance_rho":
                _cross_seed_variance(cases, damping),
        }

    central_key = f"{CENTRAL_DAMPING:.0e}"
    conditional_rho = damping_results[central_key][
        "conditional_info_vs_geometry_error_rho"]
    best_baseline_name, best_baseline_rho = max(
        baseline_correlations.items(), key=lambda item: abs(item[1]))
    gap = abs(conditional_rho) - abs(best_baseline_rho)
    oracle = [case["oracle_unrecoverable_fraction"] for case in cases]
    predicted = [case["perturb_prediction"][central_key] for case in cases]
    oracle_rho = _rho(predicted, oracle)
    ordering, n_pairs = _matched_ordering_accuracy(cases, CENTRAL_DAMPING)
    damping_direction_consistent = all(
        values["conditional_info_vs_geometry_error_rho"] < 0
        for values in damping_results.values())

    thresholds = {
        "conditional_info_geometry_error_rho_max": -0.70,
        "absolute_rho_gap_vs_best_baseline_min": 0.15,
        "matched_ordering_accuracy_min": 0.85,
        "perturb_oracle_rho_min": 0.80,
        "damping_direction_consistent": True,
    }
    checks = {
        "conditional_correlation": conditional_rho <= -0.70,
        "baseline_gap": gap >= 0.15,
        "matched_ordering": ordering >= 0.85,
        "perturb_oracle": oracle_rho >= 0.80,
        "damping_direction": damping_direction_consistent,
    }
    return {
        "thresholds": thresholds,
        "checks": checks,
        "metrics": {
            "damping_results": damping_results,
            "baseline_correlations": baseline_correlations,
            "best_baseline": best_baseline_name,
            "best_baseline_rho": best_baseline_rho,
            "absolute_rho_gap_vs_best_baseline": gap,
            "perturb_prediction_vs_oracle_rho": oracle_rho,
            "matched_ordering_accuracy": ordering,
            "matched_pairs": n_pairs,
        },
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


def main():
    parser = argparse.ArgumentParser(description="GeoGauge v8 G0 exact-reference gate")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--baselines", nargs="+", choices=BASELINES, default=list(BASELINES))
    parser.add_argument("--capacities", nargs="+", choices=CAPACITIES, default=list(CAPACITIES))
    parser.add_argument("--regimes", nargs="+", choices=REGIMES, default=list(REGIMES))
    parser.add_argument("--steps", type=int, default=FitConfig.steps)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="output/geogauge_g0/geogauge_g0.json")
    args = parser.parse_args()

    config = FitConfig(steps=args.steps)
    combinations = list(itertools.product(args.seeds, args.baselines,
                                           args.capacities, args.regimes))
    cases = []
    for index, (seed, baseline, capacity, regime) in enumerate(combinations, 1):
        case = fit_case(seed, baseline, capacity, regime, config, args.device)
        cases.append(case)
        print(f"[{index:02d}/{len(combinations)}] seed={seed} baseline={baseline} "
              f"capacity={capacity} regime={regime} "
              f"error={case['geometry_error_mean']:.5f} "
              f"mse={case['final_mse']:.3e} "
              f"oracle={case['oracle_unrecoverable_fraction']:.3f}", flush=True)

    gate = score(cases)
    report = {
        "gate": "GeoGauge-G0",
        "config": vars(args),
        **gate,
        "cases": cases,
    }
    parent = os.path.dirname(args.out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)

    metrics = gate["metrics"]
    central = metrics["damping_results"][f"{CENTRAL_DAMPING:.0e}"]
    print("\n== GeoGauge G0 ==")
    print(f"conditional info vs geometry error rho: "
          f"{central['conditional_info_vs_geometry_error_rho']:.3f}")
    print(f"best baseline: {metrics['best_baseline']} "
          f"rho={metrics['best_baseline_rho']:.3f}; |rho| gap="
          f"{metrics['absolute_rho_gap_vs_best_baseline']:.3f}")
    print(f"perturb prediction vs refit oracle rho: "
          f"{metrics['perturb_prediction_vs_oracle_rho']:.3f}")
    print(f"matched ordering: {metrics['matched_ordering_accuracy']:.3f} "
          f"over {metrics['matched_pairs']} pairs")
    print("checks:", gate["checks"])
    print(f"G0 VERDICT: {gate['verdict']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
