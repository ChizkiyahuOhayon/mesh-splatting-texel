import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

import gorfe_v1 as v1


def _statistics(rhs_values, *, feature_dim=1, groups=None):
    rhs_values = torch.as_tensor(rhs_values, dtype=torch.float64)
    if rhs_values.ndim == 1:
        rhs_values = rhs_values.unsqueeze(0)
    if rhs_values.shape[1] != v1.FOLD_COUNT:
        raise ValueError("fixture needs four folds")
    groups = rhs_values.shape[0] if groups is None else groups
    gram = torch.eye(feature_dim, dtype=torch.float64).reshape(1, 1, feature_dim, feature_dim)
    gram = gram.repeat(groups, v1.FOLD_COUNT, 1, 1)
    rhs = torch.zeros((groups, v1.FOLD_COUNT, feature_dim, 3), dtype=torch.float64)
    rhs[:, :, 0, 0] = rhs_values
    if feature_dim == 3:
        rhs[:, :, 1, 1] = 0.3 * rhs_values
        rhs[:, :, 2, 2] = -0.2 * rhs_values
    return SimpleNamespace(
        gram=gram,
        rhs=rhs,
        support_rss=torch.full((groups, v1.FOLD_COUNT), 100.0, dtype=torch.float64),
        support_pixels=torch.full((groups, v1.FOLD_COUNT), 64, dtype=torch.int64),
    )


def _passing_metrics(primary_portfolio=0.2):
    rho = {name: [0.0] * v1.FOLD_COUNT for name in v1.SELECTOR_NAMES}
    rho["primary"] = [0.4] * v1.FOLD_COUNT
    portfolio = {}
    for budget in v1.BUDGETS:
        portfolio[str(budget)] = {
            name: [0.0] * v1.FOLD_COUNT for name in v1.SELECTOR_NAMES
        }
        portfolio[str(budget)]["primary"] = [primary_portfolio] * v1.FOLD_COUNT
    return {
        "eligible_cost_units": v1.FAMILY_MINIMUM_COST,
        "rho": rho,
        "portfolio": portfolio,
        "small_budget_jaccard_primary_permuted": [0.0] * v1.FOLD_COUNT,
    }


def _policy_scores(groups):
    shape = (groups, v1.FOLD_COUNT)
    base = torch.arange(groups * v1.FOLD_COUNT, dtype=torch.float64).reshape(shape)
    return SimpleNamespace(
        primary=base + 1.0,
        outcome=base * 0.1 - 0.2,
        raw_residual=base + 10.0,
        same_view_gain=base + 2.0,
        rhs_norm=base + 3.0,
        coverage=base + 4.0,
    )


class ConstantContractTest(unittest.TestCase):
    def test_exported_constants_are_the_tracked_protocol_constants(self):
        path = Path(v1.__file__).parent / "experiments" / "gorfe_v1" / "protocol_constants.json"
        with path.open(encoding="utf-8") as handle:
            locked = json.load(handle)
        self.assertEqual(v1.FOLD_COUNT, locked["camera_folds"])
        self.assertEqual(list(v1.RIDGE_EXPONENTS), locked["gcv_ridge_exponents"])
        self.assertEqual(list(v1.BUDGETS), locked["budgets_cost_units"])
        self.assertEqual(v1.FAMILY_MINIMUM_COST, locked["family_minimum_cost_units"])
        self.assertEqual(v1.TYPE_COSTS.tolist(), [locked["cost_units"]["dc"], locked["cost_units"]["sh1"]])


class GCVTest(unittest.TestCase):
    def test_exact_zero_gcv_tie_chooses_the_larger_ridge(self):
        gram = torch.tensor([[[2.0]]], dtype=torch.float64)
        rhs = torch.zeros((1, 1, 3), dtype=torch.float64)
        rss = torch.zeros(1, dtype=torch.float64)
        fit = v1.gcv_fit(gram, rhs, rss, torch.tensor([32], dtype=torch.int64))
        self.assertEqual(float(fit.ridge[0]), 200.0)
        self.assertEqual(float(fit.gcv[0]), 0.0)

    def test_roundoff_negative_rss_is_zero_but_material_negative_is_invalid(self):
        gram = torch.ones((1, 1, 1), dtype=torch.float64)
        observations = torch.tensor([32], dtype=torch.int64)
        tiny_rhs = torch.tensor([[[1e-8, 0.0, 0.0]]], dtype=torch.float64)
        fit = v1.gcv_fit(gram, tiny_rhs, torch.zeros(1, dtype=torch.float64), observations)
        self.assertEqual(float(fit.fitted_rss[0]), 0.0)
        material_rhs = torch.tensor([[[0.1, 0.0, 0.0]]], dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "no valid GCV ridge"):
            v1.gcv_fit(gram, material_rhs, torch.zeros(1, dtype=torch.float64), observations)

    def test_both_locked_feature_dimensions_are_supported(self):
        for feature_dim in (1, 3):
            with self.subTest(feature_dim=feature_dim):
                gram = torch.eye(feature_dim, dtype=torch.float64).unsqueeze(0)
                rhs = torch.ones((1, feature_dim, 3), dtype=torch.float64) * 0.1
                fit = v1.gcv_fit(
                    gram,
                    rhs,
                    torch.tensor([10.0], dtype=torch.float64),
                    torch.tensor([64], dtype=torch.int64),
                )
                self.assertEqual(fit.coefficients.shape, (1, feature_dim, 3))
                self.assertTrue(bool(torch.isfinite(fit.ridge).all()))

    def test_multivariate_gcv_matches_an_independent_enumeration(self):
        gram = torch.tensor(
            [[[4.0, 0.7, 0.2], [0.7, 2.5, -0.1], [0.2, -0.1, 1.2]]],
            dtype=torch.float64,
        )
        rhs = torch.tensor(
            [[[0.8, -0.2, 0.1], [0.3, 0.5, -0.4], [-0.2, 0.1, 0.7]]],
            dtype=torch.float64,
        )
        rss = torch.tensor([20.0], dtype=torch.float64)
        observations = torch.tensor([48], dtype=torch.int64)
        actual = v1.gcv_fit(gram, rhs, rss, observations)

        candidates = []
        scale = float(torch.trace(gram[0])) / 3.0
        eye = torch.eye(3, dtype=torch.float64)
        for exponent in v1.RIDGE_EXPONENTS:
            ridge = scale * (10.0**exponent)
            system = gram[0] + ridge * eye
            coefficient = torch.linalg.solve(system, rhs[0])
            df = float(torch.trace(torch.linalg.solve(system, gram[0])))
            fitted_rss = float(rss[0] - 2.0 * (coefficient * rhs[0]).sum())
            fitted_rss += float(torch.einsum("qc,qr,rc->", coefficient, gram[0], coefficient))
            score = float(observations[0]) * fitted_rss / (float(observations[0]) - df) ** 2
            candidates.append((score, -ridge, ridge, coefficient, df, fitted_rss))
        expected = min(candidates)
        self.assertAlmostEqual(float(actual.ridge[0]), expected[2], places=13)
        self.assertTrue(torch.allclose(actual.coefficients[0], expected[3], atol=1e-13, rtol=1e-13))
        self.assertAlmostEqual(float(actual.effective_df[0]), expected[4], places=13)
        self.assertAlmostEqual(float(actual.fitted_rss[0]), expected[5], places=13)

    def test_zero_design_and_non_float64_are_refused(self):
        with self.assertRaisesRegex(ValueError, "no valid GCV ridge"):
            v1.gcv_fit(
                torch.zeros((1, 1, 1), dtype=torch.float64),
                torch.zeros((1, 1, 3), dtype=torch.float64),
                torch.ones(1, dtype=torch.float64),
                torch.ones(1, dtype=torch.int64),
            )
        with self.assertRaisesRegex(TypeError, "float64"):
            v1.gcv_fit(
                torch.ones((1, 1, 1), dtype=torch.float32),
                torch.zeros((1, 1, 3), dtype=torch.float64),
                torch.ones(1, dtype=torch.float64),
                torch.ones(1, dtype=torch.int64),
            )


class NestedEvaluationTest(unittest.TestCase):
    def test_outer_outcome_is_signed_and_can_be_negative(self):
        result = v1.nested_scores(_statistics([1.0, 1.0, 1.0, -8.0]))
        self.assertLess(float(result.outcome[0, 3]), 0.0)

    def test_outer_fold_residual_cannot_change_its_scores_or_selection(self):
        original = _statistics(
            [
                [0.2, 0.3, 0.4, 0.5],
                [0.3, 0.5, 0.7, 0.9],
                [0.9, 0.7, 0.5, 0.3],
                [0.4, 0.8, 0.2, 0.6],
            ]
        )
        changed = SimpleNamespace(
            gram=original.gram.clone(),
            rhs=original.rhs.clone(),
            support_rss=original.support_rss.clone(),
            support_pixels=original.support_pixels.clone(),
        )
        outer = 2
        changed.rhs[:, outer] *= -1.7
        changed.support_rss[:, outer] += 37.0
        first = v1.nested_scores(original)
        second = v1.nested_scores(changed)

        for name in (
            "primary",
            "raw_residual",
            "same_view_gain",
            "rhs_norm",
            "coverage",
            "outer_ridge",
        ):
            self.assertTrue(torch.equal(getattr(first, name)[:, outer], getattr(second, name)[:, outer]))
        self.assertTrue(
            torch.equal(
                first.outer_coefficients[:, outer], second.outer_coefficients[:, outer]
            )
        )
        self.assertFalse(torch.equal(first.outcome[:, outer], second.outcome[:, outer]))

        type_ids = torch.zeros(4, dtype=torch.int64)
        endpoints = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=torch.int64)
        costs = torch.ones(4, dtype=torch.int64)
        scores_a = v1.control_scores(first, "garden", type_ids, endpoints)
        scores_b = v1.control_scores(second, "garden", type_ids, endpoints)
        for name in v1.SELECTOR_NAMES:
            self.assertTrue(
                torch.equal(scores_a[name][:, outer], scores_b[name][:, outer]), name
            )
            selection_a = v1.select_budget(
                scores_a[name][:, outer],
                costs,
                type_ids,
                endpoints,
                2,
                divide_by_cost=name != "random_id",
            )
            selection_b = v1.select_budget(
                scores_b[name][:, outer],
                costs,
                type_ids,
                endpoints,
                2,
                divide_by_cost=name != "random_id",
            )
            self.assertTrue(torch.equal(selection_a.indices, selection_b.indices), name)

    def test_outer_fit_is_exactly_the_three_fold_complement(self):
        statistics = _statistics([0.1, 0.3, 0.6, 0.9], feature_dim=3)
        result = v1.nested_scores(statistics)
        expected = v1.gcv_fit(
            statistics.gram[:, 1:].sum(1),
            statistics.rhs[:, 1:].sum(1),
            statistics.support_rss[:, 1:].sum(1),
            statistics.support_pixels[:, 1:].sum(1),
        )
        self.assertTrue(torch.equal(result.outer_ridge[:, 0], expected.ridge))
        self.assertTrue(
            torch.equal(result.outer_coefficients[:, 0], expected.coefficients)
        )


class IdentityPolicyTest(unittest.TestCase):
    def test_half_shift_is_forward_and_stays_within_each_type(self):
        values = torch.tensor(
            [10.0, 20.0, 30.0, 100.0, 200.0, 300.0, 400.0], dtype=torch.float64
        )
        type_ids = torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.int64)
        endpoints = torch.tensor(
            [[0, 1], [1, 2], [2, 3], [10, 11], [11, 12], [12, 13], [13, 14]],
            dtype=torch.int64,
        )
        shifted = v1.within_type_half_shift(values, type_ids, endpoints)
        self.assertEqual(shifted[:3].tolist(), [30.0, 10.0, 20.0])
        self.assertEqual(shifted[3:].tolist(), [300.0, 400.0, 100.0, 200.0])

    def test_budget_skips_a_nonfitting_item_and_uses_locked_ties(self):
        # First SH1 wins outright.  With one unit left, the next SH1 is skipped;
        # DC wins its exact priority tie with that SH1 by locked type order.
        scores = torch.tensor([30.0, 27.0, 9.0], dtype=torch.float64)
        type_ids = torch.tensor([1, 1, 0], dtype=torch.int64)
        endpoints = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int64)
        costs = torch.tensor([3, 3, 1], dtype=torch.int64)
        result = v1.select_budget(scores, costs, type_ids, endpoints, 4)
        self.assertEqual(result.indices.tolist(), [0, 2])
        self.assertEqual(result.spent, 4)

    def test_random_priorities_are_deterministic_and_fold_specific(self):
        type_ids = torch.tensor([0, 1, 0], dtype=torch.int64)
        endpoints = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int64)
        first = v1.random_id_priorities("garden", type_ids, endpoints)
        second = v1.random_id_priorities("garden", type_ids, endpoints)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first[:, 0], first[:, 1]))
        expected = []
        for type_id, (u, w) in zip(type_ids.tolist(), endpoints.tolist()):
            domain = f"GoRFE-V1-random|garden|0|{v1.TYPE_NAMES[type_id]}|{u}|{w}"
            expected.append(int.from_bytes(sha256(domain.encode("utf-8")).digest(), "big"))
        self.assertEqual(
            torch.argsort(first[:, 0], descending=True).tolist(),
            sorted(range(3), key=lambda index: expected[index], reverse=True),
        )


class RankAndDecisionTest(unittest.TestCase):
    def test_spearman_uses_average_ties_and_constant_is_zero(self):
        ranks = v1.average_ranks(torch.tensor([5.0, 5.0, 9.0], dtype=torch.float64))
        self.assertEqual(ranks.tolist(), [1.5, 1.5, 3.0])
        self.assertEqual(v1.spearman([1.0, 1.0], [2.0, 3.0]), 0.0)

    def test_outcome_is_not_read_until_every_selection_is_frozen(self):
        class RevealProbe:
            def __init__(self):
                values = _policy_scores(1)
                for name in (
                    "primary",
                    "raw_residual",
                    "same_view_gain",
                    "rhs_norm",
                    "coverage",
                ):
                    setattr(self, name, getattr(values, name))
                self._outcome = values.outcome
                self.outcome_reads = 0

            @property
            def outcome(self):
                self.outcome_reads += 1
                return self._outcome

        probe = RevealProbe()
        original_scan = v1._scan_budget

        def guarded_scan(*args, **kwargs):
            self.assertEqual(probe.outcome_reads, 0)
            return original_scan(*args, **kwargs)

        with mock.patch.object(v1, "_scan_budget", side_effect=guarded_scan):
            v1.evaluate_family(
                probe,
                "garden",
                torch.tensor([0], dtype=torch.int64),
                torch.tensor([[0, 1]], dtype=torch.int64),
                [1.0] * v1.FOLD_COUNT,
                "DC",
            )
        self.assertGreater(probe.outcome_reads, 0)

    def test_scene_family_passes_only_the_full_boolean_conjunction(self):
        passing = _passing_metrics()
        self.assertTrue(v1.decide_scene_family(passing)["pass"])
        failing = _passing_metrics()
        failing["portfolio"][str(v1.BUDGETS[0])]["primary"][0] = -1.0
        self.assertFalse(v1.decide_scene_family(failing)["pass"])

    def test_empty_and_low_cost_families_produce_a_recorded_fail(self):
        empty = v1.evaluate_family(
            _policy_scores(0),
            "garden",
            torch.empty(0, dtype=torch.int64),
            torch.empty((0, 2), dtype=torch.int64),
            [1.0] * v1.FOLD_COUNT,
            "DC",
        )
        self.assertFalse(empty["metrics_applicable"])
        self.assertFalse(v1.decide_scene_family(empty)["pass"])

        low = v1.evaluate_family(
            _policy_scores(1),
            "garden",
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([[0, 1]], dtype=torch.int64),
            [1.0] * v1.FOLD_COUNT,
            "DC",
        )
        self.assertTrue(low["metrics_applicable"])
        self.assertEqual(low["eligible_cost_units"], 1)
        self.assertFalse(v1.decide_scene_family(low)["pass"])

    def test_overall_requires_a_shared_family_and_uses_exact_tie_order(self):
        all_pass = {
            scene: {family: _passing_metrics() for family in v1.FAMILY_NAMES}
            for scene in ("garden", "room")
        }
        tied = v1.decide_overall(all_pass)
        self.assertEqual(tied["decision"], "pass")
        self.assertEqual(tied["advanced_family"], "DC")

        disjoint = {
            scene: {family: _passing_metrics() for family in v1.FAMILY_NAMES}
            for scene in ("garden", "room")
        }
        for family in ("SH1", "MIXED"):
            disjoint["garden"][family]["eligible_cost_units"] = 0
        for family in ("DC", "MIXED"):
            disjoint["room"][family]["eligible_cost_units"] = 0
        result = v1.decide_overall(disjoint)
        self.assertEqual(result["decision"], "fail")
        self.assertIsNone(result["advanced_family"])


if __name__ == "__main__":
    unittest.main()
