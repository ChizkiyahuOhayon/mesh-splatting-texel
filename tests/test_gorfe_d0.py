from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from gorfe_d0 import audit_policy_state, crosscheck_v1_portfolios, topology_summary
from gorfe_v1 import FAMILY_NAMES, SELECTOR_NAMES


def _policy_fixture():
    primary = torch.tensor(
        [[4.0, 4.0, 4.0, 4.0], [-1.0, -1.0, -1.0, -1.0],
         [3.0, 3.0, 3.0, 3.0], [-2.0, -2.0, -2.0, -2.0]],
        dtype=torch.float64,
    )
    outcome = torch.tensor(
        [[2.0, 2.0, 2.0, 2.0], [-3.0, -3.0, -3.0, -3.0],
         [1.0, 1.0, 1.0, 1.0], [-2.0, -2.0, -2.0, -2.0]],
        dtype=torch.float64,
    )
    positive = primary.abs() + 1.0
    return SimpleNamespace(
        primary=primary,
        outcome=outcome,
        raw_residual=positive,
        same_view_gain=positive + 1.0,
        rhs_norm=positive + 2.0,
        coverage=positive + 3.0,
    )


class TopologySummaryTest(unittest.TestCase):
    def test_identical_edges_count_once_as_shared_vertex_pairs(self):
        endpoints = torch.tensor(
            [[0, 1], [0, 1], [1, 2], [4, 5]], dtype=torch.int64
        )
        observed = topology_summary(endpoints)
        self.assertEqual(observed["same_endpoint_pairs"], 1)
        self.assertEqual(observed["shared_vertex_pairs"], 3)
        self.assertAlmostEqual(observed["shared_vertex_pair_fraction"], 0.5)


class AuditTest(unittest.TestCase):
    def setUp(self):
        self.type_ids = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
        self.endpoints = torch.tensor(
            [[0, 1], [1, 2], [0, 1], [4, 5]], dtype=torch.int64
        )
        self.audit = audit_policy_state(
            scene="garden",
            nested=_policy_fixture(),
            type_ids=self.type_ids,
            endpoints=self.endpoints,
            outer_sse=torch.full((4,), 10.0, dtype=torch.float64),
        )

    def test_harmful_nonpositive_tail_is_visible_without_changing_v1(self):
        fold = self.audit["families"]["DC"]["folds"][0]
        prefix = fold["selectors"]["primary"]["prefix_through_largest_budget"]
        large = fold["selectors"]["primary"]["curve"]["16384"]
        self.assertEqual(prefix["peak_spent"], 1)
        self.assertAlmostEqual(prefix["peak_normalized_additive_gain"], 0.2)
        self.assertAlmostEqual(prefix["end_normalized_additive_gain"], -0.1)
        self.assertEqual(prefix["positive_score_prefix_spent"], 1)
        self.assertAlmostEqual(large["nonpositive_score_fraction"], 0.5)

    def test_every_selector_and_locked_budget_has_a_descriptive_record(self):
        for family in FAMILY_NAMES:
            self.assertTrue(self.audit["families"][family]["applicable"])
            for fold in self.audit["families"][family]["folds"]:
                self.assertEqual(set(fold["selectors"]), set(SELECTOR_NAMES))
                for selector in SELECTOR_NAMES:
                    self.assertIn("4096", fold["selectors"][selector]["curve"])
                    self.assertIn("16384", fold["selectors"][selector]["curve"])

    def test_sealed_portfolio_crosscheck_detects_drift(self):
        families = {}
        for family in FAMILY_NAMES:
            portfolio = {
                str(budget): {selector: [] for selector in SELECTOR_NAMES}
                for budget in (4096, 16384)
            }
            for fold in self.audit["families"][family]["folds"]:
                for selector in SELECTOR_NAMES:
                    for budget in (4096, 16384):
                        portfolio[str(budget)][selector].append(
                            fold["selectors"][selector]["curve"][str(budget)][
                                "v1_portfolio_value"
                            ]
                        )
            families[family] = {"metrics": {"portfolio": portfolio}}
        sealed = {"families": families}
        observed = crosscheck_v1_portfolios(self.audit, sealed)
        self.assertEqual(observed["checked_values"], 168)
        sealed["families"]["DC"]["metrics"]["portfolio"]["4096"]["primary"][0] += 1e-6
        with self.assertRaisesRegex(ValueError, "portfolio mismatch"):
            crosscheck_v1_portfolios(self.audit, sealed)


class RunnerBoundaryTest(unittest.TestCase):
    def test_runner_is_state_only_and_hides_cuda(self):
        repository = Path(__file__).resolve().parents[1]
        script = (repository / "experiments/gorfe_d0/run.sh").read_text(encoding="utf-8")
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', script)
        self.assertNotIn("GORFE_V1_GARDEN_DATA", script)
        self.assertNotIn("GORFE_V1_ROOM_DATA", script)
        audit = (repository / "gorfe_d0_audit.py").read_text(encoding="utf-8")
        self.assertNotIn("triangle_renderer", audit)
        self.assertNotIn("load_training_rgb", audit)


if __name__ == "__main__":
    unittest.main()
