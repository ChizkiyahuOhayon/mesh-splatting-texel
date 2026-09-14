"""Opacity-aware topology survival (OATS): CPU contract for the survival rule.

The rule replaces the final `max_blending > 0.5` cleanup with the integrated
useful contribution `S_f = sum over views and samples of alpha * T`, applied at
a face budget matched to what the v1 rule keeps. See handover/OATS_PLAN.md.

These tests are written before the implementation. They skip with an explicit
reason until `sota/survival.py` exists, and must all pass before any GPU time.
"""

import unittest

import torch

try:
    from sota.survival import (
        budget_matched_keep,
        rule_disagreement,
        survival_contract,
        v1_keep,
    )
    SURVIVAL_AVAILABLE = True
except Exception:  # noqa: BLE001 - not implemented yet
    SURVIVAL_AVAILABLE = False


@unittest.skipUnless(
    SURVIVAL_AVAILABLE,
    "sota/survival.py is not implemented yet (handover/OATS_PLAN.md §4)",
)
class BudgetMatchedKeepTest(unittest.TestCase):
    """The rule keeps exactly as many faces as v1 does, and no more."""

    def test_keeps_exactly_the_budget(self):
        scores = torch.tensor([0.1, 5.0, 2.0, 0.0, 3.0])
        keep = budget_matched_keep(scores, budget=3)
        self.assertEqual(int(keep.sum()), 3)
        self.assertEqual(keep.dtype, torch.bool)
        self.assertTrue(bool(keep[1] and keep[4] and keep[2]))
        self.assertFalse(bool(keep[0] or keep[3]))

    def test_ties_break_by_index_and_respect_the_budget(self):
        """Every score identical: the rule must still return exactly `budget`
        faces and must be deterministic across calls."""
        scores = torch.full((7,), 0.25)
        first = budget_matched_keep(scores, budget=4)
        second = budget_matched_keep(scores, budget=4)
        self.assertEqual(int(first.sum()), 4)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first.nonzero().flatten(), torch.arange(4)))

    def test_empty_and_full_budgets_are_exact(self):
        scores = torch.tensor([0.3, 0.9, 0.1])
        self.assertEqual(int(budget_matched_keep(scores, budget=0).sum()), 0)
        full = budget_matched_keep(scores, budget=3)
        self.assertTrue(bool(full.all()))

    def test_budget_larger_than_face_count_is_rejected(self):
        with self.assertRaises(ValueError):
            budget_matched_keep(torch.tensor([1.0, 2.0]), budget=3)

    def test_negative_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            budget_matched_keep(torch.tensor([1.0, 2.0]), budget=-1)


@unittest.skipUnless(
    SURVIVAL_AVAILABLE,
    "sota/survival.py is not implemented yet (handover/OATS_PLAN.md §4)",
)
class NeutralPointTest(unittest.TestCase):
    """Feeding `max_blending` as the score must reproduce the v1 kept set.

    This is the OATS analogue of VATO's `tau_lo == tau_hi` neutral point: the
    new code path has to contain the old rule exactly, or the comparison is
    not a controlled one.
    """

    def test_v1_rule_boundary_matches_train_py(self):
        """`train.py` prunes `importance_score <= 0.5`, so 0.5 itself dies."""
        blending = torch.tensor([0.0, 0.4999, 0.5, 0.5001, 1.0])
        keep = v1_keep(blending)
        self.assertTrue(torch.equal(
            keep, torch.tensor([False, False, False, True, True])))

    def test_max_blending_as_score_reproduces_v1(self):
        torch.manual_seed(0)
        blending = torch.rand(512)
        expected = v1_keep(blending)
        keep = budget_matched_keep(blending, budget=int(expected.sum()))
        self.assertTrue(torch.equal(keep, expected))

    def test_neutral_point_survives_heavy_ties_at_the_threshold(self):
        """Many faces sitting exactly at 0.5 is the case where a naive
        top-k and the v1 rule can disagree on the boundary."""
        blending = torch.cat([
            torch.full((40,), 0.5),
            torch.full((10,), 0.9),
            torch.full((50,), 0.2),
        ])
        expected = v1_keep(blending)
        self.assertEqual(int(expected.sum()), 10)
        keep = budget_matched_keep(blending, budget=int(expected.sum()))
        self.assertTrue(torch.equal(keep, expected))


@unittest.skipUnless(
    SURVIVAL_AVAILABLE,
    "sota/survival.py is not implemented yet (handover/OATS_PLAN.md §4)",
)
class DisagreementTest(unittest.TestCase):
    """The falsifier of OATS_PLAN.md §6 needs an honest disagreement measure."""

    def test_identical_rules_report_zero_disagreement(self):
        keep = torch.tensor([True, False, True, True])
        stats = rule_disagreement(keep, keep)
        self.assertEqual(stats["symmetric_difference"], 0)
        self.assertEqual(stats["fraction"], 0.0)
        self.assertEqual(stats["jaccard"], 1.0)

    def test_symmetric_difference_and_jaccard(self):
        a = torch.tensor([True, True, False, False])
        b = torch.tensor([True, False, True, False])
        stats = rule_disagreement(a, b)
        self.assertEqual(stats["symmetric_difference"], 2)
        self.assertAlmostEqual(stats["fraction"], 0.5)
        self.assertAlmostEqual(stats["jaccard"], 1.0 / 3.0)

    def test_disagreement_requires_matching_lengths(self):
        with self.assertRaises(ValueError):
            rule_disagreement(torch.tensor([True]), torch.tensor([True, False]))


@unittest.skipUnless(
    SURVIVAL_AVAILABLE,
    "sota/survival.py is not implemented yet (handover/OATS_PLAN.md §4)",
)
class ZeroContributionTest(unittest.TestCase):
    """A face that never contributed must not be kept unless the budget forces
    it, and a forced keep has to be visible in the returned statistics."""

    def test_zero_score_faces_are_dropped_first(self):
        scores = torch.tensor([0.0, 0.0, 1.0, 2.0])
        keep, stats = budget_matched_keep(scores, budget=2, return_stats=True)
        self.assertTrue(torch.equal(keep, torch.tensor([False, False, True, True])))
        self.assertEqual(stats["kept_with_zero_score"], 0)

    def test_budget_forcing_zero_score_faces_is_reported(self):
        scores = torch.tensor([0.0, 0.0, 1.0, 2.0])
        keep, stats = budget_matched_keep(scores, budget=3, return_stats=True)
        self.assertEqual(int(keep.sum()), 3)
        self.assertEqual(stats["kept_with_zero_score"], 1)


@unittest.skipUnless(
    SURVIVAL_AVAILABLE,
    "sota/survival.py is not implemented yet (handover/OATS_PLAN.md §4)",
)
class CheckpointContractTest(unittest.TestCase):
    """v1 and survival arms must refuse each other's checkpoints, exactly as
    the adaptive arms do in sota/main_table_eval.py."""

    def test_survival_arm_requires_the_score_tensor(self):
        with self.assertRaises(ValueError):
            survival_contract({"opacity_floor": 0.8}, expect_survival=True)

    def test_v1_arm_refuses_a_survival_checkpoint(self):
        state = {"opacity_floor": 0.8, "face_survival_score": torch.zeros(4)}
        with self.assertRaises(ValueError):
            survival_contract(state, expect_survival=False, n_faces=4)

    def test_score_length_must_match_the_face_count(self):
        state = {"opacity_floor": 0.8, "face_survival_score": torch.zeros(4)}
        with self.assertRaises(ValueError):
            survival_contract(state, expect_survival=True, n_faces=5)

    def test_well_formed_survival_checkpoint_is_accepted(self):
        state = {"opacity_floor": 0.8, "face_survival_score": torch.zeros(4)}
        survival_contract(state, expect_survival=True, n_faces=4)


if __name__ == "__main__":
    unittest.main()
