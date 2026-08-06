import unittest

import torch

from adc_allocator import (
    MAX_DEPTH,
    allocate_depths,
    call_generator,
    deepen_cost,
    depth_histogram,
    depths_from_prefix,
    leaf_cost,
)


class CostTest(unittest.TestCase):
    """Subdividing a face to depth d replaces it with 4^d leaves, so the net gain is
    4^d - 1; taking it one level deeper costs 3 * 4^d more."""

    def test_deepening_costs_grow_by_the_branching_factor(self):
        self.assertEqual(deepen_cost(0), 3)
        self.assertEqual(deepen_cost(1), 12)

    def test_leaf_cost_matches_the_deepening_costs_it_sums(self):
        self.assertEqual(leaf_cost(torch.tensor([0, 1, 2])), 0 + 3 + 15)
        self.assertEqual(leaf_cost(torch.tensor([2])), deepen_cost(0) + deepen_cost(1))

    def test_an_unsubdivided_mesh_costs_nothing(self):
        self.assertEqual(leaf_cost(torch.zeros(9, dtype=torch.int64)), 0)


class DepthsFromPrefixTest(unittest.TestCase):
    def test_multiplicity_becomes_depth(self):
        samples = torch.tensor([0, 0, 1])
        self.assertEqual(depths_from_prefix(samples, 3).tolist(), [2, 1, 0])

    def test_depth_is_capped(self):
        samples = torch.tensor([0] * 9)
        self.assertEqual(int(depths_from_prefix(samples, 2, max_depth=2)[0]), 2)

    def test_an_empty_prefix_leaves_every_face_alone(self):
        self.assertEqual(depths_from_prefix(torch.tensor([], dtype=torch.int64), 4).tolist(),
                         [0, 0, 0, 0])


class AllocateDepthsTest(unittest.TestCase):
    def setUp(self):
        self.generator = torch.Generator().manual_seed(7)

    def test_the_budget_is_never_exceeded(self):
        probs = torch.rand(200, generator=self.generator) + 1e-3
        for budget in (3, 30, 300, 3000):
            depths = allocate_depths(probs, budget, self.generator)
            self.assertLessEqual(leaf_cost(depths), budget, f"budget {budget}")

    def test_a_uniform_score_spends_essentially_the_whole_budget(self):
        """The prefix is the longest that fits, so what is left over must be less
        than the cheapest remaining move."""
        probs = torch.ones(500)
        budget = 3 * 200
        depths = allocate_depths(probs, budget, self.generator)
        self.assertGreater(leaf_cost(depths), budget - deepen_cost(MAX_DEPTH - 1))

    def test_a_concentrated_score_actually_concentrates(self):
        """The point of the change: with one face dominating the distribution, the
        allocator must give it more than one subdivision -- which the published
        `replacement=False` plus `torch.unique` allocator structurally cannot."""
        probs = torch.full((100,), 1e-6)
        probs[0] = 1.0
        depths = allocate_depths(probs, 3 * 50, self.generator)
        self.assertEqual(int(depths[0]), MAX_DEPTH)

    def test_no_face_exceeds_the_depth_cap(self):
        probs = torch.full((10,), 1e-9)
        probs[3] = 1.0
        depths = allocate_depths(probs, 3 * 1000, self.generator)
        self.assertLessEqual(int(depths.max()), MAX_DEPTH)

    def test_zero_probability_faces_are_never_selected(self):
        """Faces below `size_probs_zero` have their probability zeroed upstream and
        must stay untouched, exactly as in the published allocator."""
        probs = torch.zeros(50)
        probs[10:20] = 1.0
        depths = allocate_depths(probs, 3 * 30, self.generator)
        self.assertEqual(int(depths[:10].sum()), 0)
        self.assertEqual(int(depths[20:].sum()), 0)

    def test_an_all_zero_score_allocates_nothing(self):
        depths = allocate_depths(torch.zeros(20), 3 * 10, self.generator)
        self.assertEqual(int(depths.sum()), 0)

    def test_a_budget_below_one_subdivision_allocates_nothing(self):
        depths = allocate_depths(torch.ones(20), deepen_cost(0) - 1, self.generator)
        self.assertEqual(int(depths.sum()), 0)

    def test_an_empty_mesh_is_handled(self):
        self.assertEqual(allocate_depths(torch.zeros(0), 300, self.generator).numel(), 0)

    def test_it_is_reproducible_from_the_generator(self):
        probs = torch.rand(100, generator=torch.Generator().manual_seed(1)) + 1e-3
        first = allocate_depths(probs, 300, torch.Generator().manual_seed(3))
        second = allocate_depths(probs, 300, torch.Generator().manual_seed(3))
        self.assertTrue(torch.equal(first, second))

    def test_the_budget_matches_what_the_baseline_would_have_spent(self):
        """The baseline subdivides `n` distinct faces once each for `3n` new faces.
        Spending that same integer is what makes the arms comparable at matched
        primitive count -- the confound that invalidated SAC-G0."""
        n = 250
        probs = torch.rand(4000, generator=self.generator) + 1e-3
        depths = allocate_depths(probs, deepen_cost(0) * n, self.generator)
        self.assertLessEqual(leaf_cost(depths), deepen_cost(0) * n)


class GeneratorTest(unittest.TestCase):
    def test_it_does_not_disturb_the_ambient_stream(self):
        """The defect being fixed: `_sample_alives` calls torch.manual_seed(1) on
        every densification round, resetting the global stream under every other
        consumer of torch randomness in the process."""
        torch.manual_seed(99)
        expected = torch.rand(3)
        torch.manual_seed(99)
        call_generator("cpu", seed=0, call_index=5)
        self.assertTrue(torch.equal(torch.rand(3), expected))

    def test_successive_rounds_are_not_correlated(self):
        probs = torch.ones(400)
        first = allocate_depths(probs, 3 * 100, call_generator("cpu", 0, 0))
        second = allocate_depths(probs, 3 * 100, call_generator("cpu", 0, 1))
        self.assertFalse(torch.equal(first, second))

    def test_a_run_is_reproducible_from_its_seed(self):
        probs = torch.ones(400)
        first = allocate_depths(probs, 3 * 100, call_generator("cpu", 4, 2))
        second = allocate_depths(probs, 3 * 100, call_generator("cpu", 4, 2))
        self.assertTrue(torch.equal(first, second))

    def test_different_seeds_differ(self):
        probs = torch.ones(400)
        first = allocate_depths(probs, 3 * 100, call_generator("cpu", 0, 0))
        second = allocate_depths(probs, 3 * 100, call_generator("cpu", 1, 0))
        self.assertFalse(torch.equal(first, second))


class HistogramTest(unittest.TestCase):
    def test_it_counts_every_depth_including_empty_ones(self):
        histogram = depth_histogram(torch.tensor([0, 0, 1, 2, 2]))
        self.assertEqual(histogram, {"depth_0": 2, "depth_1": 1, "depth_2": 2})

    def test_an_untouched_mesh_reports_only_depth_zero(self):
        histogram = depth_histogram(torch.zeros(5, dtype=torch.int64))
        self.assertEqual(histogram["depth_0"], 5)
        self.assertEqual(histogram["depth_2"], 0)


if __name__ == "__main__":
    unittest.main()
