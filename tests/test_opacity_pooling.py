"""Softmin routing of the shared per-vertex opacity gradient (CPU contract).

The kernel splits a face's opacity gradient over its three vertices with the
weights this module computes, so these tests pin the two properties the method
depends on: the total gradient mass is conserved, and the published hard-argmin
routing is recovered exactly at the ends of the schedule.
"""

import math
import unittest

from sota.opacity_pooling import pool_beta, softmin_weights


class SoftminWeightsTest(unittest.TestCase):
    def test_weights_sum_to_one(self):
        """The fix redistributes gradient; it must never amplify it."""
        for beta in (0.5, 5.0, 50.0, 500.0):
            w = softmin_weights([0.81, 0.93, 0.87], beta)
            self.assertAlmostEqual(sum(w), 1.0, places=12)

    def test_hard_routing_when_beta_is_not_positive(self):
        w = softmin_weights([0.9, 0.82, 0.95], 0.0)
        self.assertEqual(w, [0.0, 1.0, 0.0])
        self.assertEqual(softmin_weights([0.9, 0.82, 0.95], -1.0), [0.0, 1.0, 0.0])

    def test_large_beta_converges_to_hard_routing(self):
        """beta -> inf is the published behaviour, so the method contains it."""
        w = softmin_weights([0.80, 0.90, 0.95], 1e4)
        self.assertAlmostEqual(w[0], 1.0, places=9)
        self.assertAlmostEqual(w[1], 0.0, places=9)
        self.assertAlmostEqual(w[2], 0.0, places=9)

    def test_minimum_always_carries_the_largest_weight(self):
        w = softmin_weights([0.88, 0.81, 0.99], 20.0)
        self.assertEqual(max(range(3), key=lambda i: w[i]), 1)

    def test_equal_opacities_split_evenly(self):
        w = softmin_weights([0.84, 0.84, 0.84], 7.0)
        for value in w:
            self.assertAlmostEqual(value, 1.0 / 3.0, places=12)

    def test_weights_depend_only_on_differences(self):
        """Shifting every opacity leaves the routing unchanged, which is what
        makes the temperature meaningful while the floor ramps."""
        a = softmin_weights([0.80, 0.85, 0.95], 12.0)
        b = softmin_weights([0.60, 0.65, 0.75], 12.0)
        for x, y in zip(a, b):
            self.assertAlmostEqual(x, y, places=12)

    def test_extreme_beta_does_not_overflow(self):
        """The kernel shifts by the minimum for exactly this reason."""
        w = softmin_weights([0.1, 0.9, 1.0], 1e6)
        self.assertTrue(all(math.isfinite(v) for v in w))
        self.assertAlmostEqual(sum(w), 1.0, places=12)

    def test_face_must_have_three_vertices(self):
        with self.assertRaises(ValueError):
            softmin_weights([0.8, 0.9], 5.0)


class PoolBetaScheduleTest(unittest.TestCase):
    RAMP = dict(start_iter=5000, end_iter=24000, k_start=1.0, k_end=100.0)

    def test_hard_before_and_after_the_ramp(self):
        """Outside the window the run is the published method exactly: no
        surrogate gradient before opacity is free, and the true objective for
        the final iterations."""
        self.assertEqual(pool_beta(4999, 0.1, **self.RAMP), 0.0)
        self.assertEqual(pool_beta(24001, 0.8, **self.RAMP), 0.0)
        self.assertEqual(pool_beta(30000, 0.8, **self.RAMP), 0.0)

    def test_endpoints_hit_the_declared_k(self):
        self.assertAlmostEqual(pool_beta(5000, 0.0, **self.RAMP), 1.0)
        self.assertAlmostEqual(pool_beta(24000, 0.0, **self.RAMP), 100.0)

    def test_beta_is_measured_in_units_of_the_open_opacity_range(self):
        """A floor of 0.8 leaves a range of 0.2, so the same k is 5x hotter."""
        self.assertAlmostEqual(pool_beta(5000, 0.8, **self.RAMP), 5.0)
        self.assertAlmostEqual(pool_beta(5000, 0.9, **self.RAMP), 10.0)

    def test_beta_increases_monotonically_across_the_ramp(self):
        previous = -1.0
        for iteration in range(5000, 24001, 500):
            beta = pool_beta(iteration, 0.5, **self.RAMP)
            self.assertGreater(beta, previous)
            previous = beta

    def test_k_start_gives_a_meaningful_share_to_the_widest_spread(self):
        """At k=1 a vertex at the top of the open range still receives about
        37% of the minimum's weight, which is what makes the early gradient
        dense rather than nominally non-zero."""
        floor = 0.8
        beta = pool_beta(5000, floor, **self.RAMP)
        w = softmin_weights([floor, 1.0, 1.0], beta)
        self.assertAlmostEqual(w[1] / w[0], math.exp(-1.0), places=6)

    def test_k_end_is_effectively_hard(self):
        floor = 0.8
        beta = pool_beta(24000, floor, **self.RAMP)
        w = softmin_weights([floor, floor + 0.05, 1.0], beta)
        self.assertGreater(w[0], 0.99)

    def test_invalid_configurations_are_rejected(self):
        with self.assertRaises(ValueError):
            pool_beta(6000, 1.0, **self.RAMP)
        with self.assertRaises(ValueError):
            pool_beta(6000, 0.5, start_iter=5000, end_iter=5000, k_start=1.0, k_end=100.0)
        with self.assertRaises(ValueError):
            pool_beta(6000, 0.5, start_iter=5000, end_iter=24000, k_start=0.0, k_end=100.0)
        with self.assertRaises(ValueError):
            pool_beta(6000, 0.5, start_iter=5000, end_iter=24000, k_start=10.0, k_end=1.0)


class NeutralPathTest(unittest.TestCase):
    def test_default_arguments_are_off(self):
        """The published training path must be the default, so every existing
        result stays reproducible from this checkout."""
        from arguments import OptimizationParams

        class _Sink:
            def add_argument_group(self, _name):
                return self

            def add_argument(self, *_args, **_kwargs):
                return None

        params = OptimizationParams(_Sink())
        self.assertFalse(params.opacity_pool)
        self.assertEqual(params.opacity_pool_k_start, 1.0)
        self.assertEqual(params.opacity_pool_k_end, 100.0)


if __name__ == "__main__":
    unittest.main()
