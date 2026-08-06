import unittest

import torch

from apx_cells import barycentric, cell_index, fit_cell_colours, gain_to_db, squared_error
from apx_f0_decide import (
    CEILING_DB,
    CONCENTRATION_CAPTURE,
    CONTROL_MARGIN,
    LIFT_MIN,
    decide,
    scene_checks,
)


TRIANGLE = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])


class BarycentricTest(unittest.TestCase):
    def test_the_corners_map_to_the_unit_vectors(self):
        corners = TRIANGLE.repeat(3, 1, 1)
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        weights = barycentric(points, corners)
        self.assertTrue(torch.allclose(weights, torch.eye(3), atol=1e-6))

    def test_the_centroid_is_a_third_each(self):
        weights = barycentric(torch.tensor([[1 / 3, 1 / 3]]), TRIANGLE)
        self.assertTrue(torch.allclose(weights, torch.full((1, 3), 1 / 3), atol=1e-6))

    def test_the_weights_sum_to_one_everywhere(self):
        corners = TRIANGLE.repeat(20, 1, 1)
        points = torch.rand(20, 2) * 2 - 0.5
        self.assertTrue(torch.allclose(barycentric(points, corners).sum(1), torch.ones(20), atol=1e-5))

    def test_a_degenerate_face_returns_the_centroid_instead_of_dividing_by_zero(self):
        """A face projecting to zero screen area must put its pixels in one cell,
        not produce NaN that silently poisons every downstream sum."""
        flat = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
        weights = barycentric(torch.tensor([[0.5, 0.0]]), flat)
        self.assertFalse(bool(weights.isnan().any()))
        self.assertTrue(torch.allclose(weights, torch.full((1, 3), 1 / 3), atol=1e-6))


class CellIndexTest(unittest.TestCase):
    def test_order_one_puts_everything_in_one_cell(self):
        weights = barycentric(torch.rand(30, 2) * 0.5, TRIANGLE.repeat(30, 1, 1))
        self.assertEqual(int(cell_index(weights, 1).unique().numel()), 1)

    def test_order_two_separates_the_corners(self):
        corners = TRIANGLE.repeat(3, 1, 1)
        points = torch.tensor([[0.05, 0.05], [0.9, 0.05], [0.05, 0.9]])
        cells = cell_index(barycentric(points, corners), 2)
        self.assertEqual(int(cells.unique().numel()), 3)

    def test_indices_stay_inside_the_grid(self):
        corners = TRIANGLE.repeat(50, 1, 1)
        cells = cell_index(barycentric(torch.rand(50, 2) * 3 - 1, corners), 4)
        self.assertGreaterEqual(int(cells.min()), 0)
        self.assertLess(int(cells.max()), 16)

    def test_a_pixel_outside_its_face_is_clamped_not_dropped(self):
        """Rasterisation and dominant-face attribution disagree at silhouettes."""
        cells = cell_index(barycentric(torch.tensor([[5.0, 5.0]]), TRIANGLE), 4)
        self.assertGreaterEqual(int(cells[0]), 0)
        self.assertLess(int(cells[0]), 16)

    def test_an_order_below_one_is_refused(self):
        with self.assertRaises(ValueError):
            cell_index(torch.full((1, 3), 1 / 3), 0)


class FitTest(unittest.TestCase):
    def test_a_cell_is_fitted_with_the_mean_of_its_pixels(self):
        bins = torch.tensor([0, 0, 1])
        colours = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.25, 0.75]])
        fitted, counts = fit_cell_colours(bins, colours, 2)
        self.assertTrue(torch.allclose(fitted[0], torch.full((3,), 0.5)))
        self.assertTrue(torch.allclose(fitted[1], torch.tensor([0.5, 0.25, 0.75])))
        self.assertEqual(counts.tolist(), [2.0, 1.0])

    def test_an_unseen_cell_is_reported_empty(self):
        fitted, counts = fit_cell_colours(
            torch.tensor([0]), torch.tensor([[1.0, 1.0, 1.0]]), 3
        )
        self.assertEqual(counts.tolist(), [1.0, 0.0, 0.0])
        self.assertTrue(torch.allclose(fitted[1], torch.zeros(3)))

    def test_the_mean_is_the_least_squares_optimum(self):
        colours = torch.rand(40, 3)
        bins = torch.zeros(40, dtype=torch.int64)
        fitted, _ = fit_cell_colours(bins, colours, 1)
        best = float(((fitted[bins] - colours) ** 2).sum())
        for shift in (-0.05, 0.05):
            worse = float(((fitted[bins] + shift - colours) ** 2).sum())
            self.assertGreater(worse, best)


class SquaredErrorTest(unittest.TestCase):
    def test_a_perfect_fit_scores_zero(self):
        bins = torch.tensor([0, 1])
        colours = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        empty = torch.zeros(2, dtype=torch.bool)
        error = squared_error(bins, colours, colours.clone(), empty, colours.clone())
        self.assertTrue(torch.allclose(error, torch.zeros(2)))

    def test_an_empty_cell_falls_back_to_the_coarser_fit(self):
        """Scoring a held-out pixel against a cell that received no training pixels
        would charge the model class with an error it never had a chance to avoid."""
        bins = torch.tensor([0])
        colours = torch.tensor([[0.5, 0.5, 0.5]])
        fitted = torch.zeros(1, 3)
        error = squared_error(
            bins, colours, fitted, torch.ones(1, dtype=torch.bool), colours.clone()
        )
        self.assertAlmostEqual(float(error[0]), 0.0, places=6)


class GainToDbTest(unittest.TestCase):
    def test_halving_the_error_is_about_three_db(self):
        self.assertAlmostEqual(gain_to_db(2.0, 1.0), 3.0103, places=3)

    def test_no_improvement_is_zero_db(self):
        self.assertAlmostEqual(gain_to_db(1.0, 1.0), 0.0, places=9)

    def test_a_worse_fit_is_negative(self):
        self.assertLess(gain_to_db(1.0, 2.0), 0.0)

    def test_degenerate_totals_do_not_raise(self):
        self.assertEqual(gain_to_db(0.0, 1.0), 0.0)
        self.assertEqual(gain_to_db(1.0, 0.0), 0.0)


def _scene(ceiling=0.5, concentration=0.7, lift=2.5, capture=0.4, control=0.2):
    return {
        "ceiling_db": {"1": 0.1, "2": 0.3, "4": ceiling},
        "concentration": {"top_10pct": concentration},
        "signals": {
            "residual_mass": {"top_10pct": {"lift": lift, "capture": capture}},
            "max_blending": {"top_10pct": {"capture": control}},
            "projected_coverage": {"top_10pct": {"capture": control}},
            "world_area": {"top_10pct": {"capture": control}},
        },
    }


class DecideTest(unittest.TestCase):
    def test_both_scenes_passing_everything_passes(self):
        result = decide({"garden": _scene(), "room": _scene()})
        self.assertEqual(result["decision"], "PASS")
        self.assertEqual(result["failed_conditions"], [])

    def test_no_scenes_is_inconclusive(self):
        self.assertEqual(decide({})["decision"], "INCONCLUSIVE")

    def test_one_scene_failing_fails_the_gate(self):
        result = decide({"garden": _scene(), "room": _scene(ceiling=CEILING_DB - 0.01)})
        self.assertEqual(result["decision"], "FAIL")
        self.assertEqual(result["failed_conditions"], ["room:ceiling"])

    def test_a_ceiling_below_what_uniform_texels_already_achieved_fails(self):
        result = decide({"garden": _scene(ceiling=0.05)})
        self.assertEqual(result["failed_conditions"], ["garden:ceiling"])

    def test_evenly_spread_gain_fails_concentration(self):
        """Adaptive allocation can only beat uniform allocation if the gain is
        concentrated, and uniform allocation already failed the 9-scene mean."""
        result = decide({"garden": _scene(concentration=CONCENTRATION_CAPTURE - 0.01)})
        self.assertEqual(result["failed_conditions"], ["garden:concentration"])

    def test_a_signal_that_only_matches_the_controls_fails(self):
        result = decide({"garden": _scene(capture=0.2, control=0.2)})
        self.assertEqual(result["failed_conditions"], ["garden:predictability"])

    def test_beating_the_controls_by_exactly_the_margin_passes(self):
        control = 0.2
        result = decide({"garden": _scene(capture=control * (1 + CONTROL_MARGIN), control=control)})
        self.assertEqual(result["decision"], "PASS")

    def test_low_lift_fails_even_when_the_controls_are_beaten(self):
        result = decide({"garden": _scene(lift=LIFT_MIN - 0.01, capture=0.9, control=0.2)})
        self.assertEqual(result["failed_conditions"], ["garden:predictability"])

    def test_every_failing_condition_is_named(self):
        result = decide({"garden": _scene(ceiling=0.0, concentration=0.0, lift=1.0)})
        self.assertEqual(
            result["failed_conditions"],
            ["garden:ceiling", "garden:concentration", "garden:predictability"],
        )

    def test_the_numbers_behind_each_check_are_reported(self):
        checks = scene_checks(_scene(ceiling=0.42))
        self.assertAlmostEqual(checks["ceiling"]["measured_db"], 0.42)
        self.assertAlmostEqual(checks["ceiling"]["required_db"], CEILING_DB)


if __name__ == "__main__":
    unittest.main()
