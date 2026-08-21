import unittest

from sota.supersampling_core import FACTORS, choose_factor, passes_test_gate


def measurements():
    return {
        factor: {"psnr": 30.0, "render_ms": 10.0 + factor}
        for factor in FACTORS
    }


class SupersamplingSelectionTest(unittest.TestCase):
    def test_fastest_eligible_factor_is_selected(self):
        rows = measurements()
        rows[1]["psnr"] = 29.949
        self.assertEqual(choose_factor(30.0, rows), 2)

    def test_timing_tie_prefers_more_samples(self):
        rows = measurements()
        rows[3]["render_ms"] = 5.0
        rows[2]["render_ms"] = 5.0
        self.assertEqual(choose_factor(30.0, rows), 3)

    def test_invalid_or_empty_selection_is_refused(self):
        rows = measurements()
        rows.pop(1)
        with self.assertRaises(ValueError):
            choose_factor(30.0, rows)
        rows = measurements()
        for row in rows.values():
            row["psnr"] = 29.0
        with self.assertRaisesRegex(ValueError, "no supersampling"):
            choose_factor(30.0, rows)

    def test_gate_requires_quality_and_speed(self):
        baseline = {"psnr": 30.0, "fps": 10.0}
        self.assertTrue(passes_test_gate(baseline, {"psnr": 29.95, "fps": 15.0}))
        self.assertFalse(passes_test_gate(baseline, {"psnr": 29.949, "fps": 15.0}))
        self.assertFalse(passes_test_gate(baseline, {"psnr": 30.0, "fps": 14.99}))


if __name__ == "__main__":
    unittest.main()
