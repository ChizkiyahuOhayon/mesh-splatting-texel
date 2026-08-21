import unittest

from sota.tail_culling_core import (
    THRESHOLDS,
    choose_threshold,
    passes_test_gate,
)


def measurements():
    return {
        threshold: {"psnr": 30.0, "render_ms": 10.0 + index}
        for index, threshold in enumerate(THRESHOLDS)
    }


class TailCullingSelectionTest(unittest.TestCase):
    def test_fastest_threshold_inside_quality_tolerance_is_selected(self):
        rows = measurements()
        rows[1e-4]["render_ms"] = 10.0
        rows[3e-4]["render_ms"] = 8.0
        rows[1e-3].update(psnr=29.981, render_ms=6.0)
        rows[3e-3].update(psnr=29.979, render_ms=4.0)
        rows[1e-2].update(psnr=29.0, render_ms=2.0)
        self.assertEqual(choose_threshold(rows), 1e-3)

    def test_timing_tie_prefers_the_smaller_threshold(self):
        rows = measurements()
        rows[1e-4]["render_ms"] = 8.0
        rows[3e-4]["render_ms"] = 5.0
        rows[1e-3]["render_ms"] = 5.0
        self.assertEqual(choose_threshold(rows), 3e-4)

    def test_grid_and_measurements_are_validated(self):
        rows = measurements()
        rows.pop(1e-2)
        with self.assertRaises(ValueError):
            choose_threshold(rows)
        rows = measurements()
        rows[1e-2]["render_ms"] = 0.0
        with self.assertRaises(ValueError):
            choose_threshold(rows)

    def test_test_gate_requires_both_quality_and_speed(self):
        baseline = {"psnr": 30.0, "fps": 10.0}
        self.assertTrue(passes_test_gate(baseline, {"psnr": 29.97, "fps": 12.5}))
        self.assertFalse(passes_test_gate(baseline, {"psnr": 29.969, "fps": 12.5}))
        self.assertFalse(passes_test_gate(baseline, {"psnr": 30.0, "fps": 12.49}))


if __name__ == "__main__":
    unittest.main()
