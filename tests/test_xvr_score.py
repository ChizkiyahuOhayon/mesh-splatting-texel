import unittest

import torch

from xvr_score import persistent_error_mass, scene_gate, top_fraction_capture


class XVRScoreTest(unittest.TestCase):
    def test_persistent_error_mass_uses_mean_visible_area(self):
        score = persistent_error_mass(
            torch.tensor([0.2, 0.4]),
            torch.tensor([30.0, 20.0]),
            torch.tensor([3.0, 2.0]),
        )
        torch.testing.assert_close(score, torch.tensor([2.0, 4.0]))

    def test_top_fraction_capture(self):
        result = top_fraction_capture(
            torch.tensor([4.0, 3.0, 2.0, 1.0]),
            torch.tensor([5.0, 3.0, 1.0, 1.0]),
            torch.tensor([True, True, True, True]),
            0.5,
        )
        self.assertEqual(result["selected_faces"], 2)
        self.assertAlmostEqual(result["capture"], 0.8)
        self.assertAlmostEqual(result["lift"], 1.6)

    def test_scene_gate_pass_and_fail(self):
        def metric(capture, lift=2.0, eligible=20_000):
            return {"top_10pct": {
                "capture": capture,
                "lift": lift,
                "eligible_faces": eligible,
                "selected_faces": 2_000,
                "actual_fraction": 0.1,
            }}

        passing = {
            "persistent_error_mass": metric(0.24, 2.4),
            "raw_error_mass": metric(0.25, 2.5),
            "max_blending": metric(0.18),
            "projected_coverage": metric(0.20),
            "world_area": metric(0.19),
        }
        self.assertTrue(scene_gate(passing)["pass"])
        passing["persistent_error_mass"] = metric(0.16, 1.6)
        self.assertFalse(scene_gate(passing)["pass"])


if __name__ == "__main__":
    unittest.main()
