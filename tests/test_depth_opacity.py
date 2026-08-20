import unittest
from pathlib import Path

from sota.depth_opacity_core import SCALES, choose_scale, evenly_spaced_indices


class SelectionTest(unittest.TestCase):
    def test_evenly_spaced_indices_are_exact_and_unique(self):
        indices = evenly_spaced_indices(272, 32)
        self.assertEqual(len(indices), 32)
        self.assertEqual(len(set(indices)), 32)
        self.assertEqual(indices[0], 0)
        self.assertLess(indices[-1], 272)

    def test_short_collection_uses_every_view(self):
        self.assertEqual(evenly_spaced_indices(3, 32), [0, 1, 2])

    def test_choose_scale_uses_psnr_and_prefers_opaque_on_exact_tie(self):
        scores = {scale: 1.0 for scale in SCALES}
        scores[0.8] = 2.0
        scores[0.9] = 2.0
        self.assertEqual(choose_scale(scores), 0.9)

    def test_choose_scale_rejects_an_unregistered_sweep(self):
        with self.assertRaisesRegex(ValueError, "locked scale grid"):
            choose_scale({1.0: 1.0})

    def test_launcher_uses_the_repository_module_entrypoint(self):
        launcher = (
            Path(__file__).resolve().parents[1] / "sota" / "batch11.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('cd "$HERE/.."', launcher)
        self.assertIn('-u -m sota.depth_opacity', launcher)
        self.assertNotIn('-u sota/depth_opacity.py', launcher)


if __name__ == "__main__":
    unittest.main()
