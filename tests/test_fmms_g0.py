import json
import tempfile
import unittest
from pathlib import Path

import torch

from fmms_native_renderer import vertex_clip_positions, vertex_sh_colors
from utils.image_utils import psnr
from utils.sh_utils import C0
from g0_decide import decide, quality_pass


class ProjectionTest(unittest.TestCase):
    def test_row_vector_projection(self):
        vertices = torch.tensor([[1.0, 2.0, 3.0], [-2.0, 0.5, 4.0]])
        transform = torch.tensor([
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 1.0],
            [5.0, 6.0, 7.0, 0.0],
        ])
        expected = torch.tensor([[7.0, 12.0, 19.0, 3.0], [1.0, 7.5, 23.0, 4.0]])
        torch.testing.assert_close(vertex_clip_positions(vertices, transform), expected)

    def test_degree_zero_vertex_color_matches_baseline_formula(self):
        class Triangles:
            active_sh_degree = 0
            get_vertices = torch.tensor([[0.0, 0.0, 1.0]])
            get_features = torch.tensor([[[2.0, 4.0, 6.0]]])

        class Camera:
            camera_center = torch.zeros(3)

        expected = torch.tensor([[2.0 * C0 + 0.5, 4.0 * C0 + 0.5, 6.0 * C0 + 0.5]])
        torch.testing.assert_close(vertex_sh_colors(Triangles(), Camera()), expected)


class DecisionTest(unittest.TestCase):
    def test_psnr_accepts_non_contiguous_render(self):
        prediction = torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 4).transpose(-1, -2)
        target = torch.zeros_like(prediction)
        self.assertFalse(prediction.is_contiguous())
        expected_mse = (prediction ** 2).reshape(1, -1).mean(1, keepdim=True)
        expected = 20 * torch.log10(1.0 / torch.sqrt(expected_mse))
        torch.testing.assert_close(psnr(prediction, target), expected)

    def test_thresholds_are_inclusive(self):
        summary = {"aa1": {"delta_vs_ssaa4": {
            "psnr": -0.10, "ssim": -0.003, "lpips_vgg": 0.005,
        }}}
        self.assertTrue(quality_pass(summary, "aa1"))

    def test_three_scene_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots = []
            for scene in ("garden", "room", "stump"):
                root = Path(tmp) / scene
                root.mkdir()
                summary = {
                    "aa1": {"delta_vs_ssaa4": {"psnr": -0.05, "ssim": -0.001, "lpips_vgg": 0.002}},
                    "aa2": {"delta_vs_ssaa4": {"psnr": -0.20, "ssim": -0.004, "lpips_vgg": 0.006}},
                }
                timing = {"scene": scene, "variants": {
                    "ssaa4": {"samples_ms": [16.0], "peak_increment_bytes": 1600},
                    "aa1": {"samples_ms": [2.0], "peak_increment_bytes": 200},
                    "aa2": {"samples_ms": [5.0], "peak_increment_bytes": 500},
                }}
                (root / "results.json").write_text(json.dumps({"scene": scene, "summary": summary}))
                (root / "timing.json").write_text(json.dumps(timing))
                roots.append(root)
            decision = decide(roots)
            self.assertEqual(decision["verdict"], "PASS")
            self.assertEqual(decision["winner"], "aa1")


if __name__ == "__main__":
    unittest.main()
