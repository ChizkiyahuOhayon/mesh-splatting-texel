import unittest

import torch

from svsr_footprint import filter_texel_detail, projected_texel_weights


class ProjectedFootprintTest(unittest.TestCase):
    def test_weights_use_final_pixel_area_per_texel(self):
        image_xy = torch.tensor([
            [0.0, 0.0], [4.0, 0.0], [0.0, 4.0],
            [0.0, 0.0], [2.0, 0.0], [0.0, 2.0],
        ])
        triangles = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32)
        weights = projected_texel_weights(image_xy, triangles, texel_order=2)
        torch.testing.assert_close(weights, torch.tensor([1.0, 0.5]))

    def test_filter_preserves_mean_and_scales_only_detail(self):
        texels = torch.tensor([[[0.0], [2.0]], [[1.0], [3.0]]])
        weights = torch.tensor([0.0, 0.5])
        filtered = filter_texel_detail(texels, weights)
        expected = torch.tensor([[[1.0], [1.0]], [[1.5], [2.5]]])
        torch.testing.assert_close(filtered, expected)
        torch.testing.assert_close(filtered.mean(1), texels.mean(1))

    def test_invalid_order_is_rejected(self):
        with self.assertRaises(ValueError):
            projected_texel_weights(torch.zeros(3, 2), torch.zeros(1, 3), 0)


if __name__ == "__main__":
    unittest.main()
