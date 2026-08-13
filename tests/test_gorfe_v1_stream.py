import unittest

import torch

from gorfe_v1_stream import GoRFEV1Accumulator, reduce_camera_design


class CameraReductionTest(unittest.TestCase):
    def test_subpixels_depths_and_incident_faces_reduce_before_outer_product(self):
        pixels = torch.tensor([4, 4, 4, 5], dtype=torch.int32)
        groups = torch.tensor([1, 1, 1, 1], dtype=torch.int32)
        features = torch.tensor(
            [
                [0.2, 0.1, 0.0, -0.1],
                [0.3, -0.2, 0.4, 0.1],
                [-0.1, 0.3, 0.2, 0.0],
                [0.5, 0.1, -0.1, 0.2],
            ],
            dtype=torch.float32,
        )
        reduced_pixels, reduced_groups, reduced = reduce_camera_design(
            pixels, groups, features, 3
        )
        self.assertEqual(reduced_pixels.tolist(), [4, 5])
        self.assertEqual(reduced_groups.tolist(), [1, 1])
        expected = torch.stack(
            (features[:3].double().sum(0), features[3].double())
        )
        torch.testing.assert_close(reduced, expected, atol=0, rtol=0)
        correct = expected.T @ expected
        naive = features.double().T @ features.double()
        self.assertGreater(float((correct - naive).abs().max()), 1e-4)

    def test_reduction_is_order_invariant_to_float64_tolerance(self):
        generator = torch.Generator().manual_seed(11)
        pixels = torch.randint(0, 8, (200,), generator=generator)
        groups = torch.randint(0, 5, (200,), generator=generator)
        features = torch.randn((200, 4), generator=generator)
        reference = reduce_camera_design(pixels, groups, features, 5)
        order = torch.randperm(200, generator=generator)
        changed = reduce_camera_design(pixels[order], groups[order], features[order], 5)
        torch.testing.assert_close(reference[0], changed[0], atol=0, rtol=0)
        torch.testing.assert_close(reference[1], changed[1], atol=0, rtol=0)
        torch.testing.assert_close(reference[2], changed[2], atol=1e-14, rtol=1e-14)

    def test_invalid_ids_are_refused(self):
        with self.assertRaises(ValueError):
            reduce_camera_design(
                torch.tensor([0]),
                torch.tensor([2]),
                torch.ones((1, 4)),
                2,
            )


class AccumulatorTest(unittest.TestCase):
    def test_dc_and_sh1_have_separate_exact_support(self):
        accumulator = GoRFEV1Accumulator(2)
        residual = torch.tensor(
            [[1.0, -0.5, 0.25], [0.2, 0.3, -0.4]], dtype=torch.float64
        )
        diagnostic = accumulator.add_camera(
            name="camera-a",
            fold=2,
            pixel_count=2,
            pixel_ids=torch.tensor([0, 1], dtype=torch.int32),
            group_ids=torch.tensor([0, 1], dtype=torch.int32),
            features=torch.tensor(
                [[0.5, 0.0, 0.0, 0.0], [0.0, 0.2, -0.1, 0.3]],
                dtype=torch.float32,
            ),
            residuals=residual,
        )
        self.assertEqual(diagnostic.dc_support_rows, 1)
        self.assertEqual(diagnostic.sh1_support_rows, 1)
        stats = accumulator.statistics()
        self.assertEqual(int(stats.dc.support_pixels[0, 2]), 1)
        self.assertEqual(int(stats.dc.support_pixels[1, 2]), 0)
        self.assertEqual(int(stats.sh1.support_pixels[0, 2]), 0)
        self.assertEqual(int(stats.sh1.support_pixels[1, 2]), 1)
        self.assertEqual(int(stats.dc.support_cameras[0, 2]), 1)
        torch.testing.assert_close(stats.fold_full_rss[2], residual.square().sum())

    def test_duplicate_camera_and_out_of_range_pixel_are_refused(self):
        accumulator = GoRFEV1Accumulator(1)
        kwargs = dict(
            name="camera-a",
            fold=0,
            pixel_count=1,
            pixel_ids=torch.tensor([], dtype=torch.int32),
            group_ids=torch.tensor([], dtype=torch.int32),
            features=torch.empty((0, 4)),
        )
        accumulator.add_camera(**kwargs)
        with self.assertRaises(ValueError):
            accumulator.add_camera(**kwargs)
        with self.assertRaises(ValueError):
            GoRFEV1Accumulator(1).add_camera(
                **{**kwargs, "pixel_ids": torch.tensor([1]),
                   "group_ids": torch.tensor([0]), "features": torch.ones((1, 4))}
            )

    def test_raw_row_ceiling_is_hard(self):
        accumulator = GoRFEV1Accumulator(1, raw_row_limit=1)
        with self.assertRaises(OverflowError):
            accumulator.add_camera(
                name="camera-a",
                fold=0,
                pixel_count=2,
                pixel_ids=torch.tensor([0, 1]),
                group_ids=torch.tensor([0, 0]),
                features=torch.ones((2, 4)),
            )


if __name__ == "__main__":
    unittest.main()
