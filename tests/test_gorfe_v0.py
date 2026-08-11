import unittest

import torch

from gorfe_v0 import (
    CameraDesignRows,
    CameraStreamingFoldAccumulator,
    FoldStatistics,
    dense_fold_statistics,
    heldout_signed_gain,
    streaming_fold_statistics,
)
from gorfe_v0_gate import build_fixture


def _camera(
    *,
    name="camera",
    fold=0,
    residual_pixel_ids=(7,),
    residuals=((1.0, -2.0, 0.5),),
    pixel_ids=(7,),
    group_ids=(0,),
    features=((1.0,),),
):
    return CameraDesignRows(
        name=name,
        fold=fold,
        residual_pixel_ids=torch.tensor(residual_pixel_ids, dtype=torch.int64),
        residuals=torch.tensor(residuals, dtype=torch.float64),
        pixel_ids=torch.tensor(pixel_ids, dtype=torch.int64),
        group_ids=torch.tensor(group_ids, dtype=torch.int64),
        features=torch.tensor(features, dtype=torch.float64),
    )


class DuplicateSafeStatisticsTest(unittest.TestCase):
    def test_duplicate_rows_are_summed_before_the_outer_product(self):
        rows = _camera(pixel_ids=(7, 7), group_ids=(0, 0), features=((2.0,), (3.0,)))
        statistics, diagnostics = streaming_fold_statistics([rows], 1, 2, 1, chunk_size=1)
        self.assertEqual(float(statistics.gram[0, 0, 0, 0]), 25.0)
        self.assertNotEqual(float(statistics.gram[0, 0, 0, 0]), 13.0)
        self.assertEqual(int(statistics.support_pixels[0, 0]), 1)
        self.assertEqual(float(statistics.support_rss[0, 0]), 5.25)
        self.assertEqual(diagnostics.input_rows, 2)
        self.assertEqual(diagnostics.reduced_rows, 1)

    def test_an_exactly_cancelled_group_pixel_is_not_support(self):
        rows = _camera(pixel_ids=(7, 7), group_ids=(0, 0), features=((2.0,), (-2.0,)))
        statistics, diagnostics = streaming_fold_statistics([rows], 1, 2, 1)
        self.assertTrue(torch.equal(statistics.gram, torch.zeros_like(statistics.gram)))
        self.assertEqual(int(statistics.support_pixels.sum()), 0)
        self.assertEqual(float(statistics.support_rss.sum()), 0.0)
        self.assertEqual(diagnostics.reduced_rows, 0)
        self.assertGreater(diagnostics.estimated_peak_temporary_bytes, 0)

    def test_a_missing_residual_pixel_is_refused(self):
        rows = _camera(pixel_ids=(8,))
        with self.assertRaisesRegex(ValueError, "missing residual pixel 8"):
            streaming_fold_statistics([rows], 1, 2, 1)

    def test_duplicate_residual_pixel_ids_are_refused(self):
        rows = _camera(
            residual_pixel_ids=(7, 7),
            residuals=((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        )
        with self.assertRaisesRegex(ValueError, "must be unique"):
            streaming_fold_statistics([rows], 1, 2, 1)

    def test_duplicate_camera_identity_is_refused(self):
        accumulator = CameraStreamingFoldAccumulator(1, 2, 1)
        rows = _camera()
        accumulator.add_camera(rows)
        with self.assertRaisesRegex(ValueError, "more than once"):
            accumulator.add_camera(rows)

    def test_nonfinite_features_are_refused(self):
        rows = _camera(features=((float("nan"),),))
        with self.assertRaisesRegex(ValueError, "must be finite"):
            streaming_fold_statistics([rows], 1, 2, 1)

    def test_out_of_range_group_is_refused(self):
        rows = _camera(group_ids=(1,))
        with self.assertRaisesRegex(ValueError, "group ids"):
            streaming_fold_statistics([rows], 1, 2, 1)


class DenseOracleTest(unittest.TestCase):
    def test_both_locked_group_dimensions_match_the_dense_oracle(self):
        for feature_dim in (1, 3):
            with self.subTest(feature_dim=feature_dim):
                cameras, _ = build_fixture(feature_dim)
                dense = dense_fold_statistics(cameras, 3, 4, feature_dim)
                sparse, _ = streaming_fold_statistics(
                    cameras, 3, 4, feature_dim, chunk_size=2
                )
                self.assertTrue(torch.allclose(sparse.gram, dense.gram, atol=1e-11, rtol=1e-11))
                self.assertTrue(torch.allclose(sparse.rhs, dense.rhs, atol=1e-11, rtol=1e-11))
                self.assertTrue(
                    torch.allclose(
                        sparse.support_rss, dense.support_rss, atol=1e-11, rtol=1e-11
                    )
                )
                self.assertTrue(torch.equal(sparse.support_pixels, dense.support_pixels))

    def test_dense_oracle_also_refuses_a_missing_residual(self):
        with self.assertRaisesRegex(ValueError, "missing residual pixel 8"):
            dense_fold_statistics([_camera(pixel_ids=(8,))], 1, 2, 1)


class HeldoutGainTest(unittest.TestCase):
    def test_formula_matches_an_independent_scalar_calculation(self):
        gram = torch.tensor([[[[2.0]], [[3.0]]]], dtype=torch.float64)
        rhs = torch.tensor(
            [[[[1.0, -0.5, 0.25]], [[0.4, 0.2, -0.1]]]], dtype=torch.float64
        )
        statistics = FoldStatistics(
            gram,
            rhs,
            torch.ones((1, 2), dtype=torch.float64),
            torch.ones((1, 2), dtype=torch.int64),
        )
        result = heldout_signed_gain(statistics)
        coefficient = rhs[0, 1, 0] / (3.0 + 0.003)
        expected = 2.0 * (coefficient * rhs[0, 0, 0]).sum()
        expected -= 2.0 * coefficient.square().sum()
        self.assertAlmostEqual(float(result.signed_gain[0, 0]), float(expected), places=13)

    def test_harmful_heldout_fold_remains_negative(self):
        gram = torch.ones((1, 4, 1, 1), dtype=torch.float64)
        rhs = torch.tensor(
            [[[[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[-8.0, 0.0, 0.0]]]],
            dtype=torch.float64,
        )
        statistics = FoldStatistics(
            gram,
            rhs,
            torch.ones((1, 4), dtype=torch.float64),
            torch.ones((1, 4), dtype=torch.int64),
        )
        result = heldout_signed_gain(statistics)
        self.assertTrue(bool((result.signed_gain < 0).any()))

    def test_zero_training_design_is_invalid(self):
        statistics = FoldStatistics(
            torch.zeros((1, 4, 1, 1), dtype=torch.float64),
            torch.zeros((1, 4, 1, 3), dtype=torch.float64),
            torch.zeros((1, 4), dtype=torch.float64),
            torch.zeros((1, 4), dtype=torch.int64),
        )
        with self.assertRaisesRegex(ValueError, "positive finite ridge"):
            heldout_signed_gain(statistics)


if __name__ == "__main__":
    unittest.main()
