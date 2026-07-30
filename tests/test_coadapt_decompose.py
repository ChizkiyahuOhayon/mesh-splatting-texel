import unittest

import torch

from coadapt_decompose import (
    recovery_fraction,
    recovery_fraction_or_none,
    texel_variants,
)


class TexelVariantsTest(unittest.TestCase):
    def test_zero_and_face_mean(self):
        texels = torch.tensor([[[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]])
        variants = texel_variants(texels)
        self.assertTrue(torch.equal(variants["zero"], torch.zeros_like(texels)))
        expected = torch.tensor([[[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]]])
        self.assertTrue(torch.equal(variants["face_mean"], expected))

    def test_recovery_fraction_for_both_metric_directions(self):
        self.assertAlmostEqual(recovery_fraction(10.0, 8.0, 9.0, True), 0.5)
        self.assertAlmostEqual(recovery_fraction(0.2, 0.4, 0.3, False), 0.5)

    def test_recovery_requires_a_regression(self):
        with self.assertRaises(ValueError):
            recovery_fraction(8.0, 10.0, 9.0, True)

    def test_optional_recovery_marks_non_regression_not_applicable(self):
        self.assertIsNone(recovery_fraction_or_none(8.0, 10.0, 9.0, True))


if __name__ == "__main__":
    unittest.main()
