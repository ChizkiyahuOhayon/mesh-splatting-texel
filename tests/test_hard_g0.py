import math
import unittest

from hard_g0_decide import decide


def record(arm, psnr, lpips, sigma=1e-4, scene="garden", seed=0):
    return {
        "scene": scene,
        "seed": seed,
        "arm": arm,
        "sigma": sigma,
        "cells": {
            "scaling_4": {
                "psnr": psnr,
                "ssim": 0.90,
                "lpips_vgg": lpips,
            }
        },
    }


class HardG0DecisionTest(unittest.TestCase):
    def test_passes_at_locked_boundaries(self):
        result = decide(record("stock", 25.0, 0.20),
                        record("early", 25.10, 0.20))
        self.assertEqual(result["decision"], "pass")
        self.assertTrue(all(result["checks"].values()))

    def test_fails_when_psnr_gain_is_too_small(self):
        result = decide(record("stock", 25.0, 0.20),
                        record("early", 25.099, 0.19))
        self.assertEqual(result["decision"], "fail")
        self.assertFalse(result["checks"]["early_psnr_gain_at_least_0p10_db"])

    def test_fails_when_lpips_is_worse(self):
        result = decide(record("stock", 25.0, 0.20),
                        record("early", 25.20, 0.200001))
        self.assertEqual(result["decision"], "fail")
        self.assertFalse(result["checks"]["early_lpips_is_nonworse"])

    def test_fails_when_endpoint_is_wrong(self):
        result = decide(record("stock", 25.0, 0.20),
                        record("early", 25.20, 0.19, sigma=2e-4))
        self.assertEqual(result["decision"], "fail")
        self.assertFalse(result["checks"]["early_endpoint_sigma_is_1e-4"])

    def test_fails_when_record_identity_is_wrong(self):
        result = decide(record("stock", 25.0, 0.20),
                        record("early", 25.20, 0.19, seed=1))
        self.assertEqual(result["decision"], "fail")
        self.assertFalse(result["checks"]["record_identity"])

    def test_rejects_nonfinite_metrics(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            decide(record("stock", 25.0, 0.20),
                   record("early", math.nan, 0.19))


if __name__ == "__main__":
    unittest.main()
