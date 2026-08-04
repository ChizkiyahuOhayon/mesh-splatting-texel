import unittest

from arguments import OptimizationParams
from sac_decide import BASELINE, MAX_PSNR_COST, MIN_RENDER_SPEEDUP, decide


class _ArgumentSink:
    """Stands in for argparse so the parameter defaults can be read directly."""

    def add_argument_group(self, _name):
        return self

    def add_argument(self, *_args, **_kwargs):
        return None


class FinalScalingParameterTest(unittest.TestCase):
    """The stock arm must be the unmodified pipeline: `final_scaling` replaced
    a literal 4 in train.py, so its default has to be that same 4."""

    def setUp(self):
        self.defaults = OptimizationParams(_ArgumentSink())

    def test_final_scaling_defaults_to_the_published_value(self):
        self.assertEqual(self.defaults.final_scaling, 4)

    def test_the_rest_of_the_upsampling_schedule_is_untouched(self):
        self.assertEqual(self.defaults.start_upsampling, 20000)
        self.assertEqual(self.defaults.upscaling_factor, 2)


def _cell(psnr, ssim, lpips, ms):
    return {"psnr": psnr, "ssim": ssim, "lpips_vgg": lpips, "render_ms_per_view": ms}


def _arm(at_2, at_4, training_seconds=1000.0):
    return {
        "cells": {"scaling_2": at_2, "scaling_4": at_4},
        "primitives": {"vertices": 3_260_939, "triangles": 6_968_004},
        "training_seconds": training_seconds,
    }


class DecideTest(unittest.TestCase):
    def setUp(self):
        self.stock = _arm(
            at_2=_cell(24.53, 0.741, 0.253, 26.0),
            at_4=_cell(BASELINE["psnr"], BASELINE["ssim"], BASELINE["lpips_vgg"], 67.7),
        )

    def test_a_cheap_arm_that_holds_quality_passes(self):
        splat2 = _arm(
            at_2=_cell(24.60, 0.745, 0.251, 26.0),
            at_4=_cell(24.66, 0.747, 0.249, 67.7),
        )
        verdict = decide(self.stock, splat2)
        self.assertTrue(verdict["pass"])
        self.assertLess(verdict["psnr_cost"], MAX_PSNR_COST)
        self.assertGreaterEqual(verdict["render_speedup"], MIN_RENDER_SPEEDUP)

    def test_an_excessive_quality_cost_fails(self):
        splat2 = _arm(
            at_2=_cell(24.20, 0.730, 0.262, 26.0),
            at_4=_cell(24.40, 0.738, 0.256, 67.7),
        )
        verdict = decide(self.stock, splat2)
        self.assertFalse(verdict["pass"])
        self.assertFalse(verdict["checks"]["psnr_cost_at_most_0p35"])

    def test_a_broken_training_platform_fails_validity(self):
        broken = _arm(
            at_2=_cell(22.00, 0.60, 0.39, 26.0), at_4=_cell(22.10, 0.61, 0.38, 67.7)
        )
        verdict = decide(broken, broken)
        self.assertFalse(verdict["pass"])
        self.assertFalse(verdict["checks"]["stock_reproduces_the_baseline_checkpoint"])

    def test_a_saving_that_does_not_materialise_fails(self):
        slow_stock = _arm(
            at_2=_cell(24.53, 0.741, 0.253, 40.0),
            at_4=_cell(BASELINE["psnr"], BASELINE["ssim"], BASELINE["lpips_vgg"], 67.7),
        )
        splat2 = _arm(
            at_2=_cell(24.60, 0.745, 0.251, 40.0),
            at_4=_cell(24.66, 0.747, 0.249, 67.7),
        )
        verdict = decide(slow_stock, splat2)
        self.assertFalse(verdict["pass"])
        self.assertFalse(verdict["checks"]["render_speedup_at_least_2p5x"])


if __name__ == "__main__":
    unittest.main()
