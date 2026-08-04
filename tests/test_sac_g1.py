import unittest

from arguments import OptimizationParams
from sac_g1_decide import BASELINES, decide
from tests.test_sac import _ArgumentSink


class CleanupScalingParameterTest(unittest.TestCase):
    """0 means "keep the factor training ended on", which is the published
    behaviour; the confound is only removed when a run opts in."""

    def test_defaults_to_the_published_behaviour(self):
        self.assertEqual(OptimizationParams(_ArgumentSink()).cleanup_scaling, 0)


def _run(scene, arm, seed, psnr, lpips=0.248, triangles=6_900_000):
    return (scene, arm, seed), {
        "scene": scene,
        "arm": arm,
        "seed": seed,
        "cells": {
            "scaling_4": {"psnr": psnr, "ssim": 0.75, "lpips_vgg": lpips},
            "scaling_2": {"psnr": psnr - 0.1, "ssim": 0.74, "lpips_vgg": lpips + 0.003},
        },
        "primitives": {"vertices": 3_200_000, "triangles": triangles},
    }


def _table(garden_pairs, room_pairs, lpips_delta=-0.001):
    runs = {}
    for scene, pairs, baseline in (
        ("garden", garden_pairs, BASELINES["garden"]),
        ("room", room_pairs, BASELINES["room"]),
    ):
        for seed, difference in enumerate(pairs):
            key, value = _run(scene, "stock", seed, baseline)
            runs[key] = value
            key, value = _run(
                scene, "splat2", seed, baseline + difference, 0.248 + lpips_delta
            )
            runs[key] = value
    return runs


class DecideTest(unittest.TestCase):
    def test_a_consistent_effect_across_scenes_and_seeds_passes(self):
        verdict = decide(_table([0.20, 0.18, 0.22], [0.16, 0.19, 0.17]))
        self.assertTrue(verdict["pass"], verdict["checks"])
        self.assertGreater(verdict["lower_bound_at_2_se"], 0)

    def test_an_effect_smaller_than_its_own_scatter_fails(self):
        # Same mean sign, but the seeds disagree wildly: exactly the case the
        # single-seed SAC-G0 observation could not rule out.
        verdict = decide(_table([0.60, -0.45, 0.10], [0.35, -0.30, 0.05]))
        self.assertFalse(verdict["pass"])
        self.assertFalse(verdict["checks"]["effect_exceeds_twice_its_standard_error"])

    def test_an_effect_present_on_one_scene_only_fails(self):
        verdict = decide(_table([0.40, 0.42, 0.38], [-0.05, -0.04, -0.06]))
        self.assertFalse(verdict["pass"])
        self.assertFalse(
            verdict["checks"]["mean_psnr_difference_positive_on_both_scenes"]
        )

    def test_a_perceptual_regression_fails(self):
        verdict = decide(_table([0.20, 0.18, 0.22], [0.16, 0.19, 0.17], lpips_delta=0.002))
        self.assertFalse(verdict["pass"])
        self.assertFalse(verdict["checks"]["lpips_not_worse_on_either_scene"])

    def test_a_drifted_training_platform_fails_validity(self):
        runs = _table([0.20, 0.18, 0.22], [0.16, 0.19, 0.17])
        for (scene, arm, seed), run in runs.items():
            if scene == "garden" and arm == "stock":
                run["cells"]["scaling_4"]["psnr"] -= 0.5
        verdict = decide(runs)
        self.assertFalse(verdict["pass"])
        self.assertFalse(verdict["checks"]["stock_reproduces_each_baseline"])

    def test_reported_quantities_are_populated(self):
        verdict = decide(_table([0.20, 0.18, 0.22], [0.16, 0.19, 0.17]))
        garden = verdict["per_scene"]["garden"]
        self.assertEqual(len(garden["psnr_differences"]), 3)
        self.assertIsNotNone(garden["seed_standard_deviation"]["splat2"])
        self.assertIn("stock", garden["mean_triangles"])


if __name__ == "__main__":
    unittest.main()
